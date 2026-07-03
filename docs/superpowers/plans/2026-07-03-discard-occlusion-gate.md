# Discard-Occlusion Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect and drop YOLO/crop labels whose box content disagrees with GT (tiles caught mid-flight during the discard→river animation), cleaning existing datasets and blocking new ones at build time; and add capture-time pixel-stability so the AI capture path stops producing them.

**Architecture:** A per-box *consistency scorer* crops each GT box, runs the production tile classifier, and flags boxes where the prediction disagrees with GT (or the box is empty felt). A per-frame *smart-drop* rule removes individual bad boxes, or the whole frame when too many are bad. This scorer hooks into (a) a one-time purge tool over `datasets/`, (b) `build_dataset` so rebuilds stay clean. Separately, the AI capture path gains a pixel-stability confirm (masked to the table ROI so the animated cloth doesn't defeat it), and `FrameSyncer`'s capped-bypass is tightened.

**Tech Stack:** Python 3.12 (conda `auto` env), PyTorch (existing `TileNet`/`TileClassifier`), numpy, OpenCV. Tests are plain scripts (pytest-compatible, no pytest dependency).

## Global Constraints

- `PY = C:/Users/zsx/miniforge3/envs/auto/python.exe`; run everything from repo root with `PYTHONPATH=.` (PowerShell: `$env:PYTHONPATH="."`).
- 38-class taxonomy order is FROZEN (`majsoul_eye/tiles.py`: `TILE_NAMES`, `NAME_TO_ID`). Do not reorder.
- `recognize/` must stay Akagi-free. The scorer lives in `annotate/` (build-time) and a `scripts/data/` purge tool; both may import `recognize/classifier.py`.
- Production classifier weights: `majsoul_eye/recognize/tile_classifier.pt` (held-out val 0.9991).
- Tests are plain scripts named `tests/test_*.py` with `test_*()` functions and a `__main__` runner; run via `PYTHONPATH=. $PY tests/test_X.py`. Match this style; do NOT introduce a pytest dependency.
- Purge/cleanup tools: **dry-run by default**, `--apply` to act, idempotent (2nd run = no-op). Mirror `scripts/data/purge_deal_frames.py`.
- After deleting frames, rewrite `datasets/detector*/{train,val}.txt` to drop lines whose image no longer exists.
- Provisional thresholds: `TAU = 0.5` (min P(gt_cls) for a box to pass on mismatch), `MAX_BAD = 2` (per-frame bad-box budget before whole-frame drop). Task 4 calibrates and may revise these constants.

---

### Task 1: `predict_proba` on TileClassifier

**Files:**
- Modify: `majsoul_eye/recognize/classifier.py` (add method after `predict`, ~line 64)
- Test: `tests/test_classifier.py` (append a test; file already exists)

**Interfaces:**
- Consumes: existing `TileClassifier.__init__`, module-level `preprocess`, `TILE_NAMES`, `self.model`, `self.device`.
- Produces: `TileClassifier.predict_proba(crops: list[np.ndarray]) -> np.ndarray` returning an `(N, 38)` float32 softmax matrix (row i = class distribution for crop i; column j = `TILE_NAMES[j]`). Empty input → shape `(0, 38)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_classifier.py`:

```python
def test_predict_proba_shape_and_normalization():
    import numpy as np
    from majsoul_eye.recognize.classifier import TileClassifier
    clf = TileClassifier()  # loads production weights
    crops = [np.full((64, 64, 3), 200, np.uint8), np.zeros((64, 64, 3), np.uint8)]
    probs = clf.predict_proba(crops)
    assert probs.shape == (2, 38), probs.shape
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4), row_sums
    # predict() must agree with argmax of predict_proba()
    names = clf.predict(crops)
    from majsoul_eye.tiles import TILE_NAMES
    assert names == [TILE_NAMES[i] for i in probs.argmax(1)]
    assert clf.predict_proba([]).shape == (0, 38)
    print("test_predict_proba_shape_and_normalization OK")
```

Add `test_predict_proba_shape_and_normalization()` to the `__main__` runner block at the bottom of the file.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. $PY tests/test_classifier.py`
Expected: FAIL with `AttributeError: 'TileClassifier' object has no attribute 'predict_proba'`.

- [ ] **Step 3: Write minimal implementation**

In `majsoul_eye/recognize/classifier.py`, add after the `predict` method:

```python
    @torch.no_grad()
    def predict_proba(self, crops: list[np.ndarray]) -> np.ndarray:
        """Softmax class distributions, shape (len(crops), 38). Column j == TILE_NAMES[j]."""
        if not crops:
            return np.zeros((0, len(TILE_NAMES)), dtype=np.float32)
        batch = torch.stack([preprocess(c) for c in crops]).to(self.device)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy().astype(np.float32)
```

Confirm `import numpy as np` is present at the top of the file (it is used by `predict`); if not, add it.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. $PY tests/test_classifier.py`
Expected: PASS (all tests including the new one print `OK`).

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/recognize/classifier.py tests/test_classifier.py
git commit -m "feat(recognize): TileClassifier.predict_proba softmax matrix for consistency gate"
```

---

### Task 2: Consistency core (pure verdict + empty-felt)

**Files:**
- Create: `majsoul_eye/annotate/consistency.py`
- Test: `tests/test_consistency.py`

**Interfaces:**
- Consumes: `majsoul_eye.tiles.NAME_TO_ID` / `TILE_NAMES`; `majsoul_eye.label.quality.is_tile_present`.
- Produces:
  - `@dataclass BoxVerdict` with fields `ok: bool`, `gt: str`, `pred: str`, `conf: float`, `reason: str` (`""` | `"mismatch"` | `"empty_felt"`).
  - `TAU: float = 0.5`, `MAX_BAD: int = 2` (module constants).
  - `verdict_from_probs(prob_row: np.ndarray, gt: str, tau: float = TAU) -> BoxVerdict` — pure; `reason=="empty_felt"` never set here.
  - `is_empty_felt(crop: np.ndarray, min_face_frac: float = 0.12) -> bool` — thin wrapper over `is_tile_present`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_consistency.py`:

```python
"""Consistency-scorer core (pure; no model weights needed)."""
import numpy as np

from majsoul_eye.annotate.consistency import (
    verdict_from_probs, is_empty_felt, BoxVerdict, TAU,
)
from majsoul_eye.tiles import NAME_TO_ID, TILE_NAMES


def _one_hot(name, p=0.99):
    row = np.full(38, (1.0 - p) / 37, np.float32)
    row[NAME_TO_ID[name]] = p
    return row


def test_agree_is_ok():
    v = verdict_from_probs(_one_hot("8s"), "8s")
    assert v.ok and v.pred == "8s" and v.reason == "" and v.conf > 0.9
    print("test_agree_is_ok OK")


def test_confident_mismatch_is_bad():
    # classifier is sure it's 3p, GT says 8s -> bad (mismatch, low P(gt))
    v = verdict_from_probs(_one_hot("3p"), "8s")
    assert not v.ok and v.pred == "3p" and v.reason == "mismatch"
    print("test_confident_mismatch_is_bad OK")


def test_mismatch_but_gt_still_plausible_is_ok():
    # top1 != gt, but P(gt) >= TAU -> keep (avoid deleting on weak classifier calls)
    row = np.full(38, 0.0, np.float32)
    row[NAME_TO_ID["5p"]] = 0.45
    row[NAME_TO_ID["5pr"]] = 0.55   # top1 = 5pr, but gt=5p has conf 0.45... make gt pass:
    row[NAME_TO_ID["5p"]] = 0.50
    row = row / row.sum()
    v = verdict_from_probs(row, "5p", tau=0.30)
    assert v.ok, (v.pred, v.conf)
    print("test_mismatch_but_gt_still_plausible_is_ok OK")


def test_empty_felt_detection():
    felt = np.zeros((64, 64, 3), np.uint8)          # flat -> no tile face
    felt[:, :] = (90, 60, 40)
    tile = np.zeros((64, 64, 3), np.uint8)
    tile[8:56, 8:56] = 240                          # bright face
    assert is_empty_felt(felt)
    assert not is_empty_felt(tile)
    print("test_empty_felt_detection OK")


if __name__ == "__main__":
    test_agree_is_ok()
    test_confident_mismatch_is_bad()
    test_mismatch_but_gt_still_plausible_is_ok()
    test_empty_felt_detection()
    print("ALL test_consistency (core) OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. $PY tests/test_consistency.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'majsoul_eye.annotate.consistency'`.

- [ ] **Step 3: Write minimal implementation**

Create `majsoul_eye/annotate/consistency.py`:

```python
"""Per-box GT-consistency gate: crop -> classifier -> compare to GT class.

Catches boxes whose pixels don't match their GT label — chiefly discard-animation
occlusion (tile caught mid-flight, box lands on empty felt/arm), but also any
mislabel/occlusion. A frame-level smart-drop rule (see frame_decision) removes bad
boxes surgically, or the whole frame when too many are bad. Not a state predicate:
occlusion is intermittent (capture-timing-dependent), so we judge pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..tiles import NAME_TO_ID, TILE_NAMES
from ..label.quality import is_tile_present

TAU: float = 0.5        # min P(gt_cls) for a top1-mismatch box to still pass
MAX_BAD: int = 2        # per-frame bad-box budget before dropping the whole frame


@dataclass
class BoxVerdict:
    ok: bool
    gt: str
    pred: str
    conf: float          # P(gt_cls)
    reason: str          # "" | "mismatch" | "empty_felt"


def verdict_from_probs(prob_row: np.ndarray, gt: str, tau: float = TAU) -> BoxVerdict:
    """Pure verdict from one softmax row. Bad iff top1 != gt AND P(gt) < tau."""
    top = int(np.argmax(prob_row))
    pred = TILE_NAMES[top]
    conf = float(prob_row[NAME_TO_ID[gt]]) if gt in NAME_TO_ID else 0.0
    if pred == gt or conf >= tau:
        return BoxVerdict(True, gt, pred, conf, "")
    return BoxVerdict(False, gt, pred, conf, "mismatch")


def is_empty_felt(crop: np.ndarray, min_face_frac: float = 0.12) -> bool:
    """True when the crop is (almost) all table felt — no tile face present."""
    return not is_tile_present(crop, min_face_frac=min_face_frac)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. $PY tests/test_consistency.py`
Expected: PASS (`ALL test_consistency (core) OK`).

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/annotate/consistency.py tests/test_consistency.py
git commit -m "feat(annotate): consistency core — verdict_from_probs + empty-felt gate"
```

---

### Task 3: Frame scoring + smart-drop decision

**Files:**
- Modify: `majsoul_eye/annotate/consistency.py`
- Test: `tests/test_consistency.py` (append)

**Interfaces:**
- Consumes: `BoxVerdict`, `verdict_from_probs`, `is_empty_felt` (Task 2); `TileClassifier.predict_proba` (Task 1).
- Produces:
  - `score_frame(crops: list[np.ndarray], gts: list[str], clf, *, tau=TAU, min_face_frac=0.12) -> list[BoxVerdict]` — one `predict_proba` batch call; a crop failing `is_empty_felt` is bad with `reason="empty_felt"` (classifier not consulted for it).
  - `frame_decision(verdicts: list[BoxVerdict], max_bad: int = MAX_BAD) -> tuple[str, list[int]]` — returns `("keep", [])` | `("drop_boxes", bad_indices)` | `("drop_frame", bad_indices)`. `drop_boxes` when `1 <= n_bad <= max_bad`; `drop_frame` when `n_bad > max_bad`.
  - `clf` is any object with `.predict_proba(list[np.ndarray]) -> np.ndarray (N,38)` (inject a stub in tests).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_consistency.py` (and add the calls to `__main__`):

```python
def test_score_frame_and_decision():
    from majsoul_eye.annotate.consistency import score_frame, frame_decision

    class StubClf:  # deterministic fake classifier; index by crop's [0,0,0] byte
        def __init__(self, names): self.names = names
        def predict_proba(self, crops):
            out = np.zeros((len(crops), 38), np.float32)
            for i, _ in enumerate(crops):
                out[i, NAME_TO_ID[self.names[i]]] = 1.0
            return out

    # 3 boxes: gt = [8s, 3p, S]; classifier "sees" [8s, 9m, S] -> box#1 is a mismatch
    crops = [np.full((64, 64, 3), 240, np.uint8) for _ in range(3)]  # all "tile present"
    gts = ["8s", "3p", "S"]
    clf = StubClf(["8s", "9m", "S"])
    verdicts = score_frame(crops, gts, clf)
    assert [v.ok for v in verdicts] == [True, False, True]
    assert frame_decision(verdicts, max_bad=2) == ("drop_boxes", [1])

    # 3 bad boxes > max_bad -> drop whole frame
    clf2 = StubClf(["1m", "9m", "N"])
    verdicts2 = score_frame(crops, gts, clf2)
    assert frame_decision(verdicts2, max_bad=2)[0] == "drop_frame"

    # empty-felt crop is bad without consulting the classifier
    felt = np.zeros((64, 64, 3), np.uint8); felt[:] = (90, 60, 40)
    v3 = score_frame([felt], ["8s"], StubClf(["8s"]))
    assert not v3[0].ok and v3[0].reason == "empty_felt"
    print("test_score_frame_and_decision OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. $PY tests/test_consistency.py`
Expected: FAIL with `ImportError: cannot import name 'score_frame'`.

- [ ] **Step 3: Write minimal implementation**

Append to `majsoul_eye/annotate/consistency.py`:

```python
def score_frame(crops, gts, clf, *, tau: float = TAU, min_face_frac: float = 0.12):
    """Verdict per (crop, gt). Empty-felt crops are bad up-front; the rest go through
    the classifier in one batch."""
    assert len(crops) == len(gts), (len(crops), len(gts))
    verdicts: list[BoxVerdict] = [None] * len(crops)  # type: ignore
    live_idx, live_crops = [], []
    for i, (crop, gt) in enumerate(zip(crops, gts)):
        if is_empty_felt(crop, min_face_frac=min_face_frac):
            verdicts[i] = BoxVerdict(False, gt, "", 0.0, "empty_felt")
        else:
            live_idx.append(i); live_crops.append(crop)
    if live_crops:
        probs = clf.predict_proba(live_crops)
        for k, i in enumerate(live_idx):
            verdicts[i] = verdict_from_probs(probs[k], gts[i], tau=tau)
    return verdicts


def frame_decision(verdicts, max_bad: int = MAX_BAD):
    bad = [i for i, v in enumerate(verdicts) if not v.ok]
    if not bad:
        return "keep", []
    if len(bad) <= max_bad:
        return "drop_boxes", bad
    return "drop_frame", bad
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. $PY tests/test_consistency.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/annotate/consistency.py tests/test_consistency.py
git commit -m "feat(annotate): score_frame + smart-drop frame_decision"
```

---

### Task 4: Calibration script + golden-frame integration test

**Files:**
- Create: `scripts/inspect/calibrate_occlusion_gate.py`
- Test: `tests/test_consistency_golden.py`

**Interfaces:**
- Consumes: `score_frame`, `frame_decision` (Task 3); `TileClassifier` (Task 1).
- Produces: a CLI that scans a dataset's `yolo/{images,labels}`, prints (a) per-frame bad-box histogram, (b) P(gt) distribution for pass vs fail boxes, (c) how many frames each `frame_decision` bucket hits — so a human sets `TAU`/`MAX_BAD`. No new importable API. Helper `iter_label_boxes(img_path, label_path) -> list[tuple[str, np.ndarray]]` (gt name + pixel crop) is defined in the script and reused conceptually by Task 5 (each re-implements its own crop read; keep them independent — do not cross-import between scripts).

- [ ] **Step 1: Write the failing test (golden frames)**

Create `tests/test_consistency_golden.py`:

```python
"""Integration: the gate flags the known mid-flight frame and passes the clean one.
Requires production weights + the ai_run_1 dataset frames on disk."""
import os
import numpy as np
import cv2

from majsoul_eye.recognize.classifier import TileClassifier
from majsoul_eye.annotate.consistency import score_frame, frame_decision
from majsoul_eye.tiles import TILE_NAMES

IMG = "datasets/precise_ai_run_1/yolo/images"
LBL = "datasets/precise_ai_run_1/yolo/labels"


def _load(seq):
    img = cv2.imread(f"{IMG}/{seq}.png")
    h, w = img.shape[:2]
    gts, crops = [], []
    with open(f"{LBL}/{seq}.txt") as f:
        for line in f:
            line = line.split()
            if not line:
                continue
            cls, cx, cy, bw, bh = int(line[0]), *[float(x) for x in line[1:5]]
            x0 = int((cx - bw / 2) * w); y0 = int((cy - bh / 2) * h)
            x1 = int((cx + bw / 2) * w); y1 = int((cy + bh / 2) * h)
            crop = img[max(0, y0):y1, max(0, x0):x1]
            if crop.size:
                gts.append(TILE_NAMES[cls]); crops.append(crop)
    return crops, gts


def test_golden_bad_and_good_frames():
    if not os.path.exists(f"{IMG}/000034.png"):
        print("test_golden_bad_and_good_frames SKIP (dataset frames absent)")
        return
    clf = TileClassifier()
    bad_crops, bad_gts = _load("000034")
    good_crops, good_gts = _load("000567")
    bad_dec = frame_decision(score_frame(bad_crops, bad_gts, clf))
    good_dec = frame_decision(score_frame(good_crops, good_gts, clf))
    assert bad_dec[0] != "keep", bad_dec        # mid-flight frame must be flagged
    assert good_dec[0] == "keep", good_dec       # settled frame must pass clean
    print(f"test_golden_bad_and_good_frames OK  bad={bad_dec[0]} good={good_dec[0]}")


if __name__ == "__main__":
    test_golden_bad_and_good_frames()
    print("ALL test_consistency_golden OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. $PY tests/test_consistency_golden.py`
Expected: FAIL — either the assert fails (thresholds not yet calibrated) or passes trivially; if it FAILS on `good_dec == keep` / `bad_dec != keep`, that is the signal to tune `TAU`/`MAX_BAD` in Step 4. (If both asserts already pass with provisional TAU=0.5/MAX_BAD=2, note that and proceed.)

- [ ] **Step 3: Write the calibration script**

Create `scripts/inspect/calibrate_occlusion_gate.py`:

```python
"""Scan a built YOLO dataset with the consistency gate and print distributions to
pick TAU / MAX_BAD. Read-only (never deletes). Run from repo root.

  PYTHONPATH=. $PY scripts/inspect/calibrate_occlusion_gate.py --datasets datasets/precise_ai_run_1
"""
from __future__ import annotations
import argparse, glob, os
from collections import Counter

import cv2
import numpy as np

from majsoul_eye.recognize.classifier import TileClassifier
from majsoul_eye.annotate.consistency import score_frame, frame_decision, TAU, MAX_BAD
from majsoul_eye.tiles import TILE_NAMES


def iter_label_boxes(img_path: str, label_path: str):
    img = cv2.imread(img_path)
    if img is None or not os.path.exists(label_path):
        return [], []
    h, w = img.shape[:2]
    gts, crops = [], []
    for line in open(label_path, encoding="utf-8"):
        f = line.split()
        if not f:
            continue
        cls = int(f[0]); cx, cy, bw, bh = (float(x) for x in f[1:5])
        x0 = max(0, int((cx - bw / 2) * w)); y0 = max(0, int((cy - bh / 2) * h))
        x1 = int((cx + bw / 2) * w); y1 = int((cy + bh / 2) * h)
        crop = img[y0:y1, x0:x1]
        if crop.size:
            gts.append(TILE_NAMES[cls]); crops.append(crop)
    return crops, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=sorted(glob.glob("datasets/precise_*")))
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--max-bad", type=int, default=MAX_BAD)
    args = ap.parse_args()

    clf = TileClassifier()
    buckets = Counter(); badhist = Counter(); pass_conf = []; fail_conf = []
    for ds in args.datasets:
        for img_path in sorted(glob.glob(os.path.join(ds, "yolo", "images", "*.png"))):
            seq = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(ds, "yolo", "labels", f"{seq}.txt")
            crops, gts = iter_label_boxes(img_path, label_path)
            if not crops:
                continue
            vs = score_frame(crops, gts, clf, tau=args.tau)
            for v in vs:
                (pass_conf if v.ok else fail_conf).append(v.conf)
            nbad = sum(1 for v in vs if not v.ok)
            badhist[min(nbad, 5)] += 1
            buckets[frame_decision(vs, max_bad=args.max_bad)[0]] += 1
    tot = sum(buckets.values())
    print(f"tau={args.tau} max_bad={args.max_bad}  frames={tot}")
    print("decision buckets:", dict(buckets))
    print("bad-boxes-per-frame histogram (5=5+):", dict(sorted(badhist.items())))
    def pct(a, q): return round(float(np.percentile(a, q)), 3) if a else None
    print(f"pass P(gt): median={pct(pass_conf,50)} p10={pct(pass_conf,10)}  (n={len(pass_conf)})")
    print(f"fail P(gt): median={pct(fail_conf,50)} p90={pct(fail_conf,90)}  (n={len(fail_conf)})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Calibrate, then set constants**

Run: `PYTHONPATH=. $PY scripts/inspect/calibrate_occlusion_gate.py`
Inspect the output. Choose `TAU`/`MAX_BAD` so `drop_frame` is a small minority and the golden test passes; if the provisional values already satisfy the golden test and the buckets look sane (most frames `keep`), leave them. If you change them, edit the `TAU` / `MAX_BAD` constants at the top of `majsoul_eye/annotate/consistency.py` and note the chosen values + the calibration output in the commit message.

- [ ] **Step 5: Run golden test to verify it passes**

Run: `PYTHONPATH=. $PY tests/test_consistency_golden.py`
Expected: PASS (`bad=drop_frame` or `drop_boxes`, `good=keep`).

- [ ] **Step 6: Commit**

```bash
git add scripts/inspect/calibrate_occlusion_gate.py tests/test_consistency_golden.py majsoul_eye/annotate/consistency.py
git commit -m "feat(inspect): occlusion-gate calibration + golden-frame test; set TAU/MAX_BAD"
```

---

### Task 5: One-time purge tool for existing datasets

**Files:**
- Create: `scripts/data/purge_occlusion_frames.py`
- Test: `tests/test_purge_occlusion.py`

**Interfaces:**
- Consumes: `score_frame`, `frame_decision` (Task 3); `TileClassifier` (Task 1); pattern of `scripts/data/purge_deal_frames.py`.
- Produces: a CLI mirroring `purge_deal_frames.py`. Dry-run default; `--apply` deletes; idempotent. For each `datasets/precise_*/`: for each `yolo/images/<seq>.png` + label, score; on `drop_boxes` rewrite the label without the bad lines (and delete each bad box's classifier crop `crops/<gt>/<seq>_*.png`); on `drop_frame` delete image + label + all `crops/*/<seq>_*.png`. Then rewrite `datasets/detector*/{train,val}.txt` dropping now-missing image lines. Testable helper: `plan_frame(img_path, label_path, clf, tau, max_bad) -> tuple[str, list[int]]` (the `frame_decision` result) so the test can assert routing without touching real data.

- [ ] **Step 1: Write the failing test**

Create `tests/test_purge_occlusion.py`:

```python
"""purge_occlusion_frames: routing + apply/idempotency on a tiny synthetic dataset."""
import os, glob, importlib.util
import numpy as np, cv2

# load the script as a module
_spec = importlib.util.spec_from_file_location("poc", "scripts/data/purge_occlusion_frames.py")
poc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(poc)

from majsoul_eye.tiles import NAME_TO_ID


class StubClf:
    def __init__(self, mapping): self.mapping = mapping   # crop-tag byte -> tile name
    def predict_proba(self, crops):
        out = np.zeros((len(crops), 38), np.float32)
        for i, c in enumerate(crops):
            out[i, NAME_TO_ID[self.mapping[int(c[0, 0, 0])]]] = 1.0
        return out


def _write(ds, seq, boxes):
    """boxes: list of (gt_cls_id, tag_byte). Writes image (tag encoded in pixel 0,0)
    + label. One box per row, all at distinct positions."""
    os.makedirs(f"{ds}/yolo/images", exist_ok=True)
    os.makedirs(f"{ds}/yolo/labels", exist_ok=True)
    img = np.full((100, 100, 3), 240, np.uint8)
    lines = []
    for k, (cls, tag) in enumerate(boxes):
        img[0, 0] = tag  # NOTE: single shared tag pixel is fine for 1-box frames in this test
        lines.append(f"{cls} {0.1+0.1*k:.3f} 0.5 0.05 0.05")
    cv2.imwrite(f"{ds}/yolo/images/{seq}.png", img)
    open(f"{ds}/yolo/labels/{seq}.txt", "w").write("\n".join(lines) + "\n")


def test_plan_frame_routing(tmpdir="scratch_purge_test"):
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    ds = f"{tmpdir}/precise_fake"
    # good frame: gt 8s, classifier sees 8s
    _write(ds, "000001", [(NAME_TO_ID["8s"], NAME_TO_ID["8s"])])
    # bad frame: gt 8s, classifier sees 3p
    _write(ds, "000002", [(NAME_TO_ID["8s"], NAME_TO_ID["3p"])])
    clf = StubClf({NAME_TO_ID["8s"]: "8s", NAME_TO_ID["3p"]: "3p"})
    d1 = poc.plan_frame(f"{ds}/yolo/images/000001.png", f"{ds}/yolo/labels/000001.txt", clf, 0.5, 2)
    d2 = poc.plan_frame(f"{ds}/yolo/images/000002.png", f"{ds}/yolo/labels/000002.txt", clf, 0.5, 2)
    assert d1[0] == "keep", d1
    assert d2[0] in ("drop_boxes", "drop_frame"), d2
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("test_plan_frame_routing OK")


if __name__ == "__main__":
    test_plan_frame_routing()
    print("ALL test_purge_occlusion OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. $PY tests/test_purge_occlusion.py`
Expected: FAIL — `FileNotFoundError`/`spec` error because `scripts/data/purge_occlusion_frames.py` does not exist yet.

- [ ] **Step 3: Write the purge tool**

Create `scripts/data/purge_occlusion_frames.py`:

```python
"""Delete/trim occlusion-corrupted YOLO boxes & frames from built datasets.

For each datasets/precise_*/ : crop every GT box from yolo/images/<seq>.png, run the
production classifier, and apply the consistency smart-drop (annotate.consistency):
- keep       -> untouched
- drop_boxes -> rewrite the label without the bad lines; delete those boxes' crops
- drop_frame -> delete image + label + all crops for the seq
Then rewrite datasets/detector*/{train,val}.txt dropping now-missing image lines.

Dry-run by DEFAULT (prints planned actions); --apply to act. Idempotent.

  PYTHONPATH=. $PY scripts/data/purge_occlusion_frames.py            # dry-run
  PYTHONPATH=. $PY scripts/data/purge_occlusion_frames.py --apply
"""
from __future__ import annotations
import argparse, glob, os

import cv2

from majsoul_eye.recognize.classifier import TileClassifier
from majsoul_eye.annotate.consistency import score_frame, frame_decision, TAU, MAX_BAD
from majsoul_eye.tiles import TILE_NAMES


def _read_boxes(img_path, label_path):
    """-> (raw_lines, gts, crops) aligned by index (only rows with a nonempty crop)."""
    img = cv2.imread(img_path)
    raw, gts, crops = [], [], []
    if img is None or not os.path.exists(label_path):
        return raw, gts, crops, img
    h, w = img.shape[:2]
    for line in open(label_path, encoding="utf-8"):
        if not line.split():
            continue
        f = line.split()
        cls = int(f[0]); cx, cy, bw, bh = (float(x) for x in f[1:5])
        x0 = max(0, int((cx - bw / 2) * w)); y0 = max(0, int((cy - bh / 2) * h))
        x1 = int((cx + bw / 2) * w); y1 = int((cy + bh / 2) * h)
        crop = img[y0:y1, x0:x1]
        if crop.size:
            raw.append(line.rstrip("\n")); gts.append(TILE_NAMES[cls]); crops.append(crop)
    return raw, gts, crops, img


def plan_frame(img_path, label_path, clf, tau=TAU, max_bad=MAX_BAD):
    raw, gts, crops, _ = _read_boxes(img_path, label_path)
    if not crops:
        return "keep", []
    return frame_decision(score_frame(crops, gts, clf, tau=tau), max_bad=max_bad)


def _crops_for(ds, gt, seq):
    return glob.glob(os.path.join(ds, "crops", gt, f"{seq}_*.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-dir", default="datasets")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--max-bad", type=int, default=MAX_BAD)
    args = ap.parse_args()

    clf = TileClassifier()
    tot_boxes = tot_frames = 0
    for ds in sorted(glob.glob(os.path.join(args.datasets_dir, "precise_*"))):
        for img_path in sorted(glob.glob(os.path.join(ds, "yolo", "images", "*.png"))):
            seq = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(ds, "yolo", "labels", f"{seq}.txt")
            raw, gts, crops, _ = _read_boxes(img_path, label_path)
            if not crops:
                continue
            decision, bad = frame_decision(score_frame(crops, gts, clf, tau=args.tau), max_bad=args.max_bad)
            if decision == "keep":
                continue
            if decision == "drop_frame":
                tot_frames += 1
                victims = [img_path, label_path]
                for gt in set(gts):
                    victims += _crops_for(ds, gt, seq)
                print(f"{os.path.basename(ds)}/{seq}: DROP FRAME ({len(bad)} bad of {len(gts)})")
                if args.apply:
                    for v in victims:
                        if os.path.exists(v):
                            os.remove(v)
            else:  # drop_boxes
                tot_boxes += len(bad)
                kept = [ln for i, ln in enumerate(raw) if i not in set(bad)]
                badtiles = [gts[i] for i in bad]
                print(f"{os.path.basename(ds)}/{seq}: drop {len(bad)} box(es) {badtiles} ({len(gts)}->{len(kept)})")
                if args.apply:
                    with open(label_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(kept) + ("\n" if kept else ""))
                    for gt in set(badtiles):
                        for c in _crops_for(ds, gt, seq):
                            os.remove(c)

    # Fix assembled detector splits: drop lines whose image no longer exists.
    for lst in glob.glob(os.path.join(args.datasets_dir, "detector*", "*.txt")):
        if os.path.basename(lst) not in ("train.txt", "val.txt"):
            continue
        lines = [ln.rstrip("\n") for ln in open(lst, encoding="utf-8") if ln.strip()]
        kept = [ln for ln in lines if os.path.exists(ln)]
        if len(kept) != len(lines):
            print(f"{lst}: drop {len(lines) - len(kept)} missing-image lines ({len(lines)} -> {len(kept)})")
            if args.apply:
                with open(lst, "w", encoding="utf-8") as f:
                    f.write("\n".join(kept) + ("\n" if kept else ""))

    mode = "DELETED" if args.apply else "would delete (dry-run; pass --apply)"
    print(f"\nTOTAL {mode}: boxes={tot_boxes}  frames={tot_frames}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. $PY tests/test_purge_occlusion.py`
Expected: PASS (`ALL test_purge_occlusion OK`).

- [ ] **Step 5: Commit**

```bash
git add scripts/data/purge_occlusion_frames.py tests/test_purge_occlusion.py
git commit -m "feat(data): purge_occlusion_frames — one-time consistency clean of built datasets"
```

---

### Task 6: Build-time hook in build_dataset

**Files:**
- Modify: `scripts/train/build_dataset.py` (imports ~line 68-70; box loop ~line 132-160)
- Test: `tests/test_build_gate.py`

**Interfaces:**
- Consumes: `score_frame`, `frame_decision` (Task 3); `TileClassifier` (Task 1); existing `iter_tile_boxes` / `crop_box` / `AnnBox` (`box.tile`, `box.reliable`, `box.sideways`).
- Produces: a gated build. New CLI flag `--occlusion-gate` (default ON) with `--no-occlusion-gate` to disable, and `--occ-tau` / `--occ-max-bad`. New helper (module-level in `build_dataset.py`) `gate_frame(frame, boxes, clf, tau, max_bad) -> set[int]` returning indices of `boxes` to SKIP (empty set = keep all; all indices = drop whole frame). Emits a `n_occluded_boxes` / `n_occluded_frames` counter alongside `n_deal`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_gate.py`:

```python
"""build_dataset.gate_frame: returns the indices of boxes to skip."""
import importlib.util
import numpy as np

_spec = importlib.util.spec_from_file_location("bd", "scripts/train/build_dataset.py")
bd = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bd)

from majsoul_eye.tiles import NAME_TO_ID


class _Box:
    def __init__(self, tile): self.tile = tile


class StubClf:
    def __init__(self, seen): self.seen = seen
    def predict_proba(self, crops):
        out = np.zeros((len(crops), 38), np.float32)
        for i in range(len(crops)):
            out[i, NAME_TO_ID[self.seen[i]]] = 1.0
        return out


def test_gate_frame_skips_bad_boxes():
    # frame with 3 boxes; middle one mislabeled -> skip index 1
    frame = np.full((100, 100, 3), 240, np.uint8)
    boxes = [_Box("8s"), _Box("3p"), _Box("S")]
    crops = [np.full((32, 32, 3), 240, np.uint8) for _ in range(3)]
    clf = StubClf(["8s", "9m", "S"])
    skip = bd.gate_frame(frame, boxes, crops, clf, tau=0.5, max_bad=2)
    assert skip == {1}, skip
    print("test_gate_frame_skips_bad_boxes OK")


if __name__ == "__main__":
    test_gate_frame_skips_bad_boxes()
    print("ALL test_build_gate OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. $PY tests/test_build_gate.py`
Expected: FAIL with `AttributeError: module 'bd' has no attribute 'gate_frame'`.

- [ ] **Step 3: Implement the gate helper and wire it in**

In `scripts/train/build_dataset.py`, add near the other imports (after line 70):

```python
    from majsoul_eye.annotate.consistency import score_frame, frame_decision, TAU as OCC_TAU, MAX_BAD as OCC_MAX_BAD
```

Add a module-level helper (top level of the file, not nested):

```python
def gate_frame(frame, boxes, crops, clf, tau, max_bad):
    """Return the set of box indices to SKIP for occlusion/mislabel. `boxes` and
    `crops` are aligned; a whole-frame drop returns every index."""
    if clf is None or not boxes:
        return set()
    gts = [b.tile for b in boxes]
    decision, bad = frame_decision(score_frame(crops, gts, clf, tau=tau), max_bad=max_bad)
    if decision == "keep":
        return set()
    if decision == "drop_frame":
        return set(range(len(boxes)))
    return set(bad)
```

Add CLI flags near the other `ap.add_argument(...)` calls:

```python
    ap.add_argument("--no-occlusion-gate", dest="occlusion_gate", action="store_false")
    ap.set_defaults(occlusion_gate=True)
    ap.add_argument("--occ-tau", type=float, default=OCC_TAU)
    ap.add_argument("--occ-max-bad", type=int, default=OCC_MAX_BAD)
```

Load the classifier once (near where the loop/counters are initialized, alongside `n_deal`):

```python
    n_occ_box = n_occ_frame = 0
    occ_clf = None
    if args.occlusion_gate:
        from majsoul_eye.recognize.classifier import TileClassifier
        occ_clf = TileClassifier()
```

Now gate the box loop. The current loop (≈lines 132-160) is:

```python
        yolo_lines = []
        ci = 0
        for box in iter_tile_boxes(rec):
            if not box.reliable:
                continue
            ...
            if not args.no_crops and not box.sideways:
                crop = crop_box(frame, box, size=args.crop_size)
                ...
```

Replace it with a two-pass version — first collect reliable boxes + their gate crops, decide skips, then emit:

```python
        reliable = [b for b in iter_tile_boxes(rec) if b.reliable]
        skip = set()
        if occ_clf is not None and reliable:
            gate_crops = [crop_box(frame, b, size=args.crop_size) for b in reliable]
            skip = gate_frame(frame, reliable, gate_crops, occ_clf,
                              args.occ_tau, args.occ_max_bad)
            if skip == set(range(len(reliable))):
                n_occ_frame += 1
            else:
                n_occ_box += len(skip)

        yolo_lines = []
        ci = 0
        for bi, box in enumerate(reliable):
            if bi in skip:
                continue
            ...  # existing per-box body: YOLO line append + classifier crop write
```

(Keep the existing per-box body verbatim inside the new loop — the YOLO-line append and the `if not args.no_crops and not box.sideways: crop = crop_box(...)` crop write. Note the gate reuses `crop_box`; the crop write still recomputes its own crop, which is fine.)

Update the final stats print to include `n_occ_box` / `n_occ_frame` next to `n_deal`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. $PY tests/test_build_gate.py`
Expected: PASS.

- [ ] **Step 5: Regression-run the touched suites**

Run: `PYTHONPATH=. $PY tests/test_consistency.py && PYTHONPATH=. $PY tests/test_classifier.py`
Expected: both PASS (no import breakage from the edits).

- [ ] **Step 6: Commit**

```bash
git add scripts/train/build_dataset.py tests/test_build_gate.py
git commit -m "feat(build): occlusion consistency gate in build_dataset (default on)"
```

---

### Task 7: Shared table-ROI frame diff (capture)

**Files:**
- Create: `majsoul_eye/capture/roi_diff.py`
- Test: `tests/test_roi_diff.py`

**Interfaces:**
- Consumes: numpy only.
- Produces:
  - `TABLE_ROI = (0.18, 0.16, 0.82, 0.92)` — normalized (x0,y0,x1,y1) of the play surface, excluding the animated cloth border and the top/side 2D HUD.
  - `roi_diff(a: np.ndarray, b: np.ndarray, roi=TABLE_ROI) -> float` — mean-abs pixel diff computed ONLY inside `roi` (shape-mismatch → large sentinel `1e9`). This is the animation-motion signal used by both capture paths.

- [ ] **Step 1: Write the failing test**

Create `tests/test_roi_diff.py`:

```python
import numpy as np
from majsoul_eye.capture.roi_diff import roi_diff, TABLE_ROI


def test_identical_is_zero():
    a = np.random.RandomState(0).randint(0, 255, (90, 160, 3), np.uint8)
    assert roi_diff(a, a) == 0.0
    print("test_identical_is_zero OK")


def test_change_outside_roi_ignored():
    a = np.zeros((100, 100, 3), np.uint8)
    b = a.copy()
    b[:10, :] = 255           # top HUD band, outside TABLE_ROI y0=0.16
    assert roi_diff(a, b) == 0.0, roi_diff(a, b)
    print("test_change_outside_roi_ignored OK")


def test_change_inside_roi_detected():
    a = np.zeros((100, 100, 3), np.uint8)
    b = a.copy()
    b[40:60, 40:60] = 255     # center, inside ROI
    assert roi_diff(a, b) > 1.0
    print("test_change_inside_roi_detected OK")


def test_shape_mismatch_sentinel():
    a = np.zeros((100, 100, 3), np.uint8)
    b = np.zeros((90, 90, 3), np.uint8)
    assert roi_diff(a, b) >= 1e8
    print("test_shape_mismatch_sentinel OK")


if __name__ == "__main__":
    test_identical_is_zero(); test_change_outside_roi_ignored()
    test_change_inside_roi_detected(); test_shape_mismatch_sentinel()
    print("ALL test_roi_diff OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. $PY tests/test_roi_diff.py`
Expected: FAIL with `ModuleNotFoundError: majsoul_eye.capture.roi_diff`.

- [ ] **Step 3: Write minimal implementation**

Create `majsoul_eye/capture/roi_diff.py`:

```python
"""Table-ROI frame diff: the discard-animation motion signal for capture stability.

Restricting the diff to the play surface excludes the always-animating cloth border
and the 2D HUD, so the threshold can be tight enough to catch a moving tile/arm
without being tripped by decoration. Shared by autoplay_ai and FrameSyncer.
"""
from __future__ import annotations

import numpy as np

# normalized (x0, y0, x1, y1) of the play surface (canonical 16:9), HUD/border excluded.
TABLE_ROI = (0.18, 0.16, 0.82, 0.92)


def roi_diff(a: np.ndarray, b: np.ndarray, roi=TABLE_ROI) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 1e9
    h, w = a.shape[:2]
    x0, y0, x1, y1 = roi
    xa, ya, xb, yb = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    ra = a[ya:yb, xa:xb].astype(np.int16)
    rb = b[ya:yb, xa:xb].astype(np.int16)
    if ra.size == 0:
        return 1e9
    return float(np.mean(np.abs(ra - rb)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. $PY tests/test_roi_diff.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/capture/roi_diff.py tests/test_roi_diff.py
git commit -m "feat(capture): table-ROI frame diff for animation-stability confirm"
```

---

### Task 8: A1 — pixel-stability confirm in the AI capture path

**Files:**
- Modify: `scripts/capture/autoplay_ai.py` (`maybe_screenshot`, ≈line 338-348; argparse ≈line 80)
- Test: `tests/test_autoplay_stability.py`

**Interfaces:**
- Consumes: `roi_diff` (Task 7).
- Produces: a testable pure decision `stable_capture_step(state, frame, thresh) -> tuple[str, dict]` where `state` is a small dict `{"ref": np.ndarray|None}`, `frame` is the freshly grabbed image, and the return `action` is `"save"` (picture settled) or `"wait"` (moved since last grab → store as new ref, try next tick). `maybe_screenshot` calls it after the quiet gate.

- [ ] **Step 1: Write the failing test**

Create `tests/test_autoplay_stability.py`:

```python
"""AI-path stability: only save once two consecutive grabs match within the table ROI."""
import importlib.util
import numpy as np

_spec = importlib.util.spec_from_file_location("ap", "scripts/capture/autoplay_ai.py")
ap = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ap)


def test_waits_for_stability():
    a = np.zeros((100, 100, 3), np.uint8)
    moving = a.copy(); moving[40:60, 40:60] = 255      # tile mid-flight in ROI
    settled = a.copy()                                  # animation done

    st = {"ref": None}
    act, st = ap.stable_capture_step(st, moving, thresh=3.0)
    assert act == "wait"                                # first grab -> set ref, wait
    act, st = ap.stable_capture_step(st, settled, thresh=3.0)
    assert act == "wait"                                # differs from moving -> still wait
    act, st = ap.stable_capture_step(st, settled, thresh=3.0)
    assert act == "save"                                # two identical grabs -> save
    print("test_waits_for_stability OK")


if __name__ == "__main__":
    test_waits_for_stability()
    print("ALL test_autoplay_stability OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. $PY tests/test_autoplay_stability.py`
Expected: FAIL with `AttributeError: module 'ap' has no attribute 'stable_capture_step'`.

- [ ] **Step 3: Implement the decision and wire it into `maybe_screenshot`**

In `scripts/capture/autoplay_ai.py`, add near the top-level helpers (module scope):

```python
from majsoul_eye.capture.roi_diff import roi_diff  # add with the other majsoul_eye imports


def stable_capture_step(state, frame, thresh):
    """Return ("save"|"wait", state). "save" once the current grab matches the previous
    one inside the table ROI (discard animation finished); else store ref and wait."""
    ref = state.get("ref")
    if ref is not None and roi_diff(frame, ref) <= thresh:
        return "save", {"ref": None}
    return "wait", {"ref": frame}
```

Add the CLI flag near the existing `--quiet` argument (≈line 80):

```python
    ap.add_argument("--stable-thresh", type=float, default=3.0,
                    help="Table-ROI frame-diff below this == settled (discard animation done).")
```

Decode the grabbed PNG to an array and gate on stability inside `maybe_screenshot` (replace the body that currently saves unconditionally after the quiet check):

```python
    _stab = {"ref": None}

    def maybe_screenshot():
        nonlocal pending_seq, fulfilled_seq, _stab
        if game_frames_dir is None or pending_seq is None or pending_seq == fulfilled_seq:
            return
        if (time.time() - last_event_t) < args.quiet:
            return
        png = screenshot_png()
        if not png:
            return
        import numpy as np, cv2
        arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        action, _stab = stable_capture_step(_stab, arr, args.stable_thresh)
        if action == "wait":
            return                                  # picture still moving; retry next tick
        with open(os.path.join(game_frames_dir, f"{pending_seq:06d}.png"), "wb") as fh:
            fh.write(png)
        fulfilled_seq = pending_seq
        _stab = {"ref": None}
```

(Ensure `_stab` is declared in the enclosing scope before `maybe_screenshot` is defined; the `nonlocal` above assumes it. If the linter flags `nonlocal _stab` because it is assigned in the outer function, keep `_stab` as a mutable dict and mutate it instead of rebinding.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. $PY tests/test_autoplay_stability.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/capture/autoplay_ai.py tests/test_autoplay_stability.py
git commit -m "feat(capture): AI-path pixel-stability confirm (wait out discard animation)"
```

---

### Task 9: A1 — tighten FrameSyncer (manual path)

**Files:**
- Modify: `majsoul_eye/capture/sync.py` (`frame_diff` usage in `_maybe_capture`, ≈line 183-196; `__init__` defaults)
- Test: `tests/test_sync.py` (append)

**Interfaces:**
- Consumes: `roi_diff` (Task 7).
- Produces: `FrameSyncer` uses `roi_diff` (table-ROI) instead of full-frame `frame_diff` for the stability confirm, and on the `capped` path does ONE stability check instead of bypassing it outright. Preserves the existing injected `grab`/`now`/`sleep` testability. `frame_diff` stays exported (diagnostics) but the confirm path calls `roi_diff`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sync.py` (add call to its `__main__`):

```python
def test_capped_still_confirms_once():
    """On a capped burst, FrameSyncer no longer saves a mid-animation frame outright:
    it requires one ROI-stable confirmation."""
    import numpy as np
    from majsoul_eye.capture.sync import FrameSyncer

    moving = np.zeros((100, 100, 3), np.uint8); moving[40:60, 40:60] = 255
    settled = np.zeros((100, 100, 3), np.uint8)
    grabs = [moving, moving, settled, settled]
    clock = {"t": 0.0}

    def grab():
        return grabs[min(len(grabs) - 1, int(clock["t"] / 0.05))]

    saved = []
    fs = FrameSyncer(grab, out_dir="scratch_sync_test", quiet=0.30, settle_cap=0.10,
                     now=lambda: clock["t"], sleep=lambda s: clock.__setitem__("t", clock["t"] + 0.05),
                     on_pair=lambda k, p, s: saved.append((k, s)))
    # simulate: event at t=0, then drive _maybe_capture past settle_cap
    fs.on_event(1)
    for _ in range(8):
        fs._maybe_capture(); clock["t"] += 0.05
    # it must have waited for a settled (ROI-stable) frame before recording "ok"
    oks = [s for _, s in saved if s == "ok"]
    assert oks, "expected a capture after stability confirmed"
    print("test_capped_still_confirms_once OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. $PY tests/test_sync.py`
Expected: FAIL (current code captures the first `moving` frame on the capped path with no confirm, or the ROI logic isn't wired — assertion/behavior mismatch). If it happens to pass, adjust `settle_cap`/grab sequence so the test exercises the capped branch and fails against current behavior before implementing.

- [ ] **Step 3: Implement the change**

In `majsoul_eye/capture/sync.py`:

Add import near the top:

```python
from .roi_diff import roi_diff
```

In `_maybe_capture`, replace the stability block (≈lines 185-190) so the confirm runs even when capped and uses `roi_diff`:

```python
        # Confirm the picture stopped moving (discard animation finished), comparing to
        # the previous tick's grab for this key — inside the table ROI so the animated
        # cloth doesn't defeat it. Runs on the capped path too (one confirmation), so a
        # long burst still waits out the sweep instead of grabbing mid-animation.
        if self.confirm_stable:
            if self._ref_key == key and self._ref is not None and roi_diff(frame, self._ref) <= self.diff_thresh:
                pass  # stable
            else:
                self._ref, self._ref_key = frame, key
                return False
```

(Remove the `and not capped` guard from the `if self.confirm_stable` condition; keep everything else.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. $PY tests/test_sync.py`
Expected: PASS (all existing sync tests still pass + the new one).

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/capture/sync.py tests/test_sync.py
git commit -m "fix(capture): FrameSyncer confirms ROI-stability even on capped bursts"
```

---

## Post-implementation (run manually, not a coding task)

After all tasks pass, run the actual cleanup and retrain (these mutate real data — do with the user, not autonomously):

```bash
# 1) dry-run the purge, review the counts + which frames/boxes it targets
PYTHONPATH=. $PY scripts/data/purge_occlusion_frames.py
# 2) apply, then re-check in FiftyOne (scripts/inspect/fiftyone_view.py --rebuild)
PYTHONPATH=. $PY scripts/data/purge_occlusion_frames.py --apply
# 3) retrain the detector on the cleaned split; compare mAP to the pre-purge baseline.
```

---

## Self-Review

**Spec coverage:**
- Spec §5.1 consistency scorer → Tasks 1 (predict_proba), 2 (verdict/empty-felt), 3 (score_frame). ✓
- Spec §5.2 smart-drop rule (M / whole-frame) → Task 3 `frame_decision`. ✓
- Spec §5.3 purge tool → Task 5; build-time hook → Task 6. ✓
- Spec §6 A1 AI path → Task 8; manual path patch → Task 9; shared ROI diff → Task 7. ✓
- Spec §7 calibration (TAU/MAX_BAD, golden frames) → Task 4. ✓
- Spec §8 testing → each task is TDD; golden frames Task 4; purge idempotency Task 5; A1 injectable tests Tasks 8-9. ✓
- Spec §10 detector-split rewrite after drops → Task 5 + Task 6 (build emits clean labels; purge rewrites splits). ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. TAU/MAX_BAD are real provisional constants with an explicit calibration step (Task 4), not placeholders.

**Type consistency:** `predict_proba -> (N,38) np.ndarray` used identically in Tasks 3/5/6 stubs and real calls. `BoxVerdict` fields (`ok/gt/pred/conf/reason`) consistent across Tasks 2/3/5. `frame_decision -> (str, list[int])` with buckets `keep|drop_boxes|drop_frame` consistent in Tasks 3/5/6. `roi_diff(a,b,roi)` signature consistent across Tasks 7/8/9. `gate_frame(frame, boxes, crops, clf, tau, max_bad) -> set[int]` matches its test and its call site.

**Note on Task 5 test:** the `_write` helper shares one tag pixel per image, so its routing test uses one box per frame (sufficient to exercise keep vs drop). Multi-box routing is covered by Task 3's `test_score_frame_and_decision` with the stub classifier.
