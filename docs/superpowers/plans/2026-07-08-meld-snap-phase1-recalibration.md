# Meld-snap Phase 1: corner recalibration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the systematic ~half-tile mis-calibration of the far-seat meld
corners so `snap_meld_strip` no longer sits at the aliasing midpoint — collapsing
pos3's one-tile lock-flip rate from ~26% to ~1% ("上家副露严重失位").

**Architecture:** Add a committed QA guard that measures each seat's systematic
meld-snap offset + mislock rate over real captures. Re-run the existing calibrator
to derive corrected `MELD_STRIP2` corners, apply them, and prove the guard passes.
No pipeline-shape change — only calibration constants move (which re-stales datasets).

**Tech Stack:** Python 3, OpenCV, NumPy; conda `auto` env; repo-root `PYTHONPATH=.`.
Tests are plain scripts under `tests/` (no pytest dependency), run with
`PYTHONPATH=. python tests/test_x.py`.

## Global Constraints

- Run everything in the conda `auto` env. In a shell where `auto` is NOT activated
  (this harness's Bash), substitute `C:/Users/zsx/miniforge3/envs/auto/python.exe`
  for `python`. Always `PYTHONPATH=.` from repo root `D:\code\phoenix\majsoul_eye`.
- Work on branch `fix/meld-snap-stability` (already created; spec at
  `docs/superpowers/specs/2026-07-08-meld-snap-recalibration-consensus-design.md`).
- **38-class / seat conventions are frozen.** Do NOT touch `tiles.py`, seat mapping,
  or `DISCARD_GRID`. Phase 1 changes ONLY `MELD_STRIP2[seat]["corner"]` values.
- **Pipeline-impact discipline (CLAUDE.md):** changing `MELD_STRIP2` re-stales every
  meld box in `datasets/*` and `datasets/backs_review`. This plan updates
  `docs/STATUS.md` (+ `docs/PIPELINE.md` if it references the calibration) and
  documents the rebuild command; the actual dataset rebuild + model retrain is a
  deliberate manual step the user runs (Task 4, not auto-executed).
- `recognize/` stays untouched and Akagi-free.
- Measured targets to expect (ai_session + ai_session3, from the investigation):
  pos3 corner `(625.0, 1797.6) → ~(624.5, 1751.6)` (y −46 along), pos0 corner
  `(2388.2, 1889.5) → ~(2388.2, 1843.5)` (y −46 cross). pos1/pos2 offsets <2px → leave.

---

## File Structure

- Create: `scripts/annotate/meld_snap_qa.py` — QA guard: per-seat dominant snap
  offset + mislock rate over captures; exits nonzero if any seat > `--max-mislock`.
- Create: `tests/test_meld_snap_qa.py` — fast unit test for the `dominant()` helper
  (no capture data).
- Modify: `majsoul_eye/annotate/pipeline.py:458-463` — the `MELD_STRIP2` dict
  (corner values for pos3 and pos0).
- Modify: `docs/STATUS.md` — new §1.50 entry.
- Modify (conditional): `docs/PIPELINE.md` — only if it references MELD_STRIP2 /
  meld calibration.

---

## Task 1: QA guard for meld-snap mislock rate

**Files:**
- Create: `scripts/annotate/meld_snap_qa.py`
- Test: `tests/test_meld_snap_qa.py`

**Interfaces:**
- Produces: `dominant(vals: list[float], tol: float = 12.0) -> tuple[float, float]`
  returning `(center, frac_in_densest_cluster)`; `([]) -> (0.0, 0.0)`.
- Produces: a CLI `meld_snap_qa.py --sources <roots...> --max-mislock <f>` that
  prints a per-seat table and `sys.exit(1)` when the worst seat mislock exceeds the
  threshold.

- [ ] **Step 1: Write the failing unit test**

Create `tests/test_meld_snap_qa.py`:

```python
"""Fast unit test for the meld_snap_qa clustering helper (no capture data)."""
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "annotate"))
from meld_snap_qa import dominant  # noqa: E402


def test_dominant_majority_over_outliers():
    # 70 frames locked at +46, 30 mislocked negative -> center ~46, frac ~0.7
    c, f = dominant([46.0] * 70 + [-20.0] * 30)
    assert abs(c - 46.0) < 2.0, c
    assert 0.65 <= f <= 0.75, f


def test_dominant_tight_cluster():
    c, f = dominant([5.0, 5.5, 4.5, 5.2])
    assert abs(c - 5.0) < 1.0, c
    assert f == 1.0, f


def test_dominant_empty():
    assert dominant([]) == (0.0, 0.0)


if __name__ == "__main__":
    for _name, _fn in sorted(list(globals().items())):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(_name, "OK")
    print("all meld_snap_qa tests passed")
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe tests/test_meld_snap_qa.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'meld_snap_qa'`.

- [ ] **Step 3: Create the QA guard script**

Create `scripts/annotate/meld_snap_qa.py`:

```python
"""QA guard: per-seat systematic meld-snap offset + mislock rate over captures.

Run after ANY change to MELD_STRIP2 / the fullwarp / the tile masks. A large
per-seat offset means a corner is mis-calibrated; a high mislock rate means the
snap sits at the aliasing midpoint and flips one tile (STATUS §1.50). Exits
nonzero if the worst seat mislock exceeds --max-mislock.

  PYTHONPATH=. python scripts/annotate/meld_snap_qa.py
  PYTHONPATH=. python scripts/annotate/meld_snap_qa.py --sources captures/raw/ai_session captures/raw/ai_session3
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

from majsoul_eye.annotate import build_homographies
from majsoul_eye.annotate import pipeline as P
from majsoul_eye.annotate.seatgt import seat_gt
from majsoul_eye.capture.gtframes import build_seq_state, load_frames
from majsoul_eye.state.replay import check_invariants, is_call_window, is_deal_window


def dominant(vals, tol=12.0):
    """(center, frac_in_densest_cluster) for a 1-D list within +-tol. ([] -> 0,0)."""
    if not vals:
        return 0.0, 0.0
    v = np.array(sorted(vals))
    best = (float(v[0]), 0)
    for x in v:
        inb = v[(v >= x - tol) & (v <= x + tol)]
        if len(inb) > best[1]:
            best = (float(np.median(inb)), len(inb))
    return round(best[0], 1), round(best[1] / len(v), 3)


def measure(sources):
    hom = build_homographies(1920, 1080)
    Hinv = hom["H_full_inv"]
    caps = []
    for root in sources:
        caps += sorted(glob.glob(os.path.join(root, "run_*", "game*", "game*.jsonl")))
    da = defaultdict(list)
    dc = defaultdict(list)
    for cap in caps:
        try:
            ss = build_seq_state(cap)
            fr = load_frames(os.path.dirname(cap))
        except Exception:
            continue
        for s in sorted(ss):
            if s not in fr:
                continue
            st = ss[s]
            if is_deal_window(st) or is_call_window(st) or check_invariants(st):
                continue
            if not any(seat_gt(st, p)[2] for p in range(4)):
                continue
            img = cv2.imread(fr[s])
            if img is None:
                continue
            if img.shape[1] != 1920:
                img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
            full = P.warp_to_full(img, hom["H_full"], hom["full_size"])
            hsv = cv2.cvtColor(full, cv2.COLOR_BGR2HSV)
            mw = P.tile_face_mask(hsv=hsv)
            mb = P.tile_back_mask(hsv=hsv)
            for pos in range(4):
                _, _, melds, _ = seat_gt(st, pos)
                if not melds:
                    continue
                boxes = P.generate_meld_boxes_v2(pos, melds, Hinv)
                if not boxes:
                    continue
                a, c, diag = P.snap_meld_strip(mw, mb, boxes, pos)
                if diag["n_along"] + diag["n_cross"] >= 4 and diag["score"] >= 3.0:
                    da[pos].append(a)
                    dc[pos].append(c)
    return da, dc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["captures/raw/ai_session"])
    ap.add_argument("--max-mislock", type=float, default=0.03)
    args = ap.parse_args()
    da, dc = measure(args.sources)
    worst = 0.0
    print(f"{'pos':>3} {'N':>5} | {'da_off':>7} {'da_mis':>7} | {'dc_off':>7} {'dc_mis':>7}")
    for pos in range(4):
        if not da[pos]:
            print(f"{pos:>3}     0 | (no confident meld frames)")
            continue
        dao, daf = dominant(da[pos])
        dco, dcf = dominant(dc[pos])
        damis, dcmis = round(1 - daf, 3), round(1 - dcf, 3)
        worst = max(worst, damis, dcmis)
        print(f"{pos:>3} {len(da[pos]):>5} | {dao:>7} {damis:>7} | {dco:>7} {dcmis:>7}")
    print(f"worst mislock={worst:.3f} (threshold {args.max_mislock})")
    if worst > args.max_mislock:
        print("FAIL: a seat exceeds the mislock threshold — check MELD_STRIP2 corners.")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe tests/test_meld_snap_qa.py`
Expected: `all meld_snap_qa tests passed`.

- [ ] **Step 5: Capture the BASELINE (pre-recalibration) — this documents the bug**

Run: `PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe scripts/annotate/meld_snap_qa.py --sources captures/raw/ai_session captures/raw/ai_session3`
Expected: a table where **pos3 `da_mis` ≈ 0.23–0.26** and the script prints
`FAIL` + `sys.exit(1)` (pos0/pos1/pos2 mislock < 0.03). Save the printed table into
the Task 2 STATUS note. (Runtime ~8–12 min; it decodes every meld frame.)

- [ ] **Step 6: Commit**

```bash
git add scripts/annotate/meld_snap_qa.py tests/test_meld_snap_qa.py
git commit -m "test(annotate): add meld-snap mislock QA guard (baseline: pos3 ~25% mislock)"
```

---

## Task 2: Recalibrate the MELD_STRIP2 corners

**Files:**
- Modify: `majsoul_eye/annotate/pipeline.py:458-463` (`MELD_STRIP2`)

**Interfaces:**
- Consumes: `dominant()` / the QA guard from Task 1 (verification).
- Produces: corrected `MELD_STRIP2` corners consumed by
  `generate_meld_boxes_v2` / `snap_meld_strip` (unchanged signatures).

- [ ] **Step 1: Re-run the calibrator to derive suggested corners**

```bash
PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe scripts/annotate/calibrate_annotation_model.py \
  --out scratchpad/calib_measure.json --per-game 40
PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe scripts/annotate/calibrate_annotation_model.py \
  --refit scratchpad/calib_measure.json
```

Read the `suggested corner: (x, y)` line under each `posN`. Expected (±1px):
`pos3 ≈ (624.5, 1751.6)`, `pos0 ≈ (2388.2, 1843.5)`, `pos1 ≈ (2454, 153)`,
`pos2 ≈ (685, 135)` (pos1/pos2 essentially unchanged). If pos3's suggestion is
NOT ~46px along from the current corner, STOP — the calibrator's plain median may
be contaminated; instead cross-check with the QA guard's `da_off` column
(dominant-cluster center, mislock-robust) and use that offset.

- [ ] **Step 2: Apply the corrected corners in `pipeline.py`**

In `majsoul_eye/annotate/pipeline.py`, replace the `MELD_STRIP2` pos0 and pos3
lines (`majsoul_eye/annotate/pipeline.py:458-463`). Current:

```python
    0: {"corner": (2388.2, 1889.5), "along": (-1.0, 0.0), "cross": (0.0, -1.0), "w": 70.2, "d": 92.1, "gap": 0.0},
    ...
    3: {"corner": (625.0, 1797.6),  "along": (0.0, -1.0), "cross": (1.0, 0.0),  "w": 71.0, "d": 93.0, "gap": 0.0},
```

New (substitute the calibrator's actual suggested corners; values below are the
expected result):

```python
    0: {"corner": (2388.2, 1843.5), "along": (-1.0, 0.0), "cross": (0.0, -1.0), "w": 70.2, "d": 92.1, "gap": 0.0},
    ...
    3: {"corner": (624.5, 1751.6),  "along": (0.0, -1.0), "cross": (1.0, 0.0),  "w": 71.0, "d": 93.0, "gap": 0.0},
```

Update the `MELD_STRIP2` block comment (`majsoul_eye/annotate/pipeline.py:454-457`)
to note the recalibration: append
`# Recalibrated 2026-07-08 (STATUS §1.50): pos3 corner was ~+46px (half-tile) off along and pos0 ~+46px off cross — the stale offset parked snap_meld_strip at the aliasing midpoint (26% one-tile flips). Re-verify with scripts/annotate/meld_snap_qa.py after any warp/mask change.`

- [ ] **Step 3: Verify the QA guard now passes**

Run: `PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe scripts/annotate/meld_snap_qa.py --sources captures/raw/ai_session captures/raw/ai_session3`
Expected: every seat `da_mis` and `dc_mis` < 0.03, all `*_off` ≈ 0, final line `OK`
(exit 0). In particular **pos3 `da_mis` drops from ~0.25 to ~0.01**.

- [ ] **Step 4: Visual spot-check on the known-bad frames**

Create `scratchpad/verify_meld_overlay.py`:

```python
"""Overlay emitted meld boxes for (game, seq) frames to confirm boxes sit on tiles."""
import os
import sys

import cv2
import numpy as np

from majsoul_eye.annotate import annotate_frame, build_homographies
from majsoul_eye.capture.gtframes import build_seq_state, load_frames

SEAT_COLOR = {0: (220, 80, 220), 1: (70, 180, 70), 2: (70, 70, 255), 3: (80, 170, 255)}
cap, out_dir = sys.argv[1], sys.argv[-1]
seqs = [int(x) for x in sys.argv[2:-1]]
os.makedirs(out_dir, exist_ok=True)
hom = build_homographies(1920, 1080)
ss = build_seq_state(cap)
fr = load_frames(os.path.dirname(cap))
gname = os.path.basename(os.path.dirname(cap))
for seq in seqs:
    img = cv2.imread(fr[seq])
    if img.shape[1] != 1920:
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
    rec = annotate_frame(img, ss[seq], hom)
    vis = img.copy()
    for pos_s, boxes in rec["meld_boxes"].items():
        for b in boxes:
            poly = np.int32(b["poly_original"])
            cv2.polylines(vis, [poly], True, SEAT_COLOR[int(pos_s)], 2)
    p = os.path.join(out_dir, f"{gname}__{seq:06d}.png")
    cv2.imwrite(p, vis)
    print("wrote", p)
```

Run for the two documented failure frames:

```bash
PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe scratchpad/verify_meld_overlay.py \
  captures/raw/ai_session/run_8/game6/game6.jsonl 1714 1715 scratchpad/verify_ov
PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe scratchpad/verify_meld_overlay.py \
  captures/raw/ai_session/run_8/game4/game4.jsonl 219 220 scratchpad/verify_ov
```

Open the 4 PNGs. Expected: pos3 (kamicha, orange) boxes on the tiles in BOTH
game6 seq1714 AND seq1715 (previously 1715 was a full tile off); pos2 (toimen, red)
boxes on the tiles in game4 seq219 and seq220. If any box is still a tile off, STOP
— the corner offset is wrong; return to Step 1.

- [ ] **Step 5: Run the full existing test suite (no regressions)**

Run: `for t in tests/test_*.py; do PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe "$t" || break; done`
Expected: every test prints its pass line; none error. (Meld geometry is GT-driven;
no committed test asserts exact corner px, so all stay green.)

- [ ] **Step 6: Commit**

```bash
git add majsoul_eye/annotate/pipeline.py
git commit -m "fix(annotate): recalibrate meld corners — pos3 +46 along / pos0 +46 cross

The pos3 MELD_STRIP2 corner was ~half a tile (46px) off along, parking
snap_meld_strip at the aliasing midpoint -> 26% one-tile lock-flips ('上家副露严重失位').
pos0 was ~46px off cross (benign, snap corrected it). Recalibrated via
calibrate_annotation_model.py; meld_snap_qa mislock pos3 ~25% -> ~1%. STATUS §1.50."
```

---

## Task 3: Documentation + stale-data note

**Files:**
- Modify: `docs/STATUS.md`
- Modify (conditional): `docs/PIPELINE.md`

- [ ] **Step 1: Add STATUS §1.50**

Prepend a new `### 1.50 副露角点重标定：远座 snap 失锁根因＝半张牌系统偏移（2026-07-08）`
entry above `### 1.49` in `docs/STATUS.md`, covering: user report (backs_review 里
上家副露严重失位 / 对家相邻帧差别); root cause (pos3 corner ~+46px≈半张 off along →
snap 顶在 aliasing 中点 → 26% 整条锁错一张; pos0 +46 cross 良性); evidence
(pre-shift +46 → mislock 25.9%→0.9%; the meld_snap_qa baseline vs post table from
Task 1 Step 5 and Task 2 Step 3); fix (re-run calibrator, apply corners,
committed `meld_snap_qa.py` guard); data impact (all meld boxes shift → datasets
stale, rebuild pending — Task 4); note that pos2 相邻帧 flicker is a per-round snap
issue NOT fixed here (Phase 2 consensus, separate plan).

- [ ] **Step 2: Update PIPELINE.md if it references the calibration**

Run: `PYTHONPATH=. grep -n "MELD_STRIP2\|calibrate_annotation\|meld.*calib" docs/PIPELINE.md`
If any hit, add a one-line note next to it that the meld corners were recalibrated
2026-07-08 (STATUS §1.50) and that `scripts/annotate/meld_snap_qa.py` is the guard.
If no hit, skip (no pipeline step/default changed — only a constant).

- [ ] **Step 3: Commit**

```bash
git add docs/STATUS.md docs/PIPELINE.md
git commit -m "docs: STATUS §1.50 meld corner recalibration + meld_snap_qa guard"
```

---

## Task 4: Rebuild affected datasets (MANUAL — user runs deliberately)

**Not auto-executed.** Changing `MELD_STRIP2` re-stales every meld box. This task
documents the rebuild so the user runs it when ready (heavy; datasets + retrain).

- [ ] **Step 1: Rebuild the review set and eyeball it in FiftyOne**

```bash
PYTHONPATH=. python scripts/inspect/build_backs_review.py            # -> datasets/backs_review
PYTHONPATH=. python scripts/inspect/fiftyone_view.py --data datasets/backs_review/obb/data.yaml --name backs_review --port 5253
```

Confirm 上家 (pos3) melds now sit on the tiles across rounds. Note any residual
pos2 相邻帧 flicker — that is the input signal for the Phase 2 (consensus) plan.

- [ ] **Step 2: Rebuild the versioned training datasets (when satisfied)**

```bash
PYTHONPATH=. python scripts/data/build_datasets.py v1 --force
# (repeat for any other live version; then retrain classifier/detector per PIPELINE.md)
```

---

## Self-Review

**Spec coverage** (against
`docs/superpowers/specs/2026-07-08-meld-snap-recalibration-consensus-design.md`):
- §3 Phase 1 (recalibrate corners + QA guard) → Tasks 1–3. ✓
- §3 Phase 1 "mislock < 3% guard, periodic check" → `meld_snap_qa.py` committed with
  `--max-mislock 0.03` default + STATUS note to re-run after warp/mask changes. ✓
- §6 pipeline impact (stale datasets, STATUS/PIPELINE, recognize untouched) → Task 3
  + Task 4. ✓
- §3 Phase 2 (consensus) → intentionally deferred to a separate plan, per spec §7
  decision 2 (Phase 1 now, Phase 2 gated on post-Phase-1 review). Task 4 Step 1
  gathers the residual-flicker signal that plan needs. ✓ (documented deferral, not a
  gap.)
- §2 "already-correct frames don't move" → verified implicitly by Task 2 Step 5
  (suite green) + Step 4 (visual on seq1714 which was already correct). ✓

**Placeholder scan:** none — all code shown in full; corner target values given with
a calibrator-confirmation step; STATUS content enumerated. ✓

**Type consistency:** `dominant()` signature identical in the script, the test, and
Task 2's cross-check reference; `MELD_STRIP2` keys unchanged (only `corner` tuples
edited). ✓
