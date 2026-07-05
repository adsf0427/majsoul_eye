# Skin-agnostic tile-back reliability gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the annotator's tile-back reliability check skin-agnostic so skin-swapped dora/ankan backs are labeled instead of dropped (and stop poisoning the detector with background negatives).

**Architecture:** Decouple the orange back-mask into two masks: a new `tile_live_mask` (skin-agnostic "a tile is rendered here", used only for the reliability fill of GT-known back slots/cells), and a saturation-based `tile_back_mask` (face/back discrimination for `snap_meld_strip`). GT still supplies *what* (which tile) and *where* (fixed slots / strip geometry); the pixel check only judges liveness.

**Tech Stack:** Python 3.12, OpenCV (cv2 4.11), NumPy 1.26. Pure-vision annotator under `majsoul_eye/annotate/`.

**Spec:** `docs/superpowers/specs/2026-07-05-skin-agnostic-back-gate-design.md`

## Global Constraints

- Run everything from the repo root with `PYTHONPATH=.`.
- Env python on this box: `/hszhao-f1/h3011050/anaconda3/envs/majsoul_eye/bin/python` (the docs write plain `python`; substitute this path — default-PATH python has no numpy). Call it `PY` below.
- Tests are plain scripts under `tests/` (also pytest-compatible); each file's `__main__` runs every `test_*` global. Run one with `PYTHONPATH=. $PY tests/test_X.py`.
- Threshold stays `FILL_OK = 0.25` (already defined in `frame.py`). Liveness signal is `(S>60) | (V>110)` coverage — data-locked (skinned backs ≥0.377, unrendered negatives 0.000).
- 38-class order is frozen; do not touch `tiles.py`.
- No AI co-author trailer in commit messages.
- Role A (Tasks 1–2, the fill gates) is unconditional. Role B (Task 3, snap mask) ships only if the default-snap regression check is clean, else falls back to leaving `tile_back_mask` orange.

---

### Task 1: Add `tile_live_mask` (skin-agnostic liveness mask)

**Files:**
- Modify: `majsoul_eye/annotate/pipeline.py` (add function right after `tile_back_mask`, ~line 609)
- Test: `tests/test_annotate_pipeline.py` (add one test + extend the import)

**Interfaces:**
- Produces: `tile_live_mask(fullwarp_bgr: np.ndarray) -> np.ndarray` — uint8 mask, 1 where a tile (any skin) is rendered: `(S>60) | (V>110)`. Applied to any BGR image/patch. Consumed by `frame.py` in Task 2.

- [ ] **Step 1: Write the failing test**

Add to the import block at the top of `tests/test_annotate_pipeline.py`:
```python
from majsoul_eye.annotate.pipeline import tile_live_mask
```
Add this test function (values verified against cv2 BGR→HSV):
```python
def test_tile_live_mask_liveness():
    def patch(bgr):
        return np.full((8, 8, 3), bgr, np.uint8)
    # skinned backs (non-orange) must read live
    assert tile_live_mask(patch((200, 40, 40))).mean() >= 0.25    # bright blue skin
    assert tile_live_mask(patch((150, 150, 150))).mean() >= 0.25  # desaturated grey skin (via V)
    assert tile_live_mask(patch((90, 0, 0))).mean() >= 0.25       # dark saturated skin (via S)
    # default orange back still reads live (no regression)
    assert tile_live_mask(patch((40, 120, 220))).mean() >= 0.25
    # unrendered / empty patches must read NOT live
    assert tile_live_mask(patch((0, 0, 0))).mean() == 0.0         # black (GT leads render)
    assert tile_live_mask(patch((80, 80, 80))).mean() == 0.0      # dark uniform
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `PYTHONPATH=. $PY tests/test_annotate_pipeline.py`
Expected: FAIL — `ImportError: cannot import name 'tile_live_mask'`.

- [ ] **Step 3: Implement `tile_live_mask`**

In `majsoul_eye/annotate/pipeline.py`, add immediately after `tile_back_mask` (~line 609):
```python
def tile_live_mask(fullwarp_bgr: np.ndarray) -> np.ndarray:
    """Skin-agnostic 'a tile is rendered here' mask: colored OR bright pixels.

    Used ONLY to judge liveness of a slot/cell GT already labels 'back' (drop the
    rare frames where GT leads the client render and the slot is still empty/black).
    Not for face/back discrimination — it lights up faces too (that is tile_back_mask's
    job). Colored-or-bright hedges both desaturated (grey) and dark skin backs.
    """
    hsv = cv2.cvtColor(fullwarp_bgr, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 1] > 60) | (hsv[..., 2] > 110)).astype(np.uint8)
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `PYTHONPATH=. $PY tests/test_annotate_pipeline.py`
Expected: PASS — prints `test_annotate_pipeline OK`.

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/annotate/pipeline.py tests/test_annotate_pipeline.py
git commit -m "feat(annotate): add skin-agnostic tile_live_mask for back liveness"
```

---

### Task 2: Wire the back fill gates (dora + meld) to `tile_live_mask`

This is the full Role-A labeling fix. After this task, skinned dora/ankan backs are marked reliable and will be emitted by `build_dataset`; `tile_back_mask`/snap are still on the orange mask (safe fallback state).

**Files:**
- Modify: `majsoul_eye/annotate/frame.py` (meld integral ~lines 47–48 and ~80; dora gate ~lines 119–135)
- Test: `tests/test_annotate_frame.py` (add one test)

**Interfaces:**
- Consumes: `P.tile_live_mask` (Task 1).
- Produces: no new signatures; `annotate_frame` behavior change only (back slots/cells reliable when rendered in any skin).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_annotate_frame.py` (imports `locate_fullscreen`, `dora_slot`, `MAX_DORA`):
```python
from majsoul_eye.normalize import locate_fullscreen
from majsoul_eye.coords import dora_slot, MAX_DORA


def _dora_state():
    s = BoardState(hero_seat=0, bakaze="E", kyoku=1, honba=0, last_actor=1)
    s.dora_markers = ["E"]          # 1 revealed -> slots 1..4 are face-down backs
    return s


def test_dora_back_reliable_on_skinned_back():
    # Paint the 4 face-down dora slots a NON-orange (blue) skin colour.
    img = np.zeros((1080, 1920, 3), np.uint8)
    region = locate_fullscreen(img)
    for i in range(1, MAX_DORA):
        x1, y1, x2, y2 = region.norm_to_px(dora_slot(i))
        img[y1:y2, x1:x2] = (200, 40, 40)     # BGR bright blue
    rec = annotate_frame(img, _dora_state(), HOM)
    backs = [d for d in rec["dora_boxes"] if d.get("back")]
    assert len(backs) == 4
    assert all(d.get("reliable", True) for d in backs)   # skin back is rendered -> reliable

    # A black frame (nothing rendered) must still drop the back slots.
    black = annotate_frame(np.zeros((1080, 1920, 3), np.uint8), _dora_state(), HOM)
    black_backs = [d for d in black["dora_boxes"] if d.get("back")]
    assert black_backs and all(not d.get("reliable", True) for d in black_backs)
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `PYTHONPATH=. $PY tests/test_annotate_frame.py`
Expected: FAIL on the `all(... reliable ...)` assertion — the blue backs read `fill=0.0` under the current orange gate, so `reliable=False`. (Verified: painting the slots blue yields `dora[i:back]:low_fill=0.00`.)

- [ ] **Step 3a: Fix the dora gate**

In `majsoul_eye/annotate/frame.py`, the dora back branch (~lines 133–135) currently reads:
```python
                if is_back:                          # orange tile-back
                    f = float(((hsv[..., 0] >= 8) & (hsv[..., 0] <= 28) &
                               (hsv[..., 1] > 80) & (hsv[..., 2] > 110)).mean())
                else:                                # white tile-face
```
Replace the `is_back` branch with:
```python
                if is_back:                          # skin-agnostic: any rendered tile back
                    f = float(P.tile_live_mask(img[y1:y2, x1:x2]).mean())
                else:                                # white tile-face
```
Also update the docstring line (~121) `# for revealed / orange-back coverage for backs;` to `# for revealed / skin-agnostic content coverage for backs;`.

- [ ] **Step 3b: Fix the meld back fill**

In `annotate_frame` near the top (~lines 47–48):
```python
    mb = P.tile_back_mask(full)
    ii_w = cv2.integral(mw)
    ii_b = cv2.integral(mb)
```
Change to (keep `mb` for the snap call; swap the back-fill integral to the live mask):
```python
    mb = P.tile_back_mask(full)                 # face/back discrimination for snap
    ii_w = cv2.integral(mw)
    ii_l = cv2.integral(P.tile_live_mask(full)) # skin-agnostic liveness for back-cell fill
```
Then in the meld loop (~line 80) change:
```python
                ii = ii_b if b["tile"] == "back" else ii_w
```
to:
```python
                ii = ii_l if b["tile"] == "back" else ii_w
```
(`ii_b` now has no other users — its only reference was this line. Removing its assignment above is correct.)

- [ ] **Step 4: Run the tests, verify they pass**

Run: `PYTHONPATH=. $PY tests/test_annotate_frame.py`
Expected: PASS — `test_annotate_frame OK`.
Run: `PYTHONPATH=. $PY tests/test_annotate_pipeline.py`
Expected: PASS (Task 1 test still green).

- [ ] **Step 5: Real-data verification (dora fix + meld no-regression)**

Write `/tmp/verify_backgate.py`:
```python
import cv2, numpy as np
from majsoul_eye.capture.gtframes import build_seq_state, load_frames
from majsoul_eye.annotate import build_homographies, annotate_frame

def counts(cap, gdir, limit=120):
    ss = build_seq_state(cap); fr = load_frames(gdir, statuses=("ok", "timeout"))
    hom = build_homographies(1920, 1080)
    dora_ok = dora_tot = meld_ok = meld_tot = 0
    for seq, fp in list(fr.items())[:limit]:
        st = ss.get(seq)
        if st is None: continue
        img = cv2.imread(fp)
        if img is None: continue
        if (img.shape[1], img.shape[0]) != (1920, 1080):
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
        rec = annotate_frame(img, st, hom)
        for d in rec["dora_boxes"]:
            if d.get("back"):
                dora_tot += 1; dora_ok += d.get("reliable", True)
        for boxes in rec["meld_boxes"].values():
            for b in boxes:
                if b["tile"] == "back":
                    meld_tot += 1; meld_ok += b.get("reliable", True)
    print(f"{gdir}: dora_back ok {dora_ok}/{dora_tot}  meld_back ok {meld_ok}/{meld_tot}")

R = "captures/raw"
counts(f"{R}/ai_session2/run_21/game1/game1.jsonl", f"{R}/ai_session2/run_21/game1")   # skinned: dora ok should jump ~0 -> ~all
counts(f"{R}/ai_session/run_8/game1/game1.jsonl", f"{R}/ai_session/run_8/game1")        # default: meld ok unchanged (~all), dora ok stays ~all
```
Run: `PYTHONPATH=. $PY /tmp/verify_backgate.py`
Expected: skinned `run_21` dora_back ok goes from ~0 to nearly all of its total; default `run_8` dora_back and meld_back ok stay near-total (no regression). Delete `/tmp/verify_backgate.py` after.

- [ ] **Step 6: Commit**

```bash
git add majsoul_eye/annotate/frame.py tests/test_annotate_frame.py
git commit -m "fix(annotate): skin-agnostic back liveness for dora + meld fill gates"
```

---

### Task 3: Skin-agnostic snap discrimination (Role B) — verify or fall back

Make `tile_back_mask` skin-agnostic so `snap_meld_strip` gets back-cell evidence on skinned ankan, but only keep it if it does not regress default snapping.

**Files:**
- Modify: `majsoul_eye/annotate/pipeline.py` (`tile_back_mask`, ~line 605–609)

**Interfaces:**
- `tile_back_mask(fullwarp_bgr) -> np.ndarray` signature unchanged; mask definition changes from orange-hue to saturation.

- [ ] **Step 1: Snapshot the default-snap baseline**

Write `/tmp/snap_baseline.py` (records per-back-cell snap offsets + reliability on a default game with ankan):
```python
import cv2, json
from majsoul_eye.capture.gtframes import build_seq_state, load_frames
from majsoul_eye.annotate import build_homographies, annotate_frame

cap = "captures/raw/ai_session/run_8/game1/game1.jsonl"
gdir = "captures/raw/ai_session/run_8/game1"
ss = build_seq_state(cap); fr = load_frames(gdir, statuses=("ok", "timeout"))
hom = build_homographies(1920, 1080)
rows = {}
for seq, fp in list(fr.items())[:200]:
    st = ss.get(seq)
    if st is None: continue
    img = cv2.imread(fp)
    if img is None: continue
    if (img.shape[1], img.shape[0]) != (1920, 1080):
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
    rec = annotate_frame(img, st, hom)
    for pos, boxes in rec["meld_boxes"].items():
        for j, b in enumerate(boxes):
            rows[f"{seq}:{pos}:{j}"] = [b["tile"], b.get("snap"), b.get("reliable", True)]
json.dump(rows, open("/tmp/snap_ref.json", "w"))
print("baseline cells:", len(rows),
      "back-ok:", sum(1 for v in rows.values() if v[0] == "back" and v[2]))
```
Run: `PYTHONPATH=. $PY /tmp/snap_baseline.py` and note the printed `back-ok` count.

- [ ] **Step 2: Change `tile_back_mask` to saturation-based**

In `majsoul_eye/annotate/pipeline.py` (~line 605), replace:
```python
def tile_back_mask(fullwarp_bgr: np.ndarray) -> np.ndarray:
    """Orange tile-back mask (ankan end tiles, walls)."""
    hsv = cv2.cvtColor(fullwarp_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h >= 8) & (h <= 32) & (s > 110) & (v > 110)).astype(np.uint8)
```
with:
```python
def tile_back_mask(fullwarp_bgr: np.ndarray) -> np.ndarray:
    """Colored tile-back mask for snap face/back discrimination (any skin).

    Saturation-based so it captures orange (default) AND skinned colored backs while
    staying disjoint from the white face mask (S<70) — snap needs to tell back cells
    from face cells. A near-white/grey skin back is (correctly) indistinguishable from
    a face here; its labeling reliability comes from tile_live_mask, not this mask.
    """
    hsv = cv2.cvtColor(fullwarp_bgr, cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] > 70).astype(np.uint8)
```

- [ ] **Step 3: Diff default snap against the baseline**

Write `/tmp/snap_diff.py`:
```python
import cv2, json
from majsoul_eye.capture.gtframes import build_seq_state, load_frames
from majsoul_eye.annotate import build_homographies, annotate_frame

ref = json.load(open("/tmp/snap_ref.json"))
cap = "captures/raw/ai_session/run_8/game1/game1.jsonl"
gdir = "captures/raw/ai_session/run_8/game1"
ss = build_seq_state(cap); fr = load_frames(gdir, statuses=("ok", "timeout"))
hom = build_homographies(1920, 1080)
moved = 0; max_d = 0.0; back_ok = 0; flipped = 0
for seq, fp in list(fr.items())[:200]:
    st = ss.get(seq)
    if st is None: continue
    img = cv2.imread(fp)
    if img is None: continue
    if (img.shape[1], img.shape[0]) != (1920, 1080):
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
    rec = annotate_frame(img, st, hom)
    for pos, boxes in rec["meld_boxes"].items():
        for j, b in enumerate(boxes):
            k = f"{seq}:{pos}:{j}"; old = ref.get(k)
            if b["tile"] == "back" and b.get("reliable", True): back_ok += 1
            if old and old[1] and b.get("snap"):
                d = max(abs(old[1][0] - b["snap"][0]), abs(old[1][1] - b["snap"][1]))
                if d > 0.05: moved += 1
                max_d = max(max_d, d)
            if old and old[2] != b.get("reliable", True): flipped += 1
print(f"cells with snap moved >0.05px: {moved}  max snap delta: {max_d:.2f}px  "
      f"reliability flips: {flipped}  back-ok now: {back_ok}")
```
Run: `PYTHONPATH=. $PY /tmp/snap_diff.py`

- [ ] **Step 4: Decide — keep or fall back**

**Keep (Role B ships)** if the diff is clean: `max snap delta` ≲ 3px and `reliability flips` is 0 (default meld backs still all reliable; snap barely moves). Proceed to Step 5.

**Fall back** if snap moved materially (e.g. `max snap delta` > ~5px on multiple cells, or any default meld-back `reliability flips`): revert `tile_back_mask` to the original orange body (Step 2, reversed) and keep only Tasks 1–2. The labeling fix is unaffected (fill gate is `tile_live_mask`); only skinned-ankan alignment precision is left as-is. Record the fallback in the STATUS entry (Task 4).

- [ ] **Step 5: Run the full annotate test suite**

Run: `PYTHONPATH=. $PY tests/test_annotate_pipeline.py` and `PYTHONPATH=. $PY tests/test_annotate_frame.py`
Expected: both PASS (the CASES/geometry tests in `test_annotate_pipeline.py` are unaffected by the mask change). Delete `/tmp/snap_*.py` and `/tmp/snap_ref.json`.

- [ ] **Step 6: Commit**

If kept:
```bash
git add majsoul_eye/annotate/pipeline.py
git commit -m "feat(annotate): skin-agnostic tile_back_mask for meld snap discrimination"
```
If fallback: no code commit for this task (tile_back_mask unchanged); note it in Task 4.

---

### Task 4: Docs + stale-data note (pipeline discipline)

**Files:**
- Modify: `docs/PIPELINE.md`, `docs/STATUS.md`

- [ ] **Step 1: Update `docs/PIPELINE.md`**

Find the annotate-stage description of the tile-back / dora reliability check and update it to state the back reliability gate is skin-agnostic (`tile_live_mask` = colored-or-bright liveness), distinct from `tile_back_mask` (saturation, snap discrimination — note "reverted to orange" if Task 3 fell back). If no such line exists, add one sentence under the annotate stage.

- [ ] **Step 2: Add a `docs/STATUS.md` entry**

Add a dated entry (2026-07-05) summarizing: orange back-gate dropped skin-swapped dora/ankan backs (0 back labels on runs 21-23, turned into YOLO background negatives); fix decouples liveness (`tile_live_mask`) from snap discrimination (`tile_back_mask` saturation — or "kept orange, snap unchanged" if fallback). State that `datasets/precise_ai_run_21..23`, `obb_precise_ai_run_21..23`, and any aggregate are now **stale for the `back` class** and must be rebuilt (`bash scripts/data/regen_detector_dataset.sh` — annotate is the stale step — or `build_datasets.py <name> --force`) and retrained; rebuild/retrain are user-triggered and out of scope here.

- [ ] **Step 3: Commit**

```bash
git add docs/PIPELINE.md docs/STATUS.md
git commit -m "docs: skin-agnostic back gate — pipeline + status, note stale datasets"
```

---

## Self-Review

- **Spec coverage:** Role A liveness mask → Task 1; dora + meld fill gates → Task 2; Role B snap mask + verify/fallback → Task 3; tests → Tasks 1–2; pipeline discipline / stale-data note → Task 4. Rejected classifier alternative and measured impact live in the spec (no task needed). All spec sections covered.
- **Placeholder scan:** every code step shows exact code; every run step shows exact command + expected output; the one data-dependent decision (Task 3 keep/fallback) has explicit numeric thresholds. No TBD/TODO.
- **Type consistency:** `tile_live_mask(fullwarp_bgr)->uint8 mask` defined in Task 1, consumed identically in Task 2 (`P.tile_live_mask(img[...])`, `cv2.integral(P.tile_live_mask(full))`). `ii_l` introduced in Task 2 Step 3b and used in the same step. `tile_back_mask` signature unchanged in Task 3.
