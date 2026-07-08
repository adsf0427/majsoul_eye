# Meld-snap Phase 2: per-round snap consensus — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kill the *occasional* whole-strip meld mis-placements (adjacent-frame flicker AND no-feature fallbacks to the raw template — the 对家/自家 "偶发整体偏移/偏下") by replacing the per-frame image snap with a per-round robust consensus offset applied to every frame of a (game, kyoku, seat).

**Architecture:** The meld strip is physically fixed within a kyoku (STATUS: within-round σ≈0.9px), so the correct `(d_along, d_cross)` is ~constant per `(game, bakaze, kyoku, honba, screen-pos)`. A new `annotate/meldsnap.py` measures each frame's per-frame snap, robustly consensuses them per round (confidence-weighted dominant 2-D cluster, both axes), and returns a per-seq override. `annotate_frame` gains a `meld_snap_override` param that, when given, shifts the meld template by the round consensus (skipping its own per-frame snap) and marks a round with no confident consensus unreliable. The dataset/QA builders compute the overrides once per game and pass them in.

**Tech Stack:** Python 3, OpenCV, NumPy; conda `auto` env; `PYTHONPATH=.` from repo root. Tests are plain scripts under `tests/` (no pytest), run with `PYTHONPATH=. python tests/test_x.py`.

## Global Constraints

- Run in the conda `auto` env. In an un-activated shell (this harness) substitute `C:/Users/zsx/miniforge3/envs/auto/python.exe` for `python`. Always `PYTHONPATH=.` from `D:\code\phoenix\majsoul_eye`.
- Branch `fix/meld-snap-stability` (Phase 1 already merged into it). Spec: `docs/superpowers/specs/2026-07-08-meld-snap-recalibration-consensus-design.md` §3 Phase 2.
- **Phase 2 is only safe AFTER Phase 1.** Phase 1 removed the pos3 half-tile offset that created large *consistent* mislock blocks; consensus assumes the true offset dominates. Do NOT run Phase 2 on un-recalibrated corners.
- **Consensus covers BOTH axes** (`d_along` AND `d_cross`). The user's occasional shifts are on both — the down-shift is cross, the sideways flip is along.
- **38-class / seat conventions frozen.** Do NOT touch `tiles.py`, seat mapping, `DISCARD_GRID`, `MELD_STRIP2` corners, or `snap_meld_strip`'s internals. Phase 2 wraps the snap, it does not change it.
- **`recognize/` stays Akagi-free and untouched.**
- **Pipeline discipline (CLAUDE.md):** the consensus changes every meld box in every dataset → rebuild (`build_datasets.py <v> --force`) + `docs/PIPELINE.md` (builds become two-pass) + `docs/STATUS.md` §1.51. Rebuild + retrain is a deliberate manual step (Task 5).
- **Cost:** the consensus adds one measure pass (decode+warp) per frame before the annotate pass — ~+60% build time. Acceptable for a deliberate rebuild; a fused single-warp optimization is noted as future, not in scope.
- Consensus tuning constants (verbatim): `CLUSTER_TOL = 12.0`, `MIN_ROUND_FRAMES = 3`, `MIN_ROUND_CONF = 0.55`, `MIN_FEATURES = 2`.

---

## File Structure

- Create: `majsoul_eye/annotate/meldsnap.py` — `round_meld_consensus` (pure), `measure_meld_snaps` (one frame → per-pos snap), `game_meld_overrides` (one game → per-seq override map).
- Create: `tests/test_meldsnap.py` — unit tests for `round_meld_consensus` + a smoke test for `game_meld_overrides` on a real game.
- Modify: `majsoul_eye/annotate/frame.py:40` (signature) and `:77-93` (meld block) — add `meld_snap_override`.
- Modify: `scripts/train/build_dataset.py` (self-contained loop, ~`:258-277`) — compute + pass overrides.
- Modify: `scripts/annotate/annotate_ai_session.py` — compute + pass overrides (the `--from-annotations` producer).
- Modify: `scripts/inspect/build_backs_review.py:129` — compute + pass overrides (QA parity).
- Create: `scripts/annotate/meld_consensus_qa.py` — Phase-2 guard: within-round adjacent-frame emitted-offset variance ≈ 0.
- Modify: `docs/STATUS.md` (§1.51), `docs/PIPELINE.md` (two-pass note).

---

## Task 1: `meldsnap.py` — measure + round consensus

**Files:**
- Create: `majsoul_eye/annotate/meldsnap.py`
- Test: `tests/test_meldsnap.py`

**Interfaces:**
- Produces: `round_meld_consensus(samples: list[tuple]) -> tuple | None` where each sample is `(d_along, d_cross, score, n_features)` for ONE `(kyoku, pos)`; returns `(d_along, d_cross, conf)` or `None`.
- Produces: `measure_meld_snaps(img, state, hom) -> dict[int, tuple]` = `{pos: (d_along, d_cross, score, n_features)}` for each screen pos 0..3 whose seat has melds.
- Produces: `game_meld_overrides(seq_states: dict[int, BoardState], seq_frames: dict[int, str], hom) -> dict[int, dict[int, tuple|None]]` = `{seq: {pos: (d_along, d_cross) | None}}`.

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_meldsnap.py`:

```python
"""Unit tests for round_meld_consensus (pure) + a smoke test for game_meld_overrides."""
import glob
import os

from majsoul_eye.annotate.meldsnap import round_meld_consensus


def test_majority_cluster_wins_over_single_outlier():
    # 17 frames agree at (0.8, 24), one flips to (51, -0.5) -> consensus is the majority
    samples = [(0.8, 24.0, 2.6, 5)] * 17 + [(51.0, -0.5, 2.9, 6)]
    r = round_meld_consensus(samples)
    assert r is not None
    assert abs(r[0] - 0.8) < 2.0 and abs(r[1] - 24.0) < 2.0, r
    assert r[2] > 0.9, r  # conf


def test_no_feature_frames_are_ignored():
    # n_features < MIN_FEATURES (2) are dropped; too few real samples -> None
    assert round_meld_consensus([(0.0, 0.0, 0.0, 1)] * 8) is None


def test_too_few_confident_frames_returns_none():
    assert round_meld_consensus([(1.0, 1.0, 2.0, 4)] * 2) is None


def test_ambiguous_split_returns_none():
    # 3 at (0,0) vs 3 at (46,0): no cluster reaches MIN_ROUND_CONF (0.55) -> None (safe)
    samples = [(0.0, 0.0, 2.0, 4)] * 3 + [(46.0, 0.0, 2.0, 4)] * 3
    assert round_meld_consensus(samples) is None


def test_cross_axis_consensus():
    # occasional cross flip: 10 at dc=24, 1 at dc=-0.5 -> consensus dc ~24
    samples = [(1.0, 24.0, 4.0, 5)] * 10 + [(1.0, -0.5, 3.0, 6)]
    r = round_meld_consensus(samples)
    assert r is not None and abs(r[1] - 24.0) < 2.0, r


def test_game_meld_overrides_smoke():
    from majsoul_eye.annotate import build_homographies
    from majsoul_eye.annotate.meldsnap import game_meld_overrides
    from majsoul_eye.capture.gtframes import build_seq_state, load_frames
    caps = glob.glob("captures/raw/ai_session/run_*/game*/game*.jsonl")
    assert caps, "no AI captures found — run from repo root"
    cap = sorted(caps)[0]
    ss = build_seq_state(cap)
    fr = load_frames(os.path.dirname(cap))
    ov = game_meld_overrides(ss, fr, build_homographies(1920, 1080))
    # every override value is a dict {pos: (da,dc) | None}; within one kyoku+pos all
    # non-None overrides are IDENTICAL (that is the whole point — one offset per round).
    from collections import defaultdict
    by_round = defaultdict(set)
    for seq, per_pos in ov.items():
        st = ss[seq]
        for pos, val in per_pos.items():
            if val is not None:
                by_round[(st.bakaze, st.kyoku, st.honba, pos)].add(val)
    for key, vals in by_round.items():
        assert len(vals) == 1, f"{key} has non-uniform override {vals}"
    print("game_meld_overrides smoke OK:", len(ov), "frames")


if __name__ == "__main__":
    for _n, _f in sorted(list(globals().items())):
        if _n.startswith("test_") and callable(_f):
            _f()
            print(_n, "OK")
    print("all meldsnap tests passed")
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe tests/test_meldsnap.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'majsoul_eye.annotate.meldsnap'`.

- [ ] **Step 3: Create the module**

Create `majsoul_eye/annotate/meldsnap.py`:

```python
"""Per-round meld-snap consensus (Phase 2, STATUS §1.51).

The meld strip is physically fixed within a kyoku, so the correct rigid-snap offset
(d_along, d_cross) is ~constant per (game, bakaze, kyoku, honba, screen-pos). Per-frame
snapping (snap_meld_strip) occasionally mislocks a single frame (flicker) or is gated
off on a low-feature frame (falls back to the raw template). This module measures every
frame's per-frame snap, takes a robust per-round consensus, and returns a per-seq
override so annotate_frame can place every frame of a round at the same, voted offset.

Only safe AFTER the Phase-1 corner recalibration (else large consistent mislock blocks
could out-vote the truth)."""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from majsoul_eye.annotate import pipeline as P
from majsoul_eye.annotate.seatgt import seat_gt
from majsoul_eye.state.replay import check_invariants, is_call_window, is_deal_window

CLUSTER_TOL = 12.0      # px; (d_along,d_cross) within this (both axes) = same cluster
MIN_ROUND_FRAMES = 3    # need >= this many confident frames to trust a round
MIN_ROUND_CONF = 0.55   # winning cluster must hold >= this fraction of total score
MIN_FEATURES = 2        # a frame's snap counts only if n_along + n_cross >= this


def _wmean(vals, weights):
    tw = float(sum(weights))
    return float(sum(v * w for v, w in zip(vals, weights)) / tw) if tw > 0 else 0.0


def round_meld_consensus(samples):
    """samples: list[(d_along, d_cross, score, n_features)] for ONE (kyoku, pos).
    Returns (d_along, d_cross, conf) — the score-weighted centre of the dominant
    (d_along, d_cross) cluster — or None if too few confident frames / no dominant
    cluster (caller then marks that round's meld boxes unreliable)."""
    conf = [(da, dc, sc) for da, dc, sc, n in samples if n >= MIN_FEATURES and sc > 0]
    if len(conf) < MIN_ROUND_FRAMES:
        return None
    best_w, best = -1.0, []
    for da0, dc0, _ in conf:
        members = [(da, dc, sc) for da, dc, sc in conf
                   if abs(da - da0) <= CLUSTER_TOL and abs(dc - dc0) <= CLUSTER_TOL]
        w = sum(sc for _, _, sc in members)
        if w > best_w:
            best_w, best = w, members
    total = sum(sc for _, _, sc in conf)
    if total <= 0 or best_w / total < MIN_ROUND_CONF:
        return None
    da_c = _wmean([m[0] for m in best], [m[2] for m in best])
    dc_c = _wmean([m[1] for m in best], [m[2] for m in best])
    return (round(da_c, 1), round(dc_c, 1), round(best_w / total, 3))


def measure_meld_snaps(img, state, hom):
    """{pos: (d_along, d_cross, score, n_features)} for each screen pos 0..3 whose seat
    has melds. Warp + meld masks + snap_meld_strip only (no river/hand/dora/hud)."""
    Hinv = hom["H_full_inv"]
    full = P.warp_to_full(img, hom["H_full"], hom["full_size"])
    hsv = cv2.cvtColor(full, cv2.COLOR_BGR2HSV)
    mw = P.tile_face_mask(hsv=hsv)
    mb = P.tile_back_mask(hsv=hsv)
    out = {}
    for pos in range(4):
        _, _, melds, _ = seat_gt(state, pos)
        if not melds:
            continue
        boxes = P.generate_meld_boxes_v2(pos, melds, Hinv)
        if not boxes:
            continue
        da, dc, diag = P.snap_meld_strip(mw, mb, boxes, pos)
        out[pos] = (da, dc, diag["score"], diag["n_along"] + diag["n_cross"])
    return out


def game_meld_overrides(seq_states, seq_frames, hom):
    """One game -> {seq: {pos: (d_along, d_cross) | None}}. Measures every settled frame,
    groups per (bakaze,kyoku,honba,pos), consensuses, and returns per-seq overrides.
    pos maps to None when that round has no confident consensus (annotate_frame then
    marks those meld boxes unreliable). Only pos with melds appear. Frames dropped by
    the build (deal/call window, invariant) are not measured (they never emit boxes)."""
    per_seq = {}
    by_round = defaultdict(list)
    for seq in sorted(seq_states):
        if seq not in seq_frames:
            continue
        st = seq_states[seq]
        if is_deal_window(st) or is_call_window(st) or check_invariants(st):
            continue
        img = cv2.imread(seq_frames[seq])
        if img is None:
            continue
        if img.shape[1] != 1920:
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
        m = measure_meld_snaps(img, st, hom)
        if not m:
            continue
        per_seq[seq] = m
        kk = (st.bakaze, st.kyoku, st.honba)
        for pos, s in m.items():
            by_round[(kk, pos)].append(s)
    consensus = {k: round_meld_consensus(v) for k, v in by_round.items()}
    overrides = {}
    for seq, m in per_seq.items():
        st = seq_states[seq]
        kk = (st.bakaze, st.kyoku, st.honba)
        ov = {}
        for pos in m:
            c = consensus.get((kk, pos))
            ov[pos] = (c[0], c[1]) if c is not None else None
        overrides[seq] = ov
    return overrides
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe tests/test_meldsnap.py`
Expected: each `test_*` prints `OK`, then `all meldsnap tests passed`. (The smoke test decodes one game's frames — ~1-2 min.)

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/annotate/meldsnap.py tests/test_meldsnap.py
git commit -m "feat(annotate): per-round meld-snap consensus (measure + round_meld_consensus + game_meld_overrides)"
```

---

## Task 2: `annotate_frame(meld_snap_override=...)`

**Files:**
- Modify: `majsoul_eye/annotate/frame.py:40` (signature), `:77-93` (meld block)
- Test: `tests/test_annotate_frame.py` (add one test)

**Interfaces:**
- Consumes: nothing new (override values are `(d_along, d_cross)` tuples from Task 1's `game_meld_overrides`).
- Produces: `annotate_frame(img, state, hom, hand_suspect=False, backs=False, meld_snap_override=None)`. When `meld_snap_override` is a dict: for each pos with melds, use `override[pos]` as `(da,dc)` (skip per-frame snap); if `override.get(pos)` is `None`, place at the raw template and mark those meld boxes `reliable=False` + flag `pos{pos}:meld:low_round_conf`. When `None`: unchanged per-frame-snap behaviour.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_annotate_frame.py` (append this test + ensure it runs in the file's `__main__` runner):

```python
def test_meld_snap_override_shifts_and_flags():
    import glob
    import numpy as np
    from majsoul_eye.annotate import annotate_frame, build_homographies
    from majsoul_eye.capture.gtframes import build_seq_state, load_frames
    import cv2, os

    cap = sorted(glob.glob("captures/raw/ai_session/run_*/game*/game*.jsonl"))[0]
    ss = build_seq_state(cap); fr = load_frames(os.path.dirname(cap))
    hom = build_homographies(1920, 1080)
    # find a settled frame with an opponent meld
    seq = next(s for s in sorted(ss) if s in fr
               and any(ss[s].melds[seat] for seat in range(4)))
    img = cv2.imread(fr[seq])
    if img.shape[1] != 1920:
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
    # which pos has melds
    from majsoul_eye.annotate.seatgt import seat_gt
    pos = next(p for p in range(4) if seat_gt(ss[seq], p)[2])

    # override (0,0) => boxes exactly at the template (no snap); a given (da,dc) shifts them
    base = annotate_frame(img, ss[seq], hom, meld_snap_override={pos: (0.0, 0.0)})
    shifted = annotate_frame(img, ss[seq], hom, meld_snap_override={pos: (10.0, 0.0)})
    b0 = np.float32(base["meld_boxes"][str(pos)][0]["poly_fullwarp"])
    b1 = np.float32(shifted["meld_boxes"][str(pos)][0]["poly_fullwarp"])
    from majsoul_eye.annotate import pipeline as P
    along = np.array(P.MELD_STRIP2[pos]["along"])
    moved = (b1 - b0).mean(axis=0)
    assert abs(float(np.dot(moved, along)) - 10.0) < 0.5, moved  # moved +10 along
    assert base["meld_boxes"][str(pos)][0]["snap"] == (0.0, 0.0)

    # override None => template + reliable False + low_round_conf flag
    lc = annotate_frame(img, ss[seq], hom, meld_snap_override={pos: None})
    assert all(b.get("reliable") is False for b in lc["meld_boxes"][str(pos)])
    assert any(f == f"pos{pos}:meld:low_round_conf" for f in lc["flags"])
    print("meld_snap_override OK")
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe tests/test_annotate_frame.py`
Expected: FAIL — `annotate_frame() got an unexpected keyword argument 'meld_snap_override'`.

- [ ] **Step 3: Implement — signature + meld block**

In `majsoul_eye/annotate/frame.py`, change the signature (line 40):

```python
def annotate_frame(img: np.ndarray, state, hom: dict, hand_suspect: bool = False,
                   backs: bool = False, meld_snap_override: dict | None = None) -> dict:
```

Add to the docstring (after the `backs` sentence): `meld_snap_override (Phase 2, STATUS §1.51): {pos: (d_along, d_cross) | None} from annotate.meldsnap.game_meld_overrides — when given, place each seat's meld strip at the per-round consensus offset instead of the per-frame snap; pos->None (round had no confident consensus) places at the raw template and marks those boxes reliable=False.`

Replace the meld block (`majsoul_eye/annotate/frame.py:77-93`):

```python
        boxes = P.generate_meld_boxes_v2(pos, melds, Hinv)
        if boxes:
            low_conf = False
            if meld_snap_override is not None:
                ov = meld_snap_override.get(pos)
                if ov is None:
                    da, dc, applied, low_conf = 0.0, 0.0, False, True
                else:
                    da, dc, applied = float(ov[0]), float(ov[1]), True
            else:
                da, dc, diag = P.snap_meld_strip(mw, mb, boxes, pos)
                applied = diag["n_along"] + diag["n_cross"] >= 2
            if applied:
                da = float(np.clip(da, -SNAP_MAX_ALONG, SNAP_MAX_ALONG))
                dc = float(np.clip(dc, -SNAP_MAX_CROSS, SNAP_MAX_CROSS))
                boxes = P.shift_boxes(boxes, pos, da, dc, Hinv)
            for b in boxes:
                ii = ii_l if b["tile"] == "back" else ii_w
                f = _fill(ii, b["poly_fullwarp"])
                b["fill"] = round(f, 3)
                b["snap"] = (round(da, 1), round(dc, 1))
                if low_conf:
                    b["reliable"] = False
                    b["low_conf"] = True
                if f < FILL_OK:
                    b["reliable"] = False
                    b["low_conf"] = True
                    rec["flags"].append(f"pos{pos}:meld[{b['tile']}]:low_fill={f:.2f}")
            if low_conf:
                rec["flags"].append(f"pos{pos}:meld:low_round_conf")
        rec["meld_boxes"][str(pos)] = boxes
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe tests/test_annotate_frame.py`
Expected: all tests pass incl. `meld_snap_override OK`.

- [ ] **Step 5: Confirm the legacy path is byte-identical**

Run the full suite: `for t in tests/test_*.py; do PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe "$t" || break; done`
Expected: all green. (`meld_snap_override=None` must reproduce the old per-frame behaviour exactly — `test_annotate_frame`/`test_annotate_pipeline`/`test_consistency` assert record shape.)

- [ ] **Step 6: Commit**

```bash
git add majsoul_eye/annotate/frame.py tests/test_annotate_frame.py
git commit -m "feat(annotate): annotate_frame meld_snap_override — per-round consensus placement + low-conf flag"
```

---

## Task 3: Wire consensus into the builders

**Files:**
- Modify: `scripts/train/build_dataset.py` (self-contained branch + loop)
- Modify: `scripts/annotate/annotate_ai_session.py` (the `--from-annotations` producer)
- Modify: `scripts/inspect/build_backs_review.py:129`

**Interfaces:**
- Consumes: `game_meld_overrides(seq_states, seq_frames, hom)` and `annotate_frame(..., meld_snap_override=...)` from Tasks 1-2.
- Produces: no new API — the builders now emit consensus-placed meld boxes.

- [ ] **Step 1: Wire `build_dataset.py` (self-contained path)**

In `scripts/train/build_dataset.py`, add the import near the other `majsoul_eye.annotate` imports at the top:

```python
from majsoul_eye.annotate import meldsnap as _meldsnap
```

In the `else` (self-contained) branch where `seq_state`/`hom` are set (around `scripts/train/build_dataset.py:258-261`), after `frames` is loaded and `hom` built, compute overrides once for the game:

```python
        seq_state = build_seq_state(args.capture)
        hom = build_homographies(1920, 1080)
        seqs = sorted(seq_state)
        meld_overrides = _meldsnap.game_meld_overrides(seq_state, frames, hom)
```

(If `frames` is loaded later than this point in the current file, place the `meld_overrides = ...` line immediately after `frames` is assigned; it needs `seq_state`, `frames`, `hom`. In the `--from-annotations` branch set `meld_overrides = {}` — those recs already carry consensus from annotate_ai_session, Step 2.)

In the per-frame loop, change the annotate call (`scripts/train/build_dataset.py:323-324`):

```python
        rec = (recs[seq] if args.from_annotations
               else annotate_frame(frame, seq_state[seq], hom, backs=args.backs,
                                   meld_snap_override=meld_overrides.get(seq)))
```

- [ ] **Step 2: Wire `annotate_ai_session.py` (the `--from-annotations` producer)**

Open `scripts/annotate/annotate_ai_session.py`, find where it loops a game's frames calling `annotate_frame(...)`. Add `from majsoul_eye.annotate import meldsnap as _meldsnap` at the top. Before the per-frame loop for a game, compute `overrides = _meldsnap.game_meld_overrides(seq_state, frames, hom)` (using that script's own `seq_state`/`frames`/`hom` names), and pass `meld_snap_override=overrides.get(seq)` to its `annotate_frame(...)` call. This keeps recs consumed by `build_dataset --from-annotations` consensus-consistent with the self-contained path.

- [ ] **Step 3: Wire `build_backs_review.py`**

In `scripts/inspect/build_backs_review.py`, add `from majsoul_eye.annotate import meldsnap as _meldsnap` at the top. In `build()` (around `scripts/inspect/build_backs_review.py:112-129`), per `cap` after `fr = load_frames(...)` and `ss = build_seq_state(...)`, compute `overrides = _meldsnap.game_meld_overrides(ss, fr, hom)` once, then change the annotate call (`:129`):

```python
            rec = annotate_frame(img, st, hom, backs=True, meld_snap_override=overrides.get(seq))
```

- [ ] **Step 4: Integration check — within-round consistency end-to-end**

Create `scratchpad/phase2_check.py`:

```python
"""Verify consensus makes every frame of a (kyoku,pos) emit the SAME meld boxes."""
import glob, os
import numpy as np
from collections import defaultdict
from majsoul_eye.annotate import annotate_frame, build_homographies
from majsoul_eye.annotate import meldsnap as M
from majsoul_eye.capture.gtframes import build_seq_state, load_frames
import cv2

cap = sorted(glob.glob("captures/raw/ai_session/run_8/game4/game*.jsonl"))[0]
ss = build_seq_state(cap); fr = load_frames(os.path.dirname(cap)); hom = build_homographies(1920, 1080)
ov = M.game_meld_overrides(ss, fr, hom)
# re-annotate a stretch and confirm within-round emitted boxes are identical
boxes_by_round = defaultdict(list)
for seq in sorted(ss):
    if seq not in fr or seq not in ov:
        continue
    img = cv2.imread(fr[seq])
    if img is None:
        continue
    if img.shape[1] != 1920:
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
    rec = annotate_frame(img, ss[seq], hom, meld_snap_override=ov.get(seq))
    st = ss[seq]
    for pos_s, bs in rec["meld_boxes"].items():
        if bs and all(b.get("reliable", True) for b in bs):
            key = (st.bakaze, st.kyoku, st.honba, pos_s, tuple(b["tile"] for b in bs))
            boxes_by_round[key].append(np.float32([b["poly_fullwarp"] for b in bs]))
bad = 0
for key, arrs in boxes_by_round.items():
    if len(arrs) < 2:
        continue
    spread = max(float(np.abs(a - arrs[0]).max()) for a in arrs)
    if spread > 1.0:
        bad += 1
        print("NONUNIFORM", key, "spread", round(spread, 1))
print("rounds checked:", len(boxes_by_round), "nonuniform:", bad)
```

Run: `PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe scratchpad/phase2_check.py`
Expected: `nonuniform: 0` — every same-meld frame of a round now emits pixel-identical boxes (the flicker is gone). If any NONUNIFORM appears, the override isn't being applied uniformly — debug before committing.

- [ ] **Step 5: Run the full suite**

Run: `for t in tests/test_*.py; do PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe "$t" || break; done`
Expected: all green (`test_build_datasets`, `test_backs`, `test_build_gate` exercise the builders).

- [ ] **Step 6: Commit**

```bash
git add scripts/train/build_dataset.py scripts/annotate/annotate_ai_session.py scripts/inspect/build_backs_review.py
git commit -m "feat(build): apply per-round meld-snap consensus in build_dataset / annotate_ai_session / backs_review"
```

---

## Task 4: Phase-2 QA guard + docs

**Files:**
- Create: `scripts/annotate/meld_consensus_qa.py`
- Modify: `docs/STATUS.md`, `docs/PIPELINE.md`

**Interfaces:**
- Consumes: `game_meld_overrides` (Task 1).
- Produces: a CLI guard exiting nonzero if any within-round emitted offset is non-uniform.

- [ ] **Step 1: Create the guard**

Create `scripts/annotate/meld_consensus_qa.py`:

```python
"""Phase-2 guard: after consensus, every frame of a (kyoku,pos) must share ONE offset.

Runs game_meld_overrides per game and asserts each (bakaze,kyoku,honba,pos) round maps
to a single override value (or None). A non-uniform round means consensus is not being
applied per-round (regression). Exits nonzero on any violation.

  PYTHONPATH=. python scripts/annotate/meld_consensus_qa.py
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

from majsoul_eye.annotate import build_homographies
from majsoul_eye.annotate import meldsnap as M
from majsoul_eye.capture.gtframes import build_seq_state, load_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["captures/raw/ai_session"])
    args = ap.parse_args()
    hom = build_homographies(1920, 1080)
    caps = []
    for root in args.sources:
        caps += sorted(glob.glob(os.path.join(root, "run_*", "game*", "game*.jsonl")))
    bad = 0
    n_rounds = 0
    for cap in caps:
        try:
            ss = build_seq_state(cap)
            fr = load_frames(os.path.dirname(cap))
        except Exception as e:
            print(f"  skip {cap}: {e}")
            continue
        ov = M.game_meld_overrides(ss, fr, hom)
        by_round = defaultdict(set)
        for seq, per_pos in ov.items():
            st = ss[seq]
            for pos, val in per_pos.items():
                by_round[(st.bakaze, st.kyoku, st.honba, pos)].add(val)
        for key, vals in by_round.items():
            n_rounds += 1
            if len(vals) != 1:
                bad += 1
                print(f"NONUNIFORM {os.path.basename(os.path.dirname(cap))} {key}: {vals}")
    print(f"rounds checked={n_rounds} nonuniform={bad}")
    if bad:
        print("FAIL: consensus is not uniform per round.")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the guard**

Run: `PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe scripts/annotate/meld_consensus_qa.py`
Expected: `rounds checked=<N> nonuniform=0` then `OK` (exit 0). (~a few min.)

- [ ] **Step 3: STATUS §1.51**

Prepend `### 1.51 副露按局吸附共识（Phase 2）（2026-07-08）` above `### GPU` in `docs/STATUS.md`: user report (自家/对家 偶发整体偏移/偏下——逐帧吸附偶发失锁 + 无特征帧回退原始模板); design (strip 每局固定 → 按 `(bakaze,kyoku,honba,screen-pos)` 对逐帧 snap 取置信度加权主簇共识，两轴一起，统一套到该局每帧；`annotate/meldsnap.py`；`annotate_frame(meld_snap_override=)`；build 变两遍); low-conf 局 → `reliable=False` + `meld:low_round_conf`; 只在 Phase 1 之后安全; guard `meld_consensus_qa.py`（同局同座唯一偏移）; cost (+1 measure warp/帧); data impact (所有 meld 框变 → 重建 + 重训, Task 5); note Phase-1 `meld_snap_qa` 的 fill 盲区结论（cross 系统偏移量法不可靠，真信号是 crevice）保留。

- [ ] **Step 4: PIPELINE.md note**

Under the annotation/build description in `docs/PIPELINE.md`, add: 副露框放置现走**按局共识**（`annotate.meldsnap.game_meld_overrides` → `annotate_frame(meld_snap_override=)`），build 因此**两遍扫描**（先 measure 每帧 snap → 每局共识 → 再 annotate）；guard `scripts/annotate/meld_consensus_qa.py`（同局同座唯一偏移）。低置信局 meld 框标 `reliable=False`。

- [ ] **Step 5: Commit**

```bash
git add scripts/annotate/meld_consensus_qa.py docs/STATUS.md docs/PIPELINE.md
git commit -m "feat(annotate): meld_consensus_qa guard + STATUS §1.51 / PIPELINE two-pass note"
```

---

## Task 5: Rebuild + verify (MANUAL — user runs deliberately)

**Not auto-executed.** Consensus changes every meld box.

- [ ] **Step 1: Rebuild the furo review set and eyeball the flicker**

Rebuild the furo-only review set (the scratchpad builder from Phase-1 review, updated to pass `game_meld_overrides` — or via `build_backs_review.py` now that it is wired), then in FiftyOne confirm the 自家/对家 occasional shifts are gone across consecutive frames of a round.

- [ ] **Step 2: Rebuild training datasets + retrain (when satisfied)**

```bash
PYTHONPATH=. python scripts/data/build_datasets.py v1 --force
# then retrain classifier/detector per PIPELINE.md
```

---

## Self-Review

**Spec coverage** (spec §3 Phase 2):
- New `annotate/meldsnap.py` with `measure_meld_snaps` + `round_meld_consensus` + `game_meld_overrides` → Task 1. ✓
- `annotate_frame` measure/apply split via `meld_snap_override` → Task 2. ✓
- Two-pass build orchestration → Task 3 (build_dataset + annotate_ai_session + backs_review). ✓
- Low-confidence rounds → `reliable=False` + flag → Task 2 (None branch) + Task 1 (None consensus). ✓
- Consensus on both axes → Task 1 (2-D cluster) + `test_cross_axis_consensus`. ✓
- Safe-only-after-Phase-1 → Global Constraints + STATUS. ✓
- Pipeline impact (stale data, STATUS/PIPELINE, recognize untouched) → Task 4 + Task 5. ✓
- Cost note (+1 warp) → Global Constraints. ✓

**Placeholder scan:** none — all code shown; the one soft spot is Task 3 Step 2 (`annotate_ai_session.py` loop location), handled by naming the exact call to change and the exact helper to insert, since that file's loop var names are read at implementation time. Not a logic placeholder.

**Type consistency:** `round_meld_consensus` returns `(da,dc,conf)`/`None`; `game_meld_overrides` unwraps to `(da,dc)`/`None`; `annotate_frame` consumes `override[pos]` as `(da,dc)` tuple or `None`; `measure_meld_snaps` emits `(da,dc,score,n)` matching `round_meld_consensus`'s sample shape. Consistent across tasks. ✓
