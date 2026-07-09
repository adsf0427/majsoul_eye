# Backs + score-anim HUD label-gate removal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the annotate stage from deleting labels of objects that are plainly on screen: remove `sorting_suspect` Condition B and the `FILL_OK_BACKS` gate in `backs.py`, and convert the score-anim HUD gate from "blanket unreliable" to "text-only unreliable".

**Architecture:** Three independent deletions/narrowings in the annotate stage plus one consumer change in `build_dataset.py`. No new modules; `tile_live_mask`, the dora/meld gates, and `recognize/` are untouched. Spec: `docs/superpowers/specs/2026-07-10-backs-hud-label-gates-design.md` (read it first — it holds the measurements that justify every deletion).

**Tech Stack:** Python 3.12, numpy/cv2, plain-script tests under `tests/` (no pytest dependency; run directly).

## Global Constraints

- Python: `PY=/hszhao-f1/h3011050/anaconda3/envs/majsoul_eye/bin/python` (the CLAUDE.md `auto` env / Windows path is dead — Linux box). Always run from the repo root with `PYTHONPATH=.`.
- Run a test file as `PYTHONPATH=. $PY tests/test_backs.py` — they are plain scripts that print `... OK` on success and throw on failure.
- Full suite: `for t in tests/test_*.py; do PYTHONPATH=. $PY "$t" || break; done` — must end green in every task.
- Git: NO `Co-Authored-By` trailers of any kind (user global rule). Commit per task.
- Do not touch: `majsoul_eye/annotate/pipeline.py` (`tile_live_mask` / `tile_back_mask`), the dora/meld back gates in `frame.py` (lines ~104-115, ~152-174), `recognize/`, the 38-class order.
- `docs/PIPELINE.md` + `docs/STATUS.md` must be updated (pipeline-impact discipline) — Task 5.

---

### Task 1: Remove `sorting_suspect` Condition B

**Files:**
- Modify: `majsoul_eye/annotate/backs.py` (function `sorting_suspect`, lines ~185-218)
- Test: `tests/test_backs.py`

**Interfaces:**
- Produces: `sorting_suspect(img, pos, row_n, n_melds, H_full_inv) -> bool` — same signature, now fires ONLY on the bare-slot signature (Condition A). Task 5 measures the recovered frames.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_backs.py` just above the `if __name__ == "__main__":` block:

```python
def _scripted_patch_stats(vals):
    """Stub for backs._patch_stats: ignore pixels, return scripted (mean, std)
    tuples in call order — row slots first, then the drawn slot, then the
    empty-felt reference (sorting_suspect's exact call order)."""
    it = iter(vals)

    def stub(img, quad):
        return next(it)
    return stub


def test_sorting_suspect_condition_b_removed():
    # Condition-B signature: every ROW slot AND the drawn slot read like tiles,
    # the empty reference reads like felt. The old "drawn slot occupied" verdict
    # false-fired on 253/256 firings of a dark-skin game and 17.6% of eligible
    # frames dataset-wide, and build_dataset drops the WHOLE frame (spec
    # 2026-07-10). A fully tile-like row must NOT be called mid-sort.
    import majsoul_eye.annotate.backs as B_mod
    TILE, FELT = (100.0, 40.0), (30.0, 5.0)
    img = np.zeros((8, 8, 3), np.uint8)                  # stub ignores pixels
    orig = B_mod._patch_stats
    try:
        B_mod._patch_stats = _scripted_patch_stats([TILE] * 13 + [TILE, FELT])
        assert B_mod.sorting_suspect(img, 1, 13, 0, HINV) is False
    finally:
        B_mod._patch_stats = orig


def test_sorting_suspect_condition_a_survives():
    # Condition-A signature: one ROW slot reads like the empty-felt reference
    # -> the row really is mid-compaction; the gate must still fire.
    import majsoul_eye.annotate.backs as B_mod
    TILE, FELT = (100.0, 40.0), (30.0, 5.0)
    img = np.zeros((8, 8, 3), np.uint8)
    orig = B_mod._patch_stats
    try:
        B_mod._patch_stats = _scripted_patch_stats(
            [TILE] * 6 + [FELT] + [TILE] * 6 + [TILE, FELT])
        assert B_mod.sorting_suspect(img, 1, 13, 0, HINV) is True
    finally:
        B_mod._patch_stats = orig
```

Why monkeypatch: the unit under change is the decision logic; quad→pixel mapping is already covered by the template-geometry tests at the top of the file. Painting pixels under warped quads would test the wrong thing brittlely.

- [ ] **Step 2: Run to verify the B test fails and the A test passes**

Run: `PYTHONPATH=. $PY tests/test_backs.py`
Expected: `AssertionError` inside `test_sorting_suspect_condition_b_removed` (current code returns True on the B signature). If instead the A test fails, the stub call-order assumption is wrong — stop and re-read `sorting_suspect`.

- [ ] **Step 3: Implement** — in `majsoul_eye/annotate/backs.py`:

Replace the tail of `sorting_suspect` (the `drawn = ...` line stays deleted too — it was only consumed by Condition B):

```python
    k = _meld_k(pos, row_n, n_melds)
    slots = [_patch_stats(img, P.fullwarp_to_original(
        _stretch_quad(BACK_SLOT_QUADS[pos][i], pos, k), H_full_inv)) for i in range(row_n)]
    slots = [s for s in slots if s]
    empty = _patch_stats(img, P.fullwarp_to_original(_drawn_fw(pos, row_n, n_melds, 1.15), H_full_inv))
    if len(slots) < 3 or empty is None:
        return False
    tm = (float(np.median([s[0] for s in slots])), float(np.median([s[1] for s in slots])))

    def dist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    return any(dist(s, empty) < 0.7 * dist(s, tm) for s in slots)
```

Replace the docstring of `sorting_suspect` with:

```python
    """Post-tedashi 理牌 (row-compaction) gate for a GT-SETTLED row.

    After an opponent discards from hand the client re-compacts the row
    (~0.5-1s) while GT is already settled; the capture stability ROI
    (roi_diff.TABLE_ROI) does NOT cover the hand rows, so canonical frames can
    catch the compaction mid-flight — template boxes then misalign. Signature:
    some ROW SLOT reads as bare felt (the pulled tile's gap, open mid-row or
    at the anchor-far end after closing). Patches are classified by (gray
    mean, std) distance to the row's own tile median vs an empty-felt
    reference sampled 1.15 tiles past the drawn slot — relative comparisons
    survive skins. Fires on 0.4-3% of settled seat-frames (the real reflow
    rate).

    A second signature ("the 13-row DRAWN slot reads occupied", Condition B of
    STATUS §1.48) was REMOVED 2026-07-10: its patch straddles the last-tile
    edge, and on dark skins the row's tile median ≈ that edge mean, so it
    false-fired on ~100% of skinned settled rows (253/256 firings on
    ai_session4/run_5/game1) and 17.6% of eligible frames dataset-wide —
    each one a WHOLE-FRAME drop in build_dataset (the largest frame-drop
    cause in the pipeline; see spec 2026-07-10)."""
```

- [ ] **Step 4: Run the backs tests**

Run: `PYTHONPATH=. $PY tests/test_backs.py`
Expected: `all backs tests passed` (both new tests + all pre-existing ones).

- [ ] **Step 5: Run the full suite**

Run: `for t in tests/test_*.py; do PYTHONPATH=. $PY "$t" || break; done`
Expected: every file prints its OK line; no traceback.

- [ ] **Step 6: Commit**

```bash
git add majsoul_eye/annotate/backs.py tests/test_backs.py
git commit -m "fix(annotate): drop sorting_suspect Condition B — 17.6% whole-frame false-drop

STATUS 1.48 recorded this removal but it was never committed. The drawn-slot
patch straddles the last-tile edge; on dark skins the row tile median equals
that edge mean, so B fired on 253/256 firings of a skinned game and 17.6% of
eligible frames dataset-wide -- each a WHOLE-frame drop in build_dataset,
the largest frame-drop cause in the pipeline (~7.5k frames, +23% dataset).
Condition A (bare-slot, the real 0.4-3% reflow signal) survives."
```

---

### Task 2: Remove the `FILL_OK_BACKS` reliability gate

**Files:**
- Modify: `majsoul_eye/annotate/backs.py` (constant line ~70; `back_boxes` lines ~253-260 post-Task-1; module docstring lines ~37-39)
- Modify: `CLAUDE.md` (the `backs.py` bullet in the `annotate/` section)
- Test: `tests/test_backs.py`

**Interfaces:**
- Produces: `back_boxes(img, state, hom) -> (rec_dict, flags)` — same shape; boxes always carry `fill` (float, diagnostic) and never get `reliable=False` from fill. `FILL_OK_BACKS` no longer exists.

- [ ] **Step 1: Flip the stale assertion + write the failing tests**

In `tests/test_backs.py`, `test_back_boxes_labels_holding_row_plus_drawn`, replace the last two lines

```python
    # black frame -> every emitted box fails the live-fill check
    assert all(b.get("reliable") is False for b in rec["1"] + rec["2"] + rec["3"])
```

with

```python
    # black frame: fill is still RECORDED (diagnostic) but no longer gates
    # reliability — the gate had zero discriminative power (empty felt reads
    # 1.00 on every table, dark backs 0.24; spec 2026-07-10).
    assert all(b.get("reliable", True) for b in rec["1"] + rec["2"] + rec["3"])
    assert all("fill" in b for b in rec["1"] + rec["2"] + rec["3"])
```

Append two new tests above the `__main__` block:

```python
def test_dark_backs_stay_reliable():
    # A dark tile back (S<=60 AND V<=110 -> tile_live_mask reads 0, e.g. the
    # RML skin: S~59 V~55) must keep its label; fill collapses but reliability
    # must not. This was the 5%-kept-backs bug on skinned games.
    img = np.full((1080, 1920, 3), 55, np.uint8)         # V=55, S=0 everywhere
    st = _state()
    rec, flags = back_boxes(img, st, HOM)
    boxes = rec["1"] + rec["2"] + rec["3"]
    assert len(boxes) == 39
    assert all(b.get("reliable", True) for b in boxes)
    assert all(b["fill"] == 0.0 for b in boxes)          # diagnostic still recorded
    assert not any("low_fill" in f for f in flags)


def test_fill_ok_backs_removed():
    import majsoul_eye.annotate.backs as B_mod
    assert not hasattr(B_mod, "FILL_OK_BACKS")
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=. $PY tests/test_backs.py`
Expected: FAIL — first at the flipped assertion in `test_back_boxes_labels_holding_row_plus_drawn` (boxes are still `reliable=False` on a black frame today).

- [ ] **Step 3: Implement** — in `majsoul_eye/annotate/backs.py`:

Delete the constant:

```python
FILL_OK_BACKS = 0.25    # tile_live_mask coverage below this = not rendered / occluded
```

In `back_boxes`, replace

```python
        for b in boxes:
            p = np.float32(b["poly_original"])
            f = P._box_fill(ii, p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
            b["fill"] = round(float(f), 3)
            if f < FILL_OK_BACKS:
                b["reliable"] = False
                flags.append(f"pos{pos}:back[{b['slot']}]:low_fill={f:.2f}")
```

with

```python
        for b in boxes:
            p = np.float32(b["poly_original"])
            f = P._box_fill(ii, p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
            b["fill"] = round(float(f), 3)   # diagnostic only, never gates (see docstring)
```

Rewrite the `back_boxes` docstring paragraph about `fill` to:

```python
    """(rec_dict, flags) for all three opponent seats of one frame.

    rec_dict maps str(pos) -> box list (empty for skipped seats). ``fill``
    (tile_live_mask coverage of the original-px bbox) is recorded as a QA
    diagnostic only. It does NOT gate reliability: for these edge-on rows the
    skewed quad's bbox is mostly felt, and every mahjong felt is colored or
    bright, so empty felt reads 1.00 on every table while a dark skinned back
    reads 0.24 — the positive class scores BELOW the negative and no threshold
    separates them (measured 2026-07-10; the mask stays valid in the dora/meld
    role it was calibrated for). GT-leads-render protection comes from
    is_deal_window / is_call_window (whole-frame drops) and sorting_suspect
    (Condition A).
    """
```

In the module docstring, replace the tail of the holding-seat bullet

```
    that reasoning actually belonged to the POST-tedashi reflow, which the
    ``sorting_suspect`` pixel gate handles on settled frames). Only the
    slide-in draw animation can leave the drawn tile unrendered — caught per-box
    by the fill check, not a whole-frame drop.
```

with

```
    that reasoning actually belonged to the POST-tedashi reflow, which the
    ``sorting_suspect`` pixel gate handles on settled frames). The slide-in
    draw animation can in principle leave the drawn tile unrendered for a
    beat; the debounce-to-quiet capture sync makes that residual rare, and no
    per-box pixel gate covers it since the fill gate was removed (2026-07-10 —
    it had zero discriminative power on these edge-on rows; ``fill`` is now a
    QA diagnostic only).
```

- [ ] **Step 4: Update the CLAUDE.md backs bullet** — replace

```
  `backs.py` = EXPERIMENTAL opt-in (`annotate_frame(..., backs=True)` / `build_datasets.py --backs`,
  default OFF, not in v1/v2): opponent concealed-hand tile-back boxes from GT counts + a calibrated
  fullwarp row grid (手摸切 groundwork; holding seats skipped + flagged, builds drop those frames whole).
```

with

```
  `backs.py` = EXPERIMENTAL opt-in (`annotate_frame(..., backs=True)` / `build_datasets.py --backs`,
  default OFF, not in v1/v2): opponent concealed-hand tile-back boxes from GT counts + manually
  clicked fullwarp slot templates (手摸切 groundwork). Holding seats ARE labeled (static n-1 row +
  drawn slot); the only pixel gate left is `sorting_suspect` Condition A (bare-slot reflow, 0.4-3%),
  which build_dataset still turns into a whole-frame drop. The per-box fill gate and the
  drawn-slot Condition B were removed 2026-07-10 (skin-dependent false drops; `fill` is QA-only).
```

- [ ] **Step 5: Run the backs tests, then the full suite**

Run: `PYTHONPATH=. $PY tests/test_backs.py`
Expected: `all backs tests passed`.
Run: `for t in tests/test_*.py; do PYTHONPATH=. $PY "$t" || break; done`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add majsoul_eye/annotate/backs.py tests/test_backs.py CLAUDE.md
git commit -m "fix(annotate): backs fill gate removed — zero discriminative power, inverted on dark skins

tile_live_mask coverage of an edge-on row box's bbox reads 1.00 on empty felt
(every felt is colored-or-bright) and 0.24 on a dark RML back (S~59 V~55 misses
both thresholds), so the gate passed everything on bright tables (never
functioned) and rejected 92-100% of rendered backs on dark ones -- which then
trained as background negatives. fill stays as a QA diagnostic. Deal/call
windows + sorting Condition A keep covering GT-leads-render."
```

---

### Task 3: score-anim window marks text, not geometry (`annotate/frame.py`)

**Files:**
- Modify: `majsoul_eye/annotate/frame.py:192-195`
- Test: `tests/test_hud_frame.py`

**Interfaces:**
- Consumes: `is_score_anim_window(state)` (unchanged), `hud_field_boxes` boxes carry `"text"`, `reach_stick_boxes` boxes do not.
- Produces: in-window HUD field boxes carry `text_reliable: False` and NO gate-driven `reliable: False`; Task 4's `hud_emit` keys off `d.get("text_reliable", True)`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_hud_frame.py` above the existing `test_score_anim_window_bundling_integration()` call line:

```python
def test_score_anim_marks_text_not_geometry():
    """A score-anim frame keeps HUD geometry labels (detector) and only marks
    the TEXT unreliable (reader): fixed-seed boxes stay valid and ink-snap
    follows the glyphs actually rendered, so the old blanket reliable=False
    deleted 410 correct train labels (spec 2026-07-10). reach_stick has no
    text and keeps its own in-window fill gate."""
    from majsoul_eye.state.replay import Replayer

    rp = Replayer(hero_seat=0)
    rp.apply({"type": "start_game", "id": 0})
    rp.apply({"type": "start_kyoku", "bakaze": "E", "dora_marker": "1m", "honba": 0,
              "kyoku": 1, "kyotaku": 0, "oya": 0, "scores": [25000] * 4,
              "tehais": [["1m"] * 13, ["?"] * 13, ["?"] * 13, ["?"] * 13]})
    st = rp.state
    st.last_event_types = frozenset({"reach", "dahai"})   # bundled reach record
    st.reach = [True, False, False, False]
    img2 = np.full((1080, 1920, 3), 200, np.uint8)        # bright: ink + stick render
    rec2 = annotate_frame(img2, st, hom)
    assert "hud:score_anim" in rec2["flags"]
    fields = [b for b in rec2["hud_boxes"] if "text" in b]
    sticks = [b for b in rec2["hud_boxes"] if b["name"] == "reach_stick"]
    assert fields and sticks
    for b in fields:
        assert b.get("reliable", True), b                 # geometry label survives
        assert b.get("text_reliable", True) is False, b   # text flagged for the reader
    for b in sticks:
        assert "text_reliable" not in b                   # no text; own fill gate rules
        assert b.get("reliable", True)                    # gray 200 >= REACH_FILL_OK
```

Add the call next to the existing one at the bottom:

```python
test_score_anim_marks_text_not_geometry()
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. $PY tests/test_hud_frame.py`
Expected: FAIL at `assert b.get("reliable", True)` — today the blanket gate sets `reliable=False` on every box.

- [ ] **Step 3: Implement** — in `majsoul_eye/annotate/frame.py`, replace

```python
        if is_score_anim_window(state):
            for b in boxes:
                b["reliable"] = False
            rec["flags"].append("hud:score_anim")
```

with

```python
        if is_score_anim_window(state):
            # The animation makes the TEXT untrustworthy, not the geometry:
            # fixed-seed boxes stay valid and ink-snap follows the glyphs
            # actually rendered (a truly blank field already came back
            # reliable=False from ink_snap). reach_stick carries no text and
            # keeps its own in-window fill gate (hud.REACH_FILL_OK).
            for b in boxes:
                if "text" in b:
                    b["text_reliable"] = False
            rec["flags"].append("hud:score_anim")
```

- [ ] **Step 4: Run the test, then the full suite**

Run: `PYTHONPATH=. $PY tests/test_hud_frame.py`
Expected: `test_hud_frame OK`.
Run: `for t in tests/test_*.py; do PYTHONPATH=. $PY "$t" || break; done`
Expected: all green (note: `tests/test_reach_stick.py` pins the reach-stick window gate — it must stay green untouched).

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/annotate/frame.py tests/test_hud_frame.py
git commit -m "fix(annotate): score-anim window flags HUD text, not geometry

Blanket reliable=False deleted every HUD box on reach frames while the frame
stayed in the dataset -- 410 train + 19 val frames taught the detector the HUD
region is background (the val 'false positives' on riichi_stick_count etc.).
Geometry is valid mid-animation: fixed seeds don't move and ink-snap tracks
rendered glyphs. Only the text is suspect -> text_reliable=False for the
reader; reach_stick keeps its own in-window fill gate."
```

---

### Task 4: `build_dataset.py` — emit HUD labels on score-anim frames, skip only reader crops

**Files:**
- Modify: `scripts/train/build_dataset.py` (module docstring lines ~26-29; `hud_emit` lines ~166-170; guard lines ~407-419; import line ~250)
- Modify: `majsoul_eye/annotate/hud.py` (the `REACH_FILL_OK` comment block, the sentence citing the wholesale build_dataset gate)
- Test: `tests/test_hud_dataset.py`

**Interfaces:**
- Consumes: `text_reliable` from Task 3.
- Produces: `hud_emit(rec, frame, w, h, obb)` — unchanged signature; a `text_reliable=False` box yields its YOLO line and no crop. Backward compat: OLD stored annotations (pre-Task-3) have `reliable=False` on score-anim boxes, so `--from-annotations` reuse of old records keeps old behavior (skipped entirely) — no compat shim needed.

- [ ] **Step 1: Write the failing test** — append to `tests/test_hud_dataset.py` before the final `print`:

```python
# text_reliable=False (score-anim window): detector line still emitted, reader
# crop skipped — geometry is right, only the rendered TEXT may lag GT.
rec2 = {"hud_boxes": [
    {"name": "score_self", "px_box": [900, 460, 1000, 500], "text": "25000",
     "text_reliable": False},
]}
lines2, crops2 = bd.hud_emit(rec2, frame, 1920, 1080, obb=False)
assert len(lines2) == 1 and lines2[0].startswith(f"{HUD_NAME_TO_ID['score_self']} ")
assert crops2 == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. $PY tests/test_hud_dataset.py`
Expected: FAIL at `assert crops2 == []` (today the crop is emitted).

- [ ] **Step 3: Implement `hud_emit`** — replace

```python
        text = d.get("text")
        if text is None or frame is None:
            # frame is None under --reuse-images (label-only, no pixels loaded) — the
            # YOLO line above still stands, but there's nothing to crop from.
            continue
```

with

```python
        text = d.get("text")
        if text is None or frame is None or not d.get("text_reliable", True):
            # No text (buttons/reach_stick), label-only reuse mode (frame is
            # None), or score-anim window (text_reliable=False: geometry good,
            # rendered text may lag GT) — the YOLO line above still stands,
            # but no reader crop.
            continue
```

- [ ] **Step 4: Remove the frame-level guard in `main()`** — replace

```python
        # HUD fields/buttons -> 55-class YOLO lines + reader crops (hud/<field>/<seq>.png).
        # belt-and-suspenders with Task 8's per-box `reliable` flag: is_score_anim_window
        # tolerates state=None (annotations-reuse path can have seqs with missing state).
        if not is_score_anim_window(state):
            hlines, hcrops = hud_emit(rec, frame, w, h, args.obb)
            if not args.no_yolo:
                yolo_lines += hlines
            if not args.no_crops:
                for rel, crop, meta in hcrops:
                    p = os.path.join(args.out, "hud", rel)
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    cv2.imwrite(p, crop)
                    hud_meta.append(meta)
```

with

```python
        # HUD fields/buttons -> 55-class YOLO lines + reader crops (hud/<field>/<seq>.png).
        # Emitted on EVERY frame: score-anim frames keep valid geometry (their
        # boxes carry text_reliable=False, so hud_emit skips only the reader
        # crops), and per-box `reliable` still gates everything else. The old
        # whole-frame skip left 410 rendered HUDs as background negatives.
        hlines, hcrops = hud_emit(rec, frame, w, h, args.obb)
        if not args.no_yolo:
            yolo_lines += hlines
        if not args.no_crops:
            for rel, crop, meta in hcrops:
                p = os.path.join(args.out, "hud", rel)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                cv2.imwrite(p, crop)
                hud_meta.append(meta)
```

Then check `is_score_anim_window` has no other use in the file and drop it from the import:

Run: `grep -n "is_score_anim_window" scripts/train/build_dataset.py`
Expected: only the import line ~250 remains → edit it to
`from majsoul_eye.state.replay import check_invariants, is_deal_window, is_call_window`.

Update the module docstring lines ~26-29 — replace

```
reader crop under ``hud/``. Score-roll/riichi-stick animation frames
(``is_score_anim_window``) are skipped for HUD entirely — belt-and-suspenders on
top of Task 8's per-box ``reliable`` flag. Old (pre-HUD) ``--from-annotations``
records simply lack ``hud_boxes`` and silently emit no HUD labels/crops.
```

with

```
reader crop under ``hud/``. Score-roll/riichi-stick animation frames keep their
YOLO lines (geometry is valid mid-animation) and skip only the reader crops via
the per-box ``text_reliable`` flag (annotate.frame sets it in-window). Old
(pre-HUD) ``--from-annotations`` records simply lack ``hud_boxes`` and silently
emit no HUD labels/crops; pre-2026-07-10 records carry the old blanket
``reliable=False`` and keep their old (fully skipped) behavior.
```

- [ ] **Step 5: Rewrite the stale contract in `majsoul_eye/annotate/hud.py`** — in the comment block above `REACH_FILL_OK = 0.35`, replace

```
# luminance-only fill conflated "not yet rendered" with "rendered but dark
# skin" and silently dropped 22.3%/19.7% of across/left sticks in datasets/v3
# (192 resp. 201 of them with fill>=0.1, i.e. rendered dim skins — measured on
# a sword-skin frame: fill 0.264) — worse than dropped: those frames still
# trained the detector with the stick as BACKGROUND. Off-window frames now
# trust GT regardless of fill. In-window frames are already excluded from HUD
# label emission wholesale by build_dataset's frame-level is_score_anim_window
# gate (working since Task 18's last_event_types fix); this per-box check
# remains as the finer, per-seat safety net for other consumers of the
# annotations. NOTE the stale-fallback residual (see is_call_window docstring):
```

with

```
# luminance-only fill conflated "not yet rendered" with "rendered but dark
# skin" and silently dropped 22.3%/19.7% of across/left sticks in datasets/v3
# (192 resp. 201 of them with fill>=0.1, i.e. rendered dim skins — measured on
# a sword-skin frame: fill 0.264) — worse than dropped: those frames still
# trained the detector with the stick as BACKGROUND. Off-window frames now
# trust GT regardless of fill. In-window this per-box check is the PRIMARY
# defense (2026-07-10): build_dataset no longer skips score-anim frames
# wholesale — HUD boxes keep their YOLO lines (fields carry text_reliable=False
# for the reader instead), so an in-window stick mid-render is dropped only by
# this gate. NOTE the stale-fallback residual (see is_call_window docstring):
```

- [ ] **Step 6: Run the test, then the full suite**

Run: `PYTHONPATH=. $PY tests/test_hud_dataset.py`
Expected: `test_hud_dataset OK`.
Run: `for t in tests/test_*.py; do PYTHONPATH=. $PY "$t" || break; done`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add scripts/train/build_dataset.py majsoul_eye/annotate/hud.py tests/test_hud_dataset.py
git commit -m "fix(dataset): score-anim frames keep HUD YOLO labels, skip only reader crops

The whole-frame is_score_anim_window skip wrote the image with tile labels but
zero HUD labels -- 410 train frames of rendered-HUD background negatives, and
19 val frames behind the fake riichi_stick_count/honba_count FPs. hud_emit now
keys reader crops off text_reliable (set by annotate.frame in-window); YOLO
lines are emitted on every frame. reach_stick's per-box in-window fill gate is
now the primary in-window defense (hud.py comment updated)."
```

---

### Task 5: Real-data regression, PIPELINE/STATUS entries

**Files:**
- Modify: `docs/PIPELINE.md` (annotate-stage notes), `docs/STATUS.md` (new § entry)
- No new committed scripts — the regression runs as inline commands; outputs land in the scratchpad and the numbers go into STATUS.

**Interfaces:**
- Consumes: everything above.
- Produces: measured before/after numbers pinned in STATUS; stale-datasets note.

- [ ] **Step 1: Rebuild the three probe games end-to-end** (self-contained mode re-runs `annotate_frame` with all four fixes + the btn fix; ~10-20 min total, run in background):

```bash
SP=<scratchpad>/regress
for g in "ai_session4/run_5/game1" "ai_session2/run_6/game9" "ai_session2/run_1/game1"; do
  out="$SP/$(echo $g | tr '/' '_')"
  PYTHONPATH=. $PY scripts/train/build_dataset.py \
    "captures/raw/$g/$(basename $g).jsonl" "captures/raw/$g" \
    --out "$out" --backs --obb --no-crops
done
```

(Check `build_dataset.py --help` first: if `--no-crops` also suppresses `hud/` crops that's fine — the regression only reads YOLO labels.)

- [ ] **Step 2: Measure** — for each output dir:

```bash
$PY - <<'EOF'
import glob, pathlib
BASE = {  # from datasets/v5 (pre-fix): kept_frames/raw, backs_per_frame
  "ai_session4_run_5_game1": (50, 306, 15.2),
  "ai_session2_run_6_game9": (59, 64, 5.6),
  "ai_session2_run_1_game1": (214, 266, 43.4),
}
for name, (k0, raw, b0) in BASE.items():
    d = f"<scratchpad>/regress/{name}"
    imgs = glob.glob(f"{d}/yolo/images/*.png")
    backs = sum(1 for lp in glob.glob(f"{d}/yolo/labels/*.txt")
                for l in pathlib.Path(lp).read_text().splitlines()
                if l.strip() and int(l.split()[0]) == 37)
    print(f"{name}: kept {k0}/{raw} -> {len(imgs)}/{raw}   "
          f"backs/frame {b0} -> {backs/max(len(imgs),1):.1f}")
EOF
```

Acceptance (spec targets, refined for the btn fix landing in the same rebuild):
- `ai_session4_run_5_game1`: kept ≥ 80% (was 16%), backs/frame 35-43 (was 15.2)
- `ai_session2_run_6_game9`: kept ≥ 85% (was 92%), backs/frame 35-43 (was 5.6)
- `ai_session2_run_1_game1` (bright control): backs/frame within ±5% of 43.4 — the fill gate was inert here, so per-frame back counts must not move. Frame retention may RISE (Condition-B recovery ~4% + btn-fix effects); it must not fall below 78%.

If a number misses: STOP, diagnose before touching docs (draw the back quads on the offending frames first — step 3).

- [ ] **Step 3: Eyeball check** — overlay labels on 3 mid-game frames per dark game:

```bash
PYTHONPATH=. $PY scripts/inspect/overlay_labels.py --help
```

Use it per its CLI (it draws label boxes on a frame) on 3 frames from `ai_session4_run_5_game1`; confirm back quads land on tiles, not felt. Save renders under the scratchpad; they are not committed.

- [ ] **Step 4: Write docs** —

`docs/STATUS.md`: append the next § entry (check `grep -n "^### 1\." docs/STATUS.md | tail -1` for the number; the btn fix took §1.55). Content: the three defects (one line each), the headline measurements from the spec (17.6% frame-level Condition-B false-fire; fill felt=1.00 vs dark back 0.24; 410/19 score-anim frames), the regression table from Step 2, and the staleness note: **all datasets v1-v5 backs + HUD labels stale; rebuild rides the pending btn-fix rebuild (`--backs` re-annotate + `build_datasets.py <name> --force` + retrain)**.

`docs/PIPELINE.md`: in the annotate-stage section, update the backs and HUD bullet(s): backs per-box fill gate removed (`fill` = QA diagnostic), `sorting_suspect` = Condition A only, score-anim frames now emit HUD YOLO labels with `text_reliable=False` steering the reader-crop skip.

- [ ] **Step 5: Full suite one last time**

Run: `for t in tests/test_*.py; do PYTHONPATH=. $PY "$t" || break; done`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add docs/STATUS.md docs/PIPELINE.md
git commit -m "docs: STATUS/PIPELINE — backs fill gate + Condition B removed, score-anim HUD labels kept

Regression on rebuilt probe games: dark-skin game frame retention 16%->NN%,
backs/frame 15.2->NN.N / 5.6->NN.N; bright control unchanged (NN.N). All
datasets v1-v5 backs+HUD labels stale; rebuild rides the btn-fix rebuild."
```

(Replace NN with the measured numbers from Step 2 before committing.)

---

## Self-review notes

- **Spec coverage:** D1 → Task 1; D2 → Task 2 (incl. CLAUDE.md bullet); D3 → Tasks 3+4 (frame.py producer, build_dataset consumer, hud.py comment); tests section → Tasks 1-4 Step 1s (incl. flipping the pre-existing black-frame assertion the spec's test list implied but did not name); real-data regression + pipeline discipline → Task 5. Out-of-scope items (re-annotate/rebuild/retrain, intra-row outlier gate, tile_live_mask) appear in no task — correct.
- **Type consistency:** `text_reliable` read via `d.get("text_reliable", True)` in Task 4 matches Task 3's write; `sorting_suspect` signature unchanged; `back_boxes` return shape unchanged.
- **Order dependency:** Tasks 1-2 are independent of 3-4; Task 4 needs Task 3 (flag producer) only for end-to-end semantics — its unit test passes standalone. Task 5 needs all.
