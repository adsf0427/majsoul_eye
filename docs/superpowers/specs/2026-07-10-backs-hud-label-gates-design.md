# Remove the two label gates that delete visible objects: backs fill + score-anim HUD

**Date:** 2026-07-10
**Status:** approved (design)
**Scope this session:** code + tests + doc updates. Re-annotate / rebuild / retrain are left
to the user to trigger.

Sibling of [`2026-07-05-skin-agnostic-back-gate-design.md`](2026-07-05-skin-agnostic-back-gate-design.md)
(which fixed the dora/meld back gate) and of the 2026-07-10 button-plate fix (STATUS §1.55).
All three are the same bug family: **a pixel gate calibrated on one skin silently deletes the
label of an object that is plainly on screen, while the frame stays in the dataset — so the
object trains as a background negative.**

## Problems

Three defects, all in the annotate stage, all measured on `datasets/v5` + its captures.

### D1 — `sorting_suspect` Condition B false-fires, dropping whole frames

`annotate/backs.py:218` still ends with

```python
return drawn is not None and dist(drawn, tm) < 0.7 * dist(drawn, empty)
```

STATUS §1.48 records this line as **removed**; it was never committed. `ee837cb` (the only
later commit to the file) shipped `meld_bias → 0` and holding-row labeling, not this.

`build_dataset.py:361` turns a `backs_sorting` flag into a **whole-frame drop**, not a
per-seat one. So every Condition-B false positive costs the frame's tiles, HUD, buttons and
reach sticks too.

Measured (12 games sampled across all sessions, every 3rd frame, `is_deal_window` /
`is_call_window` frames already excluded; n = 1266 eligible frames):

| | frames | share |
|---|---|---|
| `sorting_suspect` fires (frame dropped today) | 309 | 24.4% |
| …of which Condition B alone (recoverable) | 223 | **17.6%** |
| …Condition A only (the sane signal, keep) | 86 | 6.8% |

Per-seat, on the dark-skin `ai_session4/run_5/game1`: 256 firings, of which **253 are
Condition B**; Condition A fires 3. Even the orange-back `ai_session2/run_1/game1` shows
B-only on 4% of settled seat-frames.

This is the single largest frame-drop cause in the pipeline. v5 keeps 32,283 of 47,965 raw
`ok`/`timeout` frames (32.7% dropped). Removing Condition B is projected to recover ~7,500
frames (**+23% dataset**).

The recovered frames carry the scarcest classes. Two games alone
(`ai_session5/run_1/game19`, `ai_session4/run_5/game1` → 506 frames recovered):

| class | recovered instances | current v5 train total |
|---|---|---|
| `btn_kan` | 2 | 38 |
| `btn_tsumo` | 2 | 25 |
| `btn_ron` | 1 | 57 |
| `btn_riichi` | 1 | 124 |
| `btn_chi` / `btn_pon` / `btn_skip` | 25 / 19 / 47 | 557 / 336 / 1102 |
| `reach_stick` (on-screen) | 155 | 5922 |
| red 5 (on-screen) | 164 | 19274 |

### D2 — the `FILL_OK_BACKS` gate has no discriminative power, and is inverted on dark skins

`backs.py:263` marks a back box unreliable when `tile_live_mask` coverage of the box's
**axis-aligned bbox** falls below `FILL_OK_BACKS = 0.25`. `build_dataset` then drops the box,
keeping the frame — so the opponent's rendered hand row becomes a background negative.

`tile_live_mask` is `(S > 60) | (V > 110)` — "colored **or** bright". The 2026-07-05 spec
validated it on **dora slots and meld cells**: tight, near-upright boxes whose unrendered
negative reads black (0.000). Measured today, that role is still healthy — on dark skins the
dora backs pass at **100%**.

`backs.py` reused it for a geometry it was never validated on: **edge-on opponent hand rows**,
whose skewed quad's axis-aligned bbox is mostly table felt. Measured coverage:

| | tile-back box | genuinely empty felt |
|---|---|---|
| bright table (`ai_session2/run_1/game1`) | 0.99 | **1.000** |
| dark table (`ai_session4/run_5/game1`) | **0.24** | **1.000** |
| pale table (`ai_session2/run_6/game9`) | 0.18 | **1.000** |

Every mahjong felt is colored or bright, so the gate's negative class reads 1.000 everywhere.
On bright tables the gate therefore passes *everything* including empty slots — it has never
functioned, and all "good" backs data was collected with it effectively disabled. On dark
tables the dark `RML` tile back measures **S ≈ 59, V ≈ 55**, missing both thresholds by a
hair, so the positive class scores *below* the negative class and the gate rejects the tiles
while accepting the felt.

No threshold on this signal can separate the classes. The gate must be removed, not retuned.

Result today: opponent backs kept at **5%** on skinned games vs 100% on default. Six v5 games
sit below 30 backs/frame (median 39.8), 766 frames = 2.4% of the set — and one of them,
`ai_session4_run_5_game1`, is **in the val split**, which is why the production OBB detector
reports `back` precision 0.989 with 625 "false positives" that are in fact correct detections
of unlabeled backs.

### D3 — score-anim frames keep the image but drop every HUD label

`frame.py:192` blanket-sets `reliable = False` on all HUD field and reach-stick boxes inside
`is_score_anim_window`, and `build_dataset.py:403` additionally skips HUD emission for the
whole frame. The frame is still written with its tile labels. 410 of 31,033 train frames
(and 19 of 1,250 val frames) therefore teach the detector that the HUD region is background.

The window's premise is wrong. Reach/score animation makes the **text** unreliable, not the
**geometry**:

- `round_label`, `seat_wind_self`, `wall_count` keep a fixed seed box (no snap) — always valid.
- Score / stick / honba fields ink-snap to the glyphs *actually rendered*, which is exactly
  what the detector should learn. A genuinely blank field makes `ink_snap` return `None`,
  which already sets `reliable = False`.
- `reach_stick_boxes` already applies its own in-window fill gate (`REACH_FILL_OK`), designed
  as "the finer, per-seat safety net".

Verified on `ai_session_run_8_game1/000129.png`: the HUD is fully rendered and **static**
(東1局 / 余35 / 25000 / 24000×2), the label file holds zero HUD boxes, and the production
detector finds all 11 HUD elements at conf 0.93–0.97.

## Design

### 1. `annotate/backs.py`

- **Delete Condition B** from `sorting_suspect`: drop the trailing `return drawn is not None
  and ...` and return `False` after the Condition-A check. Rewrite the docstring, which
  currently argues *for* "phase B". `_drawn_fw`/`_patch_stats` stay — Condition A uses them.
- **Delete the reliability gate** in `back_boxes`: remove `if f < FILL_OK_BACKS: b["reliable"]
  = False` and its `low_fill` flag; remove the `FILL_OK_BACKS` constant. Keep writing
  `b["fill"]` — it is a measured diagnostic with no consumer today, costs nothing extra (the
  mask and integral are already built), and is useful for QA.

What still protects backs after this: `is_deal_window` (whole-frame drop, GT leads the deal
animation), `is_call_window` (whole-frame drop, forced-dahai animation in flight), and
`sorting_suspect` Condition A (the bare-slot reflow signal, 0.4–3%).

`tile_live_mask` is **not** touched — it is healthy in the dora/meld role it was calibrated
for, and changing it would disturb river, meld and dora labeling that measure clean today.

### 2. `annotate/frame.py`

Inside `is_score_anim_window`, stop blanket-marking boxes unreliable. Instead mark only the
text as suspect on boxes that carry one:

```python
if is_score_anim_window(state):
    for b in boxes:
        if "text" in b:
            b["text_reliable"] = False
    rec["flags"].append("hud:score_anim")
```

`reach_stick_boxes` carries no `text`, so it is untouched and its own in-window fill gate
becomes the primary defense — the role its docstring already claims.

### 3. `scripts/train/build_dataset.py`

- Remove the frame-level `if not is_score_anim_window(state):` guard around `hud_emit`.
  `hud_emit` already filters on `reliable`.
- In `hud_emit`, emit the YOLO label line always, but skip the `hud/` reader crop when
  `d.get("text_reliable", True)` is False. Reader training data stays exactly as correct as
  it is today; detector labels are recovered.

### 4. Documentation

- `annotate/hud.py`: the `REACH_FILL_OK` comment block asserts "In-window frames are already
  excluded from HUD label emission wholesale by build_dataset's frame-level
  is_score_anim_window gate". That contract is being deleted — rewrite it to say the per-box
  gate is now the only in-window defense.
- `annotate/backs.py` module docstring: drop the Condition-B / fill-gate descriptions.
- `CLAUDE.md`: the `backs.py` bullet still says "holding seats skipped + flagged, builds drop
  those frames whole" — stale since `ee837cb` labeled holding rows.
- `docs/PIPELINE.md` + `docs/STATUS.md`: new entry per the project's pipeline-impact rule.

## Tests

Written before the code (TDD). Plain scripts under `tests/`, pytest-compatible.

**`tests/test_backs.py`** (extend):
- `sorting_suspect` returns `False` for a synthetic row whose drawn slot reads like a tile but
  whose row slots all read like tiles — the Condition-B signature. (Fails today.)
- `sorting_suspect` still returns `True` when a row slot reads like the empty reference —
  Condition A survives.

  Both drive the decision logic by monkeypatching `backs._patch_stats` to return scripted
  `(gray mean, gray std)` tuples per quad, rather than painting an image whose warped quads
  happen to land on chosen pixels. The changed unit *is* the decision logic; the quad→pixel
  mapping is already covered by the template-geometry tests above.
- `back_boxes` on a synthetic dark-back frame (patch below both `S>60` and `V>110`) leaves
  every box `reliable` and still records a low `fill`. (Fails today.)
- `FILL_OK_BACKS` is gone from the module namespace.

**`tests/test_hud_frame.py`** (extend):
- In a score-anim state, HUD field boxes keep `reliable is True` and carry
  `text_reliable is False`.
- In a score-anim state, `reach_stick` boxes carry no `text_reliable` key and remain governed
  by `REACH_FILL_OK`.

**`tests/test_hud_dataset.py`** (extend):
- `hud_emit` on a `text_reliable=False` box yields one YOLO line and zero reader crops.

**Real-data regression** (not a unit test; a verification script run once, output pasted into
STATUS):
- Re-annotate `ai_session4/run_5/game1` and `ai_session2/run_6/game9`. Assert backs/frame
  rises 15.2 → ~39 and 5.6 → ~39; frame retention rises 16% → ~90% and 92% → ~92%.
- Draw back quads on 3 frames of each and confirm by eye they still land on tiles.
- Assert a default game (`ai_session2/run_1/game1`) changes by **≤1%** in backs/frame and
  frame count — the fix must be a no-op where the gate was already inert.

Full `tests/` suite must stay green.

## Pipeline discipline

This changes annotate-stage outputs, staling every derived dataset. The in-flight button-plate
fix (STATUS §1.55) already requires `--backs` re-annotate + `build_datasets.py <name> --force`
+ retrain, so these three fixes ride the same rebuild at zero marginal cost.

Expected post-fix dataset: ~39,800 frames (from 32,283), `back` ≈ 39/frame on every game,
`btn_kan` / `btn_tsumo` roughly doubled, and the `back` / HUD false positives in the val
per-class table gone.

## Out of scope

- Re-annotate / rebuild / retrain (user triggers).
- Replacing the deleted fill gate with an intra-row outlier test. Considered and rejected for
  now: no evidence that single-box occlusion occurs once `is_call_window` drops the animation
  frames, and adding a code path for an unobserved failure is speculative. Revisit if the
  rebuilt data shows phantom backs.
- `tile_live_mask` / `tile_back_mask` behaviour, and the dora + meld back gates that use them.
- The three starved button classes (`btn_kyushu` has **0** training instances and cannot be
  fixed by any label gate — it needs capture or synthesis).
