# Skin-agnostic tile-back reliability gate

**Date:** 2026-07-05
**Status:** approved (design)
**Scope this session:** code + tests + doc updates. Re-annotate / rebuild / retrain are left
to the user to trigger.

## Problem

In `annotate/frame.py`, the dora-strip face-DOWN slots are validated by an **orange**
tile-back HSV check (`hue 8-28, sat>80, val>110`). In skin-swapped capture games
(`captures/raw/ai_session2/`, runs 21-23, `skins.enabled: true`, 牌背/slot 7 randomized) the
tile back is not orange, so the check reads ~0 coverage → `fill < FILL_OK(0.25)` →
`reliable=False` → `scripts/train/build_dataset.py` drops the box. The same orange mask
(`pipeline.py tile_back_mask`) also gates ankan/meld back cells and drives `snap_meld_strip`.

### Measured impact (2026-07-05)

- Orange-gate coverage on dora backs: DEFAULT median 0.84 (100% pass FILL_OK); SKINNED
  (run_21/22 game1) median 0.000-0.002, **0% pass** — total collapse.
- Built YOLO labels, class 37 (`back`): default 3807 (run_8_game1) / 2955 (run_3_game1);
  skinned **0 / 0 / 0** (run_21/22/23 game1) despite 262/371/217 labeled frames each.
- By zone (annotation dump): `back` ≈ dora-strip unrevealed slots (~95-97%, e.g. 3697/949fr)
  plus ankan meld cells (~3-5%, 110-168/game). River never carries `back`.
- Net harm: ~4 visible-but-unlabeled backs per skinned frame sit in the training image as
  YOLO **background negatives** — actively teaching the detector NOT to detect skinned backs;
  and the classifier `back` class gets zero skinned examples.

### Rejected alternative — "classify the 5 fixed slots; no-clear-class → back"

Tested against the production classifier: OOD skinned backs are **confidently wrong**, not
low-confidence (softmax OOD). run_21 skin → all `3m` @ conf 0.33; run_22 skin → all `8s`
@ conf 0.88; 0% predicted `back`, while real faces sit at ~0.99. Confidence is skin-dependent,
so no single threshold separates OOD-back from face. It is also circular at annotate time (GT
already knows back-vs-face). The classifier-on-fixed-slots idea is a good **runtime** dora
reader, but it is *downstream* of this data fix: once skinned backs are trained into `back`,
the classifier predicts `back` directly (100%, conf 1.0, as on default backs today).

## Root cause

The orange mask conflates two distinct jobs:

- **Role A — reliability gate.** GT already knows the slot/cell *is* a back; this check only
  needs **liveness** ("is a tile rendered here"), to drop the rare transient frames where GT
  leads the client render (unrendered → empty/black patch) so we don't stamp `back` on an
  empty region (which would teach location-locked phantom backs). This is the labeling-critical
  role and the one that broke on skins.
- **Role B — snap discrimination.** `snap_meld_strip` uses the back mask to tell back cells
  from face cells when rigidly aligning a floating meld strip. This needs a signal that
  **separates backs from faces**.

A single "colored-or-bright" mask cannot do Role B (it also lights up white faces), so the
fix uses **two** masks.

## Signal choice (data-locked)

Coverage of the dora-slot / cell patch; positives must pass, unrendered negatives must fail:

| population (n) | `sat (S>60)` | `val (V>110)` | **`sat\|val`** |
|---|---|---|---|
| default backs (564) | p05 0.950 | 0.885 | **0.984** |
| skinned backs (4606) | min 0.293 | min 0.158 | **min 0.377** |
| unrendered negatives (4) | 0.000 | 0.000 | **0.000** |

`sat` alone is fragile (a desaturated skin dipped to 0.293, near the 0.25 threshold); `val`
alone worse (dark skin 0.158). **`(S>60) | (V>110)`** is robust — a colored-OR-bright hedge
catches both grey and dark skins (worst skinned back 0.377), negatives 0.000. Threshold stays
`FILL_OK = 0.25` (comfortable margin both ways).

## Design

### 1. `annotate/pipeline.py` — new `tile_live_mask` (Role A)

```python
def tile_live_mask(fullwarp_bgr):   # "a tile is rendered here" — any skin
    hsv = cv2.cvtColor(fullwarp_bgr, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 1] > 60) | (hsv[..., 2] > 110)).astype(np.uint8)
```

Used for the reliability fill of GT-known **back** cells/slots (dora + meld). It is NOT used
to discriminate face vs back — it lights up faces too, which is fine because it is only ever
applied to slots/cells GT already labels `back`.

### 2. `annotate/pipeline.py` — `tile_back_mask` → saturation-based (Role B)

```python
def tile_back_mask(fullwarp_bgr):   # colored tile back; orange ⊂ this
    hsv = cv2.cvtColor(fullwarp_bgr, cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] > 70).astype(np.uint8)
```

Skin-agnostic and complementary to `tile_face_mask` (`S<70 & V>165`) by construction, so
`snap_meld_strip` keeps its face/back discrimination. Orange (old `s>110`) is a subset, so
default snap behaviour is preserved unless green felt intrudes (see Risk).

### 3. `annotate/frame.py`

- **Dora backs** (lines ~133-135): replace the orange coverage with
  `float(P.tile_live_mask(img[y1:y2, x1:x2]).mean())`. Keep `fill < FILL_OK → reliable=False`.
  (Dora slots live on the 2D HUD / original image; the mask function is applied to the slot
  patch directly.)
- **Meld backs** (line ~80): build `ii_l = cv2.integral(P.tile_live_mask(full))`; for back
  cells use `ii_l` (not `ii_b`) in the fill. `snap_meld_strip` still receives `mb`
  (now the saturation-based back mask).
- Revealed-face and hero-hand checks: **untouched** (牌面/slot 13 not skinned by default;
  white-face gate still valid). Update the dora docstring (lines ~119-121) to describe
  `fill` as skin-agnostic content coverage for backs.

## Risk & fallback

The saturation-based `tile_back_mask` (Role B) could pick up green felt near a meld strip on
default games, shifting `snap_meld_strip` alignment. **Verification:** before/after annotate
diff on default game(s) containing ankan — compare back-cell snap `(da, dc)` and box
positions. **Fallback if default snap shifts materially:** keep `tile_back_mask` orange for
snap and ship only Role A (the fill-gate fixes). That still fully delivers "skinned ankan
backs get labeled"; only skinned-ankan *alignment precision* is left unchanged. Role A is
unconditional; Role B ships only if verified clean.

## Tests

- **Unit — `tile_live_mask`:** blue/bright (skin) patch → coverage ≥0.25; orange patch → high
  (default regression); black patch → 0.0.
- **Integration — `annotate_frame`:** synthetic frame with a non-orange (blue) dora back slot
  → that `dora_boxes` entry `reliable=True`; black frame → `reliable=False` (existing
  black-frame assertion must stay green).
- Full `tests/test_annotate_*.py` suite passes.

## Pipeline discipline

This changes an annotate-stage output, staling derived data. Actions:
- `docs/PIPELINE.md`: note the back-gate change (annotate stage).
- `docs/STATUS.md`: add an entry.
- Note that `datasets/precise_ai_run_21..23` + `obb_precise_ai_run_21..23` (and any aggregate)
  are stale for the `back` class; rebuild via `regen_detector_dataset.sh` (annotate is the
  stale step) or `build_datasets.py <name> --force`, then retrain. Rebuild/retrain are
  out of scope for this session (user-triggered).

## Out of scope

- Re-annotate / rebuild / retrain (user triggers).
- Runtime dora reading via per-slot classification (a natural follow-up once `back` is trained
  across skins).
- Skinned 桌布/场景 (slots 6/8) robustness — those affect background pixels, handled by the
  detector's general negatives, not this gate.
