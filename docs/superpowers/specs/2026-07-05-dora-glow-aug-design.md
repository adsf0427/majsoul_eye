# Dora-glow augmentation + coverage stats — design

**Date:** 2026-07-05
**Status:** approved (brainstorming)

## Problem

Mahjong Soul renders a golden bloom / rim-highlight / sparkle on **宝牌 (dora)**
tiles — red fives always, plus any tile whose value matches the current dora
(the tile *after* a dora indicator). The current training does **not** model this:

- **Detector (`scripts/train/train_detector.py`)** is a thin ultralytics wrapper
  passing **no** augmentation hyperparameters, so it inherits YOLOv8 defaults
  (`hsv_v=0.4`, `fliplr=0.5`, `mosaic=1.0`, …). Only global HSV/value jitter, no
  localized bloom.
- **Classifier (`scripts/train/train_classifier.py`)** uses a hand-written
  `_augment` (rotate ±7°, translate ±6%, global brightness ×[0.8,1.2]) — again
  global, no bloom.

Labels are auto-generated from **real** Majsoul frames (Akagi GT), so genuine
sparkle samples DO exist in the data — but coverage depends on how many dora
instances were captured and which classes happened to be dora. There is no
evidence today about whether that natural coverage is sufficient.

Two additional aug facts surfaced while scoping:
- `fliplr=0.5` (YOLO default) is **wrong** for tiles: they are directional
  (pips/characters read one way), so a horizontal flip fabricates mirror tiles
  that never occur in reality.
- The effective aug config is invisible today (nothing logged), so runs are not
  reproducible from the command line.

## Scope decisions (settled during brainstorming)

1. **Sparkle augmentation depth:** *reserve a global knob only.* Expose YOLO's
   built-in HSV/geometry params as CLI flags with a slightly brighter default;
   do **not** write a custom localized golden-bloom Albumentations transform yet.
   Whether to invest in real bloom is deferred to the stats script's output.
2. **Glow definition:** red fives (always) **plus** any tile whose normalized
   value equals `next_of(indicator)` for an active dora indicator. Indicator
   tiles themselves and `back` do not count.

Non-goals: custom bloom transform, touching the classifier's `_augment`,
retraining, changing the 38-class order.

## Component 1 — coverage stats tool `scripts/inspect/count_dora_glow.py`

A **one-shot QA tool** (PIPELINE.md §4 classification — an inspection script, not
a pipeline stage), Akagi-free at runtime (reads GT captures like the annotator).

**Purpose:** per-class "how many glowing vs total labeled-box instances does the
detector see, and which classes are starved of glowing examples" — the evidence
that decides whether real bloom aug is worth building.

**Data flow:**
1. Sources default to `captures/raw/ai_session` (same as `build_datasets`);
   `--sources` can add `manual`. Alternatively `--dataset datasets/<v>` reads
   that version's `games.json` to lock the exact game set + val game.
2. Per game: `capture.gtframes.build_seq_state(capture)` → `seq → BoardState`;
   `capture.gtframes.load_frames(frames_dir)` → `seq → frame path`. **Only count
   seqs that have a saved frame** (= frames that actually become training images).
3. Per counted frame, enumerate visible tiles from `BoardState` directly (no
   geometry / no cv2): `rivers[seat]` (`RiverTile.pai`), `melds[seat]`
   (`Meld.tiles` + `called_pai`/`added_pai`), `hero_hand`, and the `dora_markers`
   indicator strip. Assign each a zone ∈ {river, meld, hand, dora}.

**Glow rule:**
```
dora_values = { next_of(red_to_normal(ind)) for ind in state.dora_markers }
glow(tile, zone) = is_red_five(tile)                                   # red fives always glow
             or (zone in {hand, river, meld}
                 and red_to_normal(tile) in dora_values)
# zone == 'dora' (the indicator strip) and tile == 'back' never count as glow
```

`next_of(name)` is a **new shared helper in `majsoul_eye/tiles.py`** implementing
the standard dora progression: suits `1→2→…→9→1` per m/p/s; winds
`E→S→W→N→E`; dragons `P→F→C→P` (白→發→中→白); a red-five indicator collapses to
its plain five first. Returns a canonical non-red name (dora comparison is on
`red_to_normal`, so both `5m` and `5mr` count as glowing when 5m is dora).

**Counting granularity:** per-frame labeled-box instances (= training crops), not
de-duplicated physical tiles. This is the correct denominator for "does the model
see enough glowing X" (each frame is one training image); the output explicitly
labels it as frame×tile instances.

**Output:** stdout only, no artifacts written. A per-class table
`class | total | glow | glow%` (the 37 glow-eligible classes; `back` is omitted —
it never appears in the counted zones), printed once per split: **train** (all
games except the val game) and **val** (the held-out game, default
`ai_run_8_game1`, override `--val`). Each table ends with a TOTAL row (aggregate
total/glow/glow%) and a trailer highlighting classes with `glow < --min-glow`
(default 20).

## Component 2 — explicit detector aug config in `train_detector.py`

Promote YOLO augmentation hyperparameters to CLI flags, pass them explicitly into
`model.train(...)`, and print the effective aug config in the startup log for
reproducibility.

| Flag | New default | Rationale |
|---|---|---|
| `--fliplr` | **0.0** (was YOLO 0.5) | tiles are directional; mirror tiles don't exist |
| `--hsv-v` | **0.5** (was YOLO 0.4) | stronger value jitter as a global sparkle/brightness proxy |
| `--hsv-s` | 0.7 | unchanged default, now adjustable |
| `--hsv-h` | 0.015 | unchanged default, exposed |
| `--degrees` | 0.0 | unchanged default, exposed |
| `--translate` | 0.1 | unchanged default, exposed |
| `--scale` | 0.5 | unchanged default, exposed |
| `--mosaic` | 1.0 | unchanged default, exposed |
| `--close-mosaic` | 10 | unchanged default, exposed |
| `--mixup` | 0.0 | unchanged default, exposed |

`flipud` stays 0 (not exposed — never wanted for a top-down board). Custom
localized bloom is **not** implemented; the decision is deferred to Component 1's
output.

## Pipeline-impact discipline (CLAUDE.md)

- `train_detector.py` is a **training stage** and its **defaults change**
  (`fliplr` 0.5→0, `hsv_v` 0.4→0.5) → update the detector-training command/defaults
  in `docs/PIPELINE.md` and add a `docs/STATUS.md` entry.
- `count_dora_glow.py` is a **new one-shot tool** → classify it as such in
  `docs/PIPELINE.md` §4 (alongside `inspect_capture.py`, `overlay_labels.py`).
- No derived data under `out/`/`datasets/` is staled by these changes (no rebuild
  needed); the aug default change only affects the *next* detector training run.

## Testing

- `tests/test_dora_glow.py` (plain script, pytest-compatible, no GPU/frames):
  unit-test `next_of` across suits/winds/dragons/red-five wrap-around, and the
  `glow(tile, zone)` rule (red five always; dora-value match only in
  hand/river/meld; indicator strip and `back` excluded).
- Smoke-run `count_dora_glow.py` on one game to confirm it walks GT → prints a
  table without error.
- `train_detector.py`: verify the CLI parses and the effective aug dict is
  assembled/logged (inspect the kwargs / `--epochs 1` dry check); no long train.

## Open question deferred (not this spec)

Whether to build a real localized golden-bloom augmentation is intentionally left
open, to be answered by the stats tool. If coverage is adequate, no bloom needed;
if specific classes are starved, that's the trigger to invest.
