# majsoul_eye

Robust image recognition of **Mahjong Soul (雀魂)** game state (场况) from screen
images — usable by a protocol-independent vision bot, a HUD overlay, or for
recognizing arbitrary external screenshots.

> Full design, rationale, element-by-element method table, risks, and roadmap:
> **[`docs/DESIGN.md`](docs/DESIGN.md)** (mirror of the approved plan).

## Why this exists (and what it reuses)

- **`../auto/mycv`** is an existing, working *pure-vision* Mahjong Soul bot. It
  already recognizes the full board (河/副露/dora/scores/winds) and solves the
  4-seat perspective problem. majsoul_eye is a **clean rewrite that reuses
  mycv's assets** (`tile.model` classifier, 707 debug frames, coordinate
  knowledge, contour-based 河/副露 detection, seat-rotation math) as baseline +
  bootstrap — not a green-field rebuild.
- **`../Akagi`** has the full game state from MITM-parsed `liqi` protobuf. Here
  it is a **training-time oracle** that produces free, accurate labels — not a
  runtime dependency. (When you have the protocol you don't need vision; vision's
  whole value is the feed-less case.)

## Architecture (one line)

Hybrid: **deterministic ROI crop + small CNN / digit-classifier / OCR** for the
fixed "easy" zones (hand, dora, scores, buttons); a **YOLO detector (OBB where
tiles rotate)** for the perspective "hard" zones (四家河, 副露) and for
generalizing to mobile / external screenshots; an **anchor-based normalization**
front-end so fixed-slot logic survives arbitrary resolutions.

## Labeling, in one line

`Akagi protocol GT = WHAT` (which tile, who discarded, what score) +
`geometry / contour detection = WHERE` (pixel box) → auto-generated YOLO labels,
zero hand-drawing. mycv's "contour-localize + assign class from GT's ordered
discard list" *is* a free auto-annotator.

## Layout

```
majsoul_eye/
  __init__.py
  tiles.py          # unified 38-class taxonomy + MJAI interop (shared by all components)
  coords.py         # normalized ROI model (mycv-seeded easy-zone boxes, hand/dora slots, river zones)
  normalize.py      # board locators: fullscreen / letterbox / anchor(TODO) -> canonical 16:9 frame
  capture/          # ⚠️ DEV-ONLY, Akagi-coupled — recorder (never imported by the shipped recognizer)
    schema.py       #   GTRecord + JSONL I/O
    akagi_tap.py    #   monkeypatch MajsoulBridge.parse_liqi -> tee raw-liqi + MJAI to JSONL
    screen.py       #   window grab (win32 + mss)
    sync.py         #   async settle/straddle screenshot syncer (non-blocking)
  state/replay.py   # pure replayer: capture -> full 4-player BoardState + invariants
  label/autolabel.py# easy zones (hand/dora/score) -> label samples + crops + YOLO lines
  label/river.py    # 河: per-seat perspective GRID (WHERE) + GT-order assign (WHAT)
  label/meld.py     # 副露: per-seat strip + GT order (opt-in; sides approximate)
  recognize/classifier.py  # TileNet 38-class tile classifier + inference wrapper
scripts/
  record_gt.py      #   launch Akagi w/ recorder (+ --screenshots)
  crop_game.py      #   crop non-fullscreen captures back to a 16:9 canvas
  build_dataset.py  #   synced capture -> classifier crops + YOLO dataset
  train_classifier.py #  multi-game training w/ cross-game split
  inspect_capture.py / overlay_labels.py  # join + visual debug
tests/              # test_{tiles,replay,sync,label,river,meld,classifier}
docs/DESIGN.md      # design & rationale       docs/STATUS.md  # living status + roadmap
```

## Status

**P0–T6 complete — full auto-label pipeline + first tile classifier, validated end-to-end on 2 games (zero manual annotation).**

- Pipeline: `record(F11) → debounce capture → protocol-GT replay → geometric auto-label → train`.
- Auto-labeling: hand ✓, 4 rivers ✓ (98.5–99.5% on-tile, per-seat perspective grids), dora ✓; melds opt-in (3D side-seats approximate).
- Classifier: **93.5%** cross-game val (2 games; 85.7% on 1) → trajectory toward ~99% with more games.
- Datasets: ~40k crops / ~1,014 YOLO images over 2 games.

→ **Full progress, findings, and roadmap: [`docs/STATUS.md`](docs/STATUS.md).** Design rationale: [`docs/DESIGN.md`](docs/DESIGN.md).

## Run

Use the conda **`auto`** env for tests / numpy+cv2 code; Akagi itself runs in the
**`akagi`** env. (Default PATH python has no numpy.)

```bash
PY=C:/Users/zsx/miniforge3/envs/auto/python.exe

# GT-only capture (passive game, autoplay OFF; client routed through Akagi MITM)
python scripts/record_gt.py --out captures/session1.jsonl

# GT + time-synced screenshots (P2). Runs in Akagi's process, so install the two
# missing screenshot deps into the akagi env first:
#   conda run -n akagi pip install mss opencv-python      # pywin32+numpy already present
python scripts/record_gt.py --screenshots --out captures/session2.jsonl \
       --min-settle 0.30 --max-settle 1.0
#   ↳ for the WEB client, run the game FULLSCREEN (F11) so the 16:9 board fills the
#     grabbed window (no browser tab/URL chrome). Status -> captures/<...>.jsonl.log

# inspect a capture (join frames <-> GT by last_op_step; report settle quality)
$PY scripts/inspect_capture.py captures/session2.jsonl captures/session2/ --step <N>

# tests
PYTHONPATH=. $PY tests/test_tiles.py && PYTHONPATH=. $PY tests/test_replay.py && PYTHONPATH=. $PY tests/test_sync.py
```

## ⚠️ Notes

- Tile taxonomy is **38 classes** (34 tiles + 3 red fives + `back`); ordering is
  fixed by what `tile.model` was trained on — see `tiles.py`. Do not reorder.
- Coordinate baselines differ: **mycv = 1920×1080**, **Akagi/Playwright = 1600×900**.
  Always normalize to 0–1 before converting between them.
- Risk/compliance (time-sync, ban-avoidance, Akagi's AGPLv3 + Commons Clause):
  see `docs/DESIGN.md` §7. Prefer **passive capture** (观战/人工对局) over autoplay.
