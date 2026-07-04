# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`majsoul_eye` is an image recognizer for **Mahjong Soul (雀魂)** game state (场况) from
screenshots. It is a clean rewrite that **reuses two sibling repos** (not vendored — referenced):

- `../auto/mycv` — an existing working *pure-vision* Majsoul bot. Source of the baseline:
  the `tile.model` classifier, measured pixel coordinates, 4-seat perspective math,
  contour-based 河/副露 detection, and ~707 debug frames.
- `../Akagi` — MITM that parses the `liqi` protobuf into full ground truth. Here it is a
  **training-time oracle** (free, accurate labels), **never a runtime dependency**.

The whole design rests on: **`Akagi GT = WHAT` (which tile, who discarded) + `geometry = WHERE`
(pixel box) → auto-generated labels, zero hand-drawing.**

Read `README.md` and `docs/DESIGN.md` first — `DESIGN.md` is the authoritative approved plan
(element-by-element method table in §4, reuse map in §5, roadmap P0–P6 in §6, risks in §7).

## Environment & commands

Two conda envs are involved (default-PATH python has no numpy):

- **`auto`** — for all majsoul_eye code, tests, dataset building, training.
  `PY=C:/Users/zsx/miniforge3/envs/auto/python.exe`
- **`akagi`** — only for `record_gt.py`, which runs *inside* Akagi's process. The screenshot
  path needs deps Akagi lacks: `conda run -n akagi pip install mss opencv-python`.

Imports are top-level `from majsoul_eye import ...`, so **run everything with `PYTHONPATH=.`**
from the repo root.

```bash
# Tests — plain scripts (no pytest dependency; also pytest-compatible). Run one:
PYTHONPATH=. $PY tests/test_replay.py
# All of them:
PYTHONPATH=. $PY tests/test_tiles.py && PYTHONPATH=. $PY tests/test_replay.py && \
PYTHONPATH=. $PY tests/test_sync.py && PYTHONPATH=. $PY tests/test_river.py && \
PYTHONPATH=. $PY tests/test_meld.py && PYTHONPATH=. $PY tests/test_label.py && \
PYTHONPATH=. $PY tests/test_classifier.py
```

### Data pipeline (capture → dataset → model)

`captures/` uses a role-based layout (single source of truth: `majsoul_eye/paths.py`):
`raw/ai_session/` (MahjongCopilot autoplay) + `raw/manual/` (record_gt) both write our
`GTRecord` format directly — `autoplay_ai.py` writes the GTRecord + screenshot index inline
under `raw/ai_session` (same as manual); there is no separate convert step or
`intermediate/gt` anymore (retired; legacy b64 runs were migrated in place by
`scripts/data/migrate_ai_to_gtrecord.py`). `intermediate/derived/` (cropped / de-letterboxed)
+ `legacy/` (archived dupes) are unchanged. Frame indexes (`frames.jsonl`) store RELATIVE
paths; always resolve via `paths.resolve_frame_path`.

```bash
# 1. Record GT + time-synced screenshots (runs Akagi; akagi env; autoplay OFF, passive only).
#    WEB client must be FULLSCREEN (F11). Defaults to captures/raw/manual/. Status → <out>.jsonl.log
conda run -n akagi python scripts/capture/record_gt.py --screenshots --quiet 0.30 --settle-cap 2.0
# 2. Inspect sync quality (offline, no client):
PYTHONPATH=. $PY scripts/inspect/inspect_capture.py captures/raw/manual/sessionN.jsonl captures/raw/manual/sessionN/ --step <N>
# 3. (only if not captured fullscreen) crop the 16:9 canvas out of a letterboxed session:
$PY scripts/data/crop_game.py captures/raw/manual/sessionN captures/intermediate/derived/sessionN_16x9 --size 3840x2160
# 4. Build auto-labeled dataset (replay → autolabel → crops/ + yolo/):
PYTHONPATH=. $PY scripts/train/build_dataset.py captures/raw/manual/sessionN.jsonl captures/raw/manual/sessionN/ \
       --out datasets/sessionN --locator fullscreen --drop-violations
# 5. Train the 38-class tile classifier (KYOKU-level split — never split by frame):
PYTHONPATH=. $PY scripts/train/train_classifier.py \
       --data sN=datasets/sessionN/crops:captures/raw/manual/sessionN.jsonl --val sN:E3.0,S2.0 --epochs 20
# AI (MahjongCopilot) path: scripts/capture/autoplay_ai.py writes the GTRecord + screenshot
#   index directly under captures/raw/ai_session/run_N/ (same format as manual; no convert
#   step). Then annotate + build: scripts/annotate/annotate_ai_session.py (defaults to all
#   paths.ai_captures(), i.e. captures/raw/ai_session/**/*.jsonl) -> scripts/train/build_dataset.py.
# overlay_labels.py draws auto-labels onto a frame for visual coordinate calibration.
```

## Architecture

Pipeline: **GT capture → state replay → auto-label → train classifier/detector.** The runtime
recognizer (`recognize/`) is a separate, Akagi-free product. Module map:

- **`tiles.py`** — the unified **38-class taxonomy** (single source of truth) + MJAI interop.
  Shared by every component (classifier, detector, labels, state).
- **`coords.py`** — normalized ROI model. Every box is normalized 0–1 against a **canonical
  16:9 board**, so it applies at any resolution. Holds easy-zone boxes (`REGIONS`), the
  parametric `HandModel`, `DORA_STRIP`, and coarse per-quadrant `RIVER_ZONES`. (The precise
  per-seat 河/副露 geometry lives in `annotate/`, not here — the old `RIVER_QUADS`/`MELD_STRIPS`
  were removed with the `label/` river+meld modules; see §1.13.)
- **`normalize.py`** — front-end that maps an arbitrary screenshot onto the canonical frame via
  a `BoardRegion` (`locate_fullscreen` / `locate_letterbox`; `AnchorLocator` is a TODO stub).
  This is what lets fixed-slot logic survive other resolutions.
- **`capture/`** — ⚠️ **DEV-ONLY, Akagi-coupled. The shipped recognizer never imports it.**
  - `akagi_tap.py` monkeypatches `MajsoulBridge.parse_liqi` to tee each (raw liqi + derived
    MJAI) tick to a background JSONL writer. Records both because **MJAI drops superset fields**
    (`leftTileCount`, `moqie`, mid-round `scores`, full `ActionHule`) — read those from raw liqi.
  - `sync.py` (`FrameSyncer`) — the top correctness risk. Protocol events fire *before* the
    animation renders, so capture is async **debounce-to-quiet**: capture one frame once no
    board event has arrived for `quiet` s (plus optional pixel-stability confirm). Decision logic
    is injected (`grab`/`now`/`sleep`) so it is unit-testable without a client.
  - `schema.py` (`GTRecord`, JSONL I/O), `screen.py` (win32/mss window grab), `gtframes.py`
    (shared `build_seq_state`/`load_frames` for the annotator + dataset builder — Akagi-free).
- **`state/replay.py`** — pure, Akagi-free `Replayer` consuming MJAI events into a full
  seat-absolute `BoardState` (rivers with tsumogiri/riichi/called flags, melds, dora, hero hand,
  concealed counts, scores). `check_invariants()` flags desync (>4 of a kind, bad hand size) —
  drop/human-review violating frames.
- **`annotate/`** — the **precise** GT-driven annotator; the source `build_dataset.py` now consumes.
  `pipeline.py` = a fullwarp top-down homography + data-calibrated `DISCARD_GRID`/`DISCARD_ROW_OFFSETS`
  + composition-aware melds (`generate_meld_boxes_v2`/`meld_display_cells`) + per-frame mask snap
  (`snap_meld_strip`); GT drives class assignment (not detection). `frame.py` = `annotate_frame`
  (full per-frame record, original-px quads + fills/flags) plus `iter_tile_boxes`/`AnnBox`/`crop_box`
  (the crop+YOLO seam: quad crops for river/meld, px_box for hand/dora). `seatgt.py` = `seat_gt` +
  `_screen_to_seat`/`SEAT_POS` (the seat mapping, owned here); `cases.py` = the named AB validation seqs
  (`CASES`). (The precise pipeline was moved verbatim out of a former root
  `mahjong_relative_annotation_pipeline.py`, now removed — import `from majsoul_eye.annotate import pipeline as P`.)
- **`label/`** — **legacy** NormBox annotator, now just `autolabel.py` (`label_frame`): supplies the
  hero hand + dora boxes only (`annotate_frame` calls it for those zones; `DEFAULT_ZONES = {hand}`).
  The old `river.py`/`meld.py` + `coords.RIVER_QUADS`/`MELD_STRIPS` (equal-subdivision RiverGrid) were
  **removed** — superseded by `annotate/` (see docs/STATUS.md §1.13).
- **`recognize/classifier.py`** — `TileNet` (small CNN, 64px input, 38-class) + `TileClassifier`
  inference wrapper. The clean rewrite of mycv's classifier.

## Critical invariants & gotchas

- **38-class order is frozen** by what `tile.model` was trained on (m, p, s, honors, red5(m,p,s),
  back). A *dead* commented-out s/p/m ordering exists in mycv — ignore it. **Do not reorder
  without retraining.** See the header of `tiles.py`.
- **Coordinate baselines differ**: mycv = **1920×1080**, Akagi/Playwright = **1600×900**. They
  are not interchangeable — always go through normalized 0–1.
- **3D-table elements scale, 2D HUD does not.** Hand and 河/副露 (perspective table) scale
  linearly across resolutions and are calibrated; scores/dora/round-meta live in the 2D HUD and
  are **resolution-dependent** (need anchor-normalization). Hence `DEFAULT_ZONES = {hand, river}`;
  HUD zones are opt-in and are exact GT anyway. Coord seeds are marked `# CALIBRATE`.
- **Sync/dataset key is the global record `seq`, NOT `last_op_step`** — `last_op_step` resets
  every kyoku, so frame filenames would collide and later rounds overwrite earlier ones.
- **`captures/` layout is defined once in `majsoul_eye/paths.py`** — `raw/{ai_session,manual}`,
  `intermediate/derived`, `legacy/` (`intermediate/gt` is retired — AI GTRecords now live under
  `raw/ai_session`). Don't hardcode `captures/...` paths or re-derive the frames-dir stem rule;
  use `paths.frames_dir_for` / `paths.ai_captures()` (`converted_gt_captures()` is kept as a thin
  alias for old callers) and resolve every `frames.jsonl` `file` (RELATIVE now) through
  `paths.resolve_frame_path` (accepts legacy absolute too). To reorganize again,
  `scripts/data/migrate_captures_layout.py` moves dirs + rewrites indexes idempotently
  (dry-run default).
- **Train/val split by kyoku, never by frame** — the same physical discard appears in many frames
  of one kyoku; a frame split leaks it and inflates accuracy.
- **Recording must never break the bridge or the TUI.** `parse_liqi` runs under Akagi's lock on
  the MITM thread; recorder code swallows all exceptions, writes from a background thread, and
  routes status to a sidecar `.log` (printing to stdout corrupts Akagi's Textual TUI).
- **`recognize/` must stay Akagi-free** — it is the shipped product; keep the Akagi-coupled
  `capture/` import boundary intact.
- **Compliance**: capture **passively** (观战/人工对局), autoplay OFF — ban-avoidance. Akagi is
  **AGPLv3 + Commons Clause**; reusing its bridge likely makes derivatives copyleft and
  non-commercial (DESIGN.md §7). Do not extract/redistribute Majsoul sprites or screenshots.
