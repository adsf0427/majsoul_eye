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

Read `README.md` and **`docs/PIPELINE.md`** first — PIPELINE.md is the authoritative
description of the CURRENT pipeline (data flow, per-stage commands, deprecated-component
list, maintenance rules). `docs/DESIGN.md` is the original approved plan (method table §4,
reuse map §5, risks §7); `docs/STATUS.md` is the running history.

## Environment & commands

All majsoul_eye code, tests, dataset building, and training run in the conda **`auto`** env.
Docs and commands write plain `python` — the user activates the env themselves. Default-PATH
python has NO numpy, so in a shell where `auto` is not activated (e.g. this harness's Bash
tool), substitute `C:/Users/zsx/miniforge3/envs/auto/python.exe` for `python`.
The `akagi` env exists only for the **deprecated** manual `record_gt.py` path.

Imports are top-level `from majsoul_eye import ...`, so **run everything with `PYTHONPATH=.`**
from the repo root.

```bash
# Tests — plain scripts under tests/ (no pytest dependency; also pytest-compatible). One:
PYTHONPATH=. python tests/test_replay.py
# All of them:
for t in tests/test_*.py; do PYTHONPATH=. python "$t" || break; done
```

### Data pipeline (capture → dataset → model) — full detail in docs/PIPELINE.md

**Single capture path (AI autoplay).** `scripts/capture/autoplay_ai.py --live` plays via
Mortal + Playwright and writes the unified `GTRecord` + screenshot index inline under
`captures/raw/ai_session/run_N/gameM/` (self-contained: `gameM.jsonl` GTRecord +
frames/wire/metadata) — no convert step, no
`intermediate/gt` (retired; legacy b64 runs were migrated once — that one-shot migration is
since removed; conversion still lives in `convert_mjcopilot.convert_game`). The old manual path
(`record_gt.py` + Akagi MITM, `captures/raw/manual/`) is **deprecated for new capture** and its
session5/6 data is **not in the current v2 training set** (AI-only baseline). Frame indexes (`frames.jsonl`) store RELATIVE paths; always resolve via
`paths.resolve_frame_path`.

```bash
# 1. Capture (dry-run first, then --live; burner account):
PYTHONPATH=. python scripts/capture/autoplay_ai.py --live --auto-next
# 2. Build a VERSIONED dataset (annotate → per-game crops+yolo → detector split → games.json
#    manifest). Runs immediately (--dry-run to preview); --resume adds only missing games:
PYTHONPATH=. python scripts/data/build_datasets.py v2          # default --sources captures/raw/ai_session
PYTHONPATH=. python scripts/data/build_datasets.py v1 --sources captures/raw/ai_session captures/raw/manual --resume
#    (rebuild_datasets.py was REMOVED 2026-07-05 — superseded by build_datasets.py; current version: datasets/v1)
# 3. Train (GPU, deliberate). --dataset expands a version's games.json; repeat it to mix versions:
PYTHONPATH=. python scripts/train/train_classifier.py --dataset datasets/v1 --val "ai_session_run_8_game1:*" --epochs 20
PYTHONPATH=. python scripts/train/train_detector.py --data datasets/v1/detector/data.yaml
# QA: inspect_capture.py (frame↔GT join), overlay_labels.py (draw labels on a frame),
#     annotate_ai_session.py --qa-classifier (crop-consistency spot check)
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
- **`hud.py`** — the HUD-element detector taxonomy: `HUD_NAMES` (19 classes — 7 center-panel
  fields, 2 top-left counters, 9 semantic action buttons incl. the sanma `btn_babei`, 1 symmetric
  `reach_stick`) + `DET_NAMES = TILE_NAMES + HUD_NAMES` (the 57-class detector head; `btn_babei`
  is APPENDED as id 56 so 56-class weights stay a strict prefix); `OP_TO_BTN`/`buttons_for_ops`
  (liqi op type → button class, incl. 11=babei); `REACH_STICK_SLOTS` (self/right/across/left seat-attribution
  vocabulary for the single symmetric class — spec §10, revised same-day from 4 classes to 1);
  `FIELD_ROT`/`NUMERIC_FIELDS`/`ROUND_CLASSES`/`WIND_CLASSES`/`CTC_CHARSET` (micro-reader
  contracts). Pure data (no cv2/numpy) — every component imports it.
- **`capture/`** — ⚠️ **DEV-ONLY. The shipped recognizer never imports it.**
  - `akagi_tap.py` (**legacy manual path**) monkeypatches `MajsoulBridge.parse_liqi` to tee each
    (raw liqi + derived MJAI) tick to a background JSONL writer. Records both because **MJAI drops
    superset fields** (`leftTileCount`, `moqie`, mid-round `scores`, full `ActionHule`).
  - `roi_diff.py` (ROI-stability confirm — prevents discard-animation occlusion at the source),
    `overlay.py` (`DetectionOverlay` — draws live detector boxes in the browser, `--overlay`),
    `gamemeta.py` (per-game display-language `metadata.json`), `multishot.py` (`MultiShot` —
    extra-shot scheduler for uncertain-timing windows: meld→forced-dahai animation, pending
    action-button offers; purely additive `_dt{ms}.png` frames with `status="extra"` in
    `frames.jsonl`, wired via `autoplay_ai.py --op-delay`/`--multishot-offsets`),
    `bot3p.py` (`make_sanma_bot` — grafts the sibling Akagi `mjai_bot/mortal3p` stack
    (new-arch `default.pth` + its 622-channel `.libriichi` featurizer pyd — NOT the
    775-channel shinkuan `libriichi3p`) onto MJC's `BotMortalLocal` so autoplay
    plays/observes **3-player (sanma)** games too; `--model-3p`/`--akagi`/`--join-mode`.
    Capture-only: annotate/dataset/reconstruct are still 4P-only, see STATUS §1.58).
  - `sync.py` (`FrameSyncer`) — the top correctness risk. Protocol events fire *before* the
    animation renders, so capture is async **debounce-to-quiet**: capture one frame once no
    board event has arrived for `quiet` s (plus optional pixel-stability confirm). Decision logic
    is injected (`grab`/`now`/`sleep`) so it is unit-testable without a client.
  - `schema.py` (`GTRecord`, JSONL I/O), `screen.py` (win32/mss window grab), `gtframes.py`
    (shared `build_seq_state`/`load_frames` for the annotator + dataset builder — Akagi-free).
- **`state/replay.py`** — pure, Akagi-free `Replayer` consuming MJAI events into a full
  seat-absolute `BoardState` (rivers with tsumogiri/riichi/called flags, melds, dora, hero hand,
  concealed counts, scores, `pending_ops`). `check_invariants()` flags desync (>4 of a kind, bad
  hand size) — drop/human-review violating frames. **`state/ops.py`** (`ops_from_record`) is the
  sibling extractor: pending liqi op types offered to the hero, parsed from a `GTRecord`'s
  `raw_liqi.data.data.operation.operationList` — `Replayer.apply_record` sets
  `BoardState.pending_ops` from it (drives HUD button auto-labels via `hud.buttons_for_ops`).
- **`annotate/`** — the **precise** GT-driven annotator; the source `build_dataset.py` now consumes.
  `pipeline.py` = a fullwarp top-down homography + data-calibrated `DISCARD_GRID`/`DISCARD_ROW_OFFSETS`
  + composition-aware melds (`generate_meld_boxes_v2`/`meld_display_cells`) + per-frame mask snap
  (`snap_meld_strip`); GT drives class assignment (not detection). **Sanma (3P) is supported**
  (STATUS §1.59-61): `set_sanma` swaps the geometry constants IN PLACE to the `*_3P` variants
  (driven per frame by `BoardState.sanma`; the 4P seat ring `(hero+rel)%4` holds unchanged —
  chair 3 = the E1 north seat renders empty all game), and `generate_nukidora_boxes`/`NUKI_STRIP_3P`
  emit the nukidora piles as `tile='N'` meld-shaped boxes (calibrated by
  `scripts/annotate/calibrate_nukidora.py`). `frame.py` = `annotate_frame`
  (full per-frame record, original-px quads + fills/flags) plus `iter_tile_boxes`/`AnnBox`/`crop_box`
  (the crop+YOLO seam: quad crops for river/meld, px_box for hand/dora). `seatgt.py` = `seat_gt` +
  `_screen_to_seat`/`SEAT_POS` (the seat mapping, owned here); `cases.py` = the named AB validation seqs
  (`CASES`). (The precise pipeline was moved verbatim out of a former root
  `mahjong_relative_annotation_pipeline.py`, now removed — import `from majsoul_eye.annotate import pipeline as P`.)
  `hud.py` = GT-driven HUD field/button boxes: `hud_field_boxes` (seed ROI + per-frame ink-snap on
  numeric fields — EXCEPT `wall_count`, a fixed box with zero-padded text `余09` and a presence-only
  ink probe), `button_boxes` (op-GT class assignment against `BTN_ZONE` candidates; emitted boxes are
  the fixed 250×96 BANNER click area, not the language-dependent glyph blob; count-mismatch → that
  frame emits NO button labels AND `build_dataset.has_unlabeled_buttons` keeps it out of the detector
  set entirely — a visible-but-unlabeled button is a background negative, STATUS §1.55),
  `reach_stick_boxes` (fill gate applies
  only inside the reach window — settled frames trust GT so dark skinned sticks aren't dropped;
  STATUS §1.47). `annotate_frame` calls all three into `rec["hud_boxes"]`.
  `btnbg.py` = per-game BTN_ZONE background median (`game_btn_background`, from GT-no-button frames);
  `annotate_frame(..., btn_bg=)` threads it into `hud.locate_button_plates`, which segments the button
  PLATE as an overlay difference. This REPLACED the `gray>=140` glyph gate, which was skin-dependent
  and silently dropped 46% of rendered buttons (§1.55). Both pipeline entry points build it per game;
  omitting it falls back to the legacy gate (inspect tools only).
  `backs.py` = EXPERIMENTAL opt-in (`annotate_frame(..., backs=True)` / `build_datasets.py --backs`,
  default OFF, not in v1/v2): opponent concealed-hand tile-back boxes from GT counts + manually
  clicked fullwarp slot templates (手摸切 groundwork). Holding seats ARE labeled (static n-1 row +
  drawn slot); the only pixel gate left is `sorting_suspect` Condition A (bare-slot reflow, 0.4-3%),
  which build_dataset still turns into a whole-frame drop. The per-box fill gate and the
  drawn-slot Condition B were removed 2026-07-10 (skin-dependent false drops; `fill` is QA-only).
- **`label/`** — **legacy** NormBox annotator, now just `autolabel.py` (`label_frame`): supplies the
  hero hand + dora boxes only (`annotate_frame` calls it for those zones; `DEFAULT_ZONES = {hand}`).
  The old `river.py`/`meld.py` + `coords.RIVER_QUADS`/`MELD_STRIPS` (equal-subdivision RiverGrid) were
  **removed** — superseded by `annotate/` (see docs/STATUS.md §1.13).
- **`recognize/`** — the SHIPPED product: `classifier.py` (`TileNet` small CNN, 64px, 38-class +
  `TileClassifier`) and `detector.py` (`TileDetector`, YOLO HBB/OBB, lazy-loads ultralytics). The
  detector head is now **57-class** (`hud.DET_NAMES` = 38 tiles + 18 HUD/button classes incl.
  the sanma `btn_babei` + 1 `reach_stick`); `Detection.name` is valid for all 57 ids while
  `Detection.tile` is `None` for HUD-class ids (38-56) — old 38/56-class weights still load
  fine (strict id prefixes). `hudreader.py`
  (`HudReader` — `DigitCTC` segmentation-free CRNN-CTC for numeric fields + `round_label`/
  `seat_wind_self` classifier heads reusing `TileNet`) and `hudstate.py` (`assemble_hud` —
  detector HUD boxes + `HudReader` outputs → one structured HUD state dict) are the HUD reading
  half; **not yet trained** — see `docs/STATUS.md` §1.31. Production weights: `tile_classifier.pt`
  (tracked) + `tile_detector.pt` (local, still 38-class); training bases/variants live under
  `weights/`.

## Critical invariants & gotchas

- **Pipeline-impact discipline**: before committing any change that touches capture, annotate,
  dataset building, or training, ask (1) does it stale the derived data under `out/`/`datasets/`
  (→ note it, or rebuild via `build_datasets.py <name> --force`), and (2) does it change a pipeline
  input/output/step/default (→ **update `docs/PIPELINE.md`** and add a STATUS.md entry).
  New scripts must be classified as either a pipeline stage or a one-shot tool (PIPELINE.md §4).
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
  `raw/ai_session`). The AI GT jsonl is NESTED inside its frames dir
  (`run_N/gameM/gameM.jsonl`); the old sibling shape (`run_N/gameM.jsonl`, still used by
  `manual/sessionN.jsonl`) is resolved as legacy. Don't hardcode `captures/...` paths or
  re-derive that coupling; use `paths.frames_dir_for` / `paths.capture_for_frames_dir` /
  `paths.ai_captures()` (`converted_gt_captures()` is kept as a thin alias for old callers)
  and resolve every `frames.jsonl` `file` (RELATIVE now) through `paths.resolve_frame_path`
  (accepts legacy absolute too). Past layout reorganizations were one-shot idempotent
  dry-run-default scripts (`migrate_captures_layout.py` 2026-07-02, `migrate_gt_into_gamedir.py`
  2026-07-05, the latter also rewriting `datasets/*/games.json` capture paths); both completed
  and **removed 2026-07-06**.
- **Train/val split by kyoku, never by frame** — the same physical discard appears in many frames
  of one kyoku; a frame split leaks it and inflates accuracy.
- **Frame-drop predicates for GT-leads-pixels windows** (`majsoul_eye/state/replay.py`):
  `is_deal_window` (rivers all empty — deal animation still sorting the hand) and
  `is_call_window` (last event is chi/pon/daiminkan/ankan/kakan/nukidora — the meld's forced
  follow-up dahai animation is still in flight) both drop the WHOLE frame in `build_dataset`/
  `annotate_ai_session`, not just flag it unreliable, because GT has already advanced past what
  the pixels show (~4.2% of frames on the `is_call_window` measurement). `is_score_anim_window`
  (reach/reach_accepted) is narrower: it only marks HUD boxes unreliable, since only the HUD
  numbers animate, not the tiles.
- **Recording must never break the bridge or the TUI.** `parse_liqi` runs under Akagi's lock on
  the MITM thread; recorder code swallows all exceptions, writes from a background thread, and
  routes status to a sidecar `.log` (printing to stdout corrupts Akagi's Textual TUI).
- **`recognize/` must stay Akagi-free** — it is the shipped product; keep the Akagi-coupled
  `capture/` import boundary intact.
- **Ban-avoidance**: AI autoplay capture runs on **burner accounts** (active ranked play);
  don't run 24/7. Do not extract/redistribute Majsoul sprites or screenshots.
