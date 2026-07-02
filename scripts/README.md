# `scripts/` — pipeline tools, grouped by stage

Every script is an entrypoint. Run from the repo root with the `auto` conda env and
`PYTHONPATH=.` (imports are top-level `from majsoul_eye import ...`):

```powershell
$PY = "C:/Users/zsx/miniforge3/envs/auto/python.exe"; $env:PYTHONPATH = "."
& $PY scripts/<stage>/<name>.py --help
```

Pipeline: **capture → data → annotate → train → (inspect at every step).**

Legend: 🔴 dev-only / coupled to a sibling repo (`recognize/` never imports these) ·
⚙️ one-off / infrastructure · 🔁 recurring pipeline step.

---

## `capture/` — get raw (screenshots + protocol GT)

| script | role | out |
|---|---|---|
| `record_gt.py` 🔴🔁 | inject GT tap, launch **Akagi**; passive capture (观战/人工, autoplay OFF). Runs in the **`akagi`** env. | `captures/raw/manual/sessionN.jsonl` + frames |
| `autoplay_ai.py` 🔴🔁 | **Mortal-AI autoplay + capture** in one Playwright Chromium (defaults `--dry-run`). A real AI melds/riichis → covers the hard zones. | `captures/raw/ai_session/run_N/` (liqi wire + PNGs) |

## `data/` — raw → our GT format, frame fixups, layout infra

| script | role | out |
|---|---|---|
| `convert_mjcopilot.py` 🔴🔁 | MahjongCopilot raw liqi wire + PNGs → our `GTRecord` JSONL (its own `LiqiProto`→`GameState`; deep-copies events). | `captures/intermediate/gt/ai_run_*.jsonl` |
| `ingest_run.py` 🔁 | one-shot orchestrator: discover games → `convert_mjcopilot` → `build_dataset` → optional retrain (**subprocess-calls `data/convert_mjcopilot`, `train/build_dataset`, `train/train_classifier`**). | datasets + model |
| `crop_game.py` ⚙️🔁 | crop the 16:9 canvas out of non-fullscreen captures (browser chrome / pillarbox). | `captures/intermediate/derived/<name>_16x9/` |
| `deletterbox_frames.py` ⚙️🔁 | remove black bars from non-16:9 windows, resize back to 1920×1080 (fixes the **data**, not the pipeline). | `captures/intermediate/derived/<name>_fixed/` |
| `migrate_captures_layout.py` ⚙️ (once) | migrate `captures/` into the role-based layout + rewrite `frames.jsonl` to relative paths (same-volume rename, idempotent). | reorganized `captures/` |

## `annotate/` — GT + geometry → labeled boxes (+ calibration)

| script | role | out |
|---|---|---|
| `annotate_ai_session.py` 🔁 | full-frame **precise annotator** (river/meld/hand/dora boxes + fills/flags), `--qa-classifier` agreement. Calls `majsoul_eye.annotate`. | `out/ai_session_annotations/` |
| `build_case_annotations.py` 🔁 | corrected relative-seat annotations for the 11 AB validation cases (**imports `annotate/spike_topdown`**). | `out/mahjong_AB_relative_data_with_reliability.json` |
| `spike_topdown.py` ⚙️🔁 | 1.9b top-down `H_table` rectify spike; **also the shared hub** (`CASES`/`load_pair`/`_screen_to_seat`/`SEAT_POS`). | overlays |
| `calibrate_annotation_model.py` 🔁 | measure & `--refit` the fullwarp geometry (`DISCARD_GRID`/`ROW_OFFSETS`/`MELD_STRIP2`) against many real frames — the tool that produced `annotate/pipeline.py`'s constants. | suggested constants |

## `train/` — labeled → training data → model

| script | role | out |
|---|---|---|
| `build_dataset.py` 🔁 | precise path: resize→1920×1080 → `annotate_frame` → `iter_tile_boxes` → **quad crops + YOLO** from ONE calibration (`reliable`-gated, sideways excluded from crops). | `datasets/<name>/{crops,yolo}/` |
| `train_classifier.py` 🔁 | train the 38-class `TileNet`, **kyoku-level split** (never by frame). | `majsoul_eye/recognize/tile_classifier.pt` |

## `inspect/` — QA / eval / diagnostics

| script | role |
|---|---|
| `inspect_capture.py` 🔁 | join a GT capture with its frames, report sync quality (`--step N` shows the reconstructed board beside the screenshot). |
| `overlay_labels.py` ⚙️ | draw seeded `coords.py` ROIs on a frame to calibrate by eye. |
| `mycv_baseline.py` 🔴 | measure **mycv**'s real per-zone accuracy vs Akagi GT (the baseline this project rewrites). |
| `visualize_failures.py` 🔁 | run a model over labeled crops, montage misclassifications by confusion pair. |

---

## Dependencies (mind these when moving files)

The import/exec graph is intentionally shallow:

- **Python import:** `annotate/build_case_annotations.py` → `annotate/spike_topdown.py` (only inter-script import).
- **Subprocess (by path):** `data/ingest_run.py` shells out to `data/convert_mjcopilot.py`,
  `train/build_dataset.py`, `train/train_classifier.py` — move those and update the paths in `ingest_run.py`.
- Everything else imports only the **`majsoul_eye` package** (never another script). Shared annotation
  logic lives in `majsoul_eye/annotate/` + `majsoul_eye/capture/gtframes.py`, not in these scripts.
