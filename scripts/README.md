# `scripts/` — pipeline tools, grouped by stage

Per-file index. **The authoritative pipeline description is [`docs/PIPELINE.md`](../docs/PIPELINE.md)**
(data flow §0, per-stage notes §2, SOP §3, deprecated/removed components §4). This file just
says what each script is; PIPELINE.md says how the stages fit together.

Every script is an entrypoint. Run from the repo root in the `auto` conda env with
`PYTHONPATH=.` (imports are top-level `from majsoul_eye import ...`):

```bash
export PYTHONPATH=.
python scripts/<stage>/<name>.py --help          # bash launcher: bash scripts/train/launch_*.sh --help
```

Pipeline: **capture → data → annotate → build/train → (inspect at every step).**

Legend: ★ current pipeline step · 🔁 recurring tool · ⚙️ one-off / infrastructure ·
🔴 dev-only / coupled to a sibling repo (`recognize/` never imports these).

---

## `capture/` — get raw (screenshots + protocol GT)

| script | role | out |
|---|---|---|
| `autoplay_ai.py` ★🔴 | **the single main capture path**: Mortal-AI autoplay + WS tap + on-quiet screenshots in one Playwright Chromium (defaults to OBSERVE; `--live` to play, `--auto-next` to loop, `--overlay` to draw live detector boxes, `--skins` for appearance diversity). Writes `GTRecord` inline — no convert step. | `captures/raw/ai_session/run_N/gameM/{gameM.jsonl,liqi.jsonl,frames.jsonl,frames/,metadata.json}` |
| `record_gt.py` 🔴 | **legacy/deprecated** manual F11 + **Akagi** MITM capture (runs in the `akagi` env). Kept only to reproduce the archived session5/6 data; not used for new capture. | `captures/raw/manual/sessionN.jsonl` + frames |
| `build_skin_config.py` 🔴 | build the `--skins` randomization config (skin/tile-back/table ids) for autoplay appearance diversity. | skins config |
| `patch_majsoulmax.py` 🔴 | patch the MajsoulMax MITM used by `autoplay_ai --skins` to apply skins/backs/tables. | patched MITM |

## `data/` — versioned dataset build, frame fixups, shared conversion

| script | role | out |
|---|---|---|
| `build_datasets.py` ★ | **the one-command versioned builder**: orchestrates annotate → `build_dataset` (crops+YOLO) → `build_detector_dataset` (split) → `games.json`. `--sources` roots, `--resume` incremental, `--force` rebuild, `--hbb --obb` dual format, `--dry-run`. | `datasets/<name>/{annotations/, <game>/{crops,yolo}, detector/(+detector_obb/), games.json}` |
| `regen_detector_dataset.sh` 🔁 | GPU-server-side rebuild of the **flat** `datasets/detector[_obb]` layout (per-game parallel; `--obb/--obb-only/--skip-annotate`). Reuses `build_datasets.discover_games`. Overlaps `build_datasets.py --hbb --obb`; kept for the tar-and-go flat-layout flow. | `datasets/detector[_obb]/` |
| `convert_mjcopilot.py` 🔴 | shared MahjongCopilot raw-liqi-wire → `GTRecord` conversion lib (`convert_game`). New captures write `GTRecord` inline and never call this; it survives as a still-runnable CLI for any old raw-wire capture. | `GTRecord` jsonl |
| `crop_game.py` ⚙️🔁 | crop the 16:9 canvas out of non-fullscreen captures (browser chrome / pillarbox). | `captures/intermediate/derived/<name>_16x9/` |
| `deletterbox_frames.py` ⚙️🔁 | remove black bars from non-16:9 windows, resize back to 1920×1080. `--inplace` rewrites raw frames (run_5 fixed 2026-07-05) or `--out` writes a derived copy. New fullscreen 1080p capture doesn't need it. | fixed frames |
| `purge_deal_frames.py` / `apply_deal_purge.py` / `purge_occlusion_frames.py` ⚙️ | one-shot cleanups of **old** datasets (deal-window / occlusion frames). Superseded by capture-time avoidance + build-time drop; not needed after a full rebuild. | pruned dataset |

> One-shot migrations `migrate_captures_layout.py` / `migrate_ai_to_gtrecord.py` /
> `migrate_gt_into_gamedir.py` and the skins backfill `backfill_skin_meta.py` were **removed
> 2026-07-06** (their jobs are complete — see PIPELINE.md §4). The legacy orchestrator
> `ingest_run.py` was removed too (superseded by `build_datasets.py`).

## `annotate/` — GT + geometry → labeled boxes (+ calibration)

| script | role | out |
|---|---|---|
| `annotate_ai_session.py` ★ | full-frame **precise annotator** (river/meld/hand/dora boxes + fills/flags), `--qa-classifier` agreement spot-check. Defaults to all `paths.ai_captures()`. Calls `majsoul_eye.annotate`. | `out/ai_session_annotations/` |
| `build_case_annotations.py` 🔁 | corrected relative-seat annotations for the AB validation cases (imports only the `majsoul_eye` package). | `out/mahjong_AB_relative_data_with_reliability.json` |
| `calibrate_annotation_model.py` 🔁 | measure & `--refit` the fullwarp geometry (`DISCARD_GRID`/`ROW_OFFSETS`/`MELD_STRIP2`) against real frames — the tool that produced `annotate/pipeline.py`'s constants. | suggested constants |
| `spike_topdown.py` ⚙️ | **ARCHIVED** top-down rectify spike; self-contained viz, no longer load-bearing (superseded by `annotate/`). | overlays |

## `train/` — labeled → training data → model

| script | role | out |
|---|---|---|
| `build_dataset.py` 🔁 | precise per-game primitive: resize→1920×1080 → `annotate_frame` → `iter_tile_boxes` → **quad crops + YOLO** from one calibration (`--from-annotations` reuse; `--obb` 8-pt labels; `--reuse-images` OBB-labels-only, no re-encode). | `datasets/<name>/<game>/{crops,yolo}/` |
| `build_detector_dataset.py` 🔁 | assemble the detector split (train/val txt + `data.yaml`) by kyoku/game; repeatable `--dataset` merges multiple versions. | `<out>/detector/{train.txt,val.txt,data.yaml}` |
| `train_classifier.py` 🔁 | train the 38-class `TileNet`, **kyoku-level split** (never by frame). `--dataset` expands a version's `games.json` (crops). | `majsoul_eye/recognize/tile_classifier.pt` |
| `train_detector.py` 🔁 | train the YOLO detector (HBB or OBB) from a `data.yaml`; explicit augmentation CLI. | `<out>.pt` |
| `launch_classifier.sh` ★ | **classifier launcher**: single-card wrapper over `train_classifier.py`. `--dataset v2 --gpu ID`; auto-reads `games.json`'s `val` list so the holdout matches the detector; `--dry-run` previews. | `recognize/tile_classifier.pt` |
| `launch_detector.sh` ★ | **detector launcher**: multi-GPU DDP wrapper over `train_detector.py`. `{hbb\|obb} --dataset v2 --gpus IDS`; picks split/seed/output/run-dir per mode. Each run keeps a versioned `weights/detector/tile_detector_<mode>_<name>.pt`; OBB also refreshes the shipped default. | `weights/detector/tile_detector_{hbb,obb}_<ts>.pt`; OBB **also** → `recognize/tile_detector.pt` (runtime default) |

## `inspect/` — QA / eval / diagnostics

| script | role |
|---|---|
| `inspect_capture.py` 🔁 | join a GT capture with its frames, report sync quality (`--step N` shows the reconstructed board beside the screenshot). |
| `overlay_labels.py` ⚙️ | draw labels / seeded `coords.py` ROIs on a frame to calibrate by eye. |
| `visualize_failures.py` 🔁 | run a model over labeled crops, montage misclassifications by confusion pair. |
| `mycv_baseline.py` 🔴 | measure **mycv**'s real per-zone accuracy vs Akagi GT (the baseline this project rewrites). |
| `count_dora_glow.py` 🔁 | one-shot diagnostic: per-class glowing/total instance coverage, to decide whether dora-glow needs dedicated augmentation (Akagi-free, stdout only). |
| `calibrate_occlusion_gate.py` ⚙️ | tune the ROI-stability / occlusion-gate thresholds against real frames. |
| `cvat_export.py` / `cvat_import.py` ⚙️ | round-trip a game's labels to/from CVAT for manual box correction. |
| `fiftyone_view.py` ⚙️ | FiftyOne GUI review of the detector dataset (see `docs/dataset_review.md`). |

---

## `eval/` — board-reconstruction acceptance (QA, not pipeline stages)

| script | role |
|---|---|
| `eval_reconstruction.py` 🔁 | 3-layer GT eval: oracle (GT→ObservedState→reconstruct round-trip) / assemble (real frame→detector→assemble vs GT, per-zone errors + `rejected_reasons`) / engine (true vs reconstructed mjai prefix → an mjai bot, decision agreement). |
| `mortal_stdin.py` 🔴 | mjai stdin/stdout wrapper around `../auto/mycv`'s Mortal for the engine layer (`--engine-cmd "python scripts/eval/mortal_stdin.py {seat}"`). |

---

## `recognize/` — runtime-chain CLI entrypoints (Akagi-free)

| script | role |
|---|---|
| `recognize_frame.py` 🔁 | screenshot(s) → `TileDetector`+`assemble`+`reconstruct` → JSON lines (ObservedState + legal mjai sequence; rejected frames report violations). `--weights` defaults to the newest `weights/detector/tile_detector_obb_*.pt`. |

---

## Dependencies (mind these when moving files)

The import/exec graph is intentionally shallow:

- **Python import:** every script imports only the **`majsoul_eye` package** — there are no
  script→script Python imports.
- **Subprocess (by path):** `data/build_datasets.py` shells out to `annotate/annotate_ai_session.py`,
  `train/build_dataset.py`, `train/build_detector_dataset.py`; `train/launch_*.sh` exec
  `train/train_{classifier,detector}.py`. Move any of those and update the caller's path.
- Shared annotation logic lives in `majsoul_eye/annotate/` + `majsoul_eye/capture/gtframes.py`,
  not in these scripts.
