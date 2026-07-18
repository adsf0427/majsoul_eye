# majsoul_eye

Robust image recognition of **Mahjong Soul (雀魂)** game state (场况) from screen
images — usable by a protocol-independent vision bot, a HUD overlay, or for
recognizing arbitrary external screenshots.

> **当前管线的唯一权威文档：[`docs/PIPELINE.md`](docs/PIPELINE.md)**（数据流、各阶段命令、
> 过时组件清单、维护规约）。Full design & rationale: [`docs/DESIGN.md`](docs/DESIGN.md)。
> 进度史与实测结论: [`docs/STATUS.md`](docs/STATUS.md)。

## Why this exists (and what it reuses)

- **`../auto/mycv`** is an existing, working *pure-vision* Mahjong Soul bot. It
  already recognizes the full board and solves the 4-seat perspective problem.
  majsoul_eye is a **clean rewrite that reuses mycv's assets** (classifier
  baseline, coordinate knowledge, seat-rotation math) as bootstrap.
- **MahjongCopilot / Akagi** provide the full game state from the `liqi`
  protobuf. Here they are **training-time oracles** that produce free, accurate
  labels — never a runtime dependency. (When you have the protocol you don't
  need vision; vision's whole value is the feed-less case.)

## Labeling, in one line

`协议 GT = WHAT` (which tile, who discarded) + `标定几何 = WHERE` (pixel box) →
auto-generated classifier crops + YOLO labels, **zero hand-drawing**.

## The pipeline (one line each; details in [`docs/PIPELINE.md`](docs/PIPELINE.md))

1. **采集（唯一主路径）** `scripts/capture/autoplay_ai.py --live` — Playwright WS tap +
   Mortal 决策自动对局 + 事件安静截图，**实时内联写统一 `GTRecord`** 到
   `captures/raw/ai_session/run_N/`（无任何转换步骤）。
2. **标注** `scripts/annotate/annotate_ai_session.py` — 精确 fullwarp 几何 + GT 赋类 →
   `out/ai_session_annotations/`。
3. **构建（一条命令，版本化）** **`scripts/data/build_datasets.py <name>`** —— 编排 标注→建库→装配，
   产出自包含 `datasets/<name>/{annotations/, <game>/{crops,yolo}, detector/, games.json}`；
   `--sources` 指定多个采集根、`--resume` 增量并入新局、`--force` 清空重建。现役版本
   `datasets/v2`（28 局纯 AI：18 `ai_session` + 10 换肤 `ai_session2`）。
4. **训练** `train_classifier.py --dataset datasets/v2 [--dataset …]` /
   `train_detector.py --data datasets/<name>/detector/data.yaml`（GPU，手动触发；多版本可混用）。
   多卡服务器用启动器：`launch_classifier.sh` / `launch_detector.sh {hbb|obb}`（见下方「训练」）。

> ⚠️ **`record_gt.py`（Akagi MITM 手动 F11 采集）已列为过时** —— 新数据一律走
> autoplay_ai。现役 `datasets/v2` 为**纯 AI 采集基线**（`ai_session` + 换肤 `ai_session2`）；
> 早期 session5/6 手动局不在当前训练集内。

## Layout

```text
majsoul_eye/
  tiles.py          # unified 38-class taxonomy + MJAI interop (single source of truth)
  coords.py         # normalized ROI model (hand/dora slots, coarse river zones for overlays)
  normalize.py      # board locators: fullscreen / letterbox / anchor(TODO) -> canonical 16:9
  paths.py          # captures/ layout single source of truth (ai_captures / resolve_frame_path)
  capture/          # ⚠️ DEV-ONLY capture stack: schema(GTRecord) / sync / screen / gtframes /
                    #   roi_diff(遮挡防护) / overlay(浏览器检测框) / gamemeta(语言元数据);
                    #   akagi_tap = legacy manual path
  state/replay.py   # pure replayer: MJAI -> 4-player BoardState + invariants
                    #   (+ is_deal_window 发牌窗 / drawn_tile 摸牌槽)
  annotate/         # PRECISE fullwarp annotator: pipeline / frame / seatgt / cases / consistency
  label/autolabel.py# hero hand + dora boxes for annotate_frame (old river/meld geometry removed)
  recognize/        # SHIPPED, Akagi-free: classifier.py(TileNet) + detector.py(YOLO, lazy) +
                    #   tile_classifier.pt(正式) + tile_detector.pt(本地)
  baselines/        # real-mycv engine adapter + bag-matching scorer
scripts/
  capture/autoplay_ai.py         # ★ 唯一主采集（--live --auto-next [--overlay]）
  capture/record_gt.py           # 过时：手动 F11 + Akagi（存档保留）
  annotate/annotate_ai_session.py# 全帧标注器（默认全部 paths.ai_captures()）
  train/build_dataset.py         # crops + YOLO（--from-annotations 复用标注; --obb 8点标签;
                                 #   --reuse-images 仅写OBB标签、复用HBB帧不重编码）
  train/build_detector_dataset.py / train_classifier.py   # 均支持 --dataset 多版本清单展开
  train/train_detector.py
  train/launch_detector.sh       # ★ 起一次检测器训练：hbb|obb 多卡 DDP 包装 train_detector.py
  train/launch_classifier.sh     # ★ 起一次分类器训练：单卡包装 train_classifier.py（自动读 games.json val）
  data/build_datasets.py         # ★ 版本化构建 datasets/<name>/（标注→建库→装配 + games.json）
  data/regen_detector_dataset.sh # GPU 服务器侧重建检测集（分局并行, --obb/--obb-only/--skip-annotate）
  data/purge_*.py, crop_game.py, deletterbox_frames.py, convert_mjcopilot.py  # 一次性/遗留工具（非管线环节，见 PIPELINE §4）
  annotate/{build_case_annotations,calibrate_annotation_model,spike_topdown}.py  # AB case/标定/归档 spike
  inspect/…                      # QA & debug（下方 §QA）
weights/            # pretrained/ 训练基座 + detector/ 变体（gitignore; 正式权重在 recognize/）
tests/              # plain-script suites (pytest-compatible)
docs/PIPELINE.md    # ★ 权威管线   docs/STATUS.md  # 进度史(中文)   docs/DESIGN.md  # 设计
```

## Status (2026-07-06)

**采集为单一 AI 路径**（autoplay_ai 直接写 `GTRecord`，`intermediate/gt` 退役）。现役数据集
**`datasets/v2` = 28 局纯 AI 采集**（18 `ai_session` + 10 换肤 `ai_session2/run_21..23`），
`--hbb --obb` 一次出双格式（`detector/` + `detector_obb/`），held-out **两整局**
`ai_session_run_8_game1` + 换肤 `ai_session2_run_21_game1`。

- 检测器（2026-07-06，v2 重训，held-out 2 局含 1 换肤局）：现役正式
  `recognize/tile_detector.pt` = **OBB mAP50 0.994 / mAP50-95 0.981**（rotated-IoU，
  run `runs/obb/20260706_014911`）；HBB 变体 `weights/detector/tile_detector_hbb_<ts>.pt`
  = **mAP50 0.992 / mAP50-95 0.957**。val 现含换肤局，数字略低于此前非换肤 val——是更诚实的泛化测。
- 分类器 `tile_classifier.pt`：**尚未在 v2 重训**，仍是 07-03 dealfix 权重（held-out val_acc
  0.9991）。重训一条命令：`bash scripts/train/launch_classifier.sh --dataset v2 --gpu 0`。
- 轨迹（分类器）：93.5 → 95.3(P1 清洗) → 96.0(P2 erode) → 97.6(+AI) → 99.78(16 局精确) → **99.91**(dealfix)。
- ⚠️ 待办：① 分类器在 v2 重训（吃换肤外观多样性）；② 换肤局 dora 牌背被橙背门丢框（back 类
  换肤覆盖缺口，STATUS §1.31 遗留）。

→ 细节：[`docs/STATUS.md`](docs/STATUS.md)。

## 常用命令（PowerShell；管线全文见 [`docs/PIPELINE.md`](docs/PIPELINE.md)）

先自行 activate conda **`auto`** 环境；顶层 import 是 `from majsoul_eye import ...`，
一律从仓库根运行，只需设：

```powershell
$env:PYTHONPATH = "."        # bash: export PYTHONPATH=.
```

### 采集（AI 自动，唯一主路径）

```powershell
python scripts/capture/autoplay_ai.py --dry-run                    # dry-run：
python scripts/capture/autoplay_ai.py --server jp --live --auto-next --out captures/raw/ai_session   # 真跑：整场循环
# 每局写 captures/raw/ai_session/run_<N>/game<M>.jsonl (GTRecord)
#   + game<M>/{frames/*.png, frames.jsonl, liqi.jsonl 线流备份, metadata.json 语言}

python  scripts/capture/autoplay_ai.py --skins --skins-randomize --skins-all-seats
```

关键 flag：`--live`、`--auto-next`（结算自动续局）、`--overlay`（浏览器内画检测框验证）、
`--autojoin`、`--model`、`--lang`、`--quiet`。采集期已内建发牌动画跳过 + ROI 稳定确认（防遮挡）。

### 构建数据集（版本化，一条命令）

```powershell
# 现役 v2 = 纯 AI 两源 + HBB/OBB 双格式（这是当前实际在跑的命令）：
python scripts/data/build_datasets.py v2 --hbb --obb --sources captures/raw/ai_session captures/raw/ai_session2 -j 12
#   不给 --sources 时默认 captures/raw/ai_session；--resume 只补缺的局并重组 detector split；
#   --force 清空重建；--dry-run 干跑
# 产出 datasets/<name>/{annotations/, <game>/{crops,yolo}, detector/(+ --obb 时 detector_obb/), games.json}
```

### 训练（GPU；确切命令 build_datasets 收尾会按当前局清单打印好）

单机直调（PowerShell 里 --val 值必须加引号；--val 可重复 = 多留一整局）：
```powershell
# 38 类分类器（⚠️ 切分按局绝不按帧）
python scripts/train/train_classifier.py --dataset datasets/v2 `
      --val "ai_session_run_8_game1:*" --val "ai_session2_run_21_game1:*" --epochs 20
# YOLO 检测器（imgsz 1280；16GiB 卡 --batch 4 防 OOM；OBB 用 --model weights/pretrained/yolov8s-obb.pt）
python scripts/train/train_detector.py --data datasets/v2/detector/data.yaml
# 跨版本合并检测集：
python scripts/train/build_detector_dataset.py --dataset datasets/v2 --dataset datasets/v3 `
      --val "ai_session_run_8_game1:*" --out datasets/detector_combined
```

#### GPU 服务器（多卡启动器，bash）

三条命令 = 当前完整训练管线（分类器单卡 + 检测器 HBB/OBB 各一组 DDP）：
```bash
# 分类器：单卡（小 CNN，无 DDP）。--gpu 选物理卡（走 CUDA_VISIBLE_DEVICES）；不传 --val 时
#   自动读 datasets/v2/games.json 的 val 列表，与检测器留出同样的整局。~几分钟，先跑完再起下面
#   两组 DDP，或挑一张空卡。
bash scripts/train/launch_classifier.sh --dataset v2 --gpu 0        # -> recognize/tile_classifier.pt
# 检测器：--gpus 选物理卡做 DDP（**别用 CUDA_VISIBLE_DEVICES**——ultralytics select_device 会覆写它）；
#   --batch 是跨卡全局 batch。默认 batch 64 / epochs 60 / imgsz 1280；run 目录 runs/<mode>/<ts>/。
bash scripts/train/launch_detector.sh hbb --dataset v2 --gpus 0,1,2,3   # -> weights/detector/tile_detector_hbb_<ts>.pt
bash scripts/train/launch_detector.sh obb --dataset v2 --gpus 4,5,6,7   # -> weights/detector/tile_detector_obb_<ts>.pt (+ recognize/tile_detector.pt，现役默认)
#   `--` 之后原样透传底层脚本（如 -- --patience 30 --lr0 0.001）。换骨架：--model weights/pretrained/yolo11m.pt。
```
### 可视化
```powershell
python scripts/capture/autoplay_ai.py --overlay --overlay-fps 0.5 --overlay-conf 0.5 
```
### QA / 调试

```powershell
python scripts/inspect/inspect_capture.py <cap.jsonl> <frames_dir> --step 120   # 帧↔GT 对账
python scripts/inspect/overlay_labels.py <cap.jsonl> <frames_dir> --out out/overlay.png --step 120
python scripts/inspect/visualize_failures.py --crops datasets/precise_…/crops --out fails/…
python scripts/annotate/annotate_ai_session.py --captures <cap> --qa-classifier  # 分类器一致率抽查
python scripts/inspect/fiftyone_view.py --data datasets/v3/detector/data.yaml    # FiftyOne GUI 审查检测集（docs/dataset_review.md）
python scripts/inspect/cvat_export.py --game precise_ai_run_1 --out cvat_pkg --zip   # CVAT 修框往返
```

### 测试（tests/test_*.py 全部，普通脚本、兼容 pytest）

```powershell
foreach ($t in Get-ChildItem tests/test_*.py) { python $t.FullName; if ($LASTEXITCODE) { break } }
# bash: for t in tests/test_*.py; do python "$t" || break; done
```

### 运行时识别（manifest-first）

The shipped recognition chain is **manifest-first**: one
`model-manifest.internal-v1.json` names the detector/classifier/HUD-reader
asset files and pins the layout contract, and the runtime is loaded **once**
(`manifest -> one-time runtime -> draft -> override-aware reconstruct`). There
is no mtime-based weight guessing and no loose `--weights` paths. The manifest
carries no digest obligation (2026-07-18): weight bytes are pinned by the
superproject asset lock, and `metadata()` reports the digests observed at load.

```bash
# Quick single-frame inspection (JSON lines: WhatCutDraftV1 + ObservedState + mjai):
PYTHONPATH=. python scripts/recognize/recognize_frame.py --allow-experimental shot.png

# Shared worker — readiness self-check (asset presence + layout contract and,
# for supported layouts, the golden report), no bind:
EYE_REVISION="$(git rev-parse HEAD)" PYTHONPATH=. python \
  scripts/recognize/serve_worker.py \
  --manifest majsoul_eye/recognize/model-manifest.internal-v1.json --check-only
```

Internal loopback deployment (one shared process/device serves every gray-beta
caller — never a worker per request):

```bash
EYE_REVISION="$(git rev-parse HEAD)" PYTHONPATH=. \
  /hszhao-f1/h3011050/anaconda3/envs/majsoul_eye/bin/python \
  scripts/recognize/serve_worker.py \
  --manifest majsoul_eye/recognize/model-manifest.internal-v1.json \
  --device cuda --host 127.0.0.1 --port 8765
```

The remote application uses `http://127.0.0.1:8765` directly (VS Code Remote SSH
may forward remote port `8765` for debugging; the worker still binds only the
loopback interface). **Current desktop-layout support status is `experimental`**
— `majsoul-desktop-16x9-v1` stays experimental until an independent
100-image/20-game golden report passes the immutable P0 thresholds and binds the
final manifest SHA (see [`docs/WHAT_CUT_GOLDENS.md`](docs/WHAT_CUT_GOLDENS.md)).

## ⚠️ Notes

- Tile taxonomy is **38 classes** (34 tiles + 3 red fives + `back`); ordering is
  frozen by the original `tile.model` — see `tiles.py`. Do not reorder.
- Coordinate baselines differ: **mycv = 1920×1080**, **Playwright = 1600×900**.
  Always normalize to 0–1 before converting between them.
- Train/val split **by kyoku/game, never by frame** (same physical tile spans ~10
  frames; a frame split leaks).
- `frames.jsonl` `file` entries are RELATIVE — always resolve via
  `paths.resolve_frame_path`; the layout's single source of truth is `majsoul_eye/paths.py`.
- Risk/compliance (time-sync, ban-avoidance): see `docs/DESIGN.md` §7. Use burner
  accounts for autoplay capture.
