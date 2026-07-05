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
   `--sources` 指定多个采集根、`--resume` 增量并入新局、`--force` 清空重建。现役版本 `datasets/v1`。
4. **训练** `train_classifier.py --dataset datasets/v1 [--dataset …]` /
   `train_detector.py --data datasets/<name>/detector/data.yaml`（GPU，手动触发；多版本可混用）。

> ⚠️ **`record_gt.py`（Akagi MITM 手动 F11 采集）已列为过时** —— 新数据一律走
> autoplay_ai；session5/6 的存量手动数据保留在训练集中。

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
  train/build_dataset.py         # crops + YOLO（--from-annotations 复用标注; --obb 8点标签）
  train/build_detector_dataset.py / train_classifier.py   # 均支持 --dataset 多版本清单展开
  train/train_detector.py
  data/build_datasets.py         # ★ 版本化构建 datasets/<name>/（标注→建库→装配 + games.json）
  data/migrate_*.py, purge_*.py, crop_game.py, deletterbox_frames.py, convert_mjcopilot.py,
    ingest_run.py, rebuild_datasets.py(弃用)  # 一次性/遗留工具（非管线环节，见 PIPELINE §4）
  annotate/{build_case_annotations,calibrate_annotation_model,spike_topdown}.py  # AB case/标定/归档 spike
  inspect/…                      # QA & debug（下方 §QA）
weights/            # pretrained/ 训练基座 + detector/ 变体（gitignore; 正式权重在 recognize/）
tests/              # plain-script suites (pytest-compatible)
docs/PIPELINE.md    # ★ 权威管线   docs/STATUS.md  # 进度史(中文)   docs/DESIGN.md  # 设计
```

## Status (2026-07-04)

**采集已统一为单一路径**（autoplay_ai 直接写 `GTRecord`，`intermediate/gt` 退役）。
数据 **18 AI 局 + 2 手动 4K 局**，全部经精确标注 v2 重建（含 hero-tsumo 修复）。

- 分类器 `tile_classifier.pt`：held-out 整局 val_acc **0.9991**。
- 检测器：HBB `tile_detector.pt` **mAP50 0.993 / mAP50-95 0.955**；OBB 变体
  `weights/detector/tile_detector_obb.pt` **mAP50-95 0.9804**（rotated-IoU）。
- 轨迹：93.5 → 95.3(P1 清洗) → 96.0(P2 erode) → 97.6(+AI) → 99.78(16 局精确) → **99.91**(dealfix)。
- ⚠️ 待办：两模型尚未在 07-04 重建数据（hero-tsumo 手牌帧 + run_13/14）上重训。

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
python scripts/capture/autoplay_ai.py --server jp --live --auto-next --out captures/raw/ai_session# 真跑：整场循环
# 每局写 captures/raw/ai_session/run_<N>/game<M>.jsonl (GTRecord)
#   + game<M>/{frames/*.png, frames.jsonl, liqi.jsonl 线流备份, metadata.json 语言}

python  scripts/capture/autoplay_ai.py --skins --skins-randomize --skins-all-seats
```

关键 flag：`--live`、`--auto-next`（结算自动续局）、`--overlay`（浏览器内画检测框验证）、
`--autojoin`、`--model`、`--lang`、`--quiet`。采集期已内建发牌动画跳过 + ROI 稳定确认（防遮挡）。

### 构建数据集（版本化，一条命令）

```powershell
python scripts/data/build_datasets.py v1 -j 12   #默认 --sources captures/raw/ai_session）
python scripts/data/build_datasets.py v2 --sources captures/raw/ai_session captures/raw/ai_session2 captures/raw/ai_session/mannual 
#   --resume 只补缺的局并重组 detector split；--force 清空重建；--dry-run 干跑
# 产出 datasets/<name>/{annotations/, <game>/{crops,yolo}, detector/, games.json}
```

### 训练（GPU；确切命令 build_datasets 收尾会按当前局清单打印好）

```powershell
# 38 类分类器（⚠️ 切分按局绝不按帧；PowerShell 里 --val 值必须加引号）
python scripts/train/train_classifier.py --dataset datasets/v1 --val "ai_run_8_game1:*" --epochs 20
# YOLO 检测器（imgsz 1280；16GiB 卡 --batch 4 防 OOM；OBB 用 --model weights/pretrained/yolov8s-obb.pt）
python scripts/train/train_detector.py --data datasets/v1/detector/data.yaml
# 跨版本合并检测集：
python scripts/train/build_detector_dataset.py --dataset datasets/v1 --dataset datasets/v2 `
      --val "ai_run_8_game1:*" --out datasets/detector_combined
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
python scripts/inspect/fiftyone_view.py                    # FiftyOne GUI 审查检测集（docs/dataset_review.md）
python scripts/inspect/cvat_export.py --game precise_ai_run_1 --out cvat_pkg --zip   # CVAT 修框往返
```

### 测试（tests/test_*.py 全部，普通脚本、兼容 pytest）

```powershell
foreach ($t in Get-ChildItem tests/test_*.py) { python $t.FullName; if ($LASTEXITCODE) { break } }
# bash: for t in tests/test_*.py; do python "$t" || break; done
```

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
