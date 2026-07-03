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

```text
majsoul_eye/
  tiles.py          # unified 38-class taxonomy + MJAI interop (shared by all components)
  coords.py         # normalized ROI model (hand/dora slots, coarse per-quadrant river zones)
  normalize.py      # board locators: fullscreen / letterbox / anchor(TODO) -> canonical 16:9 frame
  paths.py          # captures/ layout single source of truth (frames_dir_for / resolve_frame_path)
  capture/          # ⚠️ DEV-ONLY, Akagi-coupled recorder (schema / akagi_tap / screen / sync)
  state/replay.py   # pure replayer: MJAI events -> full 4-player BoardState + invariants
  label/            # legacy easy-zone auto-annotator: autolabel (hand+dora) + quality gate
  annotate/         # PRECISE fullwarp annotator: pipeline (geometry engine) / frame / seatgt / cases
  recognize/classifier.py  # TileNet 38-class classifier (+ tile_classifier.pt, 6 games 97.6%)
  baselines/        # real-mycv engine adapter + bag-matching scorer (accuracy baseline)
scripts/
  record_gt.py / autoplay_ai.py          # capture: manual F11 (Akagi) / Mortal-AI autoplay (--auto-next)
  crop_game.py / deletterbox_frames.py   # frame repair: crop to 16:9 / strip letterbox bars
  ingest_run.py -> convert_mjcopilot.py  # AI run: discover games -> liqi wire -> our GT jsonl
  build_dataset.py / train_classifier.py # auto-labeled crops + YOLO -> 38-class classifier
  annotate_ai_session.py                 # annotation v2: per-frame river/meld/hand boxes + QA
  build_case_annotations.py              # 11-case AB JSON (out/mahjong_AB_relative_data_with_reliability.json)
  calibrate_annotation_model.py          # measure / refit the fullwarp constants
  spike_topdown.py                       # ARCHIVED H_table viz spike (self-contained; superseded by annotate/)
  inspect_capture.py / overlay_labels.py / visualize_failures.py / mycv_baseline.py  # QA & debug
  fiftyone_view.py / cvat_export.py / cvat_import.py  # dataset review/clean (FiftyOne GUI) + box-fix round-trip (CVAT)
  migrate_captures_layout.py             # one-shot captures/ layout migrator (dry-run default)
tests/              # 10 suites: tiles replay sync label classifier mycv_baseline quality coords annotate_pipeline annotate_frame
docs/DESIGN.md      # design & rationale       docs/STATUS.md  # living status + roadmap (中文)
```

## Status

**Full pipeline validated on 6 games (2 manual 4K + 4 AI 1080p) — tile classifier
97.6% (held-out AI game 98.5%), zero manual annotation. Precise annotation v2 shipped.**

- Pipeline: `record(F11)/autoplay(AI) → debounce capture → protocol-GT replay → geometric auto-label → train`.
- Classifier trajectory: 93.5 → 95.3 (label cleaning) → 96.0 (river erode) → **97.6** (+AI data).
- Annotation v2: one calibrated fullwarp geometry for all 4 rivers / melds / hand,
  per-frame snap + fill confidence; classifier-agreement QA 96.6–100% per zone
  (`annotate_ai_session.py`, all 16 AI games).
- Data: 16 AI games captured (`captures/raw/ai_session`, ~8.9k frames) all converted to GT;
  current training set 6 games / ~112k crops; red fives now abundant.
- `captures/` reorganized into `raw/intermediate/legacy` roles with relative frame
  indexes (`majsoul_eye/paths.py`).

→ 完整进度、实测结论与路线图：[`docs/STATUS.md`](docs/STATUS.md)。Design rationale:
[`docs/DESIGN.md`](docs/DESIGN.md).

## 管线与脚本用法

所有代码在 conda **`auto`** 环境跑；仅 `record_gt.py`
在 **`akagi`** 环境（跑在 Akagi 进程内）。顶层 import 是 `from majsoul_eye import ...`，
一律从仓库根运行。命令块为 **PowerShell** 语法（bash 等价见 `CLAUDE.md` /
`docs/STATUS.md` §四）；每开一个新终端先执行一次：

```powershell
$PY = "C:/Users/zsx/miniforge3/envs/auto/python.exe"
$env:PYTHONPATH = "."
```

`captures/` 布局的单一真源是 `majsoul_eye/paths.py`：`raw/{ai_session,manual}`（原始，
不可再生）→ `intermediate/{gt,derived}`（转换后 GT / 修复帧，可再生）→ `legacy/`（归档）。
`frames.jsonl` 里存相对路径，读帧永远经 `paths.resolve_frame_path` 解析（兼容旧绝对路径）。

### 1) 采集

**`record_gt.py`** — 手动 F11 局：注入 GT 录制器后启动 Akagi，事件安静即截图
（akagi 环境；autoplay OFF 被动采集、WEB 客户端 F11 全屏、默认素色桌布）。

```powershell
conda run -n akagi pip install mss opencv-python   # 一次性：截图依赖
conda run -n akagi python scripts/capture/record_gt.py --screenshots --quiet 0.30 --settle-cap 2.0
# 默认写 captures/raw/manual/session_<ts>.jsonl + 同名帧目录；状态日志 → <out>.jsonl.log
```

关键 flag：`--out`、`--screenshots`、`--quiet`（板面事件安静多少秒才截）、`--settle-cap`
（事件连发时强制截图的上限秒数）、`--akagi-dir`。

**`autoplay_ai.py`** — Mortal AI 自动对局采集（单 `auto` 环境：Playwright 抓 liqi WS +
截图 + MahjongCopilot 驱动点击）。默认 dry-run 只打日志，确认动作正确后再 `--live`；小号！

```powershell
& $PY scripts/capture/autoplay_ai.py --live --auto-next      # 整场循环采集
# 每次运行写 captures/raw/ai_session/run_<N>/（一局一个 game<M>/：liqi 线流 + 1080p PNG）
```

关键 flag：`--live`（真点击）、`--auto-next`（结算后截图守卫式点确认+"再来一局"，配
`--auto-next-confirms` / `--auto-next-timeout`）、`--autojoin`、`--model`、`--server`、`--quiet`。

**`crop_game.py`** — 非全屏手动局裁回 16:9 画布（一次检测、全局套用，非破坏性）。

```powershell
& $PY scripts/data/crop_game.py captures/raw/manual/session5 captures/intermediate/derived/session5_16x9 --size 3840x2160
```

**`deletterbox_frames.py`** — 去黑边（如 run_5 重连局）：检测黑条 → 裁剪 → resize 回
1920×1080，写新的自包含帧目录（seq 一一对应），之后用 `--frames-dir` 喂给标注器。

```powershell
& $PY scripts/data/deletterbox_frames.py --capture captures/intermediate/gt/ai_run_5_game2.jsonl `
    --out captures/intermediate/derived/ai_run_5_game2_deletterboxed
```

### 2) AI 数据接入（MahjongCopilot 线流 → 我们的 GT）

**`ingest_run.py`** — 一键编排：自动发现 run 内各局（单局/多局布局均可）→ convert →
build_dataset（→ 可选重训）。

```powershell
& $PY scripts/data/ingest_run.py captures/raw/ai_session/run_4
& $PY scripts/data/ingest_run.py captures/raw/ai_session/run_4 --train --val "ai_run_4_game1:*"
```

**`convert_mjcopilot.py`** — 底层转换器（ingest 内部调用，也可单跑）：raw liqi 线流 →
MahjongCopilot 的 `LiqiProto`/`GameState`（stub bot）→ MJAI → `captures/intermediate/gt/<NAME>.jsonl`
（帧索引指回 raw/ 的 PNG）。逐事件 deepcopy（GameState 原地改手牌）、按每条 input() 增量对 seq。

```powershell
& $PY scripts/data/convert_mjcopilot.py --game "run_3/game1=ai_run_3_game1" --mjcopilot ../MahjongCopilot
```

### 3) 数据集构建

**`build_dataset.py`** — 同步采集 → 分类裁剪 `crops/<牌>/` + YOLO `yolo/images,labels/`；
按全局 `seq` join 帧↔GT；P1 牌面占比门（`--min-face-frac`，挡空毡误标）与 P2 河格 erode
（`--river-erode-bottom/--river-erode-side`）均已默认开启。

```powershell
& $PY scripts/train/build_dataset.py captures/raw/manual/sessionN.jsonl captures/raw/manual/sessionN/ `
    --out datasets/sessionN --locator fullscreen --drop-violations
```

关键 flag：`--locator fullscreen|letterbox`、`--drop-violations`（丢弃 invariant 违例帧）、
`--min-bright` / `--min-face-frac`（P1 空毡门）。河/副露走精确 `annotate/` 管线；`label/autolabel`
只剩易区，`DEFAULT_ZONES = {hand}`（dora/score/meta 为 opt-in）。

### 4) 精准标注 v2（河/副露/手牌逐帧框）

几何引擎是包内 **`majsoul_eye/annotate/pipeline.py`**（fullwarp 单应 + 数据标定的
牌面网格/副露组成模型 + 缝隙/边缘掩膜检测器，脚本 `from majsoul_eye.annotate import pipeline as P`；
曾在根级 `mahjong_relative_annotation_pipeline.py`，已移除）。共享 GT plumbing 也已进包：
`capture.gtframes`（`build_seq_state`/`load_frames`/`load_pair`）、`annotate.seatgt`（`_screen_to_seat`/`SEAT_POS`）、
`annotate.cases`（`CASES`）。`scripts/annotate/spike_topdown.py` 现为**已归档的自足可视化 spike**（不再承重，
从包 import；其 H_table 几何被 `annotate/` 取代）。

**`annotate_ai_session.py`** — 主产品：全帧标注器。4 家河（标定网格 + GT + 逐格 fill 置信，
最新弃牌未渲染 → `unrendered`）、4 家副露（组成感知 strip + 逐帧 snap）、英雄手牌
（HandModel + 白度门）。输出 `out/ai_session_annotations/`（每局 JSONL + overlays/ + summary.json）。

```powershell
& $PY scripts/annotate/annotate_ai_session.py                 # 默认标注全部 captures/intermediate/gt/*.jsonl
& $PY scripts/annotate/annotate_ai_session.py --captures captures/intermediate/gt/ai_run_3_game1.jsonl `
    --overlay-every 40 --qa-classifier
```

关键 flag：`--qa-classifier`（用正式分类器抽查 crop 一致率，QA ≈96.6–100%）、`--frames-dir`
（指向 deletterbox 修复帧）、`--overlay-every`、`--out`。

**`build_case_annotations.py`** — 11 个固化 case 的 AB 标注 JSON（弃牌带 GT 标签 + 副露框）。

```powershell
& $PY scripts/annotate/build_case_annotations.py --overlays out/topdown_annot
# 写 out/mahjong_AB_relative_data_with_reliability.json
```

**`calibrate_annotation_model.py`** — 跨局测量缝隙/边缘特征 → 稳健线性拟合 → 打印建议常量
（引擎常量漂了就复跑再校准）。

```powershell
& $PY scripts/annotate/calibrate_annotation_model.py --per-game 40 --out scratchpad/calib.json
& $PY scripts/annotate/calibrate_annotation_model.py --refit scratchpad/calib.json
```

### 5) 训练

**`train_classifier.py`** — 38 类牌分类器（TileNet）；**按局/场切分 val，绝不按帧**
（同一物理牌横跨 ~10 帧，帧切分必泄漏）。类均衡采样 + 轻增广，GPU 自动启用。
⚠️ PowerShell 里 `--data`/`--val` 的值**必须加引号**（裸的 `,` 会被 PS 拆成数组）。

```powershell
& $PY scripts/train/train_classifier.py `
    --data "s6=datasets/session6_erode/crops:captures/raw/manual/session6.jsonl" `
    --data "ai1=datasets/ai_g1/crops:captures/intermediate/gt/ai_g1.jsonl" `
    --val "s6:E3.0,S2.0" --epochs 20 --workers 6
```

关键 flag：`--data NAME=crops:capture`（可重复）、`--val NAME:场序` / `--val NAME:*`
（`*` = 整局 held-out）、`--epochs`、`--batch`、`--workers`（GPU 建议 6）、`--out`。

### 6) QA / 调试工具

```powershell
# 帧↔GT 对账 + settle 质量（离线，不需客户端）
& $PY scripts/inspect/inspect_capture.py captures/raw/manual/session6.jsonl captures/raw/manual/session6/ --step 120
# 把自动标注画到帧上，肉眼校准坐标
& $PY scripts/inspect/overlay_labels.py captures/raw/manual/session6.jsonl captures/raw/manual/session6/ `
    --out out/overlay.png --step 120
# 分类器错例蒙太奇（按 gt→pred 混淆对分组）
& $PY scripts/inspect/visualize_failures.py --crops datasets/ai_g1/crops --out fails/ai_g1
# mycv 真实管线基线（对照精度）
& $PY scripts/inspect/mycv_baseline.py --capture captures/raw/manual/session6.jsonl `
    --frames captures/raw/manual/session6/frames
# captures/ 布局再迁移（dry-run 默认；写 MIGRATION_MANIFEST.json，幂等可续跑）
& $PY scripts/data/migrate_captures_layout.py            # 预览
& $PY scripts/data/migrate_captures_layout.py --apply --strict
```

**数据集可视化 / 清理**（YOLO 检测集，完整说明见 [`docs/dataset_review.md`](docs/dataset_review.md)）：

```powershell
& $PY -m pip install fiftyone                               # 一次性（会把 protobuf 升到 7.x）
& $PY scripts/inspect/fiftyone_view.py                      # FiftyOne GUI：按 game/split/类别筛选 → 给坏帧打 tag reject
& $PY scripts/inspect/fiftyone_view.py --export-clean datasets/detector_clean   # 导出剔除 reject 的干净 train/val
# CVAT 修框往返：export 打包 → CVAT 改框 → import 无损写回 datasets/<game>/yolo/labels/
& $PY scripts/inspect/cvat_export.py --game precise_ai_run_1 --out cvat_pkg --zip
& $PY scripts/inspect/cvat_import.py 你从CVAT导出的.zip --dry-run
```

### 测试（10 套，普通脚本、兼容 pytest）

```powershell
foreach ($t in "tiles","replay","sync","label","river","meld","classifier","mycv_baseline","quality","coords") {
  & $PY "tests/test_$t.py"; if ($LASTEXITCODE) { break }
}
```

## ⚠️ Notes

- Tile taxonomy is **38 classes** (34 tiles + 3 red fives + `back`); ordering is
  fixed by what `tile.model` was trained on — see `tiles.py`. Do not reorder.
- Coordinate baselines differ: **mycv = 1920×1080**, **Akagi/Playwright = 1600×900**.
  Always normalize to 0–1 before converting between them.
- Risk/compliance (time-sync, ban-avoidance, Akagi's AGPLv3 + Commons Clause):
  see `docs/DESIGN.md` §7. Prefer **passive capture** (观战/人工对局) over autoplay.
