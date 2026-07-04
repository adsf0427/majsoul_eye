# majsoul_eye 数据管线（权威文档）

> **本文是"当前管线"的唯一权威描述。** 任何改动（脚本、数据格式、目录布局、默认参数）
> 只要影响 采集/标注/建库/训练 任何一环，**必须同步更新本文**（维护规约见 §8）。
> 历史沿革与实测结论见 [STATUS.md](STATUS.md)；设计论证见 [DESIGN.md](DESIGN.md)。
>
> 最后更新：2026-07-04（采集统一 AI 路线；`intermediate/gt` 退役；run_13/14 补建；
> **数据集版本化**：`build_datasets.py` 构建自包含 `datasets/<name>/`，训练可吃多版本；
> `rebuild_datasets.py` 弃用）。

## 0. 一图流

```
【采集 · 唯一主路径】  scripts/capture/autoplay_ai.py --live [--auto-next] [--overlay]
   (auto 环境: Playwright WS tap + Mortal 决策 + 截图-on-quiet, 实时内联写统一 GTRecord)
   → captures/raw/ai_session/run_N/gameM.jsonl                 ← GTRecord (GT 真源)
   → captures/raw/ai_session/run_N/gameM/{frames/*.png, frames.jsonl,
                                          liqi.jsonl(线流备份), metadata.json(语言)}
        │
        ▼
【构建 · 一条命令】  scripts/data/build_datasets.py <name> [--sources 根目录...] [--resume|--force]
   默认 sources = captures/raw/ai_session（可加 captures/raw/manual、将来的 ai_session_2 等；
   立即执行，--dry-run 才是干跑）。内部按序编排三个阶段，产出自包含版本目录：

   datasets/<name>/                      ←（现役：datasets/v1，20 局）
     annotations/                        标注记录（AI 局；annotate_ai_session 产出）
     <game>/{crops/<38类>/, yolo/{images,labels}}   每局一个子文件夹（build_dataset 产出）
     detector/{train.txt, val.txt, data.yaml}       按局切分的检测器装配（build_detector_dataset）
     games.json                          清单：每局 name/capture/frames_dir/dir + val
        │
        ▼
【训练 · GPU, 手动触发 · 可吃多个版本】
   scripts/train/train_classifier.py --dataset datasets/v1 [--dataset datasets/v2 ...]
       → majsoul_eye/recognize/tile_classifier.pt   （38类, 正式）
   scripts/train/train_detector.py --data datasets/<name>/detector/data.yaml
       → weights/detector/*.pt → 择优升 recognize/tile_detector.pt
   # 跨版本合并检测集：build_detector_dataset.py --dataset datasets/v1 --dataset datasets/v2 ...
        │
        ▼
【运行时产品 · Akagi-free】  majsoul_eye/recognize/  (TileClassifier / TileDetector)
```

三个内部阶段（单独跑/调试时才手动调）：`annotate_ai_session.py`（精确 fullwarp 几何 + GT 赋类，
**可跳过**——见 §2 建库）→ `build_dataset.py`（crops+yolo）→ `build_detector_dataset.py`（split）。
（`rebuild_datasets.py` 已弃用——被版本化的 build_datasets 取代，见 §4。）

## 1. 数据目录与角色（单一真源 `majsoul_eye/paths.py`）

| 路径 | 角色 | 可再生? |
|---|---|---|
| `captures/raw/ai_session/run_N/` | **原始 GT + 帧（主采集路径产物）**：`gameM.jsonl` = GTRecord；`gameM/` = 帧目录 | ❌ 不可再生，唯一需备份的数据 |
| `captures/raw/manual/session5,6*` | 手动 F11 局（record_gt 产物，**采集方式已过时**；数据保留，仍进训练集） | ❌ 冻结存档 |
| `captures/intermediate/derived/` | 修复帧（去黑边 `*_fixed`、裁 16:9）——仅历史遗留局需要 | ✅ 由 raw 重建 |
| `captures/legacy/` | 归档的逐字节重复（ai_g*/ai_r1） | — 可删 |
| `captures/raw/temp/` | ⚠️ 无 GT 的孤儿帧（采集失败残留，只有 PNG 没有对局 jsonl）——不可用，待清理 | — 垃圾 |
| `datasets/<name>/`（版本目录，现役 `v1`） | 自包含数据集：`annotations/` + 每局 `<game>/{crops,yolo}` + `detector/`（不拷图，txt 引用）+ `games.json` 清单 | ✅ build_datasets.py |
| `out/ai_session_annotations/` | 旧全局标注位置（v1 时代产物；已拷贝进 `datasets/v1/annotations/`） | — 可删 |
| `majsoul_eye/recognize/*.pt` | **正式权重**（tile_classifier.pt 入 git；tile_detector.pt 本地） | GPU 重训 |
| `weights/` | `pretrained/` 训练基座 + `detector/` 变体（aabb/obb 等，均 gitignore） | — |

规则：
- `frames.jsonl` 的 `file` 一律**相对路径**，读取永远经 `paths.resolve_frame_path`；
  帧目录与 GT 的耦合是 `X.jsonl ↔ X/`（`paths.frames_dir_for`），不要自己重推。
- **数据集/标注/装配全部是衍生物**（gitignored）：标注代码一变，它们就"过期"，用
  `build_datasets.py <name> --force`（或建新版本）重建，不要手工修补/移动——v1 的搬家就击穿过
  detector split 的相对路径（STATUS §1.22）。

## 2. 各阶段要点

### 采集（唯一主路径 = AI 自动）
- `autoplay_ai.py`：单 `auto` 环境。Playwright 抓 liqi WS + Mortal 决策点击 + 事件安静截图，
  **边打边写统一 GTRecord**（与旧手动格式同构，无任何转换步骤）。默认 dry-run，确认后 `--live`；
  `--auto-next` 结算续局循环；`--overlay`（连续）/`--overlay-manual --overlay-key`（按键单帧）
  浏览器内画检测框（验证用）；`--skins` 经 MajsoulMax MITM 换肤/牌背/桌布（训练外观多样性）；小号。
- 采集期已内建两类脏帧规避：发牌动画不 arm 截图（ActionMJStart/NewRound）、ROI 稳定确认
  （`capture/roi_diff.py`，防弃牌动画遮挡，实测残留 ~0.4%）。
- 每局写 `metadata.json`（显示语言 BCP-47，`--lang` > localStorage 探测 > 服务器粗判）。
- **已过时**：`record_gt.py` 手动 F11 + Akagi 路线（akagi 环境）。不再用于新采集；
  脚本保留只为存档复现 session5/6。

### 标注（annotate）
- `annotate_ai_session.py` 默认标注**全部** `paths.ai_captures()`；`--captures` 指定局；
  `--frames-dir` 用于 derived 修复帧（run_5 game2/3 信箱局）；`--workers` 默认保守 4（RAM 束）。
- GT 谓词丢弃发牌窗帧（`replay.is_deal_window`：rivers 全空）；hero 摸牌槽经 `replay.drawn_tile`
  正确标注（14 张自摸态不再漏标）。

### 建库（build_dataset）
- **标注步不是必须的**：不给 `--from-annotations` 时 build_dataset 走**自足模式**（内部逐帧
  内联跑 `annotate_frame`），一步直接出 crops+yolo，输出与"先标注再 `--from-annotations`"
  **逐字节相同**（STATUS §1.14 验证）。单局快速验证用这条最短路径。
- 标注步的价值 = 可复用缓存 + 附加产物：多局进程池并行、overlays/QA（`--qa-classifier`）、
  一次标注多次建库。批量构建（build_datasets）走 标注 → `--from-annotations` 路线省时间，
  标注缓存就放在版本目录内（`datasets/<name>/annotations/`，`--resume` 据此跳过已标局）。
- manual session5/6 一律直接建（不经标注层）。
- `--drop-violations` 常开；遮挡一致性门 `--occlusion-gate` **默认关**（采集期 roi_diff 已防大头）。
- 输出既有 crops（分类）也有 yolo（检测）——同一套精确几何，一次标定两处喂。

### 装配 + 训练
- **切分铁律：按局/kyoku，绝不按帧**（同一物理牌跨 ~10 帧，帧切分必泄漏）。
  惯例 held-out：**整局 `ai_run_8_game1`**（分类器与检测器同一局，趋势可比）。
- **多版本输入**：`train_classifier.py` 与 `build_detector_dataset.py` 均支持可重复的
  `--dataset datasets/<name>`（读 `games.json` 自动展开成逐局 `--data` 条目；同名局后者覆盖
  前者并打印提示；仍可混用显式 `--data`）。
- 分类器：`train_classifier.py --dataset datasets/v1 [--dataset datasets/v2 ...] --val ai_run_8_game1:* --epochs 20`。
- 检测器：`train_detector.py --data datasets/<name>/detector/data.yaml`（imgsz 1280；16GiB 卡加
  `--batch 4` + expandable_segments 防 OOM；OBB 用 `--model weights/pretrained/yolov8s-obb.pt`）。
  跨版本先合并 split：`build_detector_dataset.py --dataset datasets/v1 --dataset datasets/v2
  --val ai_run_8_game1:* --out datasets/detector_combined`。
- 训练命令 `build_datasets.py` 收尾会按当前局清单打印好，直接复制。

## 3. SOP：新采集一个 run 后

```powershell
# 先自行 activate conda auto 环境；仓库根运行
$env:PYTHONPATH = "."        # bash: export PYTHONPATH=.
# A) 增量并入当前版本（日常推荐）：只处理缺的局，detector split + games.json 自动重组
python scripts/data/build_datasets.py v1 --sources captures/raw/ai_session captures/raw/manual --resume
# B) 建全新版本（标注代码变更后 / 要干净快照时）：
python scripts/data/build_datasets.py v2                       # 默认 sources = captures/raw/ai_session
# （--force 清空重建同名版本；--dry-run 干跑；机器好加 --workers 16 --jobs 12）
# C) GPU 训练（可吃多个版本；确切命令 build_datasets 收尾已打印）
python scripts/train/train_classifier.py --dataset datasets/v1 --val "ai_run_8_game1:*" --epochs 20
python scripts/train/train_detector.py --data datasets/v1/detector/data.yaml
```

新 run 在 `--sources` 根下自动发现，无需登记；**游戏名（run 编号）必须跨 source 根全局唯一**
（如将来 `captures/raw/ai_session_2` 从 `run_15` 起编号），冲突会直接报错。

## 4. 过时/降级组件清单（勿再当作管线环节）

| 组件 | 现状 |
|---|---|
| `scripts/data/rebuild_datasets.py` | **已弃用**——被版本化的 `build_datasets.py` 取代（原地重建旧固定布局 vs 自包含 `datasets/<name>/`）。验证期后删除 |
| `scripts/capture/record_gt.py`（+ akagi 环境、Akagi MITM） | **过时的采集方式**。新数据一律 autoplay_ai；脚本保留仅为存档 |
| `scripts/data/convert_mjcopilot.py` | 降级为**共享转换库**（`convert_game` 被迁移器复用）；不再是管线一环。可独立 CLI 处理任何遗留 b64 线流 |
| `scripts/data/ingest_run.py` | 遗留便捷入口（发现→建库）。用 §3 的 build_datasets 代替 |
| `scripts/data/migrate_ai_to_gtrecord.py` | 一次性迁移（18 局 b64 → GTRecord），**已完成**（2026-07-04）。仅新发现遗留线流时再用 |
| `scripts/data/migrate_captures_layout.py` | 一次性布局迁移，已完成（2026-07-02） |
| `scripts/data/purge_deal_frames.py` / `apply_deal_purge.py` / `purge_occlusion_frames.py` | 针对旧数据集的一次性清洗；现由采集期规避 + 建库期丢弃取代。全量重建后无需再跑 |
| `scripts/data/crop_game.py` / `deletterbox_frames.py` | 仅历史遗留局的帧修复（session5 非全屏 / run_5 信箱）。新采集全屏 1080p 用不到 |
| `scripts/annotate/spike_topdown.py` | 已归档的可视化 spike，不承重 |
| `captures/intermediate/gt/` | **已退役删除**（AI 采集直接写 GTRecord，无转换产物） |
| `label/`（`autolabel.py`） | 仅剩 hero 手牌+dora 框供 `annotate_frame` 调用；river/meld 旧几何已删 |

## 5. 数据与权重现状快照（2026-07-04）

- **原始数据**：AI 18 局（run_1, run_3×4, run_4×1(掉线), run_5×3(2 局信箱→derived 修复),
  run_7×1, run_8×6, run_13/14×1(早退迷你局, 各15帧)）+ 手动 2 局（session5/6, 4K）。
- **衍生数据**：**`datasets/v1/`**（20 局子文件夹 `precise_*` + `detector/` train 8683/val 949 +
  `annotations/` 缓存 + `games.json`）—— 2026-07-04 全量重建（含 hero-tsumo 修复；run_13/14
  同日补建；由旧平铺布局手工移入后已修复 detector 路径并补清单，`--resume` 可零成本续跑）。
- **正式权重**：
  - `recognize/tile_classifier.pt` — held-out val_acc **0.9991**（07-03 dealfix 数据训）。
  - `recognize/tile_detector.pt` — HBB mAP50 0.993 / mAP50-95 0.955；OBB 变体
    `weights/detector/tile_detector_obb.pt` mAP50-95 **0.9804**（rotated-IoU）。
- ⚠️ **待办**：分类器+检测器都还没在 07-04 重建的数据（hero-tsumo 手牌帧 + run_13/14）上重训
  ——检测器受益最大（own-turn 手牌此前是负样本信号）。

## 6. 维护规约（每次改动必过一遍）

改动涉及以下任何一项时，**提交前必须**：

1. 问一遍："这会让 out/ 或 datasets/ 里的衍生数据过期吗？" 会 → 在 PR/commit 或 STATUS.md
   里写明"需 `build_datasets.py <name> --force` 重建（或建新版本）"，重大者当场重建。
2. 问一遍："这改变了管线的输入/输出/步骤/默认值吗？" 会 → **更新本文**对应小节
   （一图流 / 目录表 / SOP / 过时清单）。
3. 新增脚本必须归位：是管线环节（进 §0/§2）还是一次性工具（进 §4）？不允许"游离脚本"。
4. STATUS.md 追加一节记录（问题→处理→验证→结果），并刷新其 TL;DR 若数字变化。
5. 涉及数据格式/目录的，`majsoul_eye/paths.py` 是唯一真源——改那里，不改散落字面量。
