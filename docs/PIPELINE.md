# majsoul_eye 数据管线（权威文档）

> **本文是"当前管线"的唯一权威描述。** 任何改动（脚本、数据格式、目录布局、默认参数）
> 只要影响 采集/标注/建库/训练 任何一环，**必须同步更新本文**（维护规约见 §8）。
> 历史沿革与实测结论见 [STATUS.md](STATUS.md)；设计论证见 [DESIGN.md](DESIGN.md)。
>
> 最后更新：2026-07-06（**分类器纳入启动器框架**：新增 `scripts/train/launch_classifier.sh`——
> 单卡包装 train_classifier.py，不传 `--val` 时自动读 `games.json` 的 val 列表、与检测器留出
> 同样的整局；现役数据集切 **`datasets/v2`**（28 局纯 AI）；清理已完成的一次性脚本
> `ingest_run.py` / `migrate_*.py` / `backfill_skin_meta.py`——见 §4）。
> 前次：2026-07-05（GT jsonl 归入对局目录 `run_N/gameM/gameM.jsonl`，每局目录自包含）；
> 2026-07-04（采集统一 AI 路线；`intermediate/gt` 退役；数据集版本化 `build_datasets.py`）。

## 0. 一图流

```
【采集 · 唯一主路径】  scripts/capture/autoplay_ai.py --live [--auto-next] [--overlay]
   (auto 环境: Playwright WS tap + Mortal 决策 + 截图-on-quiet, 实时内联写统一 GTRecord)
   → captures/raw/ai_session/run_N/gameM/                      ← 每局一个自包含目录
        {gameM.jsonl ← GTRecord (GT 真源), frames/*.png, frames.jsonl,
         liqi.jsonl(线流备份), metadata.json(语言)}
        │
        ▼
【构建 · 一条命令】  scripts/data/build_datasets.py <name> [--sources 根目录...] [--resume|--force]
   默认 sources = captures/raw/ai_session（可加 captures/raw/manual、将来的 ai_session_2 等；
   立即执行，--dry-run 才是干跑）。内部按序编排三个阶段，产出自包含版本目录：

   datasets/<name>/                      ←（现役：datasets/v2，28 局纯 AI）
     annotations/                        标注记录（AI 局；annotate_ai_session 产出）
     <game>/{crops/<38类>/, yolo/{images,labels}}   每局一个子文件夹（build_dataset 产出）
     detector/{train.txt, val.txt, data.yaml}       按局切分的检测器装配（build_detector_dataset）
     games.json                          清单：每局 name/capture/frames_dir/dir + val（held-out 局列表）+ formats（hbb/obb）
        │
        ▼
【训练 · GPU, 手动触发 · 可吃多个版本 · 多卡用启动器 launch_*.sh】
   scripts/train/launch_classifier.sh --dataset v2 --gpu 0            # 单卡；自动读 games.json val
       → majsoul_eye/recognize/tile_classifier.pt   （38类, 正式）
   scripts/train/launch_detector.sh {hbb|obb} --dataset v2 --gpus IDS # 多卡 DDP
       → weights/detector/tile_detector_<mode>_<ts>.pt（每 run 版本化，不互相覆盖）
       → OBB 另复制一份到 recognize/tile_detector.pt（现役运行时默认）
   # 直调底层：train_classifier.py --dataset datasets/v2 [...] / train_detector.py --data <ds>/detector/data.yaml
   # 跨版本合并检测集：build_detector_dataset.py --dataset datasets/v2 --dataset datasets/v3 ...
        │
        ▼
【运行时产品 · Akagi-free】  majsoul_eye/recognize/  (TileClassifier / TileDetector)
```

三个内部阶段（单独跑/调试时才手动调）：`annotate_ai_session.py`（精确 fullwarp 几何 + GT 赋类，
**可跳过**——见 §2 建库）→ `build_dataset.py`（crops+yolo）→ `build_detector_dataset.py`（split）。
（`rebuild_datasets.py` 已删除（2026-07-05）——被版本化的 build_datasets 取代，见 §4。）

## 1. 数据目录与角色（单一真源 `majsoul_eye/paths.py`）

| 路径 | 角色 | 可再生? |
|---|---|---|
| `captures/raw/ai_session/run_N/` | **原始 GT + 帧（主采集路径产物）**：每局一个自包含目录 `gameM/`，内含 `gameM.jsonl`（GTRecord）+ 帧/线流/元数据 | ❌ 不可再生，唯一需备份的数据 |
| `captures/raw/manual/session5,6*` | 手动 F11 局（record_gt 产物，**采集方式已过时**；**不在现役 v2 训练集内**——v2 为纯 AI 基线；数据冻结存档） | ❌ 冻结存档 |
| `captures/intermediate/derived/` | 修复帧（裁 16:9 等历史遗留局）。去黑边 `*_fixed` 路径**已退役**——run_5 信箱局 2026-07-05 就地修复（`deletterbox_frames.py --inplace`），raw 即修复帧 | ✅ 由 raw 重建 |
| `captures/legacy/` | 归档的逐字节重复（ai_g*/ai_r1） | — 可删 |
| `captures/raw/temp/` | ⚠️ 无 GT 的孤儿帧（采集失败残留，只有 PNG 没有对局 jsonl）——不可用，待清理 | — 垃圾 |
| `datasets/<name>/`（版本目录，现役 `v2`） | 自包含数据集：`annotations/` + 每局 `<game>/{crops,yolo}` + `detector/`（+`--obb` 时 `detector_obb/`，不拷图 txt 引用）+ `games.json` 清单 | ✅ build_datasets.py |
| `out/ai_session_annotations/` | 旧全局标注位置（早期版本产物；现每个版本自带 `datasets/<name>/annotations/`） | — 可删 |
| `majsoul_eye/recognize/*.pt` | **正式权重**（tile_classifier.pt 入 git；tile_detector.pt 本地） | GPU 重训 |
| `weights/` | `pretrained/` 训练基座 + `detector/` 变体（aabb/obb 等，均 gitignore） | — |

规则：
- `frames.jsonl` 的 `file` 一律**相对路径**，读取永远经 `paths.resolve_frame_path`；
  帧目录与 GT 的耦合规则只定义在 `paths.frames_dir_for` / `paths.capture_for_frames_dir`：
  **嵌套（现行）** `X/X.jsonl ↔ X/`，兄弟（legacy，manual 会话仍用）`X.jsonl ↔ X/`——不要自己重推。
- **数据集/标注/装配全部是衍生物**（gitignored）：标注代码一变，它们就"过期"，用
  `build_datasets.py <name> --force`（或建新版本）重建，不要手工修补/移动——v1 的搬家就击穿过
  detector split 的相对路径（STATUS §1.22）。

## 2. 各阶段要点

### 采集（唯一主路径 = AI 自动）
- `autoplay_ai.py`：单 `auto` 环境。Playwright 抓 liqi WS + Mortal 决策点击 + 事件安静截图，
  **边打边写统一 GTRecord**（与旧手动格式同构，无任何转换步骤）。默认 **OBSERVE**（记录 AI 决策、
  不点击），确认后 `--live` 真打；`--dry-run` 正交，只观察/试跑而**不落任何盘**（run/game 目录、
  截图、GT/wire/index/metadata 全不写，settings 进临时目录退出即删）——冒烟测试浏览器/tap/AI 链路用，
  可与 `--live` 组合以走完整实弹流程而不存数据。`--auto-next` 结算续局循环；`--overlay`（连续）/
  `--overlay-manual --overlay-key`（按键单帧）浏览器内画检测框（验证用）；`--skins` 经 MajsoulMax
  MITM 换肤/牌背/桌布（训练外观多样性）；小号。
- 采集期已内建两类脏帧规避：发牌动画不 arm 截图（ActionMJStart/NewRound）、ROI 稳定确认
  （`capture/roi_diff.py`，防弃牌动画遮挡，实测残留 ~0.4%）。
- 每局写 `metadata.json`（显示语言 BCP-47，`--lang` > localStorage 探测 > 服务器粗判）。
- **已过时**：`record_gt.py` 手动 F11 + Akagi 路线（akagi 环境）。不再用于新采集；
  脚本保留只为存档复现 session5/6。

### 标注（annotate）
- `annotate_ai_session.py` 默认标注**全部** `paths.ai_captures()`；`--captures` 指定局；
  `--frames-dir` 可将某局指向另一帧目录（历史上用于 run_5 信箱局的 derived 修复帧，现已就地修复不再需要）；`--workers` 默认保守 4（RAM 束）。
- GT 谓词丢弃发牌窗帧（`replay.is_deal_window`：rivers 全空）；hero 摸牌槽经 `replay.drawn_tile`
  正确标注（14 张自摸态不再漏标）。
- 牌背（`back`）可靠性门是**去皮肤化**的：`pipeline.tile_live_mask`（饱和度或亮度
  `(S>60)|(V>110)`，任意肤色都判活）判定 dora/副露反面槽是否已渲染（fill 门），与
  `tile_back_mask`（纯饱和度 `S>70`，供 `snap_meld_strip` 做吸附阶段的 face/back 几何判别）
  是两个职责分离的 mask，互不影响（STATUS §1.33）。

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
  惯例 held-out：**整局 `ai_session_run_8_game1`**（分类器与检测器同一局，趋势可比）。
  `--val` 三处（`build_datasets.py` / `build_detector_dataset.py` / `train_classifier.py`）
  **均可重复**，多传即多留一整局作 val（如
  `--val ai_session_run_8_game1 --val ai_session2_run_21_game1`）；`games.json` 的 `val`
  字段随之为**列表**（旧单字符串仍被读端容忍）。只微调 val 无需重标/重裁：
  `build_datasets.py <ver> --stage detector --sources <与构建时相同> --val A --val B --resume`
  仅重组 detector split + 改写清单。**注意** `--sources` 必须与初次构建一致（脚本从 sources
  重新发现局，而非从清单读），否则如 `ai_session2` 局不被发现、`--val` 校验会报"未在已发现局中"。
- **多版本输入**：`train_classifier.py` 与 `build_detector_dataset.py` 均支持可重复的
  `--dataset datasets/<name>`（读 `games.json` 自动展开成逐局 `--data` 条目；同名局后者覆盖
  前者并打印提示；仍可混用显式 `--data`）。
- 分类器：`train_classifier.py --dataset datasets/v2 [--dataset ...] --val ai_session_run_8_game1:* --val ai_session2_run_21_game1:* --epochs 20`；多卡服务器/日常更推荐 `launch_classifier.sh --dataset v2 --gpu 0`（见下）——不传 `--val` 时自动读 `games.json` 的 val 列表，与检测器留出同一批整局。
- 检测器：`train_detector.py --data datasets/<name>/detector/data.yaml`（OBB 版本走
  `datasets/<name>/detector_obb/data.yaml`；imgsz 1280；16GiB 卡加 `--batch 4` +
  expandable_segments 防 OOM；OBB 用 `--model weights/pretrained/yolov8s-obb.pt`）。多卡
  更推荐 `launch_detector.sh`（见下）自动按 `--dataset <name>` 定位对应 split。
  **增强现为显式 CLI**（`--fliplr/--hsv-v/--hsv-s/--mosaic/...`，启动日志打印 `aug:` 行）：
  默认 `fliplr=0`（麻将牌有方向，水平翻转造镜像牌）、`hsv_v=0.5`（亮度/宝牌闪光近似），
  其余沿用 ultralytics detect 默认。是否加真·局部 bloom 由 `count_dora_glow.py` 覆盖统计决定。
  跨版本先合并 split：`build_detector_dataset.py --dataset datasets/v2 --dataset datasets/v3
  --val ai_session_run_8_game1:* --out datasets/detector_combined`。
- **HBB/OBB 格式**：`build_datasets.py` 默认只出 HBB（`--obb` 是显式开关）。三种：不给→HBB、
  `--obb`→仅 OBB（历史布局，每局仍在 `<ds>/<game>/yolo`）、`--hbb --obb`→**一个版本同时出**
  `detector/`+`detector_obb/`。双出时 OBB 落**兄弟目录** `<ds>/<game>__obb/yolo`，其 `images` **软链**
  回 HBB（OBB/HBB 帧字节相同，零重编码，只写 9 点标签，`build_dataset.py --reuse-images --no-crops`）；
  stage-2 先跑完 HBB 再跑 OBB（reuse 依赖 HBB 帧先落盘）。`games.json` 记 `formats` 字段；`dir` 仍存 HBB
  局名，OBB 目录＝`<dir>__obb`。已建的 HBB 版本可 `--hbb --obb --resume` **原地补 OBB**（跳过已验证的
  HBB 与标注，只增量建 OBB 标签＋重装两套 split，快）。
- 训练命令 `build_datasets.py` 收尾会按当前局清单打印好，直接复制。
- **GPU 服务器（多卡 DDP，bash，tar-and-go）**——两个脚本只需 raw 采集（run_5 信箱局已就地
  去黑边，raw 即修复帧，无需单独 rsync derived），**无需 MahjongCopilot、也无需已退役的 `intermediate/gt`**：
  - `scripts/data/regen_detector_dataset.sh [--obb|--obb-only] [--skip-annotate] [--jobs=N]`
    —— 在服务器上重建**扁平** `datasets/detector`（加 `--obb` 再出 `datasets/detector_obb`）。
    局发现复用 `build_datasets.discover_games`（嵌套 `paths.ai_captures()`，与版本化
    构建同源；AI 局；`SOURCES="root..."` 可改扫描根）。OBB 复用 HBB 帧、只写 8 点标签
    （`build_dataset.py --reuse-images`，不重编码 ~17G 帧）；缺帧的局（如帧未 rsync 齐）
    **大声丢弃、不中断**。
  - `scripts/train/launch_detector.sh {hbb|obb} --dataset <name> --gpus IDS` —— `train_detector.py`
    的单次训练包装：`--dataset` 选**版本化**构建目录（裸名→`datasets/<name>`，默认 `v2`；含
    `/` 直接当目录用；`*.yaml` 逐字当 data.yaml——兼容扁平 regen 布局），变体决定 split 子目录
    （HBB→`<ds>/detector`、OBB→`<ds>/detector_obb`）、基座与输出与 run 目录 `runs/<mode>/<ts>/`。
    输出为**版本化** `weights/detector/tile_detector_<mode>_<name>.pt`（`<name>`＝run 子目录，
    默认时间戳，各 run 不互相覆盖）；**OBB 是现役默认**，故额外把 best 复制到
    `majsoul_eye/recognize/tile_detector.pt`（运行时加载的那份，无需手动 promote）。
    卡用 `--gpus` 挑**物理 id**（`4,5,6,7`；单卡 `2,`；裸数 `N`＝卡 0..N-1）——**别用
    `CUDA_VISIBLE_DEVICES`**：ultralytics `select_device` 会用 `--device` 串覆写它。`--batch` 为
    跨卡全局 batch；默认 batch64/epochs60/imgsz1280，`--` 后透传。
  - `scripts/train/launch_classifier.sh --dataset <name> --gpu ID` —— `train_classifier.py` 的
    单次训练包装。分类器是小 CNN，**单卡无 DDP**，故用 `--gpu` 经 `CUDA_VISIBLE_DEVICES` 选卡
    （与检测器相反——这里 CVD 就是正确的开关）。不传 `--val` 时自动读 `datasets/<name>/games.json`
    的 `val` 列表、逐局 `--val <game>:*` 留出，**与检测器 split 留出同样的整局**（零手动同步）；
    输出 `recognize/tile_classifier.pt`，~几分钟。`--dry-run` 只打印将执行的命令。

## 3. SOP：新采集一个 run 后

```powershell
# 先自行 activate conda auto 环境；仓库根运行
$env:PYTHONPATH = "."        # bash: export PYTHONPATH=.
# A) 增量并入当前版本（日常推荐）：只处理缺的局，detector split + games.json 自动重组
#    （--resume 校验已存在局的 yolo 完整性与标签格式——截断/HBB↔OBB 混用的局自动重建
#      而非跳过；装配 detector split 前同一校验兜底，坏局报错拒绝装配。2026-07-05 加）
python scripts/data/build_datasets.py v2 --hbb --obb --sources captures/raw/ai_session captures/raw/ai_session2 --resume
# B) 建全新版本（标注代码变更后 / 要干净快照时）：
python scripts/data/build_datasets.py v3 --hbb --obb           # 默认 sources = captures/raw/ai_session
# （--force 清空重建同名版本；--dry-run 干跑；机器好加 -j 12 一把统管两阶段并行，
#   或分开写 --workers 16 --jobs 12 分别调标注/建库）
# C) GPU 训练（多卡启动器；确切命令 build_datasets 收尾已打印）
bash scripts/train/launch_classifier.sh --dataset v2 --gpu 0
bash scripts/train/launch_detector.sh hbb --dataset v2 --gpus 0,1,2,3
bash scripts/train/launch_detector.sh obb --dataset v2 --gpus 4,5,6,7
```

新 run 在 `--sources` 根下自动发现，无需登记；**游戏名按 source 根目录 basename 加前缀**
（`captures/raw/ai_session2/run_1/game1` → `ai_session2_run_1_game1`），故同一 run 编号跨不同源根
**不再撞名，无需跨源改号**；仅真重复（同一源根传两次）才直接报错。

## 4. 过时/降级组件清单（勿再当作管线环节）

| 组件 | 现状 |
|---|---|
| `scripts/data/rebuild_datasets.py` | **已删除（2026-07-05）**——被版本化的 `build_datasets.py` 取代（原地重建旧固定布局 vs 自包含 `datasets/<name>/`） |
| `scripts/capture/record_gt.py`（+ akagi 环境、Akagi MITM） | **过时的采集方式**。新数据一律 autoplay_ai；脚本保留仅为存档 |
| `scripts/data/convert_mjcopilot.py` | 降级为**共享转换库**（`convert_game` 被迁移器复用）；不再是管线一环。可独立 CLI 处理任何遗留 b64 线流 |
| `scripts/data/ingest_run.py` | **已删除（2026-07-06）**——遗留便捷入口（发现→建库），被 §3 的 `build_datasets.py` 完全取代（`test_downstream_rewire.py` 中两条相关断言一并移除） |
| `scripts/data/migrate_ai_to_gtrecord.py` | **已删除（2026-07-06）**——18 局 b64 → GTRecord 的一次性迁移已完成（2026-07-04）；如再遇遗留 b64 线流，转换能力仍在保留的 `convert_mjcopilot.py`（`convert_game` CLI） |
| `scripts/data/migrate_captures_layout.py` | **已删除（2026-07-06）**——一次性布局迁移已完成（2026-07-02） |
| `scripts/data/migrate_gt_into_gamedir.py` | **已删除（2026-07-06）**——GT jsonl 归入对局目录 + 改写 `datasets/*/games.json` 的一次性迁移已完成（2026-07-05） |
| `scripts/capture/backfill_skin_meta.py` | **已删除（2026-07-06）**——`--skins` 局 `metadata.json` 的一次性 hero-provenance 回填已对 ai_session2 完成（2026-07-05） |
| `scripts/data/purge_deal_frames.py` / `apply_deal_purge.py` / `purge_occlusion_frames.py` | 针对旧数据集的一次性清洗；现由采集期规避 + 建库期丢弃取代。全量重建后无需再跑 |
| `scripts/data/crop_game.py` / `deletterbox_frames.py` | 帧修复（session5 非全屏裁剪 / 信箱局去黑边）。`deletterbox_frames.py` 支持 `--inplace` 就地改写 raw 帧（run_5 game2/3 已于 2026-07-05 就地修复）或 `--out` 写 derived 副本。新采集全屏 1080p 用不到 |
| `scripts/annotate/spike_topdown.py` | 已归档的可视化 spike，不承重 |
| `captures/intermediate/gt/` | **已退役删除**（AI 采集直接写 GTRecord，无转换产物） |
| `label/`（`autolabel.py`） | 仅剩 hero 手牌+dora 框供 `annotate_frame` 调用；river/meld 旧几何已删 |
| `scripts/inspect/count_dora_glow.py` | **现役一次性诊断工具**（非管线环节）：统计每个 tile 类别的「发光实例/总实例」覆盖，判断是否需要为宝牌闪光加专门增强。读 GT 采集（Akagi-free），纯 stdout。见 `docs/superpowers/specs/2026-07-05-dora-glow-aug-design.md` |

## 5. 数据与权重现状快照（2026-07-06）

- **原始数据（纯 AI）**：`ai_session` 18 局（run_1, run_3×4, run_4×1(掉线), run_5×3(2 局曾信箱，
  已就地去黑边), run_7×1, run_8×6, run_13/14×1(早退迷你局)）+ 换肤 `ai_session2` 10 局
  （run_21×2 / run_22×4 / run_23×4）。早期手动 session5/6 已退出训练集（AI-only 基线）。
- **衍生数据**：**`datasets/v2/`**（28 局子文件夹 + `--hbb --obb` 双格式 `detector/`+`detector_obb/`
  ——OBB 局 `<game>__obb/` 软链复用 HBB 帧、零重编码；`annotations/` 缓存；`games.json` 清单）。
  held-out **两整局**：`ai_session_run_8_game1` + 换肤 `ai_session2_run_21_game1`。
- **正式权重**：
  - `recognize/tile_detector.pt`（HBB，2026-07-06 v2 重训）— best mAP50 **0.992** / mAP50-95 **0.957**。
  - `weights/detector/tile_detector_obb.pt`（OBB 变体，同日）— best mAP50 **0.994** /
    mAP50-95 **0.981**（rotated-IoU）。
  - `recognize/tile_classifier.pt` — **仍是 07-03 dealfix 权重**（held-out val_acc 0.9991），
    尚未在 v2 重训。
- ⚠️ **待办**：① 分类器在 v2 重训（`launch_classifier.sh --dataset v2 --gpu 0`，吃换肤外观多样性）；
  ② 换肤局 dora 牌背橙背门覆盖缺口（STATUS §1.31 遗留）。

## 6. 维护规约（每次改动必过一遍）

改动涉及以下任何一项时，**提交前必须**：

1. 问一遍："这会让 out/ 或 datasets/ 里的衍生数据过期吗？" 会 → 在 PR/commit 或 STATUS.md
   里写明"需 `build_datasets.py <name> --force` 重建（或建新版本）"，重大者当场重建。
2. 问一遍："这改变了管线的输入/输出/步骤/默认值吗？" 会 → **更新本文**对应小节
   （一图流 / 目录表 / SOP / 过时清单）。
3. 新增脚本必须归位：是管线环节（进 §0/§2）还是一次性工具（进 §4）？不允许"游离脚本"。
4. STATUS.md 追加一节记录（问题→处理→验证→结果），并刷新其 TL;DR 若数字变化。
5. 涉及数据格式/目录的，`majsoul_eye/paths.py` 是唯一真源——改那里，不改散落字面量。
