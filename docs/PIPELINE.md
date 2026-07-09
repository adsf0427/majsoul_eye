# majsoul_eye 数据管线（权威文档）

> **本文是"当前管线"的唯一权威描述。** 任何改动（脚本、数据格式、目录布局、默认参数）
> 只要影响 采集/标注/建库/训练 任何一环，**必须同步更新本文**（维护规约见 §8）。
> 历史沿革与实测结论见 [STATUS.md](STATUS.md)；设计论证见 [DESIGN.md](DESIGN.md)。
>
> 最后更新：2026-07-07（**HUD 标签三修 + v4 重建**，STATUS §1.47：`wall_count` 固定框 + GT 补零
> `余09`（旧 42px 收紧种子把数字截出了全部标签）；立直棒 fill 门限缩到声明窗口内（暗皮肤棒不再被当
> 背景训练，across/left 各救回 16.7%/13.8%）；按钮框改恒定 250×96 banner 点击区（跨语言不变）+
> 黏连候选上限拒绝。现役数据集 **`datasets/v4`**（71 局全量重建）。）
> 前次：2026-07-06（**HUD 支线与 dev 实验线合并**。HUD：annotate 出 `hud_boxes`（字段
> ink-snap + 按钮 op-GT 赋类 + count-mismatch 丢弃）、build 出 **56 类**（38 牌 + 17 HUD/按钮 +
> 立直棒，`majsoul_eye/hud.py`）YOLO + `hud/` 读取器训练对、新丢帧谓词 `is_call_window`、采集侧
> `--op-delay` + multi-shot extras（`status="extra"` 下游默认不可见、`dt` 字段）、新入口
> `train_hudreader.py`/`eval_detector_split.py`/`qa_hud.py`——代码全通，**HUD 训练待 v2 重建后跑**
> （STATUS §1.41）。dev 线：`launch_classifier.sh` 启动器、现役数据集 **`datasets/v2`**（28 局纯 AI，
> 源根限定命名）、检测器权重版本化 + OBB 提权现役默认、一次性脚本清理（STATUS §1.32–§1.39）。
> ⚠️ v2 建于 HUD 合并前：labels 为 38 类、无 `hud/`——**跑 HUD 训练前需以合并后代码重建 v2**。）
> 前次：2026-07-05（GT jsonl 归入对局目录 `run_N/gameM/gameM.jsonl`）；2026-07-04（采集统一
> AI 路线；`intermediate/gt` 退役；数据集版本化 `build_datasets.py`）。

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

   datasets/<name>/                      ←（现役：datasets/v4，71 局全量（HUD 标签三修后重建，STATUS §1.47）；
                                            v3 = 修复前全量（wall_count 截断/暗棒缺失，勿再训 HUD 类）；
                                            v2 = 28 局纯 AI 38 类旧 build）
     annotations/                        标注记录（AI 局；annotate_ai_session 产出）
     <game>/{crops/<38类>/,
             yolo/{images,labels}}       每局一个子文件夹（build_dataset 产出，合并后代码的
                                          yolo labels 为 **56 类** = 38 牌 + 17 HUD/按钮 + 立直棒）
     <game>/hud/{<字段>/*.png, labels.jsonl}  HUD 微读取器训练对（同一 build_dataset 产出）
     detector/{train.txt, val.txt, data.yaml}       按局切分装配（build_detector_dataset，nc=56；
                                          `--obb` 时另出 detector_obb/）
     games.json                          清单：每局 name/capture/frames_dir/dir + val（held-out 局列表）+ formats（hbb/obb）
        │
        ▼
【训练 · GPU, 手动触发 · 可吃多个版本 · 多卡用启动器 launch_*.sh】
   scripts/train/launch_classifier.sh --dataset v2 --gpu 0            # 单卡；自动读 games.json val
       → majsoul_eye/recognize/tile_classifier.pt   （38类, 正式）
   scripts/train/launch_detector.sh {hbb|obb} --dataset v2 --gpus IDS # 多卡 DDP
       → weights/detector/tile_detector_<mode>_<ts>.pt（每 run 版本化，不互相覆盖）
       → OBB 另复制一份到 recognize/tile_detector.pt（现役运行时默认）
   scripts/train/train_hudreader.py --dataset datasets/<name> --out majsoul_eye/recognize/hud_reader.pt
       → CTC 数字读取器 + round/wind 分类头（一份 checkpoint 三个子模型；⚠️ 待跑，需重建后的 56 类数据集）
   # 直调底层：train_classifier.py --dataset datasets/v2 [...] / train_detector.py --data <ds>/detector/data.yaml
   # 跨版本合并检测集：build_detector_dataset.py --dataset datasets/v2 --dataset datasets/v3 ...
   # 56 类检测器回归门槛 / 端到端 QA：见 §2「装配 + 训练」末尾
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
| `datasets/<name>/`（版本目录，现役 `v4` = 71 局全量、HUD 标签三修后重建（STATUS §1.47）；`v3` = 修复前全量（HUD 类标签有系统性缺陷，勿再训）；`v2` = 38 类旧 build） | 自包含数据集：`annotations/` + 每局 `<game>/{crops,yolo(合并后 56 类),hud/(读取器训练对)}` + `detector/`（+`--obb` 时 `detector_obb/`，不拷图 txt 引用）+ `games.json` 清单 | ✅ build_datasets.py |
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
- 截图经 CDP `Page.captureScreenshot` **clip 到 `browser_width×browser_height` 左上区域**
  （scale=1，native DPR 直出，如 1280×720@1.5→1920×1080）。Playwright 把布局视口/游戏 canvas
  钉死在该尺寸左上角，但 captureScreenshot 默认抓整个 OS 窗口 surface——窗口被拖大（且经
  persistent `user_data_dir` 记忆、后续启动复原为大窗）时右/下会多出纯黑边。clip 使截图与窗口尺寸
  无关。（此 fix 前的 `ai_session2/run_5` 已离线裁回 1920×1080。）
- 采集期已内建两类脏帧规避：发牌动画不 arm 截图（ActionMJStart/NewRound）、ROI 稳定确认
  （`capture/roi_diff.py`，防弃牌动画遮挡，实测残留 ~0.4%；2026-07-07 起为**多矩形**
  `STABILITY_ROIS`＝中央桌面＋三家手牌行＋hero 行，取 MAX——此前单矩形不含手牌行，
  会放行手切后理牌收缩中段的帧，见 STATUS §1.45）。
- `--op-delay LO HI`（默认 `0.5 1.0`）拉长 hero 收到待决操作后 AI 点击前的随机等待
  （覆盖 MahjongCopilot 的 `delay_random_lower/upper`），配合默认 `--quiet 0.30` 让
  quiet-debounce 截图能在动作按钮还在屏幕上时落盘——**按钮采集专用**（如
  `--op-delay 1.5 2.5`），HUD 按钮标定/训练数据靠这条路径补足（不再需要 record_gt 人工被动局）。
- **multi-shot extras**（默认开，`--no-multishot` 关；`capture/multishot.py` 的 `MultiShot`）：
  鸣牌（吃/碰/杠/拔北）与"按钮可能出现"这类**时序不确定**的事件之后，除常规
  quiet-debounce 帧外，再按固定偏移（默认 `0.6/1.2/2.4` s，`--multishot-offsets` 可改）
  额外截 `frames/{seq:06d}_dt{ms:04d}.png`，`frames.jsonl` 对应行 `status:"extra"`——纯**增量**，
  不影响既有 `"ok"`/`"timeout"` 消费者（下游默认不读取、不参与标注/建库）；每行（含 `"ok"`）都新增
  `dt` 字段（本次截图相对触发事件 `last_event_t` 的秒数），供将来"从多帧里挑最佳一张"的
  best-shot selector 使用（尚未实现，见 STATUS §1.41 的 OWNED FOLLOW-UP）。
- `--auto-next-debug`（**诊断专用，非数据集输入**）：`--auto-next` 循环卡死时开这个，每轮把当前
  端末帧 + 四个按钮 guard 的 frac/质心/预测分支（**全在同一帧上评估**）+ lobby menu_diff 存到
  `<run>/_autonext_debug/`（PNG + `autonext_debug.jsonl`）。产物在 `frames/`、`games.json` 之外，
  下游一律不读。用来定位 auto-next 把哪一屏误判成 rematch 对话框（见 STATUS §1.43）。
- 每局写 `metadata.json`（显示语言 BCP-47，`--lang` > localStorage 探测 > 服务器粗判）。
- **已过时**：`record_gt.py` 手动 F11 + Akagi 路线（akagi 环境）。不再用于新采集；
  脚本保留只为存档复现 session5/6。

### 标注（annotate）
- `annotate_ai_session.py` 默认标注**全部** `paths.ai_captures()`；`--captures` 指定局；
  `--frames-dir` 可将某局指向另一帧目录（历史上用于 run_5 信箱局的 derived 修复帧，现已就地修复不再需要）；`--workers` 默认保守 4（RAM 束）。
- GT 谓词丢弃发牌窗帧（`replay.is_deal_window`：rivers 全空）；hero 摸牌槽经 `replay.drawn_tile`
  正确标注（14 张自摸态不再漏标）。**新增** `replay.is_call_window`（`last_event` 为
  chi/pon/daiminkan/ankan/kakan/nukidora——鸣牌动画中途，GT 已更新但像素未跟上，与
  `is_deal_window` 同策略整帧丢弃，在 `annotate_ai_session`/`build_dataset` 都生效；
  run_3/game1 实测丢弃率 ~4.2%，全部单帧命中、无过匹配）。
- `annotate_frame` 新增 `hud_boxes`（`majsoul_eye/annotate/hud.py`）：数值字段（四家分数/
  供托/本场）按标定种子 ROI（`coords.HUD_SEEDS`）做逐帧**墨迹收紧**（ink-snap，
  亮度阈值 `INK_THRESH=120`，无墨迹即标 `reliable=False`）；`round_label`/`seat_wind_self`
  定尺寸不收紧；**`wall_count` 为固定框**（客户端补零两位 ⇒ 恒宽恒位，GT 文本亦补零 `余09`，
  只在余字子区探测是否渲染——旧 42px 收紧种子曾把数字截出全部标签，STATUS §1.47）。
  按钮框：`state.pending_ops`（`state/ops.py` 从 `raw_liqi.data.data.operation.
  operationList` 提取）经 `hud.buttons_for_ops` 得到期望类别集合，与 `BTN_ZONE` 内定位到的
  候选按 x 序一一对应，发出的框是**恒定 250×96 banner（实际点击区，跨显示语言不变）**而非
  文字字形框；候选超尺寸（`BTN_MAX_W/H`，并排 banner/特效黏连块）直接拒绝；
  **检出数 ≠ 期望数则整帧按钮标签丢弃**（`flag:count_mismatch`，
  宁缺毋滥，与旧 river/meld 门同哲学）。立直宣言/分数滚动窗口（`replay.is_score_anim_window`）
  只把 HUD 框标记不可靠、不丢整帧（牌面标签不受影响）；立直棒的逐框亮度 fill 门也**只在该窗口内**
  生效——settled 帧一律信 GT（暗色皮肤棒 fill 常年 <0.35，无条件门曾把 across/left 槽 16.7%/13.8%
  的棒当背景训练，STATUS §1.47）。
- 牌背（`back`）可靠性门是**去皮肤化**的：`pipeline.tile_live_mask`（饱和度或亮度
  `(S>60)|(V>110)`，任意肤色都判活）判定 dora/副露反面槽是否已渲染（fill 门），与
  `tile_back_mask`（纯饱和度 `S>70`，供 `snap_meld_strip` 做吸附阶段的 face/back 几何判别）
  是两个职责分离的 mask，互不影响（STATUS §1.33）。
- 副露角点（`pipeline.MELD_STRIP2`）2026-07-08 重标定（pos3 沿 along +45.5，消除半张牌 aliasing
  失锁，STATUS §1.50）；**`scripts/annotate/meld_snap_qa.py`＝逐座锁错率守卫**，warp/mask/角点改动后必跑
  （<8% 锁错，否则副露框可能整张翻）。
- 副露框放置现走**按局共识**（`annotate.meldsnap.game_meld_overrides` → `annotate_frame(meld_snap_override=)`，
  STATUS §1.51），build 因此**两遍扫描**（先 measure 每帧 snap → 每局共识 → 再 annotate）取代逐帧各自吸附；
  guard `scripts/annotate/meld_consensus_qa.py`（断言同局同座只有唯一偏移，非唯一即回归）。低置信局（样本
  不足或无主簇）meld 框标 `reliable=False`，不硬套不可信偏移。

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
- **`yolo/images` 走 copy 快路径**（STATUS §1.46）：帧未经 resize 且源 PNG 是 8-bit RGB 时直接
  `copyfile` 源帧（跳过 ~100ms/帧 的 PNG 重编码，逐像素等价）；resize 过或非纯 RGB 源回退 `imwrite`。
- **HUD**：同一份 `rec["hud_boxes"]` 出两种产物：所有 `reliable` 框追加为 YOLO 行
  （**56 类** = 38 牌 + 17 HUD/按钮 + 立直棒，`majsoul_eye.hud.DET_NAMES`；旧的 38 类数据集天然是
  56 类标签空间的子集，可与新数据混训）；带 `text` 的数值/`round_label` 字段额外产出
  读取器训练对 `<out>/hud/<字段>/<seq>.png`（15% 内边距、按 `hud.FIELD_ROT` 转正）+
  `<out>/hud/labels.jsonl`（每行 `{"file","name","text","pad"}`）。按钮无 `text`——类别本身即标签。

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
  Windows 无软链权限（无开发者模式/未提权，`WinError 1314`）时 `Runner.symlink` 自动回退**目录 junction**
  （免权限、`glob` 透明穿透；但存绝对目标 → 该版本宿主本地，换机重建，见 STATUS §1.42）；
  stage-2 先跑完 HBB 再跑 OBB（reuse 依赖 HBB 帧先落盘）。`games.json` 记 `formats` 字段；`dir` 仍存 HBB
  局名，OBB 目录＝`<dir>__obb`。已建的 HBB 版本可 `--hbb --obb --resume` **原地补 OBB**（跳过已验证的
  HBB 与标注，只增量建 OBB 标签＋重装两套 split，快）。
- **对手牌背（实验，默认关）**：`build_datasets.py --backs` 额外标注三家对手暗牌行的 `back` 框
  （手摸切识别的前置；`majsoul_eye/annotate/backs.py`，标定于 run_8 真实帧，fullwarp 均匀 pitch +
  玩家左手端锚点 + 副露收缩 bias，跨皮肤/分辨率已验证）。透传路径：`--backs` →
  `annotate_ai_session.py --backs`（记录多出 `back_boxes`）→ `build_dataset.py`（YOLO 出 `back`
  类框、**不出分类器 crop**；任何带 `backs_holding` flag 的帧——该座位摸牌中、理牌插槽位置
  GT 不可知——**整帧丢弃**保证 back 信号一致，实测约丢 40–56% 帧）。⚠️ 勿混入主线 v1/v2 版本，
  用独立版本名（样例：`datasets/backs_sample/`，单局 68 帧，`detector/`(HBB)+`detector_obb/`(OBB)
  双 split；`fiftyone_view.py` 已支持 9 字段 OBB 标签渲染（Polylines），侧座建议看 OBB——HBB 会把
  倾斜 quad 坍缩成大幅重叠的轴对齐盒）。几何来源＝**人工 per-slot 模板**：`scripts/annotate/
  calibrate_backs_manual.py` 生成自包含 HTML 标注页（auto env 是 headless OpenCV，无 cv2 GUI；
  浏览器里滚轮缩放逐张点 4 角，下载 JSON 到 `out/backs_calib/`），`--ingest` 合并校验并生成
  `majsoul_eye/annotate/_backs_manual.py`（勿手改）——13 槽/座 + 摸牌槽，fullwarp quad；
  副露不重排（副露行＝前 row_n 个模板槽原位，STATUS §1.49）；holding 座位已标注（静止 n-1 行 + 摸牌槽，
  §1.48）。多局多皮肤 diversity 审查集：`scripts/inspect/build_backs_review.py`（回放扫全部 AI 局 → 按状态
  签名去重、稀有态[各座副露/听牌/摸牌]优先、跨局散布 → 只标选中帧，出扁平 `<game>__<seq>` HBB+OBB →
  `datasets/backs_review/`；`fiftyone_view.py --data datasets/backs_review/obb/data.yaml` 浏览）。
- **HUD（⚠️ 三条均待跑——先用合并后代码重建 v2（旧 v2 是 38 类、无 `hud/`），再执行）**：
  - 读取器：`train_hudreader.py --dataset datasets/<name> --out majsoul_eye/recognize/hud_reader.pt`
    ——CTC 数字读取器 + round/wind 分类头，一份 checkpoint 三个子模型；held-out 按
    `games.json` 的 `val`（**整局**，非按 kyoku——HUD 字段没有 kyoku 粒度 GT）。
  - 56 类检测器回归门槛：`eval_detector_split.py <weights> <data.yaml>` 按 id<38（牌）/≥38
    （HUD）分组报 mAP50；牌面组门槛 `0.993 − 0.005`，不达标即退回独立 HUD 检测器（spec §6）。
  - 端到端 QA：`qa_hud.py <game.jsonl>`（真实用法需 56 类检测器 + 读取器权重都到位；`--selftest`
    用假检测器/假读取器单独验证组装/比对逻辑，不需要任何权重）——按字段打印读取精确匹配率
    + 整帧全字段全对率。
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
| `scripts/eval/eval_reconstruction.py` | **QA 工具**（非管线环节，局面复原验收）：三层评测——oracle（GT `BoardState` → `ObservedState` → `reconstruct` → `Replayer` 往返一致性，无 GPU 依赖）/ assemble（真实帧 → `TileDetector` → `assemble` 装配 vs GT 投影，按 zone 报错 + 拒收帧按 violation 类别计数 `rejected_reasons`）/ engine（真实 mjai 前缀 vs 复原序列各喂 `--engine-cmd` 指定的任意 mjai bot，比较最终决策，stdin/stdout JSON lines 契约；`{seat}` 占位符按各序列 `start_game` 的 hero id 实例化——复原序列无 HUD 时 hero 恒在绝对座位 0，与真实序列不同）。oracle 在全量 `captures/raw/ai_session` 上验收 ≥99%（实测见 STATUS §1.52）。spec: `docs/superpowers/specs/2026-07-05-board-reconstruction-design.md` |
| `scripts/eval/mortal_stdin.py` | **QA 辅助工具**（非管线环节）：mjai stdin/stdout 包装 `../auto/mycv` 的 Mortal（version=4 b24c512，`mortal.pth`，cpu），供 `eval_reconstruction --level engine --engine-cmd "python scripts/eval/mortal_stdin.py {seat}"` 用。非 shipped 识别器组件，允许触及 sibling repo |
| `scripts/recognize/recognize_frame.py` | **🔁 现役工具**（非管线环节，运行时识别链路的 CLI 入口）：截图 → `TileDetector`+`assemble`+`reconstruct` → JSON lines（ObservedState + 合法 mjai 序列 + fabricated 说明；拒收帧给 violations）。`--weights` 默认取 `weights/detector/tile_detector_obb_*.pt` 最新者；`--letterbox`/`--no-reconstruct`/`--pretty`。Akagi-free，供外部调用/快速检视 |

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
