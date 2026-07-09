# majsoul_eye — 项目状态与路线图

> 活文档：**已完成的部分 + 未来计划**。当前管线的权威描述在 **[PIPELINE.md](PIPELINE.md)**；
> 设计与论证见 [DESIGN.md](DESIGN.md)。
> 最后更新对应进度：**局面复原 M1 merge 回 main**（§1.52，2026-07-09；observe/assemble/
> reconstruct/eval 四模块 + 三层评测，oracle 10063/10063）；
> 此前：**HUD 检测（分数/场风/局/余牌/供托/本场/动作按钮/立直棒）代码落地——56 类
> detector taxonomy + 微读取器 + 按钮/字段标定 + 采集侧 multi-shot/`--op-delay`/
> `is_call_window`，代码全通、训练待 v2 重建后执行**（§1.41，2026-07-06，本地线；立直棒真帧
> 标定 T17b 同日追加）；
> 与 dev 线合并（2026-07-06）：检测器权重版本化 + OBB 提权现役默认（§1.39）；分类器启动器
> `launch_classifier.sh` + 现役切 `datasets/v2`（28 局纯 AI）+ 一次性脚本清理（§1.38）；
> `--hbb --obb` 双格式（§1.37）；多整局 val（§1.36）；`launch_detector.sh` 切版本化布局（§1.35）；
> run_5 就地去黑边 + 撤 `FRAMES_OVERRIDE`（§1.34）；back 门去皮肤化（§1.33）；源根限定命名
> 解除跨源 run 编号唯一性（§1.32）；检测器增强显式化 + 宝牌闪光统计（§1.30）；
> 采集截图黑边 CDP clip 修复（§1.40，2026-07-05，本地线）；
> 此前：服务器侧 `regen_detector_dataset.sh` 切嵌套布局 + 训练启动器文档（§1.29）；skins 元数据
> hero 修正 + ai_session2 回填（§1.28）；采集统一 AI 路线 + 数据集版本化（§1.19–§1.22，2026-07-04）。
> （近期里程碑：dealfix 分类器 val 0.9991 §1.16、OBB 检测器 mAP50-95 0.9804 §1.17。）

## TL;DR

- 管线**只有一条主路径**（零手绘标注，版本化构建 `build_datasets.py <name>`，现役 `datasets/v2`）：
  `autoplay_ai(AI 自动对局, 实时写统一 GTRecord) → 精确标注 → build_dataset(crops+YOLO)
  → detector 装配 → 训练(launch_classifier.sh / launch_detector.sh；--dataset 可混多版本)`。
  手动 F11(record_gt+Akagi) 采集**已过时**，session5/6 已退出训练集（AI-only 基线）。
- 数据：现役 **`datasets/v2` = 28 局纯 AI**（18 ai_session + 10 换肤 ai_session2），`--hbb --obb`
  一次出双格式，held-out 两整局（`ai_session_run_8_game1` + 换肤 `ai_session2_run_21_game1`）。
- 模型：检测器 2026-07-06 v2 重训 HBB **mAP50 0.992 / mAP50-95 0.957**、OBB 变体 **0.994 / 0.981**
  （rotated-IoU）；分类器**尚未** v2 重训（07-03 权重 val_acc 0.9991；轨迹 93.5→…→**99.91**）。
- 核心论点已实证：`协议GT(WHAT) + 标定几何(WHERE) = 免费且正确的标签`，精度随对局数单调上升
  （mycv 基线实测见 §1.5：分类近乎已解决，瓶颈在检测/时序——由此定的路线已逐项兑现）。
- ⚠️ 待办：分类器在 v2 重训（`launch_classifier.sh --dataset v2 --gpu 0`）；换肤局 dora 牌背橙背门覆盖缺口（§1.31 遗留）。

---

## 一、已完成

### 1. 采集与 GT 旁路（P1–P2，dev-only，Akagi 耦合）
| 组件 | 文件 | 说明 |
|---|---|---|
| GT 录制器 | `majsoul_eye/capture/akagi_tap.py` | monkeypatch `MajsoulBridge.parse_liqi`，每条消息记录 raw-liqi 超集 + MJAI 事件 + `seq`，后台线程写 JSONL |
| 记录格式 | `majsoul_eye/capture/schema.py` | `GTRecord` + JSONL 读写（bytes 安全） |
| 抓窗 | `majsoul_eye/capture/screen.py` | win32 找窗 + mss 抓客户区，退化整屏 |
| 截图同步 | `majsoul_eye/capture/sync.py` | **debounce-to-quiet**：事件安静 `quiet` 秒才截 + **画面稳定确认**；不阻塞 MITM 线程；**用全局 `seq` 标帧**（非每局重置的 `last_op_step`） |
| 启动器 | `scripts/capture/record_gt.py` | 注入录制器后跑 Akagi；`--screenshots`；启动前 `loguru.remove(0)` 保护 TUI |

### 2. 状态重放（P1，纯净、无 Akagi 依赖）
| 组件 | 文件 | 说明 |
|---|---|---|
| 重放器 | `majsoul_eye/state/replay.py` | MJAI 事件 → 完整四家 `BoardState`（河含 摸切/立直横置/被鸣标记、副露、dora、手牌、各家暗牌数、分数、场况）；`check_invariants()`；**真实数据 0 违例** |
| 类别体系 | `majsoul_eye/tiles.py` | 统一 38 类 taxonomy + MJAI 互转（从 mycv 固化） |

### 3. 几何自动标注（P3–P5）
| 区域 | 文件 | 方法 | 质量 |
|---|---|---|---|
| 坐标模型 | `majsoul_eye/coords.py` | 归一化 `NormBox`、手牌槽、`RIVER_QUADS`(4家透视网格)、`MELD_STRIPS`、`DORA_STRIP` | — |
| 板面定位 | `majsoul_eye/normalize.py` | `locate_fullscreen`/`locate_letterbox`/`AnchorLocator`(TODO) | 全屏 16:9 ✓ |
| 易区 | `majsoul_eye/label/autolabel.py` | 确定性 ROI 裁剪 + GT；手牌/dora/分数/文本 | 手牌 99.8% on-tile ✓ |
| 河(难区) | `majsoul_eye/label/river.py` | **每家透视网格(单应) + GT 顺序赋类** | **98.5–99.5% on-tile** ✓ |
| 副露 | `majsoul_eye/label/meld.py` | 每家 1 行 strip + GT 顺序；ankan 2 背面 | self/left ~95–98%，**right/across 3D 侧家偏松(88–93%) → opt-in**；几何重写 spike 见 §1.9（未并） |
| 默认区 | `autolabel.DEFAULT_ZONES` | = `{hand, river}`（meld/dora/score 为 opt-in） | — |

> **校准方式**：4 家河 quad + dora + 4 家副露 strip 均由**并行 subagent**在满盘帧上"看图返回坐标"标定（不改代码、可并发）；代码任务串行。

### 4. 数据集与识别模型（T6）
| 组件 | 文件 | 说明 |
|---|---|---|
| 数据集构建 | `scripts/train/build_dataset.py` | 同步采集 → 分类裁剪 `crops/<牌>/` + YOLO `yolo/images,labels/`；按 `seq` join；亮度门 `--min-bright` 丢空格 |
| 牌分类器 | `majsoul_eye/recognize/classifier.py` | `TileNet`（64px，AdaptiveAvgPool，输入尺寸无关）+ `TileClassifier` 推理封装 |
| 训练 | `scripts/train/train_classifier.py` | 多局 `--data NAME=crops:capture` + 跨局/跨场 split `--val NAME:kyoku|*`；类均衡采样 + 轻增广 |
| 权重 | `majsoul_eye/recognize/tile_classifier.pt` | 2 局训练，**val 93.5%** |
| 非全屏修复 | `scripts/data/crop_game.py` | 裁回 16:9 游戏画布（session5 实证 99.5% 对齐） |
| 对账/可视化 | `scripts/inspect/inspect_capture.py`、`scripts/inspect/overlay_labels.py` | 帧↔GT join、覆盖率、坐标叠加调试 |

### 1.5 mycv 基线实测（2026-06-27，dev-only，6 路对抗审查已验证）
**目的**：早先"mycv 64%/56%/35%"是管线错配产物（已撤回）；这次跑 mycv **真实 native 管线**测公平精度。
| 组件 | 文件 | 说明 |
|---|---|---|
| 引擎适配器 | `majsoul_eye/baselines/mycv_engine.py` | 直接 import 真实 `myCV`（../auto/mycv，pyautogui/matplotlib 在 auto 环境可用），调它真实的 `cutPic/model/getType/getHandTiles`；自己驱动座位（raw mask k → 绝对座 `(hero+k)%4`） |
| 评分 | `majsoul_eye/baselines/score.py` | 多重集(bag)匹配：`recall`(=端到端=correct/n_gt)、`precision`(=correct/n_pred)。不依赖我们的坐标标定，对 mycv 公平 |
| 测量脚本 | `scripts/inspect/mycv_baseline.py` | 重放 GT → 逐帧跑引擎 → 聚合；`tests/test_mycv_baseline.py` 9/9 |

**结果（真实 mycv，~1029 帧，2 局，bag-matched vs Akagi GT）**：

| 区域 | recall (端到端) | precision (找到时类对) |
|---|---|---|
| 手牌 | 95% | 98–99% |
| 牌河（3 对手） | 96% | **99.7–99.8%** |
| 副露 | 94–97% | 97–99%（small-n 噪声大） |

**经对抗审查修正后的结论**（措辞很重要，原始 headline 高估了）：
- ✅ **分类器近乎已解决**：对手河/手牌上，找到的牌**precision ~98–99.8%**（座位映射、类名口径均验证无误：错误映射置换会塌到 0.21–0.53；s5 hero=0 / s6 hero=1 都 ~96% 证明映射对）。
- ⚠️ **`precision` ≠ "分类准确率"**：欠检时它数学上就是 precision，且 bag 匹配会掩盖同家内互换错误（上界 FN+FP ~3–4%）。视为偏上界的 precision。
- ⚠️ **端到端差额 ≈ 欠检（recall）**：① 牌河差额约**一半**是抓帧时序——**GT 比画面领先约 1 个动作**，最新弃牌还在动画里（settled 帧 river ~.986，fresh 帧 ~.947）；池化后"最新弃牌"只解释 **51%**（非 session6 单局的 69%）的河漏检，剩余 ~2% 是 mycv 真实漏检。② **手牌差额几乎全是 mycv 真实欠检**（`getHandTiles` floodFill），**与时序无关**（settled-13 与 fresh-14 同分）。
- ⚠️ **范围**：mycv **不识别自家河**（座位掩膜涂掉，~25% 河体积），故"牌河"=3/4 家；对任何需识别全 4 河的系统**不是同口径对照**。副露是"crop 分类器 precision"代理，非 mycv 真实副露管线。
- ⚠️ **GT 领先画面**这一条对**我们自己**最关键：它在 build_dataset 阶段把最新弃牌 crop **标错**（标了画面没渲染的牌）→ 见 §五 P1。

### 1.6 P1 完成：修抓帧时序污染 → 清洗标签 → 95.3%（2026-06-27）
mycv 基线 §1.5 暴露的"GT 领先画面 ~1 动作"在**我们自己的 build_dataset 里把最新弃牌 crop 标错了**。实测+修复：
- **诊断**（用 mycv 作像素 oracle）：**最新弃牌 cell 13% 是空毡/半渲染却被标成牌**（vs 旧弃牌 1.2%，~10×）。
  根因：旧 `--min-bright 95` 门**抓不到蓝毡**——蓝毡均值亮度 ~100–119 > 95，空 cell 直接通过被存成误标 crop。
- **修复**：新增 `majsoul_eye/label/quality.py` 的**牌面占比门**（near-white 像素占比；真牌 0.58–0.79，毡 ~0；阈值 0.35），
  接入 `build_dataset.py`（`--min-face-frac`）。`tests/test_quality.py` 5/5。
- **重建**：session6 10045→9868、session5 29982→29778（共丢 381 张误标）。重训 → **clean val 95.3%**。
- **A/B 归因**（同一 clean val 上比新旧模型）：旧 model 0.9488 / 新 model 0.9527 → **训练清洗真实收益 +0.4**；
  另 +1.4 是**测量修正**（旧 93.5% 是拿 38 张空毡误标 val crop 在罚模型）。两者都真实但含义不同。
- **更大价值是前向的**：门已修，今后每次采集（路线图→多采局）不再注入这类误标 crop；门用白占比而非亮度，更鲁棒。
- 产物：`majsoul_eye/recognize/tile_classifier_clean.pt`（在新旧 val 上都**严格优于**旧权重；建议提升为正式权重）。

### 1.7 P2 完成（部分）：河格 erode → river 93.7→94.8、overall 95.3→96.0（2026-06-27）
mycv 基线启发的"借轮廓孤立"实验，**经混淆矩阵诊断被否定**——河错误不是邻牌*点子*渗入，而是**格子几何 + 数据稀缺**：
- **诊断**（clean 模型，held-out s6 河 crop）：river 93.7%。主错 **3s→2s ×27**（占河错 25%）= 格子偏低、下家牌红色边渗入糊住第3根；S→5m = 侧家 3D 透视；红五 0% = 数据稀缺。
  廉价验证：把 crop 底部裁掉，现有模型 3s 0.39→0.98 → 确认是底/边渗入，**不是 mycv 式白底孤立能修的**。
- **修复**：`coords.NormBox.erode()` + `build_dataset --river-erode-bottom 0.18 --river-erode-side 0.08`（已设为默认）。只对 river-zone 生效。
- **GPU 重训验证**（RTX 5080，torch 2.11+cu128，~4s/epoch）：river **93.7→94.8**，**3s 0.391→0.978、4p 0.44→1.0、侧家 S 0.89→0.99**；overall **95.3→96.0**。2s→5s、8m→9m、红五 不变（另因）。
- **产物**：`tile_classifier_erode.pt` 已**提升为正式** `tile_classifier.pt`（旧权重备份 `tile_classifier_prePollutionFix.pt`，另存 `_clean.pt`）。
  ⚠️ **运行时识别器必须对河格应用同样的 erode**（与训练一致）。
- **结论**：分类几乎到顶；河 94.8→99% 的剩余差距是**数据**（红五、distinct 牌、侧家透视样本），不是预处理 → 见路线图"多采局"。

### 1.8 接入 MahjongCopilot 半自动采集 → 97.6%（2026-06-27）
手动 F11 采集太慢；接入 MahjongCopilot（autoplay + MITM）批量产数据。其格式与手动不同：
- `frames.jsonl` 是**原始 liqi WS 线流**（b64 protobuf），PNG 是 **1080p**、按 liqi 消息 `seq` 命名。
- **离线转换器** `scripts/data/convert_mjcopilot.py`（dev-only，MahjongCopilot GPL，跑 `auto` 环境/protobuf 4.25.3）：
  `LiqiProto.parse(wire)` → `GameState.input`（配 stub bot，把 libriichi 的 `bot.factory` stub 掉）→ MJAI → 我们的 `replay.py`。
  两个关键点：① GameState 按 bot 决策点批量产 MJAI，所以按**每条 input() 的增量**打 seq（逐动作对齐，98.8% 帧命中）；
  ② **GameState 原地改 AI 手牌 list**，所以捕获时必须 **deepcopy 每个事件**（否则 start_kyoku.tehais 被后续覆盖→英雄手牌 desync）。
- **教训**：起初英雄手牌 17% 违例，一度误判为 MahjongCopilot 转换 bug；实为**我的捕获存了可变引用**。deepcopy → **0% 违例**。MahjongCopilot 转换是对的。
- 4 局转换（座位 0/3/1/1，多样）→ build_dataset（P1 门+P2 erode）→ **~73k crops**，红五 286/357/246。crops 已肉眼抽查正确。
- **GPU 重训**（RTX 5080）公平验证：session6 held-out **0.9604→0.9755**（+1.5，无泄漏）；held-out AI 局 ai_g1 **0.9619→0.9851**（+2.3）。8m 0.72→1.00、3s 0.74→0.97 等大涨。
- 产物：`tile_classifier_allgames.pt` 已提升为正式 `tile_classifier.pt`（备份 `_preAI.pt`）。
- **修正**：早先"红五 0%"是 session6-val small-n（7-9 样本）假象；规模够时模型本就 ~85-100%。AI 数据的真实收益是**跨源泛化**。
- **工具**：① `scripts/data/ingest_run.py <run_dir> [--train --val NAME:*]` —— 一键 发现游戏→convert→build_dataset(→可选重训)，自动发现单局/多局布局。
  ② `scripts/inspect/visualize_failures.py --crops ... [--val-capture --val-kyoku] --out DIR` —— 按混淆对(gt→pred)出错例蒙太奇 + summary.txt。
  实测 session6 held-out（97.6% 模型）主错：**红五跨花色**(5mr/5pr→5sr)、2s→3s、个别难牌；多数"错误"是同一物理牌跨~N 近重复帧。

### 1.9 副露几何 spike：单桌面单应 H_table + 三种杠（2026-06-29，**纯可视化验证，未并管线**）
针对 §3 痛点（副露 **right/across 3D 侧家 strip 偏松 88–93%，opt-in**）做的几何重写 spike，对应路线图"副露精修"。
**思路**：一个**单桌面平面单应 `H_table`**（在河上拟合，存 `scratchpad/H_table.npy`）统一覆盖所有**共面**元素
（4 家 副露/河/立直/杠），取代每家独立 strip。**GT 驱动赋类**（非检测——副露密排 + 背面牌 defeat 亮度检测）。
新 `captures/ai_session`（MahjongCopilot 原始线流 + 1080p PNG，13 局/多 run）解锁了 **session6 没有的 大明杠/加杠** 验证；
`H_1080 = scale(0.5)·H_4k`（两者皆全屏 16:9 同板，box-y 比实测 0.497 → 精确 2×）。

**每家列布局模型**（本 session 定，经 labeled-slot + **原始牌面 ground-truth** 核对）：

| 座 | order | anchor | corner | ly | lift(dx,dy)px |
|---|---|---|---|---|---|
| self | reverse | end | -8.10 | 8.67 | (0,-18\*) |
| right | reverse | end | -7.50 | 9.01 | (0,-14) |
| across | chrono | end | -8.76 | 7.65 | (0,-11) |
| left | reverse | end | -7.20 | 9.05 | (0,-14) |

> 统一规律：**self/right/left = reverse + anchor=end**（锚定**首副露**于固定近角 → 单/多副露帧用同一 corner 对齐）；across 唯一 chrono。值绑定当前 river-fit H_table。

**三种杠**（与玩家权威写法核对一致）：
- **大明杠**：4 落地牌，1 横置；位置编码来源（上家=横最左 / 对家=横中间 / 下家=横最右）——共面，被列模型覆盖。
- **暗杠**：`[背][正][正][背]`（两端背面，中间正面）——共面。
- **加杠**：3 落地槽（= 原碰）+ 1 张沿**桌面法向**堆叠的加牌 —— **唯一离面元素**，H_table 单独放不了（需投影桌面法向；当前仅 self 标定堆叠方向）。

**z 视差 lift**：H_table 映**桌面**(z=0)，牌面在桌面上方一个牌厚 → 落地中心投影偏低 ~14px。按座加常量屏幕 lift 校正；
量级随牌像素大小（相机距离）：across(远)最小、侧家中、self(近)最大——与视差预测一致。

**本 session 两个根因 bug（证据驱动，非眼测猜测）**：
1. 侧家原 `order=chrono` → 必须 **reverse**（列从错误端构建；labeled-slot + 原始牌面核对发现 right 把 daiminkan 中 放到了底部，实际在顶部）。
2. 侧家 `anchor=start` 把**末副露**锚到固定角（位置随副露数漂移）→ 单副露帧落到空毡；必须 **anchor=end**（锚首副露，使单/多副露帧同 corner 对齐）。

**产物 / 状态**：spike 代码全在 `scratchpad/`（`meld_ai.py` 主验证、`H_table.npy`、各 sweep/diag 脚本）；验证图在
`fails/s6val_boxes/`（`ROOTCAUSE_raw_tiles_*`/`ROOTCAUSE_diag_slots_*_FLIPPED`/`FINAL_*`）。已验证：右(大明杠+碰、单加杠、吃+大明杠)、
左(碰+碰+加杠)、对家(暗杠、大明杠)，box 居中贴面。**未并入** `meld.py`/`coords.py`（按要求先看可视化效果）。
**残留**（均亚牌级，可选）：① left 长列 pitch 微漂（末牌略欠，0.82→~0.80 可收）；② self lift 未在 1080p 验证；③ kakan 堆叠 lift 方向仅 self 标定。

### 1.9b 全图俯视 spike：单 H_table rectify 整桌 + 4 重对称 + 加杠共面（2026-06-30，可视化，未并）
§1.9 的 H_table **只在河上拟合** → 桌边/远家漂；且把加杠当**离面 z-lift** → box 浮到空毡（见 `fails/bbox_demo/C_kakan_single_zoom`、`D_kakan_multi_zoom`）。本 spike 改做用户提的"把整张图 warp 成俯视"路线：
- **脚本** `scripts/annotate/spike_topdown.py`（committed，不像丢失的 scratchpad；`CASES` 字典固化验证 seq；`--list-seqs/--grid/--warp/--all-cases/--check-symmetry`）。输出 `fails/topdown_demo/`（gitignored，可再生）。验证数据：`captures/ai_g1`(大明杠/暗杠/吃/碰)、`ai_g3`(**加杠**) 1080p。
- **H_table**：从**蓝毡方框 4 角**(`PLAY_CORNERS_NORM`)拟合，覆盖整桌（修 §1.9 只拟合河的根因）；归一化存储，相机固定→常量、跨分辨率通用（缩放被 `BoardRegion` 吸收）。
- **河**：4 家全部由**可信的 self quad** `RIVER_QUADS["self"]` →rect→旋转 90°·k→映回 派生。cell 贴面、**远家漂移消失**（远胜 4 家独立 quad）。`ROT_SIGN=-1`（+1 会静默左右家互换——经副露暴露）。
- **加杠 = 共面**（证实用户判断，否定 §1.9 z-lift）：加牌建模为**面内额外 cell**，warp 实证 8m 加杠是桌面上的平块、不浮。
- **副露**：self strip + 旋转 + 径向 `MELD_OUTWARD` 外移，对**直列副露(大明杠/吃)**对齐尚可。**残留→并入前修**：① 手量毡角的 H 轻微不对称 → 左侧远端(长列/暗杠)偏 → 用 `findHomography`(毡角+中央场风牌 anchor)重拟收紧；② 1 行等距 cell 未建模**横置被鸣牌**(更宽/侧置)与**加杠并排牌**(2D 偏移)；③ 直接在 rect 空间标定 canonical 副露 strip 取代外移 hack。
- **结论**：俯视 rectify 是正确方向，**河+加杠已解决**；副露需上述 rect 标定后再并入 `coords/river/meld`。
- **2026-06-30 续：方正性修复 + AutoMajsoul 调研**。初版 warp **歪**：根因是手量 4 角**左右不对称**(等价一个旋转)。采集相机无 roll/yaw(仅 pitch)→ 牌桌在图像里**左右镜像对称**：用 Hough 可靠拿到的**左 play 边**绕竖轴 `CX` 镜像构造四角 → 对称梯形 = 真·投影正方形 → warp 不再歪(`PLAY_CORNERS_NORM` 现由 `_TL/_BL/CX` 派生)。注意**牌桌物理四角在画面外**(需外推)，我们标定的是可见的**内圈 play 边界**。AutoMajsoul(`_external/AutoMajsoul`)靠 **HSV 色块**分割毡→取四角→warp，但要求毡是**内嵌形+四周暗**(其 Android/拉远视角)；我们**全屏 16:9 毡铺满整帧**(跑其真实 `detect()` 返回≈整帧、掩膜覆盖 60%+)，Hough/边缘也分不开上下 play 边(被牌行+手牌行干扰)→ 故用**固定相机标定**取代 per-frame 自动检测;`scripts/annotate/spike_topdown.py --detect` 保留了 AutoMajsoul 端口供内嵌视角用。

### 1.9c AB case_frame 标注器：GT 驱动生成（2026-07-01，已交付；独立于 spike 的第二条线）
与 §1.9/§1.9b 的 `H_table`（majsoul_eye 包内）**不同管线**：根级独立脚本 `mahjong_relative_annotation_pipeline.py`（自带单应 `SRC_TABLE_CORNERS`→1280 方→加 pad 的 fullwarp，与 spike 的 `H_table` 是两套坐标）。用户要求"参考它继续修正 `fails/topdown_demo/case_frames` 的标注"。原脚本只会 **resize 已有多边形**；本次给它加了 **GT 驱动的生成模型**：
- `generate_discard_slots` / `generate_meld_boxes` + 常量 `DISCARD_GRID`（每家 fullwarp 河格，origin+dcol+drow，**实测为精确线性**，0px 残差，4 家互为 90° 旋转）、`DISCARD_READ`（每家读序 = 4 重旋转；right/across 为 **R→L**，用 GT pai 对齐 tile-id 实证）、riichi 横置（`RIICHI_FOOT` 旋转足迹 + 同行后续牌右移 extra）、第 4 行溢出（>18 弃牌，罕见近流局；楔在顶行左侧）、`MELD_STRIP`（**角锚**：self 右下往左 / right 右上往下 / across 左上往右 / left 左下往上；加杠 = 3 直列 + 1 垂直 stack；暗杠 = back/face/face/back）。
- 驱动 `scripts/annotate/build_case_annotations.py`：经 spike `CASES`/replay 读 GT → 调管线生成 → 写 `out/mahjong_AB_relative_data_with_reliability.json`（**全 11 case**，弃牌带 GT 标签、副露 reliable）+ overlay `fails/topdown_demo/annot/`。
- **河：稳**（格精确、ai_g1/ai_g3 通吃、深河 ≤19、riichi 4 向、溢出）。**副露：够用**——角锚 strip 有 **~半张牌的逐局容差**（Majsoul 副露随手牌位置浮动；ai_g3 外围额外漂 ~1 张）。
- 11 并行 agent 复核确认；其"漏框"报告经 GT-pai 复核为**误报**（白板/暗牌上的细线框难辨）。真实 bug（第 4 行幽灵框）已修。

### 1.10 河/副露精准标注 v2 + ai_session 全帧标注器（2026-07-02）
解决 §1.9c 遗留的"副露需逐 case 标注、河偏移"，交付 `captures/ai_session` 全帧标注。

**理论澄清（用户问题：warp 能否得到真俯视/平行投影）**：单应只能对**一个平面**精确消除透视
（该平面→度量正射）；对离面点（牌厚、立牌）2D 变换原则上无法消除视差（需 3D/换视点），warp 后
仍是原相机光线的重参数化。但所有**平放牌的上表面共面**（z=牌厚平面），把标定对准**牌面平面**即可
让全部河/副露/dora 牌面零视差、全桌等大。牌厚只剩下"白色侧裙"（朝相机方向的可见厚度面），
它不进入牌面框——这就是本次校准的关键：**所有测量特征取"背裙边"的一侧**。

**实测事实（16 局 AI 数据、每座 ~5k 缝隙对 + ~800 边缘对，rmse 0.8–1.6px）**：
- 牌面在 fullwarp 里 ≈ **70.5×92.5 px、四家一致**；河**列**间距 72.5–74.9。
- **河的行间距非线性且四家不同**（`DISCARD_ROW_OFFSETS` 逐行标定，不能用等距 drow）：
  self r1→r2 = **96.4**、across r1→r2 = **104.9**、r2→r3 各家 ≈ 97–98；across 行 1（最靠中）
  比旧拟合再低 6.5px。教训：一次线性拟合的行顶检测曾被"裙边→缝隙"假特征污染出 108–110 的假行距，
  被 11-agent 视觉复核抓出后改用**逐行 边缘/缝隙链**测量收敛到 ±1px。
- 左右河原点纵向偏 ~10px = 旧标定用了含裙边的 blob 中心（视差方向 = 朝 nadir，+y 加横向 sign(1536−x)）。
- **横置立直牌 = 行内居中**（先前"顶对齐 Δ−11.5"实为行位置偏差的混淆；修正行后重测 ±2.5px ≈ 0）。
- **副露串随局浮动可达 ~½ 张牌**——along（沿串）与 self 串的 cross（纵向，屏幕底部锚定随手牌 UI）
  都会漂 → 静态角点只是先验，**必须逐帧 snap**。副露间**无间隙**（gap=0）。局内 σ<1px。
- 大明杠/碰的**横置被鸣牌**（上家=首/对家=中/下家=末、chi=首）与加杠并排、暗杠 back/face/face/back
  的**组成建模**是旧"半张牌容差"的主因——cell 宽度随直立(70.5)/横置(92.5)变化，串长随组成变化。

**实现**：
- `replay.py`：`Meld.called_pai/added_pai`（排序 tiles 丢失的被鸣/加杠牌身份，横置渲染必需）。
- 管线 §9b/9c/9d（`mahjong_relative_annotation_pipeline.py`）：重标定 `DISCARD_GRID/FOOT/RIICHI_FOOT`
  + `DISCARD_ROW_OFFSETS`（逐行）；组成感知 `generate_meld_boxes_v2`（`meld_display_cells`）；
  `river_sideways_index`（立直牌被鸣→下一张横置）；掩膜检测器 `tile_face_mask/tile_back_mask`、
  `find_crevice`（缝隙暗谷）、`find_edge`（**非对称平台差分**，前窗 12px/后窗 5px——宽缝隙(≥5px，
  周边拉伸处)塞不进前窗，冒充不了毡→面真边缘）、`snap_meld_strip`（**多候选评分**：干净端边缘 +
  blob 覆盖扫描 + 零偏移 各自过缝隙/边精调，以特征对比度总和选优——均匀网格里缝隙是周期性的，
  单一粗定位会静默锁错半格；cross 另有 ±60 宽窗粗锁吸收 self 串纵向漂移）。
- `scripts/annotate/calibrate_annotation_model.py`：跨局测量→稳健线性拟合→建议常量（可复跑再校准）。
- `scripts/annotate/annotate_ai_session.py`：**全帧标注器**——河（网格+GT+逐格 fill 置信、最新弃牌未渲染→
  `unrendered`）、副露（v2+逐帧 snap+fill）、英雄手牌（HandModel+白度门，发牌动画→`hand:unrendered`）；
  输出 per-game JSONL + overlay 抽样 + `summary.json`（含 97.6% 分类器 crop 一致率 QA）。
- `scripts/annotate/build_case_annotations.py`：11 case 用 v2+snap 重建（`fails/topdown_demo/annot/` +
  `out/mahjong_AB_relative_data_with_reliability.json`）。
- 数据：`captures/ai_session` run_1–8 全部 16 局转换为 `captures/ai_run_*.jsonl`（~8.9k 帧）。

**QA（分类器一致率 = 端到端框质量代理；分类器自身天花板 97.6%）**：
ai_g3 河 **100%** / 副露 **100%** / 手牌 **100%**；ai_g1 河 97.7% / 副露 96.6% / 手牌 100%
（残余=游戏光标手遮挡 + 3s→4s 单牌分类混淆，非几何）。河框 99.8–99.9% 通过 fill 门。

### 1.11 captures/ 目录重构：角色分层 + 相对路径索引（2026-07-02）
`captures/` 顶层曾散落 26 个裸 `.jsonl` + ~24 个帧目录，无原始/中间区分。重构为角色分层：
- **布局**：`raw/{ai_session,manual}`（原始，不可再生）、`intermediate/{gt,derived}`（可再生：
  转换后 GT + 索引 / 裁剪·去黑边像素）、`legacy/`（归档 `ai_g*/ai_r1` 逐字节重复）。
- **单一真源** `majsoul_eye/paths.py`：布局常量 + `frames_dir_for`（`X.jsonl↔X/` stem 规则）+
  `resolve_frame_path`（解析 `frames.jsonl` 的 `file`，向后兼容旧绝对路径）+ `converted_gt_captures()`。
  ~10 处散落的 `"captures/…"` 字面量/glob 收敛到此。
- **根治绝对路径脆弱性**：所有 `frames.jsonl` 的 `file` 由绝对路径改为**相对**（自包含目录存
  `frames/NNN.png`；gt/ 空壳索引存 captures 相对 `raw/ai_session/…`）——今后移动帧目录不再破坏索引。
- **迁移器** `scripts/data/migrate_captures_layout.py`（dry-run 默认、幂等、可续跑）：同卷 rename（非拷贝，
  不遍历 PNG）+ 逐索引 `.premigrate` 备份 + `MIGRATION_MANIFEST.json`。实测：60 顶层项迁移、
  12,499 条 `file` 改写、0 未解析（--strict）。生产者（convert/ingest/record_gt/autoplay/crop/
  deletterbox/sync）全部改为写新子目录 + 相对路径；10 套测试全绿。

### 1.12 合并精确标注管线进包（§1.10 v2 → `majsoul_eye.annotate`；一套标定喂 分类+检测）（2026-07-02）
§1.10 的精确标注只活在**根级独立脚本** `mahjong_relative_annotation_pipeline.py`（fullwarp 坐标），而
`build_dataset.py` 仍走**包内旧** `RIVER_QUADS`/`MELD_STRIPS` 松框（+erode 补偿）——两套坐标，每采一局债放大。
本次把精确管线**提升进包**，让 `build_dataset` 消费精确框，一套标定同时喂 分类 crops + 检测 labels。
- **新包 `majsoul_eye/annotate/`**：`pipeline.py`（几何+证据核心，从 1213 行根脚本搬入、删死代码 §7/§8/§9b-v1/§10
  → 879 行；根名留 `sys.modules` 别名 shim，兼容 `import ... as P` 含 `P._box_fill`）、`frame.py`（`annotate_frame`
  + `AnnBox`/`iter_tile_boxes`/`crop_box`/`crop_quad`）、`seatgt.py`（`seat_gt`/`SEAT_POS`）。
  **`majsoul_eye/capture/gtframes.py`**：`build_seq_state`/`load_frames`（去重 `spike_topdown` 与 `build_dataset`
  的两份重复加载；`load_frames` 加 `statuses` 参数）。
- **`build_dataset.py` 改走精确路**：帧 resize→1920×1080、`annotate_frame` → `iter_tile_boxes` → **透视 quad 裁剪**
  （河用 `face_poly`、副露用 `poly`，96px）+ **轴对齐 YOLO**（含副露/`back`(37)/dora）；按 `reliable` 门控、
  **分类裁剪排除 sideways**（横置朝向不可几何恢复，仍进 YOLO）；**写 resize 后的帧**（修 `shutil.copy` 把 4K 图
  配 1080p 标签的 bug）；退掉 `--river-erode-*`/`--min-bright`/`--min-face-frac`；非 16:9/letterbox 帧跳过并计数。
- **零行为漂移**：Phase 1–3（纯搬移/去重）逐步 diff，`annotate ai_run_3_game1` 输出**全程逐字节等于金样**
  （河 21207 框 99.8% ok、meld 4836 99.3%、QA river .9765/meld .9402/hand 1.0）。旧 `label/river.py`·`meld.py`·
  `coords.RIVER_QUADS`/`MELD_STRIPS` 及其测试**保留保绿**（弃用留独立 PR）。
- **验收（跨局重训）**：4 AI 局精确 crops（~108k，含 `back` 278/red5 1402），held-out **g8_1 整局** → **val 98.89%
  （erode 关闭，＞旧 97.6% 基线）**；**`back` 0.000→1.000**（旧 6 局分类器从未见 meld back，本次首次可训）。红五在此
  4 局子集略降（数据量），全 16 局重建可恢复。
- **产物/后续**：子集验收模型存 `scratchpad/tile_classifier_precise.pt`（**非正式**；正式权重待全 16 局重建）；
  YOLO 检测集导出（P5）与旧 `label/` 弃用为后续 PR。新增测试 `test_annotate_pipeline`/`test_annotate_frame`。

### 1.13 旧 `label/` river/meld 弃用 PR + spike/根 shim 去承重（2026-07-02）
兑现 §1.12 末尾承诺的"独立弃用 PR"，并清掉两处技术债：包内新代码反向依赖被弃用模块、spike 名不副实地承重。
- **杀掉反向依赖**：精确管线 `annotate/seatgt.py` 曾 `from majsoul_eye.label.river import _screen_to_seat`
  （新包 → 待弃用模块）。把 `_screen_to_seat`/`SEAT_POS` 迁为 `annotate/seatgt.py` **自有定义**，包不再 reach 进 `label/`。
- **删除被取代的旧几何**：`majsoul_eye/label/river.py`、`label/meld.py`（等距 `RiverGrid` 河/副露模型）、
  `coords.RIVER_QUADS`、`coords.MELD_STRIPS` 及其测试 `test_river.py`/`test_meld.py` **全部删除**——均被 §1.10/§1.12 的
  fullwarp 精确管线取代。`label/autolabel.py` 保留（仍供 `annotate.frame` 的手牌+dora），剥掉其 river/meld 死分支，
  `DEFAULT_ZONES` 由 `{hand,river}` 改为 `{hand}`。`RIVER_ZONES`（粗象限框，overlay 用）不受影响。
- **spike 去承重 + 归档**：`scripts/annotate/spike_topdown.py` 曾是"名 spike 实承重墙"（`build_case_annotations` 从它
  import `CASES`/`load_pair`/`_screen_to_seat`/`SEAT_POS`）。共享 plumbing 迁进包：`CASES`→`annotate/cases.py`、
  `load_pair`→`capture/gtframes.py`、`_screen_to_seat`/`SEAT_POS`→`annotate/seatgt.py`。spike 改从包 import，
  被删的等距 `RiverGrid` + self 副露 strip **内联**到 spike 内 → 成为**自足、可跑、不承重**的归档可视化工具。
- **删根 shim**：`mahjong_relative_annotation_pipeline.py`（`sys.modules` 别名 → `annotate.pipeline`）删除；
  唯一消费者 `build_case_annotations.py` 改 `from majsoul_eye.annotate import pipeline as P`。
- **零行为漂移验证**：重跑 `build_case_annotations.py` 输出与已提交金样
  `out/mahjong_AB_relative_data_with_reliability.json` **逐字节相同**；全 10 套测试绿；6 个改动脚本 import 全过。
- **结果**：scripts 层**零 inter-script import**（README 声明的架构目标达成）；包不再依赖被弃用模块；死代码清零。

### 1.14 全 16 局重建 + 重训 → 正式权重 99.78%；build_dataset 复用 out/ 去冗余（2026-07-02）
接 §1.12 的"待全 16 局重建"：dora 标注补齐后，16 局全部建成精确数据集并重训分类器。
- **数据**：16 局 `datasets/precise_ai_run_*/`（~385k crops，含 dora）；红五充足（5mr 1500 / 5pr 1351 / 5sr 1424，
  远超旧 4 局的 286/357/246）、`back` 34653。ai_run_4_game1（掉线局）经 `--drop-violations` 后仅 1236 crops（0.3%，无害）。
- **训练**：15 局训 + held-out **整局 ai_run_8_game1**（`--val g81:*`，与 §1.12 同局，趋势可比）。RTX 5080、20 epoch。
  **best held-out val_acc = 99.78%**（旧 6 局正式 97.6% → 4 局精确子集 98.89% → **本次 99.78%**）。
  除 **5pr（红 5p）90.8%**（n=185，已知红五跨花色混淆）外，所有常规类 ≥99.4%。
- **升正式权重**：`majsoul_eye/recognize/tile_classifier.pt` 换为 16 局模型（旧 6 局备份 `tile_classifier_pre16games.pt`）；
  加载 + 抽样分类校验 96/96 通过（含 `back`/`5pr`）。剩余弱点 5pr → 归"红五/稀有类专项"。
- **去冗余（响应"为什么重跑相似的"）**：`annotate_ai_session`（并行 `ProcessPoolExecutor`）已把标注写进 `out/`，
  但 `build_dataset` 原本再跑一遍 `annotate_frame`（双重计算）。新增 **`build_dataset --from-annotations out/ai_session_annotations`**：
  直接从 `out/*.jsonl` 记录裁剪（`iter_tile_boxes`/`crop_box`），跳过 warp/mask/snap 重算；实测同帧 crop **逐字节相同**
  （默认自足路径不变）。管线变线性：`annotate_ai_session（并行）→ out/ → build_dataset --from-annotations → train`。

### 1.15 P5 YOLO 检测器：从 build_dataset 免费 YOLO 标签训出 → 正式 `tile_detector.pt`（mAP50 0.993）（2026-07-03）
兑现路线图近期项 #2「训 YOLO 检测器」。`build_dataset` 早已从同一套精确几何**免费导出 YOLO 标签**
（`<out>/yolo/images/<seq>.png` + `labels/<seq>.txt`，`class cx cy w h`，冻结 38 类，含 hand/河/副露/dora/back），
本次只做「装配 + 训练 + 封装」，未改标注。
- **依赖**：`pip install ultralytics`（8.4.84）装进 `auto` 环境；torch 2.11 未被降级；RTX 5080（16 GiB）。
- **导出**：16 局用 `build_dataset --from-annotations out/ai_session_annotations --no-crops` 重导 YOLO（旧 `precise_*` 建时 `--no-yolo`，无 yolo/）；全帧 PNG，共 ~8,737 帧、~1.5 GB/局。
- **装配**（新 `scripts/train/build_detector_dataset.py`）：镜像 `train_classifier` 的 `--data NAME=YOLODIR:CAPTURE` / `--val`
  接口，**按局/kyoku 切分**（内联 `seq_to_kyoku`，零跨脚本 import）。不拷图：写 `train.txt`/`val.txt`（**仓库根相对 POSIX 路径**——
  Ultralytics 按 CWD 解析图路径、靠替换 `images`→`labels` 段找标签，且加载前先 `/`→os.sep，故 Windows 用正斜杠也行）+ `data.yaml`
  （相对 `path:` + `names` 取自 `tiles.TILE_NAMES`）。**从仓库根跑** → 整个 `datasets/` 连仓库可直接打包搬到 GPU 服务器训练、**无需重生成**（实测搬走后 val 逐帧对上、mAP 不变）。
- **训练**（新 `scripts/train/train_detector.py`，薄封装 ultralytics）：`yolov8s`、imgsz **1280**（河牌在 1920 帧里~40-60px，默认 640 缩到~15px → 小目标召回崩）。
  - **M1** 单局 `ai_run_3_game1`（held-out kyoku）验管线 → mAP50 **0.915** / mAP50-95 0.864（唯 5mr 召回 0，单局稀有类）。
  - **M2** 全 16 局（7,777 训 / **held-out 整局 `ai_run_8_game1`** 960，与分类器同局）→ **best.pt @epoch13：mAP50 0.9928 / mAP50-95 0.9546、P 0.982 R 0.978**。
    红五全部修好（5mr/5pr/5sr mAP50 0.995/0.991/0.993）、`back` 0.995、常规类 ~0.994。升为正式 `majsoul_eye/recognize/tile_detector.pt`。
- **OOM 教训**：16 GiB 卡上 batch16→17.2G、batch8 也随 epoch 涨到 18.6G（mosaic × 高实例帧 + 碎片）**两次被杀**；
  epoch13 已近天花板（mAP50-95 ep8→13 仅 0.943→0.955）故直接升 best.pt。修复：`train_detector` 置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，跑满 60 epoch 建议 `--batch 4`。
- **运行时封装**（新 `majsoul_eye/recognize/detector.py`）：`TileDetector(weights, device)` + `predict(bgr)->list[Detection]`，镜像 `TileClassifier`；
  ultralytics **懒加载**（`import majsoul_eye.recognize` 不拉 ultralytics）；保持 **Akagi-free**；已从 `recognize/__init__` 导出。
- **HBB→OBB 后续**：当前 emitter 把透视四点框塌成轴对齐 HBB，远/旋转座（上家、riichi 横放）框偏松（5sr mAP50-95 0.874 为最低）；
  `box.poly_original` 已存于记录（`build_dataset.py:135-138` 丢弃），OBB 只需改 8 点标签 + `yolov8-obb`。

### 1.16 发牌动画帧污染修复：丢每局首帧（rivers-empty）+ 采集期规避（2026-07-03）
- **Bug**：`ActionNewRound` 那一帧（每局 1 张，如 `run_4/game1/frames/000018.png`）在雀魂 ~2-3s **发牌动画中途**被截：手牌只发了 8 张且**未排序**，未发的槽是空桌布。
  但 GT 手牌是排好序的整副 13 张 → 自足 `build_dataset` 把 13 个手牌框贴在排序位上 → **裁剪标签错**（`crops/1m/000018_000.png` 实为二萬、`crops/F/…` 实为空桌）。
- **根因/信号**：桥把 `[start_kyoku, tsumo]` **打包进同一条记录**，故 `last_event=='start_kyoku'` **一张都命不中**（实测 106 帧命中 0）。
  唯一稳健的 GT 信号是 **`rivers` 全空**（本局尚无 dahai）：`state.replay.is_deal_window(state)`——3P/4P 通用、不依赖 `leftTileCount` 魔数、对打包免疫、首弃即翻假。
- **修复（丢帧，非"标框不可信"）**：`is_deal_window` 帧在 `build_dataset`（自足+`--from-annotations` 两路，无条件建 `seq_state`）与 `annotate_ai_session` 处**整帧丢弃**；
  取代原 `suspect_seqs` 序数启发式（"每局前 2 seq"——会**误伤好的首弃帧**、且被打包坑到）。首弃帧（rivers 非空）现完整保留。
- **采集期规避**：AI 路 `autoplay_ai.py` 对 `ActionMJStart`/`ActionNewRound` **不 arm 截图**；手动路 `sync.py` `note()` 从 `start_kyoku` 到**首个 dahai** 抑制截图（`_in_deal_window`）。两处都有标注期兜底。
- **已有数据清洗**（新 `scripts/data/purge_deal_frames.py`，dry-run 默认、幂等）：删 16 AI 局 **840** 裁剪 + 手动局 60 = **900 裁剪**、**121** YOLO 图/标；
  并重写 `datasets/detector/{train,val}.txt` 去悬空行（train 7777→7682、val 960→949）。
- **已重训并升正式权重**（承接上一条的"待重训"）：分类器在 purge 后数据上重训 → best **val_acc 0.9991**（held-out 整局 `ai_run_8_game1`，＞旧 16 局 99.78%），已升正式 `majsoul_eye/recognize/tile_classifier.pt`（commit `0f3b481`；旧全部备份快照 ai/allgames/clean/erode/pre16games/preAI/prePollutionFix 一并删除，`.gitignore` 只追踪正式 `tile_classifier.pt`）。检测器也在 purge 数据上重建为 AABB/OBB 两个变体（见 §1.17）。
- **测试**：`test_replay` 加 `is_deal_window`（打包后 `last_event=='tsumo'` 仍判 True→首弃翻 False）；`test_sync` 加 note 发牌窗抑制。11 套仍全绿。

### 1.17 OBB 定向检测器：8 点标签 → 训练 → 运行时双分支 + weights/ 布局（2026-07-03）
兑现 §1.15 结尾的「HBB→OBB 后续」：把透视四点框 `poly_original`（build_dataset 里原本被塌成轴对齐 HBB）
直接导成**定向 8 点标签**，训 YOLO-OBB，让远座/riichi 横放牌的框贴着真实朝向而非松的外接矩形。
- **标签导出**（`build_dataset.py --obb`）：新纯函数 `box_quad`（河/副露取 `poly_original` 透视 quad、手牌/dora 取
  `px_box` 展成矩形，**单一几何真源**同喂 OBB+HBB）、`obb_label_line`（`cls x1 y1..x4 y4`，归一化+clip[0,1]，**保 quad 角序**
  = 保真实旋转）、`hbb_label_line`（历史 `cls cx cy w h` 轴对齐路径不变）。三者均单元测试（`tests/test_detector.py` +92 行）。
- **训练**（`train_detector.py`）：`--model weights/pretrained/yolov8s-obb.pt`（ultralytics 自动 `task=obb`）。
  新增 `resolve_device`/`--device`：`''` 自动、`'0'` 单卡、`'0,1,2,3'` DDP 拆一次训练、`'cpu'`；
  单卡+`CUDA_VISIBLE_DEVICES` 跑并行独立实验，多 id 拆单次 DDP（batch 变全局）。基座 `--model` 默认改指
  `weights/pretrained/<name>` 避免 ultralytics 往 cwd 重下。产物 `weights/detector/tile_detector_obb.pt`，
  **mAP50-95 0.9804（rotated-IoU）**。
- **运行时**（`recognize/detector.py`）：`_parse_result(res)` 按 `res.obb`（OBB，即使空也不碰 `res.boxes`）vs `res.boxes`（HBB）
  分支；`Detection` 加 `poly`（4 个定向角点，HBB 为 `None`）同时保留 `xyxy`（外接框）做 drop-in 兼容。
  `TileDetector.predict` 对两种权重通吃。保持 **Akagi-free + ultralytics 懒加载**。
- **`weights/` 布局落地**：`pretrained/`（训练基座：`yolov8s.pt` AABB / `yolov8s-obb.pt` OBB）+ `detector/`（可选变体
  `tile_detector_aabb.pt` / `tile_detector_obb.pt` 并列，不自动加载）。`.pt` blob 全 gitignore（超 GitHub 50MB），只追踪
  `README.md`+`.gitkeep` 保约定版本化。**shipped 权重不在此**——运行时 `tile_detector.pt` 仍在 `recognize/` 旁。
- **诚实价值评估**：本 web 全屏 16:9 采集几何**相当俯视**，只有 **~7.4% 的牌真正倾斜**（poly/bbox 面积 <0.9，即远座透视牌；
  最斜比 ~0.735）；直立牌与 90° riichi 牌都是轴对齐矩形（OBB poly == HBB bbox）。故 OBB 对 HBB 的增益真实但**集中在远座少数**，
  预计在陡透视/手机端视角收益更大（**尚未实测**——还没有外部截图）。

### 1.18 权重仓瘦身：只保留生产分类器 + 历史清除备份（2026-07-03）
- **问题**：`majsoul_eye/recognize/` 累积了 12 个 `.pt`——每个训练里程碑的"改动前/后"快照（`_prePollutionFix`/`_clean`/`_erode`/`_preAI`/`_ai`/`_allgames`/`_pre16games`/`_preDealfix`/`_dealfix` …）。多个是逐字节重复（`_preDealfix`==生产、`_pre16games`==`_allgames`、`_preAI`==`_erode`），且 8 个已进 git 历史，`.git` 达 **281 MB**（从未 gc，5379 loose 对象、最大 blob 全是这些分类器权重）。
- **处理**：（1）晋升 `_dealfix`（val_acc 0.9991）覆盖生产 `tile_classifier.pt`；（2）工作区仅留 `tile_classifier.pt` + `tile_detector.pt`（生产检测器，本地/gitignore）；（3）`git filter-repo --invert-paths` 从**全部历史**抹除 7 个已追踪备份权重；（4）`.gitignore` 修正——只追踪生产 `tile_classifier.pt`，忽略 `tile_classifier_*.pt` 变体与所有 `tile_detector*.pt`；（5）`git push --force` 已同步远端。
- **结果**：`.git` **281 MB → 14 MB**；历史中 0 个 `_*.pt` 备份残留；`tile_classifier.pt` 载入正常（38 类冒烟通过）。改写前全量 bundle 备份留于 scratchpad（原 HEAD `3b57fd1`）可回滚。

### 1.19 统一 AI 采集为标准 GTRecord；`intermediate/gt` 退役（2026-07-04）
- **问题**：`autoplay_ai.py` 原先只落**原始 liqi 线流**（b64 protobuf + 1080p PNG），要用离线
  `convert_mjcopilot.py` 转成我们的 `GTRecord` 才能喂标注/建库（`captures/intermediate/gt/ai_run_*.jsonl`）；
  手动 `record_gt.py` 路径则是**实时**直接写 `GTRecord`。两条路径格式不同，下游要各留一份适配代码，
  异常退出（Ctrl-C）时线流也没有落盘的部分 GT。
- **处理**：`autoplay_ai.py` 现在**内联**边播边写统一 `GTRecord` + 截图索引（`make_capturing_game_state`
  抽出共享的"驱动 GameState 产 MJAI"逻辑，`convert_game` 与实时写入复用同一份）；输出目录直接落在
  `captures/raw/ai_session/run_N/gameM.jsonl`（GTRecord）+ `gameM/{liqi.jsonl 原始线流备份, frames.jsonl 增量索引, frames/*.png}`，
  与手动路径同构。`paths.ai_captures()`/`paths.ai_game_name()` 取代硬编码的 `intermediate/gt` 扫描；
  `build_dataset`/`annotate_ai_session`/`rebuild_datasets`/`ingest_run` 全部改读 `raw/ai_session`，
  不再有 convert 这一步。18 局旧 b64 线流（含之前未转换的 run_13/14）用一次性
  `scripts/data/migrate_ai_to_gtrecord.py`（复用 `convert_mjcopilot.convert_game`，dry-run 默认、幂等、
  崩溃可续跑）就地迁移到新布局；迁移前全量备份于 scratchpad（`pre_migration_backup/{gt,wire}`）。
- **验证**：迁移后 `load_frames` 对比新旧索引（`raw/ai_session/run_3/game1` vs 退役前的
  `intermediate/gt/ai_run_3_game1`）帧集合与解析路径完全一致（`FRAME INDEX OK`）；
  `rebuild_datasets.py` 干跑仍发现全部 18 局 AI 游戏（`intermediate/gt` 删除前后行为不变）；
  单局冒烟（`annotate_ai_session` → `build_dataset --from-annotations`）产出 37545 crops，
  `build_dataset` 正确复用已有标注（`reuse: N records <- ...`）。
- **结果**：`captures/intermediate/gt/` 已删除（衍生数据，非 git 跟踪，不是 git 变更）；
  `convert_mjcopilot.py` 降级为共享的转换/迁移库（不再是常规 pipeline 的一环），仍可作为独立
  CLI 处理任何遗留原始线流。`intermediate/derived/*_fixed`（去信箱化帧）不受影响。

### 1.20 散件收尾（2026-07-03/04，此前漏记）：遮挡防护、hero-tsumo 修复、重建驱动器、overlay、语言元数据
- **弃牌动画遮挡**：治本改在**采集期**——`capture/roi_diff.py` ROI 稳定确认（弃牌区像素稳定才落盘），
  实测残留 ~0.4%；build_dataset 的分类器一致性门 `--occlusion-gate` 因此降为 **opt-in（默认关）**；
  一次性清洗工具 `purge_occlusion_frames.py` 在全量重建后无需再跑。
- **hero-tsumo 修复**（`148d3cf`）：autolabel 的 `len%3==1` 守卫跳过 14 张自摸态 → 玩家自己回合 ~1000 帧
  手牌无标签 = 检测器负样本信号（own-turn 抑制手牌）。经 `replay.drawn_tile` 标注摸牌槽修复；
  07-04 15:58 全量 regen 落地（`annotate --workers` 默认 16→4，防 RAM 冻死）。
- **重建驱动器**（`0ccefe8`）：`scripts/data/rebuild_datasets.py` —— annotate → build_dataset
  (--from-annotations) → build_detector_dataset 三阶段一键全量重建（dry-run 默认、`--stage` 单跑、
  `--workers`/`--jobs` 并行、训练命令打印不执行）；自动发现全部 AI 局 + manual session5/6。
- **浏览器检测框 overlay**：`autoplay_ai --overlay` 经注入 canvas 在实局浏览器画检测器输出
  （`capture/overlay.py` `DetectionOverlay` 守护线程、懒加载 TileDetector、截图对 overlay 不可见、
  页面重载自愈）。⚠️ 实局人工目验仍待做。
- **每局语言元数据**：`run_N/gameM/metadata.json` = `{"language": <BCP-47>}`（liqi 不含语言；
  解析优先级 `--lang` > localStorage 探测 > 服务器粗判；MajSoul 码 `chs`/`chs_t`/`jp`/`en`/`kr`
  已实证）。纯函数模块 `capture/gamemeta.py`。
- **HUD 检测设计稿**（未实施）：`docs/superpowers/specs/2026-07-04-hud-detection-design.md`
  （55 类 YOLO v2 + micro-readers）。

### 1.21 管线梳理收官：采集统一 + run_13/14 补建 + 权威管线文档（2026-07-04）
- **决策**：采集**统一为 AI 自动路线**（autoplay_ai）；`record_gt.py` + Akagi MITM 手动 F11 路线
  **列为过时**（脚本保留存档；session5/6 存量数据继续留在训练集，`rebuild_datasets` 仍会构建它们）。
- **run_13/14 补建**：§1.19 迁移后这两局（早退迷你局，各 15 帧）一直没进数据集——增量补跑
  annotate → build → detector 重组：+355/+485 crops、run_14 标注 QA 河 99.5%/副露 100%/dora 100%；
  detector split 现 **20 源 / train 8683 / val 949**（held-out 仍整局 ai_run_8_game1）。
  至此 `captures/raw` 下所有**有 GT 的对局**全部进入现行管线产物，无历史欠账。
- **数据审计**：`captures/raw/temp/`（5 个失败 run）为**无 GT 孤儿帧**（只有 PNG + frames.jsonl，
  没有对局 GTRecord，~76MB）→ 不可用，待清理；`captures/captures.7z`（**20.5 GB** 手工备份压缩包，
  07-03）不应常驻 captures/ → 建议移出仓库目录或删除；raw/derived/legacy 其余部分与 paths.py 布局一致。
- **文档**：新增 **[PIPELINE.md](PIPELINE.md)** 为当前管线唯一权威（一图流、目录角色表、各阶段要点、
  新 run 增量 SOP、过时组件清单、**维护规约：任何影响管线的改动必须同步更新该文档**）；
  README/CLAUDE.md 全面刷新（AI 主路径、清掉 `intermediate/gt`/已删测试等全部陈旧引用）。

### 1.22 数据集版本化：`build_datasets.py` + 多版本训练 + v1 搬家修复（2026-07-04）
- **需求**：`rebuild_datasets` 只做"原地重建固定布局"；要的是**构建**语义——指定采集根（将来的
  `captures/raw/ai_session_2`）、输出独立版本目录、训练可跨版本混用。用户已把旧平铺产物手工移入
  `datasets/v1/`。
- **新驱动 `scripts/data/build_datasets.py <name>`**：`--sources` 多根（默认 `raw/ai_session`；发现
  `run_*/game*.jsonl` / `run_*.jsonl` / `session*.jsonl` 三种形态；游戏名跨根必须唯一，冲突报错）
  → 标注缓存进 `datasets/<name>/annotations/` → 每局 `<name>/<game>/{crops,yolo}` → `<name>/detector/`
  → `games.json` 清单。**立即执行**（`--dry-run` 干跑，不再是 rebuild 的 `--yes` 语义）；`--resume`
  只补缺局并保留旧清单的 dir 映射（兼容 v1 的 `precise_` 前缀目录）；`--force` 清空重建；
  `--obb`/`--workers`/`--jobs` 透传。
- **多版本训练**：`train_classifier.py` / `build_detector_dataset.py` 新增可重复 `--dataset <dir>`，
  按 `games.json` 展开逐局 `--data`（内部走**元组**而非 `NAME=DIR:CAPTURE` 字符串——绝对路径的
  Windows 盘符冒号会打穿旧的 `:` 切分；同名局后者覆盖并打印提示）。跨版本检测集：
  `build_detector_dataset --dataset datasets/v1 --dataset datasets/v2 --out ...`。
- **v1 搬家修复**：手工移动击穿了 `detector/{train,val}.txt` 的仓库相对路径 → 重写
  `datasets/precise_*` → `datasets/v1/precise_*`（8683+949 条全部存在校验）+ `data.yaml` 的 `path:`；
  补写 `games.json`（20 局，dir 保留 precise_ 前缀）；`out/ai_session_annotations/*.jsonl` 拷入
  `v1/annotations/` → `--resume` 干跑 annotate 0 / build 0（新 run 增量成本仅为新局本身）。
- **验证**：`--dataset datasets/v1` 重装配与修复后 split **逐行一致**（忽略 CRLF）；两局迷你局端到端
  冒烟（标注→建库→装配→manifest、resume 跳过、误覆盖保护）全过；新套件 `test_build_datasets`
  （发现/冲突/FRAMES_OVERRIDE/清单往返/dir 保留/命名）；全部 27 套测试绿。顺修
  `build_data_yaml_text` 跨盘符 relpath 崩溃。
- **弃用**：`rebuild_datasets.py` 标 DEPRECATED（验证期后删）；PIPELINE.md / README / CLAUDE.md 同步。

### 1.23 `autoplay_ai.py` 真正的 `--dry-run`（不落盘）+ 术语正名（2026-07-05）
- **背景**：`--live` 只管**点不点牌**；不带 `--live` 的旧"dry-run"仍照常建 run/game 目录、写
  screenshot/GT/wire/index/metadata。用户要一个**真正的 dry-run**：任何文件夹/文件都不产生。
- **正名**：不点击的默认态改称 **OBSERVE**（banner `[OBSERVE]`、逐手日志前缀 `OBS`、结算日志
  `(observe: not confirming)`），把"dry-run"一词让给新语义；无逻辑依赖该字符串，纯展示层改名。
- **新 `--dry-run`（与 `--live` 正交）**：run/game 目录、截图、GT/wire/index/metadata **全不写**；
  `game_frames_dir` 与三个文件句柄保持 `None` 使各写点与 `maybe_screenshot` 自然 no-op；"无对局"哨兵
  由 `game_wire_fh is None` 改为 `game_state is None`（dry-run 不再有 wire 句柄）。MahjongCopilot 的
  `Settings` 必须有真实文件路径且会 re-save → dry-run 把 `ai_settings.json`（及 `--skins` 的
  `skin_proxy.log`）落到 `tempfile.mkdtemp()` 临时目录，`finally` 里 `shutil.rmtree` 删除。四组合全部有意义：
  OBSERVE/LIVE × 写/不写。默认行为与产物**零变化**（新旗标 opt-in），不 stale 任何衍生数据。
- **验证**：语法 OK；`test_autoplay_gt`（新增断言 `--dry-run` 存在）、`test_autoplay_stability` 绿。
  PIPELINE.md 采集段同步（默认 OBSERVE + `--dry-run` 说明）。

### 1.24 auto-next 误点修复：按真实终局序列 + 共存判别 + 动态质心点击（2026-07-05）
- **症状**（用户实测 + 日志/截图）：`--live --auto-next` 停在排名屏后 `auto-next guard OK next
  frac=0.250` 即自认为成功返回，`no game started ... giving up`。最终（更早的实弹）曾点进**商店**。
- **真实终局序列**（jp 段位戦实测）：`終局排名(確認) → pt/成就(確認) → 任务屏(もう一局 + 確認)
  → 再来一局确认弹窗(はい / いいえ) → 匹配(authGame)`。旧代码**完全没处理最后的弹窗**，且把结算屏
  数量写死（`--auto-next-confirms=2`）。
- **两个 bug**：①排名屏的 **2/3/4位 蓝色排名条**在滑入动画期落进旧的宽 next 框(11.0–13.6,7.88–8.85)
  → 蓝色 frac=0.250 误报"再来一局"→ 盲点+假成功返回；早期该盲点还落在大厅底栏(商店/寻觅) → 进商店。
  ②弹窗 はい 从未点击 → 即使正确点了もう一局也卡死。
- **重写**（`autoplay_ai.py` 模块级 `auto_next_flow`，依赖注入可单测）：**屏幕身份靠按钮共存判别**
  （单个颜色框不够——排名条 aliases もう一局 蓝、はい aliases 確認 黄）：大厅可见→急停（`mainmenu.png`
  掩码模板 + CDP 截图）；`はい∧いいえ∧无確認`→弹窗点はい（校验消失后返回）；`もう一局∧確認`→任务屏点
  もう一局；`確認`→结算屏推进（**无次数上限**）。每个 `button_guard(kind)` 返回 `(present, frac, 质心)`，
  **点质心**（按钮精确位置自标定，免二次校准）。框/色实测标定：確認黄右下、もう一局蓝底中(仅任务屏)、
  弹窗はい/いいえ居中(无確認)。主循环 watchdog：点了はい但 `timeout+120s` 无 authGame→failed
  （有 `--autojoin` 回退大厅重排）。`--auto-next-confirms` 随之删除。
- **验证**：`tests/test_autoplay_autonext.py`（9 例：完整序列、任务屏只点もう一局、排名条不误触、confirm
  不设限、弹窗与结算区分、点后校验重试、大厅零点击×2、超时 failed）；且**用户 4 张真实整屏截图**跑真实
  `button_guard` 逻辑，**四屏全判对**：ranking/rankpt→確認、missions→もう一局(rematch=0.596)、弹窗→はい
  (质心 6.51,6.60；confirm=False 因模态后的 確認/もう一局 变暗不触色)。ranking rematch=0.000。全套件绿。
  はい/いいえ/確認/もう一局 fallback 坐标已按整屏实测更新。
- 另查明（未在本次范围）：`--autojoin` 冷启动死角（`ui_state=NOT_RUNNING` 时 `decide_lobby_action`
  直接 pass，且 WS tap 只挂 game socket 看不到 lobby 登录）；断线"继续对局"弹窗无处理。

### 1.25 `--skins` 杀死自动打牌：mod.py 改写帧丢失空 method_name 块（2026-07-05）
- **症状**（用户实测，jp/cn 均复现）：开 `--skins` 后 Mortal 全程不出牌；日志出现
  `parse err: AssertionError:`（空消息）+ `operation seat 2 != self.seat 0`。不开 `--skins` 正常。
- **根因**：`BaseMessage` 是 **proto3**（空标量不序列化），mod.py 对**被改写**的非 Notify 帧的回写
  `buf[:3] + msg_block.SerializeToString()` 会丢掉雀魂原生 RES 帧**必带的空 `method_name` 字段**
  （线上实抓验证：RES 帧头 `0a 00`）。浏览器不在乎，但 tap 用 MahjongCopilot 的**位置式**
  `liqi.py` 解析（`assert block[0].data == b''`）→ mod.py **无条件改写**的 `authGame` RES 被断言
  丢弃 → GameState 永远学不到自家座位（`self.seat` 停在构造默认 0）、`init_bot` 不跑 → 零反应。
  旧 stale-liqi 时代改写静默失败反而"救"了 autoplay——§1.24 前修好解锁后此坑才暴露。
- **修复**：`patch_majsoulmax.py` 新增第 4 个补丁 **`ensure_res_framing`**——回写改用
  `liqi_new.toProtobuf` 显式两块（method_name 即使为空也保留），与原生帧**逐字节一致**；
  `SkinProxy` 启动时自动施加（与 protobuf-7/max_data 同批）。SKINS.md 陷阱清单同步（第 5 条）。
- **验证**：新测 `test_skins.test_patcher_res_framing`（幂等 + 功能：proto3 确实丢块、新框架逐字节
  还原）；离线端到端——真实抓帧的 authGame RES 走旧回写喂 MJC 解析器 → 精确复现 AssertionError，
  走新回写 → 解析结果与原帧完全一致。capture 相关 4 套测试全绿。**实弹 --skins 采集待重验**。
- ⚠️ **`captures/raw/ai_session2/run_1` 数据已污染**（GT 缺 authGame、metadata 无实际皮肤、
  牌局疑似客户端托管摸切打完），不要并入数据集，建议删除。

### 1.26 `build_datasets.py` 统一并行开关 `-j/--parallel`（2026-07-05）
- **动机**（用户）：`--workers`（阶段①标注）与 `--jobs`（阶段②每局建库）含义**基本相同**——
  都是"同时并行几局、每局一个 OS 进程、受内存约束"，只是作用在两个**先后串行**的阶段，
  可以合并成一个旋钮。
- **改动**：`build_datasets.py` 新增 `-j/--parallel N`，作为两阶段并行度的**共享默认**；显式
  `--workers`/`--jobs` 仍**各自覆盖**对应阶段（都不传则维持原默认：标注跟随 annotate 自身封顶 4，
  建库 `min(8, cpu//2)`）。折叠逻辑抽成纯函数 `resolve_parallelism()`（`--jobs` 默认由具体值改为
  `None`，落地由该函数兜底 `DEFAULT_JOBS`）。向后兼容——旧 `--workers 16 --jobs 12` 不变。
- **验证**：新测 `test_build_datasets.test_resolve_parallelism_shared_and_overrides`（共享/覆盖/默认
  各分支）；`--dry-run` 端到端确认 `-j 3` 同时进标注 `--workers 3` 与建库 `jobs=3`、覆盖优先、
  无旗默认不变。docstring + PIPELINE §3 SOP 同步。

### 1.27 GT jsonl 归入对局目录（嵌套布局）+ 一次性迁移（2026-07-05）
- **动机**（用户）：`run_N/gameM.jsonl` 孤悬在 `gameM/` 旁边（07-04 统一 GTRecord 时引入的形态），
  归入目录后每局一个**自包含**文件夹 `gameM/{gameM.jsonl, liqi.jsonl, frames.jsonl, frames/, metadata.json}`。
- **规则（只定义在 `paths.py`）**：嵌套 `X/X.jsonl ↔ X/` 为现行；旧"同名兄弟" `X.jsonl ↔ X/`
  降为 legacy 兼容读取（manual 会话仍用）。`frames_dir_for` 识别两种形态；新增反向
  `capture_for_frames_dir`（优先嵌套）；`ai_game_name`/`_ai_captures_in` 支持嵌套 + 迁移中态去重
  （两形态并存时只算嵌套）。写入方 `autoplay_ai.py`（GTWriter → game_dir 内）与
  `migrate_ai_to_gtrecord.py`（plan_targets）改为直接产出嵌套。`ingest_run.py` 改走
  `capture_for_frames_dir`；`annotate/cases.py`、各脚本示例路径同步。
- **迁移**：新一次性脚本 `scripts/data/migrate_gt_into_gamedir.py`（dry-run 默认、幂等、纯
  `os.rename` 同卷改名——**不用 shutil.move**，其 copy-fallback 在源文件被占用时会留下过期副本，
  实测踩到：迁移时恰有 autoplay 会话在写 `ai_session2/run_3/game4.jsonl`，锁文件改跳过+报告）。
  已执行：28 处中 27 移毕（ai_session 17 + ai_session2 9 + temp 1），剩 live 的 game4 待会话结束后
  重跑；`datasets/v1/games.json` 17 条 capture 路径同步改写（训练经清单读 GT，必须跟随）。
- **遗留发现**：`captures/raw/ai_session/run_1.jsonl`（ai_run_1 的 GTRecord）在磁盘上已丢失
  （07-04 建 v1 时还在），v1 清单该条目留原样并告警；`run_1/liqi.jsonl` 线流完好，可用
  `migrate_ai_to_gtrecord.py` 重新生成（现会产出嵌套 `run_1/run_1.jsonl`）。
- **验证**：TDD——`test_paths`（嵌套/兄弟/去重/反向）、`test_migrate_gt_layout`（计划/执行/清单
  改写/幂等）、`test_migrate_ai`、`test_build_datasets`、`test_autoplay_gt`（GTWriter 进 game_dir 的
  源码契约）、`test_downstream_rewire` 先红后绿；真实数据端到端（发现 17 局、replay 895 状态、
  帧解析正常）；全量 27 个测试文件通过。PIPELINE.md/CLAUDE.md/scripts/README.md 同步。

### 1.28 skins 元数据 `table` 误读 players[0] → 按 hero accountId 修正 + 回填（2026-07-05）
- **问题**（用户发现）：ai_session2 各局 `metadata.json` 的 `skins.table` 大多为空、个别非空但槽位
  怪异——尽管换肤明明每局都生效。逐局解码 `liqi.jsonl` 的 `authGame` RES 证实：**换肤确实局局生效**
  （hero 的 views 每局都被 mod.py 改写成随机的 {7,6,8} 牌背/桌布/场景），错的是记录——
  `extract_authgame_skins` 取 `players[0].views` 当 hero，而 RES 的 `players` 按 account_id 升序排
  （座位序在 `seat_list`），`players[0]` 基本是路人：空 = 路人没装扮；run_3/game2 非空 = 路人自己的
  装扮（误导）；run_3/game3 恰好对 = hero 碰巧 account_id 最小。
- **修复**：`extract_authgame_skins(data, hero_account=)` 按 accountId 找 hero，找不到/未给则 table
  留空（决不拿路人装扮冒充），命中时额外写 `hero_account_id`；`autoplay_ai.py` 从 `authGame` REQ
  （线流第 1 帧，含 accountId）记下 hero 传入。
- **回填**：新一次性工具 `scripts/capture/backfill_skin_meta.py`（dry-run 默认、幂等）：从每局
  `liqi.jsonl` 配对 REQ(msg_id)→RES，用 autoliqi 的 `liqi_pb2` 解码后走同一 extract 重写
  metadata。已对 ai_session2 全部 10 局执行（不影响 GT/frames/datasets——纯 provenance）。
- **验证**：TDD 先红后绿（`test_gamemeta` 新增 hero-not-first 用例、`test_backfill_skin_meta` 注入
  decoder 测 merge/dry-run/apply/跳过）；回填 dry-run 输出与手工线流解码逐局一致；全量测试通过。
  SKINS.md/PIPELINE.md §4 同步。

### 1.29 服务器侧检测集重建脚本切到嵌套布局 + 训练启动器上文档（2026-07-05）
- **背景**：远端 GPU 机的两个 bash 脚本随 merge 进主线——`scripts/train/launch_detector.sh`
  （`train_detector.py` 的 hbb/obb 多卡 DDP 单次训练包装：按变体挑好 dataset/基座/输出/run 目录，
  `CUDA_VISIBLE_DEVICES` 选卡 + `--gpus N` 定 DDP 卡数，默认 batch128/epochs60/imgsz1280）与
  `scripts/data/regen_detector_dataset.sh`（服务器上重建**扁平** `datasets/detector[_obb]`，
  OBB 复用 HBB 帧、只写 8 点标签 `build_dataset.py --reuse-images` 不重编码 ~17G）。README/PIPELINE
  此前均未记。
- **问题**（用户发现）：`regen` 仍按**已退役**的 `captures/intermediate/gt/*.jsonl` 发现对局
  （`ls` 那行），而现布局 AI GT 已嵌套进对局目录、`intermediate/gt` 已删——脚本在当前布局下扫不到局。
- **修复**：`regen` 的对局发现改为**复用 `build_datasets.discover_games`**（同一嵌套 `paths.ai_captures()`
  + 信箱 `FRAMES_OVERRIDE`，与版本化构建同源），产出 name/capture-jsonl/frames-dir/override 四列；
  annotate 阶段按 override 拆分（信箱局走 `--frames-dir` 指 de-letterboxed derived 帧，其余批量）；
  建库/装配用 capture 路径映射；`SOURCES=` 可改扫描根。**缺帧的局（如未 rsync 的 derived）大声丢弃、
  不中断**（避免深进并行建库才崩）。修了 Windows 下 Python CRLF 使末列 override flag 带 `\r` 的坑
  （末列 strip `\r`，Linux 上 no-op）。
- **验证**：`bash -n` 通过；发现块产出 18 AI 局、信箱 2 局正确路由 derived + override=1；
  本机缺 derived 帧时正确丢弃 2 局保留 16、val 在列；用 `datasets/v1/annotations` 跑真实
  stage-2（`ai_run_8_game2`，嵌套 capture+frames）产出 503 帧 503 标签 503 图，标注按
  `ai_game_name` 正确解析。README 训练节 + Layout、PIPELINE §2 装配+训练 同步。

### 1.30 检测器增强显式化 + 宝牌闪光覆盖统计（2026-07-05）
- **问题**：`train_detector.py` 是 ultralytics 薄封装、不传任何增强超参，全用 YOLOv8 默认
  （`fliplr=0.5`、`hsv_v=0.4`）。麻将牌有方向，`fliplr=0.5` 会造现实不存在的镜像牌；且雀魂
  宝牌有金色闪光特效，训练无任何针对性增强，也无证据判断自然覆盖是否足够。
- **处理**：(1) `tiles.py` 加 `next_of`/`dora_names`（标准 dora 递进，MJAI/canonical 双容）；
  (2) 新增一次性诊断工具 `scripts/inspect/count_dora_glow.py`，统计每类「发光实例/总实例」
  覆盖（红五恒亮 + 指示牌命中的牌算亮；hand/river/meld 区；帧级 = 训练 crop 数；按整局分
  train/val）；(3) `train_detector.py` 把增强超参提升为显式 CLI 参数并打印，默认改
  `fliplr=0→关`、`hsv_v=0.4→0.5`（亮度/闪光近似），其余沿用默认。真·局部 bloom 暂不做。
- **验证**：`tests/test_dora_glow.py`（next_of/dora_names/glow 规则）与
  `tests/test_train_detector_aug.py`（默认关 fliplr、hsv_v=0.5、可覆盖）均通过；
  `count_dora_glow.py` 在采集上跑通（红五行 glow%=100）。
- **结果**：检测器增强可复现、方向错误已修；宝牌闪光覆盖有了量化口径，是否投入合成 bloom
  等统计结果定。**注意**：改的是「下次」检测器训练的默认，不 stale 现有 `datasets/`；已训权重
  需重训才享受。

### 1.31 ai_session2（换肤）接入 + regen 并发竞态截断 OBB 标签 → 修复与两检测器重训（2026-07-05）
- **数据接入**：GPU 机新数据 `raw/ai_session/run_13,14`（早退迷你局）+ **`raw/ai_session2/`
  （`--skins` 换肤采集，10 局）**。`ai_game_name` 不含源根 → `ai_session2/run_3` 与主根 `run_3`
  撞名，按"run 编号跨源全局唯一"规约（PIPELINE §3）改名 `run_21..23`（`frames.jsonl` 是局目录
  相对路径，改名安全）。信箱局 derived 帧本机缺失，`deletterbox_frames.py` 从 raw 重建（563+265 帧）。
- **竞态 bug（本条核心）**：`regen_detector_dataset.sh` 把 56 个 (变体×局) 构建丢进同一 16 槽
  任务池，同局 OBB 的 reuse 检查在 HBB **写图中途**即通过（compgen 见到首张 png 就走
  `--reuse-images`，其帧集合=当时已存在的 png）→ 8 个大局共 **1092 个 OBB 标签静默截断**，
  val 局缺 218/949。无标签图=纯背景 → val 上真牌检出全算 FP，mAP50 被数学封顶
  33652/(33652+~9k)≈0.79（实测平台 0.78-0.79，val/cls 363~940 爆表）。排查依次排除了
  DDP（单卡同病）与标签内容（50.2w 行 OBB↔HBB 逐行 0 类错位、0 中心偏差、0 蝴蝶结）。
  **修复**：同局内变体串行（HBB→OBB 链）、跨局仍并行（28 链填 16 槽，并行度基本无损）；
  OBB 标签重建后 epoch1 val/cls 363→0.95，epoch2 mAP50 即 0.994。
- **build_datasets.py 加固**（用户点名；它无 reuse 竞态但有同族缺口）：新增 `verify_game_yolo`
  （images/labels 计数一致 + 每标签文件首个非空行字段数匹配变体 9/5）；`--resume` 从"目录存在
  即跳过"改为**校验失败自动重建**；stage-3 装配前全量校验，坏局 SystemExit 拒绝装配。
  `tests/test_build_datasets.py` 过；合成截断局/混格式局均被拦截。PIPELINE §3 已记。
- **ultralytics 两坑**（8.4.86）：(1) `select_device` 用 `--device` 字符串**覆写外部
  `CUDA_VISIBLE_DEVICES`** → 两个 4 卡任务落到同组卡，32 img/GPU（~22.3G）互挤 OOM；选卡必须
  直接给 `--device` 物理卡号。`launch_detector.sh --gpus` 改收物理卡号列表（兼容 count），
  默认 batch 128→64（16/GPU≈17G 安全），hbb 输出路径修为 `majsoul_eye/recognize/…`。
  (2) 相对 `--project` 会被嵌到 `runs/<task>/` 下（产出 `runs/obb/runs/obb/...`），
  `train_detector.py` 现传 `abspath`；既有 run 目录已归位 `runs/{hbb,obb}/`。
- **重训结果**（28 局扁平 `datasets/detector[_obb]`，train 10901 / val 949 = held-out
  `ai_run_8_game1`，4×4090 DDP batch64 imgsz1280 60ep）：HBB **mAP50 0.9940 / 50-95 0.9653**
  （基线 0.9937/0.9565）；OBB **0.9946 / 0.9848**（基线 0.9945/0.9804）→ 择优提升为正式
  `recognize/tile_detector.pt`（旧权重留 `weights/detector/*_0704.pt` 备份）。
- **遗留**：换肤局 dora 牌背的橙色 HSV 可靠性门失配（`annotate/frame.py` fill<FILL_OK → 丢框），
  换肤局每帧 ~4 个可见牌背成 YOLO 负样本（dora ok% 20-24 vs 素色局 93-100）；河/副露/手牌不受
  影响。若要 back 类覆盖换肤，需换肤无关的背面校验后重建+重训。

### 1.32 game 名按 source 根 basename 加前缀（拜托跨源 run 唯一限制）（2026-07-05）
- **动机**（用户）：`raw.7z` 原始布局 `ai_session2` 用 `run_1/2/3`，与 `ai_session/run_1,run_3`
  撞名——§1.31 为此把 `ai_session2` 落盘改名 `run_21..23`。目标：`build_datasets.py v2 --sources
  captures/raw/ai_session captures/raw/ai_session2` 直接吃**原始布局**，不再需要跨源改号。
- **根因**：撞名系统性——`discover_games` / `annotate_ai_session` / `build_dataset --from-annotations`
  三阶段各自调 `paths.ai_game_name(cap)`，它只看路径末两段（`run_N/gameM`）忽略源根 →
  两个 `run_1/game1` 都解析成 `ai_run_1_game1.jsonl`，标注互相覆盖（不止清单撞名）。
- **改动**：`ai_game_name` 前缀由固定 `ai_` 改为**源根目录 basename**（`run_N` 上一层）：
  `ai_session/run_8/game1 → ai_session_run_8_game1`、`ai_session2/run_1/game1 →
  ai_session2_run_1_game1`；manual `sessionN` 仍走 basename 兜底。三阶段同调一个纯函数 → 自动一致。
  第 4 个调用方 `migrate_ai_to_gtrecord.plan_targets`（弃用工具，自带重复命名）改为同调
  `paths.ai_game_name`（DRY）。`build_datasets.py` 随迁 `DEFAULT_VAL`
  (`ai_run_8_game1→ai_session_run_8_game1`)、`FRAMES_OVERRIDE` 键（值仍是磁盘物理目录
  `ai_run_5_game*_fixed` 不变）；`discover_games` 撞名分支改为"真重复"兜底（同一源根传两次才报错）。
  `count_dora_glow.py --val` 默认同步。
- **规范根也一起加前缀**（用户选 "prefix every root"，canonical `ai_session` 亦变 `ai_session_run_*`）：
  因 `datasets/`、`captures/` 均 gitignore 且无 `games.json` 版本清单，v2 全新自包含 → 无跨版本 val
  泄漏。既有磁盘 `datasets/precise_ai_run_*` 目录与按字面路径引用它们的测试**不动**（函数改名不搬
  目录）。§1.31 的 `run_21..23` 改名从此**非必需**（原始 `run_1..3` 可直建）。
- **删除**：`scripts/data/rebuild_datasets.py`（早已 DEPRECATED，被 build_datasets 取代）及其契约测试。
- **待跟进（cascade）**：`regen_detector_dataset.sh` 复用 `discover_games`（名已变）但硬编
  `VAL_GAME=ai_run_8_game1` → line 101 val 校验会失败，需改 `ai_session_run_8_game1`（该脚本是
  用户 WIP，本次未代改）。
- **验证**：TDD——`test_paths`（含 `ai_session2` 用例）、`test_downstream_rewire`、`test_build_datasets`
  （撞名测试翻转为"不同名 + 同一源根两次仍报错"）、`test_migrate_ai` 先红后绿；全量测试通过；
  `build_datasets.py v2 --sources ai_session ai_session2 --dry-run` 两根 28 局共存无撞名、默认 val
  `ai_session_run_8_game1` 命中。设计/计划：`docs/superpowers/{specs,plans}/2026-07-05-source-root-qualified-game-names*`。

### 1.33 换肤局 back 类可靠性门去皮肤化（tile_live_mask，修 §1.31 遗留）（2026-07-05）
- **根因**：`annotate/frame.py` 用**橙色 HSV** 判定牌背可靠性——dora 反面槽 fill 门、暗杠/加杠副露
  反面 fill 门都吃这个判定。§1.31 遗留已记：换肤局（`raw/ai_session2/`，`run_21..23`，`--skins`）
  牌背换色后橙色读数~0 → 全部判 `reliable=False` → 丢框 → 这三局 **`back` 类标注 0 个**（对比素色局
  2955–3807），未标的牌背区域反而当 YOLO 背景负样本喂进去，等于教模型"这里不是牌"。
- **修复**：`annotate/pipeline.py` 新增去皮肤化的 `tile_live_mask`——饱和度或亮度 `(S>60)|(V>110)`，
  只判定"这格有没有渲染出内容"（liveness），不做 face/back 二分类（commit `2b1405d`）；
  `annotate_frame` 的 dora 反面 fill 与副露反面 fill 门改吃它（commit `097d9c5`），从"只认橙色"
  变"任意肤色都可靠"。另一路 `tile_back_mask`（供 `snap_meld_strip` 做吸附阶段的 face/back 几何
  判别）也同步从橙色 HSV 改为纯饱和度 `S>70`（commit `612d659`），验证后判定 KEEP（未回退橙色，
  见下）。两个 mask 职责分离：`tile_live_mask` 管可靠性，`tile_back_mask` 管吸附判别，互不影响。
  牌面识别（河/副露正面）与 hero 手牌槽默认不换肤，本次未动、不受影响。
- **验证**：换肤局 run_21 dora 反面可靠率（全局帧）**0/1048 → 1048/1048**（修前全不可靠，修后全可靠）；素色局 run_8 无回归——
  dora 反面 3741/3741、副露反面（暗杠/加杠）110/110，均 100%；`tile_back_mask` 改动后素色局吸附
  （snap）像素偏移 max **2.00px**（<3px 判定门槛，110 个暗杠反面格采样）、**0 个可靠性翻转**。
- **待办（stale data）**：`datasets/precise_ai_run_21..23`、`datasets/obb_precise_ai_run_21..23`
  （以及任何含这几局的合并/聚合数据集）对 **`back` 类**现已过期，须重建——annotate 是过期的一环，
  `bash scripts/data/regen_detector_dataset.sh`（或 `build_datasets.py <name> --force`）重建后
  再重训检测器。重建/重训超出本任务范围，由用户后续触发。

### 1.34 run_5 信箱局就地去黑边 → 撤掉 `FRAMES_OVERRIDE` 特判（2026-07-05）
- **根因/遗留**：`run_5` 掉线重连局 game2/game3 采集时浏览器窗口非精确 16:9，Majsoul 上下补黑边，
  原始帧是 **1923×1142**（其余局干净 1920×1080）。标定同伦假设 1920×1080，信箱局标注会有随边缘
  增大的竖直偏移（~28px）。原方案：`deletterbox_frames.py` 把两局去黑边写到 `intermediate/derived/
  ai_run_5_game*_fixed`，`build_datasets.py` 用 `FRAMES_OVERRIDE` 把这两局指向 derived 修复帧
  （annotate/建库都特判）。痛点：derived 是可再生中间产物、跨机常缺失（`本机缺失`/`未 rsync`），
  规则散落在 `build_datasets.py`、`regen_detector_dataset.sh` 两处特判里。
- **修复**：给 `deletterbox_frames.py` 加 `--inplace` 模式——直接就地改写源 PNG（1920×1080），
  `frames.jsonl` 原样不动（`file` 路径已指向这些 PNG）。幂等（二次运行认不出黑边、原样重写）。
  在 `captures/raw/ai_session/run_5/game2`（563 帧全裁）与 `game3`（264 裁 + 1 全黑过渡帧 resize）
  就地执行，两局现与任何干净局字节同构。随后**彻底删除 `FRAMES_OVERRIDE`**：`build_datasets.py`
  的 dict + 两处特判分支（`discover_games` frames_dir、annotate 批量拆分）、`regen_detector_dataset.sh`
  的 `FRAMES_OVERRIDE` 导入 + override 列 + `OV_OF`/per-game `--frames-dir` 标注拆分全部收敛为
  "所有局走自己的嵌套帧目录、一次批量标注"。`annotate_ai_session.py` docstring 去掉 derived 例子。
- **验证**：两局 828 帧全部 1920×1080（逐帧核验）；`--inplace` 分支先在 scratchpad 拷贝上验证
  （尺寸/索引不动/缺帧计数/幂等，PASS）；`bash -n regen` 过、`py_compile` 全过；
  `tests/test_build_datasets.py` 8/8 绿（`test_frames_override_applies` 重写为
  `test_letterboxed_games_use_own_frames_now`：断言 `FRAMES_OVERRIDE` 已不存在且 run_5 局解析到自身
  嵌套帧目录）。captures 已 gitignore，PNG 改写不进 git。
- **数据影响**：就地修复帧与旧 derived 帧同算法同像素，故已训 `datasets/v1`（本机无）不 stale；
  服务器 tar-and-go 现只需 rsync raw（无需单独同步 derived）。PIPELINE.md 三处 + 本条同步更新。

### 1.35 `launch_detector.sh` 切版本化 `datasets/<name>` 布局 + build 收尾输出更新（2026-07-06）
- **根因**：数据集版本化后，扁平 `datasets/detector`/`datasets/detector_obb` 被整体搬进
  `datasets/v0/`，而 `build_datasets.py v2 --sources captures/raw/ai_session captures/raw/ai_session2`
  产出的检测集在 `datasets/v2/detector/`。`launch_detector.sh` 仍硬编码扁平 `datasets/detector*/data.yaml`，
  两个默认路径全部落空（`[ -f "$DATA" ]` 直接报 MISSING）。
- **修复**：
  - `launch_detector.sh` 加 `--dataset NAME`（默认 `v2`）：裸名→`datasets/NAME`、含 `/` 直接当目录、
    `*.yaml` 逐字当 data.yaml（兼容扁平 regen 布局）。变体只定 split **子目录**（`hbb→detector`、
    `obb→detector_obb`）+ 基座/输出/run 目录；`DATA` 在选项解析后由 `--dataset`＋子目录拼出。
    启动日志新增 `dataset:` 行；MISSING 提示改指 `build_datasets.py`（并保留 regen 兜底）。
  - `build_datasets.py`：OBB 检测集从 `<ds>/detector` 改名 `<ds>/detector_obb`（与 `regen_detector_dataset.sh`、
    `datasets/v0` 统一，且 `--dataset` 能各自定位）；收尾"训练命令"块从裸 `train_detector.py --data …`
    改为首选 `bash scripts/train/launch_detector.sh {hbb|obb} --dataset <name> --gpus 4`（保留裸命令兜底），
    并加 `BUILT …（detector split → detector[_obb]/）` 头 + 跨版本合并 split 示例改用 `--out datasets/combined/<split>`
    → `launch_detector.sh --dataset datasets/combined`。
- **验证**：`bash -n launch_detector.sh` 过；DATA 解析逐形式核验（裸名/目录/`*.yaml`、hbb↔detector、
  obb↔detector_obb，默认 `hbb` 命中现存 `datasets/v2/detector/data.yaml`）；`build_datasets.py … --dry-run`
  与 `--obb --dry-run` 两变体收尾输出正确（HBB→detector、OBB→detector_obb）；`py_compile` 过；
  `tests/test_build_datasets.py` 与 `tests/test_train_detector_aug.py` 全绿。PIPELINE.md 检测器/launch 两处同步更新。
- **数据影响**：无（v2 已建、HBB 名 `detector` 不变；改名只作用于未来 `--obb` 版本）。

### 1.36 `--val` 可重复：一次留出多整局作 val（train/val split 微调）（2026-07-06）
- **根因/需求**：v2 已建，需在既有 held-out `ai_session_run_8_game1` 之外**额外**把
  `ai_session2_run_21_game1` 也留作 val（换肤局的跨局泛化探针）。原 `--val` 三处
  （`build_datasets.py` / `build_detector_dataset.py` / `train_classifier.py`）**均只支持单局**
  （`val_name, val_kyoku = args.val.split(":")` → `name == val_name`），无纯 CLI 办法加第二局。
- **修复**（向后兼容，单局用法不变）：
  - `build_detector_dataset.py` + `train_classifier.py`：`--val` 改 `action="append"`；新增纯函数
    `parse_val_specs(specs)→{name: val_set}`（`'*'`/kyoku 集/空）；`split_images` 签名从
    `(sources, val_name, val_set)` 改 `(sources, val_map)`，`want_val = name in val_map`。
    分类器同改多局判定，并顺带只对需 kyoku 粒度的 val 局才 `seq_to_kyoku`（非 val 局不再空跑重放）。
  - `build_datasets.py`：`--val` 改 `action="append"`；新增 `resolve_vals(val_arg, names, default)`
    （None→`[DEFAULT_VAL]`、逐名校验在已发现局中、保序）；stage-3 每局展开一条 `--val NAME:*`；
    `games.json` 的 `val` 改写为**列表**（`write_manifest` 签名 `vals: list`）；收尾示例命令按多 val 打印。
  - `count_dora_glow.py`：读端容忍 `val` 为列表或旧单字符串（`val_set = set(...)`，桶归属改 `in`）。
- **重建 v2 detector split**（无需重标/重裁，仅 stage-3）：
  `build_datasets.py v2 --stage detector --sources captures/raw/ai_session captures/raw/ai_session2
  --val ai_session_run_8_game1 --val ai_session2_run_21_game1 --resume`
  → val 949→**1211** 图（+262 = run_21_game1 整局），train 10901→10639，两 split 局集**无交叠**；
  `games.json` `val` = 两局列表。⚠️ **`--sources` 必须与初次构建一致**（脚本从 sources 重新发现局，
  不读清单）——只给默认单根会漏掉 `ai_session2`，`--val` 校验报"未在已发现局中"（本次已踩坑并记入 PIPELINE.md）。
- **验证**：先按 TDD 写 4 个失败用例（`test_detector.py` 多局 `split_images`/`parse_val_specs`；
  `test_build_datasets.py` `resolve_vals` 校验 + 分类器 `parse_val_specs`）看红 → 实现 → 全绿；
  `tests/test_*.py` **34/34 全绿**（含 `test_count_dora_glow` 覆盖新清单 schema）。PIPELINE.md 切分节 + `games.json` 说明同步更新。
- **数据影响**：`datasets/v2/detector/{train,val}.txt` + `datasets/v2/games.json` 已就地重建（见上计数）；
  分类器 val 为训练期参数、无物化，重训时按新命令传两 `--val` 即可。其他版本（v0/v1）未动。

### 1.37 `build_datasets.py` 一次出双格式：`--hbb --obb` 同版本内 detector/ + detector_obb/（2026-07-06）
- **需求**：v2 是 HBB-only（`--obb` 默认关），生产训 OBB 时没有带新双-val 的 OBB split。要一条命令
  同时产出两套,且**不重编码 ~17G 帧**（OBB/HBB 帧字节相同）。此前扁平 `regen_detector_dataset.sh --obb`
  能双出但是**单-val + 非版本化**；版本化 `build_datasets.py` 每次只出一种格式。
- **实现**（`build_datasets.py`，向后兼容）：
  - `--obb` 保留（单独=仅 OBB，历史布局 `<ds>/<game>/yolo`）；新增 `--hbb`；`resolve_formats(hbb,obb)`
    → `['hbb']/['obb']/['hbb','obb']`（HBB 在前）。
  - **双出布局**：OBB 落兄弟目录 `<ds>/<game>__obb/yolo`，`images` 用**相对软链**指回 HBB（`Runner.symlink`，
    tar-and-go 可移植），只写 9 点标签（`build_dataset.py --obb --no-crops --reuse-images <hbb>/yolo/images`）。
    `game_yolo_dir(ds,dir,fmt,formats)`：OBB 仅在与 HBB 共存时加 `__obb` 后缀，单出仍用 `<game>`（保历史布局＋crops 路径）。
  - **stage-2 分格式两趟**（HBB 全建完 barrier 后再 OBB——reuse 依赖 HBB 帧先落盘；`run_parallel` 天然是 barrier）；
    **stage-3 按 formats 循环**装配 `detector/`＋`detector_obb/`，双-val 两套都带，verify 按 5/9 字段各校验。
  - `games.json` 加 `formats` 字段（`write_manifest(...,formats)`）。
- **应用到 v2**（`--hbb --obb --resume`，不 `--force`——跳过已验证的标注＋HBB，只增量建 OBB＋重装两套）：
  28 局 OBB 全 rc=0；`detector/`（5 字段）与 `detector_obb/`（9 字段，images 软链 HBB）各 train **10639**/val **1211**；
  OBB val 两局无交叠、软链图可解析（1.87MB 真 PNG）；`games.json` `formats=[hbb,obb]`、`val` 两局。**零帧重编码**。
- **验证**：TDD 先写 `resolve_formats`/`game_yolo_dir` 失败用例看红→实现→全绿；`tests/test_*.py` **34/34**；
  `py_compile` 过；默认(HBB)/`--obb`(单 OBB)/`--hbb --obb`(双) 三条 `--dry-run` 命令与软链路径逐一核验。
  PIPELINE.md 装配节＋清单说明同步。
- **数据影响**：`datasets/v2/` 现含两套 split＋28 个 `<game>__obb/`（OBB 标签+软链，无额外帧）；HBB `detector/` 内容不变。
  历史坑（未修，已知）：`--obb --resume` 在**已 HBB 建成**的版本上会因 `<game>` 目录里是 5 字段而判 REBUILD→
  覆盖成 OBB；要双出请用 `--hbb --obb`（各自目录，不冲突）。

### 1.38 分类器纳入启动器框架 `launch_classifier.sh` + 现役切 v2 + 清理过时一次性脚本（2026-07-06）
- **需求**：检测器有 `launch_detector.sh` 多卡包装，分类器却只能手敲 `train_classifier.py --dataset ... --val ...`；
  且 README/PIPELINE 十几处仍写 `datasets/v1`（实跑已是 v2），示例命令带 `mannual` 错路径、`--out ...#` 粘连注释、
  检测器 "batch 128"（实际默认 64）。按当前实跑管线（`build_datasets.py v2 --hbb --obb` →
  `launch_detector.sh hbb/obb --dataset v2`）核对统一。
- **实现**：
  - 新增 `scripts/train/launch_classifier.sh`——单卡包装（分类器是小 CNN，无 DDP，用 `--gpu` 经
    `CUDA_VISIBLE_DEVICES` 选卡，与检测器 ultralytics 覆写 CVD 相反）。**不传 `--val` 时读
    `datasets/<name>/games.json` 的 `val` 列表、逐局 `--val <game>:*` 留出**，与检测器 split 同一批留出局
    （零手动同步）；`--dry-run` 预览；`--` 后透传 train_classifier.py。
  - 文档核对：README + PIPELINE 全量切 v2；修上述三处 README bug；训练节改用三条启动器命令；
    PIPELINE §5 快照重写为 v2 实况；scripts/README.md 重写为当前脚本清单（补 build_datasets/launch_*/regen 等）；
    DATA_AUTOMATION.md 加「历史设计（已被 Mortal autoplay + annotate/ 取代）」横幅。
- **数据现状（v2）**：28 局纯 AI（18 ai_session + 10 换肤 ai_session2），held-out 2 整局（+换肤 run_21_game1）；
  早期手动 session5/6 退出训练集（AI-only 基线）。检测器 07-06 v2 重训 HBB best mAP50 **0.992**/50-95 **0.957**、
  OBB **0.994**/**0.981**；分类器**尚未** v2 重训（仍 07-03 权重）。
- **清理**（删除已完成的一次性脚本 + 专属测试）：`ingest_run.py`（被 build_datasets 取代）、`migrate_ai_to_gtrecord.py` /
  `migrate_gt_into_gamedir.py` / `migrate_captures_layout.py`（迁移已完成）、`backfill_skin_meta.py`（回填已完成）；
  `test_downstream_rewire.py` 去掉两条 ingest_run 断言（保留其余 3 条）；PIPELINE §4 标「已删除」、CLAUDE.md/SKINS.md
  去悬挂引用。转换能力仍在保留的 `convert_mjcopilot.convert_game`。
- **验证**：`launch_classifier.sh --dry-run` 三情形（默认自动 val=两留出局 / `--val` 覆盖 / cpu）+ `bash -n` 过；
  编辑后 `tests/test_*.py` 全绿（删脚本时同步删对应测试，见交付的 rm 清单）。

### 1.39 检测器权重版本化落盘 + OBB 自动提权为现役默认（2026-07-06）
- **需求**：`launch_detector.sh` 每次 run 直接覆盖固定的 `recognize/tile_detector.pt`（HBB）/
  `weights/detector/tile_detector_obb.pt`（OBB），跨 run 无法并存对比；且现役运行时想默认用 **OBB**
  （旋转座/立直横放贴合更好），却要手动 promote。
- **实现**：
  - `launch_detector.sh`：输出统一为**版本化** `weights/detector/tile_detector_<mode>_<name>.pt`
    （`<name>`＝run 子目录，默认时间戳，各 run 不互相覆盖）；`OUT` 从 mode case 移到 `--name` 解析之后拼装。
    **OBB 是现役默认**，故额外 `--also-out majsoul_eye/recognize/tile_detector.pt`（现役运行时那份）；
    HBB 只留版本化副本。启动日志 `best ->` 行对 OBB 追加 `(+ …/recognize/tile_detector.pt)`。
  - `train_detector.py`：加 `--also-out`（`append`，可重复）；best.pt 复制改为对 `[--out, *--also-out]`
    逐个 `makedirs`（dirname 为空则跳过）+ `copy`，一份 best 扇出到全部落点。
  - 文档同步：README 训练命令注释、scripts/README 输出列、weights/README（命名约定 + 免手动 promote）、
    PIPELINE（一图流 + §2 launch_detector 详解两处）、`autoplay_ai --overlay` 的 `--detector-weights` help
    （默认已是 OBB → 画旋转多边形）。
- **验证**：`bash -n launch_detector.sh` 过；`PY=echo` 干跑 hbb/obb 两模式核对组装命令
  （HBB 仅 `--out …_hbb_<ts>.pt`；OBB `--out …_obb_<ts>.pt --also-out …/recognize/tile_detector.pt`）；
  `train_detector.py --help` 含 `--also-out`；空数组 `"${ALSO_OUT[@]}"` 在 `set -u` 下不 unbound（bash 5.1，
  与既有 `PASSTHRU` 同惯例）。
- **数据/权重影响**：改的是**未来 run** 的命名与落点，代码不动老权重。**现役默认已切 OBB**：
  已把 `runs/obb/20260706_014911/weights/best.pt`（OBB mAP50 **0.994** / mAP50-95 **0.981**，
  v2 held-out 2 局含 1 换肤局）cp 至 `recognize/tile_detector.pt`（本地、gitignore，不入本提交）；
  README 检测器指标同步为「recognize=OBB、HBB 转版本化变体 `tile_detector_hbb_<ts>.pt`」。
### 1.40 采集截图黑边 bug：CDP 截图 clip 到视口 + run_5 离线裁回（2026-07-05）
- **现象**（用户发现）：`ai_session2/run_5` 全部 2477 帧尺寸为 **1958×1142**（另 game9 早期 9 帧
  1958×1097），右/下有纯黑边；session2 其余 run（1/2/3/4）均干净 **1920×1080**。实测每帧游戏内容都
  钉在**左上 1920×1080**，黑边纯黑（max≤16），内容无泄漏、无截断。
- **根因**（Playwright 三态复现坐实）：`screenshot_png` 走 CDP `Page.captureScreenshot`
  `{captureBeyondViewport:False}`，**抓的是页面 render surface**。Playwright 把布局视口
  （`window.innerWidth`）用 device-metrics override 钉死在 `browser_width×browser_height` = 1280×720
  CSS（Majsoul WebGL canvas 随之固定，左上角，真机 DPR 1.5 → 1920×1080 device px）。**关键**：启动时
  Playwright 把 render surface 也设成 = 视口，故 captureScreenshot 出 1920×1080 干净图——即使可见窗口
  略大（那圈是你**看得到但没被截进去**的 chrome/留白）。触发不是"窗口比视口大"（实验 A：全新启动窗口
  1296×854 比视口高，截图仍 1920×1080 无黑边），而是**启动后任何 resize**：手动拖、开成最大化、或从
  persistent `user_data_dir` 复原成大窗——Chromium 会把 render surface 重设成窗口大小（布局视口仍钉死
  1280×720），captureScreenshot 于是抓到放大后的 surface + 黑边。实验：A 全新启动=1920×1080 无边；
  B 最大化=3840×1958 有边；C 拖到 1400×900=2079×1209 有边（三态布局视口恒 1280×720、内容恒
  (0,0,1919,1079)）。窗口尺寸被 profile 记忆，某次 resize 后**后续启动复原成大窗**，故整个 run_5 从头
  带黑边（run 内 1142/1097 两高度 = 中途又 resize 过）。`ensure_res_framing`（`patch_majsoulmax.py`）是
  **protobuf 帧封装**，与窗口分辨率无关，此前记忆把两者名字搞混了。
- **修复**：`autoplay_ai.py` 截图加 `clip={x:0,y:0,width:browser_width,height:browser_height,scale:1}`。
  scale=1 让 native DPR 直出（1280×720→1920×1080），且裁掉黑边——**截图与窗口尺寸解耦**，拖窗/复原大窗
  都不再有黑边。实验证 clip 在被拖大的窗口上稳定产出 1920×1080。点击不受影响（Mahjong­Copilot 用 emulated
  视口坐标，实验证 override 下坐标恒定）。健康路径输出不变（无回归）。
- **run_5 数据修正**：一次性脚本裁全部 2477 帧回左上 1920×1080（含 9 张 1097），带守卫：仅当窗口≥视口
  且视口外全黑才裁，否则跳过报警——0 泄漏 0 截断。裁剪无重采样，内容逐像素等价。session2 现全 8150 帧
  统一 1920×1080。run_5 尚未进任何数据集（`ai_session2` 不在 `datasets/v1`），裁剪时机安全。
- PIPELINE.md 采集节已记 clip 行为。

### 1.41 HUD 检测：55 类 taxonomy + 微读取器 + 按钮/字段标定 + 采集侧扩展（代码全通，训练待排）（2026-07-06）
分支 `feat/hud-detection`（16 个任务，T1–T16，含中途追加的 T15/T16）。设计稿见
`docs/superpowers/specs/2026-07-04-hud-detection-design.md`（§1.20 已记，未实施）；本节记
**代码落地**——训练本身按计划推迟到用户另一分支完成数据集合并之后。

**55 类 taxonomy**（`majsoul_eye/hud.py`）：冻结的 38 牌类之后追加 17 类，`DET_NAMES = TILE_NAMES
+ HUD_NAMES`。按 WHERE×WHAT 来源分三组：
- **中心信息面板 7 类**（`score_self/right/across/left`、`round_label`、`wall_count`、
  `seat_wind_self`）：WHERE = 标定种子 ROI（`coords.HUD_SEEDS`）+ 逐帧墨迹收紧（`annotate/hud.py`
  的 `ink_snap`，`INK_THRESH=120`）；WHAT = `BoardState` 派生字符串（`annotate/hud.field_texts`：
  `scores`/`bakaze+kyoku`/`left_tile_count`/`(hero_seat-oya)%4`）。
- **左上角面板 2 类**（`riichi_stick_count`/`honba_count`）：同上机制，WHAT = `kyotaku`/`honba`。
- **动作按钮 8 类**（chi/pon/kan/riichi/tsumo/ron/kyushu/skip）：WHERE = `coords.BTN_ZONE` 内的
  亮度候选（`locate_button_candidates`，x 排序）；WHAT = 新提取器 `state/ops.py`
  （`ops_from_record`，从 `raw_liqi.data.data.operation.operationList` 解析待决操作类型）→
  `BoardState.pending_ops`（`state/replay.py` 新字段，`Replayer.apply_record` 末尾赋值——
  `start_kyoku` 会整体替换 `self.state`，早赋值会被静默丢弃）→ `hud.buttons_for_ops` 映射为按钮类。
  **检出数 ≠ 期望数则整帧按钮标签丢弃**（`flag:count_mismatch`，宁缺毋滥）。

**字段标定结果**（Task 6，9 个种子 ROI，跨 run_3/4/5/7/8/13/14 约 30 帧/局核对）：2 处发现是
**错框到别的元素**并重新定位（`round_label` 原来框住了 `score_across` 的倒转数字上沿；
`seat_wind_self` 原来框在面板空白装饰上，实际角标在 `score_self` 下方）；`INK_THRESH` 150→120
（青色文字灰度峰值只有 ~171，150 只框到抗锯齿最亮边缘）。收紧后残留误差 **flag rate 0.26%**
（标准 16:9 帧）；`run_5/game2,3`（1923×1142 非 16:9 信箱局）例外，属已知 `AnchorLocator` TODO
限制，非本次标定问题。

**按钮标定 + 采集盘点**（Task 3/4/7）：`scripts/inspect/inventory_ops.py` 对 v1 全量盘点：
offers=1714、button-records=493 全部有对应帧；chi 253/pon 151/skip 493/riichi 48 样本充足，
**ron 23/kan 22/tsumo 14/kyushu 1 稀缺**（一局最多出现几次）。用 `autoplay_ai.py --op-delay 1.5 2.5`
（拉长 hero 待决操作的点击延迟，参见下）针对性 harvest 出 `ai_session3/run_1`：22 帧全部框到
（chi12/pon8/kan3/ron2/riichi1/skip22）。用这批真帧标定 `BTN_ZONE`：旧种子猜测
`(0.30,0.66,0.98,0.82)` 会连带框进一个每帧恒在的"座位数+20"提示条（100% count_mismatch）+
偶发樱花标；收紧到 `(0.30,0.705,0.74,0.82)` 后 **0/22 count-mismatch**，`BTN_THRESH`/`BTN_MIN_AREA`/
`BTN_ORDER_LTR`（屏幕左右序 == `buttons_for_ops` 的 `HUD_NAMES` 序）原样成立，含 3 按钮的
pon+kan+skip、chi+ron 帧均验证。已知限制：≥4 按钮行的第 4 个 banner 会落进被排除的提示条区域
（未见过真实样本，出现时安全退化为 count_mismatch 丢弃）。

**T15：capture 侧 multi-shot + dt 时间戳**（`majsoul_eye/capture/multishot.py` 的 `MultiShot`）：
鸣牌（chi/pon/daiminkan/ankan/kakan/nukidora）与"待决操作会出按钮"这类时序不确定的事件后，
按固定偏移（默认 0.6/1.2/2.4s）额外截图 `{seq:06d}_dt{ms:04d}.png`，`frames.jsonl` 对应行
`status:"extra"`（纯增量，现有 `"ok"`/`"timeout"` 消费者不受影响、下游默认不可见不产出标注）；
`ok`/`extra` 每行都新增 `dt`（截图相对触发事件的秒数）。真实采集验证：`ai_session3/run_1`
332 ok + 92 extra 全部带 dt（实测 0.77–2.85s）。副产物修复：`write_gt` 的 `syncing` 曾硬编码
`False`——线入 MahjongCopilot `GameState.is_ms_syncing`（`_resolve_syncing_flag`），重连回放不再
被误记成新鲜 GT（DESIGN §7 的双计风险）。为解锁本任务的采集验证，把用户当时未提交的
`_shot_clip`（CDP 截图 clip，见 §1.40）WIP 一并原样提交。
**最佳帧选择器**（从 extras 里挑最有代表性的一帧，恢复本被丢弃的 call/deal 窗口帧）已规划为
**后续 owned follow-up**，尚未实现。

**T16：`is_call_window` 丢帧谓词**（`state/replay.py`）：`last_event` 为 chi/pon/daiminkan/ankan/
kakan/nukidora（鸣牌动画中途，GT 已更新但像素未跟上）时整帧丢弃，与 `is_deal_window` 同策略，
接入 `annotate_ai_session.py` + `build_dataset.py`（`n_call` 统计）+ `qa_hud.py`。实测
`run_3/game1`：30/719 = **4.2%** 丢弃，全部单帧命中（无过匹配）。已知限：同一条 GTRecord 打包
多个事件（如 鸣牌+摸 dora 同帧）时 `last_event` 会读到后一个事件，产生假阴性——留给上面的
best-shot selector 一并解决。

**detector.py 55 类解析修复**（`b021a49`，T13 复审发现的跨任务 bug）：`Detection` 原先只有
`.tile`（无条件 `TILE_NAMES[cls]`），任何 HUD 类 id（38–54）都会 `IndexError`；改为 `Detection.name`
（55 类全有效，来自 `hud.DET_NAMES`）+ `.tile`（HUD id 为 `None`），`_parse_result` 的 OBB/HBB
两条路径都跳过 `cls >= 55` 的未知类而非崩溃；`qa_hud.py`/overlay 等消费者已审计改读 `.name`。

**`datasets/v3` 流程验证 build**（T10，重定范围）：真正的训练数据合并（v1+v3 混训）由用户另一
分支负责，本分支只建一个最小 build 验证整条 HUD 管线走得通——`ai_session3` 2 局
（`ai_run_1_game1`/`ai_run_2_game1`，val=后者）：`nc=55`；`detector/{train,val}.txt` 每个字段类
（38–46）各 **694** 条（train 321 + val 373，按局切分）；按钮类（47–54）**精确匹配**上面
采集盘点的线上库存（chi12/pon8/kan3/riichi1/ron2/skip22）；HUD 读取器训练对 **6240** 行
（`hud/labels.jsonl`：`ai_run_1_game1` 2885 + `ai_run_2_game1` 3355）；0 条 extra-shot 泄漏进
train/val（多帧采样的 `status:"extra"` 帧未被 build_dataset 消费）。

**⚠️ PENDING（本分支不做，等用户合并数据集后再跑，命令见 PIPELINE.md）**：
`train_hudreader.py`、`train_detector.py`（55 类 v2 权重）、`eval_detector_split.py`（牌面组
mAP50 回归门槛 `0.993−0.005`）、`qa_hud.py` 真实端到端 QA（目前只验证过 `--selftest`，用假
检测器/读取器证明组装/比对逻辑正确，不代表真实精度）。`majsoul_eye/recognize/tile_detector.pt`
（生产权重）与 `hud_reader.pt`（尚不存在）都还是 38/无——55 类不是"已交付识别能力"，只是
"管线已就绪，数据待合并"。

**T17b：立直棒（reach_stick）真帧标定，55→56 类**（2026-07-06）：spec §10 把立直棒从"按座位 4 分类"
改成**单一对称类**（用户驱动的修订——self/across 互为 180° 旋转、left/right 互为镜像，检测框内
**外观退化**到只能靠绝对屏幕位置分辨，不泛化），T17a/17c 落的四槽种子框只是粗猜，未经真帧验证。
本次在 5 局（含 1 局不同桌布皮肤）真实 reach-accepted 帧上用 connected-components 重新定位：
`self`/`right`/`left` 在几十帧上像素级一致，`across` 因不同玩家装备的立直棒**皮肤各异**（素白棒/
针管+心形吊坠/紫色发光箭头）验证同一片屏幕区域被三种外观轮流占用；旧 `across` 种子框实际歪框在
河尾的常亮装饰边（有无立直都 fill 0.68–0.93），已重新定位到真实棒子位置。朝向算术
（`recognize.hudstate._attribute_slot`）用标定后的四槽中心 vs `round_label` 种子中心验证全部正确
归位：self dy=+122→下方、across dy=−72→上方、left dx=−140→左、right dx=+137→右。`REACH_FILL_OK`
0.05→**0.35**（self 稳定态 fill 最低 0.735、left 最低 0.428，二者真实捕获到的宣言滞后帧分别为
0.334/0.0，0.35 卡在中间；代价是 `across` 半数暗皮肤帧被保守标 `reliable=False`，宁缺毋滥）。
⚠️ 意外发现：`state.replay.is_score_anim_window`（`annotate/frame.py` 的第二道帧级门）对这些真实
宣言帧从未触发过——Majsoul 把 `reach_accepted` 和下一家的 `tsumo` 打包进同一条 GTRecord，
`last_event` 总被覆盖成 `tsumo`，所以这道 fill 检查不是"双保险"，是唯一真正拦住这些帧的机制。
**已修复（Task 18，2026-07-06）**：`BoardState.last_event_types`（记录本次实际应用的全部事件类型
集合，而非只留最后一个）让 `is_score_anim_window`/`is_call_window` 对打包记录也能正确触发。
顺带修了 `scripts/inspect/overlay_hud.py` 的自污染 bug：`cv2.rectangle` 原地画种子框后才调用
`hud_field_boxes`/`reach_stick_boxes`，导致同坐标的黄框像素反向污染了自己要可视化的那次测量
（实测把一帧真实滞后帧的 fill 从 0.334 推到 0.37，越过阈值，红框被误显示成绿框）——改为先在
干净帧拷贝上算完所有框，再统一画。**55→56 类**（`hud.DET_NAMES`）在 PIPELINE.md/CLAUDE.md 同步。

### 1.42 `build_datasets.py` 双出在 Windows 无软链权限时回退目录 junction（2026-07-06）
- **症状**：Win 下 `build_datasets.py <name> --hbb --obb`（OBB 局 `images` 软链回 HBB，见 §1.37）在
  `Runner.symlink` 的 `os.symlink(...)` 崩 `OSError [WinError 1314] A required privilege is not held
  by the client`。根因：Windows 建符号链接需 **开发者模式或管理员权限**（`SeCreateSymbolicLinkPrivilege`），
  本机两者皆无；纯 Linux 建的这条管线从没碰到过。POSIX 与非双出（HBB-only / 仅 `--obb`）路径不受影响。
- **修复**（`scripts/data/build_datasets.py`，跨平台、向后兼容）：
  - `Runner.symlink` 仍**优先相对软链**（tar-and-go 可移植）；`os.symlink` 抛 `OSError` 且在 Windows 上时，
    回退 `_winapi.CreateJunction(abspath(target), link)` 建**目录 junction**——无需权限，`glob`/`listdir`
    透明穿透（stage-3 `build_detector_dataset.py` 的 `glob images/*.png` 照常枚举）。代价：junction 存
    **绝对**目标路径，故该版本**宿主本地**（换机重建而非搬运）。非 Windows 仍原样 `raise`。
  - 新增 `_remove_dir_link()`：替换旧 `islink→unlink / isdir→rmtree` 删除逻辑。**关键**：junction 既非
    `islink` 又是 `isdir`，旧逻辑会对它 `shutil.rmtree`——Py3.12 下这会**抛 "Cannot call rmtree on a
    symbolic link"**（幸而不会穿透删掉共享的 HBB 帧）。新逻辑对 symlink/junction 一律 `os.rmdir`/`os.unlink`
    原地摘除、**绝不递归进目标**，只有真实目录才 `rmtree`。`--resume` 重跑与 `--force`（整 `ds` rmtree，
    junction 在其内）均已实测安全。
- **数据影响**：无。junction 与软链产出等价树；既有 `datasets/*` 不失效、无需重建。同版本换平台重建即可。
- **验证**：`auto` env（Py3.12.12）实探——symlink 崩 1314、junction 免权限成功、`rmtree` 拒穿透 junction；
  再以真实 `datasets/<ds>/{game,game__obb}` 布局跑补丁后的 `Runner.symlink` 五情形全绿（首建/`--resume`
  重建后 HBB 帧存活/`glob` 穿透可见/dry-run 不落盘/`--force` 整树删）。PIPELINE.md §双出布局同步。

### 1.43 auto-next 卡死诊断插桩（`--auto-next-debug`，2026-07-07）
- **症状**：`autoplay_ai --live --auto-next` 局末卡在 `rematch dialog -> clicking はい ... dialog still up
  after clicking はい; retrying` 无限循环，用户附的尾帧是**终局排名屏（有 確認）**。
- **已证据化定位（尚未定根因，故本节只插桩不改判定逻辑）**：把用户尾帧（1914×1074，≈16:9）喂进
  `button_guard` 谓词实测得 `confirm=0.397(在)`/`dialog_yes=0.027(不在, 阈 0.03)`/`dialog_no=0.239(在, 蓝
  排名条)`——**该屏应走 settlement 分支点 確認**，不会卡。而日志里的 `dialog_yes≈0.10 + conf 缺席` 对不上
  这一帧 → **卡死时代码看的是另一帧**（截图非同一时刻）。`screenshot_png` clip 到 1280×720 恰好 16:9，故
  非取景/信箱问题。两条主嫌疑待真帧确认：(a) 端末某**稳定屏无 確認**却中左有黄/中右有蓝被误判成对话框；
  (b) はい 点击点是 dialog_yes 框内**全部黄像素质心**，被装饰性黄拉偏→点空→对话框永不消失。另注：
  `button_guard` 每个 kind **各自单独截一次图**（一轮 4 张），动画期四帧可能相互矛盾——单帧诊断可区分 (a)/(b)。
- **插桩（纯增量，不改点击判定）**：
  - `button_guard(kind, img=None)` / `main_menu_visible(img=None)` 新增可选预载 RGB 帧参，让四个 guard +
    lobby 检查在**同一帧**上评估。
  - `auto_next_flow(..., debug=None)`：每轮循环顶调一次 `debug()`（try 包裹，异常不打断流程）。
  - 新 `--auto-next-debug`：`autonext_debug_dump()` 抓一帧、四 guard+menu 全在该帧上算，存
    `<run>/_autonext_debug/an_g<game>_<NNN>_<branch>.png` + `autonext_debug.jsonl`（fracs/质心/预测分支/menu_diff），
    并打印一行 `[autonext-dbg]`。**诊断产物，非数据集输入**（在 `frames/`、`games.json` 之外）。
  - 顺带把 dialog 分支/retry 日志从只印 `dialog_yes` 扩成印全四 frac + 点击质心坐标。
- **数据影响**：无（不进 `frames.jsonl`/数据集）。**判定逻辑一字未改**，仅新增可选诊断路径。
- **验证**：`tests/test_autoplay_autonext.py` 11 项全绿（新增 2 项锁 debug 钩子：每轮触发且不改点击序、
  抛异常不打断流程）；`py_compile` 通过。PIPELINE.md 采集节同步 `--auto-next-debug`。**根因已由此插桩
  在 `ai_session4/run_2` 复现定位、并已修复——见 §1.44。**

### 1.44 auto-next 卡死根因 = はい guard 被终局立绘误触；修复 = 对称双键判据 `is_rematch_dialog`（2026-07-07）
- **数据**：`--auto-next-debug`（§1.43）三轮 `ai_session4/run_2`(79)+`run_3`(31)+`run_5`(29) 共 **139 帧**，
  每轮四 guard 的 frac 全在同一帧上评估。
- **根因（实证）**：`dialog_yes`（はい，黄，中左框 5.0-8.5×5.0-7.1）**与终局排名屏的胜者立绘重叠**，而
  **確認 按钮在排名动画后 ~15-20s 才淡入**（confirm 从 i≈2 一直缺席到 i≈10）。这段窗口内蓝色排名条把
  `dialog_no` 顶到 0.65-0.93、confirm 缺席；正常时 `dialog_yes<0.03` 走 `wait` 等確認，但**当胜者立绘偏金**
  时中左框暖色把 `dialog_yes` 顶过阈 → `yes&no&!conf` 成立 → 排名屏被误判成 rematch 对话框，はい 被点进
  立绘空点，无限循环到超时。**间歇性正是因为取决于是哪张立绘**（run_2 game6、run_3 game3 都因此放弃）。
- **为何单一阈值不行（run_5 反证）**：先试把 `dialog_yes` 阈 0.03→0.13。run_5 直接推翻：game1 金皮立绘
  触 **0.1284**（仅差 0.13 一线侥幸没炸），game4 立绘更金触 **0.3882——比真 はい 的 0.2068 还高**（只因当时
  蓝条尚未渲染 dialog_no≈0 才没误触）。**立绘的黄能盖过真按钮的黄，任何单一黄阈值都分不开**。
- **真正不变量 = 对称双键**：真 rematch 对话框是**并排两个等大按钮**，はい/いいえ 两 frac **可比**（比值
  钉死 0.2068/0.2264 = 0.91，跨 3 轮 10 个对话框恒定）；排名屏则**永不对称**——要么蓝条压倒（no≫yes，如
  0.13/0.87）、要么蓝条未渲染前立绘压倒（yes≫no，如 0.39/0.0002）。
- **修复（`scripts/capture/autoplay_ai.py`）**：新模块级 `is_rematch_dialog(yes_frac, no_frac, confirm_present)`
  ——`confirm 缺席` 且 `两 frac 均 ≥ DIALOG_FLOOR(0.12)` 且 `min(yes,no) ≥ DIALOG_BALANCE(0.6)·max`。
  `auto_next_flow` 的 dialog 分支由 `yes_ok&no_ok&!conf` 改用它；点击后 recheck 改判 `still_frac<0.12`；
  `dialog_yes/dialog_no` 的 `min_frac` 回落 0.03（退化为"这里有没有色"的粗传感器，不再承担判定）。debug dump
  的 branch 预测同步走 `is_rematch_dialog`（诊断与线上一致）。guard 配置 + 判据均在模块级，回归测试直接引用。
- **全量回放验证（139 帧，跑真实模块 `is_rematch_dialog`）**：**10/10 真对话框命中、0 误触、0 漏判**；所有
  原误触/卡死/retry 帧（含 run_5 的 0.1284、0.3882 两个能击穿任何阈值的金皮帧）全转 `wait`。
  `tests/test_autoplay_autonext.py` **14 项全绿**（新增 `is_rematch_dialog` 对录得真值断言 + FracUI 跑全部误触帧
  含 run_5 金皮 + 真对话框点一次即成功）。
- **数据影响**：无（仅端末续局判定，不入数据集）。**待用户实弹复跑确认线上不再卡**（离线 139 帧 100% 通过）。

### 1.45 对手牌背标注（实验，`--backs` 默认关）——手摸切识别前置（2026-07-07）
- **动机（调研结论）**：手切时对手暗牌行短暂出现空隙；正式帧按设计（quiet 0.3s + ROI 稳定）必然错过，
  全语料 extras（ai_session3+4 共 3466 张）最早 dt=0.85s，实看 0.86s/1.98s 手切帧空隙均已闭合 →
  **空隙窗口 <0.85s，现有帧拍不到**。但空隙期间**其余牌背不动**（空隙行=闭合行的位置子集）→ 在闭合行上
  训 per-back 检测器即可泛化：运行时"行内某槽缺检测"=手切信号，模型无需见过空隙。GT 侧
  `RiverTile.tsumogiri` 早已逐张记录，标签免费。
- **标定（`majsoul_eye/annotate/backs.py`，run_8 games1-4 真实帧，V>170 fullwarp 条带 run 分析，各 extent mad=0）**：
  行在 fullwarp 内均匀 pitch，**锚在玩家左手端**（pos1 y1587 / pos2 x2123 / pos3 y277 固定），随副露向玩家
  右手收缩，侧座带小 per-meld bias（pos1≈17、pos3≈25.5，平面外 sprite 底缘伪影；pos3 三数据点 25.5/25.5/25.1
  一致，pos1 rn7 预测 1009=实测 1009）。摸牌槽=移动端外 ~25px gap（toimen 164/165 帧）。盒=fullwarp 槽矩形四角
  过 `H_full_inv` 回原像素（sprite 涂抹自动包含，站立牌无需 3D 推理）。**跨皮肤（红/蓝白 deco）+ 跨分辨率
  （1920/1600→resize）overlay 验证对齐**。
- **holding 座位（count%3==2）整座跳过 + `pos{p}:backs_holding` flag**：实测发现 tsumo 思考窗的 canonical 帧
  能拍到**理牌中段**（run_8/game1 seq347：右座 10+空隙+4 布局 + 皮肤手臂立绘）——插槽位置取决于对手暗牌，
  GT 不可知，错框比无框糟。build 侧对带该 flag 的帧**整帧丢弃**（否则未标的已渲染行会教检测器抑制 back——
  hero-hand 教训），实测丢 ~40-56% 帧。
- **接线（全部默认关）**：`build_datasets.py --backs` → `annotate_ai_session.py --backs`（记录多
  `back_boxes`，overlay/统计已接）→ `build_dataset.py`（`iter_tile_boxes` 出 zone `oppback`、YOLO `back` 类、
  不出分类器 crop、holding 帧丢弃 + 计数）。`tests/test_backs.py` 5 项 + 全套 44 测试绿。
- **样例**：`datasets/backs_sample/`（ai_session3/run_2/game5 单局、非标定来源：换肤+900p；154 帧标注 →
  4666 back 框全过 fill → build 后 68 帧/6061 检测（back 2816），HBB `detector/` + OBB `detector_obb/`
  双 split；`fiftyone_view.py` 增加 9 字段 OBB 渲染（Polylines）——侧座看 OBB（HBB 坍缩重叠大）。
  同日顺手根因修复 FiftyOne "GUI 空/len=0"：内嵌 mongod 被硬杀 → 新集合 WiredTiger 元数据计数留 0
  （文档完好，find/$count 可见），`len()`/App 网格用快速 count 命令 → 显示空；每次会话退出都会复发。
  `fiftyone_view.py` 加 `_repair_zero_count()`（加载时 len=0 但可迭代 → mongo `validate` 重算计数）
  + `session.wait(-1)`（原 `wait()` 标签一断连即返回，看起来像"程序自己退出"）。
- **人工 per-slot 模板取代均匀网格（同日，用户逐张点角）**：用户反馈均匀 fullwarp 格 HBB 太粗 → OBB 渲染
  （`fiftyone_view.py` 增 9 字段 Polylines）仍嫌侧座 quad 跟"行方向"不跟"牌自身倾斜" → 上人工标定：
  `calibrate_backs_manual.py` 生成自包含 HTML 标注页（headless OpenCV 无 GUI，浏览器画布：滚轮缩放/
  右拖平移/方向键微调/localStorage 断点续标），用户在 run_8/game1 4 帧上点了 42 quad（3×13 行位 +
  3 摸牌位）。`--ingest` 合并→规范化角序→fullwarp→生成 `annotate/_backs_manual.py`。**自洽性极强**：
  三家模板 centroid pitch 71.06/71.34/71.20（物理同一牌宽!），摸牌 gap 三家 +35.7/+36.9/+35.4，
  行范围与自动 strip-run 值重合（自动 pitch 76.9/78.2 的偏差=涂抹外延口径）。`generate_back_boxes`
  改为模板查表 + 绕锚点 meld 伸缩；`drawn_quad()`（随 moving end 平移）备用。样例重建（HBB+OBB），
  异皮肤帧验证逐张贴合。tests/test_backs.py 6 项改为模板断言，全套测试绿。
- **理牌收缩门（同日，用户在 FiftyOne 抓到错位帧）**：对手手切后客户端理牌收缩 ~0.5-1s，GT 已 settled 而像素
  未定——根因＝**`roi_diff.TABLE_ROI (0.18,0.16,0.82,0.92)` 不含三家手牌行**（当年为 river/meld 设计），稳定
  确认放行了动画中段（实测 ~3.5-10% 帧）。修：`backs.sorting_suspect` 像素门——逐槽 (gray mean,std) 与
  "行内牌中位数 vs 摸牌槽外 1.15 格空毡参考"相对比较（跨皮肤；绝对 live-mask fill 在饱和桌布上失明），
  任一行槽读作空毡（相位 A 缺口未合/相位 B 尾槽空）或 13-行摸牌槽读作牌（相位 A 早期）→ `backs_sorting`
  → build 整帧丢（与 holding 同策）。注意分离摸牌牌**跟视觉行尾滑动**（行缩至 12 时它也向锚点滑一格），
  只查固定摸牌槽会漏相位 B（实帧验证：run_2/game5 seq76 初版漏、重写后抓住；样例 68→61 帧）。
  顺带修了 `_drawn_fw` extra 位移的符号 bug。**采集侧根修（同日用户拍板执行）**：`roi_diff` 扩成
  **多矩形 STABILITY_ROIS**（原 TABLE_ROI + 三家手牌行紧包络[取自人工模板±margin、含摸牌槽] + hero 行；
  取各矩形 diff 的 MAX，量化为空的矩形跳过），dora 面板（杠宝牌闪光）/头像（右座例外——行压在头像前，
  静态肖像可接受）/HUD 角落仍排除。**离线验证**（ai_session3/4 跨皮肤 120 对同 seq extras）：各矩形静息
  med≤0.53 / p90≤1.32，远低于 diff_thresh 3.0，无慢性动画卡死区 → 阈值不变。sync/autoplay 走默认参数
  自动升级；旧单矩形调用签名兼容（test_roi_diff 断言旧 ROI 对 toimen 行确实失明）。此后新采集在手牌行
  未定时不再放行截图；存量帧仍靠 sorting 门。
- **后续（未做）**：① 空隙帧探针（`--multishot-offsets 0 0.2 0.4` + `multishot_window` 对手-dahai 开窗几行改动）
  捞验证集测"空槽幻觉率"；② dahai 窗 extras 天然 GT-leads-pixels（河多一张/行有空隙），接入训练时只能作
  backs 专用帧，**严禁当普通全盘帧标 river/meld**；③ holding 布局标注 → **已做，见 §1.47**；
  ④ stability ROI 多矩形扩展（见上，已做）。
### 1.46 build 提速：annotate HSV 一次转换 + build_dataset 免重编码；INTER_LINEAR 实测否决（2026-07-07）
- **瓶颈画像**（Linux 32 核盒，逐帧单线程实测）：stage-1 annotate 是绝对大头——27,250 源帧 ×
  (`imread` 60ms + `annotate_frame` ~320ms)，且不传 `-j` 时 annotate 只用自身保守默认 **4 workers**
  （为老 Windows 机设的上限）≈ 45 min 墙钟。`annotate_frame` 内部：fullwarp `warpPerspective`
  3072×1941 INTER_CUBIC = **191ms（59%）**、三个 mask 函数各自重复整幅 BGR→HSV = 66ms、
  `cv2.integral`×2 = 22ms。stage-2 每帧 = 解码 60ms + **全帧 PNG 重编码 98ms** + 小 crop 写。
  stage-3 只写 txt，可忽略。**大盒子跑 `build_datasets.py` 记得传 `-j 12`**（仅此一项 stage-1 ~3×）。
- **改动 1（无损）**：`annotate_frame` 把 fullwarp 的 BGR→HSV 算一次传给三个 mask（
  `tile_face/back/live_mask` 新增关键字参数 `hsv=`，原 BGR 位置参数兼容保留——标定脚本/测试不动）。
  同帧集 AB：`annotate_frame` 330→**294ms**。**验证**：普通局 `ai_run_3_game1`（719 帧）+ 换肤局
  `ai_session2_run_1_game1`（257 帧）标注记录与改前**逐字节一致**。
- **改动 2（无损）**：`build_dataset.py` 写 `yolo/images` 时，帧未经 resize（`--from-annotations`
  路径恒真；自足模式 1080p 原生也真）且源 PNG 为 8-bit RGB（PIL header 校验，防未来 alpha/16-bit
  截图悄悄换语义）→ 直接 `shutil.copyfile` 源帧，跳过 ~100ms/帧 的解码后 PNG 重编码；否则回退
  `cv2.imwrite`。**验证**：同一份标注 AB 重建换肤局——labels/crops/hud **逐字节一致**、
  257/257 张 yolo 图**逐像素一致**（字节不同：copy vs 重编码），单局 build **51.7s→29.4s**。
  两模式（自足 vs `--from-annotations`）产出逐字节相同的 §1.14 结论不受影响（copy 对两者同样生效）。
- **尝试并否决：fullwarp INTER_CUBIC→INTER_LINEAR**（本可再省 ~140ms/帧）。AB（976 帧）：river 侧
  完全干净（27.5k slots **0 个 reliable/unrendered 翻转**，fill Δ≤0.028）；但**远座 meld snap 失锁**——
  run_3 pos3 seq80–118 一段 CUBIC 锁 along=+46.0px、LINEAR 锁 +15.5px（Δ30.5px），QA meld 一致率
  **1.0→0.6348**（crop 错位读成 2m→7m 等）。根因：`snap_meld_strip` fine-stage 的
  `MIN_CREVICE_CONTRAST`/`MIN_EDGE_GRAD` 按 CUBIC 锐边标定，远座小瓦经 LINEAR 软化后对比度跌破
  阈值、候选锁评分翻转。已回退 CUBIC 并在 `warp_to_full` 注释里钉死此结论（重试需先重标定 snap）。
- **数据影响**：无——产出与改前等价（标注逐字节一致、图逐像素一致），既有 `datasets/*` 不需重建。
  测试 43/43 全绿。

### 1.47 HUD 标签三修：wall_count 固定框+补零、reach fill 门限缩至声明窗口、按钮框扩成 banner（2026-07-07）
- **起因**：用户实测截图（56 类检测器推理）——对家**剑形皮肤立直棒漏检**、wall_count 框只落在
  「余」字上。溯源均为**标注管线的系统性缺陷被检测器学走**，非推理端偶发。
- **根因 1（wall_count 全量标签截断）**：种子框 `(910,427,952,455)` 仅 42px 宽，当年 x1 收到 952 是为避开
  「面板 bezel 高光（~x956）」，但数字恰从 ~x955 起——`ink_snap` 只能在种子内收缩，于是 v3 **全部 23,263 个
  wall_count 标签宽 ≤42px（中位 29px = 只有余字）**，检测器学到的就是这个框。连带：读数 GT `余9` vs 实际渲染
  `余09`（客户端**补零两位**）——DigitCTC 若训练即错标。
- **修复 1**：跨 3 个 session（含云桌皮肤）重测——字形墨迹 **x923-997 / y432-448 恒定**（补零 ⇒ 恒宽恒位，
  单/双位数帧墨迹一致）。种子改固定框 `(918,428,1002,452)`，`FIXED_BOX_NUMERIC` 免 snap、只用
  `WALL_COUNT_INK_PROBE`（余字子区，天然避开 bezel/分数辉光）做"是否渲染"探测；`field_texts` 补零 `:02d`。
- **根因 2（暗皮肤立直棒被当背景训练）**：`REACH_FILL_OK=0.35` 亮度门**无条件**应用，把"未渲染"与"已渲染但
  暗色皮肤"混为一谈（用户截图剑皮肤实测 fill=0.264）。v3 净影响（扣除帧级 score-anim 门的 ~65/槽后）：
  **across 212/1269（16.7%）、left 143/1040（13.8%）** 纯 fill 门误杀，self/right 为 0——与标定注释的
  "注射器皮肤偏暗"完全吻合。且 `hud_emit` 只丢框不丢帧 ⇒ 这些棒的像素以**背景**身份进训练。
- **修复 2**（用户确认前提：棒只可能在声明那条记录的帧被动画/特效遮挡，settled 后一直可见到局末）：
  fill 门限缩到 `is_score_anim_window(state)` 为真时才生效（Task 18 后该谓词已 bundling-proof；零事件
  stale-fallback 只会保守多拦）。off-window 一律信 GT。QA：暗皮肤 settled 帧（fill 0.349）转绿、
  声明瞬间 hand-slam 帧（fill 0.03）保持红。
- **根因 3（按钮框=文字框，且 7.2% 是黏连坏框）**：亮度阈值只能抓 banner 内书法字（banner 底 gray 40-95
  贴近桌面），故旧标签是**文字字形框**——随显示语言变宽，且 154/2135（7.2%）是并排 banner/特效黏连的
  >300px 大块。
- **修复 3**：颜色距离分割在 189 帧干净样本 × 7 类上标定——banner（=实际点击区）为**恒定 250×96 板**、
  中心在字形块中心下方 ~10px（|dcx|≤10）。`button_boxes` 现发 `banner_box()` 固定框（跨语言不变）；
  `locate_button_candidates` 增 `BTN_MAX_W/H=300/90` 上限，黏连块被拒 ⇒ count_mismatch 整帧不出按钮标签
  （宁缺毋滥）。残留：换肤 UI（米黄主题）的字形定位本身仍会错位/黏连——被新上限过滤成无标签帧，
  locator 对皮肤主题的鲁棒性另案。
- **验证**：TDD（`test_hud_fields/test_reach_stick/test_hud_buttons` 先红后绿），全套 43 测试绿；
  `overlay_hud.py` 三类真实帧目检通过。**数据影响**：HUD 类标签全体变化 ⇒ v3 过时，已启动 **v4**
  全量重建（71 局，`build_datasets.py v4 --sources ai_session{,2,3} --hbb --obb -j 12`）；
  检测器/读取器重训待后续（本次仅数据部分）。

### 1.48 backs holding 帧改为标注（原判"GT 不可推"是错的）+ v4 空局报错随之解除（2026-07-08）
- **触发**：v4 build stage-3 `REFUSING to assemble detector — no yolo/images`，4 局（ai_session2_run_4/5/6、
  ai_session4_run_2 各一局）产出 0 帧，`--resume` 重跑必复现。逐帧查：每帧都被 `--backs` 的 holding/sorting
  整帧丢弃门清空。
- **用户纠错（关键）**：§1.45 把 holding（`concealed_count%3==2`，刚摸未打）判为"插牌缝位置取决暗牌、GT 不可推"
  是**把两个时刻混了**。时刻 A holding：牌行是**静止的 n-1 张**（摸到的牌单独摆在固定摸牌位，不插进手里），
  行＝原模板、摸牌位＝`BACK_DRAWN_QUADS`（短行随 moving-end 外推），**全部 GT 可推**。真正"缝位置取决暗牌"的是
  时刻 B——手切**之后**的理牌重排，那时 count 已回 settled，落在 `sorting_suspect` 门，与 holding 无关。docstring
  把 B 的理由错安到了 A。
- **实证**：真实 holding 帧（0 副露 row_n=13、1 副露 row_n=10）overlay——`generate_back_boxes(pos,n-1,nm)` 的
  行框 + `_drawn_fw` 外推的摸牌框**逐张贴合**（含短行外推,红框正落在分离摸牌牌上）。
- **改动**：`backs.back_boxes` holding 分支不再整帧跳过，改 emit `generate_back_boxes(pos,n-1,nm)` + `_drawn_box`
  （slot=n-1，`drawn=True`）；`sorting_suspect` 仅对 settled 行跑（holding 行不重排，摸牌滑入只影响那一张,靠
  fill 逐框判）；新增 `_drawn_box` 辅助。`build_dataset` 丢弃门去掉 holding 语义（仅 `backs_sorting` 丢帧；
  `backs_holding` 仅为旧 JSON 向后兼容保留匹配）。tests 改断言 holding→14 框（13 行+1 摸牌），全套绿。
- **效果**：样例局 `--backs` 丢帧 **93→7**（holding 全回收，86 摸牌框新出）；4 个死局 27/50/… 帧产出（原 0）→
  v4 `REFUSING` 自然解除。样例 61→**147 帧**，HBB+OBB 双 split 已重建，FiftyOne 已清待重导。
- **仍开（本次未动）**：`backs_sorting` 在换皮局（ai_session2）**过度触发**（run_6_game1 达 85%，远超均值 27%，
  pos3 偏斜）——空毡参照被皮肤桌布击穿，是回收更多帧的下一个着力点，与 holding 无关。编排器"0 帧=坏局、建议
  永远无效的 --resume"的误分类经此修复对这 4 局已 moot，但**软跳过 vs 硬失败**仍是可选的健壮性改进。

### 1.49 副露行错位根因＝虚构的 meld 伸缩（`meld_bias` 归零）（2026-07-08）
- **用户报**：FiftyOne 里"我标的很准，第三家(pos3)静态牌背仍偏移"。逐层排查：① 分辨率——样例/标定源同为
  1920×1080 16:9（FiftyOne 截图 3112×1685 只是浏览器缩放，无畸变）；② 跨会话——pos3 cross-bias 在 run_8
  (−22.3) 与 ai_session3 (−22.9) **几乎相同**，模板未漂移；③ 定位到用户看的正是 seq384 = pos3 **1 副露 10 张行**。
- **根因**：`_meld_k` 的副露"伸缩"是**虚构的**。物理上暗牌从锚点按固定 pitch 顺排，副露只是去掉远端槽、其余
  **不重排**——副露行＝前 row_n 个模板槽原位。旧 `meld_bias`(17/0/25.5) 来自自动 strip-run 的**移动端 sprite
  外延**拟合，把短行拉长（pos3 +3.6%），误差向移动端累积 → 末槽甩出行尾落到桌布上。三家 1 副露帧目验：k=1
  绿框逐张贴合、当前伸缩青框末端明显甩出（pos2 本就 bias=0 无恙）。
- **改**：`BACK_ROWS[*].meld_bias` 全置 0（`_meld_k` 恒 1.0，留作未来真·重排的 hook）；模块 doc/注释更正
  （副露不重排）；`drawn_quad`/`_drawn_fw` 随之正确（k=1 时摸牌槽仍随 moving end 平移）。tests 改为断言
  "副露行＝前 row_n 模板槽逐一相等"，全套绿。样例重建 146 帧（HBB+OBB），FiftyOne 已清待重导。
- **注**：cross 方向仍有约 −22px 的系统量，但那是**度量方法**把白色牌前脸计入亮掩膜所致，run_8/ai_session3 同值、
  模板在标定源上目验完美，非真实偏移；真正的可见偏移只在副露行、由本 bug 造成，现已消除。

### 1.50 副露角点重标定：远座 snap 失锁根因＝半张牌系统偏移（2026-07-08）
- **用户报**：`backs_review` FiftyOne 里**上家(pos3)副露区严重失位**、**对家(pos2)相邻两帧副露标注差别很大**；
  "副露标注不是比较确定吗？为什么还这么多问题？"
- **根因（两层）**：副露框＝GT 生成模板（`generate_meld_boxes_v2`，确定）+ 逐帧图像吸附（`snap_meld_strip`，脆，
  全管线最不稳一步）。① **pos3 角点系统性偏 ~+46px（半张牌深）ALONG**，恰把吸附顶在等间距同款牌的 **aliasing
  中点** → 邻位整张牌错锁 **26%**（run_8_game6 seq1714/1715：+47.5↔−48.4，95.9px 整张翻）。② 良性残留：各座
  逐帧吸附本有 ~2-5% 抖动（远座最甚）。角点自 2026-07-02 标定后经数次 warp/mask 改动未复标；
  `calibrate_annotation_model.py` 用同一吸附取 median，当前 pos3 median 即 +46 → 重跑即给出该修正。
- **实证**：预移 +46 后 pos3 mislock 25.9%→0.9%（>20px 判据，1957 帧）。新守卫 `scripts/annotate/meld_snap_qa.py`
  逐座报 dominant offset + 锁错率（1−frac@±12px）：改前 pos3 da_off 46.0 / mislock 0.231 → FAIL。
- **修复**：`MELD_STRIP2[3].corner` 沿 along +45.5（(625.0,1797.6)→(624.5,1752.1)，`calibrate_annotation_model.py`
  refit 建议，交叉核对）。改后 QA：pos3 da_off 1.5 / mislock 0.051（整张翻 <1%，残留 ~5% 是远座逐帧噪声）；
  worst 0.051 < 0.08（守卫阈＝**回归门**非精度门：逐帧地板 ~2-5%，失锁态 20%+）→ OK；suite 45/45；目验
  seq1714/1715 贴面。
- **pos0 不动**：pos0 有同样 +46 **cross** 偏移，但吸附每帧稳锁纠正（0% mislock）——重标定它只会把良性恒偏换成
  逐帧散布（0→4.4%），故保留原角点。
- **pos2 未修（本阶段）**：pos2 角点本身准（da_off~1、dc_off~0），相邻帧闪烁是**按局吸附**问题（真实按局浮动 +
  偶发翻锁），非标定偏移——留给 **Phase 2 按局共识**（另案 spec/plan：`docs/superpowers/specs|plans/2026-07-08-meld-snap-*`）。
- **数据影响**：所有 pos3 副露框位移 ⇒ `datasets/*`、`datasets/backs_review` 过时，待 `build_datasets.py <v> --force`
  重建 + 重训（用户 gated，见 plan Task 4）。**守卫**：`meld_snap_qa.py` 在 warp/mask/角点改动后必跑（<8% 锁错）。

### 1.51 副露按局吸附共识（Phase 2）（2026-07-08）
- **用户报**：`backs_review` FiftyOne 里自家/对家副露框**偶发整体偏移/偏下**——§1.50 修的是 pos3 的系统性角点
  偏移，但逐帧吸附（`snap_meld_strip`）本身仍会**偶发失锁**，且低特征帧会整帧回退到未吸附的原始模板，两者都
  导致同一局内相邻帧副露框忽然跳位，观感上"标注不稳"。
- **设计**：副露条在一局内物理固定，正确的刚体吸附偏移 `(d_along, d_cross)` 按 `(bakaze, kyoku, honba,
  screen-pos)` 应该是**常量**。新模块 `majsoul_eye/annotate/meldsnap.py`：`measure_meld_snaps` 对每帧跑一次
  warp+meld mask+`snap_meld_strip`，`round_meld_consensus` 按 `CLUSTER_TOL=12px` 对两轴一起聚类，取**置信度
  （score）加权的主簇质心**作为该局该座的唯一偏移；`game_meld_overrides(seq_states, seq_frames, hom)` 汇总一局
  内所有 seq，把该局每座的共识偏移套回**该局每一帧**（不再逐帧各测各的）。`annotate_frame` 新增
  `meld_snap_override=` 参数消费这个共识值，取代逐帧现测。因此 build 变成**两遍扫描**：第一遍按局跑
  `game_meld_overrides` 测量+聚类，第二遍才真正 `annotate_frame` 落框。
- **低置信处理**：一局的置信样本数 `< MIN_ROUND_FRAMES(3)` 或主簇得分占比 `< MIN_ROUND_CONF(0.55)` 时，
  `round_meld_consensus` 返回 `None`——该局该座不给出共识偏移，`annotate_frame` 相应把那些帧的 meld 框标
  `reliable=False` + `meld:low_round_conf`，而不是硬套一个不可信的值。
- **只在 Phase 1 之后安全**：本机制假设逐帧吸附的多数样本是对的（少数噪声/失锁被共识投票压制）；如果角点本身
  系统性偏移（§1.50 的 pos3 病灶），大量一致的错误吸附会把共识**也**投票到错误位置——所以 Phase 2 必须建立在
  Phase 1（角点重标定）完成之后，顺序不可颠倒。
- **守卫**：新增 `scripts/annotate/meld_consensus_qa.py`——对每局按 `(bakaze,kyoku,honba,pos)` 断言
  `game_meld_overrides` 吐出的偏移在该局内**唯一**（同局同座只能有一个值或 None），非唯一即回归，非零退出。
  跑 `captures/raw/ai_session` 全量：`rounds checked=149 nonuniform=0` → `OK`。
- **代价**：每帧多一次 measure-warp（img warp + mask + `snap_meld_strip`），构建时间大致翻倍（两遍扫描 vs 原来
  一遍）；只发生在 build 阶段，不影响运行时识别器。
- **数据影响**：所有含副露的帧的 meld 框位置改变（同局帧不再各自独立漂移）⇒ 现有 `datasets/*` 过时，需要
  `build_datasets.py <v> --force` 重建 + 重训（本任务序列的 Task 5，用户 gated）。
- **注**：Phase 1（`meld_snap_qa.py`）关于"fill 盲区"的结论仍然保留——cross 方向那 ~22px 的系统量是度量方法
  把白色牌前脸计入亮掩膜所致、并非真实偏移，真信号是牌缝（crevice）；Phase 2 的按局共识不改变这个结论，只是
  在"局内取哪个吸附样本当真值"这一层做投票，不重新定义吸附本身的物理量。
### 1.52 局面复原 M1：observe/assemble/reconstruct/eval 四模块 + 三层评测落地（2026-07-05，merge 回 main 2026-07-09；原分支编号 §1.30，因与主线冲突改号）
- **目标**：单帧识别结果 → `ObservedState`（可见状态，第 1 步）→ 从 `start_kyoku` 到当前状态合法
  的 hero 视角 mjai 序列（第 2 步），配三层 GT 评测（oracle/assemble/engine）。Spec/Plan：
  `docs/superpowers/specs/2026-07-05-board-reconstruction-design.md` /
  `docs/superpowers/plans/2026-07-05-board-reconstruction.md`。
- **新模块**：`state/observe.py`（`ObservedState`/`ObservedRiverTile`/`ObservedMeld` 数据模型 +
  `check_observed` 校验 + `observed_from_board` 从 GT `BoardState` 投影，用于评测基准）、
  `state/reconstruct.py`（回合模拟 + 回溯 DFS → mjai 事件序列，`reconstruct(obs) -> ReconstructionResult`）、
  `recognize/assemble.py`（检测框反用 `annotate/pipeline` 标定几何装配 `ObservedState`，Akagi-free）、
  `scripts/eval/eval_reconstruction.py`（QA 工具，非管线环节；PIPELINE.md §4 已登记）。
- **`is_call_pending`（真实采集边角案例）**：chi/pon/(dai)minkan/ankan/kakan 后到该家强制立即打牌之间
  有一个空隙——`quiet` 去抖在强制 `dahai` 事件到达前先触发，此刻副露已更新但打牌尚无落点，单帧不可复原。
  新增 `BoardState.awaiting_discard`（哪家欠一次强制打牌，-1 = 无）由 `Replayer` 在四种鸣牌事件后置位、
  首个 `dahai` 清零；`is_call_pending(s)` 据此判定。`eval_reconstruction.run_oracle` 在 `reconstruct` 之前
  检查该谓词，命中计入 `skipped_call_pending`（此前误把它当失败注解在 `fail[]` 里，只标注不跳过）。
  `tests/test_replay.py` 补 3 例（pon 后 pending / 打牌后清零 / 非鸣牌普通回合不 pending）。
- **oracle 全量验收**（`captures/raw/ai_session` 全部 18 局，10330 个非-deal-window 回合帧）：
  **10063/10063 尝试重建全部成功（0 fail、0 mismatch）**；`skipped_violations`=58（`check_observed`
  发现的 GT 自身不变式违规，跳过不计入分母）；`skipped_call_pending`=209（含此前 11 帧曾报失败的
  call-pending 帧，另 198 帧此前虽能强行复原但落在同一空隙——现统一在 `reconstruct` 之前拦截，
  不再依赖"大多数 call-pending 帧凑巧能复原"的运气）。三者相加=10330，与转换前总数一致，纯粹是
  分母口径变化，非回归。
- **assemble 冒烟**（`ai_run_8_game1`，OBB 检测器权重，cpu）：~949 有效帧中 390 帧被装配 violation
  拒收（宁缺毋滥门，非误判——不确定装配结果不参与统计），其余 559 帧中 548 帧整帧全等（98.0%）；
  剩余 zone 级错误 `drawn` 8 / `rivers` 2 / `reach` 1 / `melds` 1（检测器噪声，非复原算法问题）。
- **engine 层未跑**：无可用 `--engine-cmd`（真实 mjai bot 子进程），本次跳过；harness 已实现、
  待接入 bot 时验证。
- **数据管线无变更**（不影响 `captures/`/`datasets/`/权重）。
- **Merge 后补（2026-07-09）**：56 类权重兼容——`assemble()` 原假定每个检测都有 `det.tile`（38 类
  时代），56 类头的 HUD 检测（`tile=None`：分数/余牌/立直棒/按钮）被当牌路由，远框产生
  `stray detection None` violation、近框（立直棒）污染副露条解析，导致**整局 949 帧全拒收**。
  修复：tile 装配循环开头跳过 `tile is None` 的检测（HUD 归 `assemble_hud` 域）；回归测试
  `test_hud_class_detections_ignored`。修复后新权重（`tile_detector_obb_20260707_122157.pt`）
  assemble 冒烟：534/550 整帧全等（97.1%）、399 拒收，zone 错误 drawn 8 / rivers 7 / reach 7 / melds 1。
  逐帧诊断：rivers 7 + reach 7 全部 = 同一物理事件（某局 seat1 立直宣言牌 5s 横放被检成竖放，跨 7 个
  连续帧重复计数）；drawn 8 全是装配侧手牌空档启发式误报（与检测器无关，旧权重下同类 10 个）。
- **Post-merge TODO 清账（2026-07-09）**：三项 final-review 遗留全部落地。
  ① **engine 层实测放电**：新增 `scripts/eval/mortal_stdin.py`（mjai stdin/stdout 包装 mycv Mortal
  v4 b24c512；PIPELINE §4 已登记）+ `ask_engine` 支持 `{seat}` 占位符（复原序列无 HUD 时 hero 恒在
  绝对座位 0，真实序列 hero 在 `start_game` id——单固定 player-id 命令喂不了两者）。run_8 全 6 局
  ×20 决策点：**116/120 决策一致（96.7%）、0 engine 错误、0 复原不可行**——"libriichi 能消费复原
  序列"从推断变为实测。4 个分歧的 `mask_bits` 逐对相同（合法动作集全等 ⇒ 局面语义全对），差异仅在
  Q 值——hero 历史舍牌被 fabricate 成全摸切造成的历史特征差，单帧原理性信息丢失，非缺陷。
  ② **拒收帧按类计数**：`run_assemble` 新增 `rejected_reasons`（violation 文本→类别，
  `reject_categories`，测试 `test_eval_reconstruction.py`）。新权重冒烟分布：**meld_parse 391/399
  绝对主导**（strip unparsable/ambiguous 的宁缺毋滥门），river_geometry 12、hand_size 33、dora 1、
  tile_gt4 1——阈值标定的主攻目标即副露条解析。
  ③ **小项**：`obs_key` 副露键收紧（补 `called_pai`/`added_pai`，此前两者不同也判相等）——收紧后
  oracle 全量复跑仍 10063/10063、0 mismatch（复原连叫牌/加杠牌都完全正确）；HBB 回退路径补测试
  `test_river_hbb_fallback`（4 座位，poly=None 走 xyxy 角点，序号+横牌均恢复）。
- **backs 权重兼容（2026-07-09，权重 `tile_detector_obb_20260709_055509.pt`）**：backs 数据训出的
  检测器会真实检出对手门清立牌背（~37/帧，置信 0.97+），每家 10+ 个落进副露条 60px 窗口 →
  三条 strip 全 unparsable → **又一次 949/949 全拒收**。矩形过滤不可行（对手副露条与其门清行
  矩形重叠 11–12/16 格）；采用**拖影轴判别**：立牌的透视拖影（屏幕竖直向）映到 fullwarp 后落在
  侧家条的 ALONG 轴（seat1/3）/对家条的 CROSS 轴（seat2），实测躺平（暗杠背）≤96 vs 立牌 ≥120
  fullwarp 单位（d≈92）→ 阈值 1.15×d；hero（seat0）不可能有立牌背，豁免。测试
  `test_standing_concealed_backs_filtered_from_meld_strips`（暗杠保留侧由 `test_meld_parse_kans`
  覆盖）。修复后 07-09 权重 assemble 冒烟（run_8 game1）：**541/559 整帧全等（96.8%）、拒收 390
  （meld_parse 381 仍主导）**——接受帧数三组权重最高（07-06 基线 549、07-07 550、07-09 559）；
  zone 错误与旧权重同构（drawn 10 / rivers 7 / reach 7 / melds 1，rivers+reach 仍是那根立直横牌）。
  该权重 HUD 检出很干净（中央面板 7 字段+立直棒+供托各恰 1 框/帧），HUD 读值待 HudReader 训练
  （§1.31/§1.41 线）。
- **CLI 入口（2026-07-09）**：`scripts/recognize/recognize_frame.py`——截图 → 检测→装配→复原 →
  JSON lines（ObservedState + mjai + fabricated；拒收帧报 violations），`--weights` 默认取最新
  `tile_detector_obb_*.pt`，`--letterbox`/`--no-reconstruct`/`--pretty`。运行时库链路的外部调用
  入口（PIPELINE §4 / scripts/README 已登记；顺带补登 eval/ 区）。
- **横牌判别改边缘方向 ⇒ meld_parse 拒收坍缩 381→3（2026-07-09）**：用户手机截图（2302×1288，
  非严格 16:9）暴露根因——正放 meld cell 的 warp 后 OBB 在条坐标系里近乎正方（ext_along 99–101 vs
  ext_cross 92），旧的 `ext_along > ext_cross` 横牌判别贴边翻车（6 格里 4 个正牌被判横 → strip
  unparsable）；域内它同样是 meld_parse 拒收 381/949 的主因。改用 **OBB 自身长边方向**（长边贴条轴
  = 横牌；正牌长边 ~93 贴 cross 轴 vs 横牌 ~103 贴 along 轴，干净分离；HBB 退化为原行为），河与
  副露条两处同换（`_long_edge_sideways`）。run_8 game1 基准：**接受 559→932/949（98.2%）、整帧
  全等 541→908（占接受 97.4%）、拒收 390→17（meld_parse 381→3）**；rivers/reach 错误 7/7→2/1
  （那根立直横牌同根因、同修复）。回归 fixture `test_meld_sideways_by_edge_orientation_not_extents`
  （真实截图量取的 6 格 warp 四角）。原"阈值标定主攻副露条解析"结论就此了结——瓶颈已不在此。
  **顺带发现（已处理，见下条）**：2.17:1 超宽手机截图上 3D 桌面居中 16:9 裁剪即可对齐（手牌/河全
  对），但宝牌指示区 2D HUD 锚定屏幕角、落在 16:9 区外。
- **超宽（>16:9）手机截图支持（2026-07-09）**：`normalize.locate_wide`（居中 16:9 板矩形，
  `BoardRegion` 增可选 `fw`/`fh` 全帧尺寸）+ `locate_auto`（按宽高比分派 fullscreen/wide/letterbox，
  CLI 默认改用）。宝牌指示区在手机上锚屏幕左上角且**内缩随设备而异**（iPhone 安全区 y 77–176 vs
  Android y 52–160，实测），固定框不可标定 → 改**检测侧 stray 救援**：zone 路由结束后，落在屏幕
  左上区域（cx<0.35fw, cy<0.25fh）、构成一条水平行的 stray 牌判为岭上宝牌行（左→右序；离行者仍报
  stray）——真正的桌面牌都会被 zone 认领，不会流入。16:9 域内 `ox=0` 救援不触发，基准复跑与修复前
  完全一致（932/949）。用户 9 张真实手机截图（2.17/2.20:1，iPhone+Android）：**6/9 全链路通过**
  （含杠宝双指示 5mr+1s 正确救援），3 例失败均为正当拒绝（鸣牌待打帧、疑似动画帧、重鸣牌 hero
  的"no legal turn order"复原边界——最后者留作 reconstruct 后续案例）。测试：`test_normalize.py`
  （新文件）+ `test_wide_frame_dora_rescued_from_screen_corner`。
- **hero 鸣牌待打帧支持（2026-07-09）**：鸣牌→强制舍牌空隙对 HERO 侧是全可见的（11 张手牌 + 副露
  的 called_pai/from_rel 都在屏幕上），合法序列**终止在 call 事件即可**——且恰是真实决策点（bot 对
  它的反应就是待打选择）。`check_observed` 放行「手牌+3×副露==14 且无摸牌槽且 hero 末副露为吃/碰」
  的形态（杠空隙是 13 账形态、与稳态不可分，照旧拦截）；`reconstruct._hero_call_pending` + 搜索器
  新终止分支（pending call 必须是最后事件，禁止提前触发——提前会伪造历史）。对手侧空隙照旧由
  `is_call_pending` 拦截（欠的舍牌不可见，原理性不可复原）。**oracle 全量复跑：10121/10121 全部
  成功、0 mismatch、0 violation-skip——之前 58 帧「GT 不变式违规」之谜就此解开：全部是 hero 鸣牌
  待打帧**（旧 13 张检查误标），现全数落入新路径复原成功；209 帧对手/杠空隙照旧跳过，总账 10330
  不变。顺带加固：`_hypotheses` 拒绝**非法加杠**（加的牌与碰不同种，如真实手机截图 IMG_1963 上
  误检的「2p 加 4p4p4p」——该帧由此从含糊的 hand-size 违规改为诚实的 strip unparsable 拒收）。
  测试：`test_hero_call_pending_*`×2 + `test_hero_call_pending_shape_allowed` +
  `test_meld_parse_rejects_illegal_kakan`。
- **跨类重复框去重（2026-07-09）——"no legal turn order" 案例排查结论**：ultralytics NMS 只在类内
  抑制，同一张实体牌可带上第二个异类低分框（实测：一张 4p 的同一个框又被检成 N@0.53）。幽灵 N 把
  上家河从 5 张吹成 6 张 → 回合守恒被破（seat3 需 7 次行动、上限 6；"余46" 摸牌全局账也差 1）→
  reconstruct 的拒绝**完全正确**（全局回合账当了最后防线，且把矛盾"诊断"了出来）。修复在源头：
  `detector.predict` 后加**类无关重叠去重** `_dedup_overlaps`（IoU≥0.8 只留最高分；相邻河位/加杠
  叠放远低于阈值不受影响）。**三个失败手机样本同根因全部复活**（IMG_1963 的"非法加杠 2p"、IMG_1964
  的"河 7>6"也都是幽灵框）：**用户 16 张真实截图 16/16 全链路通过**；域内基准 911/936 全等、拒收
  13（去重再救回 1 帧 river_geometry 拒收，零回归）。测试 `test_cross_class_duplicate_suppressed`。

### 1.53 HUD 读取器首训 + 端到端全绿（v4 manifest 补齐 / val 列表兼容 / CE 头增广）（2026-07-09）
- **v4 卡点（数据完整性）**：`datasets/v4`（07-07 扩到 100 局，含 ai_session4）一直没有
  `games.json`/detector split——stage-3 因 `ai_session_run_5_game2/3` 0 帧拒绝组装。根因：这两局
  帧在磁盘上是**原始 1923×1142 信箱版**（mtime 06-28，§1.34 的 `--inplace` 去黑边成果被某次归档
  还原/迁移盖了回去；annotate 侧有信箱兼容所以统计健康，`build_dataset` 对非 16:9 硬跳）。修复 =
  重跑幂等 `deletterbox_frames.py --inplace`（563+265 帧全回 1920×1080）→ 隔离信箱帧标注 →
  `build_datasets.py v4 --resume --hbb --obb --backs`（sources = ai_session{,2,3,4}）：重标注重建
  两局、双 split 组装成功（train 31436 / val 922 帧、56 类）、manifest 落盘（val=run_8_game1）。
- **`train_hudreader.py` 两修**：① manifest `val` 是列表（`write_manifest` 多 val 约定），脚本按
  标量消费直接 `--val not among games` 拒启——归一化为 `val_names` 列表、CLI `--val` 可重复、ckpt
  meta 同步（测试补 list/标量两态）。② round/wind CE 头**零增广过拟合固定取景**：训练裁片=seed ROI
  ink-snap（帧间近乎逐像素相同），crop 级 top1 1.0，但运行时检测框（更紧）上 round_label 26.7%、
  seat_wind 72.7%，预测坍缩到 E1/S3/W；CTC 头因 `_augment_ctc` 稳健=对照组。修复：改名共享
  `_augment_crop`（pad 内随机重裁 + 亮度抖动），CE 头 train 侧启用（jitter=0.15）。
- **训练**（RTX 5080，v4 全量，val=run_8_game1 整局 hold-out）：CTC exact 0.9692、round/wind
  top1 1.0000 → `majsoul_eye/recognize/hud_reader.pt`。CTC 的 195 个 val "错" **100% 是
  wall_count 像素=GT−1**（帧 quiet 捕获时下家摸牌动画已把屏上余牌数减 1，GT 快照停在事件时刻的
  `leftTileCount`——真实误读 0，运行时读像素反而是当下真值；训练数据同源噪声 ~22% 被多数信号
  压住）。
- **端到端 `qa_hud.py`**（56 类 `tile_detector.pt` + HudReader + `assemble_hud`，val 局 906 帧）：
  检测侧 906/906 每字段恰 1 框（0 缺 0 重）；读值 scores×4 / round / seat_wind / kyotaku / honba /
  riichi×4 全 **100%**，buttons exact-set 99.0%；wall_count 78.3%（=上述时序噪声，非缺陷）、按钮帧
  recall 32/41（帧先于按钮渲染，采集时序线）。整帧全对 77.5%，缺口几乎全由 wall_count 贡献。
- **遗留**：wall_count 时序噪声可选治理（label 侧用下一 seq 的 leftTileCount 或 eval 侧容 ±1）；
  按钮帧 recall 走 multishot/op-delay 采集线（§1.41）；HUD 检测器目前用的 OBB 07-09 权重即可，
  56 类 HBB 未单独评。全测试套件 51/51 绿。

### 1.54 单帧 HUD 集成落地 + 全量回归验收（2026-07-09）
- **动机**：§1.53 训出 `hud_reader.pt`（CTC 数字读取器 + round/wind 分类头）后，运行时链路
  （`assemble`/`check_observed`/`reconstruct`/CLI）此前只吃检测框、不读 HUD——本轮把训练好的
  HudReader 接进单帧识别链，闭合"HUD 半边训练完成但没接线"的缺口。spec：
  `docs/superpowers/specs/2026-07-09-hud-integration-design.md`。commits `786d925..3bf5ec2`。
- **四处行为变化**：
  1. `qa_hud.py` 的 `assemble_hud` 调用点改吃运行时 `Detection` 对象（此前吃的是另一种表示，
     从未真正联调过，`786d925`/`cbdee2e`）。
  2. `assemble(dets, region, frame_bgr=None, hud_reader=None)` 新增两个可选参数：两者都给
     且 `region.ox==0`（16:9 全幅——宽屏手机帧 `ox>0` 直接跳过）时才用 `assemble_hud` 填充
     ObservedState 的 scores/bakaze/kyoku/honba/kyotaku/left_tile_count/seat_wind_self/
     pending_buttons（立直棒沿用横牌/GT 逻辑，见 3）（`5d4d7e8`）。
  3. `check_observed` 新增三条 HUD×视觉交叉校验，均可拒收整帧：`kyotaku < 可见立直棒数`（硬）、
     `scores` 守恒 `sum(scores) + 1000×kyotaku == 100000`（硬）、余牌墙守恒
     `pred = 70 − Σ河 − #杠 − 是否摸牌中` 允许 ±1（吸收时序噪声）（`ab468fd`）。立直棒权威化：
     `observed_from_board` 的 reach 判定改为「横牌 ∨ `state.reach`」（GT 优先于像素）；
     `reconstruct._search` 对「无横牌但 GT 已知在立直」的座位强制走 ghost 立直棒绑定
     （`must_reach`），绑不上则整帧拒收、不静默放过（`ef5d664`）。
  4. CLI `recognize_frame.py` 默认打包加载 `hud_reader.pt`，`--no-hud`/`--hud-weights` 可关/换权重
     （`7e222b6`）；`eval_reconstruction.py` 的 assemble 层新增 HUD 逐字段报告
     （`hud_ok`/`hud_err`/`hud_missing`）+ `score_anim_rejected` 计数，`rejected_reasons` 新增
     `hud_scores`/`hud_kyotaku`/`hud_wall` 三类；oracle 层对 score-anim 帧的投影改
     `include_hud=False`（`b51b4ab`，`3bf5ec2` 修了 `hud_scores` 拒收信息文本含 "kyotaku" 导致的
     分类误判顺序）。
- **Task 8 全量回归实测**（51 tests + oracle 全量 + assemble×2 + samples/，详见
  `.superpowers/sdd/task-8-report.md`；零代码改动，纯验收）：
  - 单测 **51/51 全绿**。
  - oracle 全量 `captures/raw/ai_session`：**10121/10121 ok，0 mismatch，0 skipped_violations，
    209 call-pending skipped**——与 M1 基准零漂移。
  - assemble `run_8/game1` **`--no-hud`**：909/936 全等，拒收 13（reason 分布与基准完全一致：
    meld_parse 3 / river_geometry 10 / tile_gt4 2）。全等数比旧基准 911 低 2，落在 brief 声明的
    ±2 容差内，落差归因于 3 条新增的 reach 投影口径改变（3 帧转入 `zone_errors.reach`，不影响
    拒收集合本身）。
  - assemble **带 HUD**：907/931 全等（97.42%，不低于 no-hud 的 97.12%，反而略高）；拒收升到
    18（+5，新增 `hud_kyotaku` 7 / `hud_wall` 6 / `hud_scores` 2，原三类拒收原封不动）；
    `score_anim_rejected` 7→11，即新增 5 帧拒收中 4 帧（80%）由分数动画窗口解释。`hud_ok`
    逐字段：`scores` 99.89%（1 帧 missing）、`bakaze`/`kyoku`/`honba`/`kyotaku`/`seat_wind_self`
    各 **100.00%**、`left_tile_count`（±1 容差）99.03%——较 §1.53 未加容差时的 78.3% 大幅改善，
    符合加容差本意（吸收 GT-vs-像素时序噪声，非新问题）。
  - `samples/` 16 张真实手机截图（含宽屏 >16:9）：**16/16 全过**，宽屏帧 HUD 留 null 未引入
    误拒。
- **已知缺口（留给后续，不在本轮范围）**：
  1. 宽屏手机帧（`region.ox>0`）HUD 字段留 null——HUD 框标注全部来自桌面端 16:9 采集，检测器
     没见过手机端 HUD 布局，本轮机械跳过而非误判，未追加手机 HUD 标注/训练。
  2. score-anim 帧（reach/reach_accepted 分数动画窗口）运行时会被新增的分数守恒交叉校验拒收——
     这是预期行为（HUD 数字本就在动，标记为不可靠窗口是设计目标，不是缺陷）。
  3. 按钮帧 recall 缺口沿袭 §1.53（采集时序：帧先于按钮渲染，需 multishot/`--op-delay` 线补齐，
     本轮未动）。

### GPU（2026-06-27）
`auto` 环境已装 **torch 2.11.0+cu128 + torchvision 0.26.0+cu128**（替掉 +cpu），RTX 5080 可用。
`train_classifier.py` 自动用 cuda；加了 `--workers`（GPU 建议 6）+ 逐轮 `train_loss/val_acc/耗时` 打印（`python -u` 看实时进度）。

### 测试（`tests/test_*.py` 全部，conda `auto` 环境）
跑全部（先 activate `auto` 环境）：`for t in tests/test_*.py; do PYTHONPATH=. python "$t" || break; done`
（PowerShell：`$env:PYTHONPATH="."` 后 `foreach ($t in Get-ChildItem tests/test_*.py) { python $t.FullName }`）
覆盖：核心（tiles/replay/sync/label/classifier/coords/annotate_pipeline/annotate_frame/detector）、
采集（autoplay_gt/autoplay_stability/autoplay_autonext/schema_writer/roi_diff/
mjcopilot_gt/paths/downstream_rewire）、质量门（quality/build_gate/consistency/consistency_golden/purge_occlusion）、
杂项（overlay/gamemeta/mycv_baseline）。（`test_river`/`test_meld` 已随旧几何删除，§1.13。）

---

## 二、关键经验（实测结论）

1. **3D 牌桌随分辨率线性缩放；2D HUD 不缩放** → 手牌/河/副露归一化坐标可跨分辨率；分数/局/余牌/dora 是 HUD，需锚点归一化（且本来有协议 GT）。
2. **`last_op_step` 每局重置**（非全局）→ 帧名碰撞会覆盖整局；改用全局 `seq` 标帧/join。
3. **loguru 默认 stderr sink** 把 bridge DEBUG 打到终端破坏 Textual TUI → 启动前 `remove(0)`。
4. **快速电脑对局 + 动画桌布** → "settle 后丢骑跨帧"89% 被丢；改 **debounce-to-quiet + 帧差稳定确认**（看事件不看像素）。
5. **数据多样性是精度上限**：1 局 85.7% → 2 局 **93.5%**，邻牌混淆(6p→7p、8m→9m)消失。一局的 4 万裁剪多为近重复（同一弃牌持续~10 帧），**distinct 物理牌**才是关键。
6. **自动标注阶段须用默认素色桌布**（动态桌布/立绘破坏轮廓/掩膜）；花桌布帧留作训练鲁棒性，靠 bootstrap 标注。
7. **非全屏采集可救**：`crop_game.py` 裁回 16:9 后坐标 99.5% 对齐。

---

## 三、数据集现状（2026-07-05 GPU 机 28 局扁平重建；版本化 v1 见 §1.22）

| 层 | 内容 |
|---|---|
| 原始 | `captures/raw/ai_session/` **18 AI 局**（run_1；run_3×4；run_4×1 掉线局；run_5×3，其中 2 局曾信箱、已就地去黑边为干净 1920×1080（§1.34）；run_7×1；run_8×6；run_13/14 早退迷你局各 ~15 帧）+ **`raw/ai_session2/` 10 换肤局（run_21..23，`--skins`，§1.31）** + `raw/manual/` session5/6（4K 手动，**采集方式已过时**、数据保留） |
| 数据集 | GPU 机现役 **扁平 `datasets/detector[_obb]`**（28 AI 局，**train 10901 / val 949**，held-out 整局 `ai_run_8_game1`）+ 每局 `precise_/obb_precise_<game>`；版本化 `datasets/v1/`（20 局，train 8683/949，§1.22）仍可 `--resume` 增量 |
| 权重 | 正式 `recognize/tile_classifier.pt`（val 0.9991，07-03 数据）+ `recognize/tile_detector.pt` = **OBB 0.9946/0.9848（07-05，28 局）**；变体 `weights/detector/tile_detector_{hbb,obb}.pt`（HBB **0.9940/0.9653**）+ `*_0704.pt` 备份 |

> ⚠️ 检测器已于 07-05 在 28 局数据重训并提权（§1.31）；**分类器仍未重训**（命令见 §五 0c；
> `build_datasets.py` 收尾也会打印）。
> 红五充足（每类 ~1.4k）、`back` 3.4w+。旧 `session*_erode`/`ai_g*` 数据集已被 precise_* 取代并删除。
> `captures/raw/temp/` = 无 GT 孤儿帧（垃圾，待清理）；`captures/captures.7z` = 20.5GB 手工备份（建议移出）。

## 四、怎么运行

**见 [PIPELINE.md](PIPELINE.md)**（权威：一图流、各阶段命令、增量 SOP、过时组件清单、维护规约）。
速记：`autoplay_ai --live --auto-next` 采集 → `build_datasets.py v2 --hbb --obb --resume` 增量并入
（或 `build_datasets.py v3` 建新版本）→ 训练 `launch_classifier.sh --dataset v2 --gpu 0` /
`launch_detector.sh {hbb|obb} --dataset v2 --gpus IDS`。

---

## 五、路线图（未来计划）

### 近期（高价值，低成本）
0. **【P1 ✅ DONE 2026-06-27】修抓帧时序污染 → 净化训练标签**（见 §1.6）。
   牌面占比门修掉空毡误标 crop（93.5→95.3，+0.4 真实+1.4 测量修正）。剩余可选：在 `FrameSyncer`
   加更强像素稳定确认让**未来采集**在源头就不产生半渲染帧（当前 build 阶段门已够用）。
0c. **【部分 DONE 2026-07-06，见 §1.38】在 v2（28 局纯 AI + 换肤）上重训**：检测器 HBB/OBB **已重训**
   （`launch_detector.sh`，HBB 0.992/0.957、OBB 0.994/0.981）。**剩：分类器**尚未 v2 重训——
   `launch_classifier.sh --dataset v2 --gpu 0`（吃换肤外观多样性 + hero-tsumo 手牌帧）。
1. **【采集已全自动】多采对局** —— `autoplay_ai --live --auto-next` 整场循环采集，统一 GTRecord、
   新 run 自动入管线（§1.19/§1.21）。继续多采（不同皮肤/分辨率、3人）推鲁棒性。
   ~~补 record_gt.py F11 手动局做交叉源~~（手动路线已过时，§1.21）。
1b. **【P2 ✅ 部分 DONE】river 93.7→94.8**（见 §1.7）：河格 erode 修掉 3s→2s/4p/侧家 S。混淆矩阵**否定了 mycv 白底孤立路线**（错因是几何+数据，非邻牌点子渗入）。剩余河差距（2s→5s、红五）= **数据**问题 → 归并到"多采局"。可选后续：重标定 RIVER_QUADS（根治偏移，替代 erode 补偿）。
2. **【✅ DONE 2026-07-03，见 §1.15】训 YOLO 检测器** —— `yolov8s`、16 局免费 YOLO 标签、held-out `ai_run_8_game1` **mAP50 0.993 / mAP50-95 0.955**，正式 `tile_detector.pt`。剩余：跑满 60 epoch（`--batch 4` 防 OOM）微调；**OBB**（旋转座/riichi 横放，`poly_original` 已有）；用检测器 **bootstrap 精修副露/dora**（检测器找框 → GT 顺序赋类）；域随机化 + 外部截图/手机端实测。
3. **`frame → 结构化场况` 推理封装** —— 把 分类器 + 确定性ROI + 重放器 接成单一运行时识别器（不需新数据/依赖）。

### 中期
- **锚点归一化** (`normalize.AnchorLocator`)：检测 UI 地标 → 拟合变换 → 支持任意分辨率/手机/外部截图（手机端中间不变、两侧延伸）。
- **主动学习闭环**：检测器自动标注新帧 → 用协议 GT 交叉校验 → 低置信/不一致路由人工 → 重训。
- **副露精修**【几何模型 ✅ 可视化验证 2026-06-29，见 §1.9，待并入】：单桌面单应 `H_table` + 每家 reverse/anchor=end 列模型 + 三种杠 + z 视差 lift，
  已在 AI 1080p + session6 4K 上 GT 驱动验证 box 贴面。**下一步**：并入 `coords.py`/`meld.py`（替代 strip 补偿），用其重建副露标注并实测 on-tile 精度；
  加杠离面堆叠牌需补桌面法向投影。（替代旧"per-meld strip 偏松/OBB/bootstrap"路线。）
- **HUD 区**（分数/场风/局/余牌/供托/本场/动作按钮）：设计稿
  `docs/superpowers/specs/2026-07-04-hud-detection-design.md`（55 类 YOLO v2 + micro-readers）
  **代码已落地**（§1.31，2026-07-06：55 类 taxonomy + 字段/按钮标定 + 采集侧 multi-shot/
  `--op-delay`/`is_call_window`）；**读取器已训（§1.53，2026-07-09）**——`hud_reader.pt` 在 v4
  训练，端到端 `qa_hud.py` 除 wall_count 时序噪声/按钮帧 recall 外全字段 100%。玩家名 OCR 仍出范围。
- **红五/稀有类专项**：过采样、copy-paste 增广、合成、定向采集。

### 远期
- 38 类检测器导出 **ONNX**，CPU 实时部署（~10–40ms）。
- 落地：协议无关视觉 Bot / HUD 叠加 / 识别外部截图。
- 3 人麻将、多皮肤、域随机化（鲁棒性切片）。
- 评估与数据治理：按局/会话划分、整盘精确匹配率、混淆矩阵、数据集版本/血缘。

---

## 六、风险 / 合规（提醒，详见 DESIGN.md §7）
- **Akagi = AGPLv3 + Commons Clause**：`capture/` 复用 Akagi 是衍生作品（仅训练期工具，模型权重不含 Akagi 代码）；若商业化须独立实现 liqi 解析以隔离。
- **封号**：优先被动采集（观战/人工对局），不开 autoplay；小号。
- **时间同步**：必须有人核 golden set，不能只对 auto-GT 自评。
