# majsoul_eye — 项目状态与路线图

> 活文档：**已完成的部分 + 未来计划**。当前管线的权威描述在 **[PIPELINE.md](PIPELINE.md)**；
> 设计与论证见 [DESIGN.md](DESIGN.md)。
> 最后更新对应进度：**检测器增强显式化（fliplr 关/hsv_v 提升）+ 宝牌闪光覆盖统计**（§1.30，2026-07-05）；
> 此前：服务器侧 `regen_detector_dataset.sh` 切嵌套布局 + 训练启动器文档（§1.29）；skins 元数据
> hero 修正 + ai_session2 回填（§1.28）；采集统一 AI 路线 + 数据集版本化（§1.19–§1.22，2026-07-04）。
> （近期里程碑：dealfix 分类器 val 0.9991 §1.16、OBB 检测器 mAP50-95 0.9804 §1.17。）

## TL;DR

- 管线**只有一条主路径**（零手绘标注，版本化构建 `build_datasets.py <name>`，现役 `datasets/v1`）：
  `autoplay_ai(AI 自动对局, 实时写统一 GTRecord) → 精确标注 v2 → build_dataset(crops+YOLO)
  → detector 装配 → 训练(--dataset 可混多版本)`。手动 F11(record_gt+Akagi) 采集**已过时**
  （存量 session5/6 数据保留）。
- 数据：**18 AI 局 + 2 手动 4K 局**，全部经 2026-07-04 重建（含 hero-tsumo 修复、run_13/14 补建）。
- 模型：分类器 held-out 整局 **val_acc 0.9991**；检测器 HBB **mAP50 0.993 / mAP50-95 0.955**、
  OBB 变体 **0.9804**（rotated-IoU）。分类器轨迹 93.5→95.3→96.0→97.6→99.78→**99.91**。
- 核心论点已实证：`协议GT(WHAT) + 标定几何(WHERE) = 免费且正确的标签`，精度随对局数单调上升
  （mycv 基线实测见 §1.5：分类近乎已解决，瓶颈在检测/时序——由此定的路线已逐项兑现）。
- ⚠️ 待办：两模型尚未在 07-04 重建数据（hero-tsumo 手牌帧 + run_13/14）上重训，检测器受益最大。

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

### GPU（2026-06-27）
`auto` 环境已装 **torch 2.11.0+cu128 + torchvision 0.26.0+cu128**（替掉 +cpu），RTX 5080 可用。
`train_classifier.py` 自动用 cuda；加了 `--workers`（GPU 建议 6）+ 逐轮 `train_loss/val_acc/耗时` 打印（`python -u` 看实时进度）。

### 测试（`tests/test_*.py` 全部，conda `auto` 环境）
跑全部（先 activate `auto` 环境）：`for t in tests/test_*.py; do PYTHONPATH=. python "$t" || break; done`
（PowerShell：`$env:PYTHONPATH="."` 后 `foreach ($t in Get-ChildItem tests/test_*.py) { python $t.FullName }`）
覆盖：核心（tiles/replay/sync/label/classifier/coords/annotate_pipeline/annotate_frame/detector）、
采集与迁移（autoplay_gt/autoplay_stability/autoplay_autonext/schema_writer/roi_diff/migrate_ai/
migrate_gt_layout/mjcopilot_gt/paths/downstream_rewire/backfill_skin_meta）、质量门（quality/build_gate/consistency/consistency_golden/purge_occlusion）、
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
| 原始 | `captures/raw/ai_session/` **18 AI 局**（run_1；run_3×4；run_4×1 掉线局；run_5×3，其中 2 局信箱 → `derived/*_fixed` 修复帧；run_7×1；run_8×6；run_13/14 早退迷你局各 ~15 帧）+ **`raw/ai_session2/` 10 换肤局（run_21..23，`--skins`，§1.31）** + `raw/manual/` session5/6（4K 手动，**采集方式已过时**、数据保留） |
| 数据集 | GPU 机现役 **扁平 `datasets/detector[_obb]`**（28 AI 局，**train 10901 / val 949**，held-out 整局 `ai_run_8_game1`）+ 每局 `precise_/obb_precise_<game>`；版本化 `datasets/v1/`（20 局，train 8683/949，§1.22）仍可 `--resume` 增量 |
| 权重 | 正式 `recognize/tile_classifier.pt`（val 0.9991，07-03 数据）+ `recognize/tile_detector.pt` = **OBB 0.9946/0.9848（07-05，28 局）**；变体 `weights/detector/tile_detector_{hbb,obb}.pt`（HBB **0.9940/0.9653**）+ `*_0704.pt` 备份 |

> ⚠️ 检测器已于 07-05 在 28 局数据重训并提权（§1.31）；**分类器仍未重训**（命令见 §五 0c；
> `build_datasets.py` 收尾也会打印）。
> 红五充足（每类 ~1.4k）、`back` 3.4w+。旧 `session*_erode`/`ai_g*` 数据集已被 precise_* 取代并删除。
> `captures/raw/temp/` = 无 GT 孤儿帧（垃圾，待清理）；`captures/captures.7z` = 20.5GB 手工备份（建议移出）。

## 四、怎么运行

**见 [PIPELINE.md](PIPELINE.md)**（权威：一图流、各阶段命令、增量 SOP、过时组件清单、维护规约）。
速记：`autoplay_ai --live --auto-next` 采集 → `build_datasets.py v1 --resume` 增量并入（或
`build_datasets.py v2` 建新版本）→ 训练 `--dataset datasets/v1`。

---

## 五、路线图（未来计划）

### 近期（高价值，低成本）
0. **【P1 ✅ DONE 2026-06-27】修抓帧时序污染 → 净化训练标签**（见 §1.6）。
   牌面占比门修掉空毡误标 crop（93.5→95.3，+0.4 真实+1.4 测量修正）。剩余可选：在 `FrameSyncer`
   加更强像素稳定确认让**未来采集**在源头就不产生半渲染帧（当前 build 阶段门已够用）。
0c. **【NEW 待办】在 07-04 重建数据（datasets/v1）上重训分类器 + 检测器**（hero-tsumo 手牌帧 +
   run_13/14；检测器受益最大——own-turn 手牌此前是负样本信号）。
   命令：`train_classifier.py --dataset datasets/v1 --val "ai_run_8_game1:*"` /
   `train_detector.py --data datasets/v1/detector/data.yaml`。
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
- **HUD 区**（分数/场风/局/余牌/名字）：设计稿已有——`docs/superpowers/specs/2026-07-04-hud-detection-design.md`
  （55 类 YOLO v2 + micro-readers）；实施待排期（多为协议 GT，优先级低）。
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
