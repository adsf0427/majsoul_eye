# majsoul_eye — 项目状态与路线图

> 活文档：**已完成的部分 + 未来计划**。设计与论证见 [DESIGN.md](DESIGN.md)。
> 最后更新对应进度：6 局分类器 97.6%（§1.8）；**副露单桌面单应 H_table 几何 spike 已可视化验证**（§1.9，2026-06-29，未并管线）。

## TL;DR

- 端到端管线已验证：`录制(F11) → 截图同步 → 协议GT重放 → 几何自动标注 → 训练`，**零手绘标注**。
- 牌分类器 **97.6%**（6 局：2 手动 + 4 AI 半自动；session6 held-out 公平验证）。轨迹
  93.5(原)→95.3(P1清洗)→96.0(P2 erode)→**97.6(+AI数据)**；跨局(held-out AI 局)96.2→**98.5**。
  （注：分类器=**单牌牌面分类**，非检测器。93.5→95.3 见 §1.6(P1)；→96.0、河 93.7→94.8 见 §1.7(P2)；
  →97.6 见 §1.8（接入 MahjongCopilot 半自动采集，4 AI 局）。）
- **mycv 真实精度已实测**（见 §1.5）：分类近乎已解决，瓶颈是 recall/检测与抓帧时序——定路线图的关键结论。
- 数据：**6 局 → ~190k 裁剪**（2 手动 4K + 4 AI 1080p）。红五已充足（AI 局 286/357/246 vs 手动 ~8）。
- 核心论点已实证：`协议GT(WHAT) + 几何(WHERE) = 免费且正确的标签`，且精度随被动采集的对局数单调上升。

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
  接口，**按局/kyoku 切分**（内联 `seq_to_kyoku`，零跨脚本 import）。不拷图：写 `train.txt`/`val.txt`（**原生 OS 分隔符绝对路径**——
  Ultralytics 靠替换路径里 `images`→`labels` 段找标签，Windows 上必须原生分隔符，勿 POSIX 化）+ `data.yaml`（`names` 取自 `tiles.TILE_NAMES`）。
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

### GPU（2026-06-27）
`auto` 环境已装 **torch 2.11.0+cu128 + torchvision 0.26.0+cu128**（替掉 +cpu），RTX 5080 可用。
`train_classifier.py` 自动用 cuda；加了 `--workers`（GPU 建议 6）+ 逐轮 `train_loss/val_acc/耗时` 打印（`python -u` 看实时进度）。

### 测试（11 套全绿，conda `auto` 环境）
`test_tiles / test_replay / test_sync / test_label / test_classifier / test_mycv_baseline / test_quality / test_coords / test_annotate_pipeline / test_annotate_frame / test_detector`
（§1.13：`test_river`/`test_meld` 随 `label/river.py`·`meld.py` 一并删除。§1.15：新增 `test_detector`——data.yaml 名单==TILE_NAMES、切分无泄漏、封装冒烟。）

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

## 三、数据集现状
| 数据集 | 来源 | 帧 | 分类裁剪 |
|---|---|---|---|
| `datasets/session5_erode` `session6_erode` | 2 手动局 4K（P1清洗+P2 erode 重建） | 1,029 | ~39,646 |
| `datasets/ai_g1` `ai_g2` `ai_g3` `ai_r1` | 4 AI 局 1080p（MahjongCopilot，§1.8） | ~2,031 | ~73,069 |
| **合计** | **6 局** | ~3,060 | **~112k**（含训练用增广前裁剪） |
> 旧 `session6_hr`/`session5`（污染标签前）保留作对照。erode 版是当前正式训练集。
> 红五现充足：4 AI 局合计 5mr 286 / 5pr 357 / 5sr 246（手动局每局~1 张）。
> **新增原始数据** `captures/ai_session`（MahjongCopilot 线流 + 1080p PNG，多 run/13 局）：当前用于 §1.9 副露几何 spike 验证（含 session6 缺的 大明杠/加杠），
> 尚**未**建成训练集。注：`run4` 为掉线局、`run5` 为重连局（见 `captures/ai_session/notes.txt`）——并入训练前需注意此类局的状态连续性。
> 权重：`tile_classifier.pt`=正式(6局,97.6%)；备份 `_preAI`(P2 erode)/`_erode`/`_clean`/`_prePollutionFix`。

## 四、怎么运行（conda `auto` 环境）
```bash
PY=C:/Users/zsx/miniforge3/envs/auto/python.exe
# 1) 采集（akagi 环境装 mss/opencv；F11 全屏、默认桌布、别中途重启）
python scripts/capture/record_gt.py --screenshots --quiet 0.3 --out captures/sessionN.jsonl
# 1b) 若非全屏，裁回 16:9
python scripts/data/crop_game.py captures/sessionN captures/sessionN_16x9 --size 3840x2160
# 2) 建数据集
$PY scripts/train/build_dataset.py captures/sessionN.jsonl captures/sessionN_16x9/ --out datasets/sessionN
# 3) 训练（多局跨局 val）
$PY scripts/train/train_classifier.py --data s5=datasets/session5/crops:captures/session5.jsonl \
    --data s6=datasets/session6_hr/crops:captures/session6.jsonl --val s6:E3.0,S2.0
# 4) 测试
for t in tiles replay sync label river meld classifier; do PYTHONPATH=. $PY tests/test_$t.py; done
```

---

## 五、路线图（未来计划）

### 近期（高价值，低成本）
0. **【P1 ✅ DONE 2026-06-27】修抓帧时序污染 → 净化训练标签**（见 §1.6）。
   牌面占比门修掉空毡误标 crop（93.5→95.3，+0.4 真实+1.4 测量修正）。剩余可选：在 `FrameSyncer`
   加更强像素稳定确认让**未来采集**在源头就不产生半渲染帧（当前 build 阶段门已够用）。
1. **【部分 DONE】多采对局** —— 已接入 MahjongCopilot 半自动采集 + `convert_mjcopilot.py`（见 §1.8），4 AI 局 → 97.6%、红五充足。继续多采（不同皮肤/分辨率、3人）推向 ~99% 并凑检测器集。可选：补 `record_gt.py` F11 手动局做交叉源。
1b. **【P2 ✅ 部分 DONE】river 93.7→94.8**（见 §1.7）：河格 erode 修掉 3s→2s/4p/侧家 S。混淆矩阵**否定了 mycv 白底孤立路线**（错因是几何+数据，非邻牌点子渗入）。剩余河差距（2s→5s、红五）= **数据**问题 → 归并到"多采局"。可选后续：重标定 RIVER_QUADS（根治偏移，替代 erode 补偿）。
2. **【✅ DONE 2026-07-03，见 §1.15】训 YOLO 检测器** —— `yolov8s`、16 局免费 YOLO 标签、held-out `ai_run_8_game1` **mAP50 0.993 / mAP50-95 0.955**，正式 `tile_detector.pt`。剩余：跑满 60 epoch（`--batch 4` 防 OOM）微调；**OBB**（旋转座/riichi 横放，`poly_original` 已有）；用检测器 **bootstrap 精修副露/dora**（检测器找框 → GT 顺序赋类）；域随机化 + 外部截图/手机端实测。
3. **`frame → 结构化场况` 推理封装** —— 把 分类器 + 确定性ROI + 重放器 接成单一运行时识别器（不需新数据/依赖）。

### 中期
- **锚点归一化** (`normalize.AnchorLocator`)：检测 UI 地标 → 拟合变换 → 支持任意分辨率/手机/外部截图（手机端中间不变、两侧延伸）。
- **主动学习闭环**：检测器自动标注新帧 → 用协议 GT 交叉校验 → 低置信/不一致路由人工 → 重训。
- **副露精修**【几何模型 ✅ 可视化验证 2026-06-29，见 §1.9，待并入】：单桌面单应 `H_table` + 每家 reverse/anchor=end 列模型 + 三种杠 + z 视差 lift，
  已在 AI 1080p + session6 4K 上 GT 驱动验证 box 贴面。**下一步**：并入 `coords.py`/`meld.py`（替代 strip 补偿），用其重建副露标注并实测 on-tile 精度；
  加杠离面堆叠牌需补桌面法向投影。（替代旧"per-meld strip 偏松/OBB/bootstrap"路线。）
- **HUD 区**（分数/场风/局/余牌/名字）：锚点相对定位 + 数字分类器/OCR（多为协议 GT，优先级低）。
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
