# majsoul_eye — 项目状态与路线图

> 活文档：**已完成的部分 + 未来计划**。设计与论证见 [DESIGN.md](DESIGN.md)。
> 最后更新对应进度：完成 P0–T6（自动标注全链路 + 首版牌分类器，2 局数据）。

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
| 启动器 | `scripts/record_gt.py` | 注入录制器后跑 Akagi；`--screenshots`；启动前 `loguru.remove(0)` 保护 TUI |

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
| 副露 | `majsoul_eye/label/meld.py` | 每家 1 行 strip + GT 顺序；ankan 2 背面 | self/left ~95–98%，**right/across 3D 侧家偏松(88–93%) → opt-in** |
| 默认区 | `autolabel.DEFAULT_ZONES` | = `{hand, river}`（meld/dora/score 为 opt-in） | — |

> **校准方式**：4 家河 quad + dora + 4 家副露 strip 均由**并行 subagent**在满盘帧上"看图返回坐标"标定（不改代码、可并发）；代码任务串行。

### 4. 数据集与识别模型（T6）
| 组件 | 文件 | 说明 |
|---|---|---|
| 数据集构建 | `scripts/build_dataset.py` | 同步采集 → 分类裁剪 `crops/<牌>/` + YOLO `yolo/images,labels/`；按 `seq` join；亮度门 `--min-bright` 丢空格 |
| 牌分类器 | `majsoul_eye/recognize/classifier.py` | `TileNet`（64px，AdaptiveAvgPool，输入尺寸无关）+ `TileClassifier` 推理封装 |
| 训练 | `scripts/train_classifier.py` | 多局 `--data NAME=crops:capture` + 跨局/跨场 split `--val NAME:kyoku|*`；类均衡采样 + 轻增广 |
| 权重 | `majsoul_eye/recognize/tile_classifier.pt` | 2 局训练，**val 93.5%** |
| 非全屏修复 | `scripts/crop_game.py` | 裁回 16:9 游戏画布（session5 实证 99.5% 对齐） |
| 对账/可视化 | `scripts/inspect_capture.py`、`scripts/overlay_labels.py` | 帧↔GT join、覆盖率、坐标叠加调试 |

### 1.5 mycv 基线实测（2026-06-27，dev-only，6 路对抗审查已验证）
**目的**：早先"mycv 64%/56%/35%"是管线错配产物（已撤回）；这次跑 mycv **真实 native 管线**测公平精度。
| 组件 | 文件 | 说明 |
|---|---|---|
| 引擎适配器 | `majsoul_eye/baselines/mycv_engine.py` | 直接 import 真实 `myCV`（../auto/mycv，pyautogui/matplotlib 在 auto 环境可用），调它真实的 `cutPic/model/getType/getHandTiles`；自己驱动座位（raw mask k → 绝对座 `(hero+k)%4`） |
| 评分 | `majsoul_eye/baselines/score.py` | 多重集(bag)匹配：`recall`(=端到端=correct/n_gt)、`precision`(=correct/n_pred)。不依赖我们的坐标标定，对 mycv 公平 |
| 测量脚本 | `scripts/mycv_baseline.py` | 重放 GT → 逐帧跑引擎 → 聚合；`tests/test_mycv_baseline.py` 9/9 |

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
- **离线转换器** `scripts/convert_mjcopilot.py`（dev-only，MahjongCopilot GPL，跑 `auto` 环境/protobuf 4.25.3）：
  `LiqiProto.parse(wire)` → `GameState.input`（配 stub bot，把 libriichi 的 `bot.factory` stub 掉）→ MJAI → 我们的 `replay.py`。
  两个关键点：① GameState 按 bot 决策点批量产 MJAI，所以按**每条 input() 的增量**打 seq（逐动作对齐，98.8% 帧命中）；
  ② **GameState 原地改 AI 手牌 list**，所以捕获时必须 **deepcopy 每个事件**（否则 start_kyoku.tehais 被后续覆盖→英雄手牌 desync）。
- **教训**：起初英雄手牌 17% 违例，一度误判为 MahjongCopilot 转换 bug；实为**我的捕获存了可变引用**。deepcopy → **0% 违例**。MahjongCopilot 转换是对的。
- 4 局转换（座位 0/3/1/1，多样）→ build_dataset（P1 门+P2 erode）→ **~73k crops**，红五 286/357/246。crops 已肉眼抽查正确。
- **GPU 重训**（RTX 5080）公平验证：session6 held-out **0.9604→0.9755**（+1.5，无泄漏）；held-out AI 局 ai_g1 **0.9619→0.9851**（+2.3）。8m 0.72→1.00、3s 0.74→0.97 等大涨。
- 产物：`tile_classifier_allgames.pt` 已提升为正式 `tile_classifier.pt`（备份 `_preAI.pt`）。
- **修正**：早先"红五 0%"是 session6-val small-n（7-9 样本）假象；规模够时模型本就 ~85-100%。AI 数据的真实收益是**跨源泛化**。
- **工具**：① `scripts/ingest_run.py <run_dir> [--train --val NAME:*]` —— 一键 发现游戏→convert→build_dataset(→可选重训)，自动发现单局/多局布局。
  ② `scripts/visualize_failures.py --crops ... [--val-capture --val-kyoku] --out DIR` —— 按混淆对(gt→pred)出错例蒙太奇 + summary.txt。
  实测 session6 held-out（97.6% 模型）主错：**红五跨花色**(5mr/5pr→5sr)、2s→3s、个别难牌；多数"错误"是同一物理牌跨~N 近重复帧。

### GPU（2026-06-27）
`auto` 环境已装 **torch 2.11.0+cu128 + torchvision 0.26.0+cu128**（替掉 +cpu），RTX 5080 可用。
`train_classifier.py` 自动用 cuda；加了 `--workers`（GPU 建议 6）+ 逐轮 `train_loss/val_acc/耗时` 打印（`python -u` 看实时进度）。

### 测试（10 套全绿，conda `auto` 环境）
`test_tiles / test_replay / test_sync / test_label / test_river / test_meld / test_classifier / test_mycv_baseline / test_quality / test_coords`

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
> 权重：`tile_classifier.pt`=正式(6局,97.6%)；备份 `_preAI`(P2 erode)/`_erode`/`_clean`/`_prePollutionFix`。

## 四、怎么运行（conda `auto` 环境）
```bash
PY=C:/Users/zsx/miniforge3/envs/auto/python.exe
# 1) 采集（akagi 环境装 mss/opencv；F11 全屏、默认桌布、别中途重启）
python scripts/record_gt.py --screenshots --quiet 0.3 --out captures/sessionN.jsonl
# 1b) 若非全屏，裁回 16:9
python scripts/crop_game.py captures/sessionN captures/sessionN_16x9 --size 3840x2160
# 2) 建数据集
$PY scripts/build_dataset.py captures/sessionN.jsonl captures/sessionN_16x9/ --out datasets/sessionN
# 3) 训练（多局跨局 val）
$PY scripts/train_classifier.py --data s5=datasets/session5/crops:captures/session5.jsonl \
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
2. **训 YOLO 检测器**（需 `pip install ultralytics`）—— 泛化产物（外部截图/手机端），并以此 **bootstrap 精修副露/dora**（检测器找框 → GT 顺序赋类）。
3. **`frame → 结构化场况` 推理封装** —— 把 分类器 + 确定性ROI + 重放器 接成单一运行时识别器（不需新数据/依赖）。

### 中期
- **锚点归一化** (`normalize.AnchorLocator`)：检测 UI 地标 → 拟合变换 → 支持任意分辨率/手机/外部截图（手机端中间不变、两侧延伸）。
- **主动学习闭环**：检测器自动标注新帧 → 用协议 GT 交叉校验 → 低置信/不一致路由人工 → 重训。
- **副露精修**：per-meld 几何（处理 3D 透视 + 旋转召唤牌的间隙）或 OBB；或直接靠 bootstrap。
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
