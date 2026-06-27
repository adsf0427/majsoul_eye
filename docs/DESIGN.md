# majsoul_eye — 雀魂场况图像识别 + Akagi 自动标注 方案

> 本文件是已批准方案的仓库内镜像（权威设计文档）。原始计划：
> `C:\Users\zsx\.claude\plans\steady-honking-crescent.md`。

## Context（背景与目标）

**目标**：在新项目 `d:\code\phoenix\majsoul_eye`（当前为空）里，构建一套**准确识别雀魂场况**的图像识别系统。"准确识别雀魂元素"是一切下游用途（协议无关的视觉 Bot、HUD 叠加、识别外部截图）的共同前置，因此本项目的本质交付物是**鲁棒的雀魂元素识别器**。

**用户已确认的关键约束**：
- 识别范围 = **全场况**：己方手牌+摸切、四家牌河(舍牌)、副露/dora/立直棒、分数/场风/巡目/计时，"最好是所有游戏相关元素"。
- 部署环境 = **多客户端**：Win 客户端优先（可做 16:9），同时要泛化到浏览器版、手机版(非 16:9，关键元素位置不变但两侧场景会延伸)、以及**任意来源的外部雀魂截图**。
- 技术取向 = **开放选型**，但必须容忍跨客户端的位置漂移。
- 与 `auto/mycv` 关系 = **干净重写，复用 mycv 资产**作为 baseline/bootstrap。
- 主要产物 = **两者都要**：① 喂 Bot/HUD 的结构化场况状态管线；② 能识别外部截图/手机端的可泛化检测模型 + 自动标注数据集。

**两项已验证的关键事实，直接决定方案形态**：

1. **`D:\code\phoenix\auto\mycv` 已是一套能跑的纯视觉雀魂 Bot**（非 MITM），已实证解决了本方案几乎所有难题——它提供 baseline、坐标知识、4 座位透视数学、轮廓检测河/副露逻辑、已训练分类器与初始数据。**不要绿地重造**，要站在它肩上。
2. **Akagi (`D:\code\phoenix\Akagi`) 通过 MITM 拦截 liqi protobuf 拥有完整场况 GT**。它在本方案中的角色是**训练期 Oracle（提供免费、准确的标签）**，不是运行时依赖——因为有协议时根本不需要视觉，视觉的全部价值在 feed-less 场景。

---

## 1. 核心结论（TL;DR）

| 问题 | 结论 |
|---|---|
| 用什么技术？ | **混合架构**：易区(手牌/dora/分数/按钮)用**确定性 ROI 裁剪 + 小 CNN 分类 + 数字/OCR**；难区(四家河/副露,3D 透视)用**检测器(YOLOv8/11，必要时 OBB)**；跨分辨率/外部截图用**锚点归一化 + 检测器泛化**。 |
| 需要 YOLO 吗？ | **需要**——但仅因为要识别外部截图/手机端/布局漂移。若只识别自己 16:9 客户端，确定性裁剪+分类就够（YOLO overkill）。YOLO 是泛化产物的核心，不是已知客户端的必需。 |
| 标注怎么来？ | **Akagi 当训练期 Oracle 出标签(WHAT) + 几何/轮廓定位出框(WHERE)**，自动产出 YOLO 格式标注；mycv 的"轮廓定位 + GT 顺序赋类"本身就是一个免费自动标注器。再加 bootstrap/主动学习闭环。 |
| 要不要拆游戏资源做合成数据？ | **可选，非主路径**。mycv 证明直接 crop 游戏内真实牌训练就够；合成仅作稀有类/外部鲁棒性的补充。 |
| 最大风险？ | ①标签-画面时间不同步(正确性);②封号(自动化点击);③Akagi 是 AGPLv3+Commons Clause(衍生作品/商业化约束)。 |

---

## 2. 技术路线调研与选型

雀魂是 **Unity/WebGL 轻 3D 透视场景**，棋盘元素分两类，决定了"一种方法打天下"不可行：

- **易区（近正面、可枚举、位置稳定）**：己方手牌、dora 指示、动作按钮、分数/名字/场风/本场/立直棒/牌山计数 → 位置基本固定，**确定性裁剪 + 小 CNN/数字分类器/OCR** 即可，又快又准。
- **难区（透视、变长、可遮挡、有横置）**：四家牌河、副露 → 必须有**定位**步骤；河会随回合增长、立直牌横置、副露按叫自哪家旋转。

### 2.1 候选技术对比

| 技术 | 适用 | 失效场景 | 本项目用途 |
|---|---|---|---|
| 模板匹配(OpenCV matchTemplate) | 固定 UI/按钮/图标 | 透视/换皮肤/分辨率变化即崩 | 仅按钮等 UI 锚点(mycv 已用) |
| 确定性 ROI 裁剪 + 小 CNN 分类 | 位置已知的牌(手牌/dora) | 位置未知/任意截图 | 易区主力(复用 mycv `TileNet`) |
| 轮廓检测 + 宽高比规则 + 分类 | 透视河/副露(位置已知客户端) | 强遮挡/任意分辨率不稳 | 难区**定位**(mycv 已实证) + 自动标注 |
| 目标检测 YOLOv8/v11(+OBB) | 位置未知、布局漂移、任意截图、旋转牌 | 训练/标注成本 | **泛化产物核心**(外部截图/手机端) |
| OCR(PaddleOCR/Tesseract) | 分数/名字/文字 | 小字号需限 ROI+节流 | 名字→OCR；分数→数字分类器更稳 |

### 2.2 推荐架构（统一数据集，双产物共享）

```
任意输入截图
   │
   ├─[可选]帧分类器：过滤非棋盘帧(登录/结算/广告/维护)
   │
   ├─板面定位/归一化：检测稳定 UI 锚点 → 拟合变换 → 映射到 canonical 16:9 坐标系
   │      (这一步让"固定槽逻辑"在任意分辨率/手机端/外部截图上复活)
   │
   ├─易区(fast path)：确定性 ROI 裁剪 → 38 类 CNN 批量分类(手牌/dora) + 数字分类器(分数/本场/立直棒) + OCR(名字)
   │
   ├─难区(general path)：YOLO 检测器(38 类，必要时 OBB) → 四家河/副露的牌框+类别+座位+横置
   │
   └─融合 → 结构化场况状态(产物①) ；同一检测器即产物②(外部截图识别器)
```

- **产物①(结构化状态/喂 Bot/HUD)**：在已知客户端走 fast path 为主，CPU+ONNXRuntime 实时(~10–40ms)，多半不需 GPU。
- **产物②(泛化检测器/数据集)**：YOLO 检测器吃自动标注数据集，对外部截图/手机端鲁棒。
- 二者共享：**38 类牌面 taxonomy**(`majsoul_eye/tiles.py`)、`tile.model` bootstrap、Akagi 自动标注引擎、同一数据集。

### 2.3 "何时该上 YOLO" 决策规则

| 用确定性裁剪+分类即可(YOLO overkill) | 必须上检测器 |
|---|---|
| 视口固定、已知客户端、可枚举位置(手牌/dora/分数) | 河透视格点遮挡/收缩/横置导致固定坐标不稳 |
| 有 GT 免费标签、只服务自己 | 需对 UI 改版/换皮肤/任意窗口鲁棒 |
| | **从第三方截图/直播/手机端识别(无 MITM、无固定坐标)** ← 用户硬需求 |

---

## 3. 标注数据获取方案（用户核心问题）

**总思路**：`Akagi 协议 GT = WHAT（每张牌是什么、谁打的、分数多少）`；`几何/轮廓 = WHERE（牌在屏幕哪个像素框）`。两者一拼即得 YOLO/分类标签，**零手绘**。先例成熟（Atari-ALE 从 RAM 标注、"Playing for Data" 从 GTA 图形管线重建语义标签），不是新风险。

### 3.1 GT 捕获（强烈建议**被动旁路**，不开 autoplay）

- 钩入 Akagi 的 MITM bridge（`mitm/bridge/majsoul/bridge.py` 的 `parse`），**每个游戏事件**记录两份 GT：
  - **(A) MJAI 事件流**（`MajsoulBridge.parse()`）：干净归一化，覆盖手牌/河/副露/dora/立直/风/局首分数。
  - **(B) 原始 liqi dict**（`LiqiProto.parse()`）：超集，额外有 `left_tile_count`(牌山)、`moqie`(手切/摸切)、局中 `scores`、`liqibang`、完整 `ActionHule`(结算/里宝/点移动)。**MJAI 会丢弃这些字段**，必须直接读 liqi。
- 自建**状态重放器**：消费事件流，在每个 tick 重建完整四家场况(`paihe/fulu/baopai/scores/...`)。
- **采集模式**：只需 `(截图 + GT)` 对，**不需要驱动 autoplay**。可在人工对局/**观战**时被动采集 → 绕开绝大部分封号风险（"驱动 autoplay"和"采集 GT"是两件事，后者不需前者）。
- **结算屏 GT**：MJAI 把 `ActionHule/NoTile/LiuJu` 全塌缩成裸 `end_kyoku`（`bridge.py` 中 hora 事件被注释）→ 要干净结算数据须**直接读原始 liqi**，或 fork 反注释该行。

### 3.2 截图与时间同步（**最大正确性风险，须实测**）

- 协议事件在**动画渲染之前**触发；Akagi 现有 `get_screenshot()` 但**无任何 live 调用**，同步管线完全未搭。
- 同步键 `bridge.last_op_step` 每个 `ActionPrototype` 自增（非每帧）；step 相等**不保证画面已稳定**。
- **方案**：检测到新 `last_op_step` 后 → 等固定 settle(起步 400–600ms) **并/或** 帧差稳定再截图 → 用该 step 标记帧 → 丢弃跨两事件的"骑跨帧"。可关闭游戏动画、缩小窗口降抖。**实际 settle 时长须在录像上经验测量。**
- Win 客户端用 `mss`/Win32 抓窗口；浏览器用 Playwright `page.screenshot()`（默认 1600×900 恰 16:9，`px=nx*1600, py=ny*900`，无 letterbox）。

### 3.3 三层坐标模型（全部归一化 0–1，复用 Akagi `_normalize_to_pixel()`）

1. **固定槽表（易区）**：复用 mycv 的 1080p 像素坐标(`zuobiao.py` / `main2.py` 注释里大量实测坐标，如分数 `img[473:491,920:935]`、手牌 flood fill 起点 `(235,1002)`)→ 归一化 → 适配目标分辨率。补全 Akagi 缺的：dora/四家分数/自风/立直棒/副露锚点/本场区/牌山计数器的 **bbox**（Akagi 现有 `majsoul_positions.py` 只有 14 手牌中心点 + 9 按钮 + 候选行，且只有中心点无宽高）。
2. **透视河/副露（难区，决策闸门）**：
   - **首选(mycv 已跑通)**：白掩膜 + `cv.findContours` + 按 `(x,y)` 区域 + 宽高比 `w/h` 阈值分座位/判横置（`main2.py:255-299` 的 `cutPic/lizhipai`）。
   - **类别赋值**：从 GT 的**有序弃牌列表**按顺序赋给检测到的牌框——**不靠像素推类**。这就是免费自动标注器：CV 找框，GT 给标签。
   - **备选**：每座位 4 点标定 homography `(row,col)→像素`；但 mycv 用区域+宽高比而非单一 homography，暗示单平面 homography 可能不够稳——**先试 mycv 路线**。
3. **四座位屏幕坐标系**：Akagi 是 hero 视角(self 永远在底)。复用 mycv 的 `mask_location` + `transfer1/2/3()`，`screen_quadrant=(actor-hero_seat)%4`。**4 套不同几何**：底(近正面)/右(侧视)/顶(远+倒置缩小)/左(侧视)，非简单 90° 旋转。**3 人麻将**布局不同(`scores[3]==0` 检测)，另备一套。

### 3.4 自动标注 → bootstrap/主动学习闭环

```
v0：mycv tile.model + 707 帧 + §3.3 自动框 → 训练 YOLO 检测器 v1
v1：自动标注更多真实帧 → 用 Akagi GT 交叉校验(检测结果 vs 协议 GT)
     ├ 一致 → 直接进训练集
     └ 不一致/低置信 → 路由人工(Label Studio) → 硬样本
重训循环（标注成本降 2–10x）
```

- **Akagi GT 同时是自动验证器**：检测器读出的板面与协议 GT 不符 → 该帧即硬负例。
- 工具：Label Studio + YOLO ML-backend、Ultralytics auto-annotate、FiftyOne 看数据/查错。
- **自洽不变量**（写帧前校验否则丢）：每牌种≤4(含 aka)、手牌 13/14、河长≤摸牌数、同牌不在两处。

### 3.5 跨分辨率/外部截图/手机端鲁棒性

- **锚点归一化**：检测稳定 UI 地标(角标/按钮/计分区) → 拟合相似/透视变换 → 把任意截图映射回 canonical 16:9 → 固定槽逻辑复活。手机非 16:9：中间牌桌不变、两侧延伸 → 以中间锚点定位最稳。
- **域随机化**：训练时随机分辨率/裁剪/皮肤/亮度/背景，让检测器对布局漂移鲁棒。
- **合成数据（可选）**：从游戏拆 tile 精灵合成已知框板面(`Cut,Paste,Learn` + 域随机化)；仅补稀有类/外部鲁棒性。注意资源提取涉及版权(见 §7)。

### 3.6 类别不均衡

字牌/红五/杠/拔北/远家缩小牌稀少 → 按类 recall 驱动**过采样 + copy-paste 增广**；混淆矩阵盯易混对(1m/9m、4s/字牌、aka5 vs 5、远家缩小牌、牌背 vs 面)。

---

## 4. 场况元素总表 [GT 来源 | 位置确定性 | 推荐方法]

> MJAI = `MajsoulBridge.parse()` 事件；liqi = `LiqiProto.parse()` 原始 dict(MJAI 丢弃的字段)。

| 元素 | GT 来源(Akagi 字段) | 位置确定性 | 推荐方法 |
|---|---|---|---|
| 己方手牌(13+摸,含红五) | MJAI `start_kyoku.tehais[self]`+`tsumo` | **高**，14 固定槽 | 裁剪 + 38 类 CNN |
| 对家暗手牌(仅牌数) | 仅数量(MJAI 发 `?`) | 中(需补坐标) | 不识别牌面，`back×N`，数=13−副露 |
| **四家河 ×4** | MJAI `dahai{actor,pai,tsumogiri}` 重放；摸切=liqi `moqie`；立直牌=`reach` 后首张 | **低**，透视变长无坐标 | **mycv 轮廓+宽高比定位 → CNN/检测器**；GT 顺序赋类 |
| **副露 ×4** | MJAI `chi/pon/daiminkan/ankan/kakan/nukidora{actor,target,pai,consumed}` | 低，旋转无坐标 | 同河；旋转角编码叫自哪家 |
| dora 指示(数+牌) | MJAI `start_kyoku.dora_marker`+`dora` | 中，需补坐标 | 裁剪 + 38 类 CNN |
| 里宝牌 | **仅 liqi** `ActionHule.li_doras` | 仅结算屏 | 需结算屏，读 liqi |
| 立直/立直棒/本场 | MJAI `reach`/`reach_accepted`；`start_kyoku.kyotaku/honba` | 中，需补坐标 | 横置靠宽高比；计数靠数字分类器 |
| 场风/亲/局/自风 | MJAI `start_kyoku.bakaze/oya/kyoku`；自风=(seat−oya)%4 | 中，需补坐标 | 裁剪 + 字符分类器 |
| 各家分数 | MJAI `start_kyoku.scores`；局末 **liqi** `ActionHule.delta_scores` | 中，需补坐标 | 每位数字分类器(mycv 已做) |
| 牌山计数 | **仅 liqi** `ActionDealTile.left_tile_count` | 中，需补坐标 | 数字分类器，标签读 liqi |
| 行动者/最新摸/最新弃 | MJAI 最新事件 actor | 高 | 事件流隐式；视觉读高亮 |
| 可用动作(按钮) | MJAI `_update_operation` op_map | **高**，9 槽(已有) | 模板/分类 |
| 名字/头像 | liqi `authGame` | 中，需补坐标 | 名字→OCR(节流)；头像不识别 |
| 计时器 | liqi `OptionalOperationList.time_*`(实时倒计时是动画) | 中 | 一般不作 GT |
| 结算屏(役/符番/点移动/里宝) | **仅 liqi** `ActionHule/NoTile/LiuJu` | 仅结算屏 | 必须读原始 liqi |

**视觉也读不到的隐藏信息**：对家暗手牌牌面(仅数量)、未翻牌山/里宝(直到局末揭示)。

---

## 5. 复用 mycv 资产清单（干净重写时直接搬/参考）

| 资产 | 路径 | 用途 |
|---|---|---|
| `TileNet` 38 类 CNN + 权重 | `auto/mycv/classifier2.py` + `tile.model` | 易区分类 baseline / bootstrap 自动标注 |
| ResNet 分类器 | `auto/mycv/myweight.pth` (main2.py 用) | 备选/对比 |
| 轮廓河/副露定位 | `auto/mycv/main2.py:230-299` (`cutPic/lizhipai`) | 难区定位 + 自动标注核心 |
| 4 座位透视变换 | `auto/mycv` `mask_location`/`transfer1/2/3`/`m/m0-m3.png` | 座位↔屏幕象限映射 |
| 1080p 实测坐标 | `auto/mycv/zuobiao.py`、`main2.py` 注释 | 固定槽表种子 |
| 707 帧真实截图 | `auto/mycv/debug/` | 初始训练/验证数据 |
| 多分辨率探索 | `auto/mycv/test_resolution.py`、`config2.toml` `auto_detect_resolution` | 跨分辨率适配参考 |
| 38 类 taxonomy + MJAI 编码 | `auto/mycv/CLAUDE.md`（已固化进 `majsoul_eye/tiles.py`） | 统一类别定义 |
| Akagi 归一化→像素 | `Akagi/autoplay/executor/playwright_executor.py:192-234` | 归一化 BBOX→像素 |
| Akagi 现有归一化坐标 | `Akagi/autoplay/positions/majsoul_positions.py` | 手牌/按钮槽位种子 |

> 坐标基准不一致：mycv=**1920×1080**，Akagi/Playwright=**1600×900** → **不可直接互用**，统一到归一化 0–1 再换算。

---

## 6. 分阶段路线图

| 阶段 | 目标 | 验收 |
|---|---|---|
| **P0 复用调研** | 跑通/读懂 mycv，在新截图上核对坐标是否漂移；定 38 类 taxonomy 与归一化坐标系 | mycv 在当前客户端读对手牌/分数 |
| **P1 GT 旁路** | Akagi hook：每帧记 MJAI+raw liqi；建状态重放器；取结算 liqi | 重放过自洽不变量 |
| **P2 捕获同步** | 接截图(Win32/mss + Playwright)；事件驱动 + settle + 帧差；标 step；丢骑跨帧 | golden set 上 GT↔帧一致 |
| **P3 MVP 识别(易区)** | 复用 TileNet(手牌+dora)+数字分类器(分数)+OCR(名字)；锚点归一化雏形 | 手牌 top-1 / 分数串精确达标 |
| **P4 难区+自动标注** | mycv 轮廓+宽高比定位 + GT 顺序赋类 → 自动产 YOLO 标签；4 象限几何 | 河有序序列精确 / 编辑距离达标 |
| **P5 检测器(泛化产物)** | 训 YOLOv8/11(+OBB)；域随机化；外部截图/手机端测试 | OBB mAP + 整盘精确匹配率达标 |
| **P6 鲁棒/闭环** | 主动学习 + 类均衡 + 准确率哨兵 + 皮肤/分辨率切片；导出 ONNX | golden set 退化触发重训 |

---

## 7. 风险与合规

### 时间同步 desync（最大正确性风险）
settle 未测、协议先于渲染、骑跨帧 → **必测动画时长**、settle+帧差、关动画、丢骑跨帧。**评估必须有人核 golden set，不能只对 auto-GT 自评**(auto-GT 本身可能 desync 错)。

### 封号/ToS（被低估）
雀魂内置客户端反作弊。风险主要来自**自动化点击** + 客户端篡改 + 行为信号。**建议**：①**优先被动捕获(观战/人工对局)不开 autoplay**；②一次性小号；③必须 autoplay 时保留 0.05–0.15s 点击抖动、短会话、间歇人工。mitmproxy 拦截本身亦 ToS-adverse。

### 许可与 IP（潜在阻断级）
- **Akagi = AGPLv3 + Commons Clause**(`Akagi/LICENSE.txt`)。AGPLv3:网络服务衍生作品**须公开源码**；Commons Clause:**禁止出售**。majsoul_eye 若复用 Akagi 的 bridge/liqi/mjai_bot（设计上要 tee 其流）**几乎肯定是衍生作品** → 阻断商业化、强制源码披露。**若有商业化意图，应独立实现 liqi 解析(如开源 mahjong-soul protobuf 定义)以隔离 Akagi 代码。**
- 提取/再分发雀魂精灵或截图数据集违反 Yostar 版权 + ToS，限制公开性。

### 工程风险
| 风险 | 缓解 |
|---|---|
| 重连 `syncGame` 事件重放 → 河/副露重复计数 | 显式重连检测 + 状态重置/对账 |
| 非棋盘帧(结算/登录/广告/维护) | 帧分类先过滤 |
| 坐标漂移(1080p/900p/客户端版本) | P0 新鲜截图核对；坐标记版本元数据 |
| `moqie`(手切/摸切)对家可靠性 | 协议是唯一来源，叫牌后弃牌验证几局 |

---

## 8. 评估与数据治理

- **指标按区域分别定义**：手牌/dora top-1；河 有序序列精确 / 编辑距离；检测器 mAP@.5、mAP@.5:.95、OBB mAP；分数 数字串精确；名字 CER；**整盘精确匹配率**(一张错即错，下游真正关心)。
- **按局/会话划分 train/val/test**（绝不按帧——同局近重复帧泄漏虚高）；切片：座位象限/皮肤/区域/稀有度/遮挡；感知哈希去近重复。
- **数据集版本/血缘**：每样本记 客户端版本/Akagi commit/皮肤 id/视口/settle/labeler 版本。工具 DVC/FiftyOne（注意 Roboflow 免费层数据公开，与版权冲突）。

---

## 9. 技术栈与依赖（均为净新增，须与 torch 兼容核对）

- 识别：`opencv-python`、`torch`(mycv 已用)、`ultralytics`(YOLOv8/11/OBB)、`onnx`+`onnxruntime`(CPU 部署)、`paddleocr` 或 `pytesseract`。
- 捕获：`mss`/`pywin32`(Win 客户端)、`playwright`(浏览器，Akagi 已用)。
- 标注/数据：Label Studio、FiftyOne、(可选 DVC)。
- 实验跟踪：W&B 或 MLflow。
> `Akagi/requirements.txt` 仅有 `torch==2.5.1`，无 ultralytics/onnx/opencv/ocr —— 全是新依赖。

---

## 10. 验证方式（端到端怎么测）

1. **P0**：用 `auto/mycv` 现有脚本在当前客户端跑一局，确认 mycv 识别仍准、坐标未漂移（基线锚定）。
2. **P1–P2**：录 1–2 局，对每个 step 落盘 `(截图, MJAI, liqi, 重建状态)`；人工抽查 20–30 帧，确认**截图内容与 GT 完全对得上**（同步正确性 golden set）。
3. **P3–P4**：在 golden set 上跑识别管线，按 §8 指标算手牌/河/分数准确率；用 Akagi GT 做自动对账，列混淆矩阵。
4. **P5**：拿**外部来源截图 + 手机端截图**(训练分布外)测检测器 mAP 与整盘精确匹配率，验证泛化。
5. **P6**：搭准确率哨兵——新客户端版本/皮肤上 golden set 指标退化即告警触发重训。

---

## 待实施前再决定（不阻塞）
1. 是否需要支持 **3 人麻将 / 多皮肤**（影响类别数、象限映射、域随机化面）。
2. 是否有**商业化**意图（决定是否必须独立实现 liqi 解析以规避 Akagi AGPL/Commons Clause）。
3. 结构化状态产物是否需直接产出 **MJAI 格式**(复用 mycv 的 `dictx1` 编码可无缝喂 Mortal)。
