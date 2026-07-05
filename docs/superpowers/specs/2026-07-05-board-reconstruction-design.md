# 局面复原：单帧 ObservedState + 合法 MJAI 序列重建 — 设计 spec

- 日期：2026-07-05
- 状态：调研已呈报、用户确认方向（序列消费者 = Mortal hero 视角；本 spec 覆盖第 1+2 步；
  第 3 步多帧动态暂缓，仅在 §9 记定位）
- 前置阅读：`docs/PIPELINE.md`、`majsoul_eye/state/replay.py`、
  `docs/superpowers/specs/2026-07-04-hud-detection-design.md`（本 spec 依赖其 schema 槽位，
  但**不修改**该设计，两者正交推进）

## 1. 目标与范围

从单帧截图复原完整可见局面（第 1 步），并构造一条从 `start_kyoku` 到当前状态的**合法
MJAI 事件序列**（第 2 步）——最终打通「纯视觉驱动 Mortal」。序列形态 = hero 视角部分信息
（他家 `tsumo` 的 `pai="?"`、他家 `tehais=["?"]×13`），与 Akagi/Mortal 的消费格式一致。

三个支撑事实（调研已验证）：

1. `BoardState` 就是「完整局面」的现成定义——ObservedState 是它的单帧可见投影；
   `annotate/seatgt.seat_gt()` 已在做 GT 方向的同一投影，反向即评测器。
2. MJAI 在 hero 视角本来就是部分信息协议——「看不到他家手牌」不阻塞重建。
   mycv（`refactored/game/mjai_converter.py`）证明 Mortal 消费此形态可行，但 mycv 是
   增量式（= 第 3 步），**单帧重建无先例，是本 spec 的新造部分**。
3. 每个 GTRecord 帧 = 免费测试用例（截图 + 真实 mjai 流），三层评测零标注成本（§6）。

范围内：四麻；手牌/摸牌槽、四家河（序 + 立直横牌）、四家副露（组成/来源家/红5）、
宝牌指示牌、牌背数——全部走现有 38 牌类 OBB 检测；HUD 九字段（分数/场风局数/本场/
供托/余牌/自风/按钮）只留 **Optional schema 槽位**，读值由 HUD spec 的微读取器落地后填充。

范围外：三麻（拔北）、摸切/手切恢复（单帧原理不可见，第 3 步职责）、结算/中断/选局
画面、HUD 读取器本身、多帧追踪。

## 2. 架构总览

```
截图(BGR) ─ TileDetector(OBB) ─┐
                               ├─ recognize/assemble.py ──> state/observe.ObservedState
BoardRegion(normalize) ────────┘         (几何反查装配)             │
                                                                   ▼
                    Mortal / libriichi  <── mjai events <── state/reconstruct.py
                    Replayer(自洽验证)  <──────┘              (回合模拟+回溯 DFS)
```

新增模块（均为运行时产品侧或纯逻辑，不动数据管线）：

| 模块 | 职责 | 依赖 | 边界 |
|---|---|---|---|
| `state/observe.py` | `ObservedState`/`ObservedRiverTile`/`ObservedMeld` 数据类 + 单帧一致性校验 | 仅 `tiles` | 纯数据，无 cv2/Akagi |
| `recognize/assemble.py` | `list[Detection]` + `BoardRegion` → `ObservedState` | `annotate.pipeline`（几何）、`normalize`、`coords` | Akagi-free ✅（`annotate` 包本身 Akagi-free；`pipeline` 是纯几何。**决定：recognize 直接 import `annotate.pipeline`**，不复制几何常量——单一事实源优先于「annotate=dev 包」的印象分；`capture/` 边界不破） |
| `state/reconstruct.py` | `ObservedState` → mjai 事件列表 | 仅 `tiles`/`observe` | 纯逻辑，无视觉依赖，可全合成测试 |
| `scripts/eval/eval_reconstruction.py` | 三层 GT 评测 harness | `capture.gtframes`、上述模块 | QA 工具（PIPELINE.md §4 分类），非 pipeline 阶段 |

## 3. 第 1 步：ObservedState 与装配层

### 3.1 数据模型（`state/observe.py`）

座位约定：**屏幕相对位** 0=self 1=right(下家) 2=across(对家) 3=left(上家)，与
`seatgt.SEAT_POS` 一致（单帧没有绝对座位概念；绝对化是 reconstruct 的事，§4.3）。

```python
@dataclass ObservedRiverTile:
    pai: str                  # 38 类名（红5 区分）
    sideways: bool            # 横放渲染（立直宣言位；幽灵位边角见 §4.4）

@dataclass ObservedMeld:
    type: str                 # chi | pon | daiminkan | ankan | kakan
    tiles: list[str]          # 完整组成（红5 精确；暗杠红5 由"每色一张红5"推断）
    called_pai: str           # 横牌身份（ankan 为 ""）
    added_pai: str            # kakan 叠横的加牌，其余 ""
    from_rel: int             # 被叫家相对 meld 所有者的偏移 1/2/3（ankan=0）

@dataclass ObservedState:
    # —— 3D 桌面（本期可识别）——
    hero_hand: list[str]
    drawn_tile: Optional[str]         # 摸牌分离槽；无则 None
    rivers: list[list[ObservedRiverTile]]   # ×4，屏幕相对位序
    melds: list[list[ObservedMeld]]         # ×4
    dora_markers: list[str]                 # 死墙面板左→右
    concealed_counts: list[Optional[int]]   # 牌背计数（self 位 None；对手手牌排离桌面
                                            # 标定区远，装配首期置 None——纯交叉校验字段，
                                            # 重建不依赖它）
    reach: list[bool]                       # 由 rivers 的 sideways 推导后固化
    # —— 2D HUD（Optional 槽位，55 类+微读取器落地后填充；本期恒 None）——
    scores: Optional[list[int]] = None      # 相对位序
    bakaze: Optional[str] = None
    kyoku: Optional[int] = None
    honba: Optional[int] = None
    kyotaku: Optional[int] = None
    left_tile_count: Optional[int] = None
    seat_wind_self: Optional[str] = None
    pending_buttons: Optional[list[str]] = None   # hud.HUD_NAMES 的按钮子集
    # —— 元信息 ——
    violations: list[str]                   # 空 == 可用于重建
    zone_confidence: dict[str, float]       # 逐 zone 最低检测分，诊断用
```

### 3.2 装配算法（`recognize/assemble.py`）

复用 `annotate/pipeline` 的标定几何做**反查**（正向：GT→框；反向：框→槽位/序号）：

1. `normalize.locate_fullscreen` → `P.build_homographies` → 检测框中心
   `original_to_fullwarp`。
2. **zone 归属**：hand（`coords` HandModel 域）与 dora（`DORA_STRIP`）先按原图 ROI 摘走；
   其余按 fullwarp 距离归入最近的河网格（`DISCARD_GRID`）或副露条带（`MELD_STRIP`）。
3. **河序反查**：每 seat 内，行 = 对 `DISCARD_ROW_OFFSETS` 最近匹配，列 = 沿 `dcol`
   投影取整（`DISCARD_READ` 的 colsign/disc0 约定与正向一致）；横牌判定 = OBB poly
   长边方向 vs 行进方向（HBB 退路：footprint 宽高比对照 `DISCARD_FOOT`/`RIICHI_FOOT`）。
   横牌导致的后续列挤位用与 `generate_discard_slots` 相同的 extra-shift 几何。
   **校验：占据槽位必须从 0 起前缀连续（无空洞），否则记 violation。**
4. **副露解析**：条带内沿 `MELD_STRIP.step` 方向排序 → 按「横/叠横/背」模式切组 →
   组模式定型：3 张含 1 横 = chi（顺子）/pon（刻子）；4 张含 1 横 = daiminkan；
   4 张含叠横 2 = kakan；2 背 + 2 面 = ankan。`from_rel` 由横牌在组内位置映射
   （外/中/内 ↔ 上家/对家/下家，**渲染规则需一次标定**，用 GT melds 免费验证，§7）。
5. **手牌**：hand zone 检测按 x 排序；`drawn_tile` = 与主排间距超过摸牌槽阈值的最右张
   （正向几何已在 hero-hand 修复中使用 `replay.drawn_tile` 放框，阈值取自同一标定）。
6. **dora**：`DORA_STRIP` 内非-back 检测左→右即 `dora_markers`。
7. **一致性校验**（`observe.check_observed()`，`replay.check_invariants` 的单帧版）：
   同种牌可见 ≤4；`len(hero_hand)+3×副露数 ∈ {13,14}`（含 drawn_tile 则 14）；
   河前缀连续；`len(dora_markers) − 1 ≤ 可见杠数`（明杠翻牌延迟允许 ≤）；
   牌背数 vs `13 − 3×副露数` 匹配（±摸牌）。violations 非空 → 调用方拒帧
   （静态场景等下一帧即可，与 `--drop-violations` 同哲学）。

## 4. 第 2 步：MJAI 序列重建（`state/reconstruct.py`）

### 4.1 契约

输入：violations 为空的 `ObservedState`。输出：

```python
@dataclass ReconstructionResult:
    ok: bool
    events: list[dict]        # start_kyoku..末事件；ok=False 时为 []
    reason: str               # ok=False 的不可满足说明
    fabricated: dict          # 虚构项清单：haipai、默认化的 HUD 字段、call 时机自由度
    diagnostics: dict         # 探索节点数、可行 oya 集、多解计数（上限截断）
```

末事件 = hero 决策点：`drawn_tile` 非空 → `tsumo(hero)`；有响应（按钮可见或
可推断的他家最新舍牌）→ 该家 `dahai`。

### 4.2 信息账本

| 观测钉死 | 自由变量（虚构策略） |
|---|---|
| 每家河的牌序 + 立直宣言位 | 被叫牌在原河的时间位（搜索决定） |
| 副露完整组成 + `called_pai` + `from_rel` | 全局回合交织（搜索决定） |
| hero 当前手牌 + 摸牌 | hero 起手（「全摸切」构造，§4.3） |
| 指示牌数（−1 ≤ 杠数交叉校验，明杠翻牌有延迟） | 他家摸切标志（默认 false；立直后强制 true） |
| 各家立直与宣言巡目（横牌可见） | 杠/翻 dora 的确切时机（杠后即翻简化） |

### 4.3 算法：回合模拟 + 回溯 DFS（调研已对比否决 CP-SAT 与引擎状态注入）

**预处理——oya 与绝对座位**：`kyoku` 可读时 oya_abs = kyoku−1，据 `seat_wind_self`
映射相对↔绝对并交叉校验；HUD 未落地时，按河长轮转约束枚举可行的 oya 相对位
（谁先打第一张），唯一则用之，多解取字典序第一并记入 diagnostics。hero_abs =
(oya_abs − oya_rel) mod 4。

**状态机**：状态 =（每家河 cursor、未触发副露集、当前 actor、立直标志×4）。
actor 的舍牌点按优先序分支，DFS + 失败状态 memo（cursor 元组 × 副露掩码 × actor，
上界 ≈ 河长积 × 2^副露数，实际远小）：

- (a) 打出该家可见河的下一张（优先 → 规范解 = 叫牌尽可能晚，确定性）；
- (b) 触发一个以 actor 为 target 的未触发副露：插入幽灵 `dahai`（pai =
  `called_pai`，不进可见河）→ chi/pon/daiminkan 事件 → actor 跳至叫牌者
  （daiminkan 者随后同 (c) 岭上 `tsumo` + `dora`）；
  chi 仅允许 `from_rel` 对应上家（观测本身应满足，防御校验）；
- (c) actor 自家的 ankan/kakan：`tsumo` → 杠事件 → `dora`（若还有未解释的指示牌）
  → 岭上 `tsumo` 继续。kakan 需其 pon 已在场。

终止 = 所有 cursor 耗尽、副露全触发、停在 §4.1 的末事件形态。无解（观测自相矛盾、
但 §3.2 校验漏过）→ `ok=False` + reason。

**hero「全摸切」fabrication**（已验证与 libriichi 手牌追踪自洽、恰 13 张）：
hero 每张舍牌编为 `tsumo(X) → dahai(X, tsumogiri=true)`；例外：吃/碰后的强制手切
（该张入 haipai）、杠第 4 张编为当巡摸牌。则
`haipai = 当前门清牌 + 副露手内消耗 + 吃碰后强制手切牌`。

**立直**：宣言位（`sideways`，含幽灵位边角 §4.4）前插 `reach`、后插
`reach_accepted`；该家其后 `dahai` 全部 `tsumogiri=true`。

**start_kyoku 回推**：`kyotaku_start = 读值 − 本局 reach_accepted 数`、
`scores_start[s] = 读值[s] + 1000×(s 已立直)`、honba 原样、
`dora_marker = dora_markers[0]`、tehais：hero = fabricated haipai，他家 `"?"×13`。
HUD 字段为 None 时的占位默认：`bakaze=E, kyoku=oya 推断值, honba=0, kyotaku=立直数,
scores=25000×4(+回推)`——序列合法，Mortal 顺位意识失真记入 `fabricated`，
HUD 读取器落地后自动变准。

### 4.4 边角与已知简化

- **立直宣言牌被碰走**：雀魂横放该家下一张（`river_sideways_index` 的正向逻辑）；
  反向：观测 sideways 位之前若有幽灵位，`reach` 允许绑定幽灵 `dahai`（搜索分支）。
- **明杠翻 dora 时序**：真实协议打牌后翻，简化为杠后即翻——Mortal 只累加 dora
  事件，不校验时序。
- **最后打牌者歧义**（响应帧多家河尾皆合法时）：轮转约束通常唯一；残余歧义取规范解
  并记 diagnostics。根治 = 未来给最新舍牌高亮框加一个检测类（`latest_discard_marker`，
  出本期范围，记入 HUD spec Phase B 候选）。
- **他家 moqie 失真 / call 时机非真实**：固有信息损失，用 §6 第 3 层定量，第 3 步根治。

## 5. 测试策略（TDD，纯合成优先）

1. **reconstruct 单元测试**（无视觉依赖）：合成 ObservedState 病例矩阵——每种副露型
   ×来源家×立直（宣言位首/中/幽灵位）×多杠×hero 14/13 张×按钮响应帧；断言
   `Replayer(events)` 投影全等 + `check_invariants` 空 + 无解病例给出 reason。
2. **assemble 往返测试**（无检测器）：GT BoardState → `seat_gt` 投影 +
   `generate_discard_slots`/`generate_meld_boxes_v2` 正向生成框 → 当作假 Detection
   喂 assemble → 必须还原同一 ObservedState。正向几何即反向 fixture，零标注。
3. **集成**（慢，抽样）：真帧 + 真 OBB 权重跑 §6 harness 的小切片。
   沿用 `tests/` plain-script 风格（无 pytest 依赖）。

## 6. 评测与验收（`scripts/eval/eval_reconstruction.py`，全部免费驱动）

| 层 | 输入 | 比对 | 验收 |
|---|---|---|---|
| ① 装配 | 真帧 → 检测 → assemble | vs `seat_gt` 投影，逐 zone（hand/river/meld/dora）帧级全等率 + 牌级错误分类 | 先报数基线；目标 hand ≥99%、整帧 ≥95%（OBB 0.98 mAP 推算，实测后校准阈值） |
| ② oracle 重建 | GT 投影出的完美 ObservedState → reconstruct | `Replayer` 重放投影全等 + libriichi 消费不报错 | 非拒帧（deal window/动画外）成功率 ≥99%；投影全等属定义性要求 = 100% |
| ③ 决策一致率 | 真实 GTRecord 序列截至决策点 vs 重建序列 | 分别喂 Mortal（mycv `mortal/engine.py`，依赖存在才跑），比最终决策 | 报告值，无硬门槛——它量化静态信息损失，是第 3 步立项依据 |

CLI：`--captures <dirs> --level {assemble,oracle,engine} --sample N --report out/eval/...`。

## 7. 标定任务（一次性，全 GT 驱动）

- 副露横牌位置 ↔ `from_rel` 渲染规则：现有 captures 的 GT melds 直接验证（免费）。
- 摸牌槽间距阈值：hero-hand 修复用的正向偏移反向取阈。
- ankan 面朝上两张的渲染确认（红 5 面向）：GT 对照抽查即可，识别不依赖它
  （红 5 由计数规则推断）。

## 8. 管线影响（CLAUDE.md 纪律自查）

- 不触碰 capture/annotate/dataset/training 的任何输入输出——`out/`/`datasets/` 不 stale。
- 新增脚本分类：`eval_reconstruction.py` = **QA 工具**（PIPELINE.md §4 补一行）；
  `observe/assemble/reconstruct` 为运行时产品模块，不是 pipeline 阶段。
- STATUS.md 加一节（实现落地时）。55 类检测器重训（HUD spec 已排）与本设计正交；
  assemble 本期只消费 38 牌类，权重升级后自动受益按钮/HUD 槽位。

## 9. 第 3 步定位（暂缓，另立 spec）

多帧 = 逐帧第 1 步 + 帧间 diff 出事件流（mycv 增量思路的重做），本设计的单帧重建器
天然是其**重同步原语**（掉帧/遮挡/中途进局冷启动），moqie 与 call 时机在多帧下可
直接观测。第 1+2 步是地基，不是绕路。
