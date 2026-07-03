# 出牌动画遮挡治理 —— 逐框一致性门 + 捕捉稳定确认(设计）

> 状态：**已批准设计，待写实现计划**。日期 2026-07-03。
> 相关：`state.replay.is_deal_window`（发牌帧治理，本设计的姊妹件）、`scripts/data/purge_deal_frames.py`（一次性清理模板）。

## 1. 背景与问题

在 FiftyOne 里审查 detector 数据集时发现：**相对较多的帧在出牌动画（手臂把牌甩向牌河）进行中被截取**，导致弃牌**飞行中悬在牌桌中央**、尚未落进河网格。GT 几何把该弃牌的框放在**已 settle 的网格位**（此时是空毡/手臂），于是框内容与 GT 类不符 → 坏 crop / 坏 YOLO 框。

实测两帧为证（`datasets/precise_ai_run_1/yolo/images/`）：
- `000034.png`：开局连发，多枚弃牌同时悬在中央，河网格位是空毡。**坏帧**。
- `000567.png`：中盘，四家河已全部 settle，最新弃牌在框内。**好帧**。

## 2. 根因（实测）

捕捉时机在动画结束前抓帧。且**按路径不同**：

- **AI 路径**（`scripts/capture/autoplay_ai.py` → `maybe_screenshot`，产出大部分 `precise_ai_run_*` 数据）：只等 `--quiet`（默认 **0.40s**）事件安静就截，**完全没有像素稳定确认**。Majsoul 出牌→河动画常 > 0.40s → 抓到飞行中。**主来源。**
- **手动路径**（`majsoul_eye/capture/sync.py` `FrameSyncer`）：**有**稳定确认（`confirm_stable`/`diff_thresh=3.0`，注释明确写"等出牌动画结束"），但 (a) `capped`（burst > `settle_cap=2.0s`）时**绕过**确认强截；(b) `diff_thresh=3.0` 为了忽略动画桌布调得偏松，缓慢手臂可蒙混过关。

## 3. 关键约束：无法靠 GT 状态区分

与发牌帧治理不同——发牌是干净的状态谓词（`rivers` 全空）。本问题是**间歇的、取决于捕捉时机**：**同一个 GT 状态**（"对家刚出牌"）既能产出坏帧（000034）也能产出好帧（000567）。因此**检测必须基于视觉/分类器**，或在捕捉端预防。这条否决了"`last_event==dahai` 就整帧丢"的状态谓词方案（会误杀好帧）。

## 4. 目标 / 非目标

**目标**
- B：清理现有 16 局数据集里的遮挡坏框/坏帧；并在 build 时自动挡（重建保持干净）。
- A1：给 AI 捕捉加稳定确认，未来采集不再产生（并小修手动路径的 capped 绕过 / 松阈值）。

**非目标**
- 不重采集现有 16 局（用清理而非重录）。
- 不改 38 类分类器权重、不改 GT 几何标注管线的框位置逻辑。
- 不追求零假阴性（见 §9 残余风险，已接受）。

## 5. 设计 B —— 逐框一致性门

### 5.1 一致性打分器（新模块 `majsoul_eye/annotate/consistency.py`）

输入：一帧图 + 它的 GT 框列表（`(cls, box)`）。逐框裁剪 → 跑正式分类器 → 判定坏框：

```
bad(box) := (top1 != gt_cls)  AND  (P(gt_cls) < TAU)
            OR  empty_felt(crop)          # 兜底：框内几乎全蓝毡/低亮度
```

- **分类器改动**：`recognize/classifier.py` 的 `TileClassifier.predict` 现只返回 argmax 类名。新增 `predict_proba(crops) -> list[(name, conf)]`（或返回 GT 类概率），供门取 `P(gt_cls)` 与 top1/conf。`predict` 保持不变（向后兼容）。
- **`empty_felt` 门**：复用 build_dataset 已有的 face-fraction / 亮度思路（`--min-face-frac` / `--min-bright`），把"框落在空毡"直接判坏，不依赖分类器对空毡的瞎猜。
- **裁剪来源**：现有数据直接从 `yolo/images/<seq>.png` + `yolo/labels/<seq>.txt` 裁（归一化框 → 像素框）；build 时走 `annotate/frame.py` 的 `crop_box`/`iter_tile_boxes` 同一裁剪。
- 门是**通用标签质量门**：抓的是"框内容≠GT"，不只出牌遮挡，也顺带抓错标/手臂挡/飞行中。

### 5.2 逐框智能丢规则

每帧对所有框打分后：

| 坏框数 | 动作 |
|---|---|
| 0 | 原样保留 |
| 1 ≤ n ≤ **M**（阈值，初值 2） | **逐框丢**：从 YOLO label 删这些框 + 删对应分类 crop，保留该帧与其余好框 |
| n > M | **整帧丢**：删 `yolo/images/<seq>.png` + `yolo/labels/<seq>.txt` + 该 seq 全部 crops |

M 小 → 多枚飞行的连发帧（如 000034）走整帧丢，规避"画面中央有可见牌却无框"的假阴性；单枚坏框走逐框丢，保数据。

### 5.3 挂载点

1. **一次性 purge 工具** `scripts/data/purge_occlusion_frames.py`（仿 `purge_deal_frames.py`）：遍历 `datasets/precise_*/`，逐框重打分，按 §5.2 删框/删帧，再重写 `datasets/detector*/{train,val}.txt`（丢已不存在的图行）。**dry-run 默认、幂等**。可选 `--write-manifest` 供无 captures/ 的机器上便携执行。
2. **build 时门**：在 `scripts/train/build_dataset.py`（及 `scripts/annotate/annotate_ai_session.py` 的 QA 出口）挂同一个 `consistency` 打分器，位置与 `is_deal_window` 丢帧同侧，重建自动挡。

## 6. 设计 A1 —— 捕捉端稳定确认

- **AI 路径**（`autoplay_ai.py` `maybe_screenshot`）：事件安静后，连抓两帧、`frame_diff <= thresh` 才存（照搬 `FrameSyncer` 逻辑）；否则等下一 tick。为对付动画桌布：diff 只算**牌桌 ROI**（屏蔽桌布边框与 2D HUD），阈值可收紧。新增 `--stable-thresh` / 复用/调大 `--quiet`。
- **手动路径**（`FrameSyncer`）：capped 时也做一次快确认（而非直接绕过）；`diff_thresh` 改为**区域限定** diff，使阈值能收紧到抓得住缓慢手臂而不被桌布触发。
- A1 与 B 共享一个 `frame_diff` / ROI-mask 小工具，避免两处重复。

## 7. 标定（实现第一步）

先跑量化脚本扫全部 8631 帧：统计每帧坏框数分布、分类器在"好框 vs 坏框"上的 `P(gt_cls)` 分布 → 据此定 **TAU** 与 **M**。用金标准帧（000034 坏 / 000567 好）人工校验判定方向正确。**在定阈值前不批量删数据。**

## 8. 测试

- 单测 `consistency`：对 000034（多枚飞行）断言判出坏框且走整帧丢；对 000567（干净）断言全过。
- 空毡门单测：纯蓝毡 crop 判坏、正常牌 crop 判好。
- purge 工具：dry-run 不落盘、`--apply` 幂等（二次运行零改动）、detector split 重写正确。
- A1：`FrameSyncer` 已有单测框架（注入 grab/now/sleep）；给 AI 路径的稳定确认补同类可注入单测。

## 9. 实施顺序与文件

**顺序**：B（清现有数据，你当前痛点）→ A1（防未来）。

**新增**
- `majsoul_eye/annotate/consistency.py` —— 打分器 + 智能丢决策。
- `scripts/data/purge_occlusion_frames.py` —— 一次性清理。
- 量化脚本（可临时置于 scratchpad 或 `scripts/inspect/`）。

**改动**
- `majsoul_eye/recognize/classifier.py` —— 加 `predict_proba`。
- `scripts/train/build_dataset.py` / `scripts/annotate/annotate_ai_session.py` —— 挂 build 时门。
- `scripts/capture/autoplay_ai.py` —— A1 稳定确认。
- `majsoul_eye/capture/sync.py` —— A1 手动路径小修 + 共享 ROI diff。

## 10. 残余风险（已与用户确认接受）

- **假阴性**：逐框丢后，飞行中但画面可见的牌无框 → detector 可能学到"看见牌不框"。缓解：M 设小使多飞行帧整帧丢；单枚坏框多为已被手臂挡住/飞出网格，风险小。**已接受。**
- **分类器自身错误**：一致性门以分类器为裁判，分类器在弱类（如 `5pr` 90.8%）上可能误判好框为坏 → 误删。缓解：TAU 取保守、只在 top1≠GT **且** 低置信双条件下判坏；量化阶段核查误删率。
- **detector split 一致性**：删帧后须同步重写 `detector*/{train,val}.txt`，否则训练读到缺图行（purge_deal_frames 已有此逻辑，复用）。
