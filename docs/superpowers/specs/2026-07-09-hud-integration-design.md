# 单帧 HUD 集成 — 设计（2026-07-09）

把已训练的 HUD 半边（56 类检测器 `tile_detector_obb_20260709_055509.pt` + `HudReader`
`recognize/hud_reader.pt`，STATUS §1.53）接进单帧识别链，填满 `ObservedState` 的 HUD
字段，让 `reconstruct` 用真实值替代默认值，并加 HUD×视觉交叉校验。多帧 mjai 流（局面
复原第 3 步）**不在本轮范围**，另起 spec。

## 现状与缺口

- 链已 shipped：`TileDetector → assemble → ObservedState → reconstruct → mjai`，CLI
  `scripts/recognize/recognize_frame.py`；16:9 基准 911/936 整帧全等、手机截图 16/16。
- `assemble()` 对 HUD 检测（`det.tile is None`）直接 `continue`；`ObservedState` HUD 字段
  全 `None`；`reconstruct` 回退默认（scores=25000×4、bakaze=E、kyoku=oya_rel+1、hero_abs=0）。
- `hudstate.assemble_hud` 存在但接口是 `(cls, xyxy)` 元组，与 `Detection` 对象从未接合。
- `reconstruct` 已会消费 HUD 字段（`seat_wind_self` 收敛 oya 搜索、`kyoku` 定绝对座次、
  scores/honba/kyotaku 回填 start_kyoku）——缺的只是喂进去。

## 决策（用户已确认）

1. **范围**：单帧 HUD 集成；多帧流另起一轮。
2. **冲突策略**：分字段——语义无歧义硬拒（violation 整帧拒收），wall_count ±1 容差。
3. **架构**：方案 A——扩展 `assemble()`，不新建合成层。

## 1. 数据流与接口

```
TileDetector(56类) ──┬─ tile 检测 ──→ assemble 现有牌面逻辑（不变）
                     └─ HUD 检测 ──→ hudstate.assemble_hud + HudReader
                                       ↓
        assemble(dets, region, frame_bgr=None, hud_reader=None) → ObservedState(HUD 填满)
                                       ↓
        reconstruct(obs) → 真实 scores/bakaze/kyoku/honba/kyotaku/绝对座次 的 mjai
```

- `assemble()` 加两个可选参数；都给时 HUD 检测转给 `assemble_hud`，结果经 `_fill_hud`
  回填。不给则行为逐位不变（38 类旧权重 / 无 reader 均优雅降级，字段留 `None`）。
- `assemble_hud` 签名改为直接吃 `Detection` 对象（`det.name` / `det.xyxy`），返回 dict 不变；
  更新 `tests/test_hudstate.py` fixture。
- CLI `recognize_frame.py`：默认加载打包的 `recognize/hud_reader.pt`（存在即用），加
  `--no-hud` / `--hud-weights` 开关；输出 JSON 的 `observed` 自动带 HUD 字段。
- HUD 检测的置信度并入 `zone_confidence`（`"hud"` 键，min score）。

## 2. 字段映射（`_fill_hud`，assemble.py 内部）

| assemble_hud 输出 | ObservedState | 备注 |
|---|---|---|
| `scores` 四键全非 None | `scores=[self,right,across,left]` | 缺任一 → 整体留 None（reconstruct 需全列） |
| `round` "E1".."N4" | `bakaze=text[0]`, `kyoku=int(text[1])` | ROUND_CLASSES 即此格式 |
| `wall` | `left_tile_count` | |
| `kyotaku` / `honba` | `kyotaku` / `honba` | |
| `seat_wind` | `seat_wind_self` | |
| `buttons` | `pending_buttons` | reader 在场即为 list（可为空）；不在场留 None |
| `riichi`（reach 棒+锚点归属） | **OR 进 `o.reach[r]`** | 横牌 ∨ 立直棒 |

## 3. 交叉校验（`check_observed`，仅对应字段非 None 时启用）

- **硬拒**：
  - `kyotaku < 可见立直座数（reach[] 为真计数）` → violation。
  - `sum(scores) + 1000×kyotaku ≠ 100000` → violation（顺带拒掉分数滚动动画帧——
    正是 HUD 不可靠窗口）。
- **±1 容差**：`wall_pred = 70 − Σ可见河长 − #杠(全类型) − (有摸牌槽?1:0)`；
  `|wall_pred − left_tile_count| > 1` → violation。
  推导：每次打牌前有一摸，除 chi/pon 后强制打（免摸）；被叫走的牌其摸随牌离河、恰被
  免摸打抵消；每杠净减 1（岭上补牌）。±1 吸收对手摸牌在途 + §1.53 的 pixel=GT−1
  时序噪声。这是**幻影框/漏检的强守恒防线**——每多/少一张可见牌都打破等式。
- 纯 ObservedState 算术，不依赖 reconstruct；oracle 层对 10121 帧 GT 投影免费复验
  （期望 0 新 violation）。

## 4. reconstruct：立直棒作为权威 reach 信号

已知静态盲区：宣言牌被碰/杠走且此后无再打 → 河无横牌 → 单帧看不出立直。立直棒补上：

- `obs.reach[r]=True` 且河无横牌（`side_idx[r] is None`）时，搜索必须把该座 reach 绑定
  到它的某个 ghost（被叫走的打牌）——扩展现有 `variants.append(True)` 条件。
- 终局约束（`all_done` / 终止分支）：凡 `obs.reach[r]` 为真的座，序列结束时必须已宣告；
  无可绑定 ghost → 分支不可行 → 整帧拒收（物理矛盾，宣言牌必在某处）。
- 收益：mjai 不漏 `reach`/`reach_accepted`，scores 回填 ±1000 不再错。
- oracle 投影 `observed_from_board` 的 `reach[]` 同步改为 横牌 ∨ 已接受立直，与视觉
  定义对齐（eval 公平性）。

## 5. eval 与验收

- `eval_reconstruction.py` assemble 层：HUD 字段逐项 vs GT 投影（`include_hud=True`），
  run_8 game1 基准出 per-field 准确率；`is_score_anim_window` 帧跳过 HUD 比对。
- 验收线：
  - 16:9 基准整帧全等不回归（≥911/936 量级）。
  - HUD 字段对齐 §1.53 qa_hud 水平（wall ±1、按钮帧 recall 缺口为已知）。
  - 用户 16 张手机截图重跑不回归。**宽屏 HUD 是 2D 锚定、框位不同——本轮只保证不因
    HUD 误拒（宽屏上字段可留 None），宽屏 HUD 读数质量单列 known-gap。**

## 6. 测试

- `_fill_hud` 映射（含部分 scores 缺失→None）。
- wall 算术：无杠、各类杠、叫牌（含 hero call-pending）、有/无摸牌槽。
- kyotaku / 分数和 校验触发与不触发。
- 立直棒-无横牌 reconstruct fixture（绑 ghost 成功 + 无 ghost 拒收两例）。
- `assemble_hud` 新接口（Detection 对象）。
- 降级基线：无 reader / 38 类检测时输出与现基线逐位一致。

## 7. 杂务与纪律

- 开工前先单独提交工作区 §1.53 遗留（train_hudreader.py / tests/test_hudreader.py /
  docs/PIPELINE.md / recognize/hud_reader.pt）。
- 收尾：`docs/PIPELINE.md` 同步（recognize 链默认带 HUD）、STATUS 新条目（§1.54）。
- 不触碰数据集/标注产物（纯 runtime 侧改动，`out/`、`datasets/` 不 stale）。
