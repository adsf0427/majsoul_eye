# HUD 元素识别（55 类检测器 + 微读取器）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把现有 38 类牌面检测器扩展为 55 类（+9 HUD 字段 +8 按钮语义类），配套 CTC 数字读取器 + round/wind 小分类器，端到端读出分数/场风/局数/本场/供托/余牌/可用按钮，全部标签由 GT 自动生成。

**Architecture:** 沿用本仓 "WHERE(几何)×WHAT(GT)" 自动标注公式：HUD 字段框 = 标定种子 ROI + 逐帧墨迹收紧；按钮框 = 轮廓候选 + `operationList` 按序赋类（数量不符整帧丢）。检测器一次前向出全部框；数字字段裁出转正后由 CRNN-CTC 免切分读串；round/wind 用 TileNet 小分类器。见 spec `docs/superpowers/specs/2026-07-04-hud-detection-design.md`。

**Tech Stack:** PyTorch（CTC/分类器）、ultralytics YOLO（检测器）、OpenCV（墨迹收紧/轮廓）、现有 annotate/build_datasets 管线。

**Spec deltas（管线在 spec 之后变了，两处落点更新）:**
1. spec §7 "录人工被动局补按钮帧"（record_gt）已弃用 → 改为 `autoplay_ai.py --op-delay` 让 AI 在有待决操作时迟疑，quiet 抓帧自然采到按钮帧。
2. spec §5.6 "16 局重建" → 现为版本化数据集：`build_datasets.py v2`（v1 现有 20 局不动）。

## Global Constraints

- conda `auto` env；本计划命令写 bash 形式 `PYTHONPATH=. $PY ...`，其中 `PY=C:/Users/zsx/miniforge3/envs/auto/python.exe`（用户在 PowerShell 自己跑时用 `$env:PYTHONPATH="."` + 裸 `python`）。
- 测试是 plain script（无 pytest 依赖，也 pytest 兼容）：`PYTHONPATH=. $PY tests/test_X.py`，失败即非零退出。
- **38 牌类顺序冻结**（`tiles.TILE_NAMES`）；HUD 类只能**追加**在 38 之后（id 38–54）。
- **`recognize/` 必须保持 capture-free**（torch/cv2 可以，禁止 import `majsoul_eye.capture`）。
- 工作区有别人的未提交改动（autoplay_ai/skins/docs）——每个 task 只 `git add` 自己的文件，禁止 `git add -A`。
- 管线纪律（用户指令）：改动落地后必须同步 `docs/PIPELINE.md` + `docs/STATUS.md` 新条目（Task 14）。
- 标注/训练数据基线：`captures/raw/ai_session/`（20 局，日文 UI）；held-out val 游戏 = `ai_run_8_game1`。
- 标有 **[USER RUN]** 的步骤需要真实浏览器/GPU/小号，由用户执行，执行者准备好命令并停下等结果。

---

### Task 1: HUD 检测类别表（`majsoul_eye/hud.py`）

**Files:**
- Create: `majsoul_eye/hud.py`
- Test: `tests/test_hud_taxonomy.py`

**Interfaces:**
- Produces: `HUD_NAMES: list[str]`（17 个，顺序=spec §3 表）、`DET_NAMES: list[str]`（55 = TILE_NAMES+HUD_NAMES）、`HUD_NAME_TO_ID: dict[str,int]`（38–54）、`OP_TO_BTN: dict[int,str]`、`buttons_for_ops(op_types: list[int]) -> list[str]`、`CTC_CHARSET: str`、`FIELD_ROT: dict[str,int]`（度数 0/90/180/270，占位 0 由 Task 6 标定回填）、`NUMERIC_FIELDS: tuple[str,...]`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_hud_taxonomy.py
"""HUD taxonomy: ids append after the frozen 38 tiles; op->button mapping dedupes."""
from majsoul_eye.tiles import TILE_NAMES
from majsoul_eye.hud import (HUD_NAMES, DET_NAMES, HUD_NAME_TO_ID, OP_TO_BTN,
                             buttons_for_ops, CTC_CHARSET, NUMERIC_FIELDS)

assert len(HUD_NAMES) == 17
assert DET_NAMES[:38] == TILE_NAMES and len(DET_NAMES) == 55
assert HUD_NAME_TO_ID["score_self"] == 38 and HUD_NAME_TO_ID["btn_skip"] == 54
assert len(set(DET_NAMES)) == 55
# an/dai/ka kan share one button; dapai(1)/babei(11) have none
assert OP_TO_BTN[4] == OP_TO_BTN[5] == OP_TO_BTN[6] == "btn_kan"
assert 1 not in OP_TO_BTN and 11 not in OP_TO_BTN
assert buttons_for_ops([1]) == []
# order = HUD_NAMES order (kan before riichi); on-screen order calibrated in Task 7
assert buttons_for_ops([1, 7, 4]) == ["btn_kan", "btn_riichi", "btn_skip"]
assert buttons_for_ops([2, 9]) == ["btn_chi", "btn_ron", "btn_skip"]
assert CTC_CHARSET == "0123456789-x余"
assert set(NUMERIC_FIELDS) == {"score_self", "score_right", "score_across", "score_left",
                               "wall_count", "riichi_stick_count", "honba_count"}
print("test_hud_taxonomy OK")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. $PY tests/test_hud_taxonomy.py`
Expected: `ModuleNotFoundError: No module named 'majsoul_eye.hud'`

- [ ] **Step 3: 实现**

```python
# majsoul_eye/hud.py
"""HUD-element detector taxonomy — the 17 classes appended after the frozen 38
tile classes (ids 38-54; spec docs/superpowers/specs/2026-07-04-hud-detection-design.md §3).

Pure data (no cv2/numpy) so every component can import it. Button classes are
SEMANTIC — CN/JP/TW glyphs are all training samples of the same class.
"""
from __future__ import annotations

from majsoul_eye.tiles import TILE_NAMES

HUD_NAMES: list[str] = [
    # center info panel (center-anchored; identical on PC/mobile)
    "score_self", "score_right", "score_across", "score_left",
    "round_label", "wall_count", "seat_wind_self",
    # top-left panel
    "riichi_stick_count", "honba_count",
    # action buttons (semantic; glyph varies per server language)
    "btn_chi", "btn_pon", "btn_kan", "btn_riichi",
    "btn_tsumo", "btn_ron", "btn_kyushu", "btn_skip",
]
DET_NAMES: list[str] = TILE_NAMES + HUD_NAMES          # 55-class detector head
HUD_NAME_TO_ID: dict[str, int] = {n: len(TILE_NAMES) + i for i, n in enumerate(HUD_NAMES)}
NUM_DET_CLASSES: int = len(DET_NAMES)
assert NUM_DET_CLASSES == 55, NUM_DET_CLASSES

# liqi operation type -> button class. Wire shape verified on run_13/game1:
# raw_liqi.data.data.operation = {seat, operationList:[{type, combination,...}], ...}.
# type 1 = dapai (no button), 11 = babei (3p, out of scope). An/dai/ka kan share
# ONE button. Codes follow Akagi/MahjongCopilot convention — re-check against
# Akagi's liqi parser if a mismatch shows up in calibration (spec §3.3).
OP_TO_BTN: dict[int, str] = {
    2: "btn_chi", 3: "btn_pon",
    4: "btn_kan", 5: "btn_kan", 6: "btn_kan",
    7: "btn_riichi", 8: "btn_tsumo", 9: "btn_ron", 10: "btn_kyushu",
}


def buttons_for_ops(op_types: list[int]) -> list[str]:
    """Pending liqi op types -> button classes expected on screen (dapai-only -> []).
    Order = HUD_NAMES order (stable); on-screen ordering is assigned by x-sort at
    annotation time, not here. btn_skip accompanies any other button — verify
    empirically at button calibration (Task 7) and adjust if own-turn-only
    options (riichi/ankan/tsumo) turn out to render without a skip button."""
    btns = [b for b in HUD_NAMES if b in {OP_TO_BTN.get(t) for t in op_types}]
    if btns:
        btns.append("btn_skip")
    return btns


# --- micro-reader contracts -------------------------------------------------
CTC_CHARSET: str = "0123456789-x余"   # model emits index+1; 0 = CTC blank
NUMERIC_FIELDS: tuple[str, ...] = (
    "score_self", "score_right", "score_across", "score_left",
    "wall_count", "riichi_stick_count", "honba_count",
)
ROUND_CLASSES: list[str] = [f"{w}{k}" for w in "ESWN" for k in (1, 2, 3, 4)]  # 16
WIND_CLASSES: list[str] = ["E", "S", "W", "N"]

# Per-class rotation (degrees CW) that uprights the crop before reading.
# score_across is upside down; left/right signs are CALIBRATED in Task 6 —
# the values below are the seed guess from captures/raw/ai_session frames.
FIELD_ROT: dict[str, int] = {
    "score_self": 0, "score_across": 180, "score_left": 270, "score_right": 90,
    "round_label": 0, "wall_count": 0, "seat_wind_self": 0,
    "riichi_stick_count": 0, "honba_count": 0,
}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. $PY tests/test_hud_taxonomy.py`
Expected: `test_hud_taxonomy OK`

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/hud.py tests/test_hud_taxonomy.py
git commit -m "feat(hud): 55-class detector taxonomy (38 tiles + 17 HUD/button classes)"
```

---

### Task 2: 待决操作提取器（`state/ops.py` + `BoardState.pending_ops`）

**Files:**
- Create: `majsoul_eye/state/ops.py`
- Modify: `majsoul_eye/state/replay.py`（`BoardState` 加字段 + `copy()` + `apply_record` 尾部一行）
- Test: `tests/test_ops.py`

**Interfaces:**
- Consumes: `GTRecord`（duck-typed：`.syncing/.raw_liqi/.seat`）。
- Produces: `ops_from_record(r) -> list[int] | None`；`BoardState.pending_ops: list[int] | None`（=该 seq 最后一条记录携带的 op types；无则 None）。语义：**帧上按钮可见 ⇔ 帧所属 seq 的快照 `pending_ops` 经 `buttons_for_ops` 非空**（外加 Task 7 的外观校验兜底）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_ops.py
"""ops_from_record: verified wire shape; syncing/wrong-seat/absent -> None.
Replayer surfaces pending_ops on the snapshot of the offering record and
clears it on the next record."""
from types import SimpleNamespace as NS

from majsoul_eye.state.ops import ops_from_record


def rec(op=None, syncing=False, seat=0, raw=True):
    inner = {"operation": op} if op is not None else {}
    return NS(syncing=syncing, seat=seat,
              raw_liqi={"data": {"name": "ActionDealTile", "data": inner}} if raw else None)


OP = {"seat": 0, "operationList": [{"type": 1, "combination": []},
                                   {"type": 7, "combination": []}],
      "timeAdd": 20000, "timeFixed": 5000}

assert ops_from_record(rec(OP)) == [1, 7]
assert ops_from_record(rec(None)) is None                    # no operation field
assert ops_from_record(rec(OP, syncing=True)) is None        # reconnect replay
assert ops_from_record(rec(dict(OP, seat=2), seat=0)) is None # not hero's offer
assert ops_from_record(rec(raw=False)) is None               # raw_liqi missing
assert ops_from_record(rec({"seat": 0, "operationList": []})) is None

# --- Replayer wiring: pending_ops rides the snapshot, next record clears it --
from majsoul_eye.state.replay import BoardState

s = BoardState()
assert s.pending_ops is None
s.pending_ops = [1, 7]
c = s.copy()
c.pending_ops.append(9)
assert s.pending_ops == [1, 7], "copy() must deep-copy pending_ops"
print("test_ops OK")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. $PY tests/test_ops.py`
Expected: `ModuleNotFoundError: No module named 'majsoul_eye.state.ops'`

- [ ] **Step 3: 实现 `state/ops.py`**

```python
# majsoul_eye/state/ops.py
"""Pending-operation extraction from a GTRecord's raw liqi message.

Wire shape (verified on captures/raw/ai_session/run_13/game1.jsonl):
    raw_liqi["data"]["data"]["operation"] =
        {"seat": <hero>, "operationList": [{"type": N, "combination": [...], ...}],
         "timeAdd": ..., "timeFixed": ...}
type codes: 1=dapai(no button) 2=chi 3=pon 4/5/6=kan 7=riichi 8=tsumo 9=ron
10=kyushukyuhai 11=babei(3p). Mapping to button classes lives in majsoul_eye.hud.
Semantics: a record carrying operationList OFFERS ops to the hero; any later
record supersedes it — so BoardState.pending_ops (set by Replayer.apply_record
from the LATEST record) is exactly "ops pending at this snapshot".
"""
from __future__ import annotations

from typing import Optional


def ops_from_record(r) -> Optional[list[int]]:
    """liqi op type codes offered to the hero by this record, else None.

    None for: syncing records (reconnect replays re-send stale offers), records
    without raw_liqi, offers addressed to another seat (defensive; each client
    normally only receives its own), and empty operationList.
    """
    if getattr(r, "syncing", False) or not getattr(r, "raw_liqi", None):
        return None
    try:
        op = ((r.raw_liqi.get("data") or {}).get("data") or {}).get("operation") or {}
        ol = op.get("operationList") or []
        if not ol:
            return None
        seat = getattr(r, "seat", None)
        if "seat" in op and seat is not None and op["seat"] != seat:
            return None
        types = [int(o["type"]) for o in ol if "type" in o]
        return types or None
    except Exception:
        return None
```

- [ ] **Step 4: 接进 `replay.py`（三处小改）**

在 `BoardState` 的 bookkeeping 区（`last_actor` 附近）加字段：

```python
    # liqi op types offered to the hero by the LATEST applied record (None if the
    # latest record carries no offer) — drives button auto-labels; see state/ops.py.
    pending_ops: Optional[list[int]] = None
```

在 `BoardState.copy()` 里（其他 list 拷贝旁）加：

```python
        s.pending_ops = list(self.pending_ops) if self.pending_ops else None
```

在 `Replayer.apply_record()` 开头（应用 MJAI 事件之前、raw-liqi superset 读取旁）加：

```python
        from majsoul_eye.state.ops import ops_from_record
        self.state.pending_ops = ops_from_record(record)
```

（放开头保证"最后一条记录"语义：每条记录都覆写；`import` 放模块顶部亦可，
`state/ops.py` 无重依赖。）

- [ ] **Step 5: 跑测试确认通过 + 回归旧测试**

Run: `PYTHONPATH=. $PY tests/test_ops.py && PYTHONPATH=. $PY tests/test_replay.py`
Expected: 两个都 OK（replay 原有行为不变）

- [ ] **Step 6: Commit**

```bash
git add majsoul_eye/state/ops.py majsoul_eye/state/replay.py tests/test_ops.py
git commit -m "feat(state): pending-operation extractor + BoardState.pending_ops"
```

---

### Task 3: 按钮帧存量盘点（`scripts/inspect/inventory_ops.py`）

**Files:**
- Create: `scripts/inspect/inventory_ops.py`

**Interfaces:**
- Consumes: `ops_from_record`、`buttons_for_ops`、`capture.schema.read_records`、`capture.gtframes.load_frames`、`paths`。
- Produces: 盘点报告（决定 Task 4 采集量）。无库接口。

- [ ] **Step 1: 实现脚本**

```python
# scripts/inspect/inventory_ops.py
"""Count button-visible frames across captures: records offering non-dapai ops,
and how many of those seqs actually have a saved frame. Decides how much
--op-delay harvest capture (Task 4) is needed.

Usage:  PYTHONPATH=. python scripts/inspect/inventory_ops.py [--sources captures/raw/ai_session]
"""
from __future__ import annotations

import argparse
import os

from majsoul_eye import paths
from majsoul_eye.capture.gtframes import load_frames
from majsoul_eye.capture.schema import read_records
from majsoul_eye.hud import buttons_for_ops
from majsoul_eye.state.ops import ops_from_record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=[paths.RAW_AI_SESSION])
    args = ap.parse_args()

    tot_off = tot_btn = tot_btn_framed = 0
    per_btn: dict[str, int] = {}
    for root in args.sources:
        for cap in paths._ai_captures_in(root):
            frames = {}
            fdir = paths.frames_dir_for(cap)
            if os.path.exists(os.path.join(fdir, "frames.jsonl")):
                frames = load_frames(fdir, statuses=("ok", "timeout"))
            n_off = n_btn = n_btn_framed = 0
            for r in read_records(cap):
                ops = ops_from_record(r)
                btns = buttons_for_ops(ops or [])
                if ops:
                    n_off += 1
                if btns:
                    n_btn += 1
                    if r.seq in frames:
                        n_btn_framed += 1
                        for b in btns:
                            per_btn[b] = per_btn.get(b, 0) + 1
            print(f"{paths.ai_game_name(cap):24s} offers={n_off:4d} "
                  f"button-records={n_btn:3d} with-frame={n_btn_framed:3d}")
            tot_off += n_off; tot_btn += n_btn; tot_btn_framed += n_btn_framed
    print(f"\nTOTAL offers={tot_off} button-records={tot_btn} with-frame={tot_btn_framed}")
    print("per-button (framed):", dict(sorted(per_btn.items())))
    print("\nNOTE: with-frame counts frames captured at the offering seq; whether the"
          "\nbuttons are still rendered in the pixel is verified later (Task 7 count-check).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 在全部 20 局上跑**

Run: `PYTHONPATH=. $PY scripts/inspect/inventory_ops.py`
Expected: 每局一行 + TOTAL。记录 `with-frame` 总数和 per-button 分布到 task notes。

- [ ] **Step 3: 判定（写进 commit message）**

判定规则：`btn_skip` 配对的 framed 记录 < 200，或任一常见按钮
（chi/pon/riichi/ron）framed < 40 → Task 4 的采集必做（预期就是这样，
run_13/game1 全局只有 8 条 offer 记录）。

- [ ] **Step 4: Commit**

```bash
git add scripts/inspect/inventory_ops.py
git commit -m "feat(inspect): pending-ops / button-frame inventory (harvest sizing)"
```

---

### Task 4: 按钮帧采集 —— `autoplay_ai.py --op-delay` **[USER RUN 收尾]**

**Files:**
- Modify: `scripts/capture/autoplay_ai.py`（MahjongCopilot settings 覆写处，约 L270）
- Test: `tests/test_autoplay_opdelay.py`

**Interfaces:**
- Produces: CLI `--op-delay LO HI`（秒）。作用：覆写 MahjongCopilot 的
  `delay_random_lower/upper`（AI 收到 op 到点击之间的随机迟疑），让 FrameSyncer
  的 quiet（默认 0.30s）先触发、按钮还挂在屏上时抓到帧。

- [ ] **Step 1: 写失败测试**

先读 `scripts/capture/autoplay_ai.py` L230–300 与 `tests/test_autoplay_gt.py` 的
现有测试风格（怎么隔离 import / 怎么测 settings 组装），照同一风格写：

```python
# tests/test_autoplay_opdelay.py
"""--op-delay LO HI overrides MahjongCopilot's delay_random_lower/upper."""
# 具体 import 方式仿照 tests/test_autoplay_gt.py 对 autoplay_ai 纯函数的测法。
# 断言三件事：
#  1) 默认（不传 --op-delay）settings 里 delay_random_lower==0.5, upper==1.0（现值）；
#  2) --op-delay 1.5 2.5 -> lower==1.5, upper==2.5；
#  3) LO>HI 时 argparse 报错（SystemExit）。
```

实现说明：把 L270 附近写死的 `"delay_random_lower": 0.5, "delay_random_upper": 1.0`
改为取自 `args.op_delay`（默认 `(0.5, 1.0)` 保持现行为）；argparse 定义
`ap.add_argument("--op-delay", nargs=2, type=float, metavar=("LO","HI"))`，校验
`LO<=HI`。若 settings 组装内联在 `main()` 里不可测，先抽出一个纯函数
`mjc_settings(op_delay: tuple[float, float]) -> dict`（保持返回值与现内联 dict
逐键一致），测试针对该函数。

- [ ] **Step 2: 跑测试确认失败** → **Step 3: 实现** → **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. $PY tests/test_autoplay_opdelay.py && PYTHONPATH=. $PY tests/test_autoplay_gt.py`
Expected: 新旧都 OK

- [ ] **Step 5: Commit**

```bash
git add scripts/capture/autoplay_ai.py tests/test_autoplay_opdelay.py
git commit -m "feat(capture): --op-delay (hesitate on pending ops) for button-frame harvest"
```

- [ ] **Step 6 [USER RUN]: 采集 2–4 局按钮帧**

⚠️ `autoplay_ai.py` 在工作区有他人未提交改动——采集前和用户确认当前
工作区状态可跑。用户命令（PowerShell，小号）：

```powershell
$env:PYTHONPATH = "."
python scripts/capture/autoplay_ai.py --live --auto-next --op-delay 1.5 2.5
```

跑完后执行者重跑 `inventory_ops.py` 验证新 run 的 with-frame 数显著提高
（目标：累计 framed button-records ≥ 200）。不足则再采。

---

### Task 5: HUD 字段几何 —— 种子 ROI + 墨迹收紧（`annotate/hud.py` 字段部分）

**Files:**
- Modify: `majsoul_eye/coords.py`（追加 `HUD_SEEDS`）
- Create: `majsoul_eye/annotate/hud.py`
- Test: `tests/test_hud_fields.py`

**Interfaces:**
- Consumes: `coords.NormBox/px_box`、`BoardState`（`.scores/.bakaze/.kyoku/.honba/.kyotaku/.left_tile_count/.hero_seat/.oya`）、`BoardRegion.norm_to_px`。
- Produces:
  - `coords.HUD_SEEDS: dict[str, NormBox]`（9 个字段种子框，Task 6 标定回填）
  - `annotate.hud.ink_snap(img, px_box, thresh=150, pad=3, min_px=12) -> tuple[int,int,int,int] | None`
  - `annotate.hud.field_texts(state) -> dict[str, str]`（字段名→应读出的字符串）
  - `annotate.hud.hud_field_boxes(img, state, region) -> list[dict]`，每项
    `{"name", "px_box", "text", "fill", "reliable"}`（与 `hand_boxes` 同风格；
    `reliable` 只会被置 False）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_hud_fields.py
"""ink_snap tightens to bright glyphs and clamps to the seed box; field_texts
maps BoardState -> reader-target strings (incl. seat rotation of scores)."""
import numpy as np

from majsoul_eye.annotate.hud import ink_snap, field_texts
from majsoul_eye.state.replay import BoardState

# --- ink_snap: bright "digits" on dark panel ---------------------------------
img = np.zeros((100, 200, 3), np.uint8)
img[40:52, 60:120] = 230                      # glyph band
snapped = ink_snap(img, (30, 20, 180, 80), pad=3)
x0, y0, x1, y1 = snapped
assert 55 <= x0 <= 58 and 117 <= x1 <= 125    # hugs 60..120 (+pad)
assert 35 <= y0 <= 38 and 49 <= y1 <= 56
assert ink_snap(np.zeros((100, 200, 3), np.uint8), (30, 20, 180, 80)) is None  # no ink
# clamp: glyph touching the seed edge must not escape it
img2 = np.zeros((100, 200, 3), np.uint8)
img2[20:80, 30:180] = 230
sx0, sy0, sx1, sy1 = ink_snap(img2, (30, 20, 180, 80), pad=5)
assert sx0 >= 30 and sy0 >= 20 and sx1 <= 180 and sy1 <= 80

# --- field_texts --------------------------------------------------------------
s = BoardState(hero_seat=2, bakaze="E", kyoku=3, honba=1, kyotaku=2, oya=1,
               in_round=True,   # riichi/honba fields are gated on in_round
               scores=[25000, 24000, 26000, 25000], left_tile_count=64)
t = field_texts(s)
assert t["score_self"] == "26000"             # scores[hero=2]
assert t["score_right"] == "25000"            # scores[3] (下家)
assert t["score_across"] == "25000"           # scores[0] (对家)
assert t["score_left"] == "24000"             # scores[1] (上家)
assert t["round_label"] == "E3"
assert t["wall_count"] == "余64"
assert t["riichi_stick_count"] == "x2" and t["honba_count"] == "x1"
assert t["seat_wind_self"] == "S"             # (2-1)%4=1 -> S
# missing GT -> field omitted
t2 = field_texts(BoardState())
assert "wall_count" not in t2 and "score_self" not in t2 and "seat_wind_self" not in t2
print("test_hud_fields OK")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. $PY tests/test_hud_fields.py`
Expected: `ModuleNotFoundError: No module named 'majsoul_eye.annotate.hud'`

- [ ] **Step 3: `coords.py` 追加种子（文件尾）**

```python
# --- HUD field seed ROIs (px @ 1920x1080 web client) --------------------------
# ⚠️ CALIBRATE (Task 6 of the HUD plan): score_self/right/across + riichi/honba
# seeds come from mycv's get_f1/f2/f3/bangzi/benchang; score_left / wall_count /
# round_label / seat_wind_self are eyeballed from run_1 frames. ink_snap tightens
# numeric fields per frame, so seeds only need to CONTAIN the glyphs w/ margin.
_HUD_SEEDS_PX: dict[str, tuple[int, int, int, int]] = {
    "score_self":         (900, 460, 1020, 500),   # CALIBRATE
    "score_right":        (1040, 330, 1085, 460),  # CALIBRATE (vertical digits)
    "score_across":       (900, 295, 1020, 335),   # CALIBRATE
    "score_left":         (835, 330, 880, 460),    # CALIBRATE (vertical digits)
    "round_label":        (905, 350, 1015, 385),   # CALIBRATE  東N局
    "wall_count":         (925, 385, 995, 415),    # CALIBRATE  余NN
    "seat_wind_self":     (855, 455, 900, 500),    # CALIBRATE  corner wind tag
    "riichi_stick_count": (95, 135, 175, 185),     # CALIBRATE  mycv get_bangzi
    "honba_count":        (235, 135, 315, 185),    # CALIBRATE  mycv get_benchang
}
HUD_SEEDS: dict[str, NormBox] = {k: px_box(*v) for k, v in _HUD_SEEDS_PX.items()}
```

- [ ] **Step 4: 实现 `annotate/hud.py`**

```python
# majsoul_eye/annotate/hud.py
"""GT-driven HUD field annotation: seed ROI (WHERE) + BoardState (WHAT).

Numeric fields are ink-snapped per frame (glyph width varies with the value);
fixed-glyph fields (round_label / seat_wind_self) keep the seed box. Buttons are
handled separately (button_boxes, Task 7). Output dict style matches
annotate.frame's hand_boxes: `reliable` is only ever SET False.
"""
from __future__ import annotations

import cv2
import numpy as np

from majsoul_eye.coords import HUD_SEEDS
from majsoul_eye.hud import NUMERIC_FIELDS

INK_THRESH = 150   # gray level splitting glyph from dark panel  # CALIBRATE
INK_MIN_PX = 12    # fewer bright px than this = field not rendered
INK_PAD = 3        # px of context kept around the glyph extent


def ink_snap(img: np.ndarray, px_box, thresh: int = INK_THRESH,
             pad: int = INK_PAD, min_px: int = INK_MIN_PX):
    """Tighten px_box to the bright-glyph extent inside it (clamped to px_box).
    Returns (x0,y0,x1,y1) or None when the field shows no ink (not rendered)."""
    x0, y0, x1, y1 = (int(v) for v in px_box)
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(g >= thresh)
    if len(xs) < min_px:
        return None
    return (max(x0, x0 + int(xs.min()) - pad), max(y0, y0 + int(ys.min()) - pad),
            min(x1, x0 + int(xs.max()) + 1 + pad), min(y1, y0 + int(ys.max()) + 1 + pad))


def field_texts(state) -> dict[str, str]:
    """BoardState -> {field name: exact string the reader must output}.
    Fields whose GT is missing are OMITTED (never guessed)."""
    t: dict[str, str] = {}
    hero = state.hero_seat
    if hero >= 0 and state.scores:
        for i, name in enumerate(("score_self", "score_right",
                                  "score_across", "score_left")):
            t[name] = str(state.scores[(hero + i) % 4])
    if state.bakaze and state.kyoku:
        t["round_label"] = f"{state.bakaze}{state.kyoku}"
    if state.left_tile_count is not None:
        t["wall_count"] = f"余{state.left_tile_count}"
    if state.in_round:
        t["riichi_stick_count"] = f"x{state.kyotaku}"
        t["honba_count"] = f"x{state.honba}"
    if hero >= 0 and state.oya >= 0:
        t["seat_wind_self"] = "ESWN"[(hero - state.oya) % 4]
    return t


def hud_field_boxes(img: np.ndarray, state, region) -> list[dict]:
    """Annotate every GT-known HUD field on one frame. Numeric fields are
    ink-snapped; a field with no ink is emitted unreliable (GT leads render /
    occluded), same policy as tile zones."""
    out: list[dict] = []
    for name, text in field_texts(state).items():
        seed = region.norm_to_px(HUD_SEEDS[name])
        box, fill = seed, 1.0
        if name in NUMERIC_FIELDS:
            snapped = ink_snap(img, seed)
            if snapped is None:
                out.append({"name": name, "px_box": list(seed), "text": text,
                            "fill": 0.0, "reliable": False})
                continue
            box = snapped
        d = {"name": name, "px_box": [int(v) for v in box], "text": text,
             "fill": fill}
        out.append(d)
    return out
```

- [ ] **Step 5: 跑测试确认通过**

Run: `PYTHONPATH=. $PY tests/test_hud_fields.py`
Expected: `test_hud_fields OK`

- [ ] **Step 6: Commit**

```bash
git add majsoul_eye/coords.py majsoul_eye/annotate/hud.py tests/test_hud_fields.py
git commit -m "feat(annotate): HUD field boxes — seed ROIs + per-frame ink snap + GT texts"
```

---

### Task 6: 字段标定 —— overlay QA 回填常量（人眼检查点）

**Files:**
- Create: `scripts/inspect/overlay_hud.py`
- Modify: `majsoul_eye/coords.py`（回填 `_HUD_SEEDS_PX` 实测值）、`majsoul_eye/hud.py`（回填 `FIELD_ROT` 实测方向）、`majsoul_eye/annotate/hud.py`（必要时调 `INK_THRESH`）

**Interfaces:**
- Consumes: `capture.gtframes.load_pair`、`hud_field_boxes`、`ink_snap`。
- Produces: 标定完成的常量（后续 task 直接信任）。

- [ ] **Step 1: 实现 overlay 脚本**

```python
# scripts/inspect/overlay_hud.py
"""Draw HUD seed ROIs (thin) + ink-snapped boxes (thick, with GT text) on real
frames for visual calibration. Writes <out>/hud_<game>_<seq>.png.

Usage: PYTHONPATH=. python scripts/inspect/overlay_hud.py \
           captures/raw/ai_session/run_1/game1.jsonl --seqs 28 120 400 --out scratchpad/hudcal
"""
from __future__ import annotations

import argparse
import os

import cv2

from majsoul_eye import paths
from majsoul_eye.annotate.hud import hud_field_boxes
from majsoul_eye.capture.gtframes import load_pair
from majsoul_eye.coords import HUD_SEEDS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--seqs", nargs="+", type=int, required=True)
    ap.add_argument("--out", default="scratchpad/hudcal")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    game = paths.ai_game_name(args.capture)
    for seq in args.seqs:
        frame, state, region = load_pair(args.capture, seq)
        for name, nb in HUD_SEEDS.items():                    # seeds: thin yellow
            x0, y0, x1, y1 = region.norm_to_px(nb)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 1)
        for d in hud_field_boxes(frame, state, region):       # snapped: thick
            x0, y0, x1, y1 = d["px_box"]
            color = (0, 255, 0) if d.get("reliable", True) else (0, 0, 255)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(frame, f"{d['name']}={d['text']}", (x0, max(12, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        p = os.path.join(args.out, f"hud_{game}_{seq:06d}.png")
        cv2.imwrite(p, frame)
        print("->", p)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 选帧生成 overlay**

选局面丰富的帧（有立直棒/本场>0 的中盘局面更好；用 `inventory_ops.py` 输出或
`seq_state` 里 `kyotaku>0` 的 seq）。至少覆盖 3 个不同 game、每 game 3 帧：

Run: `PYTHONPATH=. $PY scripts/inspect/overlay_hud.py captures/raw/ai_session/run_1/game1.jsonl --seqs 28 200 500 --out scratchpad/hudcal`（另两局同理）
Expected: 输出 png；执行者用 Read 工具逐张查看。

- [ ] **Step 3: 迭代回填常量，直到全部紧贴**

检查项（逐张人眼过）：
1. 9 个种子框都**包含**目标字形且不含邻近字形 → 否则改 `_HUD_SEEDS_PX`；
2. 数字字段的收紧框紧贴字串（≤3px 松弛）→ 否则调 `INK_THRESH`/`INK_PAD`；
3. 确认 `score_left/right` 的数字旋转方向 → 回填 `hud.FIELD_ROT`（把种子猜测
   改成实测值，并把注释里 "seed guess" 字样删掉）；
4. 确认左上两个图标各自对应 kyotaku/honba（找一帧 `kyotaku>0` 对照 GT 值）。

每次改完重跑 Step 2 直到通过。**此为人眼检查点：把最终 overlay 图路径贴给用户
过目后再 commit。**

- [ ] **Step 4: 回归单元测试**

Run: `PYTHONPATH=. $PY tests/test_hud_fields.py && PYTHONPATH=. $PY tests/test_hud_taxonomy.py`
Expected: OK（若改了 INK 常量导致合成测试阈值不符，同步修测试的合成亮度）

- [ ] **Step 5: Commit**

```bash
git add scripts/inspect/overlay_hud.py majsoul_eye/coords.py majsoul_eye/hud.py majsoul_eye/annotate/hud.py
git commit -m "feat(inspect): HUD overlay QA tool; calibrate HUD seeds/rotations on real frames"
```

---

### Task 7: 按钮定位器 + GT 赋类（`annotate/hud.py` 按钮部分）

依赖：Task 4 的采集数据已就位（标定需要真实按钮帧）。

**Files:**
- Modify: `majsoul_eye/annotate/hud.py`（追加 `BTN_ZONE`/`locate_button_candidates`/`button_boxes`）、`majsoul_eye/coords.py`（追加 `BTN_ZONE` 种子）
- Test: `tests/test_hud_buttons.py`

**Interfaces:**
- Consumes: `hud.buttons_for_ops`、`state.pending_ops`。
- Produces: `button_boxes(img, state, region) -> list[dict]`，每项
  `{"name": "btn_*", "px_box", "reliable"}`；候选数≠期望数时返回全部
  `reliable=False` + `"flag": "count_mismatch"`（build 侧因此不落任何按钮标签）。

- [ ] **Step 1: 写失败测试（合成按钮条）**

```python
# tests/test_hud_buttons.py
"""Button locator: bright banners on dark strip -> x-sorted candidates; GT op
set assigns classes; candidate/expected count mismatch -> all unreliable."""
import numpy as np

from majsoul_eye.annotate.hud import locate_button_candidates, button_boxes
from majsoul_eye.coords import BTN_ZONE
from majsoul_eye.normalize import locate_fullscreen
from majsoul_eye.state.replay import BoardState

W, H = 1920, 1080
img = np.zeros((H, W, 3), np.uint8)
# BTN_ZONE 内画两个高亮"按钮"（左：碰位，右：跳过位）
zx0, zy0, zx1, zy1 = (int(v) for v in (BTN_ZONE.x0 * W, BTN_ZONE.y0 * H,
                                       BTN_ZONE.x1 * W, BTN_ZONE.y1 * H))
cy = (zy0 + zy1) // 2
img[cy - 25:cy + 25, zx0 + 100:zx0 + 260] = 220
img[cy - 25:cy + 25, zx0 + 400:zx0 + 560] = 220
region = locate_fullscreen(img)

cands = locate_button_candidates(img, region)
assert len(cands) == 2 and cands[0][0] < cands[1][0]          # x-sorted

s = BoardState(hero_seat=0, pending_ops=[1, 3])               # pon offer
bb = button_boxes(img, s, region)
assert [b["name"] for b in bb] == ["btn_pon", "btn_skip"]     # left->right order rule
assert all(b.get("reliable", True) for b in bb)

s2 = BoardState(hero_seat=0, pending_ops=[1, 3, 9])           # expects 3, sees 2
bb2 = button_boxes(img, s2, region)
assert bb2 and all(b["reliable"] is False for b in bb2)
assert all(b.get("flag") == "count_mismatch" for b in bb2)

assert button_boxes(img, BoardState(pending_ops=[1]), region) == []   # dapai only
print("test_hud_buttons OK")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. $PY tests/test_hud_buttons.py`
Expected: `ImportError: cannot import name 'locate_button_candidates'`

- [ ] **Step 3: 实现**

`coords.py` 追加（`HUD_SEEDS` 旁）：

```python
# Action-button strip (bottom, above the hand). Wide on purpose — the locator
# contours inside it; only containment matters.  # CALIBRATE (Task 7 Step 5)
BTN_ZONE = NormBox(0.30, 0.66, 0.98, 0.82)
```

`annotate/hud.py` 追加：

```python
from majsoul_eye.coords import BTN_ZONE
from majsoul_eye.hud import buttons_for_ops

BTN_MIN_AREA = 2500    # px² @1080p; banners are ~160x50   # CALIBRATE
BTN_THRESH = 140       # banner glow vs table              # CALIBRATE
BTN_ORDER_LTR = True   # display order left->right == buttons_for_ops order
                       # (empirical; flip after eyeballing harvest frames)


def locate_button_candidates(img, region) -> list[tuple[int, int, int, int]]:
    """Bright banner blobs inside BTN_ZONE, x-sorted, as original-px boxes."""
    x0, y0, x1, y1 = region.norm_to_px(BTN_ZONE)
    roi = img[y0:y1, x0:x1]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    m = (g >= BTN_THRESH).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 25), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= BTN_MIN_AREA and w > h:            # wide banner shape
            out.append((x0 + x, y0 + y, x0 + x + w, y0 + y + h))
    return sorted(out)


def button_boxes(img, state, region) -> list[dict]:
    """GT-expected buttons matched to located candidates by order.
    Count mismatch -> every box unreliable + flagged (frame contributes no
    button labels; 宁缺毋滥)."""
    expected = buttons_for_ops(state.pending_ops or [])
    if not expected:
        return []
    cands = locate_button_candidates(img, region)
    ordered = expected if BTN_ORDER_LTR else expected[::-1]
    if len(cands) != len(expected):
        return [{"name": n, "px_box": list(c) if i < len(cands) else None,
                 "reliable": False, "flag": "count_mismatch"}
                for i, (n, c) in enumerate(
                    zip(ordered, list(cands) + [None] * len(expected)))
                if n][:len(expected)]
    return [{"name": n, "px_box": list(c)} for n, c in zip(ordered, cands)]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. $PY tests/test_hud_buttons.py`
Expected: `test_hud_buttons OK`

- [ ] **Step 5: 在采集的真实按钮帧上标定（人眼检查点）**

给 `overlay_hud.py` 加 `--buttons` 开关（在同一张 overlay 上叠画
`button_boxes` 结果，count_mismatch 的画红）。对 Task 4 新采 run 里
`inventory_ops.py` 报出的 button seq 各类抽 ≥5 帧：

1. `BTN_ZONE` 包含全部按钮 → 否则改；
2. 候选框贴合按钮 banner → 否则调 `BTN_THRESH/BTN_MIN_AREA`；
3. **验证顺序规则**（`BTN_ORDER_LTR`）：多按钮帧里 GT 顺序与屏幕左右序对照；
4. **验证 `btn_skip` 出现规则**：找"自摸/立直 only"的自回合帧，确认有无跳过
   按钮；若无 → 改 `hud.buttons_for_ops`（own-turn-only 不加 skip）并同步
   `tests/test_hud_taxonomy.py`；
5. 统计 count_mismatch 比例，>30% 说明外观规则错了，回到 2。

overlay 图贴给用户过目后 commit。

- [ ] **Step 6: Commit**

```bash
git add majsoul_eye/annotate/hud.py majsoul_eye/coords.py scripts/inspect/overlay_hud.py tests/test_hud_buttons.py
git commit -m "feat(annotate): button locator + op-GT class assignment (count-mismatch drop)"
```

---

### Task 8: 接入 `annotate_frame` + 丢帧谓词

**Files:**
- Modify: `majsoul_eye/annotate/frame.py`（`annotate_frame` 尾部 + `HudBox`/`iter_hud_boxes`）、`majsoul_eye/state/replay.py`（`is_score_anim_window`）
- Test: `tests/test_hud_frame.py`

**Interfaces:**
- Produces:
  - `annotate_frame` 记录新键 `rec["hud_boxes"]`（字段+按钮混合 list）
  - `HudBox(name: str, px_box: list, text: str | None, reliable: bool)` +
    `iter_hud_boxes(rec) -> Iterator[HudBox]`（按钮 `text=None`）
  - `state.replay.is_score_anim_window(state) -> bool`：`last_event` 在
    `{"reach", "reach_accepted"}` → True（立直宣言/棒动画+分数滚动窗口；
    分数动画帧按用户决定出范围）。build 侧只在该窗口跳过 **HUD 标签**，
    牌面标签照常（牌不受分数动画影响）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_hud_frame.py
"""annotate_frame emits hud_boxes; iter_hud_boxes flattens them; score-anim
window predicate flags reach frames."""
import numpy as np

from majsoul_eye.annotate import build_homographies, annotate_frame
from majsoul_eye.annotate.frame import iter_hud_boxes
from majsoul_eye.state.replay import BoardState, is_score_anim_window

s = BoardState(hero_seat=0, bakaze="E", kyoku=1, oya=0, in_round=True,
               scores=[25000] * 4, left_tile_count=64)
img = np.zeros((1080, 1920, 3), np.uint8)
hom = build_homographies(1920, 1080)
rec = annotate_frame(img, s, hom)
assert "hud_boxes" in rec
hb = list(iter_hud_boxes(rec))
names = {b.name for b in hb}
assert "round_label" in names and "seat_wind_self" in names
# black frame -> numeric fields have no ink -> unreliable, never wrong-text
for b in hb:
    if b.name == "score_self":
        assert b.reliable is False
    if b.name == "round_label":
        assert b.text == "E1"

assert is_score_anim_window(BoardState(last_event="reach_accepted"))
assert is_score_anim_window(BoardState(last_event="reach"))
assert not is_score_anim_window(BoardState(last_event="dahai"))
print("test_hud_frame OK")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. $PY tests/test_hud_frame.py`
Expected: KeyError/`hud_boxes` 断言失败

- [ ] **Step 3: 实现**

`replay.py`（`is_deal_window` 旁）：

```python
def is_score_anim_window(state) -> bool:
    """Riichi declaration/stick animation + score-roll window: HUD numeric
    fields on screen lag/animate right after a reach event, so HUD labels from
    these frames are unreliable (tile labels are unaffected). Spec'd out of
    recognition scope by the user."""
    return getattr(state, "last_event", None) in ("reach", "reach_accepted")
```

`frame.py`：`annotate_frame` 在 dora 块之后加：

```python
    # HUD fields + action buttons (GT text/ops drive labels; see annotate/hud.py)
    try:
        from majsoul_eye.annotate import hud as HUD
        from majsoul_eye.state.replay import is_score_anim_window
        region = locate_fullscreen(img)
        boxes = HUD.hud_field_boxes(img, state, region)
        if is_score_anim_window(state):
            for b in boxes:
                b["reliable"] = False
            rec["flags"].append("hud:score_anim")
        rec["hud_boxes"] = boxes + HUD.button_boxes(img, state, region)
    except Exception as e:                       # HUD is best-effort like dora
        rec["flags"].append(f"hud:error:{e}")
        rec["hud_boxes"] = []
```

`frame.py` 尾部加：

```python
@dataclass
class HudBox:
    """One HUD box from an annotate_frame record. `text` is the exact string a
    micro-reader must output (None for buttons — class IS the label)."""
    name: str
    px_box: list
    text: Optional[str]
    reliable: bool


def iter_hud_boxes(rec: dict) -> Iterator[HudBox]:
    for d in rec.get("hud_boxes", []):
        if d.get("px_box") is None:
            continue
        yield HudBox(d["name"], list(d["px_box"]), d.get("text"),
                     bool(d.get("reliable", True)))
```

- [ ] **Step 4: 跑测试确认通过 + 回归**

Run: `PYTHONPATH=. $PY tests/test_hud_frame.py && PYTHONPATH=. $PY tests/test_label.py && PYTHONPATH=. $PY tests/test_replay.py`
Expected: 全 OK

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/annotate/frame.py majsoul_eye/state/replay.py tests/test_hud_frame.py
git commit -m "feat(annotate): hud_boxes in annotate_frame; score-anim drop predicate"
```

---

### Task 9: 数据集产出 —— YOLO 55 类 + 读取器训练对

**Files:**
- Modify: `scripts/train/build_dataset.py`（HUD YOLO 行 + `hud/` crops）、`scripts/train/build_detector_dataset.py`（data.yaml 用 `DET_NAMES`）
- Test: `tests/test_hud_dataset.py`

**Interfaces:**
- Produces:
  - `<out>/yolo/labels/*.txt` 含 38–54 类行（HBB/OBB 与牌面同路径）
  - `<out>/hud/<field>/<seq>.png`（**含 15% pad** 的读取器 crop，按 `FIELD_ROT`
    已转正）+ `<out>/hud/labels.jsonl`：`{"file","name","text","pad":0.15}`
  - `detector/data.yaml`：`nc: 55`、names=`hud.DET_NAMES`（v1 旧标签只用 0–37，
    与 55 类头兼容，可混训）

- [ ] **Step 1: 写失败测试（针对可单测的纯函数）**

把 HUD 落盘逻辑抽成 `build_dataset.py` 里的纯函数 `hud_emit(rec, frame, w, h, obb)`
→ 返回 `(yolo_lines: list[str], crops: list[tuple[str, np.ndarray, dict]])`
（crop 三元组 = 相对路径、图、labels.jsonl 行），主循环只做 IO。测试：

```python
# tests/test_hud_dataset.py
"""hud_emit: reliable HUD boxes -> 55-class YOLO lines + rotated padded reader
crops; unreliable/no-text boxes emit no crop; buttons emit label only."""
import numpy as np

import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "build_dataset", pathlib.Path("scripts/train/build_dataset.py"))
bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

from majsoul_eye.hud import HUD_NAME_TO_ID

frame = np.full((1080, 1920, 3), 40, np.uint8)
rec = {"hud_boxes": [
    {"name": "score_self", "px_box": [900, 460, 1000, 500], "text": "25000"},
    {"name": "score_across", "px_box": [900, 300, 1000, 335], "text": "24000"},
    {"name": "btn_pon", "px_box": [1200, 740, 1360, 790]},
    {"name": "wall_count", "px_box": [925, 385, 995, 415], "text": "余64",
     "reliable": False},
]}
lines, crops = bd.hud_emit(rec, frame, 1920, 1080, obb=False)
assert len(lines) == 3                                   # unreliable dropped
assert lines[0].startswith(f"{HUD_NAME_TO_ID['score_self']} ")
assert any(l.startswith(f"{HUD_NAME_TO_ID['btn_pon']} ") for l in lines)
assert len(crops) == 2                                   # buttons: no crop
relpath, img, meta = crops[0]
assert meta == {"file": relpath, "name": "score_self", "text": "25000", "pad": 0.15}
assert relpath.startswith("score_self/")
# 180° field comes out rotated-to-upright: crop of across (35px tall box +pad)
_, img2, meta2 = crops[1]
assert meta2["name"] == "score_across" and img2.shape[0] > 0
print("test_hud_dataset OK")
```

- [ ] **Step 2: 跑测试确认失败** （`AttributeError: ... 'hud_emit'`）

- [ ] **Step 3: 实现 `hud_emit` + 主循环接线**

`build_dataset.py` 顶部函数区加：

```python
def hud_emit(rec, frame, w, h, obb):
    """Reliable hud_boxes -> (yolo label lines, reader crops). Crops are padded
    15% per side (jitter headroom for the trainer) and rotated upright per
    hud.FIELD_ROT; buttons contribute a label line only (class IS the label)."""
    import cv2

    from majsoul_eye.hud import HUD_NAME_TO_ID, FIELD_ROT

    ROT_CODE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    PAD = 0.15
    lines, crops = [], []
    for d in rec.get("hud_boxes", []):
        if not d.get("reliable", True) or d.get("px_box") is None:
            continue
        cls = HUD_NAME_TO_ID.get(d["name"])
        if cls is None:
            continue
        x0, y0, x1, y1 = d["px_box"]
        quad = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        lines.append(obb_label_line(cls, quad, w, h) if obb
                     else hbb_label_line(cls, quad, w, h))
        text = d.get("text")
        if text is None:
            continue
        px, py = int((x1 - x0) * PAD), int((y1 - y0) * PAD)
        cy0, cy1 = max(0, y0 - py), min(h, y1 + py)
        cx0, cx1 = max(0, x0 - px), min(w, x1 + px)
        crop = frame[cy0:cy1, cx0:cx1]
        rot = FIELD_ROT.get(d["name"], 0)
        if rot in ROT_CODE:
            crop = cv2.rotate(crop, ROT_CODE[rot])
        rel = f"{d['name']}/{rec.get('_seq', 0):06d}.png"
        crops.append((rel, crop,
                      {"file": rel, "name": d["name"], "text": text, "pad": PAD}))
    return lines, crops
```

主循环接线（`yolo_lines` 聚合之后、写盘之前）：

```python
        rec["_seq"] = seq
        hud_skip = is_score_anim_window(state)   # belt & suspenders with Task 8
        if not hud_skip:
            hlines, hcrops = hud_emit(rec, frame, w, h, args.obb)
            if not args.no_yolo:
                yolo_lines += hlines
            if not args.no_crops:
                for rel, crop, meta in hcrops:
                    p = os.path.join(args.out, "hud", rel)
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    cv2.imwrite(p, crop)
                    hud_meta.append(meta)        # hud_meta = [] before the loop
```

循环后写 `hud/labels.jsonl`（`hud_meta` 逐行 json）。import 区补
`from majsoul_eye.state.replay import is_score_anim_window`。统计行加
`hud-crops: {len(hud_meta)}`。

`build_detector_dataset.py`：`build_data_yaml_text` 把 `TILE_NAMES` 换成
`from majsoul_eye.hud import DET_NAMES`，`nc: {len(DET_NAMES)}`，names 逐行
写 55 个；docstring 的 "frozen 38 class names" 改成 "55 = frozen 38 tiles +
17 HUD (majsoul_eye.hud.DET_NAMES)"。

**--from-annotations 注意**：hud_boxes 已存在于新生成的 annotations 记录里
（Task 8 之后 annotate_ai_session 自动透传），reuse 路径无需重算——但**旧
annotations（v1 的）没有 hud_boxes**，`rec.get("hud_boxes", [])` 自然空，
不报错、只是无 HUD 标签。v2 全量重 annotate 解决。

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

Run: `PYTHONPATH=. $PY tests/test_hud_dataset.py && for t in tests/test_*.py; do PYTHONPATH=. $PY "$t" || break; done`
Expected: 全 OK

- [ ] **Step 5: Commit**

```bash
git add scripts/train/build_dataset.py scripts/train/build_detector_dataset.py tests/test_hud_dataset.py
git commit -m "feat(data): emit 55-class YOLO labels + HUD reader crops (hud/ + labels.jsonl)"
```

---

### Task 10: 构建 `datasets/v2` + 抽样 QA

**Files:** 无代码改动（跑管线 + 看产物）

- [ ] **Step 1: 全量构建（含 Task 4 新采的按钮 run）**

Run: `PYTHONPATH=. $PY scripts/data/build_datasets.py v2 --dry-run` 先看命令，
然后去掉 `--dry-run` 实跑（几十分钟量级；`--workers/--jobs` 按机器）。
Expected: `datasets/v2/` 生成，stage 输出无失败；`detector/data.yaml` `nc: 55`。

- [ ] **Step 2: 抽样 QA**

```bash
# YOLO 标签里出现过 38-54 类：
grep -rhoE "^(3[89]|4[0-9]|5[0-4]) " datasets/v2/*/yolo/labels/*.txt | sort | uniq -c
# 读取器 crop 数量与分布：
wc -l datasets/v2/*/hud/labels.jsonl
```
Expected: 9 个字段类计数≈帧数×字段数；按钮类计数≈盘点数；`hud/` 每字段目录
有图。再用 Read 抽看 10 张 `hud/score_*/...png`（应为转正后的数字条）。
异常（某类计数为 0 / crop 歪的）→ 回对应 task 修。

- [ ] **Step 3: Commit（如有修复）**；数据集本身不进 git（`datasets/` 已 ignore）。

---

### Task 11: 微读取器 —— 模型 + 训练脚本

**Files:**
- Create: `majsoul_eye/recognize/hudreader.py`（模型+解码+推理包装，capture-free）
- Create: `scripts/train/train_hudreader.py`
- Test: `tests/test_hudreader.py`

**Interfaces:**
- Produces:
  - `DigitCTC(nn.Module)`：输入 `B×1×32×W` 灰度，输出 `B×(W//4)×(len(CTC_CHARSET)+1)` log-probs
  - `ctc_decode(logits_TxC) -> str`（贪心：折叠重复、去 blank(0)）
  - `HudReader`：`__init__(path="majsoul_eye/recognize/hud_reader.pt")`，
    `read(bgr_crop: np.ndarray, cls_name: str) -> str`（内部按字段路由
    CTC / round16 / wind4；round 返回 "E1".."N4"，wind 返回 "E/S/W/N"）
  - checkpoint 格式：`{"ctc": sd, "round": sd, "wind": sd, "charset": CTC_CHARSET, "meta": {...}}`
- Consumes: `hud.CTC_CHARSET/ROUND_CLASSES/WIND_CLASSES`、`recognize.classifier.TileNet(n_classes)`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_hudreader.py
"""DigitCTC shapes; greedy CTC decode (repeat-collapse + blank-drop); charset
round-trip for every char the pipeline can emit."""
import numpy as np
import torch

from majsoul_eye.hud import CTC_CHARSET
from majsoul_eye.recognize.hudreader import DigitCTC, ctc_decode, encode_text

m = DigitCTC()
x = torch.zeros(2, 1, 32, 128)
y = m(x)
assert y.shape == (2, 32, len(CTC_CHARSET) + 1)      # T=W/4, C=13+blank

# decode: blank=0; "2 2 blank 5 5 5 blank blank 0(char '0'=idx1)" -> "250"
idx = {c: i + 1 for i, c in enumerate(CTC_CHARSET)}
seq = [idx["2"], idx["2"], 0, idx["5"], idx["5"], idx["5"], 0, 0, idx["0"]]
logits = torch.full((len(seq), len(CTC_CHARSET) + 1), -10.0)
for t, i in enumerate(seq):
    logits[t, i] = 0.0
assert ctc_decode(logits) == "250"
# encode/decode round-trip incl. 余 and x and -
for s in ("25000", "余64", "x2", "-1200"):
    enc = encode_text(s)
    assert all(1 <= i <= len(CTC_CHARSET) for i in enc)
print("test_hudreader OK")
```

- [ ] **Step 2: 跑测试确认失败** → **Step 3: 实现 `recognize/hudreader.py`**

```python
# majsoul_eye/recognize/hudreader.py
"""HUD micro-readers (shipped product; capture-free).

DigitCTC — segmentation-free CRNN-CTC over a 32px-high strip (charset
hud.CTC_CHARSET; index+1, 0=blank). Round/wind heads reuse TileNet(n_classes).
HudReader wraps all three behind one read(crop, cls_name) call; rotation to
upright happens UPSTREAM (dataset crops are saved rotated; runtime rotates by
hud.FIELD_ROT before calling read)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from majsoul_eye.hud import (CTC_CHARSET, NUMERIC_FIELDS, ROUND_CLASSES,
                             WIND_CLASSES)
from majsoul_eye.recognize.classifier import TileNet

N_CTC = len(CTC_CHARSET) + 1          # +blank at index 0


def encode_text(s: str) -> list[int]:
    return [CTC_CHARSET.index(c) + 1 for c in s]


class DigitCTC(nn.Module):
    """1x32xW -> (W/4) x N_CTC log-probs. Pools H 32->2 then collapses."""

    def __init__(self, n_out: int = N_CTC):
        super().__init__()

        def block(i, o, pool):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(pool))

        self.features = nn.Sequential(
            block(1, 32, (2, 2)), block(32, 64, (2, 2)),   # H 32->8, W /4
            block(64, 128, (2, 1)), block(128, 128, (2, 1)))  # H 8->2, W keeps /4
        self.head = nn.Linear(128 * 2, n_out)

    def forward(self, x):                       # B,1,32,W
        f = self.features(x)                    # B,128,2,W/4
        f = f.permute(0, 3, 1, 2).flatten(2)    # B,T,256
        return self.head(f).log_softmax(-1)     # B,T,N_CTC


def ctc_decode(logits: torch.Tensor) -> str:
    """Greedy best-path: argmax per step, collapse repeats, drop blank(0)."""
    out, prev = [], -1
    for i in logits.argmax(-1).tolist():
        if i != prev and i != 0:
            out.append(CTC_CHARSET[i - 1])
        prev = i
    return "".join(out)


def _strip(bgr: np.ndarray) -> torch.Tensor:
    """BGR crop -> 1x1x32xW normalized gray strip (W scaled with aspect, min 32)."""
    import cv2
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    nw = max(32, round(w * 32 / h))
    g = cv2.resize(g, (nw, 32)).astype(np.float32) / 255.0
    return torch.from_numpy(g)[None, None]


class HudReader:
    def __init__(self, path: str | None = None, device: str = "cpu"):
        import os
        path = path or os.path.join(os.path.dirname(__file__), "hud_reader.pt")
        ck = torch.load(path, map_location=device, weights_only=True)
        assert ck["charset"] == CTC_CHARSET, "charset drift vs weights"
        self.device = device
        self.ctc = DigitCTC().to(device).eval()
        self.ctc.load_state_dict(ck["ctc"])
        self.round = TileNet(n_classes=len(ROUND_CLASSES)).to(device).eval()
        self.round.load_state_dict(ck["round"])
        self.wind = TileNet(n_classes=len(WIND_CLASSES)).to(device).eval()
        self.wind.load_state_dict(ck["wind"])

    @torch.no_grad()
    def read(self, bgr_crop: np.ndarray, cls_name: str) -> str:
        if cls_name in NUMERIC_FIELDS:
            return ctc_decode(self.ctc(_strip(bgr_crop).to(self.device))[0])
        import cv2
        x = cv2.resize(bgr_crop, (64, 64)).astype(np.float32) / 255.0
        t = torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device)
        if cls_name == "round_label":
            return ROUND_CLASSES[int(self.round(t).argmax())]
        if cls_name == "seat_wind_self":
            return WIND_CLASSES[int(self.wind(t).argmax())]
        raise ValueError(f"not a readable field: {cls_name}")
```

（`TileNet` 的 preprocess 若在 classifier.py 里有现成函数，round/wind 分支改用
它保持一致——实现时看一眼 `recognize/classifier.py` 的推理预处理并对齐。）

- [ ] **Step 4: 跑测试确认通过** → **Step 5: Commit**

```bash
git add majsoul_eye/recognize/hudreader.py tests/test_hudreader.py
git commit -m "feat(recognize): DigitCTC + round/wind heads + HudReader wrapper"
```

- [ ] **Step 6: 训练脚本 `scripts/train/train_hudreader.py`**

要点（照 `train_classifier.py` 的 `--dataset` 展开惯例）：
- `--dataset datasets/v2`（可重复）→ 读每版 `games.json`，收集各 game
  `hud/labels.jsonl`；val = manifest 的 `val` game（`ai_run_8_game1`）。
- 三个子训练：
  1. CTC：样本=(`hud/<field>/<seq>.png` 灰度 32 高, `encode_text(text)`)，仅
     `NUMERIC_FIELDS`；增广=从 15% pad 里随机重裁 ±8% + 亮度 ±15%；batch 内
     W pad 到最大，`input_lengths=W//4`，`nn.CTCLoss(blank=0, zero_infinity=True)`，
     Adam 1e-3，~20 epoch；按**字段类型**均衡采样（wall_count 天然数字均匀，
     score 偏 0/2/5）。
  2. round：`round_label` crop → 16 类 CE（TileNet）。
  3. wind：`seat_wind_self` crop → 4 类 CE。
- 输出 `--out majsoul_eye/recognize/hud_reader.pt`（Task 11 checkpoint 格式），
  打印三个 held-out 指标：CTC **串精确匹配率**、round/wind top-1。
- `.gitignore`：`hud_reader.pt` 按 `tile_classifier.pt` 同策略加白名单（几 MB）。

- [ ] **Step 7 [USER RUN]: 训练 + 验收**

```powershell
$env:PYTHONPATH = "."
python scripts/train/train_hudreader.py --dataset datasets/v2 --out majsoul_eye/recognize/hud_reader.pt
```
Expected: held-out 串精确匹配 ≥99.5%（spec §6）；round/wind top-1 ≥99.5%。
未达标先查 `hud/` crop 质量（回 Task 6 标定）再调训练。

- [ ] **Step 8: Commit**

```bash
git add scripts/train/train_hudreader.py .gitignore majsoul_eye/recognize/hud_reader.pt
git commit -m "feat(train): HUD reader training (CTC + round/wind); ship hud_reader.pt"
```

---

### Task 12: 检测器 v2 重训 + 牌面回归门槛 **[USER RUN]**

**Files:**
- Create: `scripts/inspect/eval_detector_split.py`（分组 mAP 报告）

- [ ] **Step 1: 实现分组评估脚本**

```python
# scripts/inspect/eval_detector_split.py
"""Per-group mAP report for a detector checkpoint: tiles (0-37) vs HUD (38-54).
Gate: tile-group mAP50 >= 0.988 (0.993 baseline - 0.005 eps, spec §6).

Usage: PYTHONPATH=. python scripts/inspect/eval_detector_split.py \
           runs/detect/train/weights/best.pt datasets/v2/detector/data.yaml
"""
import sys

from ultralytics import YOLO

weights, data = sys.argv[1], sys.argv[2]
m = YOLO(weights)
r = m.val(data=data, imgsz=1280, plots=False)
names = r.names                       # {id: name}
ap50 = r.box.ap50                     # per-class AP50, aligned to r.box.ap_class_index
idx = list(r.box.ap_class_index)
tile = [ap50[i] for i, c in enumerate(idx) if c < 38]
hud = [ap50[i] for i, c in enumerate(idx) if c >= 38]
t = sum(tile) / len(tile) if tile else 0.0
h = sum(hud) / len(hud) if hud else 0.0
print(f"tiles mAP50={t:.4f} ({len(tile)} classes)   HUD mAP50={h:.4f} ({len(hud)} classes)")
for i, c in enumerate(idx):
    if c >= 38:
        print(f"  {names[c]:20s} AP50={ap50[i]:.4f}")
print("GATE:", "PASS" if t >= 0.988 else "FAIL (fall back to a separate HUD detector)")
```

- [ ] **Step 2 [USER RUN]: 训练**

```powershell
$env:PYTHONPATH = "."
python scripts/train/train_detector.py --data datasets/v2/detector/data.yaml
```
（16GiB 卡：expandable_segments + batch4，同 v1 经验。）

- [ ] **Step 3: 跑门槛评估**

Run: `PYTHONPATH=. $PY scripts/inspect/eval_detector_split.py runs/detect/<run>/weights/best.pt datasets/v2/detector/data.yaml`
Expected: `tiles mAP50 >= 0.988` → PASS；HUD 各类 AP 打印出来（按钮类允许偏低，
记录数值）。FAIL → 按 spec §6 退双模型：用 `--stage detector` 造一份仅 HUD 类
的 split 训独立小检测器（新任务，先向用户报告再动）。

- [ ] **Step 4: 发布权重 + Commit**

PASS 后把 best.pt 复制为 `majsoul_eye/recognize/tile_detector.pt`（本地跟踪策略
同现状：不进 git）。commit 只含评估脚本：

```bash
git add scripts/inspect/eval_detector_split.py
git commit -m "feat(inspect): grouped mAP eval (tiles-vs-HUD) with regression gate"
```

---

### Task 13: 运行时组装 + 端到端 QA

**Files:**
- Create: `majsoul_eye/recognize/hudstate.py`
- Create: `scripts/inspect/qa_hud.py`
- Test: `tests/test_hudstate.py`

**Interfaces:**
- Produces: `assemble_hud(dets: list[tuple[str, tuple[int,int,int,int]]], reader: HudReader, frame_bgr) -> dict`：
  输入=检测器输出的 `(cls_name, px_box)` 列表（只取 38–54 类），输出
  `{"scores": {"self": int|None, "right":..., "across":..., "left":...},
    "round": str|None, "wall": int|None, "kyotaku": int|None, "honba": int|None,
    "seat_wind": str|None, "buttons": list[str]}`。
  数字解析：剥 `余`/`x` 前缀 → int，解析失败 → None（不猜）。
  裁剪+转正在此完成（`FIELD_ROT`），与训练 crop 同路径。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_hudstate.py
"""assemble_hud: routes fields through a stub reader, strips 余/x, collects
buttons; unparseable -> None."""
import numpy as np

from majsoul_eye.recognize.hudstate import assemble_hud


class StubReader:
    def __init__(self, answers): self.answers = answers
    def read(self, crop, cls): return self.answers[cls]


frame = np.zeros((1080, 1920, 3), np.uint8)
dets = [("score_self", (900, 460, 1000, 500)), ("wall_count", (925, 385, 995, 415)),
        ("honba_count", (235, 135, 315, 185)), ("round_label", (905, 350, 1015, 385)),
        ("btn_pon", (1200, 740, 1360, 790)), ("btn_skip", (1400, 740, 1560, 790)),
        ("1m", (100, 900, 190, 1050))]          # tile det must be ignored
r = StubReader({"score_self": "25000", "wall_count": "余64",
                "honba_count": "x1", "round_label": "E3"})
h = assemble_hud(dets, r, frame)
assert h["scores"]["self"] == 25000 and h["scores"]["across"] is None
assert h["wall"] == 64 and h["honba"] == 1 and h["round"] == "E3"
assert h["buttons"] == ["btn_pon", "btn_skip"]

bad = StubReader({"score_self": "2x500", "wall_count": "余",
                  "honba_count": "x1", "round_label": "E3"})
h2 = assemble_hud(dets, bad, frame)
assert h2["scores"]["self"] is None and h2["wall"] is None
print("test_hudstate OK")
```

- [ ] **Step 2: 跑测试确认失败** → **Step 3: 实现**

```python
# majsoul_eye/recognize/hudstate.py
"""Assemble detector HUD boxes + micro-reader outputs into one structured dict
(the HUD half of the recognized 场况; tile half comes from detector/classifier).
Crop->rotate-upright->read happens here so runtime matches the training crops."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from majsoul_eye.hud import FIELD_ROT, HUD_NAMES, NUMERIC_FIELDS

_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE}
_SCORE_KEY = {"score_self": "self", "score_right": "right",
              "score_across": "across", "score_left": "left"}


def _to_int(s: str, strip: str = "") -> Optional[int]:
    s = s.lstrip(strip)
    try:
        return int(s)
    except ValueError:
        return None


def assemble_hud(dets, reader, frame_bgr: np.ndarray) -> dict:
    out = {"scores": {"self": None, "right": None, "across": None, "left": None},
           "round": None, "wall": None, "kyotaku": None, "honba": None,
           "seat_wind": None, "buttons": []}
    for cls, (x0, y0, x1, y1) in dets:
        if cls not in HUD_NAMES:
            continue
        if cls.startswith("btn_"):
            out["buttons"].append(cls)
            continue
        crop = frame_bgr[max(0, y0):y1, max(0, x0):x1]
        if crop.size == 0:
            continue
        rot = FIELD_ROT.get(cls, 0)
        if rot in _ROT:
            crop = cv2.rotate(crop, _ROT[rot])
        text = reader.read(crop, cls)
        if cls in _SCORE_KEY:
            out["scores"][_SCORE_KEY[cls]] = _to_int(text)
        elif cls == "wall_count":
            out["wall"] = _to_int(text, strip="余")
        elif cls == "riichi_stick_count":
            out["kyotaku"] = _to_int(text, strip="x")
        elif cls == "honba_count":
            out["honba"] = _to_int(text, strip="x")
        elif cls == "round_label":
            out["round"] = text
        elif cls == "seat_wind_self":
            out["seat_wind"] = text
    out["buttons"].sort(key=HUD_NAMES.index)
    return out
```

- [ ] **Step 4: 跑测试确认通过** → **Step 5: Commit**

```bash
git add majsoul_eye/recognize/hudstate.py tests/test_hudstate.py
git commit -m "feat(recognize): assemble_hud — detector boxes + readers -> structured HUD state"
```

- [ ] **Step 6: 端到端 QA 脚本 + 实测**

```python
# scripts/inspect/qa_hud.py
"""End-to-end HUD accuracy vs GT on a held-out game: detector v2 + HudReader
-> assemble_hud, compared field-by-field to the replayed BoardState.

Usage: PYTHONPATH=. python scripts/inspect/qa_hud.py \
           captures/raw/ai_session/run_8/game1.jsonl
"""
```
实现要点：`build_seq_state`+`load_frames` 遍历（跳 `is_deal_window`/
`is_score_anim_window` 帧）；`TileDetector` 出框（38–54 类）→ `assemble_hud`；
逐字段与 `field_texts(state)`/`buttons_for_ops(state.pending_ops)` 比对；输出
每字段精确率 + 整帧全对率（spec §6）。

Run（[USER RUN]，需要 GPU/或 CPU 慢跑）:
`PYTHONPATH=. $PY scripts/inspect/qa_hud.py captures/raw/ai_session/run_8/game1.jsonl`
Expected: 每字段 ≥99%（分数/余牌）；按钮 recall 报告数值（数据少，先记录）。
结果贴给用户。

- [ ] **Step 7: Commit**

```bash
git add scripts/inspect/qa_hud.py
git commit -m "feat(inspect): end-to-end HUD QA vs GT (per-field + whole-frame exact)"
```

---

### Task 14: 文档同步（管线纪律）

**Files:**
- Modify: `docs/PIPELINE.md`（数据流图/命令加 HUD 支线：annotate 出 `hud_boxes`、
  build 出 55 类 YOLO + `hud/` crops、train_hudreader、qa_hud）
- Modify: `docs/STATUS.md`（新条目 §1.x：HUD 检测落地——类别表、按钮采集方式、
  门槛结果、读取器指标）
- Modify: `CLAUDE.md`（Architecture 段：`hud.py`/`state/ops.py`/`annotate/hud.py`/
  `recognize/hudreader.py`/`hudstate.py` 一句话各归位；55 类口径）
- Modify: `docs/superpowers/specs/2026-07-04-hud-detection-design.md`（把 §7 的
  record_gt 按钮兜底改为 `--op-delay` 采集，与实现对齐）

- [ ] **Step 1: 按各文档现有风格写入**（PIPELINE 是权威管线文档，改动最细）
- [ ] **Step 2: 全量测试最后过一遍**

Run: `for t in tests/test_*.py; do PYTHONPATH=. $PY "$t" || break; done`
Expected: 全 OK

- [ ] **Step 3: Commit**

```bash
git add docs/PIPELINE.md docs/STATUS.md CLAUDE.md docs/superpowers/specs/2026-07-04-hud-detection-design.md
git commit -m "docs: sync PIPELINE/STATUS/CLAUDE for HUD detection (55-class + readers)"
```

---

## 任务依赖与并行性

```
T1(taxonomy) ─┬─ T2(ops) ─── T3(inventory) ─── T4(op-delay+采集[USER])
              │                                     │
              ├─ T5(字段几何) ── T6(字段标定👁) ──┐  │
              │                                  ├─ T7(按钮定位👁, 需T4数据)
              └──────────────────────────────────┘  │
T8(annotate接入, 需T5/T7) ── T9(dataset产出) ── T10(build v2)
T11(读取器, 需T10数据; 模型代码可先行) ── T12(检测器v2[USER GPU])
T13(组装+端到端QA, 需T11/T12) ── T14(docs)
```
👁 = 人眼检查点（overlay 贴给用户）。T4 的采集与 T5/T6 可并行。

## 风险备忘（执行中盯）

- 按钮外观我们**还没在 PC 帧上见过**（现有帧都无按钮）——T7 的阈值/顺序/skip
  规则全部以 T4 采到的真帧为准，预期要迭代 2–3 轮。
- `btn_kyushu`/`btn_tsumo`/`btn_ron` 样本会很稀（一局最多几次）——T12 记录
  per-class AP 即可，不阻塞；后续采集自然积累。
- v1 旧 annotations 无 `hud_boxes` → HUD 训练数据只来自 v2 全量重 annotate，
  混训 v1 只贡献牌面标签（合法，nc=55 兼容 0–37 标签）。
