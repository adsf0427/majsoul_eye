# 局面复原（ObservedState + MJAI 序列重建）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **⚠️ 已执行完毕（2026-07-06，branch feat/board-recon）。本文保留为历史规格；下列执行期
> 裁定的偏差以代码为准（详见 `.git/worktrees/majsoul_eye-recon/sdd/progress-board-recon.md`）：**
> 1. **Task 4** `test_chi_only_from_kamicha`：本文的 fixture `rivers[3]=[W]` 被证明不可行
>    （吃只能吃上家=相邻座、无跳座，r1/r2 空时上家拿不到第二次舍牌轮）——实现删去 W，断言未动。
> 2. **Task 7** `_parse_melds`：本文的"size-4 优先贪心"存在"误配尾部恰好也可解析→静默错分组"
>    漏洞（复现需 ≥7 张同种牌，合法局面不可达）——实现改为**歧义检测**（枚举完整解析，≥2 个
>    即拒绝并记 violation），新增 `test_meld_parse_rejects_ambiguous_strip`。
> 3. **Task 8** `test_full_frame_roundtrip`：本文 fixture 中 r3 的 "F" 与暗杠 F×4 凑成 5 张
>    自相矛盾——实现换成 "9s"，断言未动。
> 4. **Task 9** 真实数据发现新帧类：鸣牌事件与强制舍牌之间的间隙帧（209/10330 ≈ 2%，其中 11
>    帧不可重建）——新增 `replay.is_call_pending` 谓词（镜像 `is_deal_window`）+
>    `BoardState.awaiting_discard` 字段，oracle 按类跳过并计数；另修 `run_engine` 的
>    failure 计数器类型 bug（本文代码 `report["fail"] += 1` 打在 list 上）→ `engine_fail`。
>    engine 层本次未跑（环境无 mjai bot 命令）。

**Goal:** 单帧识别结果 → `ObservedState`（第 1 步）→ 从 `start_kyoku` 到当前状态的合法 hero 视角 MJAI 序列（第 2 步），并配三层 GT 评测。

**Architecture:** 三个新模块：`state/observe.py`（可见状态数据模型 + 校验 + GT 投影）、`recognize/assemble.py`（检测框 → ObservedState，反用 `annotate/pipeline` 标定几何）、`state/reconstruct.py`（回合模拟 + 回溯 DFS → mjai 事件）。评测 harness `scripts/eval/eval_reconstruction.py` 用现有 GTRecord 捕获免费驱动。Spec：`docs/superpowers/specs/2026-07-05-board-reconstruction-design.md`。

**Tech Stack:** Python（conda `auto` 环境）、numpy/cv2（仅 assemble/eval 路径）、现有 `majsoul_eye` 包。无新第三方依赖。

## Global Constraints

- 一切命令从仓库根运行，`PYTHONPATH=.`；用户文档写普通 `python`；**执行代理的 Bash 工具必须用 `C:/Users/zsx/miniforge3/envs/auto/python.exe` 代替 `python`**（默认 PATH 无 numpy）。
- 测试 = `tests/` 下 plain script（无 pytest 依赖，pytest 兼容）：模块级 `test_*` 函数 + 文件尾 `if __name__ == "__main__":` 循环调用并 `print("<name> OK")`。
- 38 牌类顺序冻结（`tiles.py`）；`recognize/` 保持 Akagi-free（可 import `annotate.pipeline`——纯几何，不得 import `capture/`）。
- 座位约定：**屏幕相对位** 0=self 1=right(下家) 2=across(对家) 3=left(上家)；相对↔绝对：`abs = (hero_abs + rel) % 4`。
- `ObservedState.hero_hand` **不含** `drawn_tile`（摸牌单列）。
- 提交风格：`feat(state): ...` / `feat(recognize): ...` / `docs: ...`，小步频繁提交。
- 开工前从当前 HEAD 建分支：`git checkout -b feat/board-recon`。

## File Structure

- Create: `majsoul_eye/state/observe.py` — 数据类 + `check_observed` + `observed_from_board`
- Create: `majsoul_eye/state/reconstruct.py` — `reconstruct(obs) -> ReconstructionResult`
- Create: `majsoul_eye/recognize/assemble.py` — `assemble(dets, region) -> ObservedState`
- Create: `scripts/eval/eval_reconstruction.py` + `scripts/eval/__init__.py` — 三层评测
- Create: `tests/test_observe.py`, `tests/test_reconstruct.py`, `tests/test_assemble.py`
- Modify: `docs/PIPELINE.md`（§4 QA 工具清单加一行）、`docs/STATUS.md`（新条目）

---

### Task 1: `state/observe.py` — 数据模型 + `check_observed`

**Files:**
- Create: `majsoul_eye/state/observe.py`
- Test: `tests/test_observe.py`

**Interfaces:**
- Produces: `ObservedRiverTile(pai, sideways)`、`ObservedMeld(type, tiles, called_pai, added_pai, from_rel)`、`ObservedState(...)`（字段见代码）、`check_observed(obs) -> list[str]`。后续所有 Task 依赖这些名字与语义。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_observe.py
"""ObservedState schema + single-frame consistency checks (spec 2026-07-05 §3.1)."""
from majsoul_eye.state.observe import (
    ObservedMeld, ObservedRiverTile, ObservedState, check_observed)


def _minimal():
    return ObservedState(
        hero_hand=["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
        rivers=[[], [], [], []], melds=[[], [], [], []],
        dora_markers=["5s"], concealed_counts=[None, 13, 13, 13],
        reach=[False] * 4)


def test_clean_state_has_no_violations():
    assert check_observed(_minimal()) == []


def test_fifth_copy_flagged():
    o = _minimal()
    o.rivers[1] = [ObservedRiverTile("1m") for _ in range(4)]  # + one in hand = 5
    v = check_observed(o)
    assert any("1m" in m and "5" in m for m in v)


def test_red_five_counts_with_plain():
    o = _minimal()
    o.hero_hand = ["5m", "5m", "5m", "5mr"] + o.hero_hand[4:]
    o.rivers[2] = [ObservedRiverTile("5m")]                    # 5th 5m-kind
    assert check_observed(o)


def test_hand_size_vs_melds():
    o = _minimal()
    o.hero_hand = o.hero_hand[:12]                             # 12 + 0 melds != 13
    assert any("hand" in m for m in check_observed(o))
    o2 = _minimal()
    o2.hero_hand = o2.hero_hand[:10]
    o2.melds[0] = [ObservedMeld("pon", ["P", "P", "P"], called_pai="P", from_rel=2)]
    assert check_observed(o2) == []                            # 10 + 3*1 == 13


def test_drawn_tile_is_extra():
    o = _minimal()
    o.drawn_tile = "9s"
    assert check_observed(o) == []                             # 13 + drawn == to-act


def test_dora_rules():
    o = _minimal()
    o.dora_markers = []
    assert any("dora" in m for m in check_observed(o))
    o2 = _minimal()
    o2.dora_markers = ["5s", "6s", "7s"]                       # 3 markers, 0 kans
    assert any("dora" in m or "kan" in m for m in check_observed(o2))
    o3 = _minimal()
    o3.hero_hand = o3.hero_hand[:10]
    o3.melds[0] = [ObservedMeld("ankan", ["C", "C", "C", "C"])]
    o3.dora_markers = ["5s", "6s"]                             # 2 markers, 1 kan: ok
    assert check_observed(o3) == []


def test_concealed_counts_cross_check():
    o = _minimal()
    o.concealed_counts = [None, 10, 13, 13]                    # seat1: 10 but 0 melds
    assert any("concealed" in m for m in check_observed(o))
    o2 = _minimal()
    o2.melds[1] = [ObservedMeld("chi", ["1s", "2s", "3s"], called_pai="2s", from_rel=3)]
    o2.concealed_counts = [None, 10, 13, 13]
    assert check_observed(o2) == []                            # 13 - 3*1 == 10 (or 11 mid-draw)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_observe OK")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. python tests/test_observe.py`
Expected: `ModuleNotFoundError: No module named 'majsoul_eye.state.observe'`

- [ ] **Step 3: 实现 `majsoul_eye/state/observe.py`**

```python
"""Single-frame OBSERVED board state — the vision-side mirror of replay.BoardState.

ObservedState is what one screenshot shows a human (spec 2026-07-05 §3.1):
tiles zones are recognizable today; 2D-HUD fields are Optional slots filled once
the HUD micro-readers (spec 2026-07-04) land. Seats are SCREEN-RELATIVE
(0=self 1=right 2=across 3=left, counter-clockwise = turn order).
hero_hand EXCLUDES drawn_tile (the separated tsumo slot is its own field).
Pure data + checks; no cv2/numpy/Akagi imports at module level.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from majsoul_eye.tiles import TILE_NAMES, red_to_normal

_VALID = set(TILE_NAMES)


@dataclass
class ObservedRiverTile:
    pai: str
    sideways: bool = False           # rendered sideways (riichi declaration slot)


@dataclass
class ObservedMeld:
    type: str                        # chi | pon | daiminkan | ankan | kakan
    tiles: list[str] = field(default_factory=list)   # full composition, reds exact
    called_pai: str = ""             # sideways tile ("" for ankan)
    added_pai: str = ""              # kakan's stacked tile
    from_rel: int = 0                # (target - owner) % 4: 1=shimocha 2=toimen 3=kamicha; 0=ankan


@dataclass
class ObservedState:
    hero_hand: list[str] = field(default_factory=list)
    drawn_tile: Optional[str] = None
    rivers: list[list[ObservedRiverTile]] = field(default_factory=lambda: [[] for _ in range(4)])
    melds: list[list[ObservedMeld]] = field(default_factory=lambda: [[] for _ in range(4)])
    dora_markers: list[str] = field(default_factory=list)
    concealed_counts: list[Optional[int]] = field(default_factory=lambda: [None] * 4)
    reach: list[bool] = field(default_factory=lambda: [False] * 4)
    # --- 2D HUD slots (None until the HUD readers land; relative seat order) ---
    scores: Optional[list[int]] = None
    bakaze: Optional[str] = None
    kyoku: Optional[int] = None
    honba: Optional[int] = None
    kyotaku: Optional[int] = None
    left_tile_count: Optional[int] = None
    seat_wind_self: Optional[str] = None
    pending_buttons: Optional[list[str]] = None
    # --- meta ---
    violations: list[str] = field(default_factory=list)
    zone_confidence: dict[str, float] = field(default_factory=dict)

    def n_kans(self) -> int:
        return sum(1 for ms in self.melds for m in ms
                   if m.type in ("daiminkan", "ankan", "kakan"))


def check_observed(o: ObservedState) -> list[str]:
    """Single-frame consistency checks (replay.check_invariants' vision twin)."""
    v: list[str] = []
    counts: dict[str, int] = {}

    def bump(pai: str) -> None:
        if pai and pai != "back":
            if pai not in _VALID:
                v.append(f"unknown tile name {pai!r}")
                return
            k = red_to_normal(pai)
            counts[k] = counts.get(k, 0) + 1

    for p in o.hero_hand:
        bump(p)
    if o.drawn_tile:
        bump(o.drawn_tile)
    for r in range(4):
        for t in o.rivers[r]:
            bump(t.pai)
        for m in o.melds[r]:
            for p in m.tiles:
                bump(p)
    for d in o.dora_markers:
        bump(d)
    for kind, n in counts.items():
        if n > 4:
            v.append(f"tile {kind} seen {n}>4 times across visible zones")

    n_melds = len(o.melds[0])
    if len(o.hero_hand) + 3 * n_melds != 13:
        v.append(f"hero hand {len(o.hero_hand)} + 3*{n_melds} melds != 13")

    if not o.dora_markers:
        v.append("no dora marker visible")
    elif len(o.dora_markers) - 1 > o.n_kans():
        v.append(f"{len(o.dora_markers)} dora markers but only {o.n_kans()} kans")

    for r in range(1, 4):
        c = o.concealed_counts[r]
        if c is None:
            continue
        expect = 13 - 3 * len(o.melds[r])
        if c not in (expect, expect + 1):      # +1: that seat may be mid-draw
            v.append(f"seat {r} concealed {c} != {expect}(+1) for {len(o.melds[r])} melds")

    for r in range(4):
        for m in o.melds[r]:
            if m.type in ("chi", "pon", "daiminkan") and m.from_rel not in (1, 2, 3):
                v.append(f"seat {r} {m.type} from_rel {m.from_rel} invalid")
            if m.type == "chi" and m.from_rel != 3:
                v.append(f"seat {r} chi from_rel {m.from_rel} != 3 (kamicha only)")
    return v
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. python tests/test_observe.py`
Expected: `test_observe OK`

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/state/observe.py tests/test_observe.py
git commit -m "feat(state): ObservedState schema + single-frame consistency checks"
```

---

### Task 2: `observed_from_board` — GT 投影（评测器基石）

**Files:**
- Modify: `majsoul_eye/state/observe.py`（追加函数）
- Test: `tests/test_observe.py`（追加用例）

**Interfaces:**
- Consumes: `replay.BoardState`（`visible_river`/`melds`/`hero_hand`/`drawn_tile`/`reach`/`scores`…）、`annotate.pipeline.river_sideways_index`（函数内 lazy import，避免模块级 cv2）。
- Produces: `observed_from_board(state, include_hud=True) -> ObservedState`。Task 3-5 的往返测试、Task 8 的装配对比、Task 9 的评测全部依赖它。

- [ ] **Step 1: 追加失败测试（`tests/test_observe.py` 文件尾 `__main__` 块之前）**

```python
def _played_state():
    from majsoul_eye.state.replay import Replayer
    rp = Replayer()
    for ev in [
        {"type": "start_game", "id": 1},
        {"type": "start_kyoku", "bakaze": "E", "dora_marker": "1m", "honba": 1,
         "kyoku": 2, "kyotaku": 0, "oya": 1,
         "scores": [25000, 25000, 25000, 25000],
         "tehais": [["?"] * 13,
                    ["1m", "2m", "3m", "2p", "2p", "5p", "6p", "7p", "9p", "1s", "2s", "3s", "9s"],
                    ["?"] * 13, ["?"] * 13]},
        {"type": "tsumo", "actor": 1, "pai": "4p"},
        {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": False},
        {"type": "tsumo", "actor": 2, "pai": "?"},
        {"type": "dahai", "actor": 2, "pai": "2p", "tsumogiri": True},
        {"type": "pon", "actor": 1, "target": 2, "pai": "2p", "consumed": ["2p", "2p"]},
        {"type": "dahai", "actor": 1, "pai": "9p", "tsumogiri": False},
        {"type": "tsumo", "actor": 3, "pai": "?"},
        {"type": "reach", "actor": 3},
        {"type": "dahai", "actor": 3, "pai": "W", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 3},
        {"type": "tsumo", "actor": 0, "pai": "?"},
        {"type": "dahai", "actor": 0, "pai": "E", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "8p"},
    ]:
        rp.apply(ev)
    return rp.state


def test_projection_relative_seats_and_zones():
    from majsoul_eye.state.observe import observed_from_board
    s = _played_state()                       # hero = abs seat 1
    o = observed_from_board(s)
    assert check_observed(o) == []
    # hero (rel 0): river [9s, 9p]; pon from rel target: target abs2 = hero+1 -> from_rel 1
    assert [t.pai for t in o.rivers[0]] == ["9s", "9p"]
    assert o.melds[0][0].type == "pon" and o.melds[0][0].from_rel == 1
    # abs2 = rel1: river had 2p but it was called away -> visible []
    assert o.rivers[1] == []
    # abs3 = rel2: riichi discard W is sideways; reach flag on
    assert [t.pai for t in o.rivers[2]] == ["W"] and o.rivers[2][0].sideways
    assert o.reach == [False, False, True, False]
    # abs0 = rel3
    assert [t.pai for t in o.rivers[3]] == ["E"]
    # hero hand excludes the fresh 8p draw
    assert o.drawn_tile == "8p" and "8p" not in o.hero_hand and len(o.hero_hand) == 10
    # HUD projection (include_hud default True): relative score order
    # rel order = [abs1, abs2, abs3, abs0]; abs3 paid 1000 for riichi
    assert o.scores == [25000, 25000, 24000, 25000]
    assert o.bakaze == "E" and o.kyoku == 2 and o.honba == 1 and o.kyotaku == 1
    assert o.seat_wind_self == "E"            # hero IS oya (kyoku 2, oya=1=hero)


def test_projection_without_hud():
    from majsoul_eye.state.observe import observed_from_board
    o = observed_from_board(_played_state(), include_hud=False)
    assert o.scores is None and o.bakaze is None and o.kyoku is None
    assert o.dora_markers == ["1m"]           # dora strip is detectable, not an HUD slot
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. python tests/test_observe.py`
Expected: `ImportError: cannot import name 'observed_from_board'`

- [ ] **Step 3: 在 `observe.py` 追加实现**

```python
WINDS = ["E", "S", "W", "N"]


def observed_from_board(state, include_hud: bool = True) -> ObservedState:
    """Project a replayed BoardState to what the SCREEN shows (eval oracle).

    Relative seats: rel r == abs (hero+r)%4. reach[] is derived from the
    VISIBLE sideways tile, not state.reach — a riichi whose declaration tile was
    called away with no later discard is invisible in a single frame, and the
    projection must be fair to the vision side (known static limitation).
    Lazy-imports annotate.pipeline for river_sideways_index (keeps this module
    cv2-free unless projecting).
    """
    from majsoul_eye.annotate.pipeline import river_sideways_index

    hero = state.hero_seat
    o = ObservedState()
    o.drawn_tile = state.drawn_tile
    hand = list(state.hero_hand)
    if state.drawn_tile:
        if state.drawn_tile in hand:
            hand.remove(state.drawn_tile)
        elif red_to_normal(state.drawn_tile) in hand:
            hand.remove(red_to_normal(state.drawn_tile))
    o.hero_hand = hand
    o.dora_markers = list(state.dora_markers)
    for r in range(4):
        a = (hero + r) % 4
        vis = state.visible_river(a)
        side = river_sideways_index(
            [{"riichi": t.riichi, "called": t.called} for t in state.rivers[a]])
        o.rivers[r] = [ObservedRiverTile(t.pai, sideways=(i == side))
                       for i, t in enumerate(vis)]
        o.melds[r] = [ObservedMeld(m.type, list(m.tiles), m.called_pai, m.added_pai,
                                   from_rel=((m.from_seat - a) % 4))
                      for m in state.melds[a] if m.type != "nukidora"]
        o.reach[r] = any(t.sideways for t in o.rivers[r])
        o.concealed_counts[r] = None if r == 0 else state.concealed_counts[a]
    if include_hud:
        o.scores = [state.scores[(hero + r) % 4] for r in range(4)]
        o.bakaze, o.kyoku = state.bakaze, state.kyoku
        o.honba, o.kyotaku = state.honba, state.kyotaku
        o.left_tile_count = state.left_tile_count
        if state.oya >= 0 and hero >= 0:
            o.seat_wind_self = WINDS[(hero - state.oya) % 4]
    return o
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. python tests/test_observe.py` → `test_observe OK`
Run: `PYTHONPATH=. python tests/test_replay.py` → `test_replay OK`（回归）

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/state/observe.py tests/test_observe.py
git commit -m "feat(state): observed_from_board GT projection (eval oracle)"
```

---

### Task 3: `state/reconstruct.py` — 轮转骨架（无叫牌/立直/杠）

**Files:**
- Create: `majsoul_eye/state/reconstruct.py`
- Test: `tests/test_reconstruct.py`

**Interfaces:**
- Consumes: `ObservedState`、`observed_from_board`、`replay.Replayer`（测试往返）。
- Produces: `ReconstructionResult(ok, events, reason, fabricated, diagnostics)`、`reconstruct(obs) -> ReconstructionResult`。内部 `_search(obs, oya_rel) -> Optional[list]`、`_emit(obs, ops, oya_rel) -> tuple[list, dict]`——Task 4/5 扩展这两个函数。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_reconstruct.py
"""ObservedState -> legal mjai sequence. Every case round-trips through the
Replayer and must project back to the exact same ObservedState."""
from majsoul_eye.state.observe import ObservedMeld, ObservedRiverTile, ObservedState
from majsoul_eye.state.reconstruct import reconstruct

H13 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]


def _obs(**kw):
    o = ObservedState(hero_hand=list(H13), dora_markers=["5s"],
                      rivers=[[], [], [], []], melds=[[], [], [], []])
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def _roundtrip(obs):
    from majsoul_eye.state.observe import observed_from_board
    from majsoul_eye.state.replay import Replayer, check_invariants
    r = reconstruct(obs)
    assert r.ok, r.reason
    rp = Replayer()
    for ev in r.events:
        rp.apply(ev)
    assert check_invariants(rp.state) == []
    back = observed_from_board(rp.state, include_hud=False)
    assert [[t.pai for t in riv] for riv in back.rivers] == \
           [[t.pai for t in riv] for riv in obs.rivers]
    assert [[t.sideways for t in riv] for riv in back.rivers] == \
           [[t.sideways for t in riv] for riv in obs.rivers]
    assert [[(m.type, m.from_rel, sorted(m.tiles)) for m in ms] for ms in back.melds] == \
           [[(m.type, m.from_rel, sorted(m.tiles)) for m in ms] for ms in obs.melds]
    assert sorted(back.hero_hand) == sorted(obs.hero_hand)
    assert back.drawn_tile == obs.drawn_tile
    assert back.dora_markers == obs.dora_markers
    return r


def test_empty_board_start_of_kyoku():
    r = _roundtrip(_obs())
    assert [e["type"] for e in r.events] == ["start_game", "start_kyoku"]


def test_rotation_only():
    o = _obs(rivers=[[ObservedRiverTile("9p")], [ObservedRiverTile("E")],
                     [ObservedRiverTile("S")], []])
    r = _roundtrip(o)
    kinds = [e["type"] for e in r.events]
    assert kinds == ["start_game", "start_kyoku",
                     "tsumo", "dahai", "tsumo", "dahai", "tsumo", "dahai"]
    # oya inferred = rel 0 (hero discarded first); hero_abs defaults to 0
    sk = r.events[1]
    assert sk["oya"] == 0 and sk["kyoku"] == 1 and sk["bakaze"] == "E"
    assert sorted(sk["tehais"][0]) == sorted(H13)           # all-tsumogiri: haipai == hand
    assert sk["tehais"][1] == ["?"] * 13
    # hero discard is tsumo-then-cut of the same tile
    assert r.events[2] == {"type": "tsumo", "actor": 0, "pai": "9p"}
    assert r.events[3]["tsumogiri"] is True


def test_oya_inferred_uniquely_from_river_lengths():
    o = _obs(rivers=[[ObservedRiverTile("9p")], [ObservedRiverTile("E")],
                     [ObservedRiverTile("S")], []])
    r = reconstruct(o)
    assert r.diagnostics["feasible_oya_rel"] == [0]


def test_hero_holding_draw():
    o = _obs(rivers=[[ObservedRiverTile("9p")], [ObservedRiverTile("E")],
                     [ObservedRiverTile("S")], [ObservedRiverTile("W")]],
             drawn_tile="6s")
    r = _roundtrip(o)
    assert r.events[-1] == {"type": "tsumo", "actor": 0, "pai": "6s"}


def test_hud_fields_flow_into_start_kyoku():
    o = _obs(rivers=[[], [ObservedRiverTile("E")], [ObservedRiverTile("S")],
                     [ObservedRiverTile("W")]],
             bakaze="S", kyoku=3, honba=2, kyotaku=1, seat_wind_self="N",
             scores=[24000, 26000, 25000, 25000])
    # seat_wind N -> oya_rel = (4 - 3) % 4 = 1; kyoku 3 -> oya_abs 2 -> hero_abs 1
    r = _roundtrip(o)
    sk = r.events[1]
    assert sk["bakaze"] == "S" and sk["kyoku"] == 3 and sk["oya"] == 2
    assert sk["honba"] == 2 and sk["kyotaku"] == 1
    assert r.events[0] == {"type": "start_game", "id": 1}
    assert sk["scores"][1] == 24000                          # hero_abs=1 slot = rel0 score
    assert sorted(sk["tehais"][1]) == sorted(H13)


def test_infeasible_reports_reason():
    # rel3 discarded but rel2 hasn't and nothing explains the skip -> no legal order
    o = _obs(rivers=[[ObservedRiverTile("9p")], [ObservedRiverTile("E")],
                     [], [ObservedRiverTile("W"), ObservedRiverTile("N")]])
    r = reconstruct(o)
    assert not r.ok and r.reason


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_reconstruct OK")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. python tests/test_reconstruct.py`
Expected: `ModuleNotFoundError: No module named 'majsoul_eye.state.reconstruct'`

- [ ] **Step 3: 实现骨架**

```python
# majsoul_eye/state/reconstruct.py
"""ObservedState -> legal hero-perspective MJAI sequence (spec 2026-07-05 §4).

Turn-machine simulation with backtracking DFS over call timing, then a
deterministic emission pass: hero draws are fabricated "all-tsumogiri" (every
hero discard = tsumo X, dahai X tsumogiri) so the fabricated haipai is exactly
hero_hand + meld-consumed + forced post-call tedashi. Opponents draw "?".
Canonical solution: plain discards preferred over calls (= calls as late as
feasible). Pure logic — no vision/Akagi imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from majsoul_eye.state.observe import ObservedState, check_observed

WINDS = ["E", "S", "W", "N"]


@dataclass
class ReconstructionResult:
    ok: bool
    events: list = field(default_factory=list)
    reason: str = ""
    fabricated: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


# --- search (Task 3: rotation only; Task 4 adds calls; Task 5 adds kans/riichi) ---

def _search(obs: ObservedState, oya_rel: int) -> Optional[list]:
    """Return op list or None. Ops:
    ("draw", rel) | ("discard", rel, idx) — extended by later tasks."""
    rivers = obs.rivers
    n = [len(r) for r in rivers]

    def go(cursors: tuple, actor: int) -> Optional[list]:
        if list(cursors) == n:
            if obs.drawn_tile is not None:
                return [("draw", 0)] if actor == 0 else None
            return []
        if cursors[actor] < n[actor]:
            nxt = list(cursors)
            nxt[actor] += 1
            rest = go(tuple(nxt), (actor + 1) % 4)
            if rest is not None:
                return [("draw", actor), ("discard", actor, cursors[actor])] + rest
        return None

    return go((0, 0, 0, 0), oya_rel)


# --- emission -----------------------------------------------------------------

def _emit(obs: ObservedState, ops: list, oya_rel: int):
    """ops -> (mjai events after start_kyoku, info dict for backfill)."""
    events: list = []
    haipai = list(obs.hero_hand)
    reach_count = [0] * 4
    for i, op in enumerate(ops):
        kind = op[0]
        if kind == "draw":
            r = op[1]
            if r != 0:
                events.append({"type": "tsumo", "actor": r, "pai": "?"})
                continue
            pai = obs.drawn_tile
            if i + 1 < len(ops):
                nxt = ops[i + 1]
                if nxt[0] == "discard":
                    pai = obs.rivers[0][nxt[2]].pai
            events.append({"type": "tsumo", "actor": 0, "pai": pai})
        elif kind == "discard":
            r, idx = op[1], op[2]
            pai = obs.rivers[r][idx].pai
            events.append({"type": "dahai", "actor": r, "pai": pai,
                           "tsumogiri": r == 0})
    return events, {"haipai": haipai, "reach_count": reach_count}


# --- absolute-seat mapping + start_kyoku backfill -------------------------------

def _abs_map(obs: ObservedState, oya_rel: int):
    """(hero_abs, oya_abs, kyoku). Without HUD: hero_abs=0, kyoku=oya_rel+1."""
    if obs.kyoku is not None:
        oya_abs = obs.kyoku - 1
        return (oya_abs - oya_rel) % 4, oya_abs, obs.kyoku
    return 0, oya_rel, oya_rel + 1


def _relabel(events: list, hero_abs: int) -> list:
    out = []
    for ev in events:
        ev = dict(ev)
        for k in ("actor", "target"):
            if k in ev:
                ev[k] = (hero_abs + ev[k]) % 4
        out.append(ev)
    return out


def reconstruct(obs: ObservedState) -> ReconstructionResult:
    viol = check_observed(obs)
    if viol:
        return ReconstructionResult(False, reason="; ".join(viol))
    if obs.seat_wind_self is not None:
        cand = [(4 - WINDS.index(obs.seat_wind_self)) % 4]
    else:
        cand = [0, 1, 2, 3]
    feasible, chosen, ops = [], None, None
    for oya_rel in cand:
        got = _search(obs, oya_rel)
        if got is not None:
            feasible.append(oya_rel)
            if chosen is None:
                chosen, ops = oya_rel, got
    if chosen is None:
        return ReconstructionResult(
            False, reason=f"no legal turn order for any oya in {cand}",
            diagnostics={"feasible_oya_rel": []})
    body, info = _emit(obs, ops, chosen)
    if len(info["haipai"]) != 13:
        return ReconstructionResult(
            False, reason=f"internal: fabricated haipai {len(info['haipai'])} != 13")
    hero_abs, oya_abs, kyoku = _abs_map(obs, chosen)
    n_reach = sum(info["reach_count"])
    scores_rel = list(obs.scores) if obs.scores is not None else [25000] * 4
    scores_abs = [25000] * 4
    for r in range(4):
        scores_abs[(hero_abs + r) % 4] = scores_rel[r] + 1000 * info["reach_count"][r]
    kyotaku = (obs.kyotaku if obs.kyotaku is not None else n_reach) - n_reach
    tehais: list = [["?"] * 13 for _ in range(4)]
    tehais[hero_abs] = sorted(info["haipai"])
    sk = {"type": "start_kyoku", "bakaze": obs.bakaze or "E", "kyoku": kyoku,
          "honba": obs.honba or 0, "kyotaku": max(0, kyotaku), "oya": oya_abs,
          "dora_marker": obs.dora_markers[0], "scores": scores_abs, "tehais": tehais}
    events = [{"type": "start_game", "id": hero_abs}, sk] + _relabel(body, hero_abs)
    fabricated = {"haipai": tehais[hero_abs],
                  "defaults": [k for k in ("scores", "bakaze", "kyoku", "honba", "kyotaku")
                               if getattr(obs, k) is None]}
    return ReconstructionResult(True, events=events, fabricated=fabricated,
                                diagnostics={"feasible_oya_rel": feasible,
                                             "oya_rel": chosen})
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. python tests/test_reconstruct.py`
Expected: `test_reconstruct OK`

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/state/reconstruct.py tests/test_reconstruct.py
git commit -m "feat(state): mjai reconstruction skeleton — rotation search + start_kyoku backfill"
```

---

### Task 4: 叫牌（chi/pon/daiminkan）— 幽灵舍牌 + 回溯 + memo

**Files:**
- Modify: `majsoul_eye/state/reconstruct.py`（替换 `_search`、`_emit`，新增 `_Item`/`_items_for`/`_minus`）
- Test: `tests/test_reconstruct.py`（追加用例）

**Interfaces:**
- Produces: ops 新增 `("ghost", rel, pai, reach_flag)`、`("call", _Item)`；`_Item(kind, owner, target, pai, consumed, mi)`。Task 5 复用并追加 `("ankan", _Item)`/`("kakan", _Item)`。

- [ ] **Step 1: 追加失败测试**

```python
def test_pon_ghost_discard_and_forced_tedashi():
    # hero(rel0) pon P from rel2 (across). hero hand 10 + pon = 13.
    o = _obs(hero_hand=H13[:10],
             melds=[[ObservedMeld("pon", ["P", "P", "P"], called_pai="P", from_rel=2)],
                    [], [], []],
             rivers=[[ObservedRiverTile("9p")],          # hero's forced tedashi after pon
                     [ObservedRiverTile("E")],
                     [ObservedRiverTile("S")],           # across also has a VISIBLE discard
                     []])
    r = _roundtrip(o)
    evs = r.events
    pon = next(e for e in evs if e["type"] == "pon")
    assert pon["actor"] == 0 and pon["target"] == 2 and pon["consumed"] == ["P", "P"]
    # ghost dahai P by seat 2 immediately precedes the pon
    ghost = evs[evs.index(pon) - 1]
    assert ghost == {"type": "dahai", "actor": 2, "pai": "P", "tsumogiri": False}
    # hero's discard right after the pon is tedashi and lands in haipai
    after = evs[evs.index(pon) + 1]
    assert after["type"] == "dahai" and after["actor"] == 0 and after["tsumogiri"] is False
    sk = evs[1]
    assert sorted(sk["tehais"][0]) == sorted(H13[:10] + ["P", "P", "9p"])


def test_chi_only_from_kamicha():
    o = _obs(hero_hand=H13[:10],
             melds=[[ObservedMeld("chi", ["4s", "5s", "6s"], called_pai="5s", from_rel=3)],
                    [], [], []],
             rivers=[[ObservedRiverTile("9p")], [], [], [ObservedRiverTile("W")]])
    r = _roundtrip(o)
    chi = next(e for e in r.events if e["type"] == "chi")
    assert chi["target"] == 3 and sorted(chi["consumed"]) == ["4s", "6s"]


def test_call_timing_needs_backtracking():
    # Late-call-first fails: 0:A -> 1:C -> 2 has nothing => backtrack to the
    # ghost branch (1 discards ghost P before its visible C).
    o = _obs(hero_hand=H13[:10],
             melds=[[ObservedMeld("pon", ["P", "P", "P"], called_pai="P", from_rel=1)],
                    [], [], []],
             rivers=[[ObservedRiverTile("1m"), ObservedRiverTile("9p")],
                     [ObservedRiverTile("2m")], [], []])
    _roundtrip(o)


def test_daiminkan_by_opponent():
    # rel1 daiminkan's C from hero (from_rel=3 -> target rel0): hero discards a
    # ghost C, rel1 kans + rinshan-draws + discards F. dora NOT yet flipped
    # (1 marker for 1 kan): allowed (daiminkan flip is delayed).
    o = _obs(melds=[[], [ObservedMeld("daiminkan", ["C", "C", "C", "C"],
                                      called_pai="C", from_rel=3)], [], []],
             rivers=[[ObservedRiverTile("9p")],
                     [ObservedRiverTile("E"), ObservedRiverTile("F")],
                     [ObservedRiverTile("S")], [ObservedRiverTile("W")]],
             dora_markers=["5s"])
    r = _roundtrip(o)
    kan = next(e for e in r.events if e["type"] == "daiminkan")
    assert kan["actor"] == 1 and kan["target"] == 0 and kan["consumed"] == ["C", "C", "C"]
    # the ghost C came from the hero: it was that turn's fabricated draw
    ghost = r.events[[i for i, e in enumerate(r.events)
                      if e["type"] == "dahai" and e["pai"] == "C"][0]]
    assert ghost["actor"] == 0 and ghost["tsumogiri"] is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. python tests/test_reconstruct.py`
Expected: `test_pon_ghost_discard_and_forced_tedashi` 失败（`no legal turn order`——骨架不认识副露）。

- [ ] **Step 3: 替换 `_search`/`_emit`，新增 items**

```python
from majsoul_eye.tiles import red_to_normal


def _minus(tiles: list, remove: list) -> list:
    """Multiset removal with red-five fallback both ways."""
    out = list(tiles)
    for x in remove:
        if x in out:
            out.remove(x)
            continue
        for t in list(out):
            if red_to_normal(t) == red_to_normal(x):
                out.remove(t)
                break
    return out


@dataclass(frozen=True)
class _Item:
    kind: str        # chi | pon | daiminkan | ankan | kakan
    owner: int
    target: int      # rel seat whose discard is claimed (== owner for ankan/kakan)
    pai: str         # claimed tile | kakan's added tile | "" for ankan
    consumed: tuple  # tiles leaving owner's HAND
    mi: int          # on-screen meld index (owner chronology)


def _items_for(obs: ObservedState):
    """(per-owner creation list in screen order, kakan own-turn parts)."""
    creation: list[list[_Item]] = [[] for _ in range(4)]
    kakans: list[_Item] = []
    for o in range(4):
        for mi, m in enumerate(obs.melds[o]):
            t = (o + m.from_rel) % 4
            if m.type in ("chi", "pon", "daiminkan"):
                creation[o].append(_Item(m.type, o, t, m.called_pai,
                                         tuple(_minus(m.tiles, [m.called_pai])), mi))
            elif m.type == "kakan":
                pon_cons = tuple(_minus(m.tiles, [m.called_pai, m.added_pai]))
                creation[o].append(_Item("pon", o, t, m.called_pai, pon_cons, mi))
                kakans.append(_Item("kakan", o, o, m.added_pai,
                                    tuple(_minus(m.tiles, [m.added_pai])), mi))
            elif m.type == "ankan":
                creation[o].append(_Item("ankan", o, o, "", tuple(m.tiles), mi))
    return creation, kakans


def _search(obs: ObservedState, oya_rel: int) -> Optional[list]:
    """Ops: ("draw",rel) ("discard",rel,idx) ("ghost",rel,pai,reach)
    ("call",_Item) ("ankan",_Item) ("kakan",_Item). Canonical branch order:
    visible discard > own-turn kan > ghost/call (calls as late as feasible)."""
    rivers = obs.rivers
    n = [len(r) for r in rivers]
    creation, kakans = _items_for(obs)
    ncre = [len(c) for c in creation]
    failed: set = set()

    def all_done(cur, cidx, kkmask):
        return list(cur) == n and list(cidx) == ncre and kkmask == (1 << len(kakans)) - 1

    def go(cur, cidx, kkmask, actor):
        key = (cur, cidx, kkmask, actor)
        if key in failed:
            return None
        if all_done(cur, cidx, kkmask):
            if obs.drawn_tile is not None:
                return [("draw", 0)] if actor == 0 else None
            return []
        rest = decide(cur, cidx, kkmask, actor, drew=True)
        if rest is not None:
            return [("draw", actor)] + rest
        failed.add(key)
        return None

    def decide(cur, cidx, kkmask, actor, drew):
        # success mid-turn: hero holds the final draw (incl. post-rinshan)
        if (drew and actor == 0 and obs.drawn_tile is not None
                and all_done(cur, cidx, kkmask)):
            return []
        # (a) plain visible discard
        if cur[actor] < n[actor]:
            nxt = list(cur)
            nxt[actor] += 1
            rest = go(tuple(nxt), cidx, kkmask, (actor + 1) % 4)
            if rest is not None:
                return [("discard", actor, cur[actor])] + rest
        # (b) own-turn kans — Task 5
        # (c) ghost discard + call
        for o in range(4):
            if o == actor or cidx[o] >= ncre[o]:
                continue
            it = creation[o][cidx[o]]
            if it.kind in ("chi", "pon", "daiminkan") and it.target == actor:
                ncidx = list(cidx)
                ncidx[o] += 1
                pre = [("ghost", actor, it.pai, False), ("call", it)]
                if it.kind == "daiminkan":
                    rest = decide(cur, tuple(ncidx), kkmask, o, drew=True)
                    if rest is not None:
                        return pre + [("draw", o)] + rest
                else:
                    rest = decide(cur, tuple(ncidx), kkmask, o, drew=False)
                    if rest is not None:
                        return pre + rest
        return None

    return go((0, 0, 0, 0), (0, 0, 0, 0), 0, oya_rel)
```

（`go` 统一签名 `go(cur: tuple, cidx: tuple, kkmask: int, actor: int)`；上方代码即最终形态。）

`_emit` 替换为：

```python
def _emit(obs: ObservedState, ops: list, oya_rel: int):
    events: list = []
    haipai = list(obs.hero_hand)
    reach_count = [0] * 4
    just_called_hero = False
    for i, op in enumerate(ops):
        kind = op[0]
        if kind == "draw":
            r = op[1]
            if r != 0:
                events.append({"type": "tsumo", "actor": r, "pai": "?"})
                continue
            pai = obs.drawn_tile
            if i + 1 < len(ops):
                nxt = ops[i + 1]
                if nxt[0] == "discard":
                    pai = obs.rivers[0][nxt[2]].pai
                elif nxt[0] == "ghost" and nxt[1] == 0:
                    pai = nxt[2]
                elif nxt[0] == "ankan" and nxt[1].owner == 0:
                    pai = nxt[1].consumed[0]
                elif nxt[0] == "kakan" and nxt[1].owner == 0:
                    pai = nxt[1].pai
            events.append({"type": "tsumo", "actor": 0, "pai": pai})
        elif kind in ("discard", "ghost"):
            r = op[1]
            pai = obs.rivers[r][op[2]].pai if kind == "discard" else op[2]
            if r == 0:
                tsumogiri = not just_called_hero
                if just_called_hero:
                    haipai.append(pai)            # forced tedashi came from haipai
                just_called_hero = False
            else:
                tsumogiri = False                 # refined for riichi in Task 5
            events.append({"type": "dahai", "actor": r, "pai": pai,
                           "tsumogiri": tsumogiri})
        elif kind == "call":
            it = op[1]
            events.append({"type": it.kind, "actor": it.owner, "target": it.target,
                           "pai": it.pai, "consumed": list(it.consumed)})
            if it.owner == 0:
                haipai.extend(it.consumed)
                if it.kind in ("chi", "pon"):
                    just_called_hero = True
        # ("ankan"/"kakan") handled in Task 5
    return events, {"haipai": haipai, "reach_count": reach_count}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. python tests/test_reconstruct.py`
Expected: `test_reconstruct OK`（含 Task 3 全部旧用例回归）。

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/state/reconstruct.py tests/test_reconstruct.py
git commit -m "feat(state): reconstruction calls — ghost discards + backtracking DFS with memo"
```

---

### Task 5: 立直 + 暗杠/加杠 + dora 事件 + 供托/分数回推

**Files:**
- Modify: `majsoul_eye/state/reconstruct.py`
- Test: `tests/test_reconstruct.py`（追加用例）

**Interfaces:**
- Produces: 完整 `reconstruct`。ops 全集固定；`info` 增加 `reach_count`（每 rel 座 0/1）供 backfill（Task 3 已接线）。

- [ ] **Step 1: 追加失败测试**

```python
def test_opponent_riichi_reach_events_and_tsumogiri():
    o = _obs(rivers=[[ObservedRiverTile("9p"), ObservedRiverTile("1p")],
                     [ObservedRiverTile("E"), ObservedRiverTile("W", sideways=True),
                      ObservedRiverTile("N")],
                     [ObservedRiverTile("S"), ObservedRiverTile("S")],
                     [ObservedRiverTile("C"), ObservedRiverTile("F")]],
             reach=[False, True, False, False])
    r = _roundtrip(o)
    evs = r.events
    ri = next(i for i, e in enumerate(evs) if e["type"] == "reach")
    assert evs[ri]["actor"] == 1
    assert evs[ri + 1]["type"] == "dahai" and evs[ri + 1]["pai"] == "W"
    assert evs[ri + 2] == {"type": "reach_accepted", "actor": 1}
    # post-riichi discards by seat 1 are forced tsumogiri
    later = [e for e in evs[ri + 3:] if e["type"] == "dahai" and e["actor"] == 1]
    assert later and all(e["tsumogiri"] for e in later)
    # backfill: kyotaku defaults to observed riichi count -> start 0; score +1000
    sk = evs[1]
    assert sk["kyotaku"] == 0 and sk["scores"][1] == 26000


def test_hero_ankan_with_kandora():
    o = _obs(hero_hand=H13[:9] + ["C"],                     # 10 concealed
             melds=[[ObservedMeld("ankan", ["F", "F", "F", "F"])], [], [], []],
             rivers=[[ObservedRiverTile("9p")], [ObservedRiverTile("E")],
                     [ObservedRiverTile("S")], [ObservedRiverTile("W")]],
             dora_markers=["5s", "6s"])
    r = _roundtrip(o)
    evs = r.events
    ak = next(i for i, e in enumerate(evs) if e["type"] == "ankan")
    assert evs[ak]["consumed"] == ["F", "F", "F", "F"]
    assert evs[ak + 1] == {"type": "dora", "dora_marker": "6s"}
    # the 4th F was that turn's draw: haipai holds only 3 F
    assert evs[1]["tehais"][0].count("F") == 3


def test_kakan_pon_then_upgrade():
    # On-screen kakan = TWO events at different times: its pon (needs rel1's
    # ghost P) and the own-turn upgrade. The frame ends with hero holding the
    # rinshan draw, so the search must finish 'kakan -> rinshan tsumo'.
    o = _obs(hero_hand=H13[:10],
             melds=[[ObservedMeld("kakan", ["P", "P", "P", "P"],
                                  called_pai="P", added_pai="P", from_rel=1)],
                    [], [], []],
             rivers=[[ObservedRiverTile("9p")], [ObservedRiverTile("E")],
                     [ObservedRiverTile("S")], [ObservedRiverTile("W")]],
             drawn_tile="6s", dora_markers=["5s", "6s"])
    r = _roundtrip(o)
    kinds = [e["type"] for e in r.events]
    assert kinds.index("pon") < kinds.index("kakan")
    kk = next(e for e in r.events if e["type"] == "kakan")
    assert kk["pai"] == "P" and kk["consumed"] == ["P", "P", "P"]
    ki = r.events.index(kk)
    assert r.events[ki + 1] == {"type": "dora", "dora_marker": "6s"}
    assert r.events[-1] == {"type": "tsumo", "actor": 0, "pai": "6s"}
    # haipai: 10 concealed + pon's [P,P] + forced tedashi 9p = 13
    assert sorted(r.events[1]["tehais"][0]) == sorted(H13[:10] + ["P", "P", "9p"])


def test_riichi_tile_claimed_ghost_reach():
    # rel2's riichi declaration tile was ponned by rel1 -> rel2's NEXT discard
    # renders sideways. Search must bind reach to the ghost.
    o = _obs(hero_hand=H13[:13],
             melds=[[], [ObservedMeld("pon", ["W", "W", "W"], called_pai="W",
                                      from_rel=1)], [], []],
             rivers=[[ObservedRiverTile("9p"), ObservedRiverTile("1p")],
                     [ObservedRiverTile("E"), ObservedRiverTile("F")],
                     [ObservedRiverTile("S"), ObservedRiverTile("N", sideways=True)],
                     [ObservedRiverTile("C"), ObservedRiverTile("P")]],
             reach=[False, False, True, False])
    r = _roundtrip(o)
    evs = r.events
    ri = next(i for i, e in enumerate(evs) if e["type"] == "reach")
    assert evs[ri]["actor"] == 2
    nxt = evs[ri + 1]
    assert nxt["type"] == "dahai" and nxt["actor"] == 2
    # either binding is legal; the DECLARATION discard may be W (ghost) or N
    assert nxt["pai"] in ("W", "N")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. python tests/test_reconstruct.py`
Expected: 新用例失败（roundtrip sideways/dora 不匹配或 no legal order）。

- [ ] **Step 3: 完成实现**

`_search` 内改动（在 Task 4 版本上）：

```python
    # precompute per-seat sideways visible index (None if no riichi shown)
    side_idx = [next((i for i, t in enumerate(rivers[r]) if t.sideways), None)
                for r in range(4)]

    def declared(r, cur, rghost):
        if side_idx[r] is None:
            return bool(rghost >> r & 1)
        return cur[r] > side_idx[r] or bool(rghost >> r & 1)
```

状态与 memo key 全部加 `rghost`（int 位掩码，reach 绑定幽灵的座位）：
`go(cur, cidx, kkmask, rghost, actor)` / `decide(cur, cidx, kkmask, rghost, actor, drew)` /
`key = (cur, cidx, kkmask, rghost, actor)`；`all_done` 不变。`decide` 完整替换为：

```python
    def decide(cur, cidx, kkmask, rghost, actor, drew):
        if (drew and actor == 0 and obs.drawn_tile is not None
                and all_done(cur, cidx, kkmask)):
            return []
        # (a) plain visible discard
        if cur[actor] < n[actor]:
            nxt = list(cur)
            nxt[actor] += 1
            rest = go(tuple(nxt), cidx, kkmask, rghost, (actor + 1) % 4)
            if rest is not None:
                return [("discard", actor, cur[actor])] + rest
        # (b) own-turn kans (need a fresh draw; kakan forbidden after riichi)
        if drew:
            if cidx[actor] < ncre[actor] and creation[actor][cidx[actor]].kind == "ankan":
                it = creation[actor][cidx[actor]]
                ncidx = list(cidx)
                ncidx[actor] += 1
                rest = decide(cur, tuple(ncidx), kkmask, rghost, actor, drew=True)
                if rest is not None:
                    return [("ankan", it), ("draw", actor)] + rest
            if not declared(actor, cur, rghost):
                for ki, it in enumerate(kakans):
                    if kkmask >> ki & 1 or it.owner != actor:
                        continue
                    # its pon-part must already be triggered
                    pon_pos = next(j for j, c in enumerate(creation[actor])
                                   if c.mi == it.mi)
                    if cidx[actor] <= pon_pos:
                        continue
                    rest = decide(cur, cidx, kkmask | (1 << ki), rghost, actor, drew=True)
                    if rest is not None:
                        return [("kakan", it), ("draw", actor)] + rest
        # (c) ghost discard + call (a riichi'd owner cannot call)
        for o in range(4):
            if o == actor or cidx[o] >= ncre[o]:
                continue
            it = creation[o][cidx[o]]
            if it.kind not in ("chi", "pon", "daiminkan") or it.target != actor:
                continue
            if declared(o, cur, rghost):
                continue
            ncidx = list(cidx)
            ncidx[o] += 1
            variants = [False]
            if side_idx[actor] is not None and cur[actor] == side_idx[actor] \
                    and not declared(actor, cur, rghost):
                variants.append(True)          # bind the reach to this ghost
            for reach_here in variants:
                nrg = rghost | (1 << actor) if reach_here else rghost
                pre = [("ghost", actor, it.pai, reach_here), ("call", it)]
                if it.kind == "daiminkan":
                    rest = decide(cur, tuple(ncidx), kkmask, nrg, o, drew=True)
                    if rest is not None:
                        return pre + [("draw", o)] + rest
                else:
                    rest = decide(cur, tuple(ncidx), kkmask, nrg, o, drew=False)
                    if rest is not None:
                        return pre + rest
        return None
```

`_emit` 完整替换为（最终版）：

```python
def _emit(obs: ObservedState, ops: list, oya_rel: int):
    events: list = []
    haipai = list(obs.hero_hand)
    reach_count = [0] * 4
    declared = [False] * 4
    just_called_hero = False
    side_idx = [next((i for i, t in enumerate(r) if t.sideways), None)
                for r in obs.rivers]
    dora_next = 1                      # markers[0] went into start_kyoku

    def flip_dora():
        nonlocal dora_next
        if dora_next < len(obs.dora_markers):
            events.append({"type": "dora", "dora_marker": obs.dora_markers[dora_next]})
            dora_next += 1

    for i, op in enumerate(ops):
        kind = op[0]
        if kind == "draw":
            r = op[1]
            if r != 0:
                events.append({"type": "tsumo", "actor": r, "pai": "?"})
                continue
            pai = obs.drawn_tile
            if i + 1 < len(ops):
                nxt = ops[i + 1]
                if nxt[0] == "discard":
                    pai = obs.rivers[0][nxt[2]].pai
                elif nxt[0] == "ghost" and nxt[1] == 0:
                    pai = nxt[2]
                elif nxt[0] == "ankan" and nxt[1].owner == 0:
                    pai = nxt[1].consumed[0]
                elif nxt[0] == "kakan" and nxt[1].owner == 0:
                    pai = nxt[1].pai
            events.append({"type": "tsumo", "actor": 0, "pai": pai})
        elif kind in ("discard", "ghost"):
            r = op[1]
            if kind == "discard":
                idx, pai = op[2], obs.rivers[r][op[2]].pai
                is_reach = (idx == side_idx[r]) and not declared[r]
            else:
                pai = op[2]
                is_reach = op[3]
            if is_reach:
                events.append({"type": "reach", "actor": r})
            if r == 0:
                tsumogiri = not just_called_hero
                if just_called_hero:
                    haipai.append(pai)
                just_called_hero = False
            else:
                tsumogiri = declared[r]        # post-riichi discards are forced tsumogiri
            events.append({"type": "dahai", "actor": r, "pai": pai,
                           "tsumogiri": tsumogiri})
            if is_reach:
                events.append({"type": "reach_accepted", "actor": r})
                declared[r] = True
                reach_count[r] = 1
        elif kind == "call":
            it = op[1]
            events.append({"type": it.kind, "actor": it.owner, "target": it.target,
                           "pai": it.pai, "consumed": list(it.consumed)})
            if it.owner == 0:
                haipai.extend(it.consumed)
                if it.kind in ("chi", "pon"):
                    just_called_hero = True
            if it.kind == "daiminkan":
                flip_dora()
        elif kind == "ankan":
            it = op[1]
            events.append({"type": "ankan", "actor": it.owner,
                           "consumed": list(it.consumed)})
            if it.owner == 0:
                haipai.extend(it.consumed[1:])   # 4th copy was that turn's draw
            flip_dora()
        elif kind == "kakan":
            it = op[1]
            events.append({"type": "kakan", "actor": it.owner, "pai": it.pai,
                           "consumed": list(it.consumed)})
            flip_dora()                          # added tile was the draw: nothing to haipai
    return events, {"haipai": haipai, "reach_count": reach_count}
```

同时把 `reconstruct` 里 `_search`/`decide` 调用签名对齐（`rghost` 初始 0）。

- [ ] **Step 4: 跑全部状态侧测试**

Run: `PYTHONPATH=. python tests/test_reconstruct.py && PYTHONPATH=. python tests/test_observe.py && PYTHONPATH=. python tests/test_replay.py`
Expected: 三个 OK。

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/state/reconstruct.py tests/test_reconstruct.py
git commit -m "feat(state): reconstruction complete — riichi, kans, kan-dora, kyotaku/score backfill"
```

---

### Task 6: `recognize/assemble.py` — 河序反查

**Files:**
- Create: `majsoul_eye/recognize/assemble.py`
- Test: `tests/test_assemble.py`

**Interfaces:**
- Consumes: `recognize.detector.Detection`、`normalize.BoardRegion`、`annotate.pipeline`（`build_homographies`/`DISCARD_GRID`/`DISCARD_READ`/`DISCARD_ROW_OFFSETS`/`generate_discard_slots`）。
- Produces: `_fw_points(det, region, H_full) -> np.ndarray(4,2)`（fullwarp 角点）、`_assign_river(seat, items) -> (list[ObservedRiverTile], list[str])`，`items = [(det, corners_fw)]`。Task 8 组装时调用。

- [ ] **Step 1: 写失败测试（正向几何当 fixture，反向必须还原）**

```python
# tests/test_assemble.py
"""Detections -> ObservedState. Fixtures are generated by the FORWARD annotate
geometry (generate_discard_slots / generate_meld_boxes_v2 / HAND / dora_slot),
so assembly must invert them exactly — no detector, no images needed."""
import numpy as np

from majsoul_eye.annotate import pipeline as P
from majsoul_eye.normalize import BoardRegion
from majsoul_eye.recognize.detector import Detection
from majsoul_eye.tiles import NAME_TO_ID

H = P.build_homographies(1920, 1080)
REGION = BoardRegion(0, 0, 1920, 1080)


def _det_from_poly(poly_original, tile):
    xs = [p[0] for p in poly_original]; ys = [p[1] for p in poly_original]
    return Detection(xyxy=(min(xs), min(ys), max(xs), max(ys)), tile=tile,
                     cls=NAME_TO_ID[tile], score=0.9,
                     poly=tuple((float(x), float(y)) for x, y in poly_original))


def _river_dets(seat, pais, sideways_idx=None):
    river = [{"pai": p, "riichi": i == sideways_idx, "tsumogiri": False}
             for i, p in enumerate(pais)]
    slots = P.generate_discard_slots(seat, river, H["H_full_inv"],
                                     sideways_idx=sideways_idx)
    return [_det_from_poly(s["poly_original"], s["tile"]) for s in slots]


def test_river_order_recovered_all_seats():
    from majsoul_eye.recognize.assemble import _assign_river, _fw_points
    pais = ["1m", "9p", "E", "5sr", "2s", "7p", "C", "3m"]       # 6 + 2 rows
    for seat in range(4):
        dets = _river_dets(seat, pais)
        items = [(d, _fw_points(d, REGION, H["H_full"])) for d in dets]
        np.random.shuffle(items)                                  # order must not matter
        tiles, viol = _assign_river(seat, items)
        assert viol == []
        assert [t.pai for t in tiles] == pais
        assert not any(t.sideways for t in tiles)


def test_river_sideways_riichi_detected():
    from majsoul_eye.recognize.assemble import _assign_river, _fw_points
    pais = ["1m", "9p", "E", "2s", "7p", "C", "3m"]
    for seat in range(4):
        dets = _river_dets(seat, pais, sideways_idx=2)
        items = [(d, _fw_points(d, REGION, H["H_full"])) for d in dets]
        tiles, viol = _assign_river(seat, items)
        assert viol == []
        assert [t.sideways for t in tiles] == [i == 2 for i in range(len(pais))]


def test_river_hole_flags_violation():
    from majsoul_eye.recognize.assemble import _assign_river, _fw_points
    dets = _river_dets(0, ["1m", "9p", "E", "2s", "7p", "C", "3m"])
    del dets[3]                                                   # hole in row 0
    items = [(d, _fw_points(d, REGION, H["H_full"])) for d in dets]
    _, viol = _assign_river(0, items)
    assert viol


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_assemble OK")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. python tests/test_assemble.py`
Expected: `ModuleNotFoundError: No module named 'majsoul_eye.recognize.assemble'`

- [ ] **Step 3: 实现**

```python
# majsoul_eye/recognize/assemble.py
"""Detections -> ObservedState (spec 2026-07-05 §3.2).

Runs the calibrated annotate/pipeline geometry BACKWARD: detection centers are
mapped original->canonical(1920x1080)->fullwarp, then matched to the discard
grid / meld strip. Akagi-free (annotate.pipeline is pure geometry; capture/ is
never imported)."""
from __future__ import annotations

import numpy as np

from majsoul_eye.annotate import pipeline as P
from majsoul_eye.normalize import BoardRegion
from majsoul_eye.state.observe import ObservedRiverTile

CANON_W, CANON_H = 1920, 1080


def _fw_points(det, region: BoardRegion, H_full) -> np.ndarray:
    """Detection corners (poly if OBB else xyxy box) -> fullwarp, via canonical px."""
    if det.poly:
        pts = np.float32(det.poly)
    else:
        x0, y0, x1, y1 = det.xyxy
        pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    nb = [region.px_to_norm_box(float(x), float(y), float(x), float(y)) for x, y in pts]
    canon = np.float32([[b.x0 * CANON_W, b.y0 * CANON_H] for b in nb])
    return P.original_to_fullwarp(canon, H_full)


def _river_frame(seat: int):
    g = P.DISCARD_GRID[seat]
    rd = P.DISCARD_READ[seat]
    o = np.array(g["o"], float)
    dcol = np.array(g["dcol"], float)
    drow = np.array(g["drow"], float)
    disc0 = o + (P.DISCARD_COLS - 1) * dcol if rd["disc0_at_col5"] else o.copy()
    colv = rd["colsign"] * dcol
    colu = colv / np.linalg.norm(colv)
    rowu = rd["rowsign"] * drow / np.linalg.norm(drow)
    return disc0, colu, rowu, float(np.linalg.norm(dcol))


def _assign_river(seat: int, items):
    """items = [(det, corners_fw)] -> (ordered ObservedRiverTiles, violations).

    Row = nearest DISCARD_ROW_OFFSETS entry; order within a row = along-column
    projection (handles the riichi extra-shift and the >18 overflow, since only
    ORDER matters). Sideways = footprint longer along the column axis."""
    disc0, colu, rowu, col_pitch = _river_frame(seat)
    offs = P.DISCARD_ROW_OFFSETS[seat]
    row_pitch = offs[1] - offs[0]
    rows: dict[int, list] = {0: [], 1: [], 2: []}
    viol: list[str] = []
    for det, pts in items:
        c = pts.mean(axis=0)
        v = float(np.dot(c - disc0, rowu))
        r = int(np.argmin([abs(v - x) for x in offs]))
        if abs(v - offs[r]) > 0.5 * row_pitch:
            viol.append(f"seat{seat} river det off-grid (row residual {v - offs[r]:.0f}px)")
            continue
        u = float(np.dot(c - disc0, colu))
        ext_col = float(np.ptp(pts @ colu))
        ext_row = float(np.ptp(pts @ rowu))
        rows[r].append((u, ObservedRiverTile(det.tile, sideways=ext_col > ext_row)))
    out: list[ObservedRiverTile] = []
    for r in (0, 1, 2):
        rows[r].sort(key=lambda t: t[0])
        if rows[r] and r > 0 and len(rows[r - 1]) != P.DISCARD_COLS:
            viol.append(f"seat{seat} river row{r} occupied but row{r-1} not full")
        if r < 2 and len(rows[r]) > P.DISCARD_COLS:
            viol.append(f"seat{seat} river row{r} has {len(rows[r])}>6 tiles")
        out.extend(t for _, t in rows[r])
    return out, viol
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. python tests/test_assemble.py`
Expected: `test_assemble OK`

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/recognize/assemble.py tests/test_assemble.py
git commit -m "feat(recognize): river-order inverse of the calibrated discard grid"
```

---

### Task 7: 副露条反查（模式匹配 `meld_display_cells`）

**Files:**
- Modify: `majsoul_eye/recognize/assemble.py`
- Test: `tests/test_assemble.py`（追加）

**Interfaces:**
- Produces: `_parse_melds(seat, items) -> (list[ObservedMeld], list[str])`；`items` 同 Task 6。逆向唯一事实源 = 正向 `P.meld_display_cells`：对候选 `(type, called, added, from_rel)` 假设生成 cells 并与观测 cell 串匹配（DRY——渲染规则永不复制第二份）。

- [ ] **Step 1: 追加失败测试**

```python
def _meld_dets(seat, melds_gt):
    boxes = P.generate_meld_boxes_v2(seat, melds_gt, H["H_full_inv"])
    return [_det_from_poly(b["poly_original"], b["tile"]) for b in boxes]


def _gt(type_, tiles, called="", added="", from_seat_rel=0, seat=0):
    return {"type": type_, "tiles": tiles, "called_pai": called,
            "added_pai": added, "from_seat": (seat + from_seat_rel) % 4}


def test_meld_parse_all_kinds_all_seats():
    from majsoul_eye.recognize.assemble import _fw_points, _parse_melds
    for seat in range(4):
        gt = [_gt("pon", ["P", "P", "P"], called="P", from_seat_rel=2, seat=seat),
              _gt("chi", ["4s", "5s", "6s"], called="5s", from_seat_rel=3, seat=seat)]
        items = [(d, _fw_points(d, REGION, H["H_full"])) for d in _meld_dets(seat, gt)]
        melds, viol = _parse_melds(seat, items)
        assert viol == []
        assert [(m.type, m.from_rel, m.called_pai) for m in melds] == \
               [("pon", 2, "P"), ("chi", 3, "5s")]
        assert sorted(melds[0].tiles) == ["P", "P", "P"]


def test_meld_parse_kans():
    from majsoul_eye.recognize.assemble import _fw_points, _parse_melds
    seat = 1
    gt = [_gt("ankan", ["5m", "5m", "5m", "5mr"], seat=seat),
          _gt("daiminkan", ["C", "C", "C", "C"], called="C", from_seat_rel=1, seat=seat),
          _gt("kakan", ["W", "W", "W", "W"], called="W", added="W",
              from_seat_rel=2, seat=seat)]
    items = [(d, _fw_points(d, REGION, H["H_full"])) for d in _meld_dets(seat, gt)]
    melds, viol = _parse_melds(seat, items)
    assert viol == []
    assert [m.type for m in melds] == ["ankan", "daiminkan", "kakan"]
    # ankan of fives MUST contain the red (only 4 copies exist incl. the red)
    assert sorted(melds[0].tiles) == ["5m", "5m", "5m", "5mr"]
    assert melds[1].from_rel == 1
    assert melds[2].from_rel == 2 and melds[2].added_pai == "W"


def test_meld_parse_flags_garbage():
    from majsoul_eye.recognize.assemble import _fw_points, _parse_melds
    gt = [_gt("pon", ["P", "P", "P"], called="P", from_seat_rel=2, seat=0)]
    dets = _meld_dets(0, gt)[:-1]                     # drop one tile -> unparsable
    items = [(d, _fw_points(d, REGION, H["H_full"])) for d in dets]
    _, viol = _parse_melds(0, items)
    assert viol
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. python tests/test_assemble.py`
Expected: `ImportError: cannot import name '_parse_melds'`

- [ ] **Step 3: 实现（追加到 assemble.py）**

```python
from majsoul_eye.state.observe import ObservedMeld
from majsoul_eye.tiles import red_to_normal


def _strip_cells(seat: int, items):
    """Sort meld-zone detections into display cells walking from the corner.
    Returns cells = [{label, sideways, stacked: [label]}] in CORNER order."""
    cfg = P.MELD_STRIP2[seat]
    corner = np.array(cfg["corner"], float)
    along = np.array(cfg["along"], float)
    cross = np.array(cfg["cross"], float)
    d = cfg["d"]
    raw = []
    for det, pts in items:
        c = pts.mean(axis=0)
        a = float(np.dot(c - corner, along))
        cr = float(np.dot(c - corner, cross))
        ext_a = float(np.ptp(pts @ along))
        ext_c = float(np.ptp(pts @ cross))
        raw.append({"a": a, "c": cr, "label": det.tile,
                    "sideways": ext_a > ext_c, "stacked_on": None})
    raw.sort(key=lambda x: x["a"])
    cells, i = [], 0
    while i < len(raw):
        cell = {"label": raw[i]["label"], "sideways": raw[i]["sideways"], "stacked": []}
        j = i + 1
        # a kakan's added tile shares the same along-slot, offset across (c ~ d..2d)
        while j < len(raw) and abs(raw[j]["a"] - raw[i]["a"]) < 0.4 * cfg["w"]:
            top = raw[j] if raw[j]["c"] > raw[i]["c"] else raw[i]
            base = raw[i] if top is raw[j] else raw[j]
            cell = {"label": base["label"], "sideways": True, "stacked": [top["label"]]}
            j += 1
        cells.append(cell)
        i = j
    return cells


def _hypotheses(group):
    """Candidate (type, tiles, called, added, from_rel) for a cell group."""
    labels = [c["label"] for c in group]
    out = []
    if len(group) == 4 and labels.count("back") == 2:
        face = next(l for l in labels if l != "back")
        tiles = [face] * 4
        base = red_to_normal(face)
        if base[0] == "5":                     # 4 copies of a five must include the red
            tiles = [base + "r"] + [base] * 3
        out.append(("ankan", tiles, "", "", 0))
        return out
    stacked = next((c for c in group if c["stacked"]), None)
    if stacked is not None:
        called, added = stacked["label"], stacked["stacked"][0]
        tiles = [c["label"] for c in group] + [added]
        for rel in (1, 2, 3):
            out.append(("kakan", tiles, called, added, rel))
        return out
    side = [c for c in group if c["sideways"]]
    if len(side) != 1:
        return []
    called = side[0]["label"]
    norm = [red_to_normal(l) for l in labels]
    if len(group) == 4:
        if len(set(norm)) == 1:
            for rel in (1, 2, 3):
                out.append(("daiminkan", labels, called, "", rel))
    elif len(group) == 3:
        if len(set(norm)) == 1:
            for rel in (1, 2, 3):
                out.append(("pon", labels, called, "", rel))
        elif all(len(x) == 2 and x[0].isdigit() for x in norm) \
                and len({x[1] for x in norm}) == 1:        # one suit only
            ranks = sorted(int(x[0]) for x in norm)
            if ranks[1] - ranks[0] == 1 and ranks[2] - ranks[1] == 1:
                out.append(("chi", labels, called, "", 3))
    return out


def _match_group(seat, group):
    """Find the hypothesis whose FORWARD rendering equals the observed cells."""
    obs_cells = [(c["label"], c["sideways"], tuple(c["stacked"])) for c in group]
    for type_, tiles, called, added, rel in _hypotheses(group):
        m = {"type": type_, "tiles": sorted(tiles), "from_seat": (seat + rel) % 4,
             "called_pai": called, "added_pai": added}
        cells = P.meld_display_cells(m, seat)
        if P.MELD_WITHIN_REVERSED:
            cells = list(reversed(cells))
        want = [(c["label"], bool(c["sideways"]), tuple(c.get("stacked", [])))
                for c in cells]
        if want == obs_cells:
            return ObservedMeld(type_, sorted(tiles), called, added, rel)
    return None


def _parse_melds(seat: int, items):
    """Meld-zone detections -> (melds in SCREEN order oldest-first, violations).

    Whole-strip recursive parse: a group size that matches locally but leaves an
    unparsable tail is backtracked (e.g. pon KKK followed by a meld starting
    with K would otherwise be swallowed as a fake daiminkan)."""
    cells = _strip_cells(seat, items)

    def parse(i: int):
        if i == len(cells):
            return []
        for size in (4, 3):
            if i + size <= len(cells):
                got = _match_group(seat, cells[i:i + size])
                if got is not None:
                    rest = parse(i + size)
                    if rest is not None:
                        return [got] + rest
        return None

    melds = parse(0)
    if melds is None:
        return [], [f"seat{seat} meld strip unparsable ({len(cells)} cells)"]
    return melds, []
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. python tests/test_assemble.py`
Expected: `test_assemble OK`

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/recognize/assemble.py tests/test_assemble.py
git commit -m "feat(recognize): meld-strip inverse via forward meld_display_cells pattern match"
```

---

### Task 8: `assemble()` 顶层 — 分区路由 + 手牌/dora + 整帧往返

**Files:**
- Modify: `majsoul_eye/recognize/assemble.py`
- Test: `tests/test_assemble.py`（追加）

**Interfaces:**
- Produces: `assemble(dets: list[Detection], region: BoardRegion) -> ObservedState`——Task 9 评测唯一入口。路由规则：先 hand（画面下带 + 检测框高 > 0.11，区分 hero 副露的小牌）与 dora（`DORA_STRIP` 含中心点），余下过 fullwarp 按河/副露就近归座；`back` 检测只允许归入副露区（暗杠 back/面/面/back 需要它），落在 cutoff 外的（对手手牌排、牌山）静默丢弃不记 violation（`concealed_counts` 本期恒 None，schema 槽位保留）。

- [ ] **Step 1: 追加失败测试（整帧合成 → 装配 → 与 GT 投影全等）**

```python
def _hand_dets(pais, drawn=None):
    from majsoul_eye.coords import HAND
    dets = []
    for i, p in enumerate(pais):
        b = HAND.slot_box(i)
        dets.append(_det_from_poly(
            [[b.x0 * 1920, b.y0 * 1080], [b.x1 * 1920, b.y0 * 1080],
             [b.x1 * 1920, b.y1 * 1080], [b.x0 * 1920, b.y1 * 1080]], p))
    if drawn:
        b = HAND.slot_box(len(pais), is_tsumo=True)
        dets.append(_det_from_poly(
            [[b.x0 * 1920, b.y0 * 1080], [b.x1 * 1920, b.y0 * 1080],
             [b.x1 * 1920, b.y1 * 1080], [b.x0 * 1920, b.y1 * 1080]], drawn))
    return dets


def _dora_dets(markers):
    from majsoul_eye.coords import dora_slot
    dets = []
    for i, p in enumerate(markers):
        b = dora_slot(i)
        dets.append(_det_from_poly(
            [[b.x0 * 1920, b.y0 * 1080], [b.x1 * 1920, b.y0 * 1080],
             [b.x1 * 1920, b.y1 * 1080], [b.x0 * 1920, b.y1 * 1080]], p))
    return dets


def test_full_frame_roundtrip():
    from majsoul_eye.recognize.assemble import assemble
    hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p"]
    dets = _hand_dets(hand, drawn="4p")
    dets += _dora_dets(["5s"])
    dets += _river_dets(0, ["9p", "1s"])
    dets += _river_dets(1, ["E", "W", "N"], sideways_idx=1)
    dets += _river_dets(2, ["S"])
    dets += _river_dets(3, ["C", "F"])
    dets += _meld_dets(0, [_gt("pon", ["P", "P", "P"], called="P",
                               from_seat_rel=2, seat=0)])
    dets += _meld_dets(2, [_gt("ankan", ["F", "F", "F", "F"], seat=2)])
    # a stray opponent-hand back far from every zone must be dropped silently
    dets.append(_det_from_poly([[940, 510], [980, 510], [980, 570], [940, 570]],
                               "back"))
    o = assemble(dets, REGION)
    assert o.violations == []
    assert o.hero_hand == hand and o.drawn_tile == "4p"
    assert o.dora_markers == ["5s"]
    assert [t.pai for t in o.rivers[1]] == ["E", "W", "N"]
    assert o.rivers[1][1].sideways and o.reach == [False, True, False, False]
    assert o.melds[0][0].type == "pon" and o.melds[0][0].from_rel == 2
    assert o.melds[2][0].type == "ankan"
    assert o.zone_confidence["hand"] > 0


def test_full_frame_feeds_reconstruct():
    from majsoul_eye.recognize.assemble import assemble
    from majsoul_eye.state.reconstruct import reconstruct
    hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]
    dets = _hand_dets(hand) + _dora_dets(["5s"])
    dets += _river_dets(0, ["9p"]) + _river_dets(1, ["E"]) + _river_dets(2, ["S"])
    o = assemble(dets, REGION)
    assert o.violations == []
    r = reconstruct(o)
    assert r.ok, r.reason
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. python tests/test_assemble.py`
Expected: `ImportError: cannot import name 'assemble'`

- [ ] **Step 3: 实现顶层（追加到 assemble.py）**

```python
from majsoul_eye.coords import DORA_STRIP, HAND
from majsoul_eye.state.observe import ObservedState, check_observed

HAND_MIN_H = 0.11            # hand tiles are ~0.141 canon-high; hero meld tiles ~0.083


def assemble(dets, region: BoardRegion) -> ObservedState:
    """One frame's detections -> ObservedState. HUD fields stay None (their
    readers are the 2026-07-04 spec). 'back' detections only ever route to MELD
    zones (ankan renders back/face/face/back); opponents' concealed rows sit off
    the felt plane, land outside every calibrated zone after the homography and
    are dropped silently (concealed_counts stays None — cross-check only)."""
    o = ObservedState()
    Hs = P.build_homographies(CANON_W, CANON_H)
    hand_cand, dora_cand, table = [], [], []
    conf: dict[str, list] = {}

    def note(zone, det):
        conf.setdefault(zone, []).append(det.score)

    for det in dets:
        x0, y0, x1, y1 = det.xyxy
        nb = region.px_to_norm_box(x0, y0, x1, y1)
        if DORA_STRIP.x0 <= nb.cx <= DORA_STRIP.x1 and \
                DORA_STRIP.y0 <= nb.cy <= DORA_STRIP.y1:
            if det.tile != "back":
                dora_cand.append((nb.x0, det))
                note("dora", det)
            continue
        if det.tile != "back" and nb.h >= HAND_MIN_H and nb.cy >= HAND.y0 - 0.02:
            hand_cand.append((nb.x0, nb, det))
            note("hand", det)
            continue
        table.append((det, _fw_points(det, region, Hs["H_full"])))

    # hand + drawn (gap of >= ~half a slot before the last tile)
    hand_cand.sort(key=lambda t: t[0])
    o.hero_hand = [d.tile for _, _, d in hand_cand]
    if len(hand_cand) >= 2:
        gap = hand_cand[-1][0] - hand_cand[-2][0]
        if gap > HAND.slot_w + 0.5 * HAND.tsumo_gap:
            o.drawn_tile = o.hero_hand.pop()
    o.dora_markers = [d.tile for _, d in sorted(dora_cand, key=lambda t: t[0])]

    # route table detections to the nearest seat zone (river vs meld)
    per_river: list[list] = [[] for _ in range(4)]
    per_meld: list[list] = [[] for _ in range(4)]
    for det, pts in table:
        c = pts.mean(axis=0)
        best = None                                    # (dist, kind, seat)
        for seat in range(4):
            disc0, colu, rowu, col_pitch = _river_frame(seat)
            offs = P.DISCARD_ROW_OFFSETS[seat]
            u = float(np.dot(c - disc0, colu))
            v = float(np.dot(c - disc0, rowu))
            du = max(0.0, -u, u - 10 * col_pitch)
            dv = min(abs(v - x) for x in offs)
            d_river = float(np.hypot(du, dv))
            cfg = P.MELD_STRIP2[seat]
            a = float(np.dot(c - np.array(cfg["corner"]), np.array(cfg["along"])))
            cr = float(np.dot(c - np.array(cfg["corner"]), np.array(cfg["cross"])))
            da = max(0.0, -a, a - 16 * cfg["w"])
            dc = max(0.0, -cr, cr - 2.2 * cfg["d"])
            d_meld = float(np.hypot(da, dc))
            kinds = (("meld", d_meld),) if det.tile == "back" else \
                    (("river", d_river), ("meld", d_meld))
            for kind, dist in kinds:
                if best is None or dist < best[0]:
                    best = (dist, kind, seat)
        if best[0] > 60.0:
            if det.tile != "back":     # opponents' concealed rows are expected strays
                o.violations.append(
                    f"stray detection {det.tile} ({best[0]:.0f}px off-zone)")
            continue
        (per_river if best[1] == "river" else per_meld)[best[2]].append((det, pts))
        note(f"{best[1]}{best[2]}", det)

    for seat in range(4):
        o.rivers[seat], v1 = _assign_river(seat, per_river[seat])
        melds, v2 = _parse_melds(seat, per_meld[seat])
        o.melds[seat] = melds
        o.violations.extend(v1 + v2)
        o.reach[seat] = any(t.sideways for t in o.rivers[seat])
    o.zone_confidence = {z: min(s) for z, s in conf.items()}
    o.violations.extend(check_observed(o))
    return o
```

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

Run: `PYTHONPATH=. python tests/test_assemble.py`
Expected: `test_assemble OK`
Run（bash，全部测试）: `for t in tests/test_*.py; do PYTHONPATH=. python "$t" || break; done`
Expected: 全部 OK（无回归）。

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/recognize/assemble.py tests/test_assemble.py
git commit -m "feat(recognize): assemble() — zone routing + hand/dora + full-frame ObservedState"
```

---

### Task 9: 评测 harness + 文档同步

**Files:**
- Create: `scripts/eval/__init__.py`（空文件）、`scripts/eval/eval_reconstruction.py`
- Modify: `docs/PIPELINE.md`（§4 工具清单）、`docs/STATUS.md`（新条目）

**Interfaces:**
- Consumes: `capture.gtframes.build_seq_state/load_frames`、`paths.frames_dir_for`、`observed_from_board`、`reconstruct`、`assemble`、`TileDetector`、`replay.is_deal_window`。
- Produces: CLI `scripts/eval/eval_reconstruction.py`（QA 工具；oracle 无 GPU 依赖，assemble 需权重，engine 需 `--engine-cmd`）。

- [ ] **Step 1: 实现脚本（QA 工具无单测；oracle 级即它自己的批量断言）**

```python
# scripts/eval/eval_reconstruction.py
"""Three-layer GT eval for board reconstruction (spec 2026-07-05 §6). QA tool
(PIPELINE.md §4) — not a pipeline stage.

  oracle:   GT BoardState -> perfect ObservedState -> reconstruct -> Replayer
            round-trip must project back identically (isolates the algorithm).
  assemble: real frame -> detector -> assemble vs GT projection, per zone.
  engine:   true GTRecord mjai prefix vs reconstructed sequence -> an mjai bot
            subprocess (--engine-cmd, stdin/stdout JSON lines); compare the
            final reaction (decision agreement).

Usage:
  PYTHONPATH=. python scripts/eval/eval_reconstruction.py --captures \
      captures/raw/ai_session/run_8 --level oracle
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

from majsoul_eye.capture.gtframes import build_seq_state, load_frames
from majsoul_eye import paths
from majsoul_eye.state.observe import check_observed, observed_from_board
from majsoul_eye.state.reconstruct import reconstruct
from majsoul_eye.state.replay import Replayer, is_deal_window


def find_captures(roots):
    out = []
    for root in roots:
        if os.path.isfile(root):
            out.append(root)
        else:
            out += sorted(glob.glob(os.path.join(root, "**", "game*.jsonl"),
                                    recursive=True))
    return [p for p in out if "frames" not in os.path.basename(p)]


def obs_key(o):
    return {
        "rivers": [[(t.pai, t.sideways) for t in r] for r in o.rivers],
        "melds": [[(m.type, m.from_rel, tuple(sorted(m.tiles))) for m in ms]
                  for ms in o.melds],
        "hand": sorted(o.hero_hand), "drawn": o.drawn_tile,
        "dora": list(o.dora_markers), "reach": list(o.reach),
    }


def diff_zones(a, b):
    ka, kb = obs_key(a), obs_key(b)
    return [z for z in ka if ka[z] != kb[z]]


def run_oracle(states, report):
    for seq, st in states.items():
        if not st.in_round or is_deal_window(st):
            continue
        obs = observed_from_board(st)
        if check_observed(obs):
            report["skipped_violations"] += 1
            continue
        r = reconstruct(obs)
        if not r.ok:
            report["fail"].append({"seq": seq, "reason": r.reason})
            continue
        rp = Replayer()
        for ev in r.events:
            rp.apply(ev)
        d = diff_zones(observed_from_board(rp.state, include_hud=False),
                       observed_from_board(st, include_hud=False))
        if d:
            report["mismatch"].append({"seq": seq, "zones": d})
        else:
            report["ok"] += 1


def run_assemble(cap, states, report, weights, device):
    import cv2
    from majsoul_eye.normalize import locate_fullscreen
    from majsoul_eye.recognize.assemble import assemble
    from majsoul_eye.recognize.detector import TileDetector
    det = TileDetector(weights, device=device)
    frames = load_frames(paths.frames_dir_for(cap))
    for seq, st in states.items():
        if seq not in frames or not st.in_round or is_deal_window(st):
            continue
        img = cv2.imread(frames[seq])
        if img is None:
            continue
        obs = assemble(det.predict(img), locate_fullscreen(img))
        gt = observed_from_board(st, include_hud=False)
        if obs.violations:
            report["rejected"] += 1
            continue
        d = diff_zones(obs, gt)
        report["frames"] += 1
        if not d:
            report["ok"] += 1
        for z in d:
            report["zone_errors"][z] = report["zone_errors"].get(z, 0) + 1


def ask_engine(cmd, events, timeout=60):
    inp = "\n".join(json.dumps(e) for e in events) + "\n"
    p = subprocess.run(cmd, input=inp, capture_output=True, text=True,
                       timeout=timeout, shell=True)
    lines = [l for l in p.stdout.strip().splitlines() if l.strip().startswith("{")]
    return json.loads(lines[-1]) if lines else None


def run_engine(cap, states, report, engine_cmd, sample):
    from majsoul_eye.capture.schema import read_records
    truth, taken = [], 0
    for rec in read_records(cap):
        if not rec.syncing:
            truth.extend(rec.mjai or [])
        if rec.seq not in states or taken >= sample:
            continue
        st = states[rec.seq]
        if not st.in_round or is_deal_window(st) or st.drawn_tile is None:
            continue                       # decision points = hero holds a draw
        taken += 1
        r = reconstruct(observed_from_board(st))
        if not r.ok:
            report["fail"] += 1
            continue
        a = ask_engine(engine_cmd, [e for e in truth])
        b = ask_engine(engine_cmd, r.events)
        if a is None or b is None:
            report["engine_error"] += 1
        elif a.get("type") == b.get("type") and a.get("pai") == b.get("pai"):
            report["agree"] += 1
        else:
            report["disagree"].append({"seq": rec.seq, "true": a, "recon": b})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captures", nargs="+", required=True)
    ap.add_argument("--level", choices=["oracle", "assemble", "engine"],
                    default="oracle")
    ap.add_argument("--weights", default="majsoul_eye/recognize/tile_detector.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--engine-cmd", default=None)
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    caps = find_captures(args.captures)
    if not caps:
        sys.exit("no captures found")
    if args.level == "engine" and not args.engine_cmd:
        sys.exit("--level engine requires --engine-cmd (an mjai bot: events on "
                 "stdin as JSON lines, reactions on stdout)")

    total = {"ok": 0, "fail": [], "mismatch": [], "skipped_violations": 0,
             "frames": 0, "rejected": 0, "zone_errors": {},
             "agree": 0, "disagree": [], "engine_error": 0}
    for cap in caps:
        states = build_seq_state(cap)
        if args.level == "oracle":
            run_oracle(states, total)
        elif args.level == "assemble":
            run_assemble(cap, states, total, args.weights, args.device)
        else:
            run_engine(cap, states, total, args.engine_cmd, args.sample)

    if args.level == "oracle":
        n = total["ok"] + len(total["fail"]) + len(total["mismatch"])
        print(f"[oracle] {total['ok']}/{n} ok, {len(total['fail'])} infeasible, "
              f"{len(total['mismatch'])} mismatched, "
              f"{total['skipped_violations']} skipped")
        for f in total["fail"][:10]:
            print("  FAIL", f)
        for m in total["mismatch"][:10]:
            print("  DIFF", m)
    elif args.level == "assemble":
        print(f"[assemble] {total['ok']}/{total['frames']} frames fully match, "
              f"{total['rejected']} rejected; zone errors: {total['zone_errors']}")
    else:
        print(f"[engine] agree {total['agree']}, "
              f"disagree {len(total['disagree'])}, errors {total['engine_error']}")
        for d in total["disagree"][:10]:
            print("  ", d)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(total, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 在真实捕获上跑 oracle 级**

Run: `PYTHONPATH=. python scripts/eval/eval_reconstruction.py --captures captures/raw/ai_session --level oracle --out out/eval/recon_oracle.json`
Expected: 打印 `[oracle] N/M ok ...`；**验收 = 非跳过帧成功率 ≥99%**。若有 FAIL/DIFF：逐个调查（多为 reconstruct 边角 bug——先修再继续；把复现帧固化成 `tests/test_reconstruct.py` 新用例）。

- [ ] **Step 3: 抽样跑 assemble 级（慢，验证接线即可）**

Run: `PYTHONPATH=. python scripts/eval/eval_reconstruction.py --captures captures/raw/ai_session/run_8 --level assemble --device cpu`
Expected: 正常打印 per-zone 报表（本步只验证不崩溃 + 数字合理；阈值校准归 spec §6 的后续实测）。

- [ ] **Step 4: 文档同步**

`docs/PIPELINE.md` §4 的 QA/一次性工具清单追加一行（按该节现有列表格式）：

```
- `scripts/eval/eval_reconstruction.py` — QA 工具：局面复原三层评测（oracle 重建往返 /
  assemble 装配对比 GT 投影 / engine 决策一致率，--engine-cmd 喂任意 mjai bot）。
  spec: docs/superpowers/specs/2026-07-05-board-reconstruction-design.md
```

`docs/STATUS.md` 文末追加条目（沿用现有 §1.N 编号递增）：

```
### §1.29 局面复原 M1（2026-07-XX）
单帧复原落地：`state/observe.py`（ObservedState + check_observed + GT 投影）、
`recognize/assemble.py`（检测→状态，反用 annotate 几何）、`state/reconstruct.py`
（回合模拟+回溯 → 合法 hero 视角 mjai 序列）、`scripts/eval/eval_reconstruction.py`
（三层评测）。oracle 成功率 <填实测值>；assemble 整帧一致率 <填实测值>。
数据管线无变更。spec/plan: docs/superpowers/{specs,plans}/2026-07-05-board-reconstruction*.
```

- [ ] **Step 5: 全量测试 + Commit**

Run（bash）: `for t in tests/test_*.py; do PYTHONPATH=. python "$t" || break; done`
Expected: 全部 OK。

```bash
git add scripts/eval/ docs/PIPELINE.md docs/STATUS.md
git commit -m "feat(eval): 3-layer board-reconstruction eval harness + docs sync"
```

---

## Self-Review 记录

- **Spec 覆盖**：§3.1→Task 1/2；§3.2→Task 6/7/8；§4→Task 3/4/5；§6→Task 9；§7 标定任务并入 Task 7（from_rel 经 `meld_display_cells` 正向模式匹配，无需独立标定）与 Task 8（摸牌槽阈值用 `HAND.tsumo_gap`）；§8→Task 9 Step 4。engine 级以 `--engine-cmd` 子进程契约实现（mjai bot stdin/stdout），不绑定 mycv 内部 API。
- **类型一致性**：`_Item`/ops 元组形状在 Task 4 定义、Task 5 扩展并保持；`_fw_points`/`_assign_river`/`_parse_melds` 签名在 Task 6/7 定义、Task 8 调用一致；`observed_from_board(include_hud=)` 贯穿 Task 2/3/9。
- **已知实测点**（非 placeholder，执行时按 Expected 校验）：Task 9 Step 2 的 oracle ≥99% 是验收门槛；Task 8 的 `HAND_MIN_H`/60px 路由 cutoff 若在真实帧上误路由，调参并把失败帧固化为测试。
