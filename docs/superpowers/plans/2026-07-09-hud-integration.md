# 单帧 HUD 集成 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把已训练的 HudReader + 56 类检测器接进单帧识别链，填满 `ObservedState` 的 HUD 字段，`reconstruct` 用真实值替代默认值，加 HUD×视觉交叉校验与立直棒权威化。

**Architecture:** 方案 A（spec `docs/superpowers/specs/2026-07-09-hud-integration-design.md`）：`assemble(dets, region, frame_bgr=None, hud_reader=None)` 可选接入 HUD；`assemble_hud` 改吃 `Detection` 对象；交叉校验进 `check_observed`（纯数据，oracle 免费复验）；`reconstruct` 把立直棒作为权威 reach 信号（ghost 绑定 + 终局约束）。宽屏（`region.ox > 0`）本轮跳过 HUD 填充。

**Tech Stack:** Python (conda `auto` env), PyTorch (HudReader), ultralytics YOLO (detector), 纯脚本测试（无 pytest 依赖）。

## Global Constraints

- 一切命令从仓库根跑，`PYTHONPATH=.`；用户 shell 写 `python`，Claude 的 Bash 工具用 `C:/Users/zsx/miniforge3/envs/auto/python.exe`。
- 测试是普通脚本：`PYTHONPATH=. python tests/test_X.py`，断言 + 末尾 `print("... OK")`。
- `recognize/` 必须保持 Akagi-free（不 import `capture/`）。
- 38 类顺序冻结；HUD 类是其后缀（`hud.DET_NAMES` 56 类）。
- 本轮纯 runtime 侧改动，不 stale `out/`/`datasets/`；收尾必须同步 `docs/PIPELINE.md` + STATUS 新条目（§1.54）。
- 56 类检测权重：`weights/detector/tile_detector_obb_20260709_055509.pt`；HudReader 权重：`majsoul_eye/recognize/hud_reader.pt`（已在工作区，Task 1 提交）。

---

### Task 1: 提交 §1.53 遗留（housekeeping）

工作区里有上一会话（HudReader 训练，STATUS §1.53）的未提交产物，先单独入库，保持后续 diff 干净。

**Files:**
- Commit（不修改）: `scripts/train/train_hudreader.py`, `tests/test_hudreader.py`, `docs/PIPELINE.md`, `majsoul_eye/recognize/hud_reader.pt`

- [ ] **Step 1: 验证遗留测试通过**

Run: `PYTHONPATH=. python tests/test_hudreader.py`
Expected: 末尾 OK 行，退出码 0。

- [ ] **Step 2: 查看待提交内容确认无意外**

Run: `git status --short && git diff --stat`
Expected: 仅上述 3 个修改 + 1 个未跟踪 `.pt`。

- [ ] **Step 3: 提交**

```bash
git add scripts/train/train_hudreader.py tests/test_hudreader.py docs/PIPELINE.md majsoul_eye/recognize/hud_reader.pt
git commit -m "feat(train): HudReader trained on v4 — manifest val-list fix, CE-head aug; ship hud_reader.pt (STATUS §1.53)"
```

---

### Task 2: `assemble_hud` 改吃 `Detection` 对象

**Files:**
- Modify: `majsoul_eye/recognize/hudstate.py`（`assemble_hud` 的迭代头，约 46-57 行）
- Test: `tests/test_hudstate.py`（fixture 全部改为 `Detection` 对象）

**Interfaces:**
- Consumes: `majsoul_eye.recognize.detector.Detection`（字段 `xyxy: tuple[float,...]`, `name: str`, `tile: Optional[str]`, `cls: int`, `score: float`, `poly`）。
- Produces: `assemble_hud(dets: list[Detection], reader, frame_bgr) -> dict`——返回 dict 结构不变（`scores/round/wall/kyotaku/honba/seat_wind/buttons/riichi`）。Task 4 直接调它。

- [ ] **Step 1: 改写测试为 Detection 对象**

`tests/test_hudstate.py` 顶部替换 fixture 构造（其余断言逻辑不动，把每个 `("name", box)` 元组换成 `D("name", box)`）：

```python
import numpy as np

from majsoul_eye.hud import DET_NAMES
from majsoul_eye.recognize.detector import Detection
from majsoul_eye.recognize.hudstate import assemble_hud
from majsoul_eye.tiles import TILE_NAMES


def D(name, box):
    """Test helper: a Detection as the 56-class detector would emit it."""
    cls = DET_NAMES.index(name)
    return Detection(xyxy=tuple(float(v) for v in box), name=name,
                     tile=name if cls < len(TILE_NAMES) else None,
                     cls=cls, score=0.9)


class StubReader:
    def __init__(self, answers): self.answers = answers
    def read(self, crop, cls): return self.answers[cls]


frame = np.zeros((1080, 1920, 3), np.uint8)
dets = [D("score_self", (900, 460, 1000, 500)), D("wall_count", (925, 385, 995, 415)),
        D("honba_count", (235, 135, 315, 185)), D("round_label", (905, 350, 1015, 385)),
        D("btn_pon", (1200, 740, 1360, 790)), D("btn_skip", (1400, 740, 1560, 790)),
        D("1m", (100, 900, 190, 1050))]          # tile det must be ignored
```

后续所有 `("reach_stick", (...))` 等元组同样换成 `D("reach_stick", (...))`；`no_anchor_dets`/`fallback_dets` 同理。断言全部保留。

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. python tests/test_hudstate.py`
Expected: FAIL（`assemble_hud` 解包 `for cls, (x0,...) in dets` 对 Detection 对象抛 TypeError/ValueError）。

- [ ] **Step 3: 改 `assemble_hud` 迭代头**

`majsoul_eye/recognize/hudstate.py` 中：

```python
def assemble_hud(dets, reader, frame_bgr: np.ndarray) -> dict:
```
的循环头由 `for cls, (x0, y0, x1, y1) in dets:` 改为：

```python
    for det in dets:
        cls = det.name
        x0, y0, x1, y1 = (int(round(v)) for v in det.xyxy)
        if cls not in HUD_NAMES:
            continue
```

（float 坐标必须取整才能切片；其余函数体不变。）同步把模块 docstring 里的接口描述改为 "takes runtime `Detection` objects"。

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. python tests/test_hudstate.py`
Expected: `test_hudstate OK`

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/recognize/hudstate.py tests/test_hudstate.py
git commit -m "refactor(recognize): assemble_hud takes Detection objects — close the never-joined seam"
```

---

### Task 3: `check_observed` 的 HUD×视觉交叉校验

**Files:**
- Modify: `majsoul_eye/state/observe.py`（`check_observed`，在 `return v` 前追加）
- Test: `tests/test_observe.py`（文件末尾追加）

**Interfaces:**
- Produces: `check_observed` 新增三条 violation（仅对应字段非 `None` 时启用），消息前缀分别含 `"kyotaku"`、`"scores sum"`、`"wall count"`（Task 7 的 `_REJECT_CATS` 依赖这些子串）。

- [ ] **Step 1: 写失败测试**

`tests/test_observe.py` 末尾追加：

```python
# --- HUD x vision cross-checks (spec 2026-07-09 §3) --------------------------

def _hud_obs(**kw):
    o = ObservedState()
    o.hero_hand = ["1m"] * 4 + ["2m"] * 4 + ["3m"] * 4 + ["4m"]   # 13, none >4
    o.dora_markers = ["1p"]
    for k, v in kw.items():
        setattr(o, k, v)
    return o

assert check_observed(_hud_obs()) == []                    # HUD None -> checks dormant

# kyotaku < visible riichi count -> hard violation
bad = _hud_obs(kyotaku=0, reach=[True, False, False, False])
assert any("kyotaku" in m for m in check_observed(bad))
ok = _hud_obs(kyotaku=1, reach=[True, False, False, False])
assert not any("kyotaku" in m for m in check_observed(ok))
carry = _hud_obs(kyotaku=2, reach=[False] * 4)             # carryover only: fine
assert not any("kyotaku" in m for m in check_observed(carry))

# score conservation: sum(scores) + 1000*kyotaku == 100000
bad = _hud_obs(scores=[25000, 25000, 25000, 25000], kyotaku=1)
assert any("scores sum" in m for m in check_observed(bad))
ok = _hud_obs(scores=[24000, 25000, 25000, 25000], kyotaku=1,
              reach=[True, False, False, False])
assert not any("scores sum" in m for m in check_observed(ok))
# scores present but kyotaku unread -> conservation check stays dormant
half = _hud_obs(scores=[25000, 25000, 25000, 25000])
assert not any("scores sum" in m for m in check_observed(half))

# wall conservation: pred = 70 - sum(rivers) - n_kans - (1 if drawn)
o = _hud_obs(left_tile_count=70)
assert not any("wall count" in m for m in check_observed(o))
o = _hud_obs(left_tile_count=69)                           # +-1 tolerance
assert not any("wall count" in m for m in check_observed(o))
o = _hud_obs(left_tile_count=68)
assert any("wall count" in m for m in check_observed(o))
o = _hud_obs(left_tile_count=68, drawn_tile="9p")          # pred 69 -> |69-68|<=1
assert not any("wall count" in m for m in check_observed(o))
o = _hud_obs(left_tile_count=68)                           # a kan drops pred to 69
o.melds[1] = [ObservedMeld("ankan", ["9s", "9s", "9s", "9s"])]
assert not any("wall count" in m for m in check_observed(o))
o = _hud_obs(left_tile_count=64)
o.rivers[0] = [ObservedRiverTile("1s")] * 3
o.rivers[2] = [ObservedRiverTile("2s")] * 2                # pred = 70-5 = 65
assert not any("wall count" in m for m in check_observed(o))

print("test_observe hud cross-checks OK")
```

（若文件顶部没有 `ObservedMeld`/`ObservedRiverTile` 的 import，补上。）

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. python tests/test_observe.py`
Expected: 第一条新 assert（kyotaku）FAIL。

- [ ] **Step 3: 实现**

`majsoul_eye/state/observe.py` 的 `check_observed`，在最后的 `return v` 前追加：

```python
    # --- HUD x vision cross-checks (fields are None unless a HUD reader ran) --
    n_reach = sum(1 for x in o.reach if x)
    if o.kyotaku is not None and o.kyotaku < n_reach:
        # every accepted riichi put a stick on the table; the counter can
        # only lag during the declaration animation — reject that window.
        v.append(f"kyotaku {o.kyotaku} < visible riichi count {n_reach}")
    if o.scores is not None and o.kyotaku is not None \
            and sum(o.scores) + 1000 * o.kyotaku != 100000:
        v.append(f"scores sum {sum(o.scores)} + 1000*{o.kyotaku} kyotaku != 100000")
    if o.left_tile_count is not None:
        # Conservation: each discard implies one draw except the post-chi/pon
        # forced one; a called-away discard's draw cancels against exactly that
        # exemption; each kan nets -1 (replacement). +-1 absorbs an opponent's
        # in-flight draw and the §1.53 pixel=GT-1 counter timing.
        pred = 70 - sum(len(r) for r in o.rivers) - o.n_kans() \
            - (1 if o.drawn_tile else 0)
        if abs(pred - o.left_tile_count) > 1:
            v.append(f"wall count {o.left_tile_count} vs predicted {pred} (>1 off)")
    return v
```

同步把模块 docstring 里 "2D-HUD fields are Optional slots filled once the HUD micro-readers land" 更新为 "filled by the HUD micro-readers when available"。

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. python tests/test_observe.py`
Expected: 全部 OK 行（原有 + 新增）。

- [ ] **Step 5: 回归相邻测试**

Run: `PYTHONPATH=. python tests/test_reconstruct.py && PYTHONPATH=. python tests/test_assemble.py`
Expected: 均 OK（新检查在 HUD 字段 None 时休眠，不影响现有路径）。

- [ ] **Step 6: Commit**

```bash
git add majsoul_eye/state/observe.py tests/test_observe.py
git commit -m "feat(state): HUD x vision cross-checks in check_observed — kyotaku/score/wall conservation"
```

---

### Task 4: `assemble()` 接入 HUD（`_fill_hud`）

**Files:**
- Modify: `majsoul_eye/recognize/assemble.py`
- Test: `tests/test_assemble.py`（末尾追加）

**Interfaces:**
- Consumes: Task 2 的 `assemble_hud(list[Detection], reader, frame_bgr) -> dict`。
- Produces: `assemble(dets, region, frame_bgr=None, hud_reader=None) -> ObservedState`——两个新参数都给且 `region.ox == 0`（非宽屏）时填 HUD 字段；否则行为与现状逐位一致。Task 6/7 依赖此签名。

- [ ] **Step 1: 写失败测试**

`tests/test_assemble.py` 末尾追加：

```python
# --- HUD fill via hud_reader (spec 2026-07-09 §1/§2) --------------------------
import numpy as np

from majsoul_eye.hud import DET_NAMES
from majsoul_eye.normalize import locate_fullscreen
from majsoul_eye.recognize.assemble import assemble
from majsoul_eye.recognize.detector import Detection
from majsoul_eye.tiles import TILE_NAMES


def _D(name, box):
    cls = DET_NAMES.index(name)
    return Detection(xyxy=tuple(float(v) for v in box), name=name,
                     tile=name if cls < len(TILE_NAMES) else None,
                     cls=cls, score=0.9)


class _StubReader:
    def __init__(self, answers): self.answers = answers
    def read(self, crop, cls): return self.answers[cls]


_frame = np.zeros((1080, 1920, 3), np.uint8)
_region = locate_fullscreen(_frame)
_hud_dets = [
    _D("round_label", (905, 350, 1015, 385)),
    _D("wall_count", (925, 385, 995, 415)),
    _D("score_self", (900, 460, 1000, 500)), _D("score_right", (1030, 400, 1080, 470)),
    _D("score_across", (900, 300, 1000, 340)), _D("score_left", (840, 400, 890, 470)),
    _D("riichi_stick_count", (200, 60, 280, 100)), _D("honba_count", (235, 135, 315, 185)),
    _D("seat_wind_self", (830, 470, 890, 530)),
    _D("btn_riichi", (1200, 740, 1360, 790)),
    _D("reach_stick", (900, 500, 1020, 530)),      # below anchor -> rel seat 0 (self)
]
_reader = _StubReader({"round_label": "S3", "wall_count": "余42",
                       "score_self": "24000", "score_right": "31000",
                       "score_across": "20000", "score_left": "24000",
                       "riichi_stick_count": "x1", "honba_count": "x2",
                       "seat_wind_self": "W"})

o = assemble(_hud_dets, _region, frame_bgr=_frame, hud_reader=_reader)
assert o.bakaze == "S" and o.kyoku == 3
assert o.left_tile_count == 42 and o.kyotaku == 1 and o.honba == 2
assert o.scores == [24000, 31000, 20000, 24000]
assert o.seat_wind_self == "W"
assert o.pending_buttons == ["btn_riichi"]
assert o.reach == [True, False, False, False]      # stick below anchor -> self
assert "hud" in o.zone_confidence

# graceful degrade: no reader -> HUD detections dropped, fields stay None
o2 = assemble(_hud_dets, _region)
assert o2.scores is None and o2.bakaze is None and o2.pending_buttons is None
assert o2.reach == [False] * 4 and "hud" not in o2.zone_confidence

# partial scores (one seat unread) -> whole scores list stays None
_bad = _StubReader({**_reader.answers, "score_left": ""})
o3 = assemble(_hud_dets, _region, frame_bgr=_frame, hud_reader=_bad)
assert o3.scores is None and o3.kyoku == 3         # other fields still fill

print("test_assemble hud fill OK")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. python tests/test_assemble.py`
Expected: FAIL（`assemble()` 不接受 `frame_bgr` 关键字）。

- [ ] **Step 3: 实现**

`majsoul_eye/recognize/assemble.py`：

3a. 顶部 import 区（`from majsoul_eye.state.observe import ObservedMeld, ObservedRiverTile` 附近）加：

```python
from majsoul_eye.recognize.hudstate import assemble_hud
```

3b. 模块级新函数（放在 `assemble` 定义前）：

```python
_REL_KEYS = ("self", "right", "across", "left")


def _fill_hud(o, hud: dict) -> None:
    """assemble_hud dict -> ObservedState HUD slots (spec 2026-07-09 §2).
    scores are all-or-nothing (reconstruct needs the full relative list);
    reach ORs the stick attribution over the sideways-derived flags."""
    sc = hud["scores"]
    if all(sc[k] is not None for k in _REL_KEYS):
        o.scores = [sc[k] for k in _REL_KEYS]
    if hud["round"]:
        o.bakaze, o.kyoku = hud["round"][0], int(hud["round"][1])
    o.left_tile_count = hud["wall"]
    o.kyotaku = hud["kyotaku"]
    o.honba = hud["honba"]
    o.seat_wind_self = hud["seat_wind"]
    o.pending_buttons = hud["buttons"]
    for r, k in enumerate(_REL_KEYS):
        if hud["riichi"][k]:
            o.reach[r] = True
```

3c. `assemble` 签名改为：

```python
def assemble(dets, region: BoardRegion, frame_bgr=None, hud_reader=None) -> ObservedState:
```

docstring 里 "HUD fields stay None (their readers are the 2026-07-04 spec)" 改为：

```
HUD fields fill when BOTH frame_bgr and hud_reader are given and the frame is
not wide (region.ox == 0 — the HUD detector is only trained on the 16:9
layout; wide phone frames keep HUD fields None, spec 2026-07-09 §5).
```

3d. 循环里 `if det.tile is None: continue` 改为收集：

```python
    hud_dets = []
    ...
    for det in dets:
        if det.tile is None:       # HUD-class detection — routed to assemble_hud below
            hud_dets.append(det)
            continue
```

3e. 在 per-seat 循环（`o.reach[seat] = ...` 结束）之后、`o.zone_confidence = ...` 之前插入：

```python
    if frame_bgr is not None and hud_reader is not None and hud_dets \
            and region.ox == 0:
        _fill_hud(o, assemble_hud(hud_dets, hud_reader, frame_bgr))
        for det in hud_dets:
            note("hud", det)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. python tests/test_assemble.py`
Expected: 全部 OK（原有 + `test_assemble hud fill OK`）。

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/recognize/assemble.py tests/test_assemble.py
git commit -m "feat(recognize): assemble fills the ObservedState HUD half via HudReader (16:9 only)"
```

---

### Task 5: 立直棒权威化（投影 + reconstruct ghost 绑定）

**Files:**
- Modify: `majsoul_eye/state/observe.py`（`observed_from_board` 一行）
- Modify: `majsoul_eye/state/reconstruct.py`（`_search`）
- Test: `tests/test_observe.py`、`tests/test_reconstruct.py`（各追加）

**Interfaces:**
- Produces: `observed_from_board` 的 `reach[r]` 定义变为 横牌 ∨ `state.reach[a]`（已接受立直）；`reconstruct` 对 `obs.reach[r]=True` 且无横牌的座强制 ghost 绑定 reach，无法绑定 → 整帧不可行。

- [ ] **Step 1: 写投影失败测试**

`tests/test_observe.py` 末尾追加：

```python
# --- reach projection: stick-visible riichi survives a called-away tile ------
from majsoul_eye.state.replay import Replayer

_rp = Replayer()
for _ev in [
    {"type": "start_game", "id": 0},
    {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0,
     "oya": 0, "dora_marker": "1p", "scores": [25000] * 4,
     "tehais": [["1m"] * 3 + ["2m"] * 3 + ["3m"] * 3 + ["4m"] * 3 + ["9m"],
                ["?"] * 13, ["?"] * 13, ["?"] * 13]},
    {"type": "tsumo", "actor": 0, "pai": "1s"},
    {"type": "dahai", "actor": 0, "pai": "1s", "tsumogiri": True},
    {"type": "tsumo", "actor": 1, "pai": "?"},
    {"type": "reach", "actor": 1},
    {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": False},
    {"type": "reach_accepted", "actor": 1},
    {"type": "pon", "actor": 2, "target": 1, "pai": "5p", "consumed": ["5p", "5p"]},
    {"type": "dahai", "actor": 2, "pai": "9p", "tsumogiri": False},
]:
    _rp.apply(_ev)
_o = observed_from_board(_rp.state)
assert _o.rivers[1] == []                 # declaration tile called away
assert _o.reach[1] is True                # ...but the accepted riichi is stick-visible
print("test_observe reach projection OK")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. python tests/test_observe.py`
Expected: `_o.reach[1] is True` FAIL（现投影只看横牌）。

- [ ] **Step 3: 改投影**

`majsoul_eye/state/observe.py` 的 `observed_from_board`，`o.reach[r] = any(t.sideways for t in o.rivers[r])` 改为：

```python
        # sideways tile OR the on-table stick: an accepted riichi whose
        # declaration tile was called away is still visible via its stick.
        o.reach[r] = any(t.sideways for t in o.rivers[r]) or state.reach[a]
```

docstring 里 "reach[] is derived from the VISIBLE sideways tile, not state.reach — ... (known static limitation)" 一段改为说明新定义（横牌 ∨ 已接受立直棒；stick 检测使该限制解除）。

Run: `PYTHONPATH=. python tests/test_observe.py`
Expected: 全 OK。

- [ ] **Step 4: 写 reconstruct 失败测试**

`tests/test_reconstruct.py` 末尾追加：

```python
# --- stick-authoritative reach (spec 2026-07-09 §4) ---------------------------
# seat1 declared riichi; the declaration tile 5p was pon'd away by seat2 and
# seat1 has not discarded since -> no sideways tile anywhere for seat1, only
# the stick (obs.reach[1]). The search must bind seat1's reach to its ghost.
o = ObservedState()
o.hero_hand = ["1m"] * 3 + ["2m"] * 3 + ["3m"] * 3 + ["4m"] * 3 + ["9m"]
o.dora_markers = ["1p"]
o.rivers[0] = [ObservedRiverTile("1s")]
o.rivers[2] = [ObservedRiverTile("9p")]
o.melds[2] = [ObservedMeld("pon", ["5p", "5p", "5p"], called_pai="5p", from_rel=3)]
o.reach = [False, True, False, False]
r = reconstruct(o)
assert r.ok, r.reason
evs = r.events
assert {"type": "reach", "actor": 1} in [
    {k: e[k] for k in ("type", "actor")} for e in evs if e["type"] == "reach"]
assert any(e["type"] == "reach_accepted" and e["actor"] == 1 for e in evs)

# negative: stick says reach but seat1 has ONLY an upright discard and no ghost
# (no call anywhere) -> physically contradictory, must be rejected.
o2 = ObservedState()
o2.hero_hand = list(o.hero_hand)
o2.dora_markers = ["1p"]
o2.rivers[0] = [ObservedRiverTile("1s")]
o2.rivers[1] = [ObservedRiverTile("5p")]          # upright — NOT a declaration
o2.reach = [False, True, False, False]
r2 = reconstruct(o2)
assert not r2.ok

print("test_reconstruct stick reach OK")
```

（若文件顶部没有 `ObservedState/ObservedRiverTile/ObservedMeld/reconstruct` 的 import，补上。）

- [ ] **Step 5: 跑测试确认失败**

Run: `PYTHONPATH=. python tests/test_reconstruct.py`
Expected: 第一段 FAIL——搜索找到不含 reach 的序列（`{"type":"reach","actor":1}` 不在事件里）。

- [ ] **Step 6: 实现 `_search` 扩展**

`majsoul_eye/state/reconstruct.py`，五处修改：

6a. `side_idx` 定义后加：

```python
    # riichi known from the HUD reach stick but declaration tile invisible
    # (called away with no later discard): the search MUST bind that seat's
    # reach to one of its ghosts, and may not finish until it has.
    must_reach = [bool(obs.reach[r]) and side_idx[r] is None for r in range(4)]
```

6b. `all_done` 增加 `rghost` 参与判定：

```python
    def all_done(cur, cidx, kkmask, rghost):
        return (list(cur) == n and list(cidx) == ncre
                and kkmask == (1 << len(kakans)) - 1
                and all(rghost >> r & 1 for r in range(4) if must_reach[r]))
```

6c. `go()` 内调用点改为 `if all_done(cur, cidx, kkmask, rghost):`。

6d. `decide()` 开头的 drew-terminal 改为
`and all_done(cur, cidx, kkmask, rghost)):`；
pending_it 终止分支改为 `if all_done(cur, tuple(ncidx), kkmask, nrg):`。

6e. variants 构造改为：

```python
            variants = [False]
            if not declared(actor, cur, rghost):
                if side_idx[actor] is not None and cur[actor] == side_idx[actor]:
                    variants.append(True)      # bind the reach to this ghost
                elif must_reach[actor]:
                    variants.append(True)      # stick-known riichi, tile called away
```

- [ ] **Step 7: 跑测试确认通过 + 回归**

Run: `PYTHONPATH=. python tests/test_reconstruct.py && PYTHONPATH=. python tests/test_observe.py && PYTHONPATH=. python tests/test_eval_reconstruction.py`
Expected: 全 OK。

- [ ] **Step 8: Commit**

```bash
git add majsoul_eye/state/observe.py majsoul_eye/state/reconstruct.py tests/test_observe.py tests/test_reconstruct.py
git commit -m "feat(state): reach stick authoritative — projection sees accepted riichi, reconstruct ghost-binds it"
```

---

### Task 6: CLI 接入（recognize_frame.py 默认读 HUD）

**Files:**
- Modify: `scripts/recognize/recognize_frame.py`

**Interfaces:**
- Consumes: Task 4 的 `assemble(..., frame_bgr=, hud_reader=)`；`HudReader(path=None)` 默认加载打包权重、缺文件抛 `FileNotFoundError`。

- [ ] **Step 1: 实现**

1a. argparse 加：

```python
    ap.add_argument("--hud-weights", default=None,
                    help="HudReader weights (default: packaged "
                         "majsoul_eye/recognize/hud_reader.pt)")
    ap.add_argument("--no-hud", action="store_true",
                    help="skip HUD reading (HUD fields stay null)")
```

1b. `det = TileDetector(...)` 之后：

```python
    reader = None
    if not args.no_hud:
        from majsoul_eye.recognize.hudreader import HudReader
        try:
            reader = HudReader(args.hud_weights, device=args.device)
        except FileNotFoundError:
            print("[recognize_frame] hud_reader weights not found — HUD fields "
                  "disabled", file=sys.stderr)
    print(f"[recognize_frame] hud: {'on' if reader else 'off'}", file=sys.stderr)
```

1c. assemble 调用改为：

```python
            obs = assemble(det.predict(img), locate(img),
                           frame_bgr=img, hud_reader=reader)
```

1d. 模块 docstring 里 `HUD fields null until HudReader lands` 改为
`HUD fields read by HudReader (--no-hud to disable; wide phone frames keep them null)`。

- [ ] **Step 2: 冒烟（真实帧，CPU）**

```bash
FRAME=$(ls captures/raw/ai_session/run_8/game1/*.png | head -1)
PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe scripts/recognize/recognize_frame.py --pretty "$FRAME"
```

Expected: stderr 显示 `hud: on`；JSON 的 `observed` 里 `scores`/`bakaze`/`kyoku`/`left_tile_count` 非 null（该帧若恰在动画窗被拒，换下一张帧重试）。再跑一次加 `--no-hud`：同帧 HUD 字段为 null，`ok` 与 M1 行为一致。

- [ ] **Step 3: Commit**

```bash
git add scripts/recognize/recognize_frame.py
git commit -m "feat(recognize): recognize_frame CLI reads HUD by default (--no-hud/--hud-weights)"
```

---

### Task 7: eval 扩展（HUD per-field + score-anim 门控）

**Files:**
- Modify: `scripts/eval/eval_reconstruction.py`
- Test: `tests/test_eval_reconstruction.py`（跑通即可；若它直接调 `run_assemble` 需适配新参数）

**Interfaces:**
- Consumes: Task 4/5 的行为；`majsoul_eye.state.replay.is_score_anim_window(state)`。
- Produces: assemble 层报告新增 `hud_ok/hud_err/hud_missing`（per-field 计数）与 `score_anim_rejected`；oracle 层在 score-anim 帧投影 `include_hud=False`。

- [ ] **Step 1: 实现**

1a. import 行加 `is_score_anim_window`：

```python
from majsoul_eye.state.replay import (Replayer, is_call_pending, is_deal_window,
                                      is_score_anim_window)
```

1b. `_REJECT_CATS` 头部插入三条（在 `("stray detection", ...)` 前）：

```python
    ("kyotaku", "hud_kyotaku"),
    ("scores sum", "hud_scores"),
    ("wall count", "hud_wall"),
```

1c. `run_oracle` 里 `obs = observed_from_board(st)` 改为：

```python
        # score-anim windows: HUD numbers roll on screen right after a reach —
        # project them away so the new HUD cross-checks don't reject GT frames.
        obs = observed_from_board(st, include_hud=not is_score_anim_window(st))
```

1d. `run_assemble` 签名与实现：

```python
def run_assemble(cap, states, report, weights, device, hud_weights=None,
                 no_hud=False):
    import cv2
    from majsoul_eye.normalize import locate_fullscreen
    from majsoul_eye.recognize.assemble import assemble
    from majsoul_eye.recognize.detector import TileDetector
    det = TileDetector(weights, device=device)
    reader = None
    if not no_hud:
        from majsoul_eye.recognize.hudreader import HudReader
        try:
            reader = HudReader(hud_weights, device=device)
        except FileNotFoundError:
            pass
    frames = load_frames(paths.frames_dir_for(cap))
    for seq, st in states.items():
        if seq not in frames or not st.in_round or is_deal_window(st):
            continue
        img = cv2.imread(frames[seq])
        if img is None:
            continue
        obs = assemble(det.predict(img), locate_fullscreen(img),
                       frame_bgr=img, hud_reader=reader)
        gt = observed_from_board(st)
        if obs.violations:
            report["rejected"] += 1
            if is_score_anim_window(st):
                report["score_anim_rejected"] += 1
            for cat in reject_categories(obs.violations):
                report["rejected_reasons"][cat] = \
                    report["rejected_reasons"].get(cat, 0) + 1
            continue
        d = diff_zones(obs, gt)
        report["frames"] += 1
        if not d:
            report["ok"] += 1
        for z in d:
            report["zone_errors"][z] = report["zone_errors"].get(z, 0) + 1
        if reader is not None and not is_score_anim_window(st):
            for fld in ("scores", "bakaze", "kyoku", "honba", "kyotaku",
                        "left_tile_count", "seat_wind_self"):
                got, want = getattr(obs, fld), getattr(gt, fld)
                if got is None:
                    report["hud_missing"][fld] = report["hud_missing"].get(fld, 0) + 1
                elif got == want or (fld == "left_tile_count" and want is not None
                                     and abs(got - want) <= 1):
                    report["hud_ok"][fld] = report["hud_ok"].get(fld, 0) + 1
                else:
                    report["hud_err"][fld] = report["hud_err"].get(fld, 0) + 1
```

（`gt` 由 `include_hud=False` 改为默认全填：`obs_key` 只取牌面键，diff 不受影响。）

1e. `main()`：argparse 加

```python
    ap.add_argument("--hud-weights", default=None)
    ap.add_argument("--no-hud", action="store_true")
```

`total` 初始化加 `"hud_ok": {}, "hud_err": {}, "hud_missing": {}, "score_anim_rejected": 0`；
assemble 分支调用改 `run_assemble(cap, states, total, args.weights, args.device, args.hud_weights, args.no_hud)`；
assemble 打印后追加：

```python
        print(f"  hud ok {total['hud_ok']}\n  hud err {total['hud_err']}\n"
              f"  hud missing {total['hud_missing']}; "
              f"score-anim rejected {total['score_anim_rejected']}")
```

- [ ] **Step 2: 跑 eval 自身的测试**

Run: `PYTHONPATH=. python tests/test_eval_reconstruction.py`
Expected: OK（若它以旧签名调 `run_assemble`，用默认参数即兼容；必要时在该测试里传 `no_hud=True` 保持原行为）。

- [ ] **Step 3: Commit**

```bash
git add scripts/eval/eval_reconstruction.py tests/test_eval_reconstruction.py
git commit -m "feat(eval): per-field HUD accuracy in assemble tier; score-anim HUD gating in oracle"
```

---

### Task 8: 全量回归（验收线，不产码）

**Files:** 无（只跑命令；发现问题回上游任务修）

- [ ] **Step 1: 全部单测**

```bash
for t in tests/test_*.py; do PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe "$t" || break; done
```
Expected: 全过。

- [ ] **Step 2: oracle 全量**

```bash
PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe scripts/eval/eval_reconstruction.py --captures captures/raw/ai_session --level oracle
```
Expected: **0 mismatch、0 infeasible（fail）**；ok 基线 10121。ok 数因两处语义变化可少量移动：(a) score-anim 帧现以 include_hud=False 投影（仍应重建成功，不掉数）；(b) 新 wall/kyotaku 校验若把个别 GT 帧划进 skipped_violations，必须逐帧检查确属 GT-leads-pixels 窗口才可接受，否则视为实现 bug 回修。

- [ ] **Step 3: assemble 基准——先 --no-hud 复现旧基线（规模化降级验证）**

```bash
PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe scripts/eval/eval_reconstruction.py \
  --captures captures/raw/ai_session/run_8/game1/game1.jsonl --level assemble \
  --weights weights/detector/tile_detector_obb_20260709_055509.pt --no-hud --device cuda
```
Expected: 与 M1 现基线一致量级（911/936 全等、拒收 13；容许 ±1 帧漂移——reach 投影定义变化会影响 stick-riichi 帧的 reach 键比较）。

- [ ] **Step 4: assemble 基准——带 HUD**

同命令去掉 `--no-hud`。
Expected:
- 非 score-anim 帧的牌面全等率不低于 Step 3；
- 新增拒收集中在 `hud_scores`/`hud_kyotaku` 且 `score_anim_rejected` 能解释其大头；
- `hud_ok` 各字段 ≥ §1.53 qa_hud 水平（scores/bakaze/kyoku/honba/kyotaku/seat_wind ≈100%，`left_tile_count` 在 ±1 规则下 ≈100%）。

- [ ] **Step 5: 手机截图回归（宽屏不因 HUD 误拒）**

先 `ls samples/` 确认基准集文件名与张数（记忆基线 16 张全过），再：

```bash
PYTHONPATH=. C:/Users/zsx/miniforge3/envs/auto/python.exe scripts/recognize/recognize_frame.py samples/IMG_*.PNG | C:/Users/zsx/miniforge3/envs/auto/python.exe -c "import sys,json;rs=[json.loads(l) for l in sys.stdin];print(sum(r['ok'] for r in rs),'/',len(rs))"
```

（若 `samples/` 里还有其他扩展名，把实际文件全列进去；不要用可能空匹配的 glob——bash 会把字面量传给 CLI 造成 "cannot read image" 假失败。）
Expected: 与 M1 基线同数全过（宽屏 `region.ox>0` 跳过 HUD 填充，行为不变）。

- [ ] **Step 6: 记录数字**

把 oracle / assemble(--no-hud) / assemble(HUD) / samples 四组数字存入
`scratchpad` 笔记或直接写进 Task 9 的 STATUS 草稿。

---

### Task 9: 文档同步（管线纪律）

**Files:**
- Modify: `docs/PIPELINE.md`（runtime recognizer 段：recognize 链默认带 HudReader；`recognize_frame.py` 新开关）
- Modify: `docs/STATUS.md`（新条目 §1.54：单帧 HUD 集成——接口、交叉校验、立直棒权威化、Task 8 实测数字、宽屏 known-gap）

- [ ] **Step 1: 更新 PIPELINE.md**

在 runtime/QA 相关小节补：`recognize_frame.py` 默认加载 `majsoul_eye/recognize/hud_reader.pt`（`--no-hud`/`--hud-weights`）；assemble 的 HUD 填充条件（frame+reader+非宽屏）；eval assemble 层新增 HUD per-field 报告。

- [ ] **Step 2: 追加 STATUS §1.54**

包含：动机（HUD 半边训练完成→接线）、四个行为变化（assemble 签名、check_observed 三校验、reach 投影/重建、CLI/eval）、Task 8 全部实测数字、known-gaps（宽屏 HUD 留 null；score-anim 帧运行时被分数守恒拒收=预期；按钮帧 recall 缺口沿袭 §1.53）。

- [ ] **Step 3: Commit**

```bash
git add docs/PIPELINE.md docs/STATUS.md
git commit -m "docs: PIPELINE/STATUS §1.54 — single-frame HUD integration shipped"
```
