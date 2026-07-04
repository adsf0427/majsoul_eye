# Unify AI Capture to the Manual `GTRecord` Format — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scripts/capture/autoplay_ai.py` write, at capture time, the same per-game `GTRecord` JSONL + screenshot index the manual `record_gt.py` path writes, so both capture lines produce identical data and the `convert_mjcopilot` pass + `captures/intermediate/gt/` layer disappear.

**Architecture:** Reuse two proven pieces — the manual path's background `GTWriter` and the converter's MJAI-extraction trick (`TracedGameState`/`CapList`/`deepcopy`, extracted into a shared `make_capturing_game_state` helper). `autoplay_ai` writes each `GTRecord` incrementally per liqi message (so an abnormally-ended game is still usable) and appends a screenshot-index line per saved PNG. Downstream consumers are repointed from `intermediate/gt/` to `raw/ai_session/` via new `paths` helpers, and a one-time migration re-derives the existing b64 games into the new layout.

**Tech Stack:** Python 3.12 (conda `auto` env), MahjongCopilot (`game.game_state.GameState`, `liqi.LiqiProto`) for GT derivation (dev-only, never imported by `recognize/`), the repo's plain-script test convention.

## Global Constraints

- Run everything in the conda `auto` env with `PYTHONPATH=.` from the repo root. `PY=C:/Users/zsx/miniforge3/envs/auto/python.exe`.
- `recognize/` must stay Akagi/MahjongCopilot-free. All new capture-derivation code lives under `majsoul_eye/capture/` (dev-only) or `scripts/`.
- **Emit policy:** emit a `GTRecord` only for messages that produced **new mjai** (matches today's `convert_mjcopilot` output → regenerated AI datasets stay byte-identical).
- All recording additions in the `autoplay_ai` hot path are wrapped in `try/except` and fed to the background `GTWriter`; recording must never stall or break the bot loop.
- New on-disk layout per AI game (mirrors manual `sessionN.jsonl` + `sessionN/`): `run_N/gameM.jsonl` (GTRecord) + `run_N/gameM/{liqi.jsonl (raw wire), frames.jsonl (screenshot index), frames/*.png, metadata.json}`. The single-game legacy run keeps its shape: `run_1.jsonl` (GTRecord) + `run_1/{liqi.jsonl, frames.jsonl, frames/*.png}`.
- Stable dataset names via `paths.ai_game_name(capture)`: `.../run_N/gameM.jsonl` → `ai_run_N_gameM`; `.../run_1.jsonl` → `ai_run_1`; anything else → basename stem (so manual sessions pass through unchanged).
- Test convention: plain script, `if __name__ == "__main__":` runs every `test_*` in module globals and prints `<module> OK`; also pytest-compatible. Run e.g. `PYTHONPATH=. $PY tests/test_mjcopilot_gt.py`.
- Frame-index `file` fields are written **index-relative** (`"frames/000009.png"`); always resolve via `paths.resolve_frame_path`.

**Design reference:** `docs/superpowers/specs/2026-07-04-unify-ai-capture-gtrecord-design.md`

---

## File Structure

- `majsoul_eye/capture/schema.py` — **modify**: `GTRecord` (unchanged) + move `GTWriter` here (Akagi-free background JSONL writer).
- `majsoul_eye/capture/akagi_tap.py` — **modify**: import `GTWriter` from `schema` instead of defining it; drop now-unused imports.
- `majsoul_eye/capture/mjcopilot_gt.py` — **create**: `make_capturing_game_state(game_state_cls, bot) -> (gs, drain_mjai)` + `gt_fields(msg) -> (method, action_name)`. MahjongCopilot-agnostic at import time (subclasses a class passed in).
- `scripts/data/convert_mjcopilot.py` — **modify**: `convert_game` uses `make_capturing_game_state` + `gt_fields`; wire filename parameterized (default `frames.jsonl`).
- `scripts/capture/autoplay_ai.py` — **modify**: per-game `GTWriter` + wire → `liqi.jsonl` + incremental screenshot index; `GameState(bot)` → `make_capturing_game_state`.
- `majsoul_eye/paths.py` — **modify**: add `ai_captures()`, `ai_game_name()`; repoint `converted_gt_captures()`.
- `scripts/train/build_dataset.py` — **modify**: `--from-annotations` stem uses `paths.ai_game_name`.
- `scripts/annotate/annotate_ai_session.py` — **modify**: default captures = `paths.ai_captures()`; per-game name = `paths.ai_game_name`.
- `scripts/data/rebuild_datasets.py` — **modify**: discovery = `paths.ai_captures()`; names + frames-dir + val-cap via `ai_game_name` / `frames_dir_for`.
- `scripts/data/ingest_run.py` — **modify**: drop the convert step; build directly from raw.
- `scripts/data/migrate_ai_to_gtrecord.py` — **create**: one-time, dry-run-default, idempotent migration of existing b64 games.
- `tests/test_mjcopilot_gt.py` — **create**: `make_capturing_game_state` + `gt_fields` unit tests.
- `tests/test_paths.py` — **create**: `ai_game_name` + `ai_captures` unit tests.
- `tests/test_schema_writer.py` — **create**: `GTWriter` round-trip test.
- `tests/test_migrate_ai.py` — **create**: migration path-planning + idempotency unit tests.

---

## Task 1: Move `GTWriter` into the Akagi-free `schema.py`

**Why:** `autoplay_ai` must reuse the manual path's background writer, but importing `akagi_tap` triggers the Akagi `MajsoulBridge` import. `GTWriter` has no Akagi deps, so it belongs in `schema.py` (which already owns `GTRecord` + the `_schema` header).

**Files:**
- Modify: `majsoul_eye/capture/schema.py`
- Modify: `majsoul_eye/capture/akagi_tap.py:39-79` (remove the class), `:28-36` (imports)
- Test: `tests/test_schema_writer.py`

**Interfaces:**
- Produces: `majsoul_eye.capture.schema.GTWriter(path: str)` with methods `.put(record: GTRecord) -> None`, `.next_seq() -> int`, `.close() -> None`. Writes the `{"_schema": N}` header on open, one `GTRecord` per line off a background thread. Consumed by Task 3 (`autoplay_ai`) and by `akagi_tap`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_schema_writer.py`:

```python
"""GTWriter round-trip: header + one line per record, readable by read_records.

Plain-script style: PYTHONPATH=. <auto-python> tests/test_schema_writer.py
"""
import os
import tempfile
import time

from majsoul_eye.capture.schema import GTRecord, GTWriter, read_records


def _rec(seq):
    return GTRecord(seq=seq, ts=1.0, flow_id="", seat=0, last_op_step=0,
                    syncing=False, method=".lq.ActionPrototype",
                    action_name="ActionDiscardTile",
                    raw_liqi={"method": ".lq.ActionPrototype", "data": {"name": "ActionDiscardTile"}},
                    mjai=[{"type": "dahai", "pai": "5m"}])


def test_gtwriter_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "game1.jsonl")
        w = GTWriter(path)
        w.put(_rec(0))
        w.put(_rec(1))
        w.close()
        recs = list(read_records(path))
        assert [r.seq for r in recs] == [0, 1]
        assert recs[0].mjai == [{"type": "dahai", "pai": "5m"}]
        # header line present and skipped by read_records
        with open(path, encoding="utf-8") as fh:
            first = fh.readline()
        assert first.strip() == '{"_schema": 1}'


def test_gtwriter_next_seq_monotonic():
    with tempfile.TemporaryDirectory() as d:
        w = GTWriter(os.path.join(d, "g.jsonl"))
        assert [w.next_seq() for _ in range(3)] == [0, 1, 2]
        w.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_schema_writer OK")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=. $PY tests/test_schema_writer.py`
Expected: FAIL — `ImportError: cannot import name 'GTWriter' from 'majsoul_eye.capture.schema'`.

- [ ] **Step 3: Move `GTWriter` into `schema.py`**

In `majsoul_eye/capture/schema.py`, update the imports block (top of file) to add `queue`/`threading`/`Optional`:

```python
from __future__ import annotations

import base64
import dataclasses
import json
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional
```

Then append this class to the end of `schema.py` (after `read_records`):

```python
class GTWriter:
    """Background JSONL writer fed by a thread-safe queue.

    Writes the ``{"_schema": N}`` header on open, then one :class:`GTRecord` per
    line from a daemon thread so file I/O never stalls the caller's hot path.
    Akagi-free: shared by the Akagi tap (``akagi_tap``) and the AI capture
    (``scripts/capture/autoplay_ai``).
    """

    def __init__(self, path: str):
        self.path = path
        self._q: "queue.Queue[Optional[GTRecord]]" = queue.Queue()
        self._seq = 0
        self._lock = threading.Lock()
        self._fh = open(path, "w", encoding="utf-8")
        self._fh.write('{"_schema": %d}\n' % SCHEMA_VERSION)
        self._fh.flush()
        self._thread = threading.Thread(target=self._run, name="gt-writer", daemon=True)
        self._thread.start()

    def next_seq(self) -> int:
        with self._lock:
            s = self._seq
            self._seq += 1
            return s

    def put(self, record: GTRecord) -> None:
        self._q.put(record)

    def _run(self) -> None:
        while True:
            rec = self._q.get()
            if rec is None:
                break
            try:
                self._fh.write(rec.to_json_line() + "\n")
                self._fh.flush()
            except Exception:
                pass

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=5)
        try:
            self._fh.close()
        except Exception:
            pass
```

- [ ] **Step 4: Update `akagi_tap.py` to import instead of define**

In `majsoul_eye/capture/akagi_tap.py`, change the imports block near the top (currently `import atexit, queue, threading, time`) to drop `queue`/`threading` (now only `GTWriter` used them):

```python
import atexit
import time
from typing import Any, Optional

from .schema import GTRecord, GTWriter, SCHEMA_VERSION
```

Delete the entire `class GTWriter:` definition (the block from `class GTWriter:` through its `close` method, lines ~39–79). Leave everything else (`install`, `patched_parse_liqi`, `uninstall`, `recorded_count`) unchanged — they already reference `GTWriter(path)`, which now resolves to the imported one.

- [ ] **Step 5: Run the writer test + confirm the tap still imports**

Run: `PYTHONPATH=. $PY tests/test_schema_writer.py`
Expected: `test_schema_writer OK`.

Run: `PYTHONPATH=. $PY -c "import majsoul_eye.capture.akagi_tap as m; print('akagi_tap import OK', hasattr(m, 'GTWriter'))"`
Expected: `akagi_tap import OK True` (it may fail only if the Akagi `mitm.bridge...` import is attempted — it is not at module import time, only inside `install()`, so this must succeed).

- [ ] **Step 6: Commit**

```bash
git add majsoul_eye/capture/schema.py majsoul_eye/capture/akagi_tap.py tests/test_schema_writer.py
git commit -m "refactor(capture): move GTWriter into Akagi-free schema.py"
```

---

## Task 2: Shared MJAI-extraction helper + refactor `convert_game`

**Why:** The live capture and the legacy migration must derive `mjai` identically. Extract the converter's `TracedGameState`/`CapList`/deepcopy trick into one helper both call, and prove it reproduces today's output.

**Files:**
- Create: `majsoul_eye/capture/mjcopilot_gt.py`
- Modify: `scripts/data/convert_mjcopilot.py:70-157` (`convert_game`)
- Test: `tests/test_mjcopilot_gt.py`

**Interfaces:**
- Produces: `make_capturing_game_state(game_state_cls, bot) -> (gs, drain_mjai)`. `gs` is a `game_state_cls` instance (drop-in; all attribute access/assignment works); `drain_mjai() -> list[dict]` returns the deep-copied mjai events derived since the previous call, in order. Consumed by Task 3 and by `convert_game`.
- Produces: `gt_fields(msg: dict | None) -> tuple[str | None, str | None]` = `(method, action_name)`, where `action_name` is `msg["data"]["name"]` only when `method == ".lq.ActionPrototype"`.
- Consumes: nothing MahjongCopilot-specific at import time.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mjcopilot_gt.py`:

```python
"""make_capturing_game_state + gt_fields — no MahjongCopilot needed (fake GameState).

Plain-script style: PYTHONPATH=. <auto-python> tests/test_mjcopilot_gt.py
"""
from majsoul_eye.capture.mjcopilot_gt import make_capturing_game_state, gt_fields


class FakeGameState:
    """Stand-in for MahjongCopilot's GameState: derives mjai into
    self.mjai_pending_input_msgs, which the real class also does."""
    def __init__(self, bot):
        self.bot = bot
        self.seat = 0
        self.mjai_pending_input_msgs = []          # traced -> becomes a CapList

    def input(self, msg):
        for ev in msg["events"]:
            self.mjai_pending_input_msgs.append(ev)
        return None

    def reset_pending(self):
        self.mjai_pending_input_msgs = []          # GameState flushes between turns


def test_drain_returns_new_events_each_call():
    gs, drain = make_capturing_game_state(FakeGameState, object())
    gs.input({"events": [{"type": "tsumo", "pai": "5m"}]})
    assert drain() == [{"type": "tsumo", "pai": "5m"}]
    assert drain() == []                            # nothing new since last drain
    gs.input({"events": [{"type": "dahai", "pai": "1p"}]})
    assert drain() == [{"type": "dahai", "pai": "1p"}]


def test_drain_is_deepcopied_isolated_from_later_mutation():
    gs, drain = make_capturing_game_state(FakeGameState, object())
    ev = {"type": "start_kyoku", "tehais": [["1m", "2m"]]}
    gs.input({"events": [ev]})
    out = drain()
    ev["tehais"][0].append("MUTATED")               # GameState mutates the hand in place
    assert out[0]["tehais"][0] == ["1m", "2m"]       # captured copy is frozen


def test_bot_still_sees_events_capList_transparency():
    gs, drain = make_capturing_game_state(FakeGameState, object())
    gs.input({"events": [{"type": "dahai", "pai": "9s"}]})
    # the underlying list the bot reads is still populated
    assert list(gs.mjai_pending_input_msgs) == [{"type": "dahai", "pai": "9s"}]


def test_survives_pending_reset_between_turns():
    gs, drain = make_capturing_game_state(FakeGameState, object())
    gs.input({"events": [{"type": "tsumo", "pai": "5m"}]})
    assert drain() == [{"type": "tsumo", "pai": "5m"}]
    gs.reset_pending()                              # new empty CapList installed
    gs.input({"events": [{"type": "dahai", "pai": "5m"}]})
    assert drain() == [{"type": "dahai", "pai": "5m"}]


def test_gt_fields():
    assert gt_fields({"method": ".lq.ActionPrototype",
                      "data": {"name": "ActionDiscardTile"}}) == (
        ".lq.ActionPrototype", "ActionDiscardTile")
    # non-ActionPrototype: action_name is None even if data has a name
    assert gt_fields({"method": ".lq.FastTest.authGame",
                      "data": {"name": "x"}}) == (".lq.FastTest.authGame", None)
    assert gt_fields(None) == (None, None)
    assert gt_fields({"method": ".lq.ActionPrototype", "data": None}) == (
        ".lq.ActionPrototype", None)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_mjcopilot_gt OK")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=. $PY tests/test_mjcopilot_gt.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'majsoul_eye.capture.mjcopilot_gt'`.

- [ ] **Step 3: Create the shared helper**

Create `majsoul_eye/capture/mjcopilot_gt.py`:

```python
"""Shared MJAI-extraction for the MahjongCopilot GT paths (DEV-ONLY).

MahjongCopilot's ``GameState.input(msg)`` derives MJAI events into
``self.mjai_pending_input_msgs`` and batches them to the bot. To record those
events we wrap GameState so every appended event is ``deepcopy``'d and queued —
GameState MUTATES the AI hand list in place as the game proceeds, so an
un-copied reference to ``start_kyoku.tehais`` gets overwritten later.

``make_capturing_game_state`` is MahjongCopilot-agnostic at import time: the
caller passes in whichever ``GameState`` class its import context resolved, so
this module never imports MahjongCopilot itself. Used by both the live capture
(``scripts/capture/autoplay_ai``) and the offline converter/migration
(``scripts/data/convert_mjcopilot``) — one derivation, no drift.
"""
from __future__ import annotations

import copy
from typing import Any, Callable


def make_capturing_game_state(game_state_cls, bot) -> tuple[Any, Callable[[], list]]:
    """Return ``(gs, drain_mjai)``.

    ``gs`` is a ``game_state_cls`` instance (drop-in for the real GameState);
    ``drain_mjai()`` returns the deep-copied MJAI events derived since the
    previous call, in append order.
    """
    events: list = []
    read = [0]

    class _CapList(list):
        def append(self, x):
            events.append(copy.deepcopy(x))
            list.append(self, x)

        def extend(self, xs):
            for x in xs:
                self.append(x)

    class _Traced(game_state_cls):
        def __setattr__(self, k, v):
            # Always install an EMPTY CapList for the pending list (ignore v):
            # GameState resets it to [] then populates via append/extend, so an
            # empty tracked list captures every subsequent event. Matches the
            # original convert_mjcopilot TracedGameState behavior verbatim.
            object.__setattr__(self, k, _CapList() if k == "mjai_pending_input_msgs" else v)

    gs = _Traced(bot)

    def drain_mjai() -> list:
        new = events[read[0]:]
        read[0] = len(events)
        return new

    return gs, drain_mjai


def gt_fields(msg) -> tuple:
    """``(method, action_name)`` for a parsed liqi message, matching GTRecord.

    ``action_name`` is the ActionPrototype's inner ``data.name`` (e.g.
    ``ActionDiscardTile``) and is None for any other method.
    """
    if not isinstance(msg, dict):
        return None, None
    method = msg.get("method")
    data = msg.get("data")
    action_name = data.get("name") if method == ".lq.ActionPrototype" and isinstance(data, dict) else None
    return method, action_name
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=. $PY tests/test_mjcopilot_gt.py`
Expected: `test_mjcopilot_gt OK`.

- [ ] **Step 5: Refactor `convert_game` to use the helper**

In `scripts/data/convert_mjcopilot.py`, add the import near the top (after the existing `from majsoul_eye.capture.schema import ...`):

```python
from majsoul_eye.capture.mjcopilot_gt import make_capturing_game_state, gt_fields
```

Replace the body of `convert_game` (currently lines ~70–157) with this version. It keeps `StubBot`, drops the inline `CapList`/`TracedGameState`, parameterizes the wire filename, and derives fields via `gt_fields`:

```python
def convert_game(game_dir: str, liqimod, GameState, Bot, GameMode,
                 wire_name: str = "frames.jsonl") -> tuple[list[GTRecord], list[dict]]:
    """Return (gt_records, frame_index) for one MahjongCopilot game dir.

    ``wire_name`` is the b64 liqi-wire JSONL inside ``game_dir`` (legacy captures
    name it ``frames.jsonl``; migrated captures name it ``liqi.jsonl``)."""

    class StubBot(Bot):
        def __init__(self):
            super().__init__("stub")

        @property
        def supported_modes(self):
            return [GameMode.MJ4P, GameMode.MJ3P]

        @property
        def info_str(self):
            return "stub"

        def _init_bot_impl(self, mode=GameMode.MJ4P):
            pass

        def react(self, m):
            return None

        def react_batch(self, l):
            return None

    gs, drain_mjai = make_capturing_game_state(GameState, StubBot())
    lp = liqimod.LiqiProto()

    log = os.path.join(game_dir, wire_name)
    seq_records: dict[int, dict] = {}   # seq -> {ts, method, action_name, raw, mjai[]}
    for line in open(log, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        try:
            res = lp.parse(base64.b64decode(d["b64"]))
        except Exception:
            continue
        if res is None:
            continue
        with contextlib.redirect_stdout(io.StringIO()):  # silence GameState chatter
            try:
                gs.input(res)
            except Exception:
                pass
        new = drain_mjai()
        if new:
            method, action_name = gt_fields(res)
            rec = seq_records.setdefault(d["seq"], {
                "ts": d.get("ts", 0.0),
                "method": method,
                "action_name": action_name,
                "raw": res,
                "mjai": [],
            })
            rec["mjai"].extend(new)

    seat = getattr(gs, "seat", -1) or 0
    records: list[GTRecord] = []
    for seq in sorted(seq_records):
        r = seq_records[seq]
        records.append(GTRecord(
            seq=seq, ts=r["ts"], flow_id="", seat=seat, last_op_step=0, syncing=False,
            method=r["method"], action_name=r["action_name"], raw_liqi=r["raw"], mjai=r["mjai"],
        ))

    # frame index: every png named by seq -> captures-relative path (points into raw/)
    frame_index = []
    for p in sorted(glob.glob(os.path.join(game_dir, "frames", "*.png"))):
        seq = int(os.path.splitext(os.path.basename(p))[0])
        frame_index.append({"seq": seq, "file": paths.rel_to_captures(p), "status": "ok"})
    return records, frame_index
```

The `types` import at the top of `convert_mjcopilot.py` is now unused (it was only for the `bot.factory` stub in `_import_mjcopilot`, which stays). Leave `_import_mjcopilot` and `main` unchanged.

- [ ] **Step 6: Equivalence check against the committed golden (one-time, run now, before any migration)**

The 16 files under `captures/intermediate/gt/*.jsonl` are the current validated output. Confirm the refactored `convert_game` reproduces one of them byte-for-byte. Run:

```bash
PYTHONPATH=. $PY - <<'PY'
import json, os
from majsoul_eye import paths
from scripts.data.convert_mjcopilot import _import_mjcopilot, convert_game
from majsoul_eye.capture.schema import read_records

name = "ai_run_3_game1"
golden = os.path.join(paths.GT, f"{name}.jsonl")
game_dir = os.path.join(paths.RAW_AI_SESSION, "run_3", "game1")  # legacy wire = frames.jsonl
liqimod, GameState, Bot, GameMode = _import_mjcopilot("../MahjongCopilot")
os.chdir("../MahjongCopilot")
recs, _ = convert_game(os.path.abspath(game_dir) if not os.path.isabs(game_dir) else game_dir,
                       liqimod, GameState, Bot, GameMode, wire_name="frames.jsonl")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if False else os.environ.get("PWD", "."))
# compare against golden
gold = list(read_records(golden))
a = [r.to_json_line() for r in recs]
b = [r.to_json_line() for r in gold]
print("new", len(a), "golden", len(b), "identical:", a == b)
assert a == b, "convert_game output diverged from golden — refactor changed behavior"
print("EQUIVALENCE OK")
PY
```

> Note for the implementer: `convert_game` globs PNGs relative to its `game_dir`, and `_import_mjcopilot` chdir's into MahjongCopilot for asset paths. Pass an **absolute** `game_dir`. If the inline heredoc's cwd bookkeeping is awkward, instead add a temporary `scripts/data/_equiv_check.py` with the same logic (absolute paths resolved before the MahjongCopilot chdir, exactly like `convert_mjcopilot.main` does) and run it; delete it after. Expected final line: `EQUIVALENCE OK`.

Expected: `identical: True` then `EQUIVALENCE OK`. If it diverges, STOP — the refactor is not behavior-preserving; diff `a` vs `b` to find the first differing record.

- [ ] **Step 7: Commit**

```bash
git add majsoul_eye/capture/mjcopilot_gt.py scripts/data/convert_mjcopilot.py tests/test_mjcopilot_gt.py
git commit -m "refactor(capture): extract shared make_capturing_game_state; convert_game reuses it"
```

---

## Task 3: `autoplay_ai` writes GTRecord + screenshot index inline

**Why:** The core change — produce the unified format at capture time, incrementally, so an abnormally-ended game is still a usable game.

**Files:**
- Modify: `scripts/capture/autoplay_ai.py`
- Test: `tests/test_autoplay_gt.py`

**Interfaces:**
- Consumes: `GTWriter` (Task 1), `make_capturing_game_state`, `gt_fields` (Task 2), `GTRecord` (`capture.schema`).
- Produces (on disk, per game): `out_dir/game{idx}.jsonl` (GTRecord), `out_dir/game{idx}/liqi.jsonl` (raw wire), `out_dir/game{idx}/frames.jsonl` (screenshot index), `out_dir/game{idx}/frames/*.png`.

- [ ] **Step 1: Write the failing test (factored pure helper + import smoke)**

Create `tests/test_autoplay_gt.py`. It tests a small pure helper (`_frame_index_line`) that Step 3 adds, and confirms `autoplay_ai` still imports and exposes its flags:

```python
"""autoplay_ai unified-GT plumbing: frame-index line shape + import/flag smoke.

Plain-script style: PYTHONPATH=. <auto-python> tests/test_autoplay_gt.py
"""
import argparse
import importlib.util
import os


def _load_autoplay():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "capture", "autoplay_ai.py")
    spec = importlib.util.spec_from_file_location("autoplay_ai_gt_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_frame_index_line_shape():
    mod = _load_autoplay()
    line = mod._frame_index_line(9, 123.5)
    assert line == {"seq": 9, "file": "frames/000009.png", "status": "ok", "ts": 123.5}


def test_autoplay_ai_still_imports_and_has_flags():
    mod = _load_autoplay()
    seen = {}
    real = argparse.ArgumentParser.parse_args

    def capture(self, *a, **k):
        for act in self._actions:
            seen[tuple(act.option_strings)] = act
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = capture
    try:
        try:
            mod.main()
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = real
    flat = {opt for opts in seen for opt in opts}
    assert "--out" in flat and "--server" in flat


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_autoplay_gt OK")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=. $PY tests/test_autoplay_gt.py`
Expected: FAIL — `AttributeError: module ... has no attribute '_frame_index_line'`.

- [ ] **Step 3: Add imports + the pure helper**

In `scripts/capture/autoplay_ai.py`, add to the top-level imports (the block with `from majsoul_eye.capture.roi_diff import roi_diff`):

```python
from majsoul_eye.capture.roi_diff import roi_diff
from majsoul_eye.capture.schema import GTRecord, GTWriter
from majsoul_eye.capture.mjcopilot_gt import make_capturing_game_state, gt_fields
```

Add this module-level pure helper (near `stable_capture_step`, before `main`):

```python
def _frame_index_line(seq: int, ts: float) -> dict:
    """One screenshot-index record (index-relative file path), matching the
    manual FrameSyncer's frames.jsonl shape so build_dataset consumes it the
    same way."""
    return {"seq": seq, "file": f"frames/{seq:06d}.png", "status": "ok", "ts": ts}
```

- [ ] **Step 4: Route the game-state + writers to the unified format**

All edits are inside `main()` in `scripts/capture/autoplay_ai.py`.

**(a)** Replace the bot/game-state construction. Find:

```python
    game_state = None
    game_idx = 0                # which game in this run (-> out_dir/game<idx>/)
    game_raw_fh = None          # current game's frames.jsonl handle
    game_frames_dir = None      # current game's frames/ dir
```

Replace with:

```python
    game_state = None
    drain_mjai = None           # closure from make_capturing_game_state (per game)
    game_idx = 0                # which game in this run (-> out_dir/game<idx>/)
    game_wire_fh = None         # current game's raw-wire liqi.jsonl handle
    gt_writer = None            # current game's GTRecord writer (game<idx>.jsonl)
    game_index_fh = None        # current game's screenshot-index frames.jsonl handle
    game_frames_dir = None      # current game's frames/ dir
```

**(b)** Add a small local writer for GTRecords, right after the `_stab = {"ref": None}` line (just before `def maybe_screenshot`):

```python
    def write_gt(seq, ts, msg):
        """Derive this message's mjai and, if any, append a GTRecord. Wrapped so
        recording can never break the capture loop (mirrors akagi_tap)."""
        if gt_writer is None or drain_mjai is None:
            return
        try:
            mjai = drain_mjai()
            if not mjai:
                return                                   # emit-on-new-mjai (see design §4)
            method, action_name = gt_fields(msg)
            gt_writer.put(GTRecord(
                seq=seq, ts=ts, flow_id="", seat=getattr(game_state, "seat", -1),
                last_op_step=0, syncing=False, method=method, action_name=action_name,
                raw_liqi=msg, mjai=mjai))
        except Exception as e:
            print(f"  gt write err seq {seq}: {type(e).__name__}: {e}", flush=True)
```

Note: `game_state`, `gt_writer`, `drain_mjai` are read via closure. Because `maybe_screenshot` already uses `nonlocal` for its rebinds, and `write_gt` only *reads* these, no `nonlocal` is needed in `write_gt`.

**(c)** In `maybe_screenshot`, append the index line after a successful PNG save. Find:

```python
        with open(os.path.join(game_frames_dir, f"{pending_seq:06d}.png"), "wb") as fh:
            fh.write(png)
        fulfilled_seq = pending_seq
```

Replace with:

```python
        with open(os.path.join(game_frames_dir, f"{pending_seq:06d}.png"), "wb") as fh:
            fh.write(png)
        if game_index_fh is not None:
            game_index_fh.write(json.dumps(_frame_index_line(pending_seq, time.time())) + "\n")
            game_index_fh.flush()
        fulfilled_seq = pending_seq
```

**(d)** Rework the `authGame` REQ new-game block. Find the block starting at `if (mtype, method) == (liqi.MsgType.REQ, liqi.LiqiMethod.authGame):` and replace its body with (changes: close all three per-game handles; open wire as `liqi.jsonl`, GTRecord writer as `game{idx}.jsonl`, index as `frames.jsonl`; use `make_capturing_game_state`; write the authGame wire line + GTRecord):

```python
                if (mtype, method) == (liqi.MsgType.REQ, liqi.LiqiMethod.authGame):
                    auto_next_state.update(active=False, started=0.0, clicked_next=False, failed=False)
                    if game_wire_fh is not None:
                        game_wire_fh.close()
                    if game_index_fh is not None:
                        game_index_fh.close()
                    if gt_writer is not None:
                        gt_writer.close()
                    game_idx += 1
                    game_dir = os.path.join(out_dir, f"game{game_idx}")
                    game_frames_dir = os.path.join(game_dir, "frames")
                    os.makedirs(game_frames_dir, exist_ok=True)
                    gamemeta.write_metadata(game_dir, game_language)   # game<N>/metadata.json = {"language": ...}
                    game_wire_fh = open(os.path.join(game_dir, "liqi.jsonl"), "w", encoding="utf-8")
                    game_index_fh = open(os.path.join(game_dir, "frames.jsonl"), "w", encoding="utf-8")
                    gt_writer = GTWriter(os.path.join(out_dir, f"game{game_idx}.jsonl"))
                    seq, pending_seq, fulfilled_seq = 0, None, None
                    game_state, drain_mjai = make_capturing_game_state(GameState, bot)
                    seq += 1
                    game_wire_fh.write(json.dumps({"seq": seq, "ts": ts,
                                                   "b64": base64.b64encode(raw).decode()}) + "\n")
                    game_wire_fh.flush()
                    game_state.input(msg)
                    write_gt(seq, ts, msg)
                    automation.on_enter_game()
                    print(f"  game{game_idx} start -> {game_dir}", flush=True)
                    continue
```

**(e)** Update the guard after the authGame block. Find:

```python
                if game_state is None or game_raw_fh is None:
                    continue                            # pre-game frames on the socket (before authGame)
```

Replace `game_raw_fh` with `game_wire_fh`:

```python
                if game_state is None or game_wire_fh is None:
                    continue                            # pre-game frames on the socket (before authGame)
```

**(f)** Update the per-message wire write. Find:

```python
                seq += 1
                game_raw_fh.write(json.dumps({"seq": seq, "ts": ts,
                                              "b64": base64.b64encode(raw).decode()}) + "\n")
                game_raw_fh.flush()

                try:
                    reaction = game_state.input(msg)
```

Replace with (write wire, run input, then record the GTRecord):

```python
                seq += 1
                game_wire_fh.write(json.dumps({"seq": seq, "ts": ts,
                                               "b64": base64.b64encode(raw).decode()}) + "\n")
                game_wire_fh.flush()

                try:
                    reaction = game_state.input(msg)
                except Exception as e:
                    print(f"  game_state.input err on {method}: {type(e).__name__}: {e}", flush=True)
                    continue
                write_gt(seq, ts, msg)
```

Note: the original code already has the `except ... continue` for `game_state.input`; keep exactly one copy of it — replace the original `try: reaction = game_state.input(msg)` line and its existing `except` block with the version above (which adds the `write_gt` call after the except). Do NOT duplicate the except.

**(g)** In the game-ended block, close the new handles. Find:

```python
                if game_state.is_game_ended:
                    print(f"  [g{game_idx} seq {seq}] GAME ENDED", flush=True)
                    if args.live and args.auto_next:
                        start_auto_next()
                    else:
                        automation.on_end_game()
                    game_state = None
                    if game_raw_fh is not None:
                        game_raw_fh.close()
                        game_raw_fh = None
```

Replace with:

```python
                if game_state.is_game_ended:
                    print(f"  [g{game_idx} seq {seq}] GAME ENDED", flush=True)
                    if args.live and args.auto_next:
                        start_auto_next()
                    else:
                        automation.on_end_game()
                    game_state = None
                    drain_mjai = None
                    if game_wire_fh is not None:
                        game_wire_fh.close()
                        game_wire_fh = None
                    if game_index_fh is not None:
                        game_index_fh.close()
                        game_index_fh = None
                    if gt_writer is not None:
                        gt_writer.close()
                        gt_writer = None
```

**(h)** In the `finally:` block at the end of `main`, close the new handles. Find:

```python
    finally:
        if overlay is not None:
            overlay.stop()
        if game_raw_fh is not None:
            game_raw_fh.close()
```

Replace with:

```python
    finally:
        if overlay is not None:
            overlay.stop()
        if game_wire_fh is not None:
            game_wire_fh.close()
        if game_index_fh is not None:
            game_index_fh.close()
        if gt_writer is not None:
            gt_writer.close()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `PYTHONPATH=. $PY tests/test_autoplay_gt.py`
Expected: `test_autoplay_gt OK`.

Also byte-compile to catch any stray reference to the removed `game_raw_fh`:

Run: `PYTHONPATH=. $PY -c "import py_compile; py_compile.compile('scripts/capture/autoplay_ai.py', doraise=True); print('compile OK')"`
Expected: `compile OK`. Then grep to be sure the old name is gone:

Run: `grep -n game_raw_fh scripts/capture/autoplay_ai.py || echo "no game_raw_fh remaining"`
Expected: `no game_raw_fh remaining`.

- [ ] **Step 6: Commit**

```bash
git add scripts/capture/autoplay_ai.py tests/test_autoplay_gt.py
git commit -m "feat(capture): autoplay_ai writes unified GTRecord + screenshot index inline"
```

---

## Task 4: `paths` helpers — `ai_captures()` + `ai_game_name()`

**Why:** Downstream must discover AI GTRecords in `raw/ai_session/` and derive the stable flattened dataset name, handling both the multi-game (`run_N/gameM.jsonl`) and single-game (`run_1.jsonl`) shapes.

**Files:**
- Modify: `majsoul_eye/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Produces: `ai_captures() -> list[str]` — sorted GTRecord jsonl paths under `raw/ai_session/` (both `run_*/game*.jsonl` and `run_*.jsonl`).
- Produces: `ai_game_name(capture_path: str) -> str` — `.../run_N/gameM.jsonl`→`ai_run_N_gameM`; `.../run_1.jsonl`→`ai_run_1`; else basename stem.
- `converted_gt_captures()` becomes an alias returning `ai_captures()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_paths.py`:

```python
"""paths.ai_game_name / ai_captures.

Plain-script style: PYTHONPATH=. <auto-python> tests/test_paths.py
"""
import os
import tempfile

from majsoul_eye import paths


def test_ai_game_name_multi_game():
    assert paths.ai_game_name("captures/raw/ai_session/run_3/game1.jsonl") == "ai_run_3_game1"
    assert paths.ai_game_name("captures/raw/ai_session/run_8/game6.jsonl") == "ai_run_8_game6"
    # absolute + backslash variants resolve the same
    assert paths.ai_game_name(r"D:\x\captures\raw\ai_session\run_10\game2.jsonl") == "ai_run_10_game2"


def test_ai_game_name_single_game_run():
    assert paths.ai_game_name("captures/raw/ai_session/run_1.jsonl") == "ai_run_1"


def test_ai_game_name_fallback_for_manual():
    # manual sessions (or anything not matching run/game) fall back to the stem
    assert paths.ai_game_name("captures/raw/manual/session5.jsonl") == "session5"


def test_ai_captures_globs_both_shapes():
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "raw", "ai_session")
        os.makedirs(os.path.join(base, "run_3", "game1"))
        os.makedirs(os.path.join(base, "run_3", "game1", "frames"))
        open(os.path.join(base, "run_3", "game1.jsonl"), "w").close()
        open(os.path.join(base, "run_3", "game1", "liqi.jsonl"), "w").close()   # must NOT match
        open(os.path.join(base, "run_3", "game1", "frames.jsonl"), "w").close() # must NOT match
        open(os.path.join(base, "run_1.jsonl"), "w").close()                    # single-game run
        open(os.path.join(base, "run_3", "ai_settings.json"), "w").close()      # must NOT match
        found = paths._ai_captures_in(base)      # test-seam over the real glob
        got = sorted(os.path.relpath(p, base).replace(os.sep, "/") for p in found)
        assert got == ["run_1.jsonl", "run_3/game1.jsonl"], got


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_paths OK")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=. $PY tests/test_paths.py`
Expected: FAIL — `AttributeError: module 'majsoul_eye.paths' has no attribute 'ai_game_name'`.

- [ ] **Step 3: Implement the helpers**

In `majsoul_eye/paths.py`, add `import re` to the imports at the top (next to `import glob as _glob` / `import os`):

```python
import glob as _glob
import os
import re
```

Add these functions (place them next to `converted_gt_captures`, and REPLACE the existing `converted_gt_captures`):

```python
def ai_game_name(capture_path: str) -> str:
    """Stable flattened dataset name for an AI GTRecord capture.

    ``.../run_N/gameM.jsonl`` -> ``ai_run_N_gameM``;
    ``.../run_1.jsonl``       -> ``ai_run_1`` (single-game legacy run);
    anything else             -> the basename stem (manual sessions pass through).
    """
    p = os.path.abspath(capture_path).replace("\\", "/")
    parts = p.split("/")
    stem = os.path.splitext(parts[-1])[0]                 # gameM  or  run_N
    parent = parts[-2] if len(parts) >= 2 else ""
    if re.fullmatch(r"run_\d+", parent) and re.fullmatch(r"game\d+", stem):
        return f"ai_{parent}_{stem}"
    if re.fullmatch(r"run_\d+", stem):
        return f"ai_{stem}"
    return stem


def _ai_captures_in(ai_session_dir: str) -> list:
    """AI GTRecord jsonls under a given ai_session root (test seam for ai_captures)."""
    multi = _glob.glob(os.path.join(ai_session_dir, "run_*", "game*.jsonl"))
    single = _glob.glob(os.path.join(ai_session_dir, "run_*.jsonl"))
    return sorted(multi + single)


def ai_captures() -> list:
    """Sorted AI GTRecord capture jsonls under raw/ai_session/ (both shapes)."""
    return _ai_captures_in(RAW_AI_SESSION)


def converted_gt_captures() -> list:
    """Back-compat alias: AI GT captures now live in raw/ai_session/ (no convert)."""
    return ai_captures()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=. $PY tests/test_paths.py`
Expected: `test_paths OK`.

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/paths.py tests/test_paths.py
git commit -m "feat(paths): ai_captures + ai_game_name for raw/ai_session GTRecords"
```

---

## Task 5: Repoint downstream consumers to `raw/ai_session/`

**Why:** With AI GTRecords now in `raw/ai_session/` under per-run/game names, the annotate → build → detector pipeline and `ingest_run` must discover and name them via the new helpers. `build_dataset` needs its `--from-annotations` stem aligned with `annotate_ai_session`'s output name.

**Files:**
- Modify: `scripts/train/build_dataset.py:159-162`
- Modify: `scripts/annotate/annotate_ai_session.py:91` and `:226`
- Modify: `scripts/data/rebuild_datasets.py` (`gt_frames_dir`, `main` discovery, `val_cap`)
- Modify: `scripts/data/ingest_run.py` (drop convert; build from raw)
- Test: `tests/test_downstream_rewire.py`

**Interfaces:**
- Consumes: `paths.ai_captures`, `paths.ai_game_name`, `paths.frames_dir_for` (Task 4).

- [ ] **Step 1: Write the failing test (name-consistency contract)**

Create `tests/test_downstream_rewire.py`. It locks the contract that `annotate_ai_session`'s output name and `build_dataset`'s `--from-annotations` lookup agree, and that `rebuild_datasets` no longer references `intermediate/gt` for discovery:

```python
"""Downstream rewire contracts: annotate name == build_dataset from-annotations
stem, and rebuild_datasets discovers via paths.ai_captures.

Plain-script style: PYTHONPATH=. <auto-python> tests/test_downstream_rewire.py
"""
import os

from majsoul_eye import paths


def test_annotate_and_build_agree_on_name():
    # Both derive the annotation filename the same way for an AI capture.
    cap = "captures/raw/ai_session/run_3/game1.jsonl"
    assert paths.ai_game_name(cap) == "ai_run_3_game1"


def test_build_dataset_uses_ai_game_name():
    src = open("scripts/train/build_dataset.py", encoding="utf-8").read()
    assert "paths.ai_game_name(args.capture)" in src
    # the old collision-prone stem line is gone
    assert "os.path.splitext(os.path.basename(args.capture))[0]" not in src


def test_annotate_uses_ai_game_name_and_ai_captures():
    src = open("scripts/annotate/annotate_ai_session.py", encoding="utf-8").read()
    assert "paths.ai_game_name(cap)" in src
    assert "paths.ai_captures()" in src


def test_rebuild_uses_ai_captures_and_frames_dir_for():
    src = open("scripts/data/rebuild_datasets.py", encoding="utf-8").read()
    assert "paths.ai_captures()" in src
    assert "paths.frames_dir_for" in src


def test_ingest_run_has_no_convert_step():
    src = open("scripts/data/ingest_run.py", encoding="utf-8").read()
    assert "convert_mjcopilot.py" not in src


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_downstream_rewire OK")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=. $PY tests/test_downstream_rewire.py`
Expected: FAIL on `test_build_dataset_uses_ai_game_name` (assertion) — the code still uses the basename stem.

- [ ] **Step 3: `build_dataset` — align the from-annotations stem**

In `scripts/train/build_dataset.py`, confirm `from majsoul_eye import paths` is imported (add it to the imports if missing). Find (around line 159-162):

```python
    if args.from_annotations:
        import json
        stem = os.path.splitext(os.path.basename(args.capture))[0]
        ann_path = os.path.join(args.from_annotations, f"{stem}.jsonl")
```

Replace the `stem` line:

```python
    if args.from_annotations:
        import json
        stem = paths.ai_game_name(args.capture)
        ann_path = os.path.join(args.from_annotations, f"{stem}.jsonl")
```

- [ ] **Step 4: `annotate_ai_session` — name via `ai_game_name`, default via `ai_captures`**

In `scripts/annotate/annotate_ai_session.py`, in `_process_capture` find (line ~91):

```python
    name = os.path.splitext(os.path.basename(cap))[0]
```

Replace:

```python
    name = paths.ai_game_name(cap)
```

Then in `main` find (line ~226):

```python
    captures = args.captures or paths.converted_gt_captures()
```

Replace:

```python
    captures = args.captures or paths.ai_captures()
```

(`paths` is already imported in this file.)

- [ ] **Step 5: `rebuild_datasets` — discovery, frames-dir, val-cap**

In `scripts/data/rebuild_datasets.py`:

**(a)** Replace `game_name` + `gt_frames_dir` (lines ~84-94). The frames dir now comes from the capture path, and the name from `ai_game_name`:

```python
def game_name(capture: str) -> str:
    return paths.ai_game_name(capture)


def dataset_dir(name: str) -> str:
    return os.path.join(DATASETS, f"precise_{name}")


def gt_frames_dir(capture: str, name: str) -> str:
    # AI GT frames now sit next to the capture jsonl (X.jsonl <-> X/); letterboxed
    # games still override to their de-letterboxed frames.
    return FRAMES_OVERRIDE.get(name, paths.frames_dir_for(capture))
```

**(b)** In `main`, replace the discovery + guard (lines ~137-143):

```python
    captures = sorted(paths.ai_captures())
    if not captures:
        raise SystemExit(f"no AI GT captures under {paths.RAW_AI_SESSION} — capture with "
                         f"scripts/capture/autoplay_ai.py, or migrate legacy runs with "
                         f"scripts/data/migrate_ai_to_gtrecord.py")
    names = [game_name(c) for c in captures]
    name_to_cap = dict(zip(names, captures))
    if args.val not in names:
        raise SystemExit(f"--val game {args.val!r} not among AI games: {names}")
```

**(c)** In stage 2, update the `gt_frames_dir` call (line ~177). Find:

```python
            r.run([py, "scripts/train/build_dataset.py", c, gt_frames_dir(n),
                   "--out", out, "--from-annotations", ANN_OUT, "--drop-violations"])
```

Replace:

```python
            r.run([py, "scripts/train/build_dataset.py", c, gt_frames_dir(c, n),
                   "--out", out, "--from-annotations", ANN_OUT, "--drop-violations"])
```

**(d)** Update `val_cap` (line ~211). Find:

```python
    val_cap = os.path.join(paths.GT, f"{args.val}.jsonl")
```

Replace:

```python
    val_cap = name_to_cap[args.val]
```

- [ ] **Step 6: `ingest_run` — drop the convert step, build from raw**

In `scripts/data/ingest_run.py`, replace `main` (the convert call + build loop). Find the section from `# 1) convert all games in one call` through the `# 2) build_dataset per game` loop (lines ~71-83) and replace with:

```python
    # 1) build_dataset per game — AI captures are already GTRecord (no convert).
    #    discover_games returns rel dirs like "run_13/game1"; the GTRecord jsonl is
    #    the sibling "run_13/game1.jsonl" and its frames dir is "run_13/game1/".
    for rel, name in games:
        cap = os.path.join(parent, rel) + ".jsonl"
        frames_dir = os.path.join(parent, rel)
        if not os.path.exists(cap):
            print(f"  SKIP {name}: no GTRecord at {cap} "
                  f"(capture with autoplay_ai or migrate a legacy b64 run first)")
            continue
        run([py, "scripts/train/build_dataset.py", cap, frames_dir + os.sep,
             "--out", os.path.join(args.datasets, name), "--drop-violations"], env)
```

Then update the retrain glob (step 3, lines ~91-97) — it globs `args.captures` (the old GT dir) for capture jsonls; point it at the datasets it just built instead. Find:

```python
    if args.train:
        data_args = []
        for cap in sorted(glob.glob(os.path.join(args.captures, "*.jsonl"))):
            nm = os.path.splitext(os.path.basename(cap))[0]
            crops = os.path.join(args.datasets, nm, "crops")
            if os.path.isdir(crops):
                data_args += ["--data", f"{nm}={crops}:{cap}"]
```

Replace with:

```python
    if args.train:
        data_args = []
        for rel, nm in games:
            crops = os.path.join(args.datasets, nm, "crops")
            cap = os.path.join(parent, rel) + ".jsonl"      # this game's GTRecord jsonl
            if os.path.isdir(crops) and os.path.exists(cap):
                data_args += ["--data", f"{nm}={crops}:{cap}"]
```

The now-unused `--captures` / `--mjcopilot` argparse options may be left in place (harmless) or deleted. Keep `import glob` — the crop-summary loop (lines ~85-88) still uses it.

- [ ] **Step 7: Run the rewire test + byte-compile all four scripts**

Run: `PYTHONPATH=. $PY tests/test_downstream_rewire.py`
Expected: `test_downstream_rewire OK`.

Run:
```bash
PYTHONPATH=. $PY -c "import py_compile as p; [p.compile(f, doraise=True) for f in ['scripts/train/build_dataset.py','scripts/annotate/annotate_ai_session.py','scripts/data/rebuild_datasets.py','scripts/data/ingest_run.py']]; print('compile OK')"
```
Expected: `compile OK`.

Run `rebuild_datasets` in **dry-run** to confirm it discovers raw captures and prints raw frames-dir paths (no execution):
```bash
PYTHONPATH=. $PY scripts/data/rebuild_datasets.py 2>&1 | head -30
```
Expected: it lists AI game names (`ai_run_3_game1`, …) and the printed `build_dataset.py` commands reference `captures/raw/ai_session/run_*/game*` frames dirs — NOT `captures/intermediate/gt/...`. (It may error on `--val ai_run_8_game1` only if that game isn't discovered; it should be.)

- [ ] **Step 8: Commit**

```bash
git add scripts/train/build_dataset.py scripts/annotate/annotate_ai_session.py scripts/data/rebuild_datasets.py scripts/data/ingest_run.py tests/test_downstream_rewire.py
git commit -m "refactor(pipeline): consume AI GT from raw/ai_session via ai_captures/ai_game_name"
```

---

## Task 6: One-time migration of the legacy b64 games

**Why:** Bring the existing b64 captures into the new layout so no data is left in the old format and `intermediate/gt/` can be retired.

**Files:**
- Create: `scripts/data/migrate_ai_to_gtrecord.py`
- Test: `tests/test_migrate_ai.py`

**Interfaces:**
- Produces: `plan_targets(game_dir: str) -> dict` — pure path-planning: given a b64 game dir, returns `{"name", "gt_path", "wire_dest", "index_path"}`. Consumed by the migration driver and the test.

- [ ] **Step 1: Write the failing test (pure path-planning + idempotency guard)**

Create `tests/test_migrate_ai.py`:

```python
"""Migration path-planning: b64 game dir -> new-layout targets.

Plain-script style: PYTHONPATH=. <auto-python> tests/test_migrate_ai.py
"""
import os
import tempfile
import importlib.util


def _load():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "data", "migrate_ai_to_gtrecord.py")
    spec = importlib.util.spec_from_file_location("migrate_ai", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plan_targets_multi_game():
    mod = _load()
    gd = "captures/raw/ai_session/run_3/game1"
    t = mod.plan_targets(gd)
    assert t["name"] == "ai_run_3_game1"
    assert t["gt_path"].replace(os.sep, "/").endswith("run_3/game1.jsonl")
    assert t["wire_dest"].replace(os.sep, "/").endswith("run_3/game1/liqi.jsonl")
    assert t["index_path"].replace(os.sep, "/").endswith("run_3/game1/frames.jsonl")


def test_plan_targets_single_game_run():
    mod = _load()
    gd = "captures/raw/ai_session/run_1"
    t = mod.plan_targets(gd)
    assert t["name"] == "ai_run_1"
    assert t["gt_path"].replace(os.sep, "/").endswith("ai_session/run_1.jsonl")
    assert t["wire_dest"].replace(os.sep, "/").endswith("run_1/liqi.jsonl")


def test_already_migrated_detected():
    mod = _load()
    with tempfile.TemporaryDirectory() as d:
        gd = os.path.join(d, "run_3", "game1")
        os.makedirs(gd)
        # no liqi.jsonl yet, has b64 frames.jsonl -> NOT migrated
        open(os.path.join(gd, "frames.jsonl"), "w").close()
        assert mod.is_migrated(gd) is False
        # after rename, liqi.jsonl exists -> migrated
        os.rename(os.path.join(gd, "frames.jsonl"), os.path.join(gd, "liqi.jsonl"))
        open(os.path.join(d, "run_3", "game1.jsonl"), "w").close()
        assert mod.is_migrated(gd) is True


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_migrate_ai OK")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=. $PY tests/test_migrate_ai.py`
Expected: FAIL — `ModuleNotFoundError` / file not found for `migrate_ai_to_gtrecord.py`.

- [ ] **Step 3: Write the migration script**

Create `scripts/data/migrate_ai_to_gtrecord.py`:

```python
"""One-time migration: legacy b64 AI captures -> unified GTRecord layout (DEV-ONLY).

Old (b64):  run_N/gameM/frames.jsonl (wire) + run_N/gameM/frames/*.png
New:        run_N/gameM.jsonl        (GTRecord: raw_liqi + mjai)
            run_N/gameM/liqi.jsonl    (raw wire, renamed from frames.jsonl)
            run_N/gameM/frames.jsonl  (screenshot index {seq,file,status})
            run_N/gameM/frames/*.png  (unchanged)

Single-game legacy run (run_1) keeps its shape: run_1.jsonl + run_1/{liqi.jsonl,
frames.jsonl, frames/*.png}.

Idempotent + dry-run by default. Re-derives GT from the wire via the SHARED
convert_game (same code the live capture's derivation is proven equal to), so the
output matches the retired captures/intermediate/gt/*.jsonl byte-for-byte.

Run (conda `auto` env, repo root, PYTHONPATH=.):
    PYTHONPATH=. $PY scripts/data/migrate_ai_to_gtrecord.py            # dry run
    PYTHONPATH=. $PY scripts/data/migrate_ai_to_gtrecord.py --apply    # do it
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from majsoul_eye import paths
from majsoul_eye.capture.schema import write_records


def find_b64_game_dirs(ai_session: str) -> list:
    """Dirs holding a legacy b64 wire (frames.jsonl with a 'b64' field)."""
    out = []
    for fj in glob.glob(os.path.join(ai_session, "**", "frames.jsonl"), recursive=True):
        gd = os.path.dirname(fj)
        # skip a NEW-layout screenshot index (no 'b64'): peek the first non-empty line
        if _looks_like_wire(fj):
            out.append(gd)
    return sorted(out)


def _looks_like_wire(frames_jsonl: str) -> bool:
    try:
        with open(frames_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                return "b64" in json.loads(line)
    except Exception:
        return False
    return False


def plan_targets(game_dir: str) -> dict:
    """Compute new-layout targets for a b64 game dir (pure; no I/O)."""
    gd = os.path.abspath(game_dir).replace("\\", "/")
    parts = gd.split("/")
    game = parts[-1]
    parent = parts[-2]
    if game.startswith("game"):
        name = f"ai_{parent}_{game}"                 # run_N/gameM -> ai_run_N_gameM
        gt_path = os.path.join(os.path.dirname(game_dir), f"{game}.jsonl")
    else:
        name = f"ai_{game}"                          # run_1 (single-game) -> ai_run_1
        gt_path = os.path.join(os.path.dirname(game_dir), f"{game}.jsonl")
    return {
        "name": name,
        "gt_path": gt_path,
        "wire_dest": os.path.join(game_dir, "liqi.jsonl"),
        "index_path": os.path.join(game_dir, "frames.jsonl"),
    }


def is_migrated(game_dir: str) -> bool:
    """True if this dir already has the new layout (liqi.jsonl + sibling GTRecord)."""
    t = plan_targets(game_dir)
    return os.path.exists(t["wire_dest"]) and os.path.exists(t["gt_path"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="Actually migrate (default: dry run).")
    ap.add_argument("--mjcopilot", default="../MahjongCopilot")
    ap.add_argument("--ai-session", default=paths.RAW_AI_SESSION)
    args = ap.parse_args()

    ai_session = os.path.abspath(args.ai_session)
    game_dirs = find_b64_game_dirs(ai_session)
    todo = [gd for gd in game_dirs if not is_migrated(gd)]
    print(f"{'APPLY' if args.apply else 'DRY RUN'} — {len(game_dirs)} b64 game(s), "
          f"{len(todo)} to migrate, {len(game_dirs) - len(todo)} already done")

    # Import MahjongCopilot + convert once (chdir handled by _import_mjcopilot).
    from scripts.data.convert_mjcopilot import _import_mjcopilot, convert_game
    liqimod = GameState = Bot = GameMode = None
    if todo:
        mc = os.path.abspath(args.mjcopilot)
        liqimod, GameState, Bot, GameMode = _import_mjcopilot(mc)
        os.chdir(mc)

    for gd in todo:
        t = plan_targets(gd)
        print(f"  {t['name']}: {gd}")
        print(f"    -> {t['gt_path']}")
        print(f"    -> rename frames.jsonl -> {t['wire_dest']}")
        print(f"    -> write screenshot index {t['index_path']}")
        if not args.apply:
            continue
        # 1) re-derive GT from the legacy wire (still named frames.jsonl here)
        records, frame_index = convert_game(gd, liqimod, GameState, Bot, GameMode,
                                            wire_name="frames.jsonl")
        # 2) rename the wire BEFORE overwriting frames.jsonl with the index
        os.rename(os.path.join(gd, "frames.jsonl"), t["wire_dest"])
        # 3) write GTRecord + index-relative screenshot index
        write_records(t["gt_path"], records)
        with open(t["index_path"], "w", encoding="utf-8") as f:
            for fi in frame_index:
                seq = fi["seq"]
                f.write(json.dumps({"seq": seq, "file": f"frames/{seq:06d}.png",
                                    "status": "ok"}) + "\n")

    if not args.apply:
        print("\n(dry run — nothing changed; pass --apply to migrate)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit test**

Run: `PYTHONPATH=. $PY tests/test_migrate_ai.py`
Expected: `test_migrate_ai OK`.

- [ ] **Step 5: Back up, then dry-run the real migration**

Back up the legacy GT + raw wire first (into the scratchpad, outside the repo):
```bash
BK="C:/Users/zsx/AppData/Local/Temp/claude/D--code-phoenix-majsoul-eye/aae215c2-c486-43dd-858f-6450440e07c8/scratchpad/pre_migration_backup"
mkdir -p "$BK"
cp -r captures/intermediate/gt "$BK/gt"
find captures/raw/ai_session -name frames.jsonl | while read f; do d="$BK/wire/$(dirname "$f")"; mkdir -p "$d"; cp "$f" "$d/"; done
echo "backup at $BK"
```

Dry run:
```bash
PYTHONPATH=. $PY scripts/data/migrate_ai_to_gtrecord.py
```
Expected: lists ~18 b64 games (run_1, run_3/game1-4, run_4/game1, run_5/game1-3, run_7/game1, run_8/game1-6, run_13/game1, run_14/game1), all "to migrate", and prints the target paths. No files changed.

- [ ] **Step 6: Apply the migration**

```bash
PYTHONPATH=. $PY scripts/data/migrate_ai_to_gtrecord.py --apply
```
Expected: each game printed with its three targets; afterwards every game dir has `liqi.jsonl` + a sibling `gameM.jsonl` (or `run_1.jsonl`) + a `frames.jsonl` index.

Verify a migrated GTRecord matches the golden (byte-identical) for one overlapping game:
```bash
PYTHONPATH=. $PY - <<'PY'
from majsoul_eye.capture.schema import read_records
a = [r.to_json_line() for r in read_records("captures/raw/ai_session/run_3/game1.jsonl")]
b = [r.to_json_line() for r in read_records("captures/intermediate/gt/ai_run_3_game1.jsonl")]
print("migrated", len(a), "golden", len(b), "identical:", a == b)
assert a == b
print("MIGRATION GT OK")
PY
```
Expected: `identical: True` then `MIGRATION GT OK`.

- [ ] **Step 7: Commit (script only — captures/ is gitignored)**

```bash
git add scripts/data/migrate_ai_to_gtrecord.py tests/test_migrate_ai.py
git commit -m "feat(data): one-time migration of legacy b64 AI captures to GTRecord layout"
```

---

## Task 7: Regression, cleanup, and full-suite verification

**Why:** Prove the 16 overlapping games' derived datasets are unchanged, retire `intermediate/gt/`, and confirm the whole test suite is green.

**Files:**
- Modify (docs): `docs/STATUS.md`, `CLAUDE.md` (data-pipeline section), `scripts/README.md` — repoint AI-capture description to the unified path (targeted edits only).
- No production code changes expected here.

- [ ] **Step 1: Frame-index equivalence for a migrated game**

`load_frames` must resolve the migrated (index-relative) frame set to the same PNGs the old (captures-relative) `intermediate/gt` index did:

```bash
PYTHONPATH=. $PY - <<'PY'
from majsoul_eye.capture.gtframes import load_frames
new = load_frames("captures/raw/ai_session/run_3/game1")          # migrated index
old = load_frames("captures/intermediate/gt/ai_run_3_game1")      # legacy index
def norm(d):
    import os
    return {k: os.path.normcase(os.path.abspath(v)) for k, v in d.items()}
print("new", len(new), "old", len(old), "same seqs:", set(new) == set(old))
assert set(new) == set(old)
assert norm(new) == norm(old)
print("FRAME INDEX OK")
PY
```
Expected: same seqs, `FRAME INDEX OK`. (This, plus the Task 6 GT-identity check, means `build_dataset` produces byte-identical crops/labels for that game without a full rebuild.)

- [ ] **Step 2: Run the full annotate→dataset rebuild dry-run, then a real rebuild of one game**

Dry run (confirms discovery + commands reference raw paths):
```bash
PYTHONPATH=. $PY scripts/data/rebuild_datasets.py
```
Expected: 18 AI games listed (16 original + run_13/14 now included), val `ai_run_8_game1`, all commands reference `captures/raw/ai_session/...`.

Real rebuild of a single stage-limited game to smoke the annotate+build path end-to-end (annotate one game, then build it):
```bash
PYTHONPATH=. $PY scripts/annotate/annotate_ai_session.py --captures captures/raw/ai_session/run_3/game1.jsonl --out out/_smoke_ann --workers 1
PYTHONPATH=. $PY scripts/train/build_dataset.py captures/raw/ai_session/run_3/game1.jsonl captures/raw/ai_session/run_3/game1 --out datasets/_smoke_g31 --from-annotations out/_smoke_ann --drop-violations
ls datasets/_smoke_g31/crops | head
```
Expected: `out/_smoke_ann/ai_run_3_game1.jsonl` is written; `build_dataset` reuses it (prints `reuse: N records <- out/_smoke_ann/ai_run_3_game1.jsonl`) and emits crops. Clean up: `rm -rf out/_smoke_ann datasets/_smoke_g31`.

- [ ] **Step 3: Retire `intermediate/gt/`**

The backup exists (Task 6 Step 5). Remove the retired GT dir:
```bash
rm -rf captures/intermediate/gt
echo "removed captures/intermediate/gt"
```
(It is derived + gitignored, so this is not a git change. `intermediate/derived/*_fixed` for the letterboxed games is UNTOUCHED — `FRAMES_OVERRIDE` still needs it.)

Sanity: rebuild discovery still works without `intermediate/gt`:
```bash
PYTHONPATH=. $PY scripts/data/rebuild_datasets.py 2>&1 | head -5
```
Expected: still lists the AI games (they now come from `raw/ai_session`).

- [ ] **Step 4: Update the docs that describe the AI capture path**

Make targeted edits (do not rewrite the files):

- `CLAUDE.md` — in the data-pipeline comment block, change the AI path description from "raw liqi wire → convert_mjcopilot → intermediate/gt" to "autoplay_ai writes GTRecord + screenshot index directly under raw/ai_session (same as manual); no convert step." Update the `paths.py` bullet if it names `converted_gt_captures()`.
- `scripts/README.md` — update any `convert_mjcopilot` / `ingest_run` description to reflect that new captures are already GTRecord and `migrate_ai_to_gtrecord.py` handles legacy b64.
- `docs/STATUS.md` — add a short section noting the unification (AI capture now writes GTRecord inline; `intermediate/gt` retired; `convert_mjcopilot.convert_game` lives on as the shared migration/derivation lib).

- [ ] **Step 5: Run every test suite**

```bash
PYTHONPATH=. $PY tests/test_schema_writer.py && PYTHONPATH=. $PY tests/test_mjcopilot_gt.py && \
PYTHONPATH=. $PY tests/test_autoplay_gt.py && PYTHONPATH=. $PY tests/test_paths.py && \
PYTHONPATH=. $PY tests/test_downstream_rewire.py && PYTHONPATH=. $PY tests/test_migrate_ai.py && \
PYTHONPATH=. $PY tests/test_tiles.py && PYTHONPATH=. $PY tests/test_replay.py && \
PYTHONPATH=. $PY tests/test_sync.py && PYTHONPATH=. $PY tests/test_label.py && \
PYTHONPATH=. $PY tests/test_classifier.py && PYTHONPATH=. $PY tests/test_gamemeta.py && \
PYTHONPATH=. $PY tests/test_annotate_frame.py && echo "ALL SUITES OK"
```
Expected: each prints its `... OK` line, ending with `ALL SUITES OK`. (If a heavy suite like `test_classifier` needs weights/data not present, note it and run the rest.)

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md scripts/README.md docs/STATUS.md
git commit -m "docs: AI capture now writes unified GTRecord; retire intermediate/gt"
```

---

## Manual validation (not automatable here)

The live browser path can only be fully validated by a real capture. After the plan lands, run a short passive capture and confirm the new layout:

```bash
& $PY scripts/capture/autoplay_ai.py            # dry-run; log in + 观战/enter a game
```
Confirm on disk: `captures/raw/ai_session/run_<N>/game1.jsonl` (GTRecord, grows during play), `game1/liqi.jsonl` (wire), `game1/frames.jsonl` (index grows as screenshots save), `game1/frames/*.png`, `game1/metadata.json`. Then Ctrl-C **mid-game** and confirm the partial `game1.jsonl` is still valid GTRecord (`read_records` loads it, `Replayer.check_invariants()` passes) — the abnormal-exit requirement.

---

## Self-Review

**Spec coverage:**
- §4 mechanism (inline incremental, GTWriter, make_capturing_game_state, emit-on-new-mjai) → Tasks 1, 2, 3. ✓
- §5 layout (liqi.jsonl, gameM.jsonl, incremental frames.jsonl, GTWriter move) → Tasks 1, 3. ✓
- §6 shared code + downstream rewire (mjcopilot_gt, paths, build_dataset, annotate, rebuild, ingest) → Tasks 2, 4, 5. ✓
- §7 legacy migration (all b64 games, dry-run, idempotent, reuse convert_game) → Task 6. ✓
- §8 testing (equivalence, round-trip, regression, suites) → Task 2 Step 6, Task 6 Step 6, Task 7 Steps 1-2 + 5. ✓
- §9 non-goals (no recognize/ change, no retrain, manual path behavior preserved) → respected; only GTWriter *moves* (behavior-preserving, Task 1). ✓
- §10 risks (live≠offline derivation, hot-path regression, migration corruption) → equivalence test, try/except + background writer, dry-run/idempotent/wire-kept. ✓

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to"; every code step shows full code. ✓

**Type consistency:** `make_capturing_game_state -> (gs, drain_mjai)` and `drain_mjai() -> list` used identically in Task 2 (convert_game) and Task 3 (write_gt). `gt_fields(msg) -> (method, action_name)` used in both. `paths.ai_game_name` / `paths.ai_captures` / `paths.frames_dir_for` names consistent across Tasks 4, 5, 6. `_frame_index_line(seq, ts)` defined and tested in Task 3. `plan_targets`/`is_migrated` defined and tested in Task 6. ✓
