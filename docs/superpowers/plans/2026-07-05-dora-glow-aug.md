# Dora-glow augmentation + coverage stats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the detector explicit, logged augmentation defaults (turn off the wrong `fliplr`, push `hsv_v` as a brightness/dora-glow proxy) and add a QA tool that measures per-class dora-glow coverage, so we can decide from evidence whether a real localized-bloom augmentation is worth building.

**Architecture:** Three small, independent changes. (1) Two pure taxonomy helpers in `tiles.py` (`next_of`, `dora_names`) that encode the standard dora progression. (2) A one-shot QA script `scripts/inspect/count_dora_glow.py` that walks GT captures (Akagi-free, reusing `capture.gtframes` + `build_datasets.discover_games`) and tallies per-class glowing vs total labeled-box instances, split train/val by whole game. (3) `train_detector.py` promotes YOLO aug hyperparameters to CLI flags with corrected defaults, passed explicitly and logged.

**Tech Stack:** Python 3.12, ultralytics 8.4.86, the majsoul_eye package (`tiles`, `state.replay`, `capture.gtframes`, `paths`). Tests are plain scripts (also pytest-compatible), no GPU/frames needed for the unit tests.

## Global Constraints

- **Env python (Linux box):** `PY=/hszhao-f1/h3011050/anaconda3/envs/majsoul_eye/bin/python`. The docs' Windows path `C:/Users/zsx/miniforge3/envs/auto/python.exe` is DEAD here — do not use it.
- **Always run from repo root with `PYTHONPATH=.`** (imports are top-level `from majsoul_eye import ...`).
- **38-class order is frozen** (`tiles.TILE_NAMES`) — do NOT reorder; only ADD helper functions to `tiles.py`.
- **Replay tiles are MJAI notation** (`"0m"` red five, honors may be `"E"..`/`"1z".."7z"`). Always normalize with `tiles.from_mjai(...)` before class/glow logic.
- **Glow definition (frozen for this work):** a tile glows iff it is a red five (always) OR its plain value equals `next_of(indicator)` for some active dora indicator. Indicator-strip tiles and `back` never glow and are excluded from the counted population.
- **Counts are per-frame labeled-box instances** (= training crops), split train/val by whole game; val default `ai_run_8_game1`.
- **Commits:** never add a `Co-Authored-By: Claude` (or any AI) trailer. Work on branch `feat/dora-glow-aug` (already created).
- **Pipeline discipline:** `train_detector.py` is a training stage whose defaults change → update `docs/PIPELINE.md` (§2 detector-train) + add a `docs/STATUS.md` entry. `count_dora_glow.py` is a new one-shot tool → classify it in `docs/PIPELINE.md` §4 (per §6 rule 3). No derived data under `datasets/` is staled (aug change only affects the *next* training run).

---

### Task 1: `next_of` + `dora_names` dora-progression helpers in `tiles.py`

**Files:**
- Modify: `majsoul_eye/tiles.py` (append after `red_to_normal`, ~line 85)
- Test: `tests/test_dora_glow.py` (create)

**Interfaces:**
- Consumes: existing `tiles.from_mjai`, `tiles.red_to_normal`, `tiles.is_red_five`.
- Produces:
  - `next_of(indicator: str) -> str` — canonical dora tile pointed at by a dora indicator. Tolerates MJAI or canonical, red or not. Suits wrap within suit (`9m`→`1m`); winds `E→S→W→N→E`; dragons `P→F→C→P`.
  - `dora_names(indicators) -> set[str]` — set of canonical dora tile names for an iterable of indicators.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dora_glow.py`:

```python
"""Unit tests for the dora-glow logic (tiles.next_of / tiles.dora_names) that
drives scripts/inspect/count_dora_glow.py. Pure — no frames, no GPU."""

from majsoul_eye.tiles import (
    next_of, dora_names, from_mjai, is_red_five, red_to_normal,
)


def test_next_of_suits_wrap():
    assert next_of("1m") == "2m"
    assert next_of("8p") == "9p"
    assert next_of("9m") == "1m"       # wrap within suit
    assert next_of("9p") == "1p"
    assert next_of("9s") == "1s"
    assert next_of("5s") == "6s"


def test_next_of_winds_cycle():
    assert next_of("E") == "S"
    assert next_of("S") == "W"
    assert next_of("W") == "N"
    assert next_of("N") == "E"         # wrap


def test_next_of_dragons_cycle():
    assert next_of("P") == "F"         # 白 -> 發
    assert next_of("F") == "C"         # 發 -> 中
    assert next_of("C") == "P"         # 中 -> 白 (wrap)


def test_next_of_red_five_indicator_counts_as_plain():
    assert next_of("5mr") == "6m"      # canonical red
    assert next_of("0m") == "6m"       # MJAI red
    assert next_of("0p") == "6p"
    assert next_of("0s") == "6s"


def test_next_of_accepts_mjai_honors():
    assert next_of("1z") == "S"        # MJAI East -> South
    assert next_of("5z") == "F"        # MJAI 白 -> 發
    assert next_of("7z") == "P"        # MJAI 中 -> 白


def test_dora_names_mixed_indicators():
    # 4m -> 5m; E -> S; 0p(red5p) -> 6p
    assert dora_names(["4m", "E", "0p"]) == {"5m", "S", "6p"}


def test_dora_names_empty():
    assert dora_names([]) == set()


def _glows(raw_tile, dset):
    """The exact per-tile glow expression used by count_dora_glow.py."""
    canon = from_mjai(raw_tile)
    return is_red_five(canon) or red_to_normal(canon) in dset


def test_glow_rule_red_five_always():
    assert _glows("0m", set()) is True                   # red five glows w/ no dora
    assert _glows("5mr", dora_names(["1m"])) is True


def test_glow_rule_dora_match():
    dset = dora_names(["4m"])                             # dora = 5m
    assert _glows("5m", dset) is True                    # plain 5m glows
    assert _glows("0m", dset) is True                    # red 5m also glows
    assert _glows("6m", dset) is False                   # non-dora


def test_glow_rule_honor_dora():
    dset = dora_names(["S"])                              # indicator S -> dora W
    assert _glows("W", dset) is True
    assert _glows("N", dset) is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_dora_glow OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /hszhao-f1/h3011050/workspace/phoenix/phoenix-server/majsoul_eye && PY=/hszhao-f1/h3011050/anaconda3/envs/majsoul_eye/bin/python; PYTHONPATH=. $PY tests/test_dora_glow.py`
Expected: FAIL — `ImportError: cannot import name 'next_of' from 'majsoul_eye.tiles'`.

- [ ] **Step 3: Write minimal implementation**

In `majsoul_eye/tiles.py`, insert immediately after the `red_to_normal` function (after its `return name` at ~line 85), before `name_of`:

```python
# --- dora progression -------------------------------------------------------
# A dora indicator points at the NEXT tile in its cyclic group: suited
# 1->2->...->9->1 within the suit; winds E->S->W->N->E; dragons P->F->C->P
# (白->發->中->白). A red-five indicator counts as its plain five. Used to decide
# which tiles carry Majsoul's dora glow (see scripts/inspect/count_dora_glow.py).
_WIND_CYCLE: list[str] = ["E", "S", "W", "N"]
_DRAGON_CYCLE: list[str] = ["P", "F", "C"]


def next_of(indicator: str) -> str:
    """Canonical dora tile pointed at by a dora *indicator*.

    Tolerates MJAI or canonical, red or not (``'0m'``/``'5mr'`` -> ``'6m'``,
    ``'1z'`` -> ``'S'``). Suits wrap within their suit (``9m`` -> ``1m``); winds
    cycle E->S->W->N->E; dragons cycle P->F->C->P. Returns a canonical, non-red
    tile name.
    """
    name = red_to_normal(from_mjai(indicator))     # '0m'/'5mr' -> '5m'; '1z' -> 'E'
    if name in _WIND_CYCLE:
        return _WIND_CYCLE[(_WIND_CYCLE.index(name) + 1) % len(_WIND_CYCLE)]
    if name in _DRAGON_CYCLE:
        return _DRAGON_CYCLE[(_DRAGON_CYCLE.index(name) + 1) % len(_DRAGON_CYCLE)]
    num, suit = int(name[0]), name[1]              # suited 'Xm'/'Xp'/'Xs'
    return f"{1 if num == 9 else num + 1}{suit}"


def dora_names(indicators) -> set[str]:
    """Set of canonical dora tile names for an iterable of dora indicators
    (each MJAI or canonical, red or not)."""
    return {next_of(i) for i in indicators}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /hszhao-f1/h3011050/workspace/phoenix/phoenix-server/majsoul_eye && PY=/hszhao-f1/h3011050/anaconda3/envs/majsoul_eye/bin/python; PYTHONPATH=. $PY tests/test_dora_glow.py`
Expected: PASS — prints `test_dora_glow OK`.

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/tiles.py tests/test_dora_glow.py
git commit -m "feat(tiles): next_of/dora_names dora-progression helpers + tests"
```

---

### Task 2: `count_dora_glow.py` per-class glow coverage tool

**Files:**
- Create: `scripts/inspect/count_dora_glow.py`
- Modify: `docs/PIPELINE.md` (§4 one-shot-tool classification)
- Test: smoke run on one game (no unit-test file — the pure logic is already covered by Task 1)

**Interfaces:**
- Consumes: `tiles.next_of`/`dora_names`/`from_mjai`/`is_red_five`/`red_to_normal`/`TILE_NAMES` (Task 1), `capture.gtframes.build_seq_state`/`load_frames`, `scripts.data.build_datasets.discover_games`, `paths.RAW_AI_SESSION`. Reads `BoardState.hero_hand`, `.rivers[seat][i].pai`, `.melds[seat][j].tiles`, `.dora_markers`.
- Produces: a CLI tool (stdout only, no artifacts). No importable API relied on by later tasks.

- [ ] **Step 1: Write the tool**

Create `scripts/inspect/count_dora_glow.py`:

```python
"""Per-class dora-glow coverage stats for the detector training data.

For each of the 38 tile classes, count how many GLOWING vs total labeled-box
instances the detector sees, and flag classes starved of glowing examples —
evidence for whether a real localized-bloom augmentation is worth building
(see docs/superpowers/specs/2026-07-05-dora-glow-aug-design.md). Reads GT
captures the same way the annotator / dataset builder do (Akagi-free).

A tile GLOWS when it is a red five (always aka dora) or its value matches the
current dora (``tiles.next_of(indicator)``). Only glow-eligible zones (hero
hand / river / meld) are counted; the dora-indicator strip and face-down 'back'
tiles are a different visual population and are excluded. Counts are per-frame
labeled-box instances (= training crops), NOT de-duplicated physical tiles —
that is the right denominator for "does the model see enough glowing X". Split
train/val by whole game (val default ai_run_8_game1).

    PYTHONPATH=. python scripts/inspect/count_dora_glow.py
    PYTHONPATH=. python scripts/inspect/count_dora_glow.py \
        --sources captures/raw/ai_session captures/raw/manual
    PYTHONPATH=. python scripts/inspect/count_dora_glow.py --dataset datasets/v1 --min-glow 30
"""
from __future__ import annotations

import argparse
import json
import os

from majsoul_eye import paths
from majsoul_eye.capture.gtframes import build_seq_state, load_frames
from majsoul_eye.tiles import (
    TILE_NAMES, from_mjai, is_red_five, red_to_normal, dora_names,
)

# match the dataset builder's frame set (build_dataset uses ('ok', 'timeout'))
FRAME_STATUSES = ("ok", "timeout")


def glow_eligible_tiles(state):
    """The raw (MJAI) tile strings of every glow-eligible box in one frame:
    hero hand + every seat's river + every seat's meld tiles. Excludes the dora
    indicator strip (never glows) and 'back' (face-down, filtered by caller)."""
    out = list(state.hero_hand)
    for river in state.rivers:
        out.extend(rt.pai for rt in river)
    for melds in state.melds:
        for m in melds:
            out.extend(m.tiles)
    return out


def count_game(capture: str, frames_dir: str):
    """Return {class_name: [total, glow]} for one game (per-frame box instances)."""
    seq_state = build_seq_state(capture)
    frames = load_frames(frames_dir, statuses=FRAME_STATUSES)
    tally = {name: [0, 0] for name in TILE_NAMES}
    for seq in frames:                       # only seqs that have a saved frame
        state = seq_state.get(seq)
        if state is None:
            continue
        dset = dora_names(state.dora_markers)
        for raw in glow_eligible_tiles(state):
            canon = from_mjai(raw)
            if canon == "back":              # defensive; hand/river/meld never back
                continue
            glow = is_red_five(canon) or red_to_normal(canon) in dset
            tally[canon][0] += 1
            if glow:
                tally[canon][1] += 1
    return tally


def load_games(args):
    """Return (games, val_name). games = list of {name, capture, frames_dir}."""
    if args.dataset:
        with open(os.path.join(args.dataset, "games.json"), encoding="utf-8") as f:
            man = json.load(f)
        return man["games"], man.get("val", args.val)
    # reuse the builder's discovery so this matches what actually gets trained on
    from scripts.data.build_datasets import discover_games
    return discover_games(args.sources), args.val


def _merge(dst, src):
    for name, (t, g) in src.items():
        dst[name][0] += t
        dst[name][1] += g


def print_table(title, tally, min_glow):
    print(f"\n=== {title} ===")
    print(f"{'class':>6} | {'total':>8} | {'glow':>7} | {'glow%':>6}")
    print("-" * 38)
    starved, tot_all, glow_all = [], 0, 0
    for name in TILE_NAMES:
        t, g = tally[name]
        tot_all += t
        glow_all += g
        pct = (100.0 * g / t) if t else 0.0
        flag = "  <-- starved" if g < min_glow else ""
        print(f"{name:>6} | {t:>8} | {g:>7} | {pct:>5.1f}%{flag}")
        if g < min_glow:
            starved.append((name, g))
    print("-" * 38)
    op = (100.0 * glow_all / tot_all) if tot_all else 0.0
    print(f"{'TOTAL':>6} | {tot_all:>8} | {glow_all:>7} | {op:>5.1f}%")
    if starved:
        print(f"\n{len(starved)} class(es) with glow < {min_glow}: "
              + ", ".join(f"{n}({g})" for n, g in starved))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", nargs="+", default=[paths.RAW_AI_SESSION],
                    help="capture roots to scan (default: captures/raw/ai_session)")
    ap.add_argument("--dataset", default=None,
                    help="datasets/<v>: read its games.json for the exact game set + val "
                         "(overrides --sources)")
    ap.add_argument("--val", default="ai_run_8_game1",
                    help="held-out game name counted as val (default ai_run_8_game1)")
    ap.add_argument("--min-glow", type=int, default=20,
                    help="flag classes whose glow count is below this (default 20)")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N games (0 = all; for a quick smoke run)")
    args = ap.parse_args()

    games, val = load_games(args)
    if args.limit:
        games = games[:args.limit]
    train = {name: [0, 0] for name in TILE_NAMES}
    valt = {name: [0, 0] for name in TILE_NAMES}
    for g in games:
        tally = count_game(g["capture"], g["frames_dir"])
        _merge(valt if g["name"] == val else train, tally)
        print(f"  counted {g['name']:>18}  "
              f"(total={sum(t for t, _ in tally.values())}, "
              f"glow={sum(gl for _, gl in tally.values())})")

    print_table(f"TRAIN (all games except {val})", train, args.min_glow)
    print_table(f"VAL ({val})", valt, args.min_glow)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on one game**

Run: `cd /hszhao-f1/h3011050/workspace/phoenix/phoenix-server/majsoul_eye && PY=/hszhao-f1/h3011050/anaconda3/envs/majsoul_eye/bin/python; PYTHONPATH=. $PY scripts/inspect/count_dora_glow.py --limit 1`
Expected: prints one `counted ai_run_1 (total=..., glow=...)` line with non-zero totals, then a `TRAIN` table of 38 rows ending in a `TOTAL` row, then an (empty) `VAL` table. No traceback.

- [ ] **Step 3: Sanity-check the numbers**

Run: `cd /hszhao-f1/h3011050/workspace/phoenix/phoenix-server/majsoul_eye && PY=/hszhao-f1/h3011050/anaconda3/envs/majsoul_eye/bin/python; PYTHONPATH=. $PY scripts/inspect/count_dora_glow.py --limit 1 2>&1 | grep -E "5mr|5pr|5sr|TOTAL"`
Expected: the three red-five rows have `glow% = 100.0%` (red fives always glow) whenever their `total > 0`; the `TOTAL` glow% is a small positive number (dora are a minority of tiles). If a red-five row shows total>0 but glow<total, the glow rule is wrong — stop and fix.

- [ ] **Step 4: Classify the tool in PIPELINE.md §4**

In `docs/PIPELINE.md`, inside the §4 table ("过时/降级组件清单（勿再当作管线环节）"), add a row at the end of the table (after the `label/（autolabel.py）` row):

```markdown
| `scripts/inspect/count_dora_glow.py` | **现役一次性诊断工具**（非管线环节）：统计每个 tile 类别的「发光实例/总实例」覆盖，判断是否需要为宝牌闪光加专门增强。读 GT 采集（Akagi-free），纯 stdout。见 `docs/superpowers/specs/2026-07-05-dora-glow-aug-design.md` |
```

- [ ] **Step 5: Commit**

```bash
git add scripts/inspect/count_dora_glow.py docs/PIPELINE.md
git commit -m "feat(inspect): count_dora_glow per-class glow coverage tool"
```

---

### Task 3: explicit, corrected augmentation config in `train_detector.py`

**Files:**
- Modify: `scripts/train/train_detector.py` (refactor `main()` into `build_parser` + `build_train_kwargs`)
- Modify: `docs/PIPELINE.md` (§2 detector-train note) and `docs/STATUS.md` (new entry)
- Test: `tests/test_train_detector_aug.py` (create)

**Interfaces:**
- Consumes: existing `resolve_device(device_arg, cuda_available)`.
- Produces:
  - `build_parser() -> argparse.ArgumentParser` — the full CLI incl. new aug flags.
  - `build_train_kwargs(args, device) -> tuple[dict, dict]` — returns `(kw, aug)` where `kw` is the full `model.train(**kw)` kwargs (including the aug block + optional `project`) and `aug` is just the aug sub-dict (for logging).

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_detector_aug.py`:

```python
"""Detector aug-config tests: the corrected defaults (fliplr off, hsv_v boosted)
and overridability. No torch/ultralytics/GPU — build_train_kwargs is pure."""

from scripts.train.train_detector import build_parser, build_train_kwargs


def test_aug_defaults_turn_off_fliplr_and_boost_value():
    args = build_parser().parse_args(["--data", "d.yaml"])
    kw, aug = build_train_kwargs(args, device=0)
    assert kw["fliplr"] == 0.0        # directional tiles: no mirror flip
    assert kw["flipud"] == 0.0        # never wanted on a top-down board
    assert kw["hsv_v"] == 0.5         # brightness / dora-glow proxy (was YOLO 0.4)
    assert kw["hsv_s"] == 0.7         # unchanged default, now explicit
    assert kw["mosaic"] == 1.0
    assert kw["close_mosaic"] == 10
    assert kw["data"] == "d.yaml"
    assert "project" not in kw        # not set unless --project given
    assert aug["fliplr"] == 0.0 and "data" not in aug   # aug is the sub-dict only


def test_aug_overridable_from_cli():
    args = build_parser().parse_args(
        ["--data", "d.yaml", "--fliplr", "0.5", "--hsv-v", "0.9", "--project", "runs/x"])
    kw, _ = build_train_kwargs(args, device="0,1")
    assert kw["fliplr"] == 0.5
    assert kw["hsv_v"] == 0.9
    assert kw["device"] == "0,1"
    assert kw["project"] == "runs/x"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_train_detector_aug OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /hszhao-f1/h3011050/workspace/phoenix/phoenix-server/majsoul_eye && PY=/hszhao-f1/h3011050/anaconda3/envs/majsoul_eye/bin/python; PYTHONPATH=. $PY tests/test_train_detector_aug.py`
Expected: FAIL — `ImportError: cannot import name 'build_parser' from 'scripts.train.train_detector'`.

- [ ] **Step 3: Refactor `main()` and add aug flags**

In `scripts/train/train_detector.py`, replace the whole `def main() -> None:` function (from `def main() -> None:` down to just before `if __name__ == "__main__":`) with the three functions below. The augmentation flags are new; `resolve_device` above is unchanged.

```python
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml from build_detector_dataset.py")
    ap.add_argument("--model", default="weights/pretrained/yolov8s.pt",
                    help="base weights / arch. A bare name (yolov8s.pt / yolo11s-obb.pt) makes "
                         "ultralytics auto-download to cwd; prefer weights/pretrained/<name> to "
                         "keep base seeds under weights/ (see weights/README.md)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=100,
                    help="early-stop after N epochs without val-mAP gain (ultralytics default 100)")
    ap.add_argument("--batch", type=int, default=8, help="images per batch (-1 = auto-batch); "
                    "with multi-GPU --device this is the GLOBAL batch, split across GPUs")
    ap.add_argument("--device", default="",
                    help="CUDA device(s): '' auto (GPU0/CPU), '0', '0,1,2,3' for DDP, 'cpu'")
    ap.add_argument("--project", default="", help="run dir parent (default: ultralytics runs/detect)")
    ap.add_argument("--name", default="tile_detector")
    ap.add_argument("--out", default="majsoul_eye/recognize/tile_detector.pt")
    # --- augmentation (explicit; ultralytics detect defaults EXCEPT fliplr/hsv_v) ---
    # fliplr defaults to 0.0 (NOT ultralytics' 0.5): mahjong tiles are directional, so
    # a horizontal flip fabricates mirror tiles that never occur in reality. hsv_v is
    # pushed to 0.5 (from 0.4) as a global brightness / dora-glow robustness proxy —
    # Majsoul renders a golden bloom on dora tiles, which we don't model otherwise
    # (see docs/superpowers/specs/2026-07-05-dora-glow-aug-design.md). All other knobs
    # keep the ultralytics detect defaults but are now exposed + logged.
    ap.add_argument("--fliplr", type=float, default=0.0, help="P(horizontal flip); 0 for directional tiles")
    ap.add_argument("--hsv-v", type=float, default=0.5, help="HSV-Value jitter (brightness/glow proxy)")
    ap.add_argument("--hsv-s", type=float, default=0.7, help="HSV-Saturation jitter")
    ap.add_argument("--hsv-h", type=float, default=0.015, help="HSV-Hue jitter")
    ap.add_argument("--degrees", type=float, default=0.0, help="rotation degrees")
    ap.add_argument("--translate", type=float, default=0.1, help="translation fraction")
    ap.add_argument("--scale", type=float, default=0.5, help="scale gain")
    ap.add_argument("--mosaic", type=float, default=1.0, help="P(mosaic)")
    ap.add_argument("--close-mosaic", type=int, default=10, help="disable mosaic for last N epochs")
    ap.add_argument("--mixup", type=float, default=0.0, help="P(mixup)")
    return ap


def build_train_kwargs(args, device):
    """Assemble model.train(**kw); returns (kw, aug) with aug the loggable sub-dict.

    flipud is pinned to 0.0 (never wanted on a top-down board) and not exposed.
    """
    aug = dict(fliplr=args.fliplr, flipud=0.0, hsv_h=args.hsv_h, hsv_s=args.hsv_s,
               hsv_v=args.hsv_v, degrees=args.degrees, translate=args.translate,
               scale=args.scale, mosaic=args.mosaic, close_mosaic=args.close_mosaic,
               mixup=args.mixup)
    kw = dict(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
              patience=args.patience, device=device, name=args.name, **aug)
    # project defaults to ultralytics' own runs/detect; passing our own "runs/detect"
    # here would nest it (runs/detect/runs/detect/...), so only override when non-empty.
    if args.project:
        kw["project"] = args.project
    return kw, aug


def main() -> None:
    args = build_parser().parse_args()

    import torch
    from ultralytics import YOLO

    device = resolve_device(args.device, torch.cuda.is_available())
    if device == "cpu":
        desc = "CPU"
    else:
        desc = ", ".join(f"cuda:{i} {torch.cuda.get_device_name(i)}"
                         for i in (int(x) for x in str(device).split(",")))
    print(f"device={device} ({desc})  model={args.model}  imgsz={args.imgsz}  "
          f"epochs={args.epochs}  batch={args.batch}", flush=True)

    kw, aug = build_train_kwargs(args, device)
    print("aug: " + " ".join(f"{k}={v}" for k, v in aug.items()), flush=True)

    model = YOLO(args.model)
    model.train(**kw)

    best = getattr(getattr(model, "trainer", None), "best", None)
    if best and os.path.exists(str(best)):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        shutil.copy(str(best), args.out)
        print(f"\nbest weights {best} -> {args.out}")
    else:
        print(f"\nWARNING: best.pt not found (trainer.best={best}); "
              f"look under {args.project}/{args.name}/weights/")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /hszhao-f1/h3011050/workspace/phoenix/phoenix-server/majsoul_eye && PY=/hszhao-f1/h3011050/anaconda3/envs/majsoul_eye/bin/python; PYTHONPATH=. $PY tests/test_train_detector_aug.py`
Expected: PASS — prints `test_train_detector_aug OK`.

- [ ] **Step 5: Verify the CLI still parses (help) and no long train is triggered**

Run: `cd /hszhao-f1/h3011050/workspace/phoenix/phoenix-server/majsoul_eye && PY=/hszhao-f1/h3011050/anaconda3/envs/majsoul_eye/bin/python; PYTHONPATH=. $PY scripts/train/train_detector.py --help`
Expected: help text listing `--fliplr`, `--hsv-v`, `--hsv-s`, `--mosaic`, `--close-mosaic`, etc. No training runs.

- [ ] **Step 6: Update PIPELINE.md §2 detector-train note**

In `docs/PIPELINE.md` §2 ("装配 + 训练"), the detector bullet currently reads (line ~110):

```markdown
- 检测器：`train_detector.py --data datasets/<name>/detector/data.yaml`（imgsz 1280；16GiB 卡加
  `--batch 4` + expandable_segments 防 OOM；OBB 用 `--model weights/pretrained/yolov8s-obb.pt`）。
```

Append one sentence to that bullet (keep the existing text; add after the closing `）。`):

```markdown
  **增强现为显式 CLI**（`--fliplr/--hsv-v/--hsv-s/--mosaic/...`，启动日志打印 `aug:` 行）：
  默认 `fliplr=0`（麻将牌有方向，水平翻转造镜像牌）、`hsv_v=0.5`（亮度/宝牌闪光近似），
  其余沿用 ultralytics detect 默认。是否加真·局部 bloom 由 `count_dora_glow.py` 覆盖统计决定。
```

- [ ] **Step 7: Add a STATUS.md entry**

In `docs/STATUS.md`, under `## 一、已完成`, add a new subsection. Find the highest existing `### 1.NN` number in the file (currently up to §1.29 per the header) and use the next integer. Insert this block (replace `1.30` with the actual next number if different):

```markdown
### 1.30 检测器增强显式化 + 宝牌闪光覆盖统计（2026-07-05）
- **问题**：`train_detector.py` 是 ultralytics 薄封装、不传任何增强超参，全用 YOLOv8 默认
  （`fliplr=0.5`、`hsv_v=0.4`）。麻将牌有方向，`fliplr=0.5` 会造现实不存在的镜像牌；且雀魂
  宝牌有金色闪光特效，训练无任何针对性增强，也无证据判断自然覆盖是否足够。
- **处理**：(1) `tiles.py` 加 `next_of`/`dora_names`（标准 dora 递进，MJAI/canonical 双容）；
  (2) 新增一次性诊断工具 `scripts/inspect/count_dora_glow.py`，统计每类「发光实例/总实例」
  覆盖（红五恒亮 + 指示牌命中的牌算亮；hand/river/meld 区；帧级 = 训练 crop 数；按整局分
  train/val）；(3) `train_detector.py` 把增强超参提升为显式 CLI 参数并打印，默认改
  `fliplr=0→关`、`hsv_v=0.4→0.5`（亮度/闪光近似），其余沿用默认。真·局部 bloom 暂不做。
- **验证**：`tests/test_dora_glow.py`（next_of/dora_names/glow 规则）与
  `tests/test_train_detector_aug.py`（默认关 fliplr、hsv_v=0.5、可覆盖）均通过；
  `count_dora_glow.py` 在采集上跑通（红五行 glow%=100）。
- **结果**：检测器增强可复现、方向错误已修；宝牌闪光覆盖有了量化口径，是否投入合成 bloom
  等统计结果定。**注意**：改的是「下次」检测器训练的默认，不 stale 现有 `datasets/`；已训权重
  需重训才享受。
```

Also update the STATUS.md header's "最后更新对应进度" line (near the top) to point at the new subsection, mirroring the existing style (prepend the new item, demote the current one to "此前").

- [ ] **Step 8: Commit**

```bash
git add scripts/train/train_detector.py tests/test_train_detector_aug.py docs/PIPELINE.md docs/STATUS.md
git commit -m "feat(train): explicit detector aug (fliplr off, hsv_v boosted) + docs"
```

---

### Task 4: Full test sweep

**Files:** none (verification only)

- [ ] **Step 1: Run every test**

Run: `cd /hszhao-f1/h3011050/workspace/phoenix/phoenix-server/majsoul_eye && PY=/hszhao-f1/h3011050/anaconda3/envs/majsoul_eye/bin/python; for t in tests/test_*.py; do echo "== $t =="; PYTHONPATH=. $PY "$t" || { echo "FAILED: $t"; break; }; done`
Expected: every test file prints its `... OK` line; no `FAILED:`.

- [ ] **Step 2: Confirm no stray artifacts / clean tree**

Run: `cd /hszhao-f1/h3011050/workspace/phoenix/phoenix-server/majsoul_eye && git status --short`
Expected: clean (all changes committed across Tasks 1-3); no untracked debug/output files.

---

## Self-Review

**1. Spec coverage:**
- Component 1 (stats tool) → Task 2 (+ helpers in Task 1). Glow rule, zones, per-frame granularity, train/val split, `--min-glow`, sources vs `--dataset` — all present. ✓
- Component 2 (explicit aug) → Task 3. `fliplr`→0, `hsv_v`→0.5, others exposed, `flipud` pinned 0, log line, no custom bloom. ✓
- `next_of` helper (spec §Component 1) → Task 1. Suits/winds/dragons/red wrap. ✓
- Pipeline discipline (spec) → Task 2 Step 4 (§4 classify) + Task 3 Steps 6-7 (PIPELINE §2 + STATUS). ✓
- Testing (spec) → Task 1 (next_of/dora_names/glow), Task 2 (smoke), Task 3 (aug kwargs), Task 4 (sweep). ✓
- Deferred bloom question → left open (spec §Open question); no task, by design. ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N". All code blocks complete. The only variable is the STATUS.md subsection number (`1.30`) — Step 7 gives an explicit rule to resolve it (next integer after the current max). ✓

**3. Type consistency:** `next_of(str)->str`, `dora_names(iterable)->set[str]`, `build_parser()->ArgumentParser`, `build_train_kwargs(args, device)->(dict, dict)`, `count_game(capture, frames_dir)->dict[str,[int,int]]` — names/signatures match between definition and use. `from_mjai`/`red_to_normal`/`is_red_five`/`TILE_NAMES` used exactly as they exist in `tiles.py`. Field accesses (`state.hero_hand`, `state.rivers[i].pai` via `RiverTile.pai`, `state.melds[i].tiles`, `state.dora_markers`) verified against `state/replay.py`. ✓
