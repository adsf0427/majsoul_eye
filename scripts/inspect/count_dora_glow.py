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
    hero hand + every seat's visible river + every seat's meld tiles. Excludes
    the dora indicator strip (never glows) and 'back' (face-down, filtered by
    caller). Rivers use visible_river() to exclude called-away tiles (which are
    counted in the meld zone instead, avoiding double-counting)."""
    out = list(state.hero_hand)
    for seat in range(len(state.rivers)):
        # visible_river excludes called-away tiles: a called tile is physically
        # moved into the caller's meld, so counting it in both river and meld
        # would double-count (mirrors replay.check_invariants / annotate.seat_gt).
        out.extend(rt.pai for rt in state.visible_river(seat))
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
            if canon == "back":              # unreachable in practice ('back' has no MJAI form); defensive only
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
