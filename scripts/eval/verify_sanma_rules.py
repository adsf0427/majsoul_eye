"""Sanma physics gate — the measured laws the 3P recognizer/reconstructor rely on.

Every sanma constant in ``state/observe.py`` and ``state/reconstruct.py`` is a
CLAIM ABOUT MAJSOUL, not a rule of arithmetic. Each one is cheap to get subtly
wrong and expensive to detect afterwards (a wrong wall constant does not raise —
it silently rejects, or silently accepts, real boards). This script measures each
claim against the captured 3P ground truth, so the constants can never drift away
from the game without a test going red.

  V1 wall    live wall is 55, and a nukidora costs one tile like a kan:
             L + Σ|visible river| + kans + Σnuki + (1 if hero holds a draw) ∈ {54,55}
             (the ±1 is the §1.53 pixel=GT-1 counter timing, same as observe.py)
  V2 dora    a nukidora NEVER flips a dora; per game #dora events == #kan events
  V3 turn    a nukidora does not end the turn: it is immediately preceded AND
             followed by a tsumo of the same actor
  V4 tiles   no chi, and no 2m-8m (hence no 5m, no 5mr) anywhere
  V5 oya     oya == (kyoku-1) % 3, kyoku ∈ {1,2,3}, bakaze ∈ {E,S}
  V6 shape   start_kyoku carries 4-wide scores [35000,35000,35000,0] and 4 tehais
  V7 phantom absolute chair 3 never acts and never holds anything
  V8 budget  the 4-copies-per-tile budget INCLUDES the nukidora piles

Usage:
  PYTHONPATH=. python scripts/eval/verify_sanma_rules.py \
      --captures captures/raw/ai_session_3p
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

from majsoul_eye.capture.schema import read_records
from majsoul_eye.state.replay import (Replayer, is_deal_window)
from majsoul_eye.tiles import red_to_normal

SANMA_WALL = 55                 # 108 tiles - 14 dead wall - 3*13 haipai
KAN_TYPES = ("ankan", "kakan", "daiminkan")
ILLEGAL_MANZU = re.compile(r"^[2-8]m$|^5mr$")


def find_captures(roots):
    out = []
    for root in roots:
        if os.path.isfile(root):
            out.append(root)
        else:
            out += sorted(glob.glob(os.path.join(root, "**", "game*.jsonl"),
                                    recursive=True))
    return [p for p in out if "frames" not in os.path.basename(p)]


def _tiles_in(ev):
    """Every tile string anywhere in an mjai event (pai / consumed / tehais / dora)."""
    out = []
    def walk(v):
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    for key in ("pai", "consumed", "tehais", "dora_marker", "target_pai"):
        if key in ev:
            walk(ev[key])
    return out


def check_game(path, report):
    """One capture -> violations appended to report. Returns True if it is sanma."""
    events = []                          # flat mjai stream, in order
    ticks = []                           # (seq, BoardState) at every record
    rp = Replayer()
    for rec in read_records(path):
        rp.apply_record(rec)
        for ev in (rec.mjai or []):
            events.append(ev)
        ticks.append((rec.seq, rp.state.copy()))

    sanma = any(t.sanma for _, t in ticks)
    if not sanma:
        report["skipped_4p"] += 1
        return False
    report["games"] += 1
    tag = os.path.basename(os.path.dirname(path))

    def fail(rule, msg):
        report["violations"].append(f"[{rule}] {tag}: {msg}")
        report["by_rule"][rule] = report["by_rule"].get(rule, 0) + 1

    # --- V1: wall identity, at every in-round tick with a fresh counter -------
    for seq, s in ticks:
        if not s.in_round or is_deal_window(s) or s.left_tile_count is None:
            continue
        kans = sum(1 for ms in s.melds for m in ms if m.type in KAN_TYPES)
        nuki = sum(s.nukidora)
        consumed = (sum(len(s.visible_river(a)) for a in range(4))
                    + kans + nuki
                    + (1 if s.drawn_tile is not None else 0))
        total = s.left_tile_count + consumed
        report["ticks"] += 1
        report["wall_hist"][total] = report["wall_hist"].get(total, 0) + 1
        if total not in (SANMA_WALL - 1, SANMA_WALL):
            fail("V1", f"seq {seq}: wall identity = {total}, want {SANMA_WALL}±1 "
                       f"(L={s.left_tile_count} river+kan+nuki+drawn={consumed})")
        # NEGATIVE CONTROL for the -1-per-nuki claim. Only meaningful at >=2
        # nukis: the ±1 tolerance would absorb a single one, so a 1-nuki frame
        # cannot distinguish "nuki costs a tile" from "nuki is free" and asserting
        # on it would be theatre. At >=2 the two hypotheses are >=2 apart and the
        # band can no longer hide the difference.
        if nuki >= 2:
            report["nuki_control"] += 1
            if (total - nuki) in (SANMA_WALL - 1, SANMA_WALL):
                fail("V1", f"seq {seq}: with {nuki} nukis the identity ALSO holds "
                           f"without the nuki term — -1-per-nuki is not pinned")

    # --- V2/V3: nukidora semantics -------------------------------------------
    n_dora = sum(1 for e in events if e.get("type") == "dora")
    n_kan = sum(1 for e in events if e.get("type") in KAN_TYPES)
    n_nuki = sum(1 for e in events if e.get("type") == "nukidora")
    report["nukidora"] += n_nuki
    if n_dora != n_kan:
        fail("V2", f"{n_dora} dora events but {n_kan} kans — a nukidora flipped a dora "
                   f"(or a kan did not)")
    for i, ev in enumerate(events):
        if ev.get("type") != "nukidora":
            continue
        actor = ev.get("actor")
        if ev.get("pai") != "N":
            fail("V3", f"nukidora #{i} pai={ev.get('pai')!r}, want 'N'")
        prev = events[i - 1] if i else None
        nxt = events[i + 1] if i + 1 < len(events) else None
        if not (prev and prev.get("type") == "tsumo" and prev.get("actor") == actor):
            fail("V3", f"nukidora #{i} (actor {actor}) not preceded by that actor's tsumo "
                       f"(got {prev and prev.get('type')})")
        if not (nxt and nxt.get("type") == "tsumo" and nxt.get("actor") == actor):
            fail("V3", f"nukidora #{i} (actor {actor}) not followed by a replacement tsumo "
                       f"(got {nxt and nxt.get('type')}) — it would END the turn")

    # --- V10: under an ACCEPTED riichi, only the JUST-DRAWN north may be pulled
    # (riichi freezes the hand). Opponent draws are redacted to "?", so only the
    # pulls with an observable preceding draw can testify; among those the rule
    # held 13/13 with zero counterexamples. decision.py gates on this.
    reached = [False] * 4
    for i, ev in enumerate(events):
        t = ev.get("type")
        if t == "start_kyoku":
            reached = [False] * 4
        elif t == "reach_accepted":
            reached[ev["actor"]] = True
        elif t == "nukidora" and reached[ev["actor"]]:
            actor = ev["actor"]
            prev = events[i - 1] if i else {}
            drew = (prev.get("pai")
                    if prev.get("type") == "tsumo" and prev.get("actor") == actor
                    else None)
            if drew is not None and drew != "?" and drew != "N":
                fail("V10", f"event #{i}: actor {actor} pulled a north while in riichi "
                            f"after drawing {drew!r} — riichi should freeze the hand")
            if drew is None or drew == "?":
                report["v10_unobservable"] += 1
            else:
                report["v10_observed"] += 1

    # --- V4: no chi, no 2m-8m ------------------------------------------------
    for i, ev in enumerate(events):
        if ev.get("type") == "chi":
            fail("V4", f"event #{i} is a chi — sanma has no chi")
        for t in _tiles_in(ev):
            if ILLEGAL_MANZU.match(t):
                fail("V4", f"event #{i} ({ev.get('type')}) carries {t!r} — "
                           f"sanma has no 2m-8m")

    # --- V5/V6: round meta + start_kyoku shape -------------------------------
    for ev in events:
        if ev.get("type") != "start_kyoku":
            continue
        report["kyoku"] += 1
        kyoku, oya, bakaze = ev["kyoku"], ev["oya"], ev["bakaze"]
        if kyoku not in (1, 2, 3):
            fail("V5", f"kyoku {kyoku} not in 1..3")
        if oya != (kyoku - 1) % 3:
            fail("V5", f"oya {oya} != (kyoku-1)%3 = {(kyoku - 1) % 3} (kyoku {kyoku})")
        if bakaze not in ("E", "S"):
            fail("V5", f"bakaze {bakaze!r} not in E/S")
        scores, tehais = ev["scores"], ev["tehais"]
        if len(scores) != 4 or scores[3] != 0:
            fail("V6", f"start_kyoku scores {scores} — want 4-wide with a 0 phantom")
        if len(tehais) != 4:
            fail("V6", f"start_kyoku has {len(tehais)} tehais — want 4 (phantom included)")

    # --- V7: the phantom chair never acts, never holds -----------------------
    for i, ev in enumerate(events):
        if ev.get("actor") == 3:
            fail("V7", f"event #{i} ({ev.get('type')}) has actor 3 — the phantom acted")
    for seq, s in ticks:
        # in_round only: before the first start_kyoku the state is a default
        # BoardState whose scores are the 4P [25000]*4 placeholder.
        if not s.in_round:
            continue
        if s.rivers[3] or s.melds[3] or s.nukidora[3] or s.scores[3]:
            fail("V7", f"seq {seq}: phantom chair 3 is not inert "
                       f"(river {len(s.rivers[3])} melds {len(s.melds[3])} "
                       f"nuki {s.nukidora[3]} score {s.scores[3]})")

    # --- V8: the 4-copy budget includes the nukidora piles -------------------
    for seq, s in ticks:
        if not s.in_round:
            continue
        counts = {}
        def bump(pai):
            if pai and pai != "?":
                k = red_to_normal(pai)
                counts[k] = counts.get(k, 0) + 1
        for pai in s.hero_hand:
            bump(pai)
        for seat in range(4):
            for t in s.visible_river(seat):
                bump(t.pai)
            for m in s.melds[seat]:
                for pai in m.tiles:
                    bump(pai)
            for _ in range(s.nukidora[seat]):    # <- the piles ARE visible N tiles
                bump("N")
        for d in s.dora_markers:
            bump(d)
        for kind, n in counts.items():
            if n > 4:
                fail("V8", f"seq {seq}: tile {kind} seen {n}>4 with the nuki piles counted")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", nargs="+",
                    default=["captures/raw/ai_session_3p"])
    ap.add_argument("--max-report", type=int, default=20)
    args = ap.parse_args()

    caps = find_captures(args.captures)
    if not caps:
        # LOUD skip, never a silent pass: the corpus is gitignored, and a quiet
        # "0 violations" on an empty corpus is exactly how a broken constant
        # would sail through CI.
        sys.exit(f"SKIP (LOUD): no captures under {args.captures} — the 3P GT corpus "
                 f"is gitignored. This gate proves nothing without it.")

    report = {"games": 0, "skipped_4p": 0, "ticks": 0, "kyoku": 0,
              "nukidora": 0, "nuki_control": 0, "wall_hist": {},
              "v10_observed": 0, "v10_unobservable": 0,
              "by_rule": {}, "violations": []}
    for cap in caps:
        check_game(cap, report)

    print(f"[sanma-rules] {report['games']} sanma games "
          f"({report['skipped_4p']} 4P skipped), {report['ticks']} in-round ticks, "
          f"{report['kyoku']} kyoku, {report['nukidora']} nukidora")
    print(f"  wall identity histogram: "
          f"{dict(sorted(report['wall_hist'].items()))}  (want only {SANMA_WALL-1}/{SANMA_WALL})")
    print(f"  -1-per-nuki negative control exercised on {report['nuki_control']} "
          f"ticks with >=2 nukis")
    print(f"  V10 riichi pulls: {report['v10_observed']} observable, "
          f"{report['v10_unobservable']} redacted (cannot testify)")
    if report["violations"]:
        print(f"  {len(report['violations'])} VIOLATIONS {report['by_rule']}:")
        for v in report["violations"][:args.max_report]:
            print("   ", v)
        if len(report["violations"]) > args.max_report:
            print(f"    ... and {len(report['violations']) - args.max_report} more")
        sys.exit(1)
    if not report["games"]:
        sys.exit("SKIP (LOUD): no SANMA games found in the given captures")
    print("  V1..V8 all hold.")


if __name__ == "__main__":
    main()
