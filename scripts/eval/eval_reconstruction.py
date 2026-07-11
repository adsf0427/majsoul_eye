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
from majsoul_eye.state.history import (ReconstructionOverrides,
                                       UserTsumogiriOverride)
from majsoul_eye.state.observe import check_observed, observed_from_board
from majsoul_eye.state.reconstruct import _hero_call_pending, reconstruct
from majsoul_eye.state.replay import (Replayer, is_call_pending, is_deal_window,
                                      is_score_anim_window)


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
        "melds": [[(m.type, m.from_rel, tuple(sorted(m.tiles)),
                    m.called_pai, m.added_pai) for m in ms]
                  for ms in o.melds],
        "hand": sorted(o.hero_hand), "drawn": o.drawn_tile,
        "dora": list(o.dora_markers), "reach": list(o.reach),
    }


def diff_zones(a, b):
    ka, kb = obs_key(a), obs_key(b)
    return [z for z in ka if ka[z] != kb[z]]


def history_overrides_from_board(state, observed):
    hero = state.hero_seat
    result = ReconstructionOverrides()
    for rel_seat in range(4):
        absolute = (hero + rel_seat) % 4
        visible = [tile for tile in state.rivers[absolute] if not tile.called]
        for index, tile in enumerate(visible):
            item_id = f"gt-river:{rel_seat}:{index}"
            result.river_ids[(rel_seat, index)] = item_id
            result.user_visible[(rel_seat, index)] = UserTsumogiriOverride(
                tile.tsumogiri, item_id, f"gt.river.{rel_seat}.{index}")
    called_by_key = {}
    for absolute_target in range(4):
        for tile in state.rivers[absolute_target]:
            if not tile.called:
                continue
            if tile.called_by is None or tile.called_meld_index is None:
                raise ValueError("GT called discard lacks replay association")
            caller = (tile.called_by - hero) % 4
            called_by_key[(caller, tile.called_meld_index)] = tile
    for caller in range(4):
        for meld_index, meld in enumerate(observed.melds[caller]):
            if meld.type not in ("chi", "pon", "daiminkan", "kakan"):
                continue
            key, item_id = (caller, meld_index), f"gt-ghost:{caller}:{meld_index}"
            called = called_by_key[key]
            if called.pai != meld.called_pai:
                raise ValueError(f"GT ghost tile mismatch for {key}")
            result.ghost_ids[key] = item_id
            result.ghost_order.append(key)
            result.user_ghosts[key] = UserTsumogiriOverride(
                called.tsumogiri, item_id,
                f"gt.ghost.{caller}.{meld_index}")
    return result


def _history_marks_mismatch(gt_overrides, recon_state, recon_obs) -> bool:
    """Do the reconstruction's emitted visible/ghost tsumogiri marks reproduce
    the GT constraint that history_overrides_from_board forced? Compared in
    REL-seat space by IDENTITY — visible marks by (relSeat, visibleIndex), ghost
    marks by (callerRelSeat, meldIndex) — never by full-river position. The
    solver may legally place a called-away discard later than a still-visible one
    (calls as late as feasible), which reorders that ghost tile inside the raw
    river list without changing any tsumogiri value; a positional compare would
    flag those as false mismatches. With --history-from-gt this must be zero: the
    solver was handed every GT tsumogiri as a hard override."""
    re_ov = history_overrides_from_board(recon_state, recon_obs)
    if (set(gt_overrides.user_visible) != set(re_ov.user_visible)
            or set(gt_overrides.user_ghosts) != set(re_ov.user_ghosts)):
        return True
    return (any(gt_overrides.user_visible[k].value != re_ov.user_visible[k].value
                for k in gt_overrides.user_visible)
            or any(gt_overrides.user_ghosts[k].value != re_ov.user_ghosts[k].value
                   for k in gt_overrides.user_ghosts))


_REJECT_CATS = (            # ordered: first substring hit wins per message
    ("scores sum", "hud_scores"),  # "scores sum" first: its message text also contains "kyotaku"
    ("kyotaku", "hud_kyotaku"),
    ("wall count", "hud_wall"),
    ("stray detection", "stray"),
    ("meld strip", "meld_parse"),          # unparsable + ambiguous
    ("river", "river_geometry"),           # off-grid / hole / prefix / >6
    (">4 times", "tile_gt4"),
    ("hero hand", "hand_size"),
    ("dora marker", "dora"),
    ("concealed", "concealed"),
)


def reject_categories(violations):
    """Violation messages of ONE rejected frame -> distinct category set, so
    threshold calibration can see WHY frames are rejected, not just how many
    (post-M1 final-review item). Vocabulary tracks assemble.py/observe.py."""
    cats = set()
    for msg in violations:
        cats.add(next((cat for sub, cat in _REJECT_CATS if sub in msg), "other"))
    return cats


def run_oracle(states, report, history_from_gt=False):
    for seq, st in states.items():
        if not st.in_round or is_deal_window(st):
            continue
        # score-anim windows: HUD numbers roll on screen right after a reach —
        # project them away so the new HUD cross-checks don't reject GT frames.
        obs = observed_from_board(st, include_hud=not is_score_anim_window(st))
        if check_observed(obs):
            report["skipped_violations"] += 1
            continue
        if is_call_pending(st) and not (st.awaiting_discard == st.hero_seat
                                        and _hero_call_pending(obs)):
            # KNOWN real-capture edge case: frame taken between a call and the
            # caller's mandatory immediate discard (see is_call_pending's
            # docstring in state/replay.py). An OPPONENT's gap is genuinely
            # un-reconstructable (their withheld discard is invisible) — a
            # counted skip. HERO's chi/pon gap is fully visible and now
            # reconstructs (sequence ends at the call), so it falls through;
            # hero KAN gaps (13-count shape, indistinguishable from steady)
            # stay skipped.
            report["skipped_call_pending"] += 1
            continue
        # --history-from-gt: hand the solver the exact GT tsumogiri for every
        # visible/ghost discard so the emitted history is pinned to truth, then
        # assert the round-trip reproduces those marks (history_mismatch == 0).
        overrides = history_overrides_from_board(st, obs) if history_from_gt else None
        r = reconstruct(obs, overrides)
        if not r.ok:
            report["fail"].append({"seq": seq, "reason": r.reason})
            continue
        rp = Replayer()
        for ev in r.events:
            rp.apply(ev)
        recon = observed_from_board(rp.state, include_hud=False)
        if history_from_gt and _history_marks_mismatch(overrides, rp.state, recon):
            report["history_mismatch"] += 1
        d = diff_zones(recon, observed_from_board(st, include_hud=False))
        if d:
            report["mismatch"].append({"seq": seq, "zones": d})
        else:
            report["ok"] += 1


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
    skeys = sorted(states)
    succ_of = dict(zip(skeys, skeys[1:]))
    for seq, st in states.items():
        if seq not in frames or not st.in_round or is_deal_window(st):
            continue
        img = cv2.imread(frames[seq])
        if img is None:
            continue
        obs = assemble(det.predict(img), locate_fullscreen(img),
                       frame_bgr=img, hud_reader=reader)
        gt = observed_from_board(st)
        # Hero-draw race: the quiet-capture can fire AFTER the next record's
        # draw animation rendered (same timing skew as the §1.53 wall_count
        # pixel=GT-1 noise), so the frame depicts the SUCCESSOR state, not
        # states[seq]. Accept the join correction only when it is exact:
        # GT has no drawn tile, the frame shows one, and the successor
        # projection matches the observation zone-for-zone.
        if (gt.drawn_tile is None and obs.drawn_tile is not None
                and seq in succ_of and states[succ_of[seq]].in_round):
            succ_gt = observed_from_board(states[succ_of[seq]])
            if succ_gt.drawn_tile == obs.drawn_tile \
                    and not diff_zones(obs, succ_gt):
                report["drawn_race"] += 1
                gt, st = succ_gt, states[succ_of[seq]]
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


def hero_id(events):
    return next((e.get("id", 0) for e in events
                 if e.get("type") == "start_game"), 0)


def ask_engine(cmd, events, timeout=60):
    # {seat} placeholder -> the sequence's own hero seat: truth events keep the
    # real seat (start_game id) while reconstructed events put hero at abs seat
    # 0 when no HUD was read — one fixed player-id cmd cannot serve both.
    cmd = cmd.replace("{seat}", str(hero_id(events)))
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
            report["engine_fail"] += 1
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
    ap.add_argument("--hud-weights", default=None)
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--history-from-gt", action="store_true",
                    help="oracle only: force GT tsumogiri overrides into "
                         "reconstruct and count history_mismatch (must be 0)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    caps = find_captures(args.captures)
    if not caps:
        sys.exit("no captures found")
    if args.level == "engine" and not args.engine_cmd:
        sys.exit("--level engine requires --engine-cmd (an mjai bot: events on "
                 "stdin as JSON lines, reactions on stdout)")

    total = {"ok": 0, "fail": [], "mismatch": [], "skipped_violations": 0,
             "skipped_call_pending": 0, "frames": 0, "rejected": 0,
             "rejected_reasons": {}, "zone_errors": {}, "agree": 0,
             "disagree": [], "engine_error": 0, "engine_fail": 0,
             "hud_ok": {}, "hud_err": {}, "hud_missing": {},
             "score_anim_rejected": 0, "drawn_race": 0, "history_mismatch": 0}
    for cap in caps:
        states = build_seq_state(cap)
        if args.level == "oracle":
            run_oracle(states, total, history_from_gt=args.history_from_gt)
        elif args.level == "assemble":
            run_assemble(cap, states, total, args.weights, args.device,
                         args.hud_weights, args.no_hud)
        else:
            run_engine(cap, states, total, args.engine_cmd, args.sample)

    if args.level == "oracle":
        n = total["ok"] + len(total["fail"]) + len(total["mismatch"])
        print(f"[oracle] {total['ok']}/{n} ok, {len(total['fail'])} infeasible, "
              f"{len(total['mismatch'])} mismatched, "
              f"{total['skipped_violations']} skipped, "
              f"{total['skipped_call_pending']} call-pending skipped, "
              f"history_mismatch {total['history_mismatch']}")
        for f in total["fail"][:10]:
            print("  FAIL", f)
        for m in total["mismatch"][:10]:
            print("  DIFF", m)
    elif args.level == "assemble":
        print(f"[assemble] {total['ok']}/{total['frames']} frames fully match, "
              f"{total['rejected']} rejected {total['rejected_reasons']}; "
              f"zone errors: {total['zone_errors']}")
        print(f"  hud ok {total['hud_ok']}\n  hud err {total['hud_err']}\n"
              f"  hud missing {total['hud_missing']}; "
              f"score-anim rejected {total['score_anim_rejected']}; "
              f"drawn-race rejoined {total['drawn_race']}")
    else:
        print(f"[engine] agree {total['agree']}, "
              f"disagree {len(total['disagree'])}, "
              f"recon-infeasible {total['engine_fail']}, "
              f"errors {total['engine_error']}")
        for d in total["disagree"][:10]:
            print("  ", d)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(total, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
