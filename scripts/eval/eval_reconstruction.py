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
from majsoul_eye.state.replay import Replayer, is_call_pending, is_deal_window


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
        if is_call_pending(st):
            # KNOWN real-capture edge case: frame taken between a chi/pon/
            # (dai)minkan/ankan/kakan and the caller's mandatory immediate
            # discard (see is_call_pending's docstring in state/replay.py) —
            # genuinely un-reconstructable from this single frame, so it is a
            # counted skip rather than a reconstruction failure.
            report["skipped_call_pending"] += 1
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
             "skipped_call_pending": 0, "frames": 0, "rejected": 0,
             "zone_errors": {}, "agree": 0, "disagree": [], "engine_error": 0}
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
              f"{total['skipped_violations']} skipped, "
              f"{total['skipped_call_pending']} call-pending skipped")
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
