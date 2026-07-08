"""Phase-2 guard: after consensus, every frame of a (kyoku,pos) must share ONE offset.

Runs game_meld_overrides per game and asserts each (bakaze,kyoku,honba,pos) round maps
to a single override value (or None). A non-uniform round means consensus is not being
applied per-round (regression). Exits nonzero on any violation.

  PYTHONPATH=. python scripts/annotate/meld_consensus_qa.py
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

from majsoul_eye.annotate import build_homographies
from majsoul_eye.annotate import meldsnap as M
from majsoul_eye.capture.gtframes import build_seq_state, load_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["captures/raw/ai_session"])
    args = ap.parse_args()
    hom = build_homographies(1920, 1080)
    caps = []
    for root in args.sources:
        caps += sorted(glob.glob(os.path.join(root, "run_*", "game*", "game*.jsonl")))
    bad = 0
    n_rounds = 0
    for cap in caps:
        try:
            ss = build_seq_state(cap)
            fr = load_frames(os.path.dirname(cap))
        except Exception as e:
            print(f"  skip {cap}: {e}")
            continue
        ov = M.game_meld_overrides(ss, fr, hom)
        by_round = defaultdict(set)
        for seq, per_pos in ov.items():
            st = ss[seq]
            for pos, val in per_pos.items():
                by_round[(st.bakaze, st.kyoku, st.honba, pos)].add(val)
        for key, vals in by_round.items():
            n_rounds += 1
            if len(vals) != 1:
                bad += 1
                print(f"NONUNIFORM {os.path.basename(os.path.dirname(cap))} {key}: {vals}")
    print(f"rounds checked={n_rounds} nonuniform={bad}")
    if bad:
        print("FAIL: consensus is not uniform per round.")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
