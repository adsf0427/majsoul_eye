"""QA guard: per-seat systematic meld-snap offset + mislock rate over captures.

Run after ANY change to MELD_STRIP2 / the fullwarp / the tile masks. A large
per-seat offset means a corner is mis-calibrated; a high mislock rate means the
snap sits at the aliasing midpoint and flips one tile (STATUS §1.50). Exits
nonzero if the worst seat mislock exceeds --max-mislock.

  PYTHONPATH=. python scripts/annotate/meld_snap_qa.py
  PYTHONPATH=. python scripts/annotate/meld_snap_qa.py --sources captures/raw/ai_session captures/raw/ai_session3
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

from majsoul_eye.annotate import build_homographies
from majsoul_eye.annotate import pipeline as P
from majsoul_eye.annotate.seatgt import seat_gt
from majsoul_eye.capture.gtframes import build_seq_state, load_frames
from majsoul_eye.state.replay import check_invariants, is_call_window, is_deal_window


def dominant(vals, tol=12.0):
    """(center, frac_in_densest_cluster) for a 1-D list within +-tol. ([] -> 0,0)."""
    if not vals:
        return 0.0, 0.0
    v = np.array(sorted(vals))
    best = (float(v[0]), 0)
    for x in v:
        inb = v[(v >= x - tol) & (v <= x + tol)]
        if len(inb) > best[1]:
            best = (float(np.median(inb)), len(inb))
    return round(best[0], 1), round(best[1] / len(v), 3)


def measure(sources):
    hom = build_homographies(1920, 1080)
    Hinv = hom["H_full_inv"]
    caps = []
    for root in sources:
        caps += sorted(glob.glob(os.path.join(root, "run_*", "game*", "game*.jsonl")))
    da = defaultdict(list)
    dc = defaultdict(list)
    for cap in caps:
        try:
            ss = build_seq_state(cap)
            fr = load_frames(os.path.dirname(cap))
        except Exception:
            continue
        for s in sorted(ss):
            if s not in fr:
                continue
            st = ss[s]
            if is_deal_window(st) or is_call_window(st) or check_invariants(st):
                continue
            if not any(seat_gt(st, p)[2] for p in range(4)):
                continue
            img = cv2.imread(fr[s])
            if img is None:
                continue
            if img.shape[1] != 1920:
                img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
            full = P.warp_to_full(img, hom["H_full"], hom["full_size"])
            hsv = cv2.cvtColor(full, cv2.COLOR_BGR2HSV)
            mw = P.tile_face_mask(hsv=hsv)
            mb = P.tile_back_mask(hsv=hsv)
            for pos in range(4):
                _, _, melds, _ = seat_gt(st, pos)
                if not melds:
                    continue
                boxes = P.generate_meld_boxes_v2(pos, melds, Hinv)
                if not boxes:
                    continue
                a, c, diag = P.snap_meld_strip(mw, mb, boxes, pos)
                if diag["n_along"] + diag["n_cross"] >= 4 and diag["score"] >= 3.0:
                    da[pos].append(a)
                    dc[pos].append(c)
    return da, dc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["captures/raw/ai_session"])
    ap.add_argument("--max-mislock", type=float, default=0.03)
    args = ap.parse_args()
    da, dc = measure(args.sources)
    worst = 0.0
    print(f"{'pos':>3} {'N':>5} | {'da_off':>7} {'da_mis':>7} | {'dc_off':>7} {'dc_mis':>7}")
    for pos in range(4):
        if not da[pos]:
            print(f"{pos:>3}     0 | (no confident meld frames)")
            continue
        dao, daf = dominant(da[pos])
        dco, dcf = dominant(dc[pos])
        damis, dcmis = round(1 - daf, 3), round(1 - dcf, 3)
        worst = max(worst, damis, dcmis)
        print(f"{pos:>3} {len(da[pos]):>5} | {dao:>7} {damis:>7} | {dco:>7} {dcmis:>7}")
    print(f"worst mislock={worst:.3f} (threshold {args.max_mislock})")
    if worst > args.max_mislock:
        print("FAIL: a seat exceeds the mislock threshold — check MELD_STRIP2 corners.")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
