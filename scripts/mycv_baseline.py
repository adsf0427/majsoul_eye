"""Measure mycv's REAL tile-face recognition accuracy on our captured frames.

Runs mycv's actual pipeline (via majsoul_eye.baselines.mycv_engine) over every
in-round captured frame, scoring its output against Akagi GT (via Replayer) with
position-agnostic bag matching (baselines.score). Reports per-zone accuracy:

    hand   : self hand        (getHandTiles + tile.model)
    river  : 3 OPPONENT rivers (seat masks m1/m2/m3 + myweight ResNet)
             [the hero's own river is masked out of mycv's vision by design]
    meld   : opponent melds    (getType type 1/2/3 + myweight ResNet, de-duped)

Usage:
    PYTHONPATH=. $PY scripts/mycv_baseline.py \
        --capture captures/session6.jsonl --frames captures/session6/frames \
        --overlay-dir <scratch>/mycv_gate --limit 0
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np

from majsoul_eye.capture.schema import read_records
from majsoul_eye.state.replay import replay_capture, check_invariants
from majsoul_eye.baselines.mycv_engine import MycvEngine, OPPONENT_MASKS
from majsoul_eye.baselines.score import bag_tally, ZoneTally


def frame_path(frames_dir: str, seq: int) -> str:
    return os.path.join(frames_dir, f"{seq:06d}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--mycv-dir", default=None)
    ap.add_argument("--limit", type=int, default=0, help="max frames (0 = all)")
    ap.add_argument("--overlay-dir", default=None, help="save first N overlays here")
    ap.add_argument("--overlay-n", type=int, default=4)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--drop-violations", action="store_true", default=True)
    args = ap.parse_args()

    import cv2

    eng = MycvEngine(args.mycv_dir) if args.mycv_dir else MycvEngine()

    zones = {"hand": ZoneTally(), "river": ZoneTally(), "meld": ZoneTally()}
    river_by_pos = {"right": ZoneTally(), "across": ZoneTally(), "left": ZoneTally()}
    per_frame = []
    n_frames = n_skipped = n_overlay = 0

    if args.overlay_dir:
        os.makedirs(args.overlay_dir, exist_ok=True)

    for rec, st in replay_capture(read_records(args.capture)):
        seq = rec.seq
        p = frame_path(args.frames, seq)
        if st.hero_seat < 0 or not st.in_round or not os.path.exists(p):
            continue
        if args.drop_violations and check_invariants(st):
            n_skipped += 1
            continue
        frame = cv2.imread(p)
        if frame is None:
            continue
        res = eng.recognize(frame, st.hero_seat)

        # --- hand ---
        gt_hand = [t for t in st.hero_hand if t != "?"]
        ht = bag_tally(res.hand, gt_hand)
        zones["hand"].add(ht)

        # --- rivers (3 opponents) ---
        rt_frame = ZoneTally()
        pos_for_k = {1: "right", 2: "across", 3: "left"}
        for k in OPPONENT_MASKS:
            seat = (st.hero_seat + k) % 4
            pred = [d.name for d in res.rivers.get(seat, [])]
            gt = [t.pai for t in st.visible_river(seat)]
            t = bag_tally(pred, gt)
            zones["river"].add(t); rt_frame.add(t)
            river_by_pos[pos_for_k[k]].add(t)

        # --- melds (opponent seats, global bag) ---
        gt_meld = []
        for k in OPPONENT_MASKS:
            seat = (st.hero_seat + k) % 4
            for m in st.melds[seat]:
                gt_meld.extend(m.tiles)
        mt = bag_tally([d.name for d in res.melds], gt_meld)
        zones["meld"].add(mt)

        per_frame.append({
            "seq": seq, "hero": st.hero_seat,
            "hand": [ht.correct, ht.n_gt, ht.n_pred],
            "river": [rt_frame.correct, rt_frame.n_gt, rt_frame.n_pred],
            "meld": [mt.correct, mt.n_gt, mt.n_pred],
            "err": res.error,
        })
        n_frames += 1

        # --- overlay gate ---
        if args.overlay_dir and n_overlay < args.overlay_n:
            _save_overlay(cv2, frame, res, st, os.path.join(args.overlay_dir, f"gate_{seq:06d}.png"))
            n_overlay += 1

        if args.limit and n_frames >= args.limit:
            break

    # --- report ---
    print(f"\n=== mycv baseline on {args.capture} ===")
    print(f"frames scored: {n_frames}  (dropped {n_skipped} invariant violations)")
    print(f"\n{'zone':8s} {'n_gt':>6s} {'n_pred':>6s} {'correct':>7s} "
          f"{'recall':>8s} {'precision':>10s} {'len_rec':>8s}")
    print("  (recall = end-to-end = correct/n_gt ; precision = correct/n_pred)")
    for name, z in zones.items():
        print(f"{name:8s} {z.n_gt:6d} {z.n_pred:6d} {z.correct:7d} "
              f"{z.recall:8.3f} {z.precision:10.3f} {z.recall_lenient:8.3f}")
    print("\nriver by screen position:")
    for pos, z in river_by_pos.items():
        print(f"  {pos:7s} {z.summary()}")

    print("\ntop river miss/extra (class-level diagnostic):")
    for kind_name, c in zones["river"].confusions.most_common(12):
        print(f"  {kind_name[0]:5s} {kind_name[1]:5s}  x{c}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "capture": args.capture, "n_frames": n_frames, "n_skipped": n_skipped,
                "zones": {k: {"n_gt": v.n_gt, "n_pred": v.n_pred, "correct": v.correct,
                              "recall": v.recall, "precision": v.precision,
                              "recall_lenient": v.recall_lenient}
                          for k, v in zones.items()},
                "river_by_pos": {k: {"n_gt": v.n_gt, "n_pred": v.n_pred, "correct": v.correct,
                                     "recall": v.recall, "precision": v.precision}
                                 for k, v in river_by_pos.items()},
                "per_frame": per_frame,
            }, f, indent=2, default=str)
        print(f"\nwrote {args.out_json}")


def _save_overlay(cv2, frame, res, st, path):
    f = cv2.resize(frame, (1920, 1080))
    COL = {"river": (0, 255, 0), "meld": (0, 165, 255)}
    for seat, dets in res.rivers.items():
        for d in dets:
            cv2.rectangle(f, (d.x - d.w // 2, d.y - d.h // 2),
                          (d.x + d.w // 2, d.y + d.h // 2), COL["river"], 2)
            cv2.putText(f, d.name, (d.x - d.w // 2, d.y - d.h // 2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL["river"], 1)
    for d in res.melds:
        cv2.rectangle(f, (d.x - d.w // 2, d.y - d.h // 2),
                      (d.x + d.w // 2, d.y + d.h // 2), COL["meld"], 2)
        cv2.putText(f, d.name, (d.x - d.w // 2, d.y - d.h // 2 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL["meld"], 1)
    # GT caption
    txt = "hand_pred=" + "".join(res.hand)
    cv2.putText(f, txt[:120], (20, 1060), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(path, f)


if __name__ == "__main__":
    main()
