"""Scan a built YOLO dataset with the consistency gate and print distributions to
pick TAU / MAX_BAD. Read-only (never deletes). Run from repo root.

  PYTHONPATH=. $PY scripts/inspect/calibrate_occlusion_gate.py --datasets datasets/precise_ai_run_1
"""
from __future__ import annotations
import argparse, glob, os
from collections import Counter

import cv2
import numpy as np

from majsoul_eye.recognize.classifier import TileClassifier
from majsoul_eye.annotate.consistency import score_frame, frame_decision, TAU, MAX_BAD
from majsoul_eye.tiles import TILE_NAMES


def iter_label_boxes(img_path: str, label_path: str):
    img = cv2.imread(img_path)
    if img is None or not os.path.exists(label_path):
        return [], []
    h, w = img.shape[:2]
    gts, crops = [], []
    for line in open(label_path, encoding="utf-8"):
        f = line.split()
        if not f:
            continue
        cls = int(f[0]); cx, cy, bw, bh = (float(x) for x in f[1:5])
        x0 = max(0, int((cx - bw / 2) * w)); y0 = max(0, int((cy - bh / 2) * h))
        x1 = int((cx + bw / 2) * w); y1 = int((cy + bh / 2) * h)
        crop = img[y0:y1, x0:x1]
        if crop.size:
            gts.append(TILE_NAMES[cls]); crops.append(crop)
    return crops, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=sorted(glob.glob("datasets/precise_*")))
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--max-bad", type=int, default=MAX_BAD)
    args = ap.parse_args()

    clf = TileClassifier()
    buckets = Counter(); badhist = Counter(); pass_conf = []; fail_conf = []
    for ds in args.datasets:
        for img_path in sorted(glob.glob(os.path.join(ds, "yolo", "images", "*.png"))):
            seq = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(ds, "yolo", "labels", f"{seq}.txt")
            crops, gts = iter_label_boxes(img_path, label_path)
            if not crops:
                continue
            vs = score_frame(crops, gts, clf, tau=args.tau)
            for v in vs:
                (pass_conf if v.ok else fail_conf).append(v.conf)
            nbad = sum(1 for v in vs if not v.ok)
            badhist[min(nbad, 5)] += 1
            buckets[frame_decision(vs, max_bad=args.max_bad)[0]] += 1
    tot = sum(buckets.values())
    print(f"tau={args.tau} max_bad={args.max_bad}  frames={tot}")
    print("decision buckets:", dict(buckets))
    print("bad-boxes-per-frame histogram (5=5+):", dict(sorted(badhist.items())))
    def pct(a, q): return round(float(np.percentile(a, q)), 3) if a else None
    print(f"pass P(gt): median={pct(pass_conf,50)} p10={pct(pass_conf,10)}  (n={len(pass_conf)})")
    print(f"fail P(gt): median={pct(fail_conf,50)} p90={pct(fail_conf,90)}  (n={len(fail_conf)})")


if __name__ == "__main__":
    main()
