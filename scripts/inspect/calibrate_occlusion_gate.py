"""Scan a built dataset with the consistency gate and print distributions to pick
TAU / MAX_BAD. Read-only (never deletes). Run from repo root.

Classifies build_dataset's saved crops/<tile>/<seq>_<ci>.png (the perspective-warped
crop_box outputs the classifier trained on) — NOT a raw re-crop of the YOLO AABB, which
is skewed for tilted river/meld tiles and yields false positives. Sideways tiles have no
saved crop and are (correctly) not measured.

  PYTHONPATH=. $PY scripts/inspect/calibrate_occlusion_gate.py --datasets datasets/precise_ai_run_1
"""
from __future__ import annotations
import argparse, glob, os, re
from collections import Counter

import cv2
import numpy as np

from majsoul_eye.recognize.classifier import TileClassifier
from majsoul_eye.annotate.consistency import score_frame, frame_decision, TAU, MAX_BAD

_CI_RE = re.compile(r"_(\d+)\.png$")


def frame_saved_crops(ds: str, seq: str):
    """The frame's PROPER saved crops -> (crops, gts), ordered by crop index ci."""
    items = []
    for cdir in glob.glob(os.path.join(ds, "crops", "*")):
        gt = os.path.basename(cdir)
        for p in glob.glob(os.path.join(cdir, f"{seq}_*.png")):
            m = _CI_RE.search(p)
            if not m:
                continue
            img = cv2.imread(p)
            if img is not None and img.size:
                items.append((int(m.group(1)), gt, img))
    items.sort(key=lambda t: t[0])
    return [im for _, _, im in items], [gt for _, gt, _ in items]


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
            crops, gts = frame_saved_crops(ds, seq)
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
