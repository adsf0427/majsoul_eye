"""Delete/trim occlusion-corrupted YOLO boxes & frames from built datasets.

For each datasets/precise_*/ : crop every GT box from yolo/images/<seq>.png, run the
production classifier, and apply the consistency smart-drop (annotate.consistency):
- keep       -> untouched
- drop_boxes -> rewrite the label without the bad lines; glob-delete a class's crops
  only when EVERY box of that class in the frame is bad (a partially-bad class's
  crops can't be safely mapped label-line -> crop-file, so they're left in place)
- drop_frame -> delete image + label + all crops for the seq
Then rewrite datasets/detector*/{train,val}.txt dropping now-missing image lines.

Dry-run by DEFAULT (prints planned actions); --apply to act. Idempotent.

  PYTHONPATH=. $PY scripts/data/purge_occlusion_frames.py            # dry-run
  PYTHONPATH=. $PY scripts/data/purge_occlusion_frames.py --apply
"""
from __future__ import annotations
import argparse, glob, os

import cv2

from majsoul_eye.recognize.classifier import TileClassifier
from majsoul_eye.annotate.consistency import score_frame, frame_decision, TAU, MAX_BAD
from majsoul_eye.tiles import TILE_NAMES


def _read_boxes(img_path, label_path):
    """-> (raw_lines, gts, crops) aligned by index (only rows with a nonempty crop)."""
    img = cv2.imread(img_path)
    raw, gts, crops = [], [], []
    if img is None or not os.path.exists(label_path):
        return raw, gts, crops, img
    h, w = img.shape[:2]
    for line in open(label_path, encoding="utf-8"):
        if not line.split():
            continue
        f = line.split()
        cls = int(f[0]); cx, cy, bw, bh = (float(x) for x in f[1:5])
        x0 = max(0, int((cx - bw / 2) * w)); y0 = max(0, int((cy - bh / 2) * h))
        x1 = int((cx + bw / 2) * w); y1 = int((cy + bh / 2) * h)
        crop = img[y0:y1, x0:x1]
        if crop.size:
            raw.append(line.rstrip("\n")); gts.append(TILE_NAMES[cls]); crops.append(crop)
    return raw, gts, crops, img


def plan_frame(img_path, label_path, clf, tau=TAU, max_bad=MAX_BAD):
    raw, gts, crops, _ = _read_boxes(img_path, label_path)
    if not crops:
        return "keep", []
    return frame_decision(score_frame(crops, gts, clf, tau=tau), max_bad=max_bad)


def _crops_for(ds, gt, seq):
    return glob.glob(os.path.join(ds, "crops", gt, f"{seq}_*.png"))


def _drop_boxes_plan(gts, bad):
    """Partition the bad boxes' classes into ones safe to glob-delete crops for (EVERY
    box of that class in the frame is bad) vs ones to leave alone (mixed good/bad — a
    class's crops are named `<seq>_NNN.png` by build_dataset's per-frame global crop
    counter, which does not line up with per-class label-line order, so there is no
    cheap way to know which specific crop file is the bad one). Returns (full, partial)
    sorted class-name lists."""
    full, partial = [], []
    for gt in sorted(set(gts[i] for i in bad)):
        if gts.count(gt) == [gts[i] for i in bad].count(gt):
            full.append(gt)
        else:
            partial.append(gt)
    return full, partial


def apply_drop_boxes(ds, seq, label_path, raw, gts, bad, apply=True):
    """Rewrite the label file dropping `bad` line indices, and glob-delete a class's
    crops only when EVERY box of that class in this frame is bad (see _drop_boxes_plan).
    With apply=False, computes the plan without touching disk. Returns
    (kept_lines, full_classes, partial_classes)."""
    kept = [ln for i, ln in enumerate(raw) if i not in set(bad)]
    full, partial = _drop_boxes_plan(gts, bad)
    if apply:
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(kept) + ("\n" if kept else ""))
        for gt in full:
            for c in _crops_for(ds, gt, seq):
                os.remove(c)
    return kept, full, partial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-dir", default="datasets")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--max-bad", type=int, default=MAX_BAD)
    args = ap.parse_args()

    clf = TileClassifier()
    tot_boxes = tot_frames = 0
    for ds in sorted(glob.glob(os.path.join(args.datasets_dir, "precise_*"))):
        for img_path in sorted(glob.glob(os.path.join(ds, "yolo", "images", "*.png"))):
            seq = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(ds, "yolo", "labels", f"{seq}.txt")
            raw, gts, crops, _ = _read_boxes(img_path, label_path)
            if not crops:
                continue
            decision, bad = frame_decision(score_frame(crops, gts, clf, tau=args.tau), max_bad=args.max_bad)
            if decision == "keep":
                continue
            if decision == "drop_frame":
                tot_frames += 1
                victims = [img_path, label_path]
                for gt in set(gts):
                    victims += _crops_for(ds, gt, seq)
                print(f"{os.path.basename(ds)}/{seq}: DROP FRAME ({len(bad)} bad of {len(gts)})")
                if args.apply:
                    for v in victims:
                        if os.path.exists(v):
                            os.remove(v)
            else:  # drop_boxes
                tot_boxes += len(bad)
                badtiles = [gts[i] for i in bad]
                kept, full, partial = apply_drop_boxes(ds, seq, label_path, raw, gts, bad, apply=args.apply)
                msg = f"{os.path.basename(ds)}/{seq}: drop {len(bad)} box(es) {badtiles} ({len(gts)}->{len(kept)})"
                if full:
                    msg += f"; crops deleted for {full}"
                if partial:
                    msg += f"; crops LEFT for mixed-class {partial} (can't map box->file safely)"
                print(msg)

    # Fix assembled detector splits: drop lines whose image no longer exists.
    for lst in glob.glob(os.path.join(args.datasets_dir, "detector*", "*.txt")):
        if os.path.basename(lst) not in ("train.txt", "val.txt"):
            continue
        lines = [ln.rstrip("\n") for ln in open(lst, encoding="utf-8") if ln.strip()]
        kept = [ln for ln in lines if os.path.exists(ln)]
        if len(kept) != len(lines):
            print(f"{lst}: drop {len(lines) - len(kept)} missing-image lines ({len(lines)} -> {len(kept)})")
            if args.apply:
                with open(lst, "w", encoding="utf-8") as f:
                    f.write("\n".join(kept) + ("\n" if kept else ""))

    mode = "DELETED" if args.apply else "would delete (dry-run; pass --apply)"
    print(f"\nTOTAL {mode}: boxes={tot_boxes}  frames={tot_frames}")


if __name__ == "__main__":
    main()
