"""Delete/trim occlusion-corrupted YOLO boxes & frames from built datasets.

For each datasets/precise_*/ : verify each box with the production classifier and apply
the consistency smart-drop (annotate.consistency): keep / drop_boxes / drop_frame.

CROP SOURCE — the fix that matters: we classify the PROPER perspective-warped crops
build_dataset already saved under crops/<tile>/<seq>_<ci>.png (the `crop_box` outputs
the classifier trained on). Re-cropping the raw axis-aligned YOLO box (img[y0:y1,x0:x1])
is WRONG for tilted river/meld tiles — a skewed box full of background — and produces
mass false positives. Sideways tiles (riichi/called) have NO saved crop and are never
judged, matching the build-time gate.

Dropping:
- drop_frame -> delete image + label + all crops for the seq.
- drop_boxes, no sideways in frame (#crops == #yolo lines, classes aligned in order) ->
  surgically drop the bad YOLO lines and delete exactly those bad crop files.
- drop_boxes, sideways present -> crop<->line order can't be recovered from a built
  dataset, so conservatively delete the whole DETECTOR frame (image+label) while keeping
  the frame's good classifier crops (only the bad crops are deleted).
Then rewrite datasets/detector*/{train,val}.txt dropping now-missing image lines.

Dry-run by DEFAULT (prints planned actions); --apply to act. Idempotent.

  PYTHONPATH=. $PY scripts/data/purge_occlusion_frames.py            # dry-run
  PYTHONPATH=. $PY scripts/data/purge_occlusion_frames.py --apply
"""
from __future__ import annotations
import argparse, glob, os, re

import cv2

from majsoul_eye.recognize.classifier import TileClassifier
from majsoul_eye.annotate.consistency import score_frame, frame_decision, TAU, MAX_BAD
from majsoul_eye.tiles import TILE_NAMES

_CI_RE = re.compile(r"_(\d+)\.png$")


def saved_crops(ds, seq):
    """build_dataset's PROPER (crop_box-warped) crops for a frame, ordered by crop index
    ``ci``. Returns [(ci, gt_class, path, img)]. Only non-sideways boxes have crops."""
    out = []
    for cdir in glob.glob(os.path.join(ds, "crops", "*")):
        gt = os.path.basename(cdir)
        for p in glob.glob(os.path.join(cdir, f"{seq}_*.png")):
            m = _CI_RE.search(p)
            if not m:
                continue
            img = cv2.imread(p)
            if img is not None and img.size:
                out.append((int(m.group(1)), gt, p, img))
    out.sort(key=lambda t: t[0])
    return out


def yolo_lines(label_path):
    if not os.path.exists(label_path):
        return []
    return [ln.rstrip("\n") for ln in open(label_path, encoding="utf-8") if ln.split()]


def _yolo_classes(lines):
    return [TILE_NAMES[int(ln.split()[0])] for ln in lines]


def plan_frame(ds, seq, clf, tau=TAU, max_bad=MAX_BAD):
    """Verdict on the frame's PROPER saved crops. Returns
    (decision, bad_local_idx, crops_meta, lines) where bad_local_idx indexes crops_meta
    (which is ordered by ci). ("keep", [], ...) when there is nothing to judge."""
    crops_meta = saved_crops(ds, seq)
    lines = yolo_lines(os.path.join(ds, "yolo", "labels", f"{seq}.txt"))
    if not crops_meta:
        return "keep", [], crops_meta, lines
    gts = [m[1] for m in crops_meta]
    imgs = [m[3] for m in crops_meta]
    decision, bad = frame_decision(score_frame(imgs, gts, clf, tau=tau), max_bad=max_bad)
    return decision, bad, crops_meta, lines


def _write_label(label_path, kept):
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + ("\n" if kept else ""))


def apply_plan(ds, seq, decision, bad, crops_meta, lines, apply=True):
    """Execute (or, with apply=False, describe) the plan for one frame.
    Returns (mode, n_bad) where mode in {"frame", "boxes", "frame(sideways)"}."""
    img_path = os.path.join(ds, "yolo", "images", f"{seq}.png")
    label_path = os.path.join(ds, "yolo", "labels", f"{seq}.txt")

    if decision == "drop_frame":
        victims = [img_path, label_path] + [m[2] for m in crops_meta]
        if apply:
            for v in victims:
                if os.path.exists(v):
                    os.remove(v)
        return "frame", len(bad)

    # drop_boxes: always delete exactly the bad crop files (we know each bad crop's path).
    bad_paths = [crops_meta[i][2] for i in bad]
    if apply:
        for p in bad_paths:
            if os.path.exists(p):
                os.remove(p)

    # Detector (YOLO) side. Only surgically drop lines when crop<->line order is
    # recoverable: that requires no sideways box in the frame, i.e. one crop per line
    # with matching class order (verified empirically to align exactly).
    clean_map = (len(crops_meta) == len(lines)
                 and [m[1] for m in crops_meta] == _yolo_classes(lines))
    if clean_map:
        bad_lines = set(bad)  # crop local index == line index in the clean-map case
        kept = [ln for i, ln in enumerate(lines) if i not in bad_lines]
        if apply:
            _write_label(label_path, kept)
        return "boxes", len(bad)

    # Sideways present -> can't map crop->line safely. Drop the whole DETECTOR frame
    # (image+label); the frame's good classifier crops are kept (only bad ones deleted).
    if apply:
        for v in (img_path, label_path):
            if os.path.exists(v):
                os.remove(v)
    return "frame(sideways)", len(bad)


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
            decision, bad, crops_meta, lines = plan_frame(ds, seq, clf, args.tau, args.max_bad)
            if decision == "keep":
                continue
            badtiles = [crops_meta[i][1] for i in bad]
            mode, n = apply_plan(ds, seq, decision, bad, crops_meta, lines, apply=args.apply)
            if mode.startswith("frame"):
                tot_frames += 1
                extra = " (sideways in frame; can't map box->line)" if mode == "frame(sideways)" else ""
                print(f"{os.path.basename(ds)}/{seq}: DROP FRAME{extra} "
                      f"({n} bad of {len(crops_meta)} judged {badtiles})")
            else:
                tot_boxes += n
                print(f"{os.path.basename(ds)}/{seq}: drop {n} box(es) {badtiles} "
                      f"({len(lines)}->{len(lines) - n}); crops deleted")

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
