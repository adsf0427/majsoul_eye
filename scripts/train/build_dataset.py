"""Build an auto-labeled dataset from a synced capture (GT + screenshots), using
the PRECISE fullwarp annotation pipeline (majsoul_eye.annotate).

For every 'ok'/'timeout' screenshot, reconstruct the BoardState at its record,
run the precise annotator, and emit — from ONE calibration — both:
  <out>/crops/<tile>/<seq>_<i>.png   classifier dataset (hand + 4 rivers + melds
                                     + dora; perspective-deskewed 96px face crops)
  <out>/yolo/images/<seq>.png        detector images (the RESIZED 1920x1080 frame)
  <out>/yolo/labels/<seq>.txt        detector labels (YOLO: class cx cy w h, axis-
                                     aligned bbox of each box's original-px quad)

Only boxes the annotator marks ``reliable`` are emitted (drops unrendered newest
discards + low-fill/occluded cells). ``sideways`` tiles (riichi discard, called
meld tile) still go to YOLO but are EXCLUDED from classifier crops — their upright
glyph orientation is not recoverable from geometry, so an upright-only crop set
stays clean (runtime classifies both rotations for these).

The precise geometry is calibrated at 1920x1080 fullscreen 16:9; frames are
resized to that. Non-16:9 / letterboxed frames are skipped with a warning (their
river/meld boxes would be garbage — see session4).

Usage (conda `auto` env, repo root, PYTHONPATH=.):
  python scripts/train/build_dataset.py captures/intermediate/gt/ai_run_3_game1.jsonl \
         captures/raw/ai_session/run_3/game1 --out datasets/ai_g_run3_1
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("frames_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-violations", action="store_true",
                    help="Skip frames whose reconstructed state fails invariants.")
    ap.add_argument("--crop-size", type=int, default=96,
                    help="Saved classifier crop size (px). preprocess() resizes to 64 at "
                         "train/inference; a larger saved crop gives augmentation headroom.")
    ap.add_argument("--no-crops", action="store_true", help="Skip classifier crops.")
    ap.add_argument("--no-yolo", action="store_true", help="Skip YOLO detector labels.")
    args = ap.parse_args()

    import cv2  # auto env
    import numpy as np

    from majsoul_eye import paths
    from majsoul_eye.tiles import NAME_TO_ID
    from majsoul_eye.state.replay import check_invariants
    from majsoul_eye.capture.gtframes import build_seq_state, load_frames
    from majsoul_eye.annotate import build_homographies, annotate_frame, iter_tile_boxes, crop_box

    # seq -> reconstructed state; seq -> frame path (keep 'timeout' frames as before)
    seq_state = build_seq_state(args.capture)
    frames = load_frames(args.frames_dir, statuses=("ok", "timeout"))
    hom = build_homographies(1920, 1080)

    crops_dir = os.path.join(args.out, "crops")
    img_dir = os.path.join(args.out, "yolo", "images")
    lbl_dir = os.path.join(args.out, "yolo", "labels")
    for d in (crops_dir, img_dir, lbl_dir):
        os.makedirs(d, exist_ok=True)

    n_frames = n_crops = n_yolo = n_skip = n_letterbox = 0
    for seq in sorted(seq_state):
        if seq not in frames:
            continue
        state = seq_state[seq]
        if args.drop_violations and check_invariants(state):
            n_skip += 1
            continue
        frame = cv2.imread(frames[seq])
        if frame is None:
            n_skip += 1
            continue
        h, w = frame.shape[:2]
        if abs(w / h - 16 / 9) > 0.02:       # letterboxed / non-16:9 → precise geom invalid
            n_letterbox += 1
            continue
        if (w, h) != (1920, 1080):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)

        rec = annotate_frame(frame, state, hom)
        yolo_lines = []
        ci = 0
        for box in iter_tile_boxes(rec):
            if not box.reliable:
                continue
            cls = NAME_TO_ID.get(box.tile)
            if cls is None:
                continue
            # axis-aligned bbox (px) for YOLO, from the quad (river/meld) or px_box (hand/dora)
            if box.poly_original is not None:
                p = np.asarray(box.poly_original, dtype=np.float32)
                x0, y0 = float(p[:, 0].min()), float(p[:, 1].min())
                x1, y1 = float(p[:, 0].max()), float(p[:, 1].max())
            else:
                x0, y0, x1, y1 = (float(v) for v in box.px_box)
            if not args.no_yolo:
                cx, cy = (x0 + x1) / 2 / 1920, (y0 + y1) / 2 / 1080
                bw, bh = (x1 - x0) / 1920, (y1 - y0) / 1080
                yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            # classifier crop: skip sideways (upright orientation not geometry-recoverable)
            if not args.no_crops and not box.sideways:
                crop = crop_box(frame, box, size=args.crop_size)
                if crop.size:
                    cdir = os.path.join(crops_dir, box.tile)
                    os.makedirs(cdir, exist_ok=True)
                    cv2.imwrite(os.path.join(cdir, f"{seq:06d}_{ci:03d}.png"), crop)
                    ci += 1
                    n_crops += 1
        if yolo_lines and not args.no_yolo:
            cv2.imwrite(os.path.join(img_dir, f"{seq:06d}.png"), frame)   # RESIZED frame
            with open(os.path.join(lbl_dir, f"{seq:06d}.txt"), "w") as lf:
                lf.write("\n".join(yolo_lines) + "\n")
            n_yolo += 1
        n_frames += 1

    print(f"frames labeled: {n_frames}  crops: {n_crops}  yolo-imgs: {n_yolo}  "
          f"skipped: {n_skip}  letterbox-skipped: {n_letterbox}")
    print(f"dataset -> {args.out}")


if __name__ == "__main__":
    main()
