"""Build an auto-labeled dataset from a synced capture (GT + screenshots).

For every 'ok' screenshot, reconstruct the BoardState at its `last_op_step`, locate
the board, auto-label the easy zones, and emit:
  <out>/crops/<tile>/*.png         classifier dataset (hand + dora tiles)
  <out>/yolo/images/<step>.png     detector images
  <out>/yolo/labels/<step>.txt     detector labels (YOLO: class cx cy w h)

Usage:
  python scripts/build_dataset.py captures/raw/manual/session2.jsonl captures/raw/manual/session2/ \
         --out datasets/session2 --locator fullscreen

⚠️ Coordinates are seeded from mycv (web 1080p) and NOT yet calibrated — eyeball a
few crops first (this is exactly what session2 is for). Run with the `auto` env.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

from majsoul_eye import paths
from majsoul_eye.capture.schema import read_records
from majsoul_eye.capture.sync import RELEVANT_EVENTS
from majsoul_eye.state.replay import Replayer, check_invariants
from majsoul_eye.label import label_frame, save_classification_crops, to_yolo_lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("frames_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--locator", choices=["fullscreen", "letterbox"], default="fullscreen")
    ap.add_argument("--drop-violations", action="store_true",
                    help="Skip frames whose reconstructed state fails invariants.")
    ap.add_argument("--min-bright", type=float, default=95.0,
                    help="Drop tile boxes whose crop mean brightness is below this.")
    ap.add_argument("--min-face-frac", type=float, default=0.35,
                    help="Drop tile boxes whose tile-face fraction is below this — catches "
                         "empty-felt / in-flight FRESHEST-discard cells that the brightness "
                         "gate misses (blue felt mean ~100-119 > min-bright). See label/quality.py.")
    # Defaults are the VALIDATED-optimal values (session5/6, default cloth): eroding the
    # packed river cell trims next-tile bottom-bleed + side 3D-perspective bleed. Measured:
    # river 93.7->94.8, 3s 0.39->0.98, 4p->1.0, side-seat S 0.89->0.99, overall 95.3->96.0.
    # Set to 0 to disable. NOTE: the runtime recognizer must apply the SAME erode to river cells.
    ap.add_argument("--river-erode-bottom", type=float, default=0.18,
                    help="Shrink RIVER cell crops by this frac off the bottom (trims the next "
                         "discard bleeding in — fixes 3s->2s etc). 0 disables.")
    ap.add_argument("--river-erode-side", type=float, default=0.08,
                    help="Shrink RIVER cell crops by this frac off each side (3D side-seat bleed).")
    args = ap.parse_args()

    import cv2  # auto env
    import numpy as np
    from majsoul_eye.normalize import locate_fullscreen, locate_letterbox
    from majsoul_eye.label.quality import is_tile_present
    locate = locate_fullscreen if args.locator == "fullscreen" else locate_letterbox

    # seq -> reconstructed state at that record (seq is globally unique; last_op_step is not)
    rp = Replayer()
    seq_state = {}
    for r in read_records(args.capture):
        rp.apply_record(r)
        if r.mjai and any(ev.get("type") in RELEVANT_EVENTS for ev in r.mjai):
            seq_state[r.seq] = rp.state.copy()

    # frames index
    idx = os.path.join(args.frames_dir, "frames.jsonl")
    if not os.path.exists(idx):
        raise SystemExit(f"no frames.jsonl in {args.frames_dir} (run record_gt --screenshots)")

    crops_dir = os.path.join(args.out, "crops")
    img_dir = os.path.join(args.out, "yolo", "images")
    lbl_dir = os.path.join(args.out, "yolo", "labels")
    for d in (crops_dir, img_dir, lbl_dir):
        os.makedirs(d, exist_ok=True)

    n_frames = n_crops = n_yolo = n_skip = 0
    with open(idx, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") not in ("ok", "timeout") or not rec.get("file"):
                continue
            seq = rec.get("seq", rec.get("step"))   # tolerate old field name
            state = seq_state.get(seq)
            if state is None:
                n_skip += 1
                continue
            if args.drop_violations and check_invariants(state):
                n_skip += 1
                continue
            src = paths.resolve_frame_path(rec["file"], args.frames_dir)
            frame = cv2.imread(src)
            if frame is None:
                n_skip += 1
                continue
            region = locate(frame)
            samples = label_frame(frame, state, region)
            # erode packed river cells to trim next-tile bleed (off by default)
            if args.river_erode_bottom or args.river_erode_side:
                import dataclasses
                eroded = []
                for s in samples:
                    if s.zone == "river" and s.kind == "tile":
                        nb = s.norm_box.erode(bottom=args.river_erode_bottom,
                                              left=args.river_erode_side, right=args.river_erode_side)
                        s = dataclasses.replace(s, norm_box=nb, px_box=region.norm_to_px(nb))
                    eroded.append(s)
                samples = eroded
            # emptiness gates: drop tile boxes that landed on empty felt / in-flight tiles.
            # brightness alone misses blue felt (mean ~100-119); tile-face fraction catches it.
            kept = []
            for x in samples:
                if x.kind == "tile":
                    c = region.crop(frame, x.norm_box)
                    if (c.size == 0 or float(np.asarray(c).mean()) < args.min_bright
                            or not is_tile_present(c, args.min_face_frac)):
                        n_skip += 1
                        continue
                kept.append(x)
            samples = kept
            n_crops += save_classification_crops(frame, region, samples, crops_dir, prefix=f"{seq:06d}_")
            lines = to_yolo_lines(samples)
            if lines:
                shutil.copy(src, os.path.join(img_dir, f"{seq:06d}.png"))
                with open(os.path.join(lbl_dir, f"{seq:06d}.txt"), "w") as lf:
                    lf.write("\n".join(lines) + "\n")
                n_yolo += 1
            n_frames += 1

    print(f"frames labeled: {n_frames}  crops: {n_crops}  yolo-imgs: {n_yolo}  skipped: {n_skip}")
    print(f"dataset -> {args.out}")


if __name__ == "__main__":
    main()
