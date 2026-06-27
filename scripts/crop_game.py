"""Crop the 16:9 MajSoul game canvas out of non-fullscreen captures.

Some sessions (e.g. session5) were grabbed with the browser chrome and
pillarbox bars visible, so the raw frame is not 16:9. The game canvas is at a
fixed location across every frame of a session, so we detect it once and apply
the same crop to all frames, optionally resizing to a target resolution to
match the fullscreen sessions (3840x2160).

Layout produced (self-contained, non-destructive):
    <out_dir>/frames/*.png          cropped + resized frames (same filenames)
    <out_dir>/frames.jsonl          index with "file" rewritten to the crops

Usage:
    python scripts/crop_game.py captures/session5 captures/session5_16x9
    python scripts/crop_game.py captures/session5 captures/session5_16x9 --size 3840x2160
    python scripts/crop_game.py captures/session5 out --box top,bottom,left,right
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np


def _gray(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def detect_box(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Return (top, bottom, left, right) of the 16:9 game canvas.

    Strategy: the browser chrome at the top is bright; the game below it is
    dark. Find the chrome->game row transition for `top`. The game canvas
    spans down to the bottom (or the next bright chrome row). The browser
    pillarboxes the 16:9 canvas centered horizontally, so derive the width
    from the canvas height and center it.
    """
    h, w = gray.shape
    # Brightness of the central column band, immune to left-edge widgets.
    band = gray[:, int(w * 0.30):int(w * 0.70)].mean(axis=1)
    bright, dark = 150.0, 120.0

    # top: sharp chrome->game edge (a bright row immediately followed by a
    # sustained dark run). Requiring the very next rows to be dark avoids
    # stopping on a dark bookmark/tab bar inside the chrome.
    top = 0
    for r in range(0, h - 40):
        if (band[r] > bright and band[r + 1] < dark
                and np.median(band[r + 1:r + 41]) < dark):
            top = r + 1
            break

    # bottom: scan down from `top`; stop if chrome reappears for a long run.
    bottom = h
    for r in range(top + 50, h - 30):
        if band[r] > bright and np.median(band[r:r + 30]) > bright:
            bottom = r
            break

    # Pillarbox bars are pure black (near-zero column std). Measure the actual
    # game span over the canvas rows, then snap to an exact 16:9 box anchored
    # on the centre of that span (robust to left-edge browser widgets).
    canvas = gray[top:bottom, :]
    col_active = canvas.std(axis=0) > 3.0
    cols = np.where(col_active)[0]
    canvas_h = bottom - top
    canvas_w = round(canvas_h * 16 / 9)
    if canvas_w > w:  # letterboxed instead: width-constrained
        canvas_w = w
        canvas_h = round(w * 9 / 16)
    if len(cols):
        # centre on the right-hand content block (game), ignoring far-left widgets
        right_edge = int(cols[-1]) + 1
        left = right_edge - canvas_w
    else:
        left = (w - canvas_w) // 2
    left = max(0, min(left, w - canvas_w))
    right = left + canvas_w
    return top, bottom, left, right


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_dir", help="session dir containing frames.jsonl + frames/")
    ap.add_argument("out_dir", help="output session dir (mirrors layout)")
    ap.add_argument("--size", default="3840x2160",
                    help="WxH to resize crops to, or 'native' to keep crop size")
    ap.add_argument("--box", default=None,
                    help="override detection: top,bottom,left,right")
    args = ap.parse_args()

    idx_path = os.path.join(args.in_dir, "frames.jsonl")
    if not os.path.exists(idx_path):
        raise SystemExit(f"no frames.jsonl in {args.in_dir}")

    records = []
    with open(idx_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Pick a reference frame to detect the crop box.
    ref = next((r["file"] for r in records
                if r.get("file") and os.path.exists(r["file"])), None)
    if ref is None:
        raise SystemExit("no readable frame files referenced in frames.jsonl")

    if args.box:
        top, bottom, left, right = (int(x) for x in args.box.split(","))
    else:
        top, bottom, left, right = detect_box(_gray(ref))
    cw, ch = right - left, bottom - top
    print(f"crop box: rows[{top}:{bottom}] cols[{left}:{right}]  "
          f"= {cw}x{ch}  aspect={cw / ch:.4f} (16:9={16 / 9:.4f})")

    if args.size == "native":
        target = None
    else:
        tw, th = (int(x) for x in args.size.lower().split("x"))
        target = (tw, th)
        print(f"resizing crops -> {tw}x{th}")

    frames_out = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_out, exist_ok=True)

    out_records = []
    done = skipped = 0
    for rec in records:
        src = rec.get("file")
        if not src or not os.path.exists(src):
            out_records.append(rec)  # preserve non-ok rows as-is
            skipped += 1
            continue
        img = cv2.imread(src)
        crop = img[top:bottom, left:right]
        if target is not None:
            # upscale -> cubic; downscale -> area, for best quality
            interp = cv2.INTER_CUBIC if target[0] > cw else cv2.INTER_AREA
            crop = cv2.resize(crop, target, interpolation=interp)
        dst = os.path.join(frames_out, os.path.basename(src))
        cv2.imwrite(dst, crop)
        new_rec = dict(rec)
        new_rec["file"] = os.path.abspath(dst)
        out_records.append(new_rec)
        done += 1
        if done % 100 == 0:
            print(f"  {done} frames...")

    out_idx = os.path.join(args.out_dir, "frames.jsonl")
    with open(out_idx, "w", encoding="utf-8") as fh:
        for rec in out_records:
            fh.write(json.dumps(rec) + "\n")

    print(f"done: {done} cropped, {skipped} skipped -> {args.out_dir}")
    print(f"index: {out_idx}")


if __name__ == "__main__":
    main()
