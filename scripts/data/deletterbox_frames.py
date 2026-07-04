"""De-letterbox captured frames into a new, self-contained capture dir.

Some capture sessions render the 16:9 board *letterboxed* — the browser window
was not exactly 16:9, so Majsoul fit its canvas to the window width and padded
black bars top/bottom. Concretely, ``captures/raw/ai_session/run_5`` game2/game3
(the reconnect session) are 1923x1142 with ~30px bars, whereas every other game
is a clean 1920x1080. ``annotate_ai_session.py``'s calibrated homography assumes
the board fills a 1920x1080 frame, so those two games annotate with a growing
vertical offset (~28px at the frame edges).

This tool fixes the *data*, not the pipeline (the letterbox is a one-off): for
each frame it auto-detects the black bars, crops to the content bbox, and
resizes the content back to 1920x1080. Corrected PNGs + a rewritten
``frames.jsonl`` go to a NEW directory, so the originals stay untouched and the
per-frame ``seq`` correspondence is preserved 1:1.

Run (conda `auto` env, repo root, PYTHONPATH=.):
  $PY scripts/data/deletterbox_frames.py --capture captures/raw/ai_session/run_5/game2.jsonl \
      --out captures/intermediate/derived/ai_run_5_game2_fixed
Then annotate the corrected frames without touching the original folder:
  $PY scripts/annotate/annotate_ai_session.py --captures captures/raw/ai_session/run_5/game2.jsonl \
      --frames-dir captures/intermediate/derived/ai_run_5_game2_fixed --qa-classifier
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from majsoul_eye import paths

TARGET = (1920, 1080)   # (w, h) canonical board frame the calibration assumes


def content_bbox(img: np.ndarray, thr: float = 12.0, min_frac: float = 0.5):
    """Bounding box of the non-black content. Returns (top, bot, left, right,
    degenerate). ``degenerate`` is True when the frame is (near) all-black or the
    content region is implausibly small — in that case callers should NOT crop
    (e.g. loading/transition frames), only resize."""
    gray = img.mean(axis=2) if img.ndim == 3 else img.astype(np.float32)
    rows = np.where(gray.mean(axis=1) >= thr)[0]
    cols = np.where(gray.mean(axis=0) >= thr)[0]
    h, w = img.shape[:2]
    if len(rows) == 0 or len(cols) == 0:
        return 0, h, 0, w, True
    t, b = int(rows[0]), int(rows[-1]) + 1
    l, r = int(cols[0]), int(cols[-1]) + 1
    if (b - t) < min_frac * h or (r - l) < min_frac * w:
        return 0, h, 0, w, True
    return t, b, l, r, False


def process(capture: str, out_dir: str, frames_dir: str | None = None,
            thr: float = 12.0) -> dict:
    src_dir = frames_dir or paths.frames_dir_for(capture)
    src_jsonl = os.path.join(src_dir, "frames.jsonl")
    out_frames = os.path.join(out_dir, "frames")
    os.makedirs(out_frames, exist_ok=True)

    recs, n_fixed, n_pass, n_degenerate, n_missing = [], 0, 0, 0, 0
    with open(src_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("status") != "ok" or not d.get("file"):
                recs.append(d)                 # keep non-ok rows verbatim
                continue

            file = paths.resolve_frame_path(d["file"], src_dir)
            img = cv2.imread(file)
            if img is None:
                n_missing += 1
                recs.append({**d, "status": "missing"})
                continue

            t, b, l, r, degenerate = content_bbox(img, thr)
            crop = img[t:b, l:r]
            out_img = (crop if (crop.shape[1], crop.shape[0]) == TARGET
                       else cv2.resize(crop, TARGET, interpolation=cv2.INTER_AREA))
            cropped = (t, b, l, r) != (0, img.shape[0], 0, img.shape[1])
            if degenerate:
                n_degenerate += 1
            elif cropped:
                n_fixed += 1
            else:
                n_pass += 1

            out_path = os.path.join(out_frames, os.path.basename(file))
            cv2.imwrite(out_path, out_img)
            recs.append({**d, "file": paths.rel_frame(out_path, out_dir),  # index-relative (portable)
                         "orig_dims": [img.shape[1], img.shape[0]],
                         "crop": [t, b, l, r]})

    with open(os.path.join(out_dir, "frames.jsonl"), "w", encoding="utf-8") as f:
        for d in recs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return {"total": len(recs), "cropped": n_fixed, "passthrough": n_pass,
            "degenerate": n_degenerate, "missing": n_missing}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True,
                    help="capture jsonl whose frames dir (stem/) to de-letterbox")
    ap.add_argument("--out", required=True,
                    help="new dir for corrected frames + rewritten frames.jsonl")
    ap.add_argument("--frames-dir", default=None,
                    help="override source frames dir (default: capture stem)")
    ap.add_argument("--thr", type=float, default=12.0,
                    help="row/col mean below this = black bar (default 12)")
    args = ap.parse_args()

    st = process(args.capture, args.out, frames_dir=args.frames_dir, thr=args.thr)
    print(f"{args.out}: {st['total']} frames  "
          f"cropped={st['cropped']} passthrough={st['passthrough']} "
          f"degenerate={st['degenerate']} missing={st['missing']}", flush=True)


if __name__ == "__main__":
    main()
