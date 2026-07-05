"""Draw HUD seed ROIs (thin) + ink-snapped boxes (thick, with GT text) on real
frames for visual calibration. Writes <out>/hud_<game>_<seq>.png.

Usage: PYTHONPATH=. python scripts/inspect/overlay_hud.py \
           captures/raw/ai_session/run_1/game1/game1.jsonl --seqs 28 120 400 --out scratchpad/hudcal
"""
from __future__ import annotations

import argparse
import os

import cv2

from majsoul_eye import paths
from majsoul_eye.annotate.hud import hud_field_boxes
from majsoul_eye.capture.gtframes import load_pair
from majsoul_eye.coords import HUD_SEEDS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--seqs", nargs="+", type=int, required=True)
    ap.add_argument("--out", default="scratchpad/hudcal")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    game = paths.ai_game_name(args.capture)
    for seq in args.seqs:
        try:
            frame, state, region = load_pair(args.capture, seq)
        except SystemExit as e:
            print(f"skip seq {seq}: {e}")
            continue
        for name, nb in HUD_SEEDS.items():                    # seeds: thin yellow
            x0, y0, x1, y1 = region.norm_to_px(nb)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 1)
        for d in hud_field_boxes(frame, state, region):       # snapped: thick
            x0, y0, x1, y1 = d["px_box"]
            color = (0, 255, 0) if d.get("reliable", True) else (0, 0, 255)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(frame, f"{d['name']}={d['text']}", (x0, max(12, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        p = os.path.join(args.out, f"hud_{game}_{seq:06d}.png")
        cv2.imwrite(p, frame)
        print("->", p)


if __name__ == "__main__":
    main()
