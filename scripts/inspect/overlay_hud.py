"""Draw HUD seed ROIs (thin) + ink-snapped boxes (thick, with GT text) on real
frames for visual calibration. Writes <out>/hud_<game>_<seq>.png.

Usage: PYTHONPATH=. python scripts/inspect/overlay_hud.py \
           captures/raw/ai_session/run_1/game1/game1.jsonl --seqs 28 120 400 --out scratchpad/hudcal

--buttons additionally overlays the action-button locator (Task 7 Step 5):
locate_button_candidates in cyan (thin, unlabeled — raw contour candidates)
and button_boxes assignments in green (ok, labeled with the button class) or
red (count_mismatch), plus the GT-expected button list printed top-left.
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
    ap.add_argument("--buttons", action="store_true",
                     help="also overlay locate_button_candidates/button_boxes")
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
        if args.buttons:
            from majsoul_eye.annotate.hud import button_boxes, locate_button_candidates
            from majsoul_eye.hud import buttons_for_ops

            for cx0, cy0, cx1, cy1 in locate_button_candidates(frame, region):
                cv2.rectangle(frame, (cx0, cy0), (cx1, cy1), (255, 255, 0), 1)
            expected = buttons_for_ops(state.pending_ops or [])
            for d in button_boxes(frame, state, region):
                box = d.get("px_box")
                if box is None:
                    continue
                bx0, by0, bx1, by1 = box
                mismatch = d.get("flag") == "count_mismatch"
                color = (0, 0, 255) if mismatch else (0, 255, 0)
                cv2.rectangle(frame, (bx0, by0), (bx1, by1), color, 2)
                label = d["name"] + (" MISMATCH" if mismatch else "")
                cv2.putText(frame, label, (bx0, max(12, by0 - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"GT expected: {expected}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        p = os.path.join(args.out, f"hud_{game}_{seq:06d}.png")
        cv2.imwrite(p, frame)
        print("->", p)


if __name__ == "__main__":
    main()
