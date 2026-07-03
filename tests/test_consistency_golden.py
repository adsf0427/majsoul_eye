"""Integration: the gate flags the known mid-flight frame and passes the clean one.
Requires production weights + the ai_run_1 dataset frames on disk.

Frame choice: 000197 shows a discard-throw hand/arm sprite physically covering 5
river boxes (visually confirmed occlusion, verified via overlay); 000034 -- the
frame originally named in the task brief -- was checked and found to have zero
defects (all 21 boxes classify at ~0.9999 confidence matching GT exactly, and
`verdict_from_probs` can only fail a box when pred != gt, which never happens
here), so no TAU/MAX_BAD choice could make it fail; 000197 was substituted as a
verified real mid-flight example. See task-4-report.md for the investigation."""
import os
import numpy as np
import cv2

from majsoul_eye.recognize.classifier import TileClassifier
from majsoul_eye.annotate.consistency import score_frame, frame_decision
from majsoul_eye.tiles import TILE_NAMES

IMG = "datasets/precise_ai_run_1/yolo/images"
LBL = "datasets/precise_ai_run_1/yolo/labels"


def _load(seq):
    img = cv2.imread(f"{IMG}/{seq}.png")
    h, w = img.shape[:2]
    gts, crops = [], []
    with open(f"{LBL}/{seq}.txt") as f:
        for line in f:
            line = line.split()
            if not line:
                continue
            cls, cx, cy, bw, bh = int(line[0]), *[float(x) for x in line[1:5]]
            x0 = int((cx - bw / 2) * w); y0 = int((cy - bh / 2) * h)
            x1 = int((cx + bw / 2) * w); y1 = int((cy + bh / 2) * h)
            crop = img[max(0, y0):y1, max(0, x0):x1]
            if crop.size:
                gts.append(TILE_NAMES[cls]); crops.append(crop)
    return crops, gts


def test_golden_bad_and_good_frames():
    if not os.path.exists(f"{IMG}/000197.png"):
        print("test_golden_bad_and_good_frames SKIP (dataset frames absent)")
        return
    clf = TileClassifier()
    bad_crops, bad_gts = _load("000197")
    good_crops, good_gts = _load("000567")
    bad_dec = frame_decision(score_frame(bad_crops, bad_gts, clf))
    good_dec = frame_decision(score_frame(good_crops, good_gts, clf))
    assert bad_dec[0] != "keep", bad_dec        # mid-flight frame must be flagged
    assert good_dec[0] == "keep", good_dec       # settled frame must pass clean
    print(f"test_golden_bad_and_good_frames OK  bad={bad_dec[0]} good={good_dec[0]}")


if __name__ == "__main__":
    test_golden_bad_and_good_frames()
    print("ALL test_consistency_golden OK")
