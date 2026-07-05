"""Per-group mAP report for a detector checkpoint: tiles (0-37) vs HUD (38-54).
Gate: tile-group mAP50 >= 0.988 (0.993 baseline - 0.005 eps, spec §6).

Usage: PYTHONPATH=. python scripts/inspect/eval_detector_split.py \
           runs/detect/train/weights/best.pt datasets/v2/detector/data.yaml
"""
import sys

if len(sys.argv) < 3:
    print("Usage: python scripts/inspect/eval_detector_split.py <weights> <data>")
    sys.exit(1)

from ultralytics import YOLO

weights, data = sys.argv[1], sys.argv[2]
m = YOLO(weights)
r = m.val(data=data, imgsz=1280, plots=False)
names = r.names                       # {id: name}
ap50 = r.box.ap50                     # per-class AP50, aligned to r.box.ap_class_index
idx = list(r.box.ap_class_index)
tile = [ap50[i] for i, c in enumerate(idx) if c < 38]
hud = [ap50[i] for i, c in enumerate(idx) if c >= 38]
t = sum(tile) / len(tile) if tile else 0.0
h = sum(hud) / len(hud) if hud else 0.0
print(f"tiles mAP50={t:.4f} ({len(tile)} classes)   HUD mAP50={h:.4f} ({len(hud)} classes)")
for i, c in enumerate(idx):
    if c >= 38:
        print(f"  {names[c]:20s} AP50={ap50[i]:.4f}")
print("GATE:", "PASS" if t >= 0.988 else "FAIL (fall back to a separate HUD detector)")
