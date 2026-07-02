"""Train the 38-class tile DETECTOR (Ultralytics YOLO) on the free auto-labeled
YOLO export. Thin wrapper over ultralytics; the dataset (data.yaml + train/val
image lists, split by kyoku/game) comes from ``build_detector_dataset.py``.

    python scripts/train/train_detector.py --data datasets/detector_g1/data.yaml \
        --model yolov8s.pt --imgsz 1280 --epochs 50

Small tiles are the whole story: river tiles are ~40-60px in a 1920-wide frame, so
``imgsz>=1280`` is the main recall lever (the YOLO-default 640 shrinks them to
~15px). Ultralytics saves ``best.pt`` at the best-mAP epoch; we copy it to ``--out``
(default ``recognize/tile_detector.pt``), alongside the classifier weight, keeping
``recognize/`` the single home for shipped models.
"""
from __future__ import annotations

import argparse
import os
import shutil

# YOLO GPU memory creeps up over epochs (mosaic aug on high-instance mahjong frames)
# and fragments — a 16 GiB card OOM'd mid-run at batch 16 AND batch 8. Expandable
# segments let the allocator grow without fragmenting, which keeps a full run alive;
# on a 16 GiB card also prefer --batch 4 for an unattended 60-epoch run.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml from build_detector_dataset.py")
    ap.add_argument("--model", default="yolov8s.pt", help="base weights / arch (yolov8n/s/m.pt)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=100,
                    help="early-stop after N epochs without val-mAP gain (ultralytics default 100)")
    ap.add_argument("--batch", type=int, default=8, help="images per batch (-1 = auto-batch)")
    ap.add_argument("--project", default="", help="run dir parent (default: ultralytics runs/detect)")
    ap.add_argument("--name", default="tile_detector")
    ap.add_argument("--out", default="majsoul_eye/recognize/tile_detector.pt")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print(f"device={device} ({gpu})  model={args.model}  imgsz={args.imgsz}  "
          f"epochs={args.epochs}  batch={args.batch}", flush=True)

    # project defaults to ultralytics' own runs/detect; passing our own "runs/detect"
    # here would nest it (runs/detect/runs/detect/...), so only override when non-empty.
    kw = dict(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
              patience=args.patience, device=device, name=args.name)
    if args.project:
        kw["project"] = args.project
    model = YOLO(args.model)
    model.train(**kw)

    best = getattr(getattr(model, "trainer", None), "best", None)
    if best and os.path.exists(str(best)):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        shutil.copy(str(best), args.out)
        print(f"\nbest weights {best} -> {args.out}")
    else:
        print(f"\nWARNING: best.pt not found (trainer.best={best}); "
              f"look under {args.project}/{args.name}/weights/")


if __name__ == "__main__":
    main()
