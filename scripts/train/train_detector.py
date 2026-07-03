"""Train the 38-class tile DETECTOR (Ultralytics YOLO) on the free auto-labeled
YOLO export. Thin wrapper over ultralytics; the dataset (data.yaml + train/val
image lists, split by kyoku/game) comes from ``build_detector_dataset.py``.

    python scripts/train/train_detector.py --data datasets/detector_g1/data.yaml \
        --model weights/pretrained/yolov8s.pt --imgsz 1280 --epochs 50

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


def resolve_device(device_arg: str, cuda_available: bool):
    """Map the --device CLI value to an ultralytics ``device=`` spec.

    ``""`` keeps the historical default (GPU 0 when CUDA is present, else CPU).
    A single id (``"0"``) -> that int; multiple ids (``"0,1,2,3"``) -> a comma
    string, which is how ultralytics requests DDP across those GPUs; ``"cpu"``
    forces CPU. Whitespace around ids is tolerated. Pass a single GPU per process
    (with ``CUDA_VISIBLE_DEVICES``) to run independent experiments in parallel;
    pass several ids here to split ONE run across GPUs via DDP.
    """
    if not device_arg:
        return 0 if cuda_available else "cpu"
    d = device_arg.strip()
    if d.lower() == "cpu":
        return "cpu"
    ids = [p.strip() for p in d.split(",") if p.strip()]
    return int(ids[0]) if len(ids) == 1 else ",".join(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml from build_detector_dataset.py")
    ap.add_argument("--model", default="weights/pretrained/yolov8s.pt",
                    help="base weights / arch. A bare name (yolov8s.pt / yolo11s-obb.pt) makes "
                         "ultralytics auto-download to cwd; prefer weights/pretrained/<name> to "
                         "keep base seeds under weights/ (see weights/README.md)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=100,
                    help="early-stop after N epochs without val-mAP gain (ultralytics default 100)")
    ap.add_argument("--batch", type=int, default=8, help="images per batch (-1 = auto-batch); "
                    "with multi-GPU --device this is the GLOBAL batch, split across GPUs")
    ap.add_argument("--device", default="",
                    help="CUDA device(s): '' auto (GPU0/CPU), '0', '0,1,2,3' for DDP, 'cpu'")
    ap.add_argument("--project", default="", help="run dir parent (default: ultralytics runs/detect)")
    ap.add_argument("--name", default="tile_detector")
    ap.add_argument("--out", default="majsoul_eye/recognize/tile_detector.pt")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    device = resolve_device(args.device, torch.cuda.is_available())
    if device == "cpu":
        desc = "CPU"
    else:
        desc = ", ".join(f"cuda:{i} {torch.cuda.get_device_name(i)}"
                         for i in (int(x) for x in str(device).split(",")))
    print(f"device={device} ({desc})  model={args.model}  imgsz={args.imgsz}  "
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
