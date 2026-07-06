# weights/

Local home for **base seeds** and **optional detector variants**. The `.pt` blobs
here are **git-ignored** (too big for GitHub's 50 MB limit); only this README and the
`.gitkeep` files are tracked, so the layout/convention is versioned while the weights
stay local. See the `weights/` block in `.gitignore`.

> **Shipped / runtime weights do NOT live here.** The recognizer is a self-contained,
> Akagi-free product, so its production weights sit next to the code that loads them:
> `majsoul_eye/recognize/tile_classifier.pt` (tracked) and
> `majsoul_eye/recognize/tile_detector.pt` (local-only). `weights/` is for training
> seeds and swap-in experiments, not for what the app loads by default.

## Layout

```
weights/
  pretrained/    # base models used as training start points (ultralytics downloads)
    yolov8s.pt   # default --model for train_detector.py (AABB detector seed)
    yolo26n.pt
  detector/      # TRAINED tile-detector variants, one VERSIONED file per run
    tile_detector_hbb_<name>.pt    # axis-aligned boxes (standard YOLO detect)
    tile_detector_obb_<name>.pt    # oriented boxes (YOLO-OBB, tilted 3D-table tiles)
    #   <name> = the launch_detector.sh run subdir (a timestamp by default), so runs
    #   never clobber each other; OBB additionally copies to recognize/tile_detector.pt.
```

## pretrained/ — training seeds

Base checkpoints fed to `--model` when training. Prefer a repo-relative path so
ultralytics loads the local copy instead of re-downloading to the cwd:

```bash
PYTHONPATH=. $PY scripts/train/train_detector.py --data datasets/detectorN/data.yaml \
    --model weights/pretrained/yolov8s.pt --imgsz 1280 --epochs 50
```

An OBB run seeds from an OBB base instead (ultralytics auto-downloads a bare name the
first time; move it here afterwards to keep the folder tidy):

```bash
--model weights/pretrained/yolo11s-obb.pt
```

## detector/ — optional variants (AABB vs OBB)

`launch_detector.sh` drops every run's `best.pt` here as a versioned
`tile_detector_<mode>_<name>.pt`, so **AABB** (`hbb`, standard axis-aligned boxes) and
**OBB** (`obb`, oriented boxes, a better fit for the perspective-tilted 河/副露 tiles)
runs accumulate side by side without clobbering. **OBB is the shipped default**: an `obb`
run additionally copies its `best.pt` to `majsoul_eye/recognize/tile_detector.pt` (the
weight the runtime loads), so no manual promotion is needed. To evaluate an HBB variant
or an older run, point the runtime `TileDetector(weights=...)` at that versioned file.
