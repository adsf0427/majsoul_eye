#!/usr/bin/env bash
# Launch ONE YOLO tile-detector training run. Thin wrapper over train_detector.py that
# only fills in the per-variant dataset / base weights / output path and the run dir;
# you pick the GPUs. See --help for the full menu.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PY=${PY:-python}

usage() {
  cat <<'EOF'
launch_detector.sh — start ONE tile-detector training run (HBB or OBB).

USAGE
  [PY=...] bash scripts/train/launch_detector.sh {hbb|obb} [options]

  Run from the repo root in the `majsoul_eye` conda env, e.g.
    PY=/hszhao-f1/h3011050/anaconda3/envs/majsoul_eye/bin/python

MODES  (pick detector split + default seed + output weight + run dir)
  hbb   axis-aligned boxes   <dataset>/detector/data.yaml
        seed weights/pretrained/yolov8s.pt      -> majsoul_eye/recognize/tile_detector.pt
        runs/hbb/<timestamp>/
  obb   oriented boxes       <dataset>/detector_obb/data.yaml
        seed weights/pretrained/yolov8s-obb.pt  -> weights/detector/tile_detector_obb.pt
        runs/obb/<timestamp>/

  <dataset> is a VERSIONED build dir from build_datasets.py (datasets/<name>,
  e.g. datasets/v2); choose it with --dataset [v2]. The mode picks the split
  subdir inside it (hbb -> detector/, obb -> detector_obb/).

GPUS
  Pick cards with --gpus, using PHYSICAL ids: "--gpus 4,5,6,7" DDPs over those
  cards; a bare count "--gpus 4" means cards 0..3; one specific card: "--gpus 2,".
  Do NOT pick cards via CUDA_VISIBLE_DEVICES: ultralytics select_device OVERWRITES
  it with the --device string, so an external CVD is silently ignored (two 4-GPU
  jobs launched that way landed on the same cards and OOM'd, 2026-07-05).
  --batch is the GLOBAL batch, split across the GPUs (64 / 4 = 16 per GPU).
  16 imgs/GPU ≈ 17 GiB at imgsz 1280; 32/GPU hits ~22.3 GiB — too tight on 24 GiB.

OPTIONS  (forwarded to train_detector.py; per-variant defaults in [brackets])
  --gpus N|IDS   physical GPU ids "4,5,6,7" (single card "2,"), or a count N
                 = cards 0..N-1 [4]. "cpu" forces CPU.
  --batch N      global batch across the run's GPUs [64]
  --epochs N     [60]
  --imgsz N      [1280]  small river tiles need this; 640 shrinks them to ~15px
  --dataset NAME versioned build dir the split lives in: a bare NAME -> datasets/NAME,
                 a value with a '/' is used as the dir as-is, a '*.yaml' is used
                 verbatim as the data.yaml (escape hatch for the flat regen layout) [v2]
  --model PATH   base seed weights [per-variant default below]
  --name NAME    run subdir under runs/<mode>/ [<timestamp>, e.g. 20260704_153012]
  --project DIR  run dir parent [runs/<mode>]
  -h, --help     this help
  --             everything after -- is forwarded verbatim to train_detector.py
                 (e.g. -- --patience 30 --lr0 0.001 --resume)

--model MENU  (head type MUST match the dataset: OBB dataset needs an -obb seed,
              HBB needs a plain seed. Local seeds are in weights/pretrained/; a bare
              name auto-downloads to cwd on first use — move it into pretrained/ after.)
  hbb (detect):  weights/pretrained/yolov8s.pt   [default]
                 yolov8{n,m,l,x}.pt   yolo11{n,s,m,l,x}.pt   yolo26n.pt (local)
  obb (obb):     weights/pretrained/yolov8s-obb.pt [default]
                 yolov8{n,m,l,x}-obb.pt   yolo11{n,s,m,l,x}-obb.pt
  s = the baseline arch (HBB mAP50-95 ~0.965, OBB ~0.985). m/l trade VRAM+time for a
  little accuracy; yolo26n is nano/detect-only (HBB only, lower capacity).

EXAMPLES
  # HBB on cards 0-3, defaults (4 GPUs, global bs 64, dataset v2) -> runs/hbb/<ts>/
  bash scripts/train/launch_detector.sh hbb --gpus 4
  # HBB against a specific build version
  bash scripts/train/launch_detector.sh hbb --gpus 4 --dataset v2
  # OBB on cards 4-7 at the same time (reads <dataset>/detector_obb)
  bash scripts/train/launch_detector.sh obb --gpus 4,5,6,7 --dataset v0
  # single-GPU smoke run on card 2 with a named subdir
  bash scripts/train/launch_detector.sh hbb --gpus 2, --batch 16 --name smoke
  # bigger backbone + extra ultralytics flags
  bash scripts/train/launch_detector.sh hbb --gpus 4 \
      --model weights/pretrained/yolo11m.pt -- --patience 30
EOF
}

GPUS=4
BATCH=64
EPOCHS=60
IMGSZ=1280
DATASET=v2
MODEL=""
NAME=""
PROJECT=""
PASSTHRU=()

# allow `-h`/`--help` before the mode too
case "${1:-}" in -h|--help) usage; exit 0 ;; esac
if [ $# -lt 1 ]; then
  echo "usage: bash $0 {hbb|obb} [options]   (try --help)" >&2
  exit 2
fi
MODE="$1"; shift
# The mode fixes the split SUBDIR + seed + output; the DATA path is resolved after the
# option loop, once --dataset (the versioned build dir) is known.
case "$MODE" in
  hbb) SUBDIR=detector;     DEF_MODEL=weights/pretrained/yolov8s.pt;     OUT=majsoul_eye/recognize/tile_detector.pt ;;
  obb) SUBDIR=detector_obb; DEF_MODEL=weights/pretrained/yolov8s-obb.pt; OUT=weights/detector/tile_detector_obb.pt ;;
  *) echo "mode must be hbb|obb (got: $MODE) — try --help" >&2; exit 2 ;;
esac

while [ $# -gt 0 ]; do
  case "$1" in
    --gpus)    GPUS="$2";    shift 2 ;;
    --batch)   BATCH="$2";   shift 2 ;;
    --epochs)  EPOCHS="$2";  shift 2 ;;
    --imgsz)   IMGSZ="$2";   shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --model)   MODEL="$2";   shift 2 ;;
    --name)    NAME="$2";    shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --) shift; PASSTHRU=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1 — try --help" >&2; exit 2 ;;
  esac
done

# --gpus -> the --device string, in PHYSICAL ids. Card picking must go through
# --device (not CUDA_VISIBLE_DEVICES): ultralytics select_device sets CVD to the
# --device string, clobbering any externally exported value.
if [ "$GPUS" = cpu ]; then
  DEVICE=cpu
elif [[ "$GPUS" == *,* ]]; then                   # id list "4,5,6,7" ("2," = one card)
  DEVICE="${GPUS%,}"
  [[ "$DEVICE" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "--gpus id list must be comma-separated ints (got: $GPUS)" >&2; exit 2; }
elif [[ "$GPUS" =~ ^[0-9]+$ ]] && [ "$GPUS" -ge 1 ]; then
  DEVICE=$(seq -s, 0 $((GPUS-1)))                 # bare count -> cards 0..N-1
else
  echo "--gpus must be a count, an id list like 4,5,6,7 (one card: '2,'), or 'cpu' (got: $GPUS)" >&2; exit 2
fi

# --dataset -> the detector data.yaml. A *.yaml/*.yml value is used verbatim (escape
# hatch for the flat regen layout, datasets/detector*/data.yaml); otherwise it names a
# versioned build dir — a bare NAME means datasets/NAME, a value with a '/' is used as
# the dir as-is — and the mode's SUBDIR (detector | detector_obb) picks the split.
case "$DATASET" in
  *.yaml|*.yml) DATA="$DATASET" ;;
  */*)          DATA="$DATASET/$SUBDIR/data.yaml" ;;
  *)            DATA="datasets/$DATASET/$SUBDIR/data.yaml" ;;
esac

[ -f "$DATA" ] || { echo "MISSING dataset: $DATA — build it via scripts/data/build_datasets.py <name> (pick the version with --dataset), or scripts/data/regen_detector_dataset.sh for the flat layout" >&2; exit 1; }

# run dir: runs/<mode>/<timestamp>/  (override parent with --project, subdir with --name)
PROJECT="${PROJECT:-runs/$MODE}"
NAME="${NAME:-$(date +%Y%m%d_%H%M%S)}"

echo ">>> [$MODE] device=$DEVICE (physical ids) batch=$BATCH imgsz=$IMGSZ epochs=$EPOCHS"
echo "    dataset: $DATA"
echo "    run dir: $PROJECT/$NAME/   best -> $OUT"
exec env PYTHONPATH=. "$PY" scripts/train/train_detector.py \
  --data "$DATA" --model "${MODEL:-$DEF_MODEL}" --out "$OUT" \
  --imgsz "$IMGSZ" --epochs "$EPOCHS" --batch "$BATCH" \
  --device "$DEVICE" --project "$PROJECT" --name "$NAME" "${PASSTHRU[@]}"
