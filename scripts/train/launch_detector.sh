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
  [PY=...] [CUDA_VISIBLE_DEVICES=...] bash scripts/train/launch_detector.sh {hbb|obb} [options]

  Run from the repo root in the `majsoul_eye` conda env, e.g.
    PY=/hszhao-f1/h3011050/anaconda3/envs/majsoul_eye/bin/python

MODES  (pick dataset + default seed + output weight + run dir)
  hbb   axis-aligned boxes   datasets/detector/data.yaml
        seed weights/pretrained/yolov8s.pt      -> recognize/tile_detector.pt
        runs/hbb/<timestamp>/
  obb   oriented boxes       datasets/detector_obb/data.yaml
        seed weights/pretrained/yolov8s-obb.pt  -> weights/detector/tile_detector_obb.pt
        runs/obb/<timestamp>/

GPUS
  WHICH physical cards -> set CUDA_VISIBLE_DEVICES (e.g. 0,1,2,3 or 4,5,6,7).
  HOW MANY of them     -> --gpus N. The script spans the first N *visible* ids
                          "0,1,..,N-1", which ultralytics runs as one DDP job.
  --batch is the GLOBAL batch, split across those N GPUs (128 / 4 = 32 per GPU).

OPTIONS  (forwarded to train_detector.py; per-variant defaults in [brackets])
  --gpus N       how many visible GPUs to DDP over [4]. "cpu" forces CPU.
  --batch N      global batch across the run's GPUs [128]
  --epochs N     [60]
  --imgsz N      [1280]  small river tiles need this; 640 shrinks them to ~15px
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
  s = the baseline arch (HBB mAP50-95 ~0.957, OBB ~0.980). m/l trade VRAM+time for a
  little accuracy; yolo26n is nano/detect-only (HBB only, lower capacity).

EXAMPLES
  # HBB on cards 0-3, defaults (4 GPUs, bs 128) -> runs/hbb/<ts>/
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/launch_detector.sh hbb --gpus 4
  # OBB on the other half at the same time
  CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/train/launch_detector.sh obb --gpus 4
  # single-GPU smoke run with a named subdir
  CUDA_VISIBLE_DEVICES=0 bash scripts/train/launch_detector.sh hbb --gpus 1 --batch 32 --name smoke
  # bigger backbone + extra ultralytics flags
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/launch_detector.sh hbb --gpus 4 \
      --model weights/pretrained/yolo11m.pt -- --patience 30
EOF
}

GPUS=4
BATCH=128
EPOCHS=60
IMGSZ=1280
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
case "$MODE" in
  hbb) DATA=datasets/detector/data.yaml;     DEF_MODEL=weights/pretrained/yolov8s.pt;     OUT=recognize/tile_detector.pt ;;
  obb) DATA=datasets/detector_obb/data.yaml; DEF_MODEL=weights/pretrained/yolov8s-obb.pt; OUT=weights/detector/tile_detector_obb.pt ;;
  *) echo "mode must be hbb|obb (got: $MODE) — try --help" >&2; exit 2 ;;
esac

while [ $# -gt 0 ]; do
  case "$1" in
    --gpus)    GPUS="$2";    shift 2 ;;
    --batch)   BATCH="$2";   shift 2 ;;
    --epochs)  EPOCHS="$2";  shift 2 ;;
    --imgsz)   IMGSZ="$2";   shift 2 ;;
    --model)   MODEL="$2";   shift 2 ;;
    --name)    NAME="$2";    shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --) shift; PASSTHRU=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1 — try --help" >&2; exit 2 ;;
  esac
done

# --gpus N -> DDP device string over the first N *visible* ids (CUDA_VISIBLE_DEVICES
# already picked WHICH physical cards; here we just say how many of them to span).
if [ "$GPUS" = cpu ]; then
  DEVICE=cpu
elif [[ "$GPUS" =~ ^[0-9]+$ ]] && [ "$GPUS" -ge 1 ]; then
  DEVICE=$(seq -s, 0 $((GPUS-1)))
else
  echo "--gpus must be a positive integer or 'cpu' (got: $GPUS)" >&2; exit 2
fi

[ -f "$DATA" ] || { echo "MISSING dataset: $DATA — build it via scripts/data/regen_detector_dataset.sh" >&2; exit 1; }

# run dir: runs/<mode>/<timestamp>/  (override parent with --project, subdir with --name)
PROJECT="${PROJECT:-runs/$MODE}"
NAME="${NAME:-$(date +%Y%m%d_%H%M%S)}"

echo ">>> [$MODE] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} device=$DEVICE batch=$BATCH imgsz=$IMGSZ epochs=$EPOCHS"
echo "    run dir: $PROJECT/$NAME/   best -> $OUT"
exec env PYTHONPATH=. "$PY" scripts/train/train_detector.py \
  --data "$DATA" --model "${MODEL:-$DEF_MODEL}" --out "$OUT" \
  --imgsz "$IMGSZ" --epochs "$EPOCHS" --batch "$BATCH" \
  --device "$DEVICE" --project "$PROJECT" --name "$NAME" "${PASSTHRU[@]}"
