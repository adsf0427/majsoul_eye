#!/usr/bin/env bash
# Launch ONE 38-class tile-classifier training run. Thin wrapper over train_classifier.py
# that fills in the versioned dataset (crops), holds out the SAME whole games the detector
# split used (read from the dataset's games.json), and picks the GPU. See --help.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PY=${PY:-python}

usage() {
  cat <<'EOF'
launch_classifier.sh — start ONE 38-class tile-classifier training run.

USAGE
  [PY=...] bash scripts/train/launch_classifier.sh [options]

  Run from the repo root in the `majsoul_eye` conda env, e.g.
    PY=/hszhao-f1/h3011050/anaconda3/envs/majsoul_eye/bin/python

WHAT IT DOES
  Trains TileNet on the crops of a VERSIONED build dir (build_datasets.py), saving the
  best-val weight to majsoul_eye/recognize/tile_classifier.pt. With no explicit --val it
  reads <dataset>/games.json's `val` list and holds out those WHOLE games — the SAME
  held-out games the detector uses — so the two models stay comparable with zero manual
  sync. (Split is by game/kyoku, NEVER by frame: the same physical tile spans ~10 frames,
  so a frame split leaks it into val and inflates accuracy.)

GPU
  The classifier is a small plain-PyTorch CNN (single GPU, no DDP), so pick the card with
  --gpu, which sets CUDA_VISIBLE_DEVICES — the OPPOSITE of launch_detector.sh, whose
  ultralytics select_device would clobber CVD. Training is quick (~minutes); run it on any
  free card, e.g. after / beside the two detector DDP jobs.

OPTIONS  (forwarded to train_classifier.py; defaults in [brackets])
  --dataset NAME  versioned build dir with the crops + games.json: a bare NAME ->
                  datasets/NAME, a value with a '/' is used as the dir as-is [v2]
  --gpu ID        physical GPU id via CUDA_VISIBLE_DEVICES [0]; "cpu" forces CPU
  --epochs N      [20]
  --batch N       [128]
  --workers N     DataLoader workers (keeps cv2.imread from starving the GPU) [6]
  --val SPEC      hold-out spec 'NAME:*' (whole game) or 'NAME:k1,k2' (kyoku ids);
                  repeatable. Given even once, it REPLACES the games.json auto-holdout.
  --out PATH      output weight [majsoul_eye/recognize/tile_classifier.pt]
  --dry-run       print the resolved train_classifier.py command and exit (no training)
  -h, --help      this help
  --              everything after -- is forwarded verbatim to train_classifier.py

EXAMPLES
  # default: dataset v2, card 0, holdout = the val games in datasets/v2/games.json
  bash scripts/train/launch_classifier.sh --dataset v2 --gpu 0
  # override the holdout to a single game, train longer
  bash scripts/train/launch_classifier.sh --dataset v2 --gpu 5 --val ai_session_run_8_game1:* --epochs 30
  # preview the exact command without training
  bash scripts/train/launch_classifier.sh --dataset v2 --dry-run
EOF
}

DATASET=v2
GPU=0
EPOCHS=20
BATCH=128
WORKERS=6
OUT=majsoul_eye/recognize/tile_classifier.pt
DRYRUN=0
USER_VAL=0
VAL_FLAGS=()
PASSTHRU=()

case "${1:-}" in -h|--help) usage; exit 0 ;; esac

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --gpu)     GPU="$2";     shift 2 ;;
    --epochs)  EPOCHS="$2";  shift 2 ;;
    --batch)   BATCH="$2";   shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --val)     VAL_FLAGS+=(--val "$2"); USER_VAL=1; shift 2 ;;
    --out)     OUT="$2";     shift 2 ;;
    --dry-run) DRYRUN=1;     shift ;;
    --) shift; PASSTHRU=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1 — try --help" >&2; exit 2 ;;
  esac
done

# --dataset -> the versioned build dir (bare NAME -> datasets/NAME; a '/'-path used as-is).
case "$DATASET" in
  */*) DSDIR="$DATASET" ;;
  *)   DSDIR="datasets/$DATASET" ;;
esac
MANIFEST="$DSDIR/games.json"
[ -f "$MANIFEST" ] || { echo "MISSING dataset manifest: $MANIFEST — build it via scripts/data/build_datasets.py <name> (pick the version with --dataset)" >&2; exit 1; }

# No explicit --val? Hold out the SAME whole games the detector split used, straight from
# the manifest's `val` list (a string or a list; each entry -> '<game>:*').
if [ "$USER_VAL" -eq 0 ]; then
  VAL_LIST=$("$PY" -c 'import json,sys; m=json.load(open(sys.argv[1],encoding="utf-8")); v=m.get("val") or []; v=[v] if isinstance(v,str) else v; print("\n".join(v))' "$MANIFEST")
  while IFS= read -r g; do
    [ -n "$g" ] && VAL_FLAGS+=(--val "$g:*")
  done <<< "$VAL_LIST"
  [ ${#VAL_FLAGS[@]} -gt 0 ] || echo "WARNING: $MANIFEST has no 'val' — training with NO held-out val (accuracy will be optimistic)" >&2
fi

# --gpu -> CUDA_VISIBLE_DEVICES. The classifier is plain torch, so masking with CVD is the
# correct way to pin the card (unlike the ultralytics detector, which overwrites CVD).
if [ "$GPU" = cpu ]; then CVD=""; else CVD="$GPU"; fi

echo ">>> [classifier] gpu=$GPU dataset=$DSDIR epochs=$EPOCHS batch=$BATCH workers=$WORKERS"
echo "    val: ${VAL_FLAGS[*]:-(none — no holdout)}"
echo "    out: $OUT"

CMD=("$PY" scripts/train/train_classifier.py --dataset "$DSDIR"
     "${VAL_FLAGS[@]}" --epochs "$EPOCHS" --batch "$BATCH" --workers "$WORKERS"
     --out "$OUT" "${PASSTHRU[@]}")

if [ "$DRYRUN" -eq 1 ]; then
  echo "DRY-RUN: env PYTHONPATH=. CUDA_VISIBLE_DEVICES=$CVD ${CMD[*]}"
  exit 0
fi
exec env PYTHONPATH=. CUDA_VISIBLE_DEVICES="$CVD" "${CMD[@]}"
