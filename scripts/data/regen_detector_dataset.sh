#!/usr/bin/env bash
# Regenerate the YOLO detector dataset from captures/ — for a fresh machine (e.g. a
# GPU server) that has only the raw captures + the (unchanged) intermediate GT.
#
# INPUTS the machine must have (nothing else is needed — NO MahjongCopilot):
#   captures/raw/ai_session/...              the PNG frames (rsync from local;
#                                            include any manual deletions e.g. run_4's
#                                            disconnect tail — build_dataset skips
#                                            missing PNGs, so deleting = dropping them)
#   captures/intermediate/gt/                GT jsonl + <game>/frames.jsonl indexes.
#                                            UNCHANGED by the tsumo fix; copy the ~9MB
#                                            from local. (Regenerating it needs
#                                            convert_mjcopilot -> ../MahjongCopilot;
#                                            copying avoids that entirely.)
#   this repo at the fix commit              (replay.drawn_tile + autolabel tsumo slot)
#
# OUTPUTS (regenerated, overwriting any stale copies):
#   out/ai_session_annotations/*.jsonl       precise per-frame boxes (incl. tsumo hand)
#   datasets/precise_<game>/yolo/            per-game YOLO images+labels (HBB, axis-aligned)
#   datasets/detector/{train,val}.txt,data.yaml   the tar-and-go HBB detector dataset
#   with --obb, ALSO (from the SAME annotations — step 1 is shared, not re-run):
#   datasets/obb_precise_<game>/yolo/        per-game 8-point oriented labels
#   datasets/detector_obb/{train,val}.txt,data.yaml   the OBB detector dataset
#
# Run from the repo root in the `auto` conda env. Set PY to that python.
#   PY=/path/to/envs/auto/python  bash scripts/data/regen_detector_dataset.sh [--obb]
set -euo pipefail
PY=${PY:-python}
ANN=out/ai_session_annotations
VAL_GAME=ai_run_8_game1                       # held-out cross-game val (unchanged split)

# --- flags -------------------------------------------------------------------
DO_OBB=0
for arg in "$@"; do
  case "$arg" in
    --obb)     DO_OBB=1 ;;               # also emit the oriented (OBB) dataset
    --yes|-y)  : ;;                      # accepted, no-op (no interactive prompt exists)
    -h|--help) echo "usage: [PY=...] bash $0 [--obb] [--yes]"; exit 0 ;;
    *) echo "unknown arg: $arg (use --obb / --yes)" >&2; exit 2 ;;
  esac
done

# Discover games from the GT jsonls present (the 16 training games).
mapfile -t GAMES < <(ls captures/intermediate/gt/*.jsonl | xargs -n1 basename | sed 's/\.jsonl$//' | sort)
echo "games (${#GAMES[@]}): ${GAMES[*]}"

echo "=== 1/3 annotate all games -> $ANN (shared by HBB + OBB) ==="
# RAM-bound (each worker holds full-frame + homography buffers). Default --workers is
# now min(4, cpu//2); a big server can go higher, e.g. add: --workers 8
PYTHONPATH=. "$PY" scripts/annotate/annotate_ai_session.py --out "$ANN" --overlay-every 0 --workers 32

# build_variant TAG PRECISE_PREFIX OUT_DATASET [EXTRA_BUILD_FLAG]
# Steps 2/3 for one label format: per-game YOLO labels then the assembled dataset.
build_variant() {
  local tag="$1" pfx="$2" outds="$3" extra="${4:-}"
  echo "=== 2/3 [$tag] per-game YOLO labels (single-process; deal-drop + drop-violations) ==="
  local g gt fr out
  for g in "${GAMES[@]}"; do
    gt="captures/intermediate/gt/${g}.jsonl"
    fr=$(PYTHONPATH=. "$PY" -c "from majsoul_eye import paths; print(paths.frames_dir_for('${gt}'))")
    out="datasets/${pfx}${g}"
    rm -f "${out}/yolo/labels/"*.txt "${out}/yolo/images/"*.png 2>/dev/null || true  # no stale files
    PYTHONPATH=. "$PY" scripts/train/build_dataset.py "$gt" "$fr" \
      --out "$out" --from-annotations "$ANN" --drop-violations --no-crops $extra | tail -1 | sed "s/^/  [$tag $g] /"
  done
  echo "=== 3/3 [$tag] assemble detector dataset -> ${outds} (val = ${VAL_GAME}) ==="
  local DATA=() name
  for g in "${GAMES[@]}"; do
    name="$g"; [ "$g" = "$VAL_GAME" ] && name="v"
    DATA+=(--data "${name}=datasets/${pfx}${g}/yolo:captures/intermediate/gt/${g}.jsonl")
  done
  PYTHONPATH=. "$PY" scripts/train/build_detector_dataset.py "${DATA[@]}" --val "v:*" --out "$outds"
}

build_variant HBB "precise_" datasets/detector ""
[ "$DO_OBB" = 1 ] && build_variant OBB "obb_precise_" datasets/detector_obb "--obb"

echo "=== DONE ==="
echo "  HBB -> datasets/detector ; train: PYTHONPATH=. \$PY scripts/train/train_detector.py --data datasets/detector/data.yaml --batch <fit-your-GPU>"
[ "$DO_OBB" = 1 ] && echo "  OBB -> datasets/detector_obb ; train: PYTHONPATH=. \$PY scripts/train/train_detector.py --data datasets/detector_obb/data.yaml --model weights/pretrained/yolov8s-obb.pt --batch <fit-your-GPU>"
