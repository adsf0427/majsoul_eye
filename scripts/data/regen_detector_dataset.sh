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
#   datasets/precise_<game>/yolo/            per-game YOLO images+labels
#   datasets/detector/{train,val}.txt,data.yaml   the tar-and-go detector dataset
#
# Run from the repo root in the `auto` conda env. Set PY to that python.
#   PY=/path/to/envs/auto/python  bash scripts/data/regen_detector_dataset.sh
set -euo pipefail
PY=${PY:-python}
ANN=out/ai_session_annotations
VAL_GAME=ai_run_8_game1                       # held-out cross-game val (unchanged split)

# Discover games from the GT jsonls present (the 16 training games).
mapfile -t GAMES < <(ls captures/intermediate/gt/*.jsonl | xargs -n1 basename | sed 's/\.jsonl$//' | sort)
echo "games (${#GAMES[@]}): ${GAMES[*]}"

echo "=== 1/3 annotate all games -> $ANN ==="
# RAM-bound (each worker holds full-frame + homography buffers). Default --workers is
# now min(4, cpu//2); a big server can go higher, e.g. add: --workers 8
PYTHONPATH=. "$PY" scripts/annotate/annotate_ai_session.py --out "$ANN" --overlay-every 0

echo "=== 2/3 per-game YOLO labels (single-process; deal-drop + drop-violations) ==="
for g in "${GAMES[@]}"; do
  gt="captures/intermediate/gt/${g}.jsonl"
  fr=$(PYTHONPATH=. "$PY" -c "from majsoul_eye import paths; print(paths.frames_dir_for('${gt}'))")
  out="datasets/precise_${g}"
  rm -f "${out}/yolo/labels/"*.txt "${out}/yolo/images/"*.png 2>/dev/null || true  # no stale files
  PYTHONPATH=. "$PY" scripts/train/build_dataset.py "$gt" "$fr" \
    --out "$out" --from-annotations "$ANN" --drop-violations --no-crops | tail -1 | sed "s/^/  [$g] /"
done

echo "=== 3/3 assemble detector dataset (val = ${VAL_GAME}) ==="
DATA=()
for g in "${GAMES[@]}"; do
  name="$g"; [ "$g" = "$VAL_GAME" ] && name="v"
  DATA+=(--data "${name}=datasets/precise_${g}/yolo:captures/intermediate/gt/${g}.jsonl")
done
PYTHONPATH=. "$PY" scripts/train/build_detector_dataset.py "${DATA[@]}" --val "v:*" --out datasets/detector
echo "=== DONE -> datasets/detector ; now: PYTHONPATH=. \$PY scripts/train/train_detector.py --data datasets/detector/data.yaml --batch <fit-your-GPU> ==="
