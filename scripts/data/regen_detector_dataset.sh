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
# FLAGS:
#   --obb             also build the oriented (OBB) dataset (shares step 1's annotations)
#   --obb-only        build ONLY the OBB dataset — skip the HBB variant entirely (implies --obb).
#                     Use when HBB is already built and you just want to add OBB.
#   --skip-annotate   reuse an existing out/ai_session_annotations (step 1) — the slow,
#                     already-done part; fails fast if any game's annotations are missing
#   --jobs=N          parallelism for the per-game builds (step 2). Default min(nproc,16).
#                     Also settable via env JOBS=N. Each game is independent.
#   --yes             accepted, no-op (no interactive prompt exists)
#
# OBB reuses HBB's frames automatically: when datasets/precise_<game>/yolo/images exists, the
# OBB build symlinks it and writes ONLY labels (no 17G frame re-encode — HBB & OBB images are
# byte-identical). If HBB images are absent it falls back to a full OBB build.
#
# Run from the repo root in the `auto` conda env. Set PY to that python.
#   PY=/path/to/envs/auto/python  bash scripts/data/regen_detector_dataset.sh [--obb|--obb-only] [--skip-annotate] [--jobs=N]
set -euo pipefail
PY=${PY:-python}
ANN=out/ai_session_annotations
VAL_GAME=ai_run_8_game1                       # held-out cross-game val (unchanged split)

# --- flags -------------------------------------------------------------------
DO_OBB=0
OBB_ONLY=0
SKIP_ANNOTATE=0
JOBS=${JOBS:-0}                               # 0 => auto (min(nproc,16)); env or --jobs overrides
for arg in "$@"; do
  case "$arg" in
    --obb)                DO_OBB=1 ;;         # also emit the oriented (OBB) dataset
    --obb-only)           DO_OBB=1; OBB_ONLY=1 ;;   # ONLY OBB — skip the (already-built) HBB
    --skip-annotate|--no-annotate) SKIP_ANNOTATE=1 ;;
    --jobs=*)             JOBS="${arg#*=}" ;;
    --yes|-y)             : ;;                # accepted, no-op (no interactive prompt exists)
    -h|--help) echo "usage: [PY=... JOBS=N] bash $0 [--obb|--obb-only] [--skip-annotate] [--jobs=N] [--yes]"; exit 0 ;;
    *) echo "unknown arg: $arg (use --obb / --obb-only / --skip-annotate / --jobs=N / --yes)" >&2; exit 2 ;;
  esac
done
if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [ "$JOBS" -eq 0 ]; then
  JOBS=$(nproc 2>/dev/null || echo 8); [ "$JOBS" -gt 16 ] && JOBS=16
fi

# Discover games from the GT jsonls present (the 16 training games).
mapfile -t GAMES < <(ls captures/intermediate/gt/*.jsonl | xargs -n1 basename | sed 's/\.jsonl$//' | sort)
echo "games (${#GAMES[@]}): ${GAMES[*]}"

# --- 1/3 annotate (the slow shared step; skippable once done) -----------------
if [ "$SKIP_ANNOTATE" = 1 ]; then
  echo "=== 1/3 SKIP annotate — reusing existing $ANN ==="
  miss=0
  for g in "${GAMES[@]}"; do
    [ -s "$ANN/${g}.jsonl" ] || { echo "  MISSING $ANN/${g}.jsonl" >&2; miss=1; }
  done
  [ "$miss" = 1 ] && { echo "annotations incomplete — rerun WITHOUT --skip-annotate" >&2; exit 1; }
else
  echo "=== 1/3 annotate all games -> $ANN (shared by HBB + OBB) ==="
  # RAM-bound (each worker holds full-frame + homography buffers). Default --workers is
  # now min(4, cpu//2); a big server can go higher, e.g. add: --workers 8
  PYTHONPATH=. "$PY" scripts/annotate/annotate_ai_session.py --out "$ANN" --overlay-every 0 --workers 32
fi

# --- 2/3 per-game YOLO labels — parallel across ALL (variant × game) builds ---
# Every (variant, game) build is independent (distinct out dir), so fan them out
# through one job pool instead of the old serial per-game loop.
VARIANTS=()
[ "$OBB_ONLY" = 1 ] || VARIANTS+=("HBB|precise_||datasets/detector")
[ "$DO_OBB" = 1 ]   && VARIANTS+=("OBB|obb_precise_|--obb|datasets/detector_obb")

FAILS="$(pwd)/.regen_fails.$$"; : > "$FAILS"
build_one() {                                 # tag pfx extra game
  local tag="$1" pfx="$2" extra="$3" g="$4"
  local gt="captures/intermediate/gt/${g}.jsonl" out="datasets/${2}${4}" fr log
  local hbb_imgs="datasets/precise_${g}/yolo/images"
  local reuse=()
  fr=$(PYTHONPATH=. "$PY" -c "from majsoul_eye import paths; print(paths.frames_dir_for('${gt}'))")
  log="${out%/}.build.log"
  if [ "$tag" = OBB ] && compgen -G "${hbb_imgs}/*.png" >/dev/null 2>&1; then
    # HBB & OBB frames are byte-identical — reuse them: symlink the images dir, write only
    # OBB labels. rm ONLY labels (NOT images: they'd delete HBB's frames through the symlink).
    mkdir -p "${out}/yolo/labels"
    rm -f "${out}/yolo/labels/"*.txt 2>/dev/null || true
    [ -L "${out}/yolo/images" ] || rm -rf "${out}/yolo/images" 2>/dev/null || true
    ln -sfn "$(cd "$hbb_imgs" && pwd)" "${out}/yolo/images"
    reuse=(--reuse-images "$hbb_imgs")
  else
    rm -f "${out}/yolo/labels/"*.txt "${out}/yolo/images/"*.png 2>/dev/null || true  # no stale files
  fi
  if PYTHONPATH=. "$PY" scripts/train/build_dataset.py "$gt" "$fr" \
       --out "$out" --from-annotations "$ANN" --drop-violations --no-crops $extra "${reuse[@]}" > "$log" 2>&1; then
    echo "  [ok   $tag $g${reuse:+ (reuse-hbb)}] $(tail -1 "$log")"
    rm -f "$log"                              # success: no stray log left in datasets/ (kept only on FAIL below)
  else
    echo "  [FAIL $tag $g] see $log" >&2
    echo "$tag $g" >> "$FAILS"                # short append; atomic under PIPE_BUF
  fi
}

n_builds=$(( ${#VARIANTS[@]} * ${#GAMES[@]} ))
echo "=== 2/3 per-game YOLO labels — ${#VARIANTS[@]} variant(s) × ${#GAMES[@]} games = ${n_builds} builds, JOBS=${JOBS} ==="
for v in "${VARIANTS[@]}"; do
  IFS='|' read -r tag pfx extra _outds <<<"$v"
  for g in "${GAMES[@]}"; do
    build_one "$tag" "$pfx" "$extra" "$g" &
    while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done   # throttle to JOBS in flight
  done
done
wait
if [ -s "$FAILS" ]; then
  echo "=== per-game build FAILURES ($(wc -l < "$FAILS")): ===" >&2; cat "$FAILS" >&2
  rm -f "$FAILS"; exit 1
fi
rm -f "$FAILS"

# --- 3/3 assemble the detector dataset(s) — quick, one call each -------------
for v in "${VARIANTS[@]}"; do
  IFS='|' read -r tag pfx _extra outds <<<"$v"
  echo "=== 3/3 [$tag] assemble detector dataset -> ${outds} (val = ${VAL_GAME}) ==="
  DATA=()
  for g in "${GAMES[@]}"; do
    name="$g"; [ "$g" = "$VAL_GAME" ] && name="v"
    DATA+=(--data "${name}=datasets/${pfx}${g}/yolo:captures/intermediate/gt/${g}.jsonl")
  done
  PYTHONPATH=. "$PY" scripts/train/build_detector_dataset.py "${DATA[@]}" --val "v:*" --out "$outds"
done

echo "=== DONE ==="
[ "$OBB_ONLY" = 0 ] && echo "  HBB -> datasets/detector ; train: PYTHONPATH=. \$PY scripts/train/train_detector.py --data datasets/detector/data.yaml --batch <fit-your-GPU>"
[ "$DO_OBB" = 1 ]   && echo "  OBB -> datasets/detector_obb ; train: PYTHONPATH=. \$PY scripts/train/train_detector.py --data datasets/detector_obb/data.yaml --model weights/pretrained/yolov8s-obb.pt --batch <fit-your-GPU>"
