#!/usr/bin/env bash
# Regenerate the YOLO detector dataset from captures/ — for a fresh machine (e.g. a
# GPU server) that has only the raw captures. Reads the CURRENT nested AI layout
# directly (captures/raw/ai_session/run_N/gameM/gameM.jsonl) via paths.ai_captures();
# the retired captures/intermediate/gt/ tree is NO LONGER used or needed.
#
# INPUTS the machine must have (nothing else — NO MahjongCopilot, NO intermediate/gt):
#   captures/raw/ai_session/run_N/gameM/      per-game GTRecord jsonl + frames/ + frames.jsonl
#                                             (rsync from local; any manual frame deletions —
#                                             e.g. run_4's disconnect tail — just carry over,
#                                             build_dataset skips missing PNGs)
#   captures/intermediate/derived/…_fixed/    ONLY for the letterboxed games (ai_run_5_game2/3):
#                                             their de-letterboxed frames + frames.jsonl (see
#                                             deletterbox_frames.py / build_datasets FRAMES_OVERRIDE).
#                                             Discovery applies the override automatically.
#   this repo at the fix commit               (replay.drawn_tile + autolabel tsumo slot)
#
# Game discovery + names + frames-dir (incl. the letterbox override) are shared with the
# versioned builder — this reuses scripts/data/build_datasets.discover_games, so the game
# set and split match `build_datasets.py`. AI games only (manual session*.jsonl are skipped).
# Override the scanned roots with env SOURCES="captures/raw/ai_session captures/raw/ai_session2".
#
# OUTPUTS (regenerated, overwriting any stale copies):
#   out/ai_session_annotations/<game>.jsonl  precise per-frame boxes (incl. tsumo hand)
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
SOURCES=${SOURCES:-captures/raw/ai_session}   # capture roots to scan (space-separated); AI games only
ANN_WORKERS=${ANN_WORKERS:-32}                # step-1 annotate workers for the batched (non-override) games

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
    -h|--help) echo "usage: [PY=... JOBS=N SOURCES='root..'] bash $0 [--obb|--obb-only] [--skip-annotate] [--jobs=N] [--yes]"; exit 0 ;;
    *) echo "unknown arg: $arg (use --obb / --obb-only / --skip-annotate / --jobs=N / --yes)" >&2; exit 2 ;;
  esac
done
if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [ "$JOBS" -eq 0 ]; then
  JOBS=$(nproc 2>/dev/null || echo 8); [ "$JOBS" -gt 16 ] && JOBS=16
fi

# --- discover the AI training games from the nested raw layout ---------------
# Reuse build_datasets.discover_games so game NAMES (ai_run_N_gameM), capture jsonl paths,
# and frames dirs (incl. the letterbox FRAMES_OVERRIDE -> de-letterboxed derived frames)
# match the versioned builder exactly. TSV: name<TAB>capture<TAB>frames_dir<TAB>override(0/1).
declare -A GT_OF FR_OF OV_OF                  # name -> capture jsonl / frames dir / is-letterbox-override
GAMES=(); DROPPED=()
while IFS=$'\t' read -r name cap fr ov; do
  [ -n "$name" ] || continue
  ov="${ov%$'\r'}"                            # strip a trailing CR (Python emits CRLF on Windows; no-op on Linux)
  if [ ! -d "$fr" ]; then                     # frames dir absent — usually a letterbox game whose de-letterboxed
    DROPPED+=("$name")                        # derived frames weren't rsync'd. Drop it LOUDLY, don't crash mid-build.
    echo "  DROP $name: frames dir missing ($fr)" >&2
    continue
  fi
  GAMES+=("$name"); GT_OF["$name"]="$cap"; FR_OF["$name"]="$fr"; OV_OF["$name"]="$ov"
done < <(PYTHONPATH=. "$PY" - $SOURCES <<'PY'
import sys
sys.path.insert(0, "scripts/data")
from build_datasets import discover_games, FRAMES_OVERRIDE
for g in discover_games(sys.argv[1:] or ["captures/raw/ai_session"]):
    if g["kind"] != "ai":                     # manual sessions are out of scope for regen
        continue
    print(g["name"], g["capture"], g["frames_dir"],
          "1" if g["name"] in FRAMES_OVERRIDE else "0", sep="\t")
PY
)
[ "${#GAMES[@]}" -gt 0 ] || { echo "no AI games discovered under: $SOURCES" >&2; exit 1; }
[ -n "${GT_OF[$VAL_GAME]:-}" ] || { echo "val game $VAL_GAME not among usable games (missing frames?) — set VAL_GAME" >&2; exit 1; }
[ "${#DROPPED[@]}" -eq 0 ] || echo "dropped ${#DROPPED[@]} game(s) with missing frames: ${DROPPED[*]}  (rsync their frames to build them)" >&2
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
  # Batch the non-letterbox games in one call; each letterbox-override game gets its own
  # --frames-dir call against the de-letterboxed derived frames (mirrors build_datasets).
  BATCH=(); OVR=()
  for g in "${GAMES[@]}"; do
    if [ "${OV_OF[$g]}" = 1 ]; then OVR+=("$g"); else BATCH+=("${GT_OF[$g]}"); fi
  done
  if [ "${#BATCH[@]}" -gt 0 ]; then
    # RAM-bound (each worker holds full-frame + homography buffers). --workers is ANN_WORKERS
    # (default 32 for a big server; lower it on a small box).
    PYTHONPATH=. "$PY" scripts/annotate/annotate_ai_session.py \
      --captures "${BATCH[@]}" --out "$ANN" --overlay-every 0 --workers "$ANN_WORKERS"
  fi
  for g in "${OVR[@]}"; do
    echo "  (letterbox) $g <- ${FR_OF[$g]}"
    PYTHONPATH=. "$PY" scripts/annotate/annotate_ai_session.py \
      --captures "${GT_OF[$g]}" --frames-dir "${FR_OF[$g]}" --out "$ANN" --overlay-every 0 --workers 1
  done
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
  local gt="${GT_OF[$g]}" fr="${FR_OF[$g]}" out="datasets/${2}${4}" log
  local hbb_imgs="datasets/precise_${g}/yolo/images"
  local reuse=()
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
    DATA+=(--data "${name}=datasets/${pfx}${g}/yolo:${GT_OF[$g]}")
  done
  PYTHONPATH=. "$PY" scripts/train/build_detector_dataset.py "${DATA[@]}" --val "v:*" --out "$outds"
done

echo "=== DONE ==="
[ "$OBB_ONLY" = 0 ] && echo "  HBB -> datasets/detector ; train: PYTHONPATH=. \$PY scripts/train/train_detector.py --data datasets/detector/data.yaml --batch <fit-your-GPU>"
[ "$DO_OBB" = 1 ]   && echo "  OBB -> datasets/detector_obb ; train: PYTHONPATH=. \$PY scripts/train/train_detector.py --data datasets/detector_obb/data.yaml --model weights/pretrained/yolov8s-obb.pt --batch <fit-your-GPU>"
