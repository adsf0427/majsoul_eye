"""Regenerate ALL derived datasets from ``captures/raw/`` with the CURRENT code.

Everything except ``captures/raw/`` is gitignored (see .gitignore) — the GT
conversions (``captures/intermediate/gt``), the annotation records
(``out/ai_session_annotations``), the classifier crops + YOLO labels
(``datasets/precise_*``) and the assembled detector split (``datasets/detector``)
are all DERIVED. When the labeling code changes (e.g. the hero-tsumo fix that
made ``autolabel`` place the hero's drawn tile), the on-disk datasets are stale
until re-annotated. This driver re-runs the canonical linear pipeline
(STATUS.md §1.14/§1.15):

    annotate_ai_session (parallel)  ->  out/ai_session_annotations/*.jsonl
      -> build_dataset --from-annotations  ->  datasets/precise_<game>/{crops,yolo}
      -> build_detector_dataset            ->  datasets/detector/{train,val}.txt

It orchestrates the EXISTING scripts as subprocesses (each call is exactly the
vetted invocation), so there is no reimplemented logic to drift.

SCOPE — the AI (MahjongCopilot) games under ``captures/intermediate/gt`` (the
games with a playing hero, seat 0). Two of them (``ai_run_5_game2/3``) were
captured letterboxed and use de-letterboxed frames from
``captures/intermediate/derived/*_fixed`` — that per-game override is encoded in
``FRAMES_OVERRIDE`` below. The manual ``session5/6`` games and un-converted runs
(``run_13/14``) are NOT auto-run — their steps are PRINTED at the end (session5
needs ``crop_game.py`` first; runs 13/14 need ``ingest_run.py`` to convert).

This driver DOES NOT train — model weights need a GPU and are a deliberate step.
The classifier + detector train commands are printed at the end.

Run (conda ``auto`` env, repo root):
    # dry run — print every command that WOULD execute, touch nothing:
    PYTHONPATH=. $PY scripts/data/rebuild_datasets.py
    # actually rebuild (cleans the derived target dirs first, then regenerates):
    PYTHONPATH=. $PY scripts/data/rebuild_datasets.py --yes
    # limit to one stage:
    PYTHONPATH=. $PY scripts/data/rebuild_datasets.py --yes --stage annotate
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

from majsoul_eye import paths

# ---- config ---------------------------------------------------------------

# Games whose GT frames are letterboxed; annotate + crop from the de-letterboxed
# frames instead (see annotate_ai_session docstring / captures/intermediate/derived).
FRAMES_OVERRIDE = {
    "ai_run_5_game2": os.path.join(paths.DERIVED, "ai_run_5_game2_fixed"),
    "ai_run_5_game3": os.path.join(paths.DERIVED, "ai_run_5_game3_fixed"),
}

# Held-out validation game (whole game out — the cross-game val; STATUS.md §1.14).
VAL_GAME = "ai_run_8_game1"

ANN_OUT = os.path.join("out", "ai_session_annotations")
DATASETS = "datasets"
DETECTOR_OUT = os.path.join(DATASETS, "detector")
CLASSIFIER_OUT = os.path.join("majsoul_eye", "recognize", "tile_classifier.pt")


def game_name(capture: str) -> str:
    return os.path.splitext(os.path.basename(capture))[0]


def dataset_dir(name: str) -> str:
    return os.path.join(DATASETS, f"precise_{name}")


def gt_frames_dir(name: str) -> str:
    # convert_mjcopilot writes <out>/<name>/frames.jsonl next to <out>/<name>.jsonl
    return FRAMES_OVERRIDE.get(name, os.path.join(paths.GT, name))


# ---- runner ---------------------------------------------------------------

class Runner:
    def __init__(self, execute: bool):
        self.execute = execute
        self.env = dict(os.environ, PYTHONPATH=os.getcwd())

    def run(self, cmd: list[str]) -> None:
        print("  $", " ".join(cmd))
        if not self.execute:
            return
        r = subprocess.run(cmd, env=self.env)
        if r.returncode != 0:
            raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}")

    def rm(self, path: str) -> None:
        exists = os.path.exists(path)
        print(f"  rm -rf {path}" + ("" if exists else "   (absent)"))
        if self.execute and exists:
            shutil.rmtree(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true",
                    help="Actually execute (default is a dry run that only prints commands).")
    ap.add_argument("--stage", choices=["annotate", "dataset", "detector", "all"],
                    default="all", help="Run only one stage (default: all).")
    ap.add_argument("--no-clean", action="store_true",
                    help="Do NOT delete the derived target dirs before regenerating "
                         "(default cleans them for a from-scratch rebuild).")
    ap.add_argument("--workers", type=int, default=None,
                    help="annotate_ai_session parallel workers (default: the script's own cap).")
    ap.add_argument("--val", default=VAL_GAME, help=f"held-out game (default {VAL_GAME}).")
    args = ap.parse_args()

    py = sys.executable
    r = Runner(args.yes)

    captures = sorted(paths.converted_gt_captures())
    if not captures:
        raise SystemExit(f"no converted GT captures under {paths.GT} — convert AI runs first "
                         f"(scripts/data/ingest_run.py captures/raw/ai_session/run_N)")
    names = [game_name(c) for c in captures]
    if args.val not in names:
        raise SystemExit(f"--val game {args.val!r} not among converted games: {names}")

    print(f"{'EXECUTE' if args.yes else 'DRY RUN'} - {len(captures)} AI game(s): {names}")
    print(f"val (held out whole): {args.val}\n")

    do = lambda s: args.stage in ("all", s)

    # ---- stage 1: annotate (writes out/ai_session_annotations) -------------
    if do("annotate"):
        print("[1/3] annotate  ->", ANN_OUT)
        if not args.no_clean:
            r.rm(ANN_OUT)
        # Batch the games that use their DEFAULT frames in one parallel run; the
        # letterboxed overrides must go one-at-a-time (--frames-dir needs a single
        # --captures). Same --out => summaries merge (annotate_ai_session handles it).
        batch = [c for c, n in zip(captures, names) if n not in FRAMES_OVERRIDE]
        wk = ["--workers", str(args.workers)] if args.workers else []
        if batch:
            r.run([py, "scripts/annotate/annotate_ai_session.py",
                   "--captures", *batch, "--out", ANN_OUT, *wk])
        for c, n in zip(captures, names):
            if n in FRAMES_OVERRIDE:
                r.run([py, "scripts/annotate/annotate_ai_session.py",
                       "--captures", c, "--frames-dir", FRAMES_OVERRIDE[n],
                       "--out", ANN_OUT, "--workers", "1"])
        print()

    # ---- stage 2: build_dataset per game (crops + YOLO) --------------------
    if do("dataset"):
        print("[2/3] build_dataset --from-annotations  ->", dataset_dir("<game>"))
        for c, n in zip(captures, names):
            out = dataset_dir(n)
            if not args.no_clean:
                r.rm(out)
            r.run([py, "scripts/train/build_dataset.py", c, gt_frames_dir(n),
                   "--out", out, "--from-annotations", ANN_OUT, "--drop-violations"])
        print()

    # ---- stage 3: assemble the detector split ------------------------------
    if do("detector"):
        print("[3/3] build_detector_dataset  ->", DETECTOR_OUT)
        if not args.no_clean:
            r.rm(DETECTOR_OUT)
        data_args = []
        for c, n in zip(captures, names):
            data_args += ["--data", f"{n}={os.path.join(dataset_dir(n), 'yolo')}:{c}"]
        r.run([py, "scripts/train/build_detector_dataset.py", *data_args,
               "--val", f"{args.val}:*", "--out", DETECTOR_OUT])
        print()

    # ---- not run: training + the manual/uncoverted-game steps --------------
    val_cap = os.path.join(paths.GT, f"{args.val}.jsonl")
    clf_data = " ".join(
        f"--data {n}={os.path.join(dataset_dir(n), 'crops')}:{c}" for c, n in zip(captures, names))
    print("=" * 70)
    print("NOT run here (need a GPU / manual attention):\n")
    print("# retrain the 38-class classifier on the regenerated crops:")
    print(f"  {py} scripts/train/train_classifier.py {clf_data} \\")
    print(f"      --val {args.val}:* --epochs 20 --out {CLASSIFIER_OUT}\n")
    print("# retrain the YOLO detector on the regenerated split "
          "(see train_detector.py --help for the OOM flags):")
    print(f"  {py} scripts/train/train_detector.py --data {os.path.join(DETECTOR_OUT, 'data.yaml')}\n")
    print("# manual 'human-play' games (session5/6) - they also have a hero, so the "
          "tsumo fix applies;\n#   session5 was letterboxed, de-letterbox it first:")
    print("  # $PY scripts/data/crop_game.py captures/raw/manual/session5 "
          "captures/intermediate/derived/session5_16x9 --size 3840x2160")
    print("  # $PY scripts/train/build_dataset.py captures/raw/manual/session6.jsonl "
          "captures/raw/manual/session6 --out datasets/precise_session6 --drop-violations\n")
    print("# un-converted new runs (run_13/14): convert + build first, then re-run this driver:")
    print("  # $PY scripts/data/ingest_run.py captures/raw/ai_session/run_13")
    print("  # $PY scripts/data/ingest_run.py captures/raw/ai_session/run_14")

    if not args.yes:
        print("\n(dry run - nothing was executed; pass --yes to rebuild)")


if __name__ == "__main__":
    main()
