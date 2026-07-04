"""One-shot ingest of a MahjongCopilot capture run: discover games -> build_dataset
(P1 gate + P2 erode) -> optionally retrain. Captures are already our `GTRecord`
format (autoplay_ai.py writes it inline under captures/raw/ai_session/), so there
is no convert step here anymore — legacy b64-wire runs are migrated in place
once by scripts/data/migrate_ai_to_gtrecord.py, not by this script.

A "run" dir is either a single game (has frames.jsonl directly, like run_1) or a
parent of game*/ subdirs (each with frames.jsonl, like run_3). Both are handled.

Examples (PowerShell):
  $PY = "C:/Users/zsx/miniforge3/envs/auto/python.exe"; $env:PYTHONPATH = "."
  # build every game in a run, naming them ai_<run>_<game>:
  & $PY scripts/data/ingest_run.py captures/raw/ai_session/run_4
  # then retrain on ALL ingested games, holding out one as val:
  & $PY scripts/data/ingest_run.py captures/raw/ai_session/run_4 --train --val ai_run4_game1:*

Run in the `auto` env.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

from majsoul_eye import paths


def discover_games(run_dir: str) -> list[tuple[str, str]]:
    """Return [(rel_dir_under_session, name), ...] for each game in run_dir.
    `rel` is relative to the run_dir's parent (matches the GTRecord jsonl's own
    layout: "run_13/game1" <-> sibling "run_13/game1.jsonl")."""
    run_dir = os.path.normpath(run_dir)
    parent = os.path.dirname(run_dir)
    run = os.path.basename(run_dir)
    games = []
    if os.path.exists(os.path.join(run_dir, "frames.jsonl")):
        games.append((run, f"ai_{run}"))                       # single-game run
    else:
        for sub in sorted(os.listdir(run_dir)):
            if os.path.exists(os.path.join(run_dir, sub, "frames.jsonl")):
                games.append((f"{run}/{sub}", f"ai_{run}_{sub}"))
    return parent, games


def run(cmd: list[str], env: dict) -> None:
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="MahjongCopilot run dir (single game or parent of game*/)")
    ap.add_argument("--mjcopilot", default="../MahjongCopilot")
    ap.add_argument("--captures", default=paths.GT)
    ap.add_argument("--datasets", default="datasets")
    ap.add_argument("--train", action="store_true", help="retrain on ALL ingested+manual datasets after")
    ap.add_argument("--val", default="", help="train val spec, e.g. ai_run4_game1:* (whole game held out)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="majsoul_eye/recognize/tile_classifier.pt")
    args = ap.parse_args()

    py = sys.executable
    env = dict(os.environ, PYTHONPATH=os.getcwd())

    parent, games = discover_games(args.run_dir)
    if not games:
        raise SystemExit(f"no games (frames.jsonl) found under {args.run_dir}")
    print(f"discovered {len(games)} game(s): {[n for _, n in games]}")

    # 1) build_dataset per game — AI captures are already GTRecord (no convert
    #    step: discover_games returns rel dirs like "run_13/game1"; the GTRecord
    #    jsonl is the sibling "run_13/game1.jsonl" and its frames dir is "run_13/game1/").
    for rel, name in games:
        cap = os.path.join(parent, rel) + ".jsonl"
        frames_dir = os.path.join(parent, rel)
        if not os.path.exists(cap):
            print(f"  SKIP {name}: no GTRecord at {cap} "
                  f"(capture with autoplay_ai or migrate a legacy b64 run first)")
            continue
        run([py, "scripts/train/build_dataset.py", cap, frames_dir + os.sep,
             "--out", os.path.join(args.datasets, name), "--drop-violations"], env)

    # crop summary
    for _, name in games:
        n = len(glob.glob(os.path.join(args.datasets, name, "crops", "*", "*.png")))
        print(f"  {name}: {n} crops")

    # 2) optional retrain on EVERY dataset that has a matching capture jsonl
    if args.train:
        data_args = []
        for rel, nm in games:
            crops = os.path.join(args.datasets, nm, "crops")
            cap = os.path.join(parent, rel) + ".jsonl"      # this game's GTRecord jsonl
            if os.path.isdir(crops) and os.path.exists(cap):
                data_args += ["--data", f"{nm}={crops}:{cap}"]
        # also pick up the erode-rebuilt manual sets if present
        for nm, cap in (("session5_erode", os.path.join(paths.RAW_MANUAL, "session5.jsonl")),
                        ("session6_erode", os.path.join(paths.RAW_MANUAL, "session6.jsonl"))):
            crops = os.path.join(args.datasets, nm, "crops")
            if os.path.isdir(crops) and os.path.exists(cap):
                data_args += ["--data", f"{nm}={crops}:{cap}"]
        cmd = [py, "scripts/train/train_classifier.py", *data_args,
               "--epochs", str(args.epochs), "--workers", "6", "--out", args.out]
        if args.val:
            cmd += ["--val", args.val]
        print("\nretraining on all datasets:")
        run(cmd, env)
    else:
        print("\n(skip --train) to retrain, re-run with --train [--val NAME:*]")


if __name__ == "__main__":
    main()
