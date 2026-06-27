"""One-shot ingest of a MahjongCopilot capture run: discover games -> convert
(liqi wire -> our GT) -> build_dataset (P1 gate + P2 erode) -> optionally retrain.

A "run" dir is either a single game (has frames.jsonl directly, like run_1) or a
parent of game*/ subdirs (each with frames.jsonl, like run_3). Both are handled.

Examples (PowerShell):
  $PY = "C:/Users/zsx/miniforge3/envs/auto/python.exe"; $env:PYTHONPATH = "."
  # convert + build every game in a run, naming them ai_<run>_<game>:
  & $PY scripts/ingest_run.py captures/ai_session/run_4
  # then retrain on ALL ingested games, holding out one as val:
  & $PY scripts/ingest_run.py captures/ai_session/run_4 --train --val ai_run4_game1:*

Dev-only (convert step reaches into ../MahjongCopilot). Run in the `auto` env.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys


def discover_games(run_dir: str) -> list[tuple[str, str]]:
    """Return [(rel_dir_under_session, name), ...] for each game in run_dir.
    `rel` is relative to the run_dir's parent (so convert_mjcopilot --session=parent works)."""
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
    ap.add_argument("--captures", default="captures")
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

    # 1) convert all games in one call
    game_args = []
    for rel, name in games:
        game_args += ["--game", f"{rel}={name}"]
    run([py, "scripts/convert_mjcopilot.py", *game_args,
         "--session", parent, "--mjcopilot", args.mjcopilot, "--out", args.captures], env)

    # 2) build_dataset per game
    for _, name in games:
        run([py, "scripts/build_dataset.py",
             os.path.join(args.captures, f"{name}.jsonl"),
             os.path.join(args.captures, name) + os.sep,
             "--out", os.path.join(args.datasets, name), "--drop-violations"], env)

    # crop summary
    for _, name in games:
        n = len(glob.glob(os.path.join(args.datasets, name, "crops", "*", "*.png")))
        print(f"  {name}: {n} crops")

    # 3) optional retrain on EVERY dataset that has a matching capture jsonl
    if args.train:
        data_args = []
        for cap in sorted(glob.glob(os.path.join(args.captures, "*.jsonl"))):
            nm = os.path.splitext(os.path.basename(cap))[0]
            crops = os.path.join(args.datasets, nm, "crops")
            if os.path.isdir(crops):
                data_args += ["--data", f"{nm}={crops}:{cap}"]
        # also pick up the erode-rebuilt manual sets if present
        for nm, cap in (("session5_erode", "captures/session5.jsonl"),
                        ("session6_erode", "captures/session6.jsonl")):
            crops = os.path.join(args.datasets, nm, "crops")
            if os.path.isdir(crops) and os.path.exists(cap):
                data_args += ["--data", f"{nm}={crops}:{cap}"]
        cmd = [py, "scripts/train_classifier.py", *data_args,
               "--epochs", str(args.epochs), "--workers", "6", "--out", args.out]
        if args.val:
            cmd += ["--val", args.val]
        print("\nretraining on all datasets:")
        run(cmd, env)
    else:
        print("\n(skip --train) to retrain, re-run with --train [--val NAME:*]")


if __name__ == "__main__":
    main()
