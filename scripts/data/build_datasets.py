"""Build a VERSIONED dataset from capture source roots — the pipeline's build driver.

Discovers every game under the given ``--sources`` roots and builds a self-contained
``datasets/<NAME>/``:

    datasets/<NAME>/
      annotations/            annotate_ai_session records (AI games only) + overlays + summary
      <game>/{crops,yolo}     per-game classifier crops + YOLO labels
      detector/               assembled train/val split (train.txt / val.txt / data.yaml)
      games.json              manifest: name -> {capture, frames_dir, dir, kind}  (+ val game)

The manifest is what lets the training scripts consume MULTIPLE dataset versions:
``train_classifier.py --dataset datasets/v1 --dataset datasets/v2 ...`` and
``build_detector_dataset.py --dataset ...`` expand each ``games.json`` into their
usual per-game ``--data`` entries.

Sources: each root is scanned for the AI shapes (``run_*/game*.jsonl`` and
``run_*.jsonl``) plus the manual shape (``session*.jsonl``). Game names come from
``paths.ai_game_name`` and must be UNIQUE across all sources — keep run numbering
global (e.g. a future ``captures/raw/ai_session_2`` starts at ``run_15``), or the
build aborts on the collision. AI games go annotate -> build ``--from-annotations``;
manual games build direct (no annotate stage).

Unlike ``rebuild_datasets.py`` (DEPRECATED — in-place regen of the fixed legacy
layout), this runs IMMEDIATELY (``--dry-run`` to preview) and never touches other
dataset versions. Training is deliberately NOT run (GPU) — commands are printed.

Run (activate the conda ``auto`` env yourself; repo root;
PowerShell ``$env:PYTHONPATH = "."`` / bash ``export PYTHONPATH=.``):
    # everything under captures/raw/ai_session -> datasets/v2:
    python scripts/data/build_datasets.py v2
    # several roots (incl. the deprecated-but-kept manual data):
    python scripts/data/build_datasets.py v2 --sources captures/raw/ai_session captures/raw/manual
    # after capturing new runs: pick up only what's missing
    python scripts/data/build_datasets.py v2 --resume
    # big-RAM server:
    python scripts/data/build_datasets.py v2 --workers 16 --jobs 12
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

from majsoul_eye import paths

# Games whose GT frames are letterboxed; annotate + crop from the de-letterboxed
# frames instead (captures/intermediate/derived, see deletterbox_frames.py).
FRAMES_OVERRIDE = {
    "ai_run_5_game2": os.path.join(paths.DERIVED, "ai_run_5_game2_fixed"),
    "ai_run_5_game3": os.path.join(paths.DERIVED, "ai_run_5_game3_fixed"),
}

DEFAULT_VAL = "ai_run_8_game1"   # held-out whole game (classifier + detector convention)


def _posix(p: str) -> str:
    return str(p).replace(os.sep, "/")


def _spec_path(p: str) -> str:
    """Repo-root-relative POSIX when possible (all tooling runs from the repo root) —
    keeps ``NAME=DIR:CAPTURE`` CLI specs free of Windows drive colons."""
    try:
        return _posix(os.path.relpath(p))
    except ValueError:          # different drive
        return _posix(p)


def discover_games(sources: list) -> list:
    """Scan source roots for captures -> manifest entries (pure; no side effects).

    Returns [{name, dir, kind, capture, frames_dir}] with repo-relative POSIX paths.
    Raises SystemExit on an empty root or a cross-source name collision.
    """
    games, seen = [], {}
    for root in sources:
        ai = paths._ai_captures_in(root)
        manual = sorted(glob.glob(os.path.join(root, "session*.jsonl")))
        if not ai and not manual:
            raise SystemExit(f"no captures under {root!r} (expected run_*/game*.jsonl, "
                             f"run_*.jsonl or session*.jsonl)")
        for cap in ai + manual:
            name = paths.ai_game_name(cap)
            if name in seen:
                raise SystemExit(f"game name collision: {name!r} from both {seen[name]} and "
                                 f"{cap} — run numbering must be unique across sources")
            seen[name] = cap
            frames = FRAMES_OVERRIDE.get(name, paths.frames_dir_for(cap))
            games.append({"name": name, "dir": name,
                          "kind": "manual" if cap in manual else "ai",
                          "capture": _posix(cap), "frames_dir": _posix(frames)})
    return games


def apply_existing_dirs(games: list, ds_dir: str) -> list:
    """When resuming into an existing dataset, keep each known game's on-disk ``dir``
    from its games.json (e.g. v1's hand-moved ``precise_<name>`` dirs) instead of
    re-deriving ``dir = name`` — otherwise the rewritten manifest would point at
    dirs that don't exist. New games keep the plain-name default."""
    mpath = os.path.join(ds_dir, "games.json")
    if os.path.exists(mpath):
        try:
            old = json.load(open(mpath, encoding="utf-8"))
            old_dirs = {g["name"]: g["dir"] for g in old.get("games", [])}
        except Exception:
            old_dirs = {}
        for g in games:
            if g["name"] in old_dirs:
                g["dir"] = old_dirs[g["name"]]
    return games


def write_manifest(ds_dir: str, games: list, val: str) -> str:
    path = os.path.join(ds_dir, "games.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"val": val, "games": games}, f, ensure_ascii=False, indent=1)
    return path


def dataset_root(name: str) -> str:
    """'v2' -> datasets/v2; anything with a path separator is used as-is."""
    if "/" in name or os.sep in name:
        return name
    return os.path.join("datasets", name)


# ---- runner (same shape as the deprecated rebuild_datasets.py) ---------------

class Runner:
    def __init__(self, execute: bool):
        self.execute = execute
        self.env = dict(os.environ, PYTHONPATH=os.getcwd())

    def run(self, cmd: list) -> None:
        print("  $", " ".join(cmd))
        if not self.execute:
            return
        r = subprocess.run(cmd, env=self.env)
        if r.returncode != 0:
            raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}")

    def run_parallel(self, cmds: list, jobs: int) -> None:
        """Independent commands, ``jobs`` at a time (RAM-bound; see rebuild notes)."""
        for cmd in cmds:
            print("  $", " ".join(cmd))
        if not self.execute or not cmds:
            return
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"  ... running {len(cmds)} build(s), {jobs} at a time")

        def _one(cmd):
            return cmd, subprocess.run(cmd, env=self.env, capture_output=True, text=True)

        failed = []
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            for fut in as_completed([ex.submit(_one, c) for c in cmds]):
                cmd, res = fut.result()
                tag = os.path.basename(cmd[2]) if len(cmd) > 2 else " ".join(cmd)
                last = (res.stdout or "").strip().splitlines()
                print(f"  [rc={res.returncode}] {tag}  {last[-1] if last else ''}")
                if res.returncode != 0:
                    print((res.stderr or "")[-800:])
                    failed.append(cmd)
        if failed:
            raise SystemExit(f"{len(failed)} of {len(cmds)} build(s) failed: "
                             + ", ".join(os.path.basename(c[2]) for c in failed))

    def rm(self, path: str) -> None:
        exists = os.path.exists(path)
        print(f"  rm -rf {path}" + ("" if exists else "   (absent)"))
        if self.execute and exists:
            shutil.rmtree(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="dataset version name -> datasets/<name>/ "
                                 "(a path with '/' is used as-is)")
    ap.add_argument("--sources", nargs="+", default=[paths.RAW_AI_SESSION],
                    help=f"capture roots to scan (default: {paths.RAW_AI_SESSION}). "
                         f"Add e.g. {paths.RAW_MANUAL} to include the legacy manual sessions.")
    ap.add_argument("--val", default=None,
                    help=f"held-out whole game for the detector split (default {DEFAULT_VAL} "
                         f"when present among the discovered games; else you must pass one).")
    ap.add_argument("--stage", choices=["annotate", "dataset", "detector", "all"], default="all")
    ap.add_argument("--resume", action="store_true",
                    help="continue into an existing datasets/<name>: skip games that already "
                         "have annotations / a built game dir (detector split is redone).")
    ap.add_argument("--force", action="store_true",
                    help="delete datasets/<name> first and rebuild from scratch.")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, touch nothing.")
    ap.add_argument("--workers", type=int, default=None,
                    help="annotate_ai_session parallel workers (default: its own cap of 4).")
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 2)),
                    help="parallel build_dataset processes (RAM-bound; default min(8, cpu//2)).")
    ap.add_argument("--obb", action="store_true", help="emit OBB (8-point) YOLO labels.")
    args = ap.parse_args()

    py = sys.executable
    ds = dataset_root(args.name)
    ann = os.path.join(ds, "annotations")
    r = Runner(not args.dry_run)

    if os.path.exists(ds) and not (args.resume or args.force or args.dry_run):
        raise SystemExit(f"{ds} already exists — pass --resume to continue it, "
                         f"or --force to wipe and rebuild")
    if args.force:
        r.rm(ds)

    games = apply_existing_dirs(discover_games(args.sources), ds)
    names = [g["name"] for g in games]
    val = args.val or (DEFAULT_VAL if DEFAULT_VAL in names else None)
    if val is None or val not in names:
        raise SystemExit(f"--val {args.val or DEFAULT_VAL!r} not among discovered games "
                         f"{names} — pass --val <NAME>")
    print(f"{'DRY RUN' if args.dry_run else 'BUILD'} {ds}  ({len(games)} game(s), val={val})")
    print(f"sources: {args.sources}\ngames: {names}\n")

    do = lambda s: args.stage in ("all", s)

    # ---- stage 1: annotate (AI games only) ---------------------------------
    if do("annotate"):
        todo = [g for g in games if g["kind"] == "ai"
                and not (args.resume and os.path.exists(os.path.join(ann, g["name"] + ".jsonl")))]
        print(f"[1/3] annotate {len(todo)} game(s) -> {ann}")
        wk = ["--workers", str(args.workers)] if args.workers else []
        batch = [g["capture"] for g in todo if g["name"] not in FRAMES_OVERRIDE]
        if batch:
            r.run([py, "scripts/annotate/annotate_ai_session.py",
                   "--captures", *batch, "--out", ann, *wk])
        for g in todo:
            if g["name"] in FRAMES_OVERRIDE:
                r.run([py, "scripts/annotate/annotate_ai_session.py",
                       "--captures", g["capture"], "--frames-dir", g["frames_dir"],
                       "--out", ann, "--workers", "1"])
        print()

    # ---- stage 2: per-game crops + YOLO ------------------------------------
    if do("dataset"):
        cmds = []
        for g in games:
            out = os.path.join(ds, g["dir"])
            if args.resume and os.path.isdir(out):
                continue
            cmd = [py, "scripts/train/build_dataset.py", g["capture"], g["frames_dir"],
                   "--out", out, "--drop-violations"]
            if g["kind"] == "ai":
                cmd += ["--from-annotations", ann]
            if args.obb:
                cmd += ["--obb"]
            cmds.append(cmd)
        print(f"[2/3] build_dataset {len(cmds)} game(s) -> {os.path.join(ds, '<game>')}"
              f"   (jobs={args.jobs})")
        r.run_parallel(cmds, args.jobs)
        print()

    # ---- stage 3: detector split (always rebuilt over ALL games) -----------
    if do("detector"):
        det = os.path.join(ds, "detector")
        print(f"[3/3] build_detector_dataset -> {det}")
        data_args = []
        for g in games:
            data_args += ["--data",
                          f"{g['name']}={_spec_path(os.path.join(ds, g['dir'], 'yolo'))}:{g['capture']}"]
        r.run([py, "scripts/train/build_detector_dataset.py", *data_args,
               "--val", f"{val}:*", "--out", det])
        print()

    if not args.dry_run:
        print("manifest ->", write_manifest(ds, games, val))

    print("=" * 70)
    print("NOT run here (GPU / deliberate). Train on one or MORE dataset versions:\n")
    print(f"  {py} scripts/train/train_classifier.py --dataset {_posix(ds)} \\")
    print(f"      --val {val}:* --epochs 20 --out majsoul_eye/recognize/tile_classifier.pt")
    print(f"  {py} scripts/train/train_detector.py --data {_posix(os.path.join(ds, 'detector', 'data.yaml'))}")
    print("  # multi-version: expand several manifests into one run / one combined split:")
    print(f"  #   train_classifier.py --dataset datasets/v1 --dataset {_posix(ds)} --val {val}:*")
    print(f"  #   build_detector_dataset.py --dataset datasets/v1 --dataset {_posix(ds)} \\")
    print(f"  #       --val {val}:* --out datasets/detector_combined")


if __name__ == "__main__":
    main()
