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

SCOPE — the AI (MahjongCopilot) games under ``captures/intermediate/gt`` PLUS the
manual ``session5/6`` human-play games under ``captures/raw/manual``. All have a
playing hero (seat 0), so all feed BOTH the classifier crops and the YOLO detector.
The manual GT is already MJAI (no convert / no intermediate step), so those build
DIRECT from raw. Two AI games (``ai_run_5_game2/3``) were captured letterboxed and
use de-letterboxed frames from ``captures/intermediate/derived/*_fixed`` (see
``FRAMES_OVERRIDE``). The un-converted new runs (``run_13/14``) are NOT auto-run —
they need ``ingest_run.py`` to convert first; that step is PRINTED at the end.

This driver DOES NOT train — model weights need a GPU and are a deliberate step.
The classifier + detector train commands are printed at the end.

Both heavy stages parallelize per game: stage 1 (annotate) via ``--workers``
(forwarded to annotate_ai_session's process pool) and stage 2 (build_dataset) via
``--jobs`` (this driver fans the per-game builds out itself). Both are RAM-bound —
each worker/job holds full-frame + homography (+ crop) buffers — so scale them to
the box: big defaults freeze a laptop but a server can go much higher.

Run (conda ``auto`` env, repo root):
    # dry run — print every command that WOULD execute, touch nothing:
    PYTHONPATH=. $PY scripts/data/rebuild_datasets.py
    # actually rebuild (cleans the derived target dirs first, then regenerates):
    PYTHONPATH=. $PY scripts/data/rebuild_datasets.py --yes
    # on a big-RAM server, crank both stages' per-game parallelism:
    PYTHONPATH=. $PY scripts/data/rebuild_datasets.py --yes --workers 16 --jobs 12
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

# Manual "human-play" sessions (record_gt): the GT jsonl is ALREADY MJAI (Akagi tees
# raw+mjai live), so there is NO convert step and NO intermediate/gt — build_dataset
# reads captures/raw/manual/<name>.jsonl + its frames dir directly. They also have a
# playing hero (seat 0), so the hero-tsumo fix applies — include them in BOTH the
# classifier crops and the YOLO detector split. (session5/6 are 3840x2160 16:9, so no
# de-letterboxing is needed; build_dataset resizes to the canonical 1920x1080.)
MANUAL_SESSIONS = ["session5", "session6"]

# Held-out validation game (whole game out — the cross-game val; STATUS.md §1.14).
VAL_GAME = "ai_run_8_game1"


def manual_cap(name: str) -> str:
    return os.path.join(paths.RAW_MANUAL, f"{name}.jsonl")


def manual_frames_dir(name: str) -> str:
    return os.path.join(paths.RAW_MANUAL, name)

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

    def run_parallel(self, cmds: list[list[str]], jobs: int) -> None:
        """Run independent commands concurrently, ``jobs`` at a time (per-game fan-out).
        Each build_dataset is a single-threaded process, so this is a process pool;
        it is RAM-bound (each holds a frame + homography + crop buffers), so keep
        ``jobs`` within the machine's memory. Output is captured and printed per
        command on completion (not interleaved). Raises if any command fails."""
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
    ap.add_argument("--yes", action="store_true",
                    help="Actually execute (default is a dry run that only prints commands).")
    ap.add_argument("--stage", choices=["annotate", "dataset", "detector", "all"],
                    default="all", help="Run only one stage (default: all).")
    ap.add_argument("--no-clean", action="store_true",
                    help="Do NOT delete the derived target dirs before regenerating "
                         "(default cleans them for a from-scratch rebuild).")
    ap.add_argument("--workers", type=int, default=None,
                    help="stage-1 annotate_ai_session parallel workers (default: that "
                         "script's own conservative cap of 4; raise on a big-RAM server).")
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 2)),
                    help="stage-2 build_dataset processes to run in parallel (per-game "
                         "fan-out). RAM-bound — each holds a frame + homography + crop "
                         "buffers. Default min(8, cpu//2); on a big server pass more, "
                         "lower it if memory-constrained.")
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
        print(f"[2/3] build_dataset  ->  {dataset_dir('<game>')}   (jobs={args.jobs})")
        cmds = []                                    # each game is independent -> fan out
        for c, n in zip(captures, names):
            out = dataset_dir(n)
            if not args.no_clean:
                r.rm(out)                            # clean is fast; do it up front
            cmds.append([py, "scripts/train/build_dataset.py", c, gt_frames_dir(n),
                         "--out", out, "--from-annotations", ANN_OUT, "--drop-violations"])
        # manual sessions: GT is already MJAI, so build DIRECT (annotate_frame inline,
        # no separate annotate stage / --from-annotations).
        for name in MANUAL_SESSIONS:
            cap = manual_cap(name)
            if not os.path.exists(cap):
                print(f"  (skip {name}: {cap} absent)")
                continue
            out = dataset_dir(name)
            if not args.no_clean:
                r.rm(out)
            cmds.append([py, "scripts/train/build_dataset.py", cap, manual_frames_dir(name),
                         "--out", out, "--drop-violations"])
        r.run_parallel(cmds, args.jobs)
        print()

    # ---- stage 3: assemble the detector split ------------------------------
    if do("detector"):
        print("[3/3] build_detector_dataset  ->", DETECTOR_OUT)
        if not args.no_clean:
            r.rm(DETECTOR_OUT)
        data_args = []
        for c, n in zip(captures, names):
            data_args += ["--data", f"{n}={os.path.join(dataset_dir(n), 'yolo')}:{c}"]
        # manual sessions join the TRAIN side (val stays the held-out AI game).
        for name in MANUAL_SESSIONS:
            cap = manual_cap(name)
            if os.path.exists(cap) and os.path.isdir(os.path.join(dataset_dir(name), "yolo", "images")):
                data_args += ["--data", f"{name}={os.path.join(dataset_dir(name), 'yolo')}:{cap}"]
        r.run([py, "scripts/train/build_detector_dataset.py", *data_args,
               "--val", f"{args.val}:*", "--out", DETECTOR_OUT])
        print()

    # ---- not run: training + the manual/uncoverted-game steps --------------
    val_cap = os.path.join(paths.GT, f"{args.val}.jsonl")
    clf_data = " ".join(
        f"--data {n}={os.path.join(dataset_dir(n), 'crops')}:{c}" for c, n in zip(captures, names))
    for name in MANUAL_SESSIONS:                       # session5/6 crops too
        cap = manual_cap(name)
        if os.path.exists(cap):
            clf_data += f" --data {name}={os.path.join(dataset_dir(name), 'crops')}:{cap}"
    print("=" * 70)
    print("NOT run here (need a GPU / manual attention):\n")
    print("# retrain the 38-class classifier on the regenerated crops:")
    print(f"  {py} scripts/train/train_classifier.py {clf_data} \\")
    print(f"      --val {args.val}:* --epochs 20 --out {CLASSIFIER_OUT}\n")
    print("# retrain the YOLO detector on the regenerated split "
          "(see train_detector.py --help for the OOM flags):")
    print(f"  {py} scripts/train/train_detector.py --data {os.path.join(DETECTOR_OUT, 'data.yaml')}\n")
    print("# un-converted new runs (run_13/14): convert + build first, then re-run this driver:")
    print("  # $PY scripts/data/ingest_run.py captures/raw/ai_session/run_13")
    print("  # $PY scripts/data/ingest_run.py captures/raw/ai_session/run_14")

    if not args.yes:
        print("\n(dry run - nothing was executed; pass --yes to rebuild)")


if __name__ == "__main__":
    main()
