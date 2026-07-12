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
``paths.ai_game_name``, which tags each name with its SOURCE-ROOT basename
(``captures/raw/ai_session2/run_1/game1`` -> ``ai_session2_run_1_game1``), so the same
run number under different roots never collides — no cross-source run-renumbering
needed. AI games go annotate -> build ``--from-annotations``; manual games build
direct (no annotate stage).

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
    # big-RAM server — one knob for both stages, or tune each:
    python scripts/data/build_datasets.py v2 -j 12
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

# (The letterboxed run_5 game2/game3 frames used to be routed through a
# FRAMES_OVERRIDE map to de-letterboxed derived copies; they were fixed in place
# on 2026-07-05 — see deletterbox_frames.py --inplace — so every game now uses its
# own frames dir with no special-casing.)

DEFAULT_VAL = "ai_session_run_8_game1"   # held-out whole game (classifier + detector convention)

# Stage-2 default: per-game build_dataset processes are RAM-bound, so cap at 8.
DEFAULT_JOBS = max(1, min(8, (os.cpu_count() or 4) // 2))

# Per-format layout. Both HBB and OBB detector splits can live in ONE version; the OBB
# per-game build then goes in a sibling "<dir>__obb" whose yolo/images is a SYMLINK to the
# HBB build's images (OBB & HBB frames are byte-identical — no ~17G re-encode) and holds
# only the 9-point labels. A SINGLE-format build keeps the plain "<dir>" (so OBB-only matches
# the historical layout, crops included). launch_detector.sh {hbb|obb} --dataset <name>
# resolves the split subdir per format.
FORMATS = {
    "hbb": {"split": "detector",     "obb": False},
    "obb": {"split": "detector_obb", "obb": True},
}
OBB_SIBLING_SUFFIX = "__obb"


def resolve_formats(hbb: bool, obb: bool) -> list:
    """--hbb/--obb -> ordered format list. Neither -> ['hbb'] (historical default);
    --obb alone -> ['obb'] (historical OBB-only); both -> ['hbb','obb'] (one version
    carrying detector/ + detector_obb/). HBB is first so the OBB pass can reuse its frames."""
    fmts = []
    if hbb:
        fmts.append("hbb")
    if obb:
        fmts.append("obb")
    return fmts or ["hbb"]


def game_yolo_dir(ds: str, game_dir: str, fmt: str, formats) -> str:
    """Per-game yolo dir for a format. OBB uses the sibling '<dir>__obb' ONLY when it
    coexists with HBB (a dual build); otherwise the plain '<dir>' — preserving the
    single-format (incl. OBB-only) layout and its crops path for the classifier."""
    dual = "hbb" in formats and "obb" in formats
    suffix = OBB_SIBLING_SUFFIX if (fmt == "obb" and dual) else ""
    return os.path.join(ds, game_dir + suffix, "yolo")


def resolve_parallelism(parallel, workers, jobs):
    """Fold the shared ``-j/--parallel`` knob into the two per-stage degrees.

    Both heavy stages are "N games at once, one OS process each" — stage-1 annotate
    (``--workers``) and stage-2 per-game build (``--jobs``) — and they run
    sequentially, so one number is safe for both. ``-j`` sets the default for each;
    an explicit ``--workers``/``--jobs`` overrides its own stage. ``workers`` may stay
    ``None`` (annotate then falls back to its own conservative cap of 4); ``jobs``
    always resolves to a concrete int for ``run_parallel``.
    """
    eff_workers = workers if workers is not None else parallel
    eff_jobs = jobs if jobs is not None else (parallel if parallel is not None else DEFAULT_JOBS)
    return eff_workers, eff_jobs


def _posix(p: str) -> str:
    return str(p).replace(os.sep, "/")


def _spec_path(p: str) -> str:
    """Repo-root-relative POSIX when possible (all tooling runs from the repo root) —
    keeps ``NAME=DIR:CAPTURE`` CLI specs free of Windows drive colons."""
    try:
        return _posix(os.path.relpath(p))
    except ValueError:          # different drive
        return _posix(p)


def verify_game_yolo(yolo_dir: str, obb: bool):
    """Integrity-check one game's ``yolo/`` dir; return a problem string or None.

    Guards the two silent-corruption modes that poison a detector split: a
    TRUNCATED build (image without a label file -> ultralytics scores every real
    tile on that frame as a false positive; a concurrent-build race truncated
    1092 OBB labels across 8 games on 2026-07-05, capping val mAP50 at ~0.79)
    and a FORMAT mix (5-field HBB rows in an --obb dataset or vice versa, e.g.
    a --resume over a version built with the other flag). Pure and cheap: counts
    files, then checks the first non-empty row of each label file (a build emits
    one uniform format per file)."""
    imgs = glob.glob(os.path.join(yolo_dir, "images", "*.png"))
    labs = glob.glob(os.path.join(yolo_dir, "labels", "*.txt"))
    if not imgs:
        return "no yolo/images"
    if len(imgs) != len(labs):
        return f"truncated: {len(imgs)} images vs {len(labs)} labels"
    want = 9 if obb else 5
    for lp in labs:
        with open(lp, encoding="utf-8") as fh:
            for ln in fh:
                n = len(ln.split())
                if n:
                    if n != want:
                        return (f"{os.path.basename(lp)}: {n}-field rows "
                                f"(want {want} for {'OBB' if obb else 'HBB'})")
                    break
    return None


def discover_games(sources: list, exclude=()) -> list:
    """Scan source roots for captures -> manifest entries (pure; no side effects).

    Returns [{name, dir, kind, capture, frames_dir}] with repo-relative POSIX paths.
    Raises SystemExit on an empty root or a duplicate game name (e.g. the same source
    root listed twice; distinct roots are source-root-qualified and never collide).
    Captures whose frames.jsonl exists but holds zero entries (a run aborted before
    any frame was recorded) are dropped with a note — they can never be annotated or
    built, and stage 3 would reject their empty yolo dir as a poisoned split.

    ``exclude`` names games to keep OUT of the manifest for good — the capture stays on
    disk with its GT, but no stage ever sees it, so a later ``--resume`` cannot pull it
    back in. For games that are discoverable and annotatable yet whose PIXELS do not
    show what GT says (see --exclude's help). An unknown name raises rather than being
    ignored: a typo would silently readmit the game it was meant to keep out.
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
                raise SystemExit(f"duplicate game name {name!r} from both {seen[name]} and "
                                 f"{cap} — pass each source root once (names are already "
                                 f"source-root-qualified, so distinct roots never collide)")
            seen[name] = cap
            frames = paths.frames_dir_for(cap)
            idx = os.path.join(frames, "frames.jsonl")
            if os.path.exists(idx) and os.path.getsize(idx) == 0:
                print(f"  skipping frameless capture (aborted run): {_posix(cap)}")
                continue
            games.append({"name": name, "dir": name,
                          "kind": "manual" if cap in manual else "ai",
                          "capture": _posix(cap), "frames_dir": _posix(frames)})
    if exclude:
        unknown = [x for x in exclude if x not in seen]
        if unknown:
            raise SystemExit(f"--exclude {unknown} not among discovered games — a name that "
                             f"matches nothing would silently readmit the game it was meant "
                             f"to keep out. Discovered: {sorted(seen)}")
        for x in exclude:
            print(f"  excluding game (--exclude): {x}")
        games = [g for g in games if g["name"] not in set(exclude)]
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


def resolve_vals(val_arg, names, default: str = DEFAULT_VAL) -> list:
    """Fold the repeatable ``--val`` into a validated list of held-out game names.

    ``None``/empty falls back to ``[default]`` when that game was discovered (the
    historical single-game convention); otherwise at least one must be given. Every
    name must be among the discovered ``names`` — else the split would silently hold
    out nothing for it. Order is preserved so the manifest reads left-to-right."""
    vals = list(val_arg) if val_arg else ([default] if default in names else [])
    if not vals:
        raise SystemExit(f"--val not given and default {default!r} not among discovered "
                         f"games {names} — pass --val <NAME>")
    bad = [v for v in vals if v not in names]
    if bad:
        raise SystemExit(f"--val {bad} not among discovered games {names} — pass --val <NAME>")
    return vals


def write_manifest(ds_dir: str, games: list, vals: list, formats=("hbb",)) -> str:
    path = os.path.join(ds_dir, "games.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"val": vals, "formats": list(formats), "games": games},
                  f, ensure_ascii=False, indent=1)
    return path


def dataset_root(name: str) -> str:
    """'v2' -> datasets/v2; anything with a path separator is used as-is."""
    if "/" in name or os.sep in name:
        return name
    return os.path.join("datasets", name)


# ---- runner (same shape as the deprecated rebuild_datasets.py) ---------------

def _remove_dir_link(link: str) -> None:
    """Remove ``link`` if it exists, NEVER recursing through a symlink/junction into its
    target (that would delete the shared HBB frames the OBB build points at). A real
    directory is rmtree'd; a symlink/junction is unlinked in place."""
    if not os.path.lexists(link):
        return
    is_junction = os.name == "nt" and getattr(os.path, "isjunction", lambda p: False)(link)
    if os.path.islink(link):
        # POSIX: unlink handles a symlink-to-dir; Windows: a dir symlink needs rmdir.
        os.rmdir(link) if (os.name == "nt" and os.path.isdir(link)) else os.unlink(link)
    elif is_junction:
        os.rmdir(link)                       # drops the reparse point, keeps the target
    elif os.path.isdir(link):
        shutil.rmtree(link)                  # a real directory
    else:
        os.remove(link)


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

    def symlink(self, target: str, link: str) -> None:
        """Point dir `link` -> `target`, replacing any stale entry, so an OBB per-game build
        shares the HBB frames dir instead of re-encoding the (byte-identical) images.

        Prefers a RELATIVE symlink (tar-and-go portable). On Windows without the symlink
        privilege (no Developer Mode / not elevated -> WinError 1314) it falls back to a
        directory JUNCTION, which needs no privilege but stores an ABSOLUTE target, so the
        built version is host-local (rebuild rather than relocate it)."""
        rel = os.path.relpath(target, os.path.dirname(link))
        if not self.execute:
            print(f"  ln -sfn {rel} {link}")
            return
        os.makedirs(os.path.dirname(link), exist_ok=True)
        _remove_dir_link(link)
        try:
            os.symlink(rel, link, target_is_directory=True)
            print(f"  ln -sfn {rel} {link}")
        except OSError:
            if os.name != "nt":
                raise
            import _winapi                    # Windows-only; junction = privilege-free symlink
            _winapi.CreateJunction(os.path.abspath(target), link)
            print(f"  mklink /J {link} {os.path.abspath(target)}   (no symlink privilege)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="dataset version name -> datasets/<name>/ "
                                 "(a path with '/' is used as-is)")
    ap.add_argument("--sources", nargs="+", default=[paths.RAW_AI_SESSION],
                    help=f"capture roots to scan (default: {paths.RAW_AI_SESSION}). "
                         f"Add e.g. {paths.RAW_MANUAL} to include the legacy manual sessions.")
    ap.add_argument("--val", action="append", default=None,
                    help=f"held-out whole game for the detector split (default {DEFAULT_VAL} "
                         f"when present among the discovered games; else you must pass one). "
                         f"Repeatable — pass --val twice to hold out two whole games at once.")
    ap.add_argument("--exclude", action="append", default=None, metavar="NAME",
                    help="keep a discovered game OUT of this dataset for good (repeatable). "
                         "The capture stays on disk; it just never enters the manifest, so a "
                         "later --resume cannot pull it back in. For captures whose PIXELS "
                         "disagree with GT even though the GT itself is fine — e.g. a game the "
                         "client played DISCONNECTED (a modal covers the table: river boxes "
                         "land on nothing and are dropped as unreliable, while meld boxes land "
                         "on the wall, pass the fill gate, and ship as phantom labels). An "
                         "unknown NAME is an error, not a no-op.")
    ap.add_argument("--stage", choices=["annotate", "dataset", "detector", "all"], default="all")
    ap.add_argument("--resume", action="store_true",
                    help="continue into an existing datasets/<name>: skip games that already "
                         "have annotations / a built game dir (detector split is redone).")
    ap.add_argument("--force", action="store_true",
                    help="delete datasets/<name> first and rebuild from scratch.")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, touch nothing.")
    ap.add_argument("-j", "--parallel", type=int, default=None,
                    help="degree of parallelism shared by BOTH heavy stages (stage-1 annotate "
                         "workers and stage-2 per-game build jobs — each is one OS process per "
                         "game, RAM-bound, and the stages run sequentially). Sets the default for "
                         "--workers and --jobs; pass either to override that one stage.")
    ap.add_argument("--workers", type=int, default=None,
                    help="override stage-1 annotate parallel workers only (default: follows "
                         "--parallel, else annotate_ai_session's own cap of 4).")
    ap.add_argument("--jobs", type=int, default=None,
                    help=f"override stage-2 parallel build_dataset processes only (RAM-bound; "
                         f"default: follows --parallel, else min(8, cpu//2)={DEFAULT_JOBS}).")
    ap.add_argument("--hbb", action="store_true",
                    help="emit HBB (axis-aligned) YOLO labels (the default when no format flag "
                         "is given). Pass together with --obb to build BOTH splits in one version.")
    ap.add_argument("--obb", action="store_true",
                    help="emit OBB (8-point) YOLO labels. Alone -> OBB-only (historical layout); "
                         "with --hbb -> both detector/ + detector_obb/ in one version, the OBB "
                         "build reusing the HBB frames via symlink (no re-encode).")
    ap.add_argument("--backs", action="store_true",
                    help="EXPERIMENTAL (default OFF): also label opponent hand-row tile backs "
                         "(手摸切 groundwork — annotate/backs.py). Threads --backs to the annotate "
                         "stage and the per-game builds; frames where an opponent is mid-draw "
                         "(backs_holding) are dropped from YOLO for label consistency. Keep this "
                         "OUT of the mainline v1/v2 versions — use a scratch version name.")
    args = ap.parse_args()

    workers, jobs = resolve_parallelism(args.parallel, args.workers, args.jobs)
    py = sys.executable
    ds = dataset_root(args.name)
    ann = os.path.join(ds, "annotations")
    formats = resolve_formats(args.hbb, args.obb)
    dual = len(formats) > 1
    r = Runner(not args.dry_run)

    if os.path.exists(ds) and not (args.resume or args.force or args.dry_run):
        raise SystemExit(f"{ds} already exists — pass --resume to continue it, "
                         f"or --force to wipe and rebuild")
    if args.force:
        r.rm(ds)

    games = apply_existing_dirs(discover_games(args.sources, args.exclude or ()), ds)
    names = [g["name"] for g in games]
    vals = resolve_vals(args.val, names)
    print(f"{'DRY RUN' if args.dry_run else 'BUILD'} {ds}  "
          f"({len(games)} game(s), val={vals}, formats={formats})")
    print(f"sources: {args.sources}\ngames: {names}\n")

    do = lambda s: args.stage in ("all", s)

    # ---- stage 1: annotate (AI games only) ---------------------------------
    if do("annotate"):
        todo = [g for g in games if g["kind"] == "ai"
                and not (args.resume and os.path.exists(os.path.join(ann, g["name"] + ".jsonl")))]
        print(f"[1/3] annotate {len(todo)} game(s) -> {ann}")
        wk = ["--workers", str(workers)] if workers else []
        bk = ["--backs"] if args.backs else []
        batch = [g["capture"] for g in todo]
        if batch:
            r.run([py, "scripts/annotate/annotate_ai_session.py",
                   "--captures", *batch, "--out", ann, *wk, *bk])
        print()

    # ---- stage 2: per-game crops + YOLO ------------------------------------
    # One pass PER FORMAT, HBB first (formats is ordered): the OBB pass reuses the HBB
    # frames, so they must already be on disk — run_parallel is a barrier between passes.
    if do("dataset"):
        for fmt in formats:
            spec = FORMATS[fmt]
            cmds = []
            for g in games:
                yolo = game_yolo_dir(ds, g["dir"], fmt, formats)
                out = os.path.dirname(yolo)
                if args.resume and os.path.isdir(out):
                    prob = verify_game_yolo(yolo, spec["obb"])
                    if prob is None:
                        continue        # complete + right format -> safe to skip
                    print(f"  resume: REBUILD {g['name']} [{fmt}] ({prob})")
                cmd = [py, "scripts/train/build_dataset.py", g["capture"], g["frames_dir"],
                       "--out", out, "--drop-violations"]
                if args.backs:
                    cmd += ["--backs"]
                if g["kind"] == "ai":
                    cmd += ["--from-annotations", ann]
                if spec["obb"]:
                    cmd += ["--obb"]
                    if dual:    # reuse the byte-identical HBB frames instead of re-encoding
                        hbb_imgs = os.path.join(ds, g["dir"], "yolo", "images")
                        r.symlink(hbb_imgs, os.path.join(yolo, "images"))
                        cmd += ["--no-crops", "--reuse-images", hbb_imgs]
                cmds.append(cmd)
            print(f"[2/3] build_dataset [{fmt}] {len(cmds)} game(s) -> "
                  f"{os.path.dirname(game_yolo_dir(ds, '<game>', fmt, formats))}   (jobs={jobs})")
            r.run_parallel(cmds, jobs)
        print()

    # ---- stage 3: detector split(s) (always rebuilt over ALL games) -----------
    if do("detector"):
        val_args = []
        for v in vals:
            val_args += ["--val", f"{v}:*"]
        for fmt in formats:
            spec = FORMATS[fmt]
            det = os.path.join(ds, spec["split"])
            if not args.dry_run:    # refuse to assemble a poisoned split (see verify_game_yolo)
                bad = [f"{g['name']}: {p}" for g in games
                       if (p := verify_game_yolo(game_yolo_dir(ds, g["dir"], fmt, formats), spec["obb"]))]
                if bad:
                    raise SystemExit(f"[3/3] REFUSING to assemble {spec['split']} — broken game "
                                     "builds:\n  " + "\n  ".join(bad)
                                     + "\n  rebuild them with --resume (bad games are re-run).")
            print(f"[3/3] build_detector_dataset [{fmt}] -> {det}")
            data_args = []
            for g in games:
                data_args += ["--data",
                              f"{g['name']}={_spec_path(game_yolo_dir(ds, g['dir'], fmt, formats))}:{g['capture']}"]
            r.run([py, "scripts/train/build_detector_dataset.py", *data_args,
                   *val_args, "--out", det])
        print()

    if not args.dry_run:
        print("manifest ->", write_manifest(ds, games, vals, formats))

    val_flags = " ".join(f"--val {v}:*" for v in vals)
    splits = ", ".join(FORMATS[f]["split"] + "/" for f in formats)
    print("=" * 70)
    print(f"BUILT datasets/{args.name}/  (val={vals}, splits -> {splits})")
    print("NOT run here (GPU / deliberate). Train on one or MORE dataset versions:\n")
    print("  # classifier (crops — the HBB per-game dirs hold the crops):")
    print(f"  {py} scripts/train/train_classifier.py --dataset {_posix(ds)} \\")
    print(f"      {val_flags} --epochs 20 --out majsoul_eye/recognize/tile_classifier.pt")
    for f in formats:
        det_yaml = _posix(os.path.join(ds, FORMATS[f]["split"], "data.yaml"))
        print(f"  # detector ({f.upper()}): launch_detector.sh picks seed/out/run-dir per mode")
        print(f"  bash scripts/train/launch_detector.sh {f} --dataset {args.name} --gpus 4")
        print(f"      # or raw: {py} scripts/train/train_detector.py --data {det_yaml}")
    print("  # multi-version: merge several manifests into one combined split, then train it:")
    print(f"  #   train_classifier.py --dataset datasets/v0 --dataset {_posix(ds)} {val_flags}")
    print(f"  #   build_detector_dataset.py --dataset datasets/v0 --dataset {_posix(ds)} \\")
    print(f"  #       {val_flags} --out datasets/combined/{FORMATS[formats[0]]['split']}")
    print(f"  #   launch_detector.sh {formats[0]} --dataset datasets/combined")


if __name__ == "__main__":
    main()
