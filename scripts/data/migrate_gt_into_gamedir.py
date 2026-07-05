"""One-shot layout migration: move sibling-shape AI GT jsonls INTO their frames dirs.

Old (sibling):  captures/raw/<root>/run_N/gameM.jsonl   next to  run_N/gameM/
New (nested):   captures/raw/<root>/run_N/gameM/gameM.jsonl      (self-contained game dir)
                (single-game runs likewise: run_N.jsonl -> run_N/run_N.jsonl)

Also rewrites every ``datasets/*/games.json`` manifest whose ``capture`` entries
still point at a moved sibling path (training reads the capture jsonl through the
manifest, so it must follow). ``frames_dir`` entries are unchanged — the frames
dir itself does not move. Manual sessions (``raw/manual/session*.jsonl``) keep the
sibling shape and are never touched.

Same-volume ``shutil.move`` = instant rename. Idempotent: already-nested files and
already-rewritten manifests are skipped; a second run is a no-op. Dry-run by
default. Run (conda ``auto`` env, repo root, PYTHONPATH=.):

  python scripts/data/migrate_gt_into_gamedir.py            # preview
  python scripts/data/migrate_gt_into_gamedir.py --apply
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from majsoul_eye import paths


def plan_moves(raw_root: str) -> list:
    """[(src, dst)] for every sibling-shape AI GT jsonl under raw_root that has a
    frames-dir twin to nest into. Scans every root (ai_session, ai_session2, temp,
    ...) but the run_*-shaped globs never match manual/session*.jsonl."""
    moves = []
    for pat in (os.path.join("*", "run_*", "game*.jsonl"),      # multi-game runs
                os.path.join("*", "run_*.jsonl")):              # single-game runs
        for src in sorted(glob.glob(os.path.join(raw_root, pat))):
            stem = os.path.splitext(src)[0]
            if not os.path.isdir(stem):
                continue                    # no frames-dir twin -> nothing to nest into
            dst = os.path.join(stem, os.path.basename(src))
            if os.path.exists(dst):
                continue                    # nested twin already there (mid-state) -> skip
            moves.append((src, dst))
    return moves


def apply_moves(moves: list) -> int:
    """os.rename ONLY (same-volume atomic; never shutil.move — its copy-fallback
    would leave a stale duplicate when the source is a LIVE capture held open by
    a running autoplay session). A locked/failed file is skipped and reported;
    re-run after the session ends."""
    n = 0
    for src, dst in moves:
        try:
            os.rename(src, dst)
            n += 1
        except OSError as e:
            print(f"  SKIP (in use / failed): {src}: {e}")
    return n


def rewrite_manifests(datasets_dir: str, apply: bool, pending: set = frozenset()) -> int:
    """Point ``capture`` entries of every datasets/*/games.json at the nested file
    when the sibling path is gone (or is in ``pending`` — dry-run preview of paths
    that WILL move). Returns the number of entries (to be) rewritten."""
    total = 0
    for mpath in sorted(glob.glob(os.path.join(datasets_dir, "*", "games.json"))):
        with open(mpath, encoding="utf-8") as f:
            m = json.load(f)
        changed = 0
        for g in m.get("games", []):
            cap = g.get("capture", "")
            gone = cap and (not os.path.exists(cap) or os.path.normpath(cap) in pending)
            if not gone:
                continue
            nested = os.path.join(os.path.splitext(cap)[0], os.path.basename(cap))
            if os.path.exists(nested) or os.path.normpath(cap) in pending:
                g["capture"] = nested.replace(os.sep, "/")
                changed += 1
            else:
                print(f"  WARN {mpath}: {cap} missing and no nested twin — left as-is")
        if changed and apply:
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump(m, f, ensure_ascii=False, indent=1)
        if changed:
            print(f"  {mpath}: {changed} capture path(s) {'rewritten' if apply else 'to rewrite'}")
        total += changed
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default=paths.RAW,
                    help=f"captures raw root to scan (default: {paths.RAW})")
    ap.add_argument("--datasets-dir", default="datasets")
    ap.add_argument("--apply", action="store_true", help="perform the moves (default: dry-run)")
    args = ap.parse_args()

    moves = plan_moves(args.raw_root)
    print(f"{'APPLY' if args.apply else 'DRY-RUN'}: {len(moves)} GT jsonl(s) to nest")
    for src, dst in moves:
        print(f"  {src}  ->  {dst}")
    if args.apply:
        print(f"moved {apply_moves(moves)}")
        rewrite_manifests(args.datasets_dir, apply=True)
    else:
        n = rewrite_manifests(args.datasets_dir, apply=False,
                              pending={os.path.normpath(s) for s, _ in moves})
        print(f"would rewrite {n} manifest capture path(s); re-run with --apply")


if __name__ == "__main__":
    main()
