"""One-shot migration of ``captures/`` into a clean role-based layout.

Moves the loose top-level entries into::

    captures/raw/ai_session/          (MahjongCopilot raw, moved wholesale)
    captures/raw/manual/              (record_gt sessions: session*.jsonl/dir/.log)
    captures/intermediate/gt/         (ai_run_*.jsonl + their index dirs)
    captures/intermediate/derived/    (session5_16x9, *_fixed)
    captures/legacy/                  (ai_g*/ai_r1 byte-identical duplicates, archived)

and rewrites every ``frames.jsonl`` (+ ``.letterboxed`` backups) so ``file`` fields
become RELATIVE (index-relative ``frames/000009.png`` for self-contained dirs,
captures-relative ``raw/ai_session/...`` for the hollow gt/ indexes). This is the
root-cause fix for the absolute-path fragility: after this, moving frame dirs never
breaks an index again (resolution goes through majsoul_eye.paths.resolve_frame_path).

All moves are within the same volume, so ``shutil.move`` is an instant rename, never
a multi-GB copy — the script never iterates PNGs. Idempotent + resumable (writes
``MIGRATION_MANIFEST.json`` with the full old->new map before moving; skips already-
moved entries; backs each index up to ``frames.jsonl.premigrate`` before rewriting).

Dry-run by default. Run (conda `auto` env, repo root, PYTHONPATH=.):
  $PY scripts/migrate_captures_layout.py                 # preview the plan
  # BACK UP FIRST:  robocopy captures captures_backup /E
  $PY scripts/migrate_captures_layout.py --apply
  $PY scripts/migrate_captures_layout.py --apply --strict   # fail if any index unresolved
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from majsoul_eye import paths

CAP = paths.CAPTURES              # "captures" (relative; cwd == repo root)
CAP_ABS = paths.CAPTURES_ABS
MANIFEST = os.path.join(CAP, "MIGRATION_MANIFEST.json")

LEGACY_STEMS = {"ai_g1", "ai_g2", "ai_g3", "ai_r1"}
NEW_DIRS = ["raw", "raw/manual", "intermediate", "intermediate/gt",
            "intermediate/derived", "legacy"]
SKIP_TOPLEVEL = {"raw", "intermediate", "legacy", "MIGRATION_MANIFEST.json"}


def _stem(name: str) -> str:
    if name.endswith(".jsonl.log"):
        return name[:-len(".jsonl.log")]
    if name.endswith(".jsonl"):
        return name[:-len(".jsonl")]
    return name


def classify(name: str):
    """Return the captures-relative destination DIR for a top-level entry, or None
    to leave it in place. Order matters: _fixed / session5_16x9 are checked before
    the generic ai_run_ / session prefixes."""
    if name in SKIP_TOPLEVEL:
        return None
    st = _stem(name)
    if st == "ai_session":        return "raw"
    if st in LEGACY_STEMS:        return "legacy"
    if st.endswith("_fixed"):     return "intermediate/derived"
    if st == "session5_16x9":     return "intermediate/derived"
    if st.startswith("ai_run_"):  return "intermediate/gt"
    if st.startswith("session"):  return "raw/manual"
    return None


def build_plan():
    """(moves, topmove, unknown) from the CURRENT top-level of captures/."""
    moves, topmove, unknown = [], {}, []
    for name in sorted(os.listdir(CAP_ABS)):
        dest = classify(name)
        if dest is None:
            if name not in SKIP_TOPLEVEL and not name.startswith("."):
                unknown.append(name)
            continue
        new_rel = f"{dest}/{name}"
        moves.append((name, dest, new_rel))
        topmove[name] = new_rel
    return moves, topmove, unknown


def _captures_rel_old(path: str):
    """Strip everything up to and including the last 'captures' segment."""
    parts = path.replace("\\", "/").split("/")
    if "captures" in parts:
        i = len(parts) - 1 - parts[::-1].index("captures")
        return "/".join(parts[i + 1:])
    return None


def remap(file_field: str, index_dir: str, topmove: dict):
    """Absolute on-disk path of a frame after migration, or None if unresolved."""
    # already-relative and resolvable (idempotent re-run)
    if not os.path.isabs(file_field):
        cur = paths.resolve_frame_path(file_field, index_dir)
        if os.path.exists(cur):
            return os.path.abspath(cur)
    rel_old = _captures_rel_old(file_field)
    if rel_old:
        top, _, rest = rel_old.partition("/")
        new_top = topmove.get(top, top)
        cand = os.path.join(CAP_ABS, *new_top.split("/"), *(rest.split("/") if rest else []))
        if os.path.exists(cand):
            return os.path.abspath(cand)
    cur = paths.resolve_frame_path(file_field, index_dir)
    return os.path.abspath(cur) if os.path.exists(cur) else None


def _rel_form(new_abs: str, index_dir: str) -> str:
    idx_abs = os.path.abspath(index_dir)
    na = os.path.abspath(new_abs)
    if na.startswith(idx_abs + os.sep):
        return os.path.relpath(na, idx_abs).replace(os.sep, "/")   # index-relative
    return paths.rel_to_captures(na)                                # captures-relative


def target_indexes():
    found = []
    for root in (paths.GT, paths.DERIVED, paths.RAW_MANUAL):
        if not os.path.isdir(root):
            continue
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn == "frames.jsonl" or fn == "frames.jsonl.letterboxed":
                    found.append(os.path.join(dp, fn))
    return found


def rewrite_index(idx_path: str, topmove: dict, apply: bool):
    index_dir = os.path.dirname(idx_path)
    with open(idx_path, encoding="utf-8") as f:
        raw = f.read().splitlines()
    out, n, unresolved = [], 0, []
    for line in raw:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("file"):
            na = remap(d["file"], index_dir, topmove)
            if na is None:
                unresolved.append(d["file"])
            else:
                d = dict(d)
                d["file"] = _rel_form(na, index_dir)
                n += 1
        out.append(d)
    if apply:
        pre = idx_path + ".premigrate"
        if not os.path.exists(pre):
            shutil.copy(idx_path, pre)
        with open(idx_path, "w", encoding="utf-8") as f:
            for d in out:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return n, unresolved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="perform the migration (default: dry-run)")
    ap.add_argument("--strict", action="store_true", help="exit nonzero if any index record is unresolved")
    args = ap.parse_args()

    if not (os.path.isdir("majsoul_eye") and os.path.isdir(CAP)):
        raise SystemExit("run from the repo root (need ./majsoul_eye and ./captures)")

    moves, computed_topmove, unknown = build_plan()
    topmove = json.load(open(MANIFEST, encoding="utf-8")) if os.path.exists(MANIFEST) else computed_topmove

    print(f"{'APPLY' if args.apply else 'DRY-RUN'}: {len(moves)} top-level entries to move")
    for name, dest, new_rel in moves:
        print(f"  {name}  ->  {new_rel}")
    if unknown:
        print(f"  left in place (unrecognized): {unknown}")

    if not args.apply:
        # Preview index rewrite count against CURRENT locations (best-effort).
        idxs = target_indexes()
        print(f"would rewrite ~{len(idxs)} index files under gt/ derived/ raw/manual/ (post-move)")
        print("re-run with --apply (after: robocopy captures captures_backup /E)")
        return

    for d in NEW_DIRS:
        os.makedirs(os.path.join(CAP, d), exist_ok=True)
    if not os.path.exists(MANIFEST):
        with open(MANIFEST, "w", encoding="utf-8") as f:
            json.dump(computed_topmove, f, ensure_ascii=False, indent=1)

    moved = skipped = 0
    for name, dest, new_rel in moves:
        src = os.path.join(CAP, name)
        dst = os.path.join(CAP, dest, name)
        if os.path.exists(dst):
            skipped += 1
            continue
        if os.path.exists(src):
            shutil.move(src, dst)
            moved += 1
    print(f"moved {moved}, skipped {skipped} (already at destination)")

    idxs = target_indexes()
    total_rw, all_unresolved = 0, []
    for idx in idxs:
        n, unres = rewrite_index(idx, topmove, apply=True)
        total_rw += n
        all_unresolved += [(idx, u) for u in unres]
    print(f"rewrote {total_rw} file entries across {len(idxs)} indexes; "
          f"unresolved={len(all_unresolved)}")
    for idx, u in all_unresolved[:20]:
        print(f"  UNRESOLVED {idx}: {u}")
    if args.strict and all_unresolved:
        raise SystemExit(f"strict: {len(all_unresolved)} unresolved index records")
    print("done. captures/ top-level is now: raw/ intermediate/ legacy/ + MIGRATION_MANIFEST.json")


if __name__ == "__main__":
    main()
