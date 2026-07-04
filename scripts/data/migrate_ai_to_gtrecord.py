"""One-time migration: legacy b64 AI captures -> unified GTRecord layout (DEV-ONLY).

Old (b64):  run_N/gameM/frames.jsonl (wire) + run_N/gameM/frames/*.png
New:        run_N/gameM.jsonl        (GTRecord: raw_liqi + mjai)
            run_N/gameM/liqi.jsonl    (raw wire, renamed from frames.jsonl)
            run_N/gameM/frames.jsonl  (screenshot index {seq,file,status})
            run_N/gameM/frames/*.png  (unchanged)

Single-game legacy run (run_1) keeps its shape: run_1.jsonl + run_1/{liqi.jsonl,
frames.jsonl, frames/*.png}.

Idempotent + dry-run by default. Re-derives GT from the wire via the SHARED
convert_game (same code the live capture's derivation is proven equal to), so the
output matches the retired captures/intermediate/gt/*.jsonl byte-for-byte.

Run (conda `auto` env, repo root, PYTHONPATH=.):
    PYTHONPATH=. $PY scripts/data/migrate_ai_to_gtrecord.py            # dry run
    PYTHONPATH=. $PY scripts/data/migrate_ai_to_gtrecord.py --apply    # do it
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from majsoul_eye import paths


def find_game_dirs(ai_session: str) -> list:
    """Game dirs holding a liqi wire — either a legacy b64 ``frames.jsonl`` OR an
    already-renamed ``liqi.jsonl`` (so a partially-migrated dir from an interrupted
    run is re-picked and finished on retry)."""
    dirs = set()
    for fj in glob.glob(os.path.join(ai_session, "**", "frames.jsonl"), recursive=True):
        if _looks_like_wire(fj):                       # legacy b64 wire (not a screenshot index)
            dirs.add(os.path.dirname(fj))
    for lj in glob.glob(os.path.join(ai_session, "**", "liqi.jsonl"), recursive=True):
        dirs.add(os.path.dirname(lj))                  # wire already renamed (done or partial)
    return sorted(dirs)


def _looks_like_wire(frames_jsonl: str) -> bool:
    try:
        with open(frames_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                return "b64" in json.loads(line)
    except Exception:
        return False
    return False


def plan_targets(game_dir: str) -> dict:
    """Compute new-layout targets for a b64 game dir (pure; no I/O)."""
    gd = os.path.abspath(game_dir).replace("\\", "/")
    parts = gd.split("/")
    game = parts[-1]
    parent = parts[-2]
    if game.startswith("game"):
        name = f"ai_{parent}_{game}"                 # run_N/gameM -> ai_run_N_gameM
        gt_path = os.path.join(os.path.dirname(game_dir), f"{game}.jsonl")
    else:
        name = f"ai_{game}"                          # run_1 (single-game) -> ai_run_1
        gt_path = os.path.join(os.path.dirname(game_dir), f"{game}.jsonl")
    return {
        "name": name,
        "gt_path": gt_path,
        "wire_dest": os.path.join(game_dir, "liqi.jsonl"),
        "index_path": os.path.join(game_dir, "frames.jsonl"),
    }


def is_migrated(game_dir: str) -> bool:
    """True only if the new layout is COMPLETE: GTRecord + renamed wire + index all
    present (so an interrupted run that produced only some of them is redone)."""
    t = plan_targets(game_dir)
    return (os.path.exists(t["gt_path"]) and os.path.exists(t["wire_dest"])
            and os.path.exists(t["index_path"]))


def _atomic_write(path: str, text: str) -> None:
    """Write via a temp file + os.replace so a crash can't leave a truncated file
    that later looks 'present' to is_migrated."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _render_records(records) -> str:
    """Serialize GTRecords exactly like capture.schema.write_records (schema header
    + one record per line) so migrated files stay byte-identical to the golden."""
    from majsoul_eye.capture.schema import SCHEMA_VERSION
    out = [json.dumps({"_schema": SCHEMA_VERSION})]
    out += [r.to_json_line() for r in records]
    return "".join(l + "\n" for l in out)


def migrate_one(game_dir: str, records, frame_index) -> dict:
    """Write one game's new-layout outputs in crash-safe order and return its
    plan_targets. Resumable: renames the wire ONLY if it is still the legacy
    frames.jsonl (a resumed run reads from the already-renamed liqi.jsonl, so the
    rename is skipped). Atomic writes so an interrupted write can't leave a
    truncated file that later looks 'done'."""
    t = plan_targets(game_dir)
    wire_is_legacy = not os.path.exists(t["wire_dest"])   # liqi.jsonl absent => still legacy
    # 1) GTRecord to its own path first (atomic).
    _atomic_write(t["gt_path"], _render_records(records))
    # 2) rename wire only if still legacy (frames.jsonl -> liqi.jsonl).
    if wire_is_legacy:
        os.rename(os.path.join(game_dir, "frames.jsonl"), t["wire_dest"])
    # 3) index-relative screenshot index last (frames.jsonl now free), atomic.
    lines = [json.dumps({"seq": fi["seq"], "file": f"frames/{fi['seq']:06d}.png",
                         "status": "ok"}) for fi in frame_index]
    _atomic_write(t["index_path"], "".join(l + "\n" for l in lines))
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="Actually migrate (default: dry run).")
    ap.add_argument("--mjcopilot", default="../MahjongCopilot")
    ap.add_argument("--ai-session", default=paths.RAW_AI_SESSION)
    args = ap.parse_args()

    ai_session = os.path.abspath(args.ai_session)
    game_dirs = find_game_dirs(ai_session)
    todo = [gd for gd in game_dirs if not is_migrated(gd)]
    print(f"{'APPLY' if args.apply else 'DRY RUN'} — {len(game_dirs)} game(s), "
          f"{len(todo)} to migrate, {len(game_dirs) - len(todo)} already done")

    # Import MahjongCopilot + convert once (chdir handled by _import_mjcopilot).
    from scripts.data.convert_mjcopilot import _import_mjcopilot, convert_game
    liqimod = GameState = Bot = GameMode = None
    if todo:
        mc = os.path.abspath(args.mjcopilot)
        liqimod, GameState, Bot, GameMode = _import_mjcopilot(mc)
        os.chdir(mc)

    for gd in todo:
        t = plan_targets(gd)
        # wire source: the renamed liqi.jsonl if a prior interrupted run already moved it,
        # else the legacy frames.jsonl.
        wire_name = "liqi.jsonl" if os.path.exists(t["wire_dest"]) else "frames.jsonl"
        print(f"  {t['name']}: {gd}  (wire={wire_name})")
        print(f"    -> {t['gt_path']}")
        if wire_name == "frames.jsonl":
            print(f"    -> rename frames.jsonl -> {t['wire_dest']}")
        print(f"    -> write screenshot index {t['index_path']}")
        if not args.apply:
            continue
        records, frame_index = convert_game(gd, liqimod, GameState, Bot, GameMode,
                                            wire_name=wire_name)
        migrate_one(gd, records, frame_index)

    if not args.apply:
        print("\n(dry run — nothing changed; pass --apply to migrate)")


if __name__ == "__main__":
    main()
