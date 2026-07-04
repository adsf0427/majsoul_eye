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
from majsoul_eye.capture.schema import write_records


def find_b64_game_dirs(ai_session: str) -> list:
    """Dirs holding a legacy b64 wire (frames.jsonl with a 'b64' field)."""
    out = []
    for fj in glob.glob(os.path.join(ai_session, "**", "frames.jsonl"), recursive=True):
        gd = os.path.dirname(fj)
        # skip a NEW-layout screenshot index (no 'b64'): peek the first non-empty line
        if _looks_like_wire(fj):
            out.append(gd)
    return sorted(out)


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
    """True if this dir already has the new layout (liqi.jsonl + sibling GTRecord)."""
    t = plan_targets(game_dir)
    return os.path.exists(t["wire_dest"]) and os.path.exists(t["gt_path"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="Actually migrate (default: dry run).")
    ap.add_argument("--mjcopilot", default="../MahjongCopilot")
    ap.add_argument("--ai-session", default=paths.RAW_AI_SESSION)
    args = ap.parse_args()

    ai_session = os.path.abspath(args.ai_session)
    game_dirs = find_b64_game_dirs(ai_session)
    todo = [gd for gd in game_dirs if not is_migrated(gd)]
    print(f"{'APPLY' if args.apply else 'DRY RUN'} — {len(game_dirs)} b64 game(s), "
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
        print(f"  {t['name']}: {gd}")
        print(f"    -> {t['gt_path']}")
        print(f"    -> rename frames.jsonl -> {t['wire_dest']}")
        print(f"    -> write screenshot index {t['index_path']}")
        if not args.apply:
            continue
        # 1) re-derive GT from the legacy wire (still named frames.jsonl here)
        records, frame_index = convert_game(gd, liqimod, GameState, Bot, GameMode,
                                            wire_name="frames.jsonl")
        # 2) rename the wire BEFORE overwriting frames.jsonl with the index
        os.rename(os.path.join(gd, "frames.jsonl"), t["wire_dest"])
        # 3) write GTRecord + index-relative screenshot index
        write_records(t["gt_path"], records)
        with open(t["index_path"], "w", encoding="utf-8") as f:
            for fi in frame_index:
                seq = fi["seq"]
                f.write(json.dumps({"seq": seq, "file": f"frames/{seq:06d}.png",
                                    "status": "ok"}) + "\n")

    if not args.apply:
        print("\n(dry run — nothing changed; pass --apply to migrate)")


if __name__ == "__main__":
    main()
