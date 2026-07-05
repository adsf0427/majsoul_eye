"""Backfill 换肤 provenance in existing --skins capture metadata.json (one-shot tool, 2026-07-05).

``extract_authgame_skins`` used to read ``players[0].views`` as the hero's table 装扮
(牌背/桌布/场景), but the authGame RES orders ``players`` by account_id — NOT hero-first (seat
order lives in ``seat_list``) — so ``skins.table`` recorded a stranger's own cosmetics (usually
``{}``). MajsoulMax's mod.py rewrites only the HERO's views, and only those render the captured
table, so the applied swap was fine; just the metadata record was wrong.

This tool re-derives the truth from each game's recorded wire (``liqi.jsonl``): frame 1 is the
authGame REQ (carries the hero accountId); the RES with the matching msg_id carries the
post-rewrite per-player views. It rewrites ``skins.table`` / ``skins.characters`` and adds
``skins.hero_account_id``. Games without a ``skins`` key (non ``--skins`` runs) are untouched.
Idempotent; dry-run by default:

  PYTHONPATH=. python scripts/capture/backfill_skin_meta.py            # preview captures/raw/ai_session2
  PYTHONPATH=. python scripts/capture/backfill_skin_meta.py --apply
  PYTHONPATH=. python scripts/capture/backfill_skin_meta.py captures/raw/ai_sessionX --apply

Needs ``_external/autoliqi-asserts/liqi_pb2.py`` (the current protocol) to decode the wire.
"""
from __future__ import annotations

import argparse
import base64
import copy
import glob
import json
import os
import pathlib
import sys

from majsoul_eye import paths
from majsoul_eye.capture import gamemeta

AUTHGAME = ".lq.FastTest.authGame"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_liqi_pb2():
    sys.path.insert(0, str(REPO_ROOT / "_external" / "autoliqi-asserts"))
    import liqi_pb2
    return liqi_pb2


def _player_dict(p):
    """PlayerGameView -> the camelCase dict shape MahjongCopilot's parser emits (what
    extract_authgame_skins expects). Manual build: proto3 accessors always yield defaults,
    so slot 0 / itemId 0 survive (MessageToDict would need always_print... to keep them)."""
    return {"accountId": p.account_id,
            "character": {"charid": p.character.charid, "skin": p.character.skin},
            "views": [{"slot": v.slot, "itemId": v.item_id} for v in p.views]}


def decode_authgame(wire_path):
    """(hero_account_id, authGame RES data dict) from a game's recorded wire, or (None, None).
    Wire frames: 0x02=REQ / 0x03=RES, then 2-byte LE msg_id, then a Wrapper{name, data};
    the RES's name is empty so it is paired to the REQ by msg_id."""
    pb = _load_liqi_pb2()
    hero = req_id = None
    try:
        with open(wire_path, encoding="utf-8") as fh:
            for line in fh:
                raw = base64.b64decode(json.loads(line)["b64"])
                if len(raw) < 4 or raw[0] not in (2, 3):
                    continue
                mid = raw[1] | (raw[2] << 8)
                w = pb.Wrapper()
                try:
                    w.ParseFromString(raw[3:])
                except Exception:
                    continue
                if raw[0] == 2 and w.name == AUTHGAME:
                    req = pb.ReqAuthGame()
                    req.ParseFromString(w.data)
                    hero, req_id = req.account_id, mid
                elif raw[0] == 3 and req_id is not None and mid == req_id:
                    res = pb.ResAuthGame()
                    res.ParseFromString(w.data)
                    return hero, {"players": [_player_dict(p) for p in res.players],
                                  "robots": [_player_dict(p) for p in res.robots],
                                  "seatList": list(res.seat_list)}
    except (OSError, ValueError, KeyError):
        pass
    return None, None


def merge_game_metadata(meta, actual):
    """metadata.json dict with ``actual`` (extract_authgame_skins output) merged over its
    ``skins`` — same merge the live capture does. Pure; inputs untouched."""
    out = copy.deepcopy(meta)
    out["skins"] = {**(out.get("skins") or {}), **actual}
    return out


def process_game(game_dir, decode=decode_authgame, apply=False):
    """Fix one game's metadata.json from its wire. Returns a status string:
    ``no-skins`` / ``no-authgame`` skips, else ``would-fix ...`` / ``fixed ...``."""
    meta_path = os.path.join(game_dir, "metadata.json")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    if "skins" not in meta:
        return "no-skins"
    hero, data = decode(os.path.join(game_dir, "liqi.jsonl"))
    if not hero or not data:
        return "no-authgame"
    actual = gamemeta.extract_authgame_skins(data, hero_account=hero)
    merged = merge_game_metadata(meta, actual)
    if apply:
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, ensure_ascii=False, indent=2)
    return (f"{'fixed' if apply else 'would-fix'}: hero={hero} "
            f"table {meta['skins'].get('table')} -> {actual['table']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=[os.path.join(paths.RAW, "ai_session2")],
                    help="capture session roots holding run_*/game*/ (default: raw/ai_session2)")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run preview)")
    args = ap.parse_args(argv)
    n = 0
    for root in args.roots:
        for meta_path in sorted(glob.glob(os.path.join(root, "run_*", "game*", "metadata.json"))):
            game_dir = os.path.dirname(meta_path)
            print(f"{game_dir}: {process_game(game_dir, apply=args.apply)}", flush=True)
            n += 1
    if not n:
        print("no games found")
    if not args.apply:
        print("dry-run only -- pass --apply to write")


if __name__ == "__main__":
    main()
