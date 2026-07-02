"""Convert a MahjongCopilot semi-automated capture (raw liqi WS wire + 1080p PNGs)
into our GT capture format, so build_dataset/replay can consume it.

A MahjongCopilot capture dir holds:
  frames.jsonl   -- one {seq, ts, b64} per liqi WS message (b64 = liqi protobuf wire)
  frames/NNNNNN.png  -- screenshots, named by the liqi msg `seq` they were grabbed at

We decode the wire and derive MJAI with MahjongCopilot's OWN proven stack:
  liqi.LiqiProto.parse(bytes) -> liqi dict        (wire -> dict)
  game.game_state.GameState.input(dict)           (dict -> MJAI, fed to a stub bot)

⚠️ DEV-ONLY, MahjongCopilot-coupled (GPLv3) — never imported by recognize/. Run in the
`auto` env (protobuf 4.25.3, which MahjongCopilot's liqi_pb2 decodes correctly).

Two gotchas this handles:
  * GameState batches MJAI to the bot at decision points, so we capture per-`input()`
    deltas (one liqi msg = its own seq) for per-action seq tagging — needed because the
    screenshots are grabbed per game action, not per bot turn.
  * GameState MUTATES the AI hand list in place as the game proceeds, so we DEEP-COPY
    each MJAI event at capture (otherwise start_kyoku.tehais gets overwritten with a
    later hand state and the hero hand desyncs — see git history / STATUS).

Outputs (per game NAME; defaults land under captures/intermediate/gt/):
  <out>/NAME.jsonl          -- our GTRecord JSONL (seq, mjai, raw_liqi, ...)
  <out>/NAME/frames.jsonl   -- frame index {seq, file=<captures-relative png>, status}

Usage (defaults: --session captures/raw/ai_session, --out captures/intermediate/gt):
  PYTHONPATH=. $PY scripts/convert_mjcopilot.py \
      --game run_3/game1=ai_run_3_game1 --game run_1=ai_run_1 --mjcopilot ../MahjongCopilot
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import glob
import io
import json
import os
import sys
import types

from majsoul_eye import paths
from majsoul_eye.capture.schema import GTRecord, write_records


def _import_mjcopilot(mc_dir: str):
    """Import MahjongCopilot's LiqiProto + GameState + Bot, stubbing the heavy
    bot.factory (Mortal/libriichi) which we don't need."""
    mc_dir = os.path.abspath(mc_dir)
    if mc_dir not in sys.path:
        sys.path.insert(0, mc_dir)
    # bot/__init__ does `from .factory import *` which loads the libriichi Rust ext;
    # we only need the Bot ABC, so stub the factory submodule.
    sys.modules.setdefault("bot.factory", types.ModuleType("bot.factory"))
    prev = os.getcwd()
    try:
        os.chdir(mc_dir)  # MahjongCopilot uses cwd-relative asset paths (liqi_proto/liqi.json)
        import liqi as liqimod
        from game.game_state import GameState
        from bot import Bot
        from common.utils import GameMode
    finally:
        os.chdir(prev)
    return liqimod, GameState, Bot, GameMode


def convert_game(game_dir: str, liqimod, GameState, Bot, GameMode) -> tuple[list[GTRecord], list[dict]]:
    """Return (gt_records, frame_index) for one MahjongCopilot game dir."""
    captured: list[tuple[int, dict]] = []
    cur = {"seq": None}

    class CapList(list):  # records (seq, deep-copied event) for every append
        def append(self, x):
            captured.append((cur["seq"], copy.deepcopy(x)))
            super().append(x)

        def extend(self, xs):
            for x in xs:
                self.append(x)

    class TracedGameState(GameState):
        def __setattr__(self, k, v):
            object.__setattr__(self, k, CapList() if k == "mjai_pending_input_msgs" else v)

    class StubBot(Bot):
        def __init__(self):
            super().__init__("stub")

        @property
        def supported_modes(self):
            return [GameMode.MJ4P, GameMode.MJ3P]

        @property
        def info_str(self):
            return "stub"

        def _init_bot_impl(self, mode=GameMode.MJ4P):
            pass

        def react(self, m):
            return None

        def react_batch(self, l):
            return None

    gs = TracedGameState(StubBot())
    lp = liqimod.LiqiProto()

    log = os.path.join(game_dir, "frames.jsonl")
    seq_records: dict[int, dict] = {}   # seq -> {ts, method, action_name, raw, mjai[]}
    for line in open(log, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        cur["seq"] = d["seq"]
        before = len(captured)
        try:
            res = lp.parse(base64.b64decode(d["b64"]))
        except Exception:
            continue
        if res is None:
            continue
        with contextlib.redirect_stdout(io.StringIO()):  # silence GameState chatter
            try:
                gs.input(res)
            except Exception:
                pass
        new = [e for (_, e) in captured[before:]]
        if new:
            rec = seq_records.setdefault(d["seq"], {
                "ts": d.get("ts", 0.0),
                "method": res.get("method"),
                "action_name": res["data"].get("name") if res.get("method") == ".lq.ActionPrototype" else None,
                "raw": res,
                "mjai": [],
            })
            rec["mjai"].extend(new)

    seat = getattr(gs, "seat", -1) or 0
    records: list[GTRecord] = []
    for seq in sorted(seq_records):
        r = seq_records[seq]
        records.append(GTRecord(
            seq=seq, ts=r["ts"], flow_id="", seat=seat, last_op_step=0, syncing=False,
            method=r["method"], action_name=r["action_name"], raw_liqi=r["raw"], mjai=r["mjai"],
        ))

    # frame index: every png named by seq -> captures-relative path (points into raw/)
    frame_index = []
    for p in sorted(glob.glob(os.path.join(game_dir, "frames", "*.png"))):
        seq = int(os.path.splitext(os.path.basename(p))[0])
        frame_index.append({"seq": seq, "file": paths.rel_to_captures(p), "status": "ok"})
    return records, frame_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", action="append", required=True,
                    help="REL_DIR=NAME (relative to --session). Repeatable.")
    ap.add_argument("--session", default=paths.RAW_AI_SESSION)
    ap.add_argument("--mjcopilot", default="../MahjongCopilot")
    ap.add_argument("--out", default=paths.GT)
    args = ap.parse_args()

    # Resolve all our paths to absolute BEFORE switching cwd to MahjongCopilot
    # (LiqiProto/GameState load assets via cwd-relative paths).
    session_abs = os.path.abspath(args.session)
    out_abs = os.path.abspath(args.out)
    os.makedirs(out_abs, exist_ok=True)          # create the GT dir if it doesn't exist yet
    mc_abs = os.path.abspath(args.mjcopilot)
    liqimod, GameState, Bot, GameMode = _import_mjcopilot(mc_abs)
    os.chdir(mc_abs)

    for spec in args.game:
        rel, name = spec.split("=", 1)
        game_dir = os.path.join(session_abs, rel)
        if not os.path.exists(os.path.join(game_dir, "frames.jsonl")):
            print(f"SKIP {name}: no frames.jsonl in {game_dir}")
            continue
        records, frame_index = convert_game(game_dir, liqimod, GameState, Bot, GameMode)
        gt_path = os.path.join(out_abs, f"{name}.jsonl")
        write_records(gt_path, records)
        fi_dir = os.path.join(out_abs, name)
        os.makedirs(fi_dir, exist_ok=True)
        with open(os.path.join(fi_dir, "frames.jsonl"), "w", encoding="utf-8") as f:
            for fi in frame_index:
                f.write(json.dumps(fi) + "\n")
        n_mjai = sum(len(r.mjai) for r in records)
        print(f"{name:10s}: {len(records)} GT records ({n_mjai} mjai) seat={records[0].seat if records else '?'}, "
              f"{len(frame_index)} frames -> {gt_path}")


if __name__ == "__main__":
    main()
