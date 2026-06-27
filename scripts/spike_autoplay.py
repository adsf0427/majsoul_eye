"""Spike #5 — minimal AUTOPLAY: one Playwright Chromium that taps the liqi WS,
decodes it, and plays the simplest always-legal game (tsumogiri on your turn,
skip on call-chances). Strength is irrelevant — we only want the game to advance
so (screenshot, GT) pairs can be collected unattended.

Turn detection (proven on real captures): the server only attaches an `operation`
field to YOUR seat. So:
  ActionNewRound / ActionDealTile  with operation  -> your self-turn  -> DISCARD rightmost (tsumogiri)
  ActionDiscardTile / *Gang        with operation  -> a call-chance   -> SKIP (decline)

Threading: Playwright sync objects must be used on the creating thread, so WS
`framereceived` callbacks only ENQUEUE a decision; the main loop drains the queue
and performs the clicks. (MahjongCopilot's action-queue model, single-threaded.)

SAFETY: defaults to --dry-run (logs what it WOULD do, never clicks). Watch a few
of your own turns, confirm "would tsumogiri / would skip" line up, THEN add --live.
Coordinates are MahjongCopilot's 16:9 tables (Positions) — recalibrate if your
client's layout differs.

Usage:
    & $PY scripts/spike_autoplay.py                 # dry-run, logs decisions
    & $PY scripts/spike_autoplay.py --live          # actually click
    & $PY scripts/spike_autoplay.py --live --dump captures/autoplay   # + record frames
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import spike_decode_frames as F  # walk_protobuf, xor_decode, _first_string  # noqa: E402

DEFAULT_URL = "https://game.mahjongsoul.com/"


# --- MahjongCopilot/game/automation.py Positions (16x9 board), reused as data ---
TEHAI_X = [2.23125, 3.021875, 3.8125, 4.603125, 5.39375, 6.184375, 6.975,
           7.765625, 8.55625, 9.346875, 10.1375, 10.928125, 11.71875, 12.509375]
TEHAI_Y = 8.3625
TRUMO_SPACE = 0.246875
BUTTON_SKIP = (10.875, 7.0)   # None/Pass button (bottom-right of the action buttons)
CENTER = (8.0, 4.5)

SELF_TURN = ("ActionNewRound", "ActionDealTile")           # operation here => my discard turn
CALL_CHANCE = ("ActionDiscardTile", "ActionChiPengGang", "ActionAnGangAddGang", "ActionBaBei")


def decode_action(raw: bytes, pb):
    """Return (action_name, decoded_dict) for an ActionPrototype NOTIFY frame, else (None, None)."""
    if not raw or raw[0] != 1:
        return None, None
    blocks = F.walk_protobuf(raw[1:])
    method = (F._first_string(blocks) or b"").decode("ascii", "ignore")
    if method != ".lq.ActionPrototype" or len(blocks) < 2:
        return None, None
    ap = F.walk_protobuf(blocks[1]["data"])
    name_b = F._first_string(ap, lambda d: d[:6] == b"Action")
    body = F._first_string(ap, lambda d: d[:6] != b"Action")
    if not name_b:
        return None, None
    name = name_b.decode("ascii", "ignore")
    try:
        from google.protobuf.json_format import MessageToDict
        inner = getattr(pb, name).FromString(F.xor_decode(body)) if body else getattr(pb, name)()
        try:
            d = MessageToDict(inner, including_default_value_fields=True, preserving_proto_field_name=True)
        except TypeError:
            d = MessageToDict(inner, always_print_fields_with_no_presence=True, preserving_proto_field_name=True)
    except Exception:
        d = {}
    return name, d


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal Majsoul autoplay (tsumogiri + skip) over an in-browser WS tap.")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--seconds", type=float, default=1800.0)
    ap.add_argument("--live", action="store_true", help="Actually click. Default is dry-run (log only).")
    ap.add_argument("--proto-dir", default="../MahjongCopilot")
    ap.add_argument("--dump", default=None, help="Also dump frames.jsonl here (for offline GT).")
    ap.add_argument("--user-data-dir", default=".spike_browser_data")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.proto_dir))
    try:
        from liqi_proto import liqi_pb2 as pb
    except Exception as e:
        sys.exit(f"could not import liqi_pb2 from {args.proto_dir}: {e}\n  (pip install protobuf)")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("playwright not installed: pip install playwright && playwright install chromium")

    scaler = args.width / 16.0
    state = {"my_seat": None, "last_step": -1}
    decisions: deque = deque()
    t0 = time.time()
    dump_fh = None
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        dump_fh = open(os.path.join(args.dump, "frames.jsonl"), "w", encoding="utf-8")

    mode = "LIVE" if args.live else "DRY-RUN"
    print(f"[{mode}] autoplay: tsumogiri on your turn, skip call-chances. Watching {args.seconds:.0f}s.\n")

    def on_decode(name: str, d: dict) -> None:
        if name == "ActionMJStart":
            state["my_seat"] = None
            return
        op = d.get("operation") or {}
        oplist = op.get("operation_list") or op.get("operationList") or []
        if not oplist:
            return                                   # not my move
        seat = op.get("seat")
        if seat is not None:
            state["my_seat"] = seat
        kind = "dahai" if name in SELF_TURN else ("skip" if name in CALL_CHANCE else None)
        if kind is None:
            return
        decisions.append((kind, name, time.time() - t0))

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=args.user_data_dir, headless=False,
            viewport={"width": args.width, "height": args.height},
            ignore_default_args=["--enable-automation"],
            args=["--noerrdialogs", "--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        ctx.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
        page = ctx.new_page()

        def on_ws(ws):
            if "game" not in ws.url:                 # only the game-gateway socket carries ActionPrototype
                return
            def on_frame(payload):
                raw = payload if isinstance(payload, (bytes, bytearray)) else None
                if raw is None:
                    return
                if dump_fh is not None:
                    dump_fh.write(json.dumps({"t": round(time.time() - t0, 3), "dir": "recv",
                                              "url": ws.url, "opcode": "bin",
                                              "b64": base64.b64encode(raw).decode()}) + "\n")
                    dump_fh.flush()
                name, d = decode_action(bytes(raw), pb)
                if name:
                    on_decode(name, d)
            ws.on("framereceived", on_frame)
        page.on("websocket", on_ws)

        print(f"Opening {args.url} ... log in and enter a game.\n")
        try:
            page.goto(args.url, timeout=60000)
        except Exception as e:
            print(f"  goto warning: {e}")

        def click(x16: float, y16: float, *, hold: float = 0.08) -> None:
            page.mouse.move(x16 * scaler, y16 * scaler, steps=random.randint(3, 6))
            page.wait_for_timeout(int(random.uniform(40, 90)))
            page.mouse.down()
            page.wait_for_timeout(int(hold * 1000))
            page.mouse.up()

        def do_tsumogiri(name: str) -> None:
            # rightmost tile = the just-drawn tsumohai (dealer ActionNewRound has 14 tiles, no draw-gap)
            x = TEHAI_X[13] + (0.0 if name == "ActionNewRound" else TRUMO_SPACE)
            click(x, TEHAI_Y)
            page.mouse.move(CENTER[0] * scaler, CENTER[1] * scaler, steps=4)   # deselect

        def do_skip() -> None:
            click(*BUTTON_SKIP)

        end = t0 + args.seconds
        try:
            while time.time() < end:
                while decisions:
                    kind, name, t = decisions.popleft()
                    seat = state["my_seat"]
                    if not args.live:
                        print(f"  [{t:7.1f}s] DRY would {kind.upper():5} (my_seat={seat}, from {name})")
                        continue
                    print(f"  [{t:7.1f}s] {kind.upper():5} (my_seat={seat}, from {name})")
                    page.wait_for_timeout(int(random.uniform(500, 1400)))   # human-ish think time
                    try:
                        do_tsumogiri(name) if kind == "dahai" else do_skip()
                    except Exception as e:
                        print(f"      click error: {e}")
                page.wait_for_timeout(80)
        except KeyboardInterrupt:
            print("\n  interrupted.")
        finally:
            if dump_fh is not None:
                dump_fh.close()
                print(f"\nDumped frames -> {os.path.join(args.dump, 'frames.jsonl')}")
            try:
                ctx.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
