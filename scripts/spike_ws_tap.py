"""Gating spike: can ONE Playwright Chromium tap Majsoul's liqi WebSocket frames
in-process (no mitmproxy / no proxinject / no CA cert)?

This is the single load-bearing unknown for the unattended-capture pipeline
(docs/DATA_AUTOMATION.md §3.1). It runs BOTH candidate capture paths against the
same live browser at once, so one run tells you which works:

  (A) page.on('websocket') -> ws.on('framereceived')   — the clean primary path,
      but Playwright only surfaces MAIN-THREAD sockets (gh issue #37048).
  (B) CDP: context.new_cdp_session(page) + Network.enable
      -> 'Network.webSocketFrameReceived'              — the fallback; sees more,
      payloads arrive base64 for binary (opcode 2) frames.

It does NOT decode protobuf — it just scans each frame's raw bytes for the ASCII
method marker ".lq.<...>" (e.g. ".lq.Lobby.oauth2Login" on the lobby socket,
".lq.FastTest.authGame" / ".lq.ActionPrototype" on the GAME socket). That alone
proves we are receiving the right frames, with zero GPL/Akagi code.

PASS  := at least one BINARY game-socket frame ('.lq.FastTest*' / 'ActionPrototype'
         / 'NotifyGame*' / 'NotifyRoom*') was seen via path (A).
         If only (B) sees it, the socket is worker-thread -> use the CDP fallback.

Usage (conda `auto` env, after `pip install playwright && playwright install chromium`):
    PYTHONPATH=. $PY scripts/spike_ws_tap.py
    PYTHONPATH=. $PY scripts/spike_ws_tap.py --url https://mahjongsoul.game.yo-star.com/ --seconds 360
    PYTHONPATH=. $PY scripts/spike_ws_tap.py --mode websocket   # only path (A)

Then: log in MANUALLY in the opened window and ENTER ANY GAME (ranked/观战/友人战).
The game socket only opens once you are in a match. Ctrl-C anytime for the summary.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

# Known Majsoul web entry points (server-dependent; pick yours via --url):
#   JP/EN: https://game.mahjongsoul.com/
#   EN   : https://mahjongsoul.game.yo-star.com/
#   CN   : https://game.maj-soul.com/1/   (often requires the downloaded client)
DEFAULT_URL = "https://game.mahjongsoul.com/"

# method-name markers (ASCII, appear verbatim in the liqi Wrapper payload)
LOBBY_MARKERS = ("Lobby", "oauth2Login", "fetchGameRecord", "heatbeat", "Heartbeat")
GAME_MARKERS = ("FastTest", "ActionPrototype", "NotifyGame", "NotifyRoom",
                "authGame", "syncGame", "NotifyPlayerLoadGame")

_LQ = b".lq."
_TOKEN_CHARS = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._")


def extract_lq_method(raw: bytes) -> str | None:
    """Find the first ``.lq.<token>`` ASCII marker in a raw liqi frame, if any."""
    i = raw.find(_LQ)
    if i < 0:
        return None
    j = i
    while j < len(raw) and raw[j] in _TOKEN_CHARS:
        j += 1
    return raw[i:j].decode("ascii", "ignore")


def classify(method: str | None) -> str:
    """Return 'game', 'lobby', or 'other' for a method marker."""
    if not method:
        return "other"
    if any(m in method for m in GAME_MARKERS):
        return "game"
    if any(m in method for m in LOBBY_MARKERS):
        return "lobby"
    return "other"


@dataclass
class Stats:
    """Per capture-path tally."""
    frames: int = 0
    binary: int = 0
    game_binary: int = 0
    lobby_frames: int = 0
    methods: Counter = field(default_factory=Counter)
    sockets: set = field(default_factory=set)
    first_game_t: float | None = None

    def record(self, *, raw: bytes | None, is_binary: bool, url: str | None, t0: float) -> str | None:
        self.frames += 1
        if url:
            self.sockets.add(url)
        if is_binary:
            self.binary += 1
        method = extract_lq_method(raw) if raw else None
        kind = classify(method)
        if method:
            self.methods[method] += 1
        if kind == "lobby":
            self.lobby_frames += 1
        if kind == "game" and is_binary:
            self.game_binary += 1
            if self.first_game_t is None:
                self.first_game_t = time.time() - t0
        return method if kind == "game" else None


def _as_bytes(payload) -> tuple[bytes | None, bool]:
    """Normalise a Playwright framereceived payload to (bytes, is_binary)."""
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload), True
    if isinstance(payload, str):
        # text frame; still scan it (some control msgs are text JSON)
        return payload.encode("utf-8", "ignore"), False
    return None, False


def main() -> None:
    ap = argparse.ArgumentParser(description="Spike: in-browser Majsoul liqi WS tap (Playwright + CDP).")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"Majsoul web URL (default: {DEFAULT_URL})")
    ap.add_argument("--seconds", type=float, default=300.0, help="How long to watch (default 300s).")
    ap.add_argument("--mode", choices=["websocket", "cdp", "both"], default="both",
                    help="Which capture path(s) to run (default both).")
    ap.add_argument("--user-data-dir", default=".spike_browser_data",
                    help="Persistent profile dir (so login survives reruns).")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--verbose", action="store_true", help="Print every frame, not just game frames.")
    ap.add_argument("--dump", default=None,
                    help="If set, write every frame (raw base64) to DIR/frames.jsonl for offline "
                         "decoding with scripts/spike_decode_frames.py.")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("playwright not installed. In the `auto` env:\n"
                 "  pip install playwright && playwright install chromium")

    ws_stats = Stats()    # path (A): page.on('websocket')
    cdp_stats = Stats()   # path (B): CDP Network.webSocketFrameReceived
    t0 = time.time()

    dump_fh = None
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        dump_fh = open(os.path.join(args.dump, "frames.jsonl"), "w", encoding="utf-8")

    def dump_frame(direction: str, raw: bytes | None, is_bin: bool, url: str | None) -> None:
        """Append one frame (raw bytes base64-encoded) to the dump JSONL, in arrival order."""
        if dump_fh is None or raw is None:
            return
        dump_fh.write(json.dumps({
            "t": round(time.time() - t0, 3), "dir": direction, "url": url,
            "opcode": "bin" if is_bin else "txt",
            "method": extract_lq_method(raw) or "",
            "b64": base64.b64encode(raw).decode("ascii"),
        }) + "\n")
        dump_fh.flush()

    def log_game(path: str, method: str | None, url: str | None) -> None:
        if method:
            print(f"  [{time.time()-t0:6.1f}s] ({path}) GAME frame  {method}"
                  f"{'  '+url if url else ''}", flush=True)

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=args.user_data_dir,
            headless=False,
            viewport={"width": args.width, "height": args.height},
            ignore_default_args=["--enable-automation"],
            args=["--noerrdialogs", "--no-sandbox",
                  "--disable-blink-features=AutomationControlled"],
        )
        # Hide the residual automation fingerprint before any page script runs.
        ctx.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")

        page = ctx.new_page()

        # --- path (A): high-level WebSocket events (must register BEFORE goto) ---
        if args.mode in ("websocket", "both"):
            def on_ws(ws):
                print(f"  [{time.time()-t0:6.1f}s] (ws-event) socket OPEN  {ws.url}", flush=True)

                def make_handler(direction: str, _url=ws.url):
                    def on_frame(payload):
                        raw, is_bin = _as_bytes(payload)
                        method = ws_stats.record(raw=raw, is_binary=is_bin, url=_url, t0=t0)
                        if args.mode in ("websocket", "both"):
                            dump_frame(direction, raw, is_bin, _url)
                        if method:
                            log_game("ws-event", method, _url)
                        elif args.verbose:
                            print(f"  [{time.time()-t0:6.1f}s] (ws-event) {'bin' if is_bin else 'txt'} "
                                  f"{len(raw) if raw else 0}B {extract_lq_method(raw) or ''}", flush=True)
                    return on_frame

                ws.on("framereceived", make_handler("recv"))
                ws.on("framesent", make_handler("sent"))
            page.on("websocket", on_ws)

        # --- path (B): raw CDP (sees worker-thread sockets too) ---
        cdp = None
        if args.mode in ("cdp", "both"):
            cdp = ctx.new_cdp_session(page)
            cdp.send("Network.enable")

            def on_cdp_frame(evt):
                resp = evt.get("response", {}) or {}
                opcode = resp.get("opcode")              # 1=text, 2=binary
                data = resp.get("payloadData", "")
                is_bin = opcode == 2
                try:
                    raw = base64.b64decode(data) if is_bin else (data or "").encode("utf-8", "ignore")
                except Exception:
                    raw = None
                method = cdp_stats.record(raw=raw, is_binary=is_bin, url=None, t0=t0)
                if args.mode == "cdp":
                    dump_frame("recv", raw, is_bin, "cdp")
                if method:
                    log_game("cdp", method, None)
            cdp.on("Network.webSocketFrameReceived", on_cdp_frame)
            cdp.on("Network.webSocketFrameSent", on_cdp_frame)

        print(f"Opening {args.url} ...", flush=True)
        try:
            page.goto(args.url, timeout=60000)
        except Exception as e:
            print(f"  goto warning: {e}", flush=True)

        print("\n>>> Log in MANUALLY and ENTER A GAME. Watching for liqi frames "
              f"for {args.seconds:.0f}s (Ctrl-C to stop early).\n", flush=True)

        deadline = t0 + args.seconds
        try:
            while time.time() < deadline:
                # touch the page so a dead tab is noticed; also keeps the loop alive
                try:
                    page.evaluate("() => 0")
                except Exception:
                    print("  page closed; stopping.", flush=True)
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n  interrupted.", flush=True)
        finally:
            try:
                ctx.close()
            except Exception:
                pass

    if dump_fh is not None:
        dump_fh.close()
        print(f"\nDumped frames -> {os.path.join(args.dump, 'frames.jsonl')}", flush=True)

    # ---------------------------- summary ----------------------------
    def report(name: str, s: Stats) -> None:
        print(f"\n=== {name} ===")
        print(f"  sockets seen        : {len(s.sockets)}")
        for u in sorted(s.sockets):
            print(f"      {u}")
        print(f"  frames total        : {s.frames}  (binary {s.binary}, lobby {s.lobby_frames})")
        print(f"  GAME binary frames  : {s.game_binary}"
              + (f"  (first at {s.first_game_t:.1f}s)" if s.first_game_t is not None else ""))
        top = s.methods.most_common(8)
        if top:
            print("  top methods         : " + ", ".join(f"{m}×{n}" for m, n in top))

    print("\n" + "=" * 60)
    if args.mode in ("websocket", "both"):
        report("path (A) page.on('websocket')", ws_stats)
    if args.mode in ("cdp", "both"):
        report("path (B) CDP webSocketFrameReceived", cdp_stats)

    print("\n" + "=" * 60)
    a_pass = ws_stats.game_binary > 0
    b_pass = cdp_stats.game_binary > 0
    if args.mode == "cdp":
        verdict = "PASS (CDP)" if b_pass else "FAIL"
    elif args.mode == "websocket":
        verdict = "PASS (page.on websocket)" if a_pass else "FAIL"
    else:
        if a_pass:
            verdict = "PASS — page.on('websocket') captures the game socket. Use the clean path."
        elif b_pass:
            verdict = ("PARTIAL — only CDP saw game frames => socket is worker-thread "
                       "(gh #37048). Use the CDP fallback (new_cdp_session).")
        else:
            verdict = ("FAIL — no game frames on either path. Did you ENTER a game? "
                       "If yes, try a window.WebSocket init-script shim.")
    print(f"VERDICT: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
