"""Real-Mortal-AI autoplay + capture, single env (Python 3.12 `auto`).

Reuses MahjongCopilot's brain+driver VERBATIM and swaps its mitmproxy frame source
for our in-browser Playwright WebSocket tap:

  ONE Chromium (game.browser.GameBrowser, own thread + action queue)
    page.on('websocket') -> framereceived -> enqueue raw bytes        [browser thread]
    main loop drains:  liqi.LiqiProto.parse(bytes)
                       -> GameState.input(msg) -> Mortal reaction (mjai)
                       -> Automation.automate_action(reaction, game_state)  [clicks: dahai/chi/pon/kan/reach]
                       -> automate_end_kyoku on round end
    + dump raw game frames -> <out>/frames.jsonl   (offline GT via our Replayer/decoder)
    + screenshot-on-quiet  -> <out>/frames/<seq>.png

Why Mortal (not tsumogiri): a real AI melds / riichis / varies discards, so the
dataset covers 副露 / 立直 / tedashi — the hard zones a tsumogiri stub never produces.

Threading: GameBrowser owns the Playwright page on its own thread; its mouse_* are
thread-safe queue puts, so Automation's worker threads marshal clicks correctly.
The WS callback only ENQUEUES bytes; all decode/react/click/screenshot run on the
main thread.

SAFETY: defaults to --dry-run (logs Mortal's chosen action; NO clicking). Watch a few
of your turns, confirm the actions look right, then add --live. Use a burner account
(this is active ranked play). Requires you to log in + enter a game manually (auto-join
is --autojoin, opt-in, and depends on a lobby template that may need recalibration).

Env (one-time):
  & $PY -m pip install playwright protobuf cryptography requests Pillow
  & $PY -m playwright install chromium
  # libriichi already built+installed (Mortal source, cp312). torch already present.

Usage (PowerShell):
  $PY = "C:/Users/zsx/miniforge3/envs/auto/python.exe"
  & $PY scripts/capture/autoplay_ai.py                         # dry-run
  & $PY scripts/capture/autoplay_ai.py --live --out captures/ai1
  & $PY scripts/capture/autoplay_ai.py --live --model ensemble_borda_models.pth   # stronger, slower
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import random
import sys
import time
from collections import deque

MJC = r"D:/code/phoenix/MahjongCopilot"

SERVERS = {                                              # Majsoul web entry points by server
    "jp": "https://game.mahjongsoul.com/",
    "cn": "https://game.maj-soul.com/1/",
    "en": "https://mahjongsoul.game.yo-star.com/",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-Mortal-AI Majsoul autoplay + capture (single env).")
    ap.add_argument("--server", choices=list(SERVERS), default="jp",
                    help="Majsoul server: jp=game.mahjongsoul.com, cn=game.maj-soul.com/1/, en=yo-star.")
    ap.add_argument("--url", default=None, help="Override the server URL directly (else derived from --server).")
    ap.add_argument("--out", default=None,
                    help="Session PARENT dir; each run writes a fresh run_<N>/ subdir. Default: captures/ai_session.")
    ap.add_argument("--model", default="v4_js_09260526.pth",
                    help="Model file under MahjongCopilot/models/ (v4_js… is ~6x faster than the ensemble).")
    ap.add_argument("--live", action="store_true", help="Actually click. Default = dry-run (log only).")
    ap.add_argument("--randomize", type=int, default=2, help="ai_randomize_choice 0-5 (discard diversity).")
    ap.add_argument("--autojoin", action="store_true", help="Auto-join ranked + rejoin (lobby template; may need recalibration).")
    ap.add_argument("--auto-next", action="store_true",
                    help="After a complete game, guarded-click the result confirms and the 'one more game' button.")
    ap.add_argument("--auto-next-confirms", type=int, default=2,
                    help="How many guarded yellow confirm clicks to try before clicking 'one more game'.")
    ap.add_argument("--auto-next-timeout", type=float, default=90.0,
                    help="Seconds to wait for the guarded auto-next UI flow before falling back to --autojoin, if enabled.")
    ap.add_argument("--quiet", type=float, default=0.40, help="Screenshot once the board is event-quiet this long.")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--mjc", default=MJC, help="Path to the MahjongCopilot repo.")
    args = ap.parse_args()
    url = args.url or SERVERS[args.server]

    # --- env setup: chdir into MahjongCopilot so its data files (liqi.json, models) resolve,
    #     and drop CWD from sys.path so its bundled cp311 libriichi can't shadow site-packages cp312.
    mjc = os.path.abspath(args.mjc)
    os.chdir(mjc)
    sys.path[:] = [p for p in sys.path if p not in ("", ".")]
    sys.path.append(mjc)

    import libriichi  # noqa: F401  (sanity; must be the site-packages cp312 build)
    if "site-packages" not in (libriichi.__file__ or ""):
        sys.exit(f"wrong libriichi loaded: {libriichi.__file__} (rebuild via maturin into the auto env)")
    import liqi
    from common.settings import Settings
    from common.utils import GameMode  # noqa: F401
    from game.browser import GameBrowser
    from game.game_state import GameState
    from game.automation import (
        ActionStepClick,
        ActionStepDelay,
        ActionStepMove,
        Automation,
        AutomationTask,
        Positions,
    )
    from bot.local.bot_local import BotMortalLocal

    if args.out is None:
        # raw/ai_session/ per the captures layout (see majsoul_eye/paths.py).
        args.out = os.path.join("captures", "raw", "ai_session")
    parent = os.path.abspath(os.path.join(_ORIG_CWD, args.out))    # session parent; each run -> run_<N>/
    os.makedirs(parent, exist_ok=True)
    _runs = [d[4:] for d in os.listdir(parent) if d.startswith("run_") and d[4:].isdigit()]
    run_n = max((int(x) for x in _runs), default=0) + 1
    out_dir = os.path.join(parent, f"run_{run_n}")        # this run; each GAME -> out_dir/game<M>/
    os.makedirs(out_dir, exist_ok=True)

    # --- settings: seed a COMPLETE settings file first so Settings() loads it cleanly
    #     (no "use default value" warning spam), with all our values already correct. ---
    settings_path = os.path.join(out_dir, "ai_settings.json")
    seed = {
        "update_url": "https://update.mjcopilot.com", "auto_launch_browser": False, "gui_set_dpi": True,
        "browser_width": args.width, "browser_height": args.height, "ms_url": url,
        "enable_chrome_ext": False, "mitm_port": 10999, "upstream_proxy": "", "enable_proxinject": False,
        "inject_process_name": "jantama_mahjongsoul", "language": "ZHS", "enable_overlay": False,
        "model_type": "Local", "model_file": args.model, "model_file_3p": "",
        "akagi_ot_url": "", "akagi_ot_apikey": "", "mjapi_url": "https://mjai.7xcnnw11phu.eu.org",
        "mjapi_user": "", "mjapi_secret": "", "mjapi_models": [], "mjapi_model_select": "baseline",
        "enable_automation": bool(args.live), "auto_idle_move": False, "auto_random_move": True,
        "auto_reply_emoji_rate": 0.0, "auto_emoji_intervel": 5.0, "auto_dahai_drag": False,
        "game_end_reminder": False, "ai_randomize_choice": max(0, min(5, args.randomize)),
        "delay_random_lower": 0.5, "delay_random_upper": 1.0, "auto_retry_interval": 1.5,
        "auto_join_game": bool(args.autojoin), "auto_join_level": 1, "auto_join_mode": "4E",
    }
    with open(settings_path, "w", encoding="utf-8") as fh:
        json.dump(seed, fh, indent=2)
    st = Settings(settings_path)

    print(f"[{'LIVE' if args.live else 'DRY-RUN'}] server={args.server} model={args.model} "
          f"randomize={st.ai_randomize_choice} autojoin={args.autojoin} "
          f"auto_next={args.auto_next}  out={out_dir}")
    print("Loading Mortal …", flush=True)
    model_path = os.path.join(mjc, "models", args.model)
    if not os.path.exists(model_path):
        sys.exit(f"model not found: {model_path}")
    bot = BotMortalLocal({GameMode.MJ4P: model_path})   # 4p only -> no missing-3p warning
    browser = GameBrowser(st.browser_width, st.browser_height)
    automation = Automation(browser, st)
    parser = liqi.LiqiProto()

    frame_q: deque = deque()                            # (ts, raw_bytes) from the WS tap

    METHODS_TO_IGNORE = {
        liqi.LiqiMethod.checkNetworkDelay, liqi.LiqiMethod.heartbeat, liqi.LiqiMethod.loginBeat,
        liqi.LiqiMethod.fetchAccountActivityData, liqi.LiqiMethod.fetchServerTime,
    }
    # ActionPrototype names whose frame is the deal-in animation (~2-3s): the hero
    # hand is still dealing/sorting and undealt slots are empty, so a screenshot here
    # won't match GT. Don't arm a shot for them (the annotator also drops this window
    # via state.replay.is_deal_window). ActionMJStart is the pre-deal VS splash.
    DEAL_ACTION_NAMES = {"ActionMJStart", "ActionNewRound"}

    def on_ws(ws):
        if "game" not in ws.url:                        # only the game-gateway socket carries ActionPrototype
            return
        print(f"  game socket: {ws.url}", flush=True)
        def on_frame(payload):
            if isinstance(payload, (bytes, bytearray)):
                frame_q.append((time.time(), bytes(payload)))
        ws.on("framereceived", on_frame)   # server->client: ActionPrototype, authGame RES, ...
        ws.on("framesent", on_frame)       # client->server: authGame REQ, inputOperation, ... (needed for LiqiProto REQ/RES pairing)
    # register the tap on the browser thread (page is owned there)

    print(f"Opening {url} ({args.server}) … log in and ENTER A GAME (burner account).", flush=True)
    browser.start(st.ms_url, None, st.browser_width, st.browser_height, False)
    for _ in range(150):                                # wait up to 30s for the page
        if browser.is_page_normal():
            break
        time.sleep(0.2)
    browser._action_queue.put(lambda: browser.page.on("websocket", on_ws))

    # CDP screenshot: snapshots the compositor surface directly — NO viewport/device-metrics
    # override, so Majsoul's WebGL canvas doesn't resize/flicker (page.screenshot() does).
    cdp_holder = [None]
    def _mk_cdp():
        try:
            cdp_holder[0] = browser.context.new_cdp_session(browser.page)
        except Exception as e:
            print(f"  cdp session err: {e}", flush=True)
    browser._action_queue.put(_mk_cdp)

    def screenshot_png():
        cdp = cdp_holder[0]
        if cdp is None:
            return None
        rq: queue.Queue = queue.Queue()
        def _do():
            try:
                res = cdp.send("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
                rq.put(base64.b64decode(res["data"]))
            except Exception:
                rq.put(None)
        browser._action_queue.put(_do)
        try:
            return rq.get(True, 5)
        except Exception:
            return None

    # Game-end "one more game" flow.  Coordinates are in MahjongCopilot's 16x9
    # logical space; guard boxes are sampled from the current screenshot before
    # each click so late animations / missing buttons do not receive blind clicks.
    AUTO_NEXT_BUTTON = (12.25, 8.45)
    BUTTON_GUARDS = {
        "confirm": {
            "label": "yellow confirm",
            "box": (13.30, 7.88, 15.85, 8.85),
            "min_frac": 0.035,
            "pred": lambda r, g, b: r > 170 and g > 115 and b < 150 and r > b + 45,
        },
        "next": {
            "label": "blue one-more-game",
            "box": (11.00, 7.88, 13.60, 8.85),
            "min_frac": 0.025,
            "pred": lambda r, g, b: b > 105 and g > 65 and r < 165 and b > r + 20,
        },
    }

    def button_guard(kind: str) -> tuple[bool, float]:
        spec = BUTTON_GUARDS[kind]
        png = screenshot_png()
        if not png:
            return False, 0.0
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(png)).convert("RGB")
        except Exception as e:
            print(f"  auto-next guard image error: {e}", flush=True)
            return False, 0.0

        w, h = img.size
        x0, y0, x1, y1 = spec["box"]
        px0, py0 = max(0, round(x0 / 16 * w)), max(0, round(y0 / 9 * h))
        px1, py1 = min(w, round(x1 / 16 * w)), min(h, round(y1 / 9 * h))
        if px1 <= px0 or py1 <= py0:
            return False, 0.0

        step = max(1, min(px1 - px0, py1 - py0) // 80)
        total = hits = 0
        pred = spec["pred"]
        pix = img.load()
        for y in range(py0, py1, step):
            for x in range(px0, px1, step):
                r, g, b = pix[x, y]
                total += 1
                if pred(r, g, b):
                    hits += 1
        frac = hits / total if total else 0.0
        return frac >= spec["min_frac"], frac

    def guarded_click_steps(x16: float, y9: float,
                            jitter_x: float = 0.18, jitter_y: float = 0.10):
        rx = (x16 + random.uniform(-jitter_x, jitter_x)) * automation.scaler
        ry = (y9 + random.uniform(-jitter_y, jitter_y)) * automation.scaler
        yield ActionStepMove(rx, ry, random.randint(2, 5))
        yield ActionStepDelay(random.uniform(0.12, 0.25))
        yield ActionStepClick(random.randint(60, 100))

    auto_next_state = {"active": False, "started": 0.0, "clicked_next": False, "failed": False}

    def auto_next_iter():
        deadline = time.time() + args.auto_next_timeout
        confirm_clicks = 0

        yield ActionStepDelay(random.uniform(2.0, 4.0))
        while confirm_clicks < max(0, args.auto_next_confirms) and time.time() < deadline:
            ok, frac = button_guard("confirm")
            if ok:
                print(f"  auto-next guard OK confirm frac={frac:.3f}", flush=True)
                x, y = Positions.END_KYOKU_CONFIRM
                yield from guarded_click_steps(x, y, 0.20, 0.12)
                confirm_clicks += 1
                yield ActionStepDelay(random.uniform(1.0, 2.0))
                continue

            next_ok, next_frac = button_guard("next")
            if next_ok:
                print(f"  auto-next sees next button before confirm #{confirm_clicks + 1} "
                      f"(next frac={next_frac:.3f})", flush=True)
                break

            print(f"  auto-next waiting confirm frac={frac:.3f}", flush=True)
            yield ActionStepDelay(0.5)

        while time.time() < deadline:
            ok, frac = button_guard("next")
            if ok:
                print(f"  auto-next guard OK next frac={frac:.3f}", flush=True)
                yield from guarded_click_steps(*AUTO_NEXT_BUTTON, 0.18, 0.10)
                auto_next_state["clicked_next"] = True
                # Keep the flow active while matchmaking; the next authGame clears it.
                return

            ok_confirm, confirm_frac = button_guard("confirm")
            if ok_confirm:
                print(f"  auto-next extra confirm frac={confirm_frac:.3f}", flush=True)
                x, y = Positions.END_KYOKU_CONFIRM
                yield from guarded_click_steps(x, y, 0.20, 0.12)
                yield ActionStepDelay(random.uniform(1.0, 2.0))
                continue

            print(f"  auto-next waiting next frac={frac:.3f}", flush=True)
            yield ActionStepDelay(0.5)

        auto_next_state["failed"] = True
        auto_next_state["active"] = False
        print("  auto-next guard timed out before clicking next", flush=True)

    def start_auto_next() -> None:
        auto_next_state.update(active=True, started=time.time(), clicked_next=False, failed=False)
        automation.stop_previous()
        task = AutomationTask(browser, "Auto_NextGame", "Confirming game end and requesting another game")
        automation._task = task
        task.start_action_steps(auto_next_iter(), None)

    game_state = None
    game_idx = 0                # which game in this run (-> out_dir/game<idx>/)
    game_raw_fh = None          # current game's frames.jsonl handle
    game_frames_dir = None      # current game's frames/ dir
    seq = 0                     # per-GAME frame counter
    pending_seq = None          # latest board-changing seq awaiting a screenshot
    fulfilled_seq = None
    last_event_t = 0.0

    def maybe_screenshot():
        nonlocal pending_seq, fulfilled_seq
        if game_frames_dir is None or pending_seq is None or pending_seq == fulfilled_seq:
            return
        if (time.time() - last_event_t) < args.quiet:
            return
        png = screenshot_png()                          # CDP capture (no viewport flicker)
        if png:
            with open(os.path.join(game_frames_dir, f"{pending_seq:06d}.png"), "wb") as fh:
                fh.write(png)
        fulfilled_seq = pending_seq

    print("Watching. Ctrl-C to stop.\n", flush=True)
    try:
        while True:
            while frame_q:
                ts, raw = frame_q.popleft()
                try:
                    msg = parser.parse(raw)             # parse EVERY frame in order (LiqiProto is stateful)
                except Exception as e:
                    print(f"  parse err: {type(e).__name__}: {e}", flush=True); continue
                if msg is None:
                    continue
                mtype, method = msg.get("type"), msg.get("method")

                # new game: authGame REQ -> new game<M>/ dir + fresh GameState (sets seat + re-inits bot)
                if (mtype, method) == (liqi.MsgType.REQ, liqi.LiqiMethod.authGame):
                    auto_next_state.update(active=False, started=0.0, clicked_next=False, failed=False)
                    if game_raw_fh is not None:
                        game_raw_fh.close()
                    game_idx += 1
                    game_dir = os.path.join(out_dir, f"game{game_idx}")
                    game_frames_dir = os.path.join(game_dir, "frames")
                    os.makedirs(game_frames_dir, exist_ok=True)
                    game_raw_fh = open(os.path.join(game_dir, "frames.jsonl"), "w", encoding="utf-8")
                    seq, pending_seq, fulfilled_seq = 0, None, None
                    game_state = GameState(bot)
                    seq += 1
                    game_raw_fh.write(json.dumps({"seq": seq, "ts": ts,
                                                  "b64": base64.b64encode(raw).decode()}) + "\n")
                    game_raw_fh.flush()
                    game_state.input(msg)
                    automation.on_enter_game()
                    print(f"  game{game_idx} start -> {game_dir}", flush=True)
                    continue

                if game_state is None or game_raw_fh is None:
                    continue                            # pre-game frames on the socket (before authGame)
                if method in METHODS_TO_IGNORE:
                    continue                            # parsed (LiqiProto state kept) but not recorded/routed

                seq += 1
                game_raw_fh.write(json.dumps({"seq": seq, "ts": ts,
                                              "b64": base64.b64encode(raw).decode()}) + "\n")
                game_raw_fh.flush()

                try:
                    reaction = game_state.input(msg)
                except Exception as e:
                    print(f"  game_state.input err on {method}: {type(e).__name__}: {e}", flush=True)
                    continue
                kyoku_just_ended = game_state.kyoku_just_ended
                game_state.kyoku_just_ended = False

                if method == liqi.LiqiMethod.ActionPrototype:   # board changed -> arm a screenshot
                    if (msg.get("data") or {}).get("name") not in DEAL_ACTION_NAMES:
                        pending_seq = seq                        # skip the deal-in animation frame
                        last_event_t = time.time()

                if reaction:
                    rtype = reaction.get("type")
                    pai = reaction.get("pai", "")
                    if args.live:
                        try:
                            automation.automate_action(reaction, game_state)
                        except Exception as e:
                            print(f"  automate_action err {rtype}: {e}", flush=True)
                        print(f"  [g{game_idx} seq {seq}] ACT {rtype} {pai}", flush=True)
                    else:
                        print(f"  [g{game_idx} seq {seq}] DRY {rtype} {pai}", flush=True)
                elif kyoku_just_ended and not game_state.is_ms_syncing:
                    if args.live:
                        automation.automate_end_kyoku(game_state)
                    print(f"  [g{game_idx} seq {seq}] kyoku ended"
                          + ("" if args.live else " (dry: not confirming)"), flush=True)

                if game_state.is_game_ended:
                    print(f"  [g{game_idx} seq {seq}] GAME ENDED", flush=True)
                    if args.live and args.auto_next:
                        start_auto_next()
                    else:
                        automation.on_end_game()
                    game_state = None
                    if game_raw_fh is not None:
                        game_raw_fh.close()
                        game_raw_fh = None

            maybe_screenshot()
            if auto_next_state["failed"] and args.live and args.autojoin and game_state is None:
                print("  auto-next failed; falling back to lobby autojoin", flush=True)
                auto_next_state["failed"] = False
                automation.on_end_game()
            if args.live and args.autojoin and game_state is None and not auto_next_state["active"]:
                automation.decide_lobby_action()           # auto-join next game (lobby template)
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nstopping …", flush=True)
    finally:
        if game_raw_fh is not None:
            game_raw_fh.close()
        try:
            browser.stop(True)
        except Exception:
            pass
        print(f"frames + screenshots -> {out_dir}", flush=True)


_ORIG_CWD = os.path.abspath(os.getcwd())

if __name__ == "__main__":
    main()
