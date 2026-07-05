"""Real-Mortal-AI autoplay + capture, single env (Python 3.12 `auto`).

Reuses MahjongCopilot's brain+driver VERBATIM and swaps its mitmproxy frame source
for our in-browser Playwright WebSocket tap:

  ONE Chromium (game.browser.GameBrowser, own thread + action queue)
    page.on('websocket') -> framereceived -> enqueue raw bytes        [browser thread]
    main loop drains:  liqi.LiqiProto.parse(bytes)
                       -> GameState.input(msg) -> Mortal reaction (mjai)
                       -> Automation.automate_action(reaction, game_state)  [clicks: dahai/chi/pon/kan/reach]
                       -> automate_end_kyoku on round end
    + per game a self-contained <out>/game<M>/: game<M>.jsonl (GTRecord), liqi.jsonl
      (raw wire), frames.jsonl (index) + screenshot-on-quiet -> frames/<seq>.png

Why Mortal (not tsumogiri): a real AI melds / riichis / varies discards, so the
dataset covers 副露 / 立直 / tedashi — the hard zones a tsumogiri stub never produces.

Threading: GameBrowser owns the Playwright page on its own thread; its mouse_* are
thread-safe queue puts, so Automation's worker threads marshal clicks correctly.
The WS callback only ENQUEUES bytes; all decode/react/click/screenshot run on the
main thread.

SAFETY: defaults to OBSERVE mode (logs Mortal's chosen action; NO clicking). Watch a few
of your turns, confirm the actions look right, then add --live to actually play. Use a
burner account (this is active ranked play). Requires you to log in + enter a game manually
(auto-join is --autojoin, opt-in, and depends on a lobby template that may need recalibration).

--dry-run is ORTHOGONAL to --live: it writes NOTHING to disk — no run/game dirs, no
screenshots, no GT/wire/index/metadata files (the MahjongCopilot settings file goes to a
temp dir removed on exit). Use it to smoke-test the browser/tap/AI plumbing (with or without
--live) without creating a run under captures/.

Env (one-time):
  & $PY -m pip install playwright protobuf cryptography requests Pillow
  & $PY -m playwright install chromium
  # libriichi already built+installed (Mortal source, cp312). torch already present.

Usage (PowerShell):
  $PY = "C:/Users/zsx/miniforge3/envs/auto/python.exe"
  & $PY scripts/capture/autoplay_ai.py                         # observe (log only), still writes files
  & $PY scripts/capture/autoplay_ai.py --dry-run               # observe + write NOTHING (plumbing smoke test)
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
import shutil
import sys
import tempfile
import time
from collections import deque

from majsoul_eye.capture.roi_diff import roi_diff
from majsoul_eye.capture.schema import GTRecord, GTWriter
from majsoul_eye.capture.mjcopilot_gt import make_capturing_game_state, gt_fields

MJC = r"D:/code/phoenix/MahjongCopilot"

SERVERS = {                                              # Majsoul web entry points by server
    "jp": "https://game.mahjongsoul.com/",
    "cn": "https://game.maj-soul.com/1/",
    "en": "https://mahjongsoul.game.yo-star.com/",
}


def stable_capture_step(state, frame, thresh):
    """Return ("save"|"wait", state). "save" once the current grab matches the previous
    one inside the table ROI (discard animation finished); else store ref and wait."""
    ref = state.get("ref")
    if ref is not None and roi_diff(frame, ref) <= thresh:
        return "save", {"ref": None}
    return "wait", {"ref": frame}


def _frame_index_line(seq: int, ts: float) -> dict:
    """One screenshot-index record (index-relative file path), matching the
    manual FrameSyncer's frames.jsonl shape so build_dataset consumes it the
    same way."""
    return {"seq": seq, "file": f"frames/{seq:06d}.png", "status": "ok", "ts": ts}


def auto_next_flow(*, button_guard, main_menu_visible, click_at, delay_step,
                   timeout, state, now=time.time, log=print):
    """Game-end "one more game" step generator (deps injected; unit-testable).

    Real Majsoul game-end sequence (measured live 2026-07-05, jp 段位戦):
      終局 ranking(確認) -> pt/achievement(確認) -> missions(もう一局 + 確認)
      -> rematch dialog(はい / いいえ) -> matchmaking(authGame).
    Screen identity is decided by button CO-PRESENCE (a single color box is not
    enough — the ranking screen's blue 2/3/4位 bars alias the もう一局 color, and
    the dialog's はい aliases 確認's yellow):
      - lobby main menu visible          -> STOP, never click (a blind click on the
                                            lobby's bottom icon row opens 商店).
      - はい AND いいえ present, 確認 ABSENT  -> rematch dialog: click はい (done).
      - もう一局 AND 確認 present            -> missions screen: click もう一局 (NOT
                                            確認, which would exit to the lobby).
      - 確認 present                       -> ranking/pt settlement: click 確認.
                                            No fixed count — screens vary per game.
      - nothing                          -> short wait.
    Each button_guard(kind) returns (present, frac, (x16,y9)); the click point is
    the guard's own centroid so exact button position needs no separate calibration.
    On timeout / lobby-stop: state failed=True, active=False (main loop may fall back
    to --autojoin). After clicking はい the state stays active until the next authGame
    clears it (matchmaking in progress); a main-loop watchdog covers a missed はい."""
    deadline = now() + timeout
    yield delay_step(random.uniform(2.0, 4.0))
    while now() < deadline:
        menu_ok, menu_diff = main_menu_visible()
        if menu_ok:
            log(f"  auto-next: lobby detected (diff={menu_diff:.1f}); stopping — never clicking in the lobby")
            state["failed"] = True
            state["active"] = False
            return

        yes_ok, yes_frac, yes_xy = button_guard("dialog_yes")
        no_ok, _, _ = button_guard("dialog_no")
        conf_ok, conf_frac, conf_xy = button_guard("confirm")

        if yes_ok and no_ok and not conf_ok:          # rematch confirmation dialog
            log(f"  auto-next: rematch dialog -> clicking はい (frac={yes_frac:.3f})")
            yield from click_at(yes_xy, kind="dialog_yes")
            yield delay_step(random.uniform(2.0, 3.0))
            still, _, _ = button_guard("dialog_yes")
            if not still:
                state["clicked_next"] = True          # matchmaking; next authGame clears state
                return
            log("  auto-next: dialog still up after clicking はい; retrying")
            continue

        rem_ok, rem_frac, rem_xy = button_guard("rematch")
        if rem_ok and conf_ok:                         # missions screen: もう一局, NOT 確認
            log(f"  auto-next: missions -> clicking もう一局 (frac={rem_frac:.3f})")
            yield from click_at(rem_xy, kind="rematch")
            yield delay_step(random.uniform(1.5, 2.5))
            continue

        if conf_ok:                                    # ranking / pt settlement: advance
            log(f"  auto-next: settlement -> clicking 確認 (frac={conf_frac:.3f})")
            yield from click_at(conf_xy, kind="confirm")
            yield delay_step(random.uniform(1.0, 2.0))
            continue

        log(f"  auto-next waiting (confirm={conf_frac:.3f} rematch={button_guard('rematch')[1]:.3f} "
            f"dialog_yes={yes_frac:.3f})")
        yield delay_step(0.5)

    log("  auto-next timed out before reaching matchmaking")
    state["failed"] = True
    state["active"] = False


def mjc_settings(op_delay: tuple[float, float] = (0.5, 1.0), *, url: str = SERVERS["jp"],
                 width: int = 1280, height: int = 720, model: str = "v4_js_09260526.pth",
                 live: bool = False, randomize: int = 2, autojoin: bool = False) -> dict:
    """The MahjongCopilot settings-file seed dict (`Settings()` loads this so no "use
    default value" warning spam). Pure/testable — extracted so `--op-delay` (the random
    hesitation between the AI receiving an operation offer and clicking) has a unit-testable
    home; every other key matches the pre-extraction inline literal byte-for-byte."""
    lo, hi = op_delay
    return {
        "update_url": "https://update.mjcopilot.com", "auto_launch_browser": False, "gui_set_dpi": True,
        "browser_width": width, "browser_height": height, "ms_url": url,
        "enable_chrome_ext": False, "mitm_port": 10999, "upstream_proxy": "", "enable_proxinject": False,
        "inject_process_name": "jantama_mahjongsoul", "language": "ZHS", "enable_overlay": False,
        "model_type": "Local", "model_file": model, "model_file_3p": "",
        "akagi_ot_url": "", "akagi_ot_apikey": "", "mjapi_url": "https://mjai.7xcnnw11phu.eu.org",
        "mjapi_user": "", "mjapi_secret": "", "mjapi_models": [], "mjapi_model_select": "baseline",
        "enable_automation": bool(live), "auto_idle_move": False, "auto_random_move": True,
        "auto_reply_emoji_rate": 0.0, "auto_emoji_intervel": 5.0, "auto_dahai_drag": False,
        "game_end_reminder": False, "ai_randomize_choice": max(0, min(5, randomize)),
        "delay_random_lower": lo, "delay_random_upper": hi, "auto_retry_interval": 1.5,
        "auto_join_game": bool(autojoin), "auto_join_level": 1, "auto_join_mode": "4E",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-Mortal-AI Majsoul autoplay + capture (single env).")
    ap.add_argument("--server", choices=list(SERVERS), default="jp",
                    help="Majsoul server: jp=game.mahjongsoul.com, cn=game.maj-soul.com/1/, en=yo-star.")
    ap.add_argument("--url", default=None, help="Override the server URL directly (else derived from --server).")
    ap.add_argument("--out", default=None,
                    help="Session PARENT dir; each run writes a fresh run_<N>/ subdir. Default: captures/raw/ai_session.")
    ap.add_argument("--model", default="v4_js_09260526.pth",
                    help="Model file under MahjongCopilot/models/ (v4_js… is ~6x faster than the ensemble).")
    ap.add_argument("--live", action="store_true",
                    help="Actually click (AI auto-plays). Default = OBSERVE (log Mortal's action, no clicking).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Write NOTHING to disk: no run/game dirs, no screenshots, no GT/wire/index/metadata "
                         "(the settings file goes to a temp dir, removed on exit). Orthogonal to --live — "
                         "combine them to exercise the full live flow without saving any data.")
    ap.add_argument("--op-delay", nargs=2, type=float, default=(0.5, 1.0), metavar=("LO", "HI"),
                    help="Random hesitation (seconds) between the AI receiving an operation offer and "
                         "clicking; overrides MahjongCopilot's delay_random_lower/upper (default 0.5 1.0). "
                         "Widen for button-frame harvest runs (e.g. --op-delay 1.5 2.5) so the FrameSyncer's "
                         "quiet capture (default 0.30s) fires while the action buttons are still on screen. "
                         "LO must be <= HI.")
    ap.add_argument("--randomize", type=int, default=2, help="ai_randomize_choice 0-5 (discard diversity).")
    ap.add_argument("--autojoin", action="store_true", help="Auto-join ranked + rejoin (lobby template; may need recalibration).")
    ap.add_argument("--auto-next", action="store_true",
                    help="After a complete game, guarded-click the result confirms and the 'one more game' button.")
    ap.add_argument("--auto-next-timeout", type=float, default=90.0,
                    help="Seconds to wait for the guarded auto-next UI flow before falling back to --autojoin, if enabled.")
    ap.add_argument("--quiet", type=float, default=0.40, help="Screenshot once the board is event-quiet this long.")
    ap.add_argument("--stable-thresh", type=float, default=3.0,
                    help="Table-ROI frame-diff below this == settled (discard animation done).")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--overlay", action="store_true",
                    help="Draw the tile detector's boxes live onto the browser (visualizer; off by default).")
    ap.add_argument("--detector-weights", default="majsoul_eye/recognize/tile_detector.pt",
                    help="Detector weights for --overlay (pass weights/detector/tile_detector_obb.pt for rotated OBB polys).")
    ap.add_argument("--overlay-fps", type=float, default=12.0, help="Overlay redraw rate (Hz); ignored with --overlay-manual.")
    ap.add_argument("--overlay-manual", action="store_true",
                    help="Manual overlay: detect once per --overlay-key press in the browser (no fps loop).")
    ap.add_argument("--overlay-key", default="Space",
                    help="KeyboardEvent.code that triggers a manual detect (e.g. Space, KeyD, Enter).")
    ap.add_argument("--overlay-conf", type=float, default=0.25, help="Detector confidence threshold for the overlay.")
    ap.add_argument("--overlay-device", default="cuda", help="Torch device for the overlay detector (cuda/cpu).")
    ap.add_argument("--lang", default=None,
                    help="Force the captured display language (zh-Hans/zh-Hant/ja/en); else server-coarse + page probe. "
                         "Written to each game<N>/metadata.json.")
    ap.add_argument("--mjc", default=MJC, help="Path to the MahjongCopilot repo.")
    ap.add_argument("--skins", action="store_true",
                    help="Route the browser through MajsoulMax (mitmproxy) to swap character/skin/牌背/桌布 "
                         "client-side for training-data diversity. Needs the 'majsoulmax' conda env + a "
                         "one-time CA trust (reuses MahjongCopilot's mitm_config cert). Bot games only.")
    ap.add_argument("--skins-port", type=int, default=23410, help="Local port for the skin MITM proxy.")
    ap.add_argument("--skins-env", default="auto",
                    help="Conda env with mitmproxy for MajsoulMax's mitmdump (auto has it after setup).")
    ap.add_argument("--skins-dir", default=os.path.join("_external", "MajsoulMax"),
                    help="Path to the MajsoulMax repo (mitmproxy addon).")
    ap.add_argument("--skins-randomize", action="store_true",
                    help="With --skins: randomize skin/牌背/桌布/牌面 per game (else unlock-all; set skins in-lobby).")
    ap.add_argument("--skins-slots", default="7,6,8",
                    help="Decoration slots to randomize (7=牌背 6=桌布 8=场景; add 13=牌面 to also randomize "
                         "the tile face, 3=手 0=立直棒 …; slot 5 unsupported). Default excludes 牌面.")
    ap.add_argument("--skins-all-seats", action="store_true",
                    help="Randomize EVERY seat's 立绘 (incl. AI opponents), not just the hero. "
                         "Confirm opponents render injected skins first (see docs).")
    args = ap.parse_args()
    if args.op_delay[0] > args.op_delay[1]:
        ap.error(f"--op-delay LO must be <= HI (got {args.op_delay[0]} {args.op_delay[1]})")
    url = args.url or SERVERS[args.server]
    if args.skins and not os.path.isabs(args.skins_dir):
        args.skins_dir = os.path.join(_ORIG_CWD, args.skins_dir)   # survive the chdir into MJC (like --out)
    from majsoul_eye.capture import overlay as overlay_mod   # light: no ultralytics until detector built
    from majsoul_eye.capture import gamemeta                 # per-game metadata (display language)
    overlay_canvas_id = overlay_mod.OVERLAY_CANVAS_ID if args.overlay else None

    if args.overlay and not os.path.isabs(args.detector_weights):
        args.detector_weights = os.path.join(_ORIG_CWD, args.detector_weights)   # survive the chdir into MJC (like --out)

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
    )
    from game.img_proc import ImgTemp, img_avg_diff   # lobby (main-menu) template check
    from bot.local.bot_local import BotMortalLocal

    if args.out is None:
        # raw/ai_session/ per the captures layout (see majsoul_eye/paths.py).
        args.out = os.path.join("captures", "raw", "ai_session")
    parent = os.path.abspath(os.path.join(_ORIG_CWD, args.out))    # session parent; each run -> run_<N>/
    if not args.dry_run:
        os.makedirs(parent, exist_ok=True)
    _runs = ([d[4:] for d in os.listdir(parent) if d.startswith("run_") and d[4:].isdigit()]
             if os.path.isdir(parent) else [])
    run_n = max((int(x) for x in _runs), default=0) + 1
    out_dir = os.path.join(parent, f"run_{run_n}")        # this run; each GAME -> out_dir/game<M>/
    if not args.dry_run:                                  # --dry-run: create no dirs, write no files
        os.makedirs(out_dir, exist_ok=True)

    # --- settings: seed a COMPLETE settings file first so Settings() loads it cleanly
    #     (no "use default value" warning spam), with all our values already correct.
    #     --dry-run has no run dir, so the settings file (which Settings also re-saves)
    #     goes to a throwaway temp dir removed in the finally. ---
    dry_tmp = tempfile.mkdtemp(prefix="autoplay_ai_dry_") if args.dry_run else None
    settings_path = os.path.join(dry_tmp if dry_tmp else out_dir, "ai_settings.json")
    seed = mjc_settings(tuple(args.op_delay), url=url, width=args.width, height=args.height,
                        model=args.model, live=args.live, randomize=args.randomize, autojoin=args.autojoin)
    with open(settings_path, "w", encoding="utf-8") as fh:
        json.dump(seed, fh, indent=2)
    st = Settings(settings_path)

    mode = ("LIVE" if args.live else "OBSERVE") + ("+DRY-RUN" if args.dry_run else "")
    print(f"[{mode}] server={args.server} model={args.model} "
          f"randomize={st.ai_randomize_choice} autojoin={args.autojoin} "
          f"auto_next={args.auto_next}  out={out_dir}"
          f"{'  (dry-run: nothing written)' if args.dry_run else ''}")
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

    # Optional skin-swap MITM: route the browser through MajsoulMax so character/skin/牌背/桌布
    # are rewritten client-side (data diversity). GT is unaffected — mod only rewrites lobby/
    # authGame, never ActionPrototype game actions. Torn down in the finally below.
    skin_proxy = None
    browser_proxy = None
    # Per-game provenance: what skin randomization was applied (written into each metadata.json).
    skin_meta = ({"enabled": True, "randomize": bool(args.skins_randomize),
                  "slots": (args.skins_slots if args.skins_randomize else None),
                  "all_seats": bool(args.skins_all_seats)}
                 if args.skins else None)
    if args.skins:
        from majsoul_eye.capture import skins as skins_mod
        from common import utils as mjc_utils              # MJC on sys.path after the chdir above
        confdir = str(mjc_utils.sub_folder(mjc_utils.Folder.MITM_CONF))   # <mjc>/mitm_config: share MJC's CA
        def _ensure_cert(cert_path):
            installed, _ = mjc_utils.is_certificate_installed(cert_path)
            if installed:
                return True
            ok, _ = mjc_utils.install_root_cert(cert_path)   # certutil -addstore Root (may prompt UAC once)
            return ok
        randomize = ({"slots": args.skins_slots, "all_seats": args.skins_all_seats}
                     if args.skins_randomize else None)
        skin_proxy = skins_mod.SkinProxy(
            args.skins_dir, port=args.skins_port, env=args.skins_env,
            confdir=confdir, ensure_cert=_ensure_cert, randomize=randomize,
            log_path=os.path.join(dry_tmp if dry_tmp else out_dir, "skin_proxy.log"))
        skin_proxy.__enter__()
        browser_proxy = skin_proxy.proxy_str

    print(f"Opening {url} ({args.server}) … log in and ENTER A GAME (burner account).", flush=True)
    browser.start(st.ms_url, browser_proxy, st.browser_width, st.browser_height, False)
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

    # Resolve the captured client display language ONCE: --lang override > page probe > server coarse.
    # (Language is a client render setting, NOT in the liqi protocol — see capture/gamemeta.py.)
    _lang_dump = None
    try:
        _rq_lang: queue.Queue = queue.Queue()
        browser._action_queue.put(lambda: _rq_lang.put(browser.page.evaluate(gamemeta.probe_language_js())))
        _lang_dump = _rq_lang.get(True, 5)
    except Exception as e:
        print(f"  lang probe failed: {e}", flush=True)
    _probe_lang = gamemeta.parse_probe_dump(_lang_dump)
    if _lang_dump and _probe_lang is None:                    # couldn't decode: dump candidates for debugging
        print(f"  [lang probe] {_lang_dump}", flush=True)
    game_language = gamemeta.resolve_language(args.server, probe=_probe_lang, override=args.lang)
    print(f"  captured language = {game_language} "
          f"(server={args.server} probe={_probe_lang} override={args.lang})", flush=True)

    def screenshot_png():
        cdp = cdp_holder[0]
        if cdp is None:
            return None
        rq: queue.Queue = queue.Queue()
        cid = overlay_canvas_id
        def _do():
            try:
                if cid:
                    browser.page.evaluate(overlay_mod.hide_canvas_js(cid))
                try:
                    res = cdp.send("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
                    data = base64.b64decode(res["data"])
                finally:
                    if cid:
                        browser.page.evaluate(overlay_mod.show_canvas_js(cid))
                rq.put(data)
            except Exception:
                rq.put(None)
        browser._action_queue.put(_do)
        try:
            return rq.get(True, 5)
        except Exception:
            return None

    # Game-end "one more game" flow.  Coordinates are in MahjongCopilot's 16x9
    # logical space; each guard samples the current CDP screenshot and returns
    # (present, colour-fraction, centroid) — the centroid is where auto_next_flow
    # actually clicks, so exact button positions self-calibrate. Boxes + colours +
    # fallbacks were MEASURED on real jp 段位戦 end-game full-screen captures
    # (2026-07-05, all four screens): 確認 yellow bottom-right (centroid ≈14.3,8.1);
    # もう一局 blue bottom-centre, missions only (≈12.2,8.3); the rematch dialog's
    # はい (≈6.5,6.6) / いいえ (≈9.5,6.6) sit centre-screen with NO active 確認 (the
    # dimmed settlement buttons behind the modal are too dark to trip the colours),
    # which is how the dialog is told apart from a settlement screen.
    _YELLOW = lambda r, g, b: r > 170 and g > 115 and b < 150 and r > b + 45
    _BLUE = lambda r, g, b: b > 105 and g > 65 and r < 165 and b > r + 20
    BUTTON_GUARDS = {
        # kind: box (x0,y0,x1,y1) in 16x9, colour pred, min hit-fraction, fallback click pt
        "confirm":    {"box": (13.30, 7.55, 15.75, 8.80), "pred": _YELLOW, "min_frac": 0.035, "fallback": (14.35, 8.10)},
        "rematch":    {"box": (11.10, 7.95, 13.40, 8.75), "pred": _BLUE,   "min_frac": 0.045, "fallback": (12.21, 8.32)},
        "dialog_yes": {"box": (5.00, 5.00, 8.50, 7.10),   "pred": _YELLOW, "min_frac": 0.030, "fallback": (6.50, 6.60)},
        "dialog_no":  {"box": (8.60, 5.00, 11.00, 7.10),  "pred": _BLUE,   "min_frac": 0.030, "fallback": (9.50, 6.57)},
    }

    def button_guard(kind: str) -> tuple[bool, float, tuple[float, float]]:
        """Return (present, colour-fraction, click-point-in-16x9). The click point
        is the centroid of the colour-matching pixels (falls back to a fixed point
        when the button isn't present)."""
        spec = BUTTON_GUARDS[kind]
        fallback = spec["fallback"]
        png = screenshot_png()
        if not png:
            return False, 0.0, fallback
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(png)).convert("RGB")
        except Exception as e:
            print(f"  auto-next guard image error: {e}", flush=True)
            return False, 0.0, fallback

        w, h = img.size
        x0, y0, x1, y1 = spec["box"]
        px0, py0 = max(0, round(x0 / 16 * w)), max(0, round(y0 / 9 * h))
        px1, py1 = min(w, round(x1 / 16 * w)), min(h, round(y1 / 9 * h))
        if px1 <= px0 or py1 <= py0:
            return False, 0.0, fallback

        step = max(1, min(px1 - px0, py1 - py0) // 80)
        total = hits = 0
        sx = sy = 0
        pred = spec["pred"]
        pix = img.load()
        for y in range(py0, py1, step):
            for x in range(px0, px1, step):
                r, g, b = pix[x, y]
                total += 1
                if pred(r, g, b):
                    hits += 1
                    sx += x
                    sy += y
        frac = hits / total if total else 0.0
        if hits:
            click = (sx / hits / w * 16, sy / hits / h * 9)   # centroid -> 16x9
        else:
            click = fallback
        return frac >= spec["min_frac"], frac, click

    def guarded_click_steps(x16: float, y9: float,
                            jitter_x: float = 0.18, jitter_y: float = 0.10):
        rx = (x16 + random.uniform(-jitter_x, jitter_x)) * automation.scaler
        ry = (y9 + random.uniform(-jitter_y, jitter_y)) * automation.scaler
        yield ActionStepMove(rx, ry, random.randint(2, 5))
        yield ActionStepDelay(random.uniform(0.12, 0.25))
        yield ActionStepClick(random.randint(60, 100))

    def click_at(xy, kind=None):
        yield from guarded_click_steps(xy[0], xy[1], 0.12, 0.08)

    auto_next_state = {"active": False, "started": 0.0, "clicked_next": False, "failed": False}

    def main_menu_visible() -> tuple[bool, float]:
        """True if the current screen is the lobby main menu (MahjongCopilot's
        masked-template diff, threshold as in GameVisual.comp_temp). Uses OUR CDP
        screenshot, not browser.screen_shot() (page.screenshot flickers WebGL)."""
        png = screenshot_png()
        if not png:
            return False, -1.0
        try:
            from PIL import Image
            base, mask = automation.g_v.temp_dict[ImgTemp.MAIN_MENU]
            diff = img_avg_diff(base, Image.open(io.BytesIO(png)).convert("RGB"), mask)
            return diff < 30, diff
        except Exception as e:
            print(f"  auto-next menu check error: {type(e).__name__}: {e}", flush=True)
            return False, -1.0

    def start_auto_next() -> None:
        auto_next_state.update(active=True, started=time.time(), clicked_next=False, failed=False)
        automation.stop_previous()
        task = AutomationTask(browser, "Auto_NextGame", "Confirming game end and requesting another game")
        automation._task = task
        flow = auto_next_flow(
            button_guard=button_guard, main_menu_visible=main_menu_visible,
            click_at=click_at, delay_step=ActionStepDelay,
            timeout=args.auto_next_timeout, state=auto_next_state,
            log=lambda m: print(m, flush=True))
        task.start_action_steps(flow, None)

    game_state = None
    drain_mjai = None           # closure from make_capturing_game_state (per game)
    game_hero_account = None    # hero accountId from the authGame REQ (identifies OUR player in the RES)
    game_idx = 0                # which game in this run (-> out_dir/game<idx>/)
    game_wire_fh = None         # current game's raw-wire liqi.jsonl handle
    gt_writer = None            # current game's GTRecord writer (game<idx>.jsonl)
    game_index_fh = None        # current game's screenshot-index frames.jsonl handle
    game_frames_dir = None      # current game's frames/ dir
    seq = 0                     # per-GAME frame counter
    pending_seq = None          # latest board-changing seq awaiting a screenshot
    fulfilled_seq = None
    last_event_t = 0.0
    _stab = {"ref": None}       # pixel-stability ref for the currently-armed pending_seq

    def write_gt(seq, ts, msg):
        """Derive this message's mjai and, if any, append a GTRecord. Wrapped so
        recording can never break the capture loop (mirrors akagi_tap)."""
        if gt_writer is None or drain_mjai is None:
            return
        try:
            mjai = drain_mjai()
            if not mjai:
                return                                   # emit-on-new-mjai (see design §4)
            method, action_name = gt_fields(msg)
            gt_writer.put(GTRecord(
                seq=seq, ts=ts, flow_id="", seat=getattr(game_state, "seat", -1),
                last_op_step=0, syncing=False, method=method, action_name=action_name,
                raw_liqi=msg, mjai=mjai))
        except Exception as e:
            print(f"  gt write err seq {seq}: {type(e).__name__}: {e}", flush=True)

    def maybe_screenshot():
        nonlocal pending_seq, fulfilled_seq
        if game_frames_dir is None or pending_seq is None or pending_seq == fulfilled_seq:
            return
        if (time.time() - last_event_t) < args.quiet:
            return
        png = screenshot_png()                          # CDP capture (no viewport flicker)
        if not png:
            return
        import numpy as np
        import cv2
        arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        action, new_stab = stable_capture_step(_stab, arr, args.stable_thresh)
        _stab["ref"] = new_stab["ref"]                   # mutate in place (no nonlocal rebind needed)
        if action == "wait":
            return                                       # picture still moving; retry next tick
        with open(os.path.join(game_frames_dir, f"{pending_seq:06d}.png"), "wb") as fh:
            fh.write(png)
        if game_index_fh is not None:
            game_index_fh.write(json.dumps(_frame_index_line(pending_seq, time.time())) + "\n")
            game_index_fh.flush()
        fulfilled_seq = pending_seq

    overlay = None
    if args.overlay:
        eval_js = lambda js: browser._action_queue.put(lambda: browser.page.evaluate(js))
        def eval_js_result(js):                     # round-trip eval (returns the JS value); manual-mode poll
            rq: queue.Queue = queue.Queue()
            browser._action_queue.put(lambda: rq.put(browser.page.evaluate(js)))
            try:
                return rq.get(True, 5)
            except Exception:
                return None
        try:
            overlay = overlay_mod.DetectionOverlay(
                capture_png=screenshot_png, eval_js=eval_js,
                weights=args.detector_weights, device=args.overlay_device,
                fps=args.overlay_fps, conf=args.overlay_conf,
                canvas_id=overlay_canvas_id,
                manual=args.overlay_manual, key=args.overlay_key, eval_js_result=eval_js_result,
            )
            print(f"[overlay] loading detector {args.detector_weights} on {args.overlay_device} …", flush=True)
            overlay.start()
            if args.overlay_manual:
                print(f"[overlay] manual: press '{args.overlay_key}' in the browser to detect once", flush=True)
            else:
                print(f"[overlay] live @ {args.overlay_fps:g} fps", flush=True)
        except Exception as e:
            print(f"[overlay] disabled (init failed): {type(e).__name__}: {e}", flush=True)
            overlay = None

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
                    if game_wire_fh is not None:
                        game_wire_fh.close()
                    if game_index_fh is not None:
                        game_index_fh.close()
                    if gt_writer is not None:
                        gt_writer.close()
                    game_idx += 1
                    game_dir = os.path.join(out_dir, f"game{game_idx}")
                    # --dry-run: leave frames-dir + every file handle None, so the write paths
                    # below and maybe_screenshot no-op; the game still plays and logs decisions.
                    game_frames_dir = None if args.dry_run else os.path.join(game_dir, "frames")
                    if not args.dry_run:
                        os.makedirs(game_frames_dir, exist_ok=True)
                        gamemeta.write_metadata(game_dir, game_language,   # game<N>/metadata.json
                                                extra={"skins": skin_meta} if skin_meta else None)
                        game_wire_fh = open(os.path.join(game_dir, "liqi.jsonl"), "w", encoding="utf-8")
                        game_index_fh = open(os.path.join(game_dir, "frames.jsonl"), "w", encoding="utf-8")
                        gt_writer = GTWriter(os.path.join(game_dir, f"game{game_idx}.jsonl"))
                    seq, pending_seq, fulfilled_seq = 0, None, None
                    game_hero_account = (msg.get("data") or {}).get("accountId")
                    game_state, drain_mjai = make_capturing_game_state(GameState, bot)
                    seq += 1
                    if game_wire_fh is not None:
                        game_wire_fh.write(json.dumps({"seq": seq, "ts": ts,
                                                       "b64": base64.b64encode(raw).decode()}) + "\n")
                        game_wire_fh.flush()
                    game_state.input(msg)
                    write_gt(seq, ts, msg)
                    automation.on_enter_game()
                    print(f"  game{game_idx} start -> {game_dir}"
                          f"{' (dry-run: not written)' if args.dry_run else ''}", flush=True)
                    continue

                if game_state is None:                  # pre-game frames on the socket (before authGame)
                    continue                            # (--dry-run keeps no wire handle; gate on game_state)
                if method in METHODS_TO_IGNORE:
                    continue                            # parsed (LiqiProto state kept) but not recorded/routed

                seq += 1
                if game_wire_fh is not None:
                    game_wire_fh.write(json.dumps({"seq": seq, "ts": ts,
                                                   "b64": base64.b64encode(raw).decode()}) + "\n")
                    game_wire_fh.flush()

                # authGame RES (post-mod.py rewrite) carries the ACTUAL per-seat skins → record the
                # real 牌背/桌布/场景 + 立绘 into this game's metadata.json (over the config-only stub).
                if not args.dry_run and skin_meta and (mtype, method) == (liqi.MsgType.RES, liqi.LiqiMethod.authGame):
                    try:
                        actual = gamemeta.extract_authgame_skins(msg.get("data") or {},
                                                                 hero_account=game_hero_account)
                        gamemeta.write_metadata(game_dir, game_language,
                                                extra={"skins": {**skin_meta, **actual}})
                        print(f"  skins: table={actual['table']} "
                              f"chars={[(c['charid'], c['skin']) for c in actual['characters']]}", flush=True)
                    except Exception as e:
                        print(f"  skin-meta extract err: {type(e).__name__}: {e}", flush=True)

                try:
                    reaction = game_state.input(msg)
                except Exception as e:
                    print(f"  game_state.input err on {method}: {type(e).__name__}: {e}", flush=True)
                    continue
                write_gt(seq, ts, msg)
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
                        print(f"  [g{game_idx} seq {seq}] OBS {rtype} {pai}", flush=True)
                elif kyoku_just_ended and not game_state.is_ms_syncing:
                    if args.live:
                        automation.automate_end_kyoku(game_state)
                    print(f"  [g{game_idx} seq {seq}] kyoku ended"
                          + ("" if args.live else " (observe: not confirming)"), flush=True)

                if game_state.is_game_ended:
                    print(f"  [g{game_idx} seq {seq}] GAME ENDED", flush=True)
                    if args.live and args.auto_next:
                        start_auto_next()
                    else:
                        automation.on_end_game()
                    game_state = None
                    drain_mjai = None
                    if game_wire_fh is not None:
                        game_wire_fh.close()
                        game_wire_fh = None
                    if game_index_fh is not None:
                        game_index_fh.close()
                        game_index_fh = None
                    if gt_writer is not None:
                        gt_writer.close()
                        gt_writer = None

            maybe_screenshot()
            # Watchdog: next was clicked (flow returned, state stays active) but no authGame
            # ever arrived — misclick or matchmaking never started. Give up so the run doesn't
            # sit "active" forever; with --autojoin the fallback below then takes over.
            if (auto_next_state["active"] and game_state is None
                    and time.time() - auto_next_state["started"] > args.auto_next_timeout + 120):
                print("  auto-next: no game started after clicking next; giving up on this flow", flush=True)
                auto_next_state.update(active=False, failed=True)
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
        if overlay is not None:
            overlay.stop()
        if game_wire_fh is not None:
            game_wire_fh.close()
        if game_index_fh is not None:
            game_index_fh.close()
        if gt_writer is not None:
            gt_writer.close()
        try:
            browser.stop(True)
        except Exception:
            pass
        if skin_proxy is not None:
            skin_proxy.__exit__(None, None, None)          # stop the mitmdump subprocess (no orphan)
        if dry_tmp is not None:
            shutil.rmtree(dry_tmp, ignore_errors=True)     # throwaway settings dir
            print("dry-run: nothing written to disk", flush=True)
        else:
            print(f"frames + screenshots -> {out_dir}", flush=True)


_ORIG_CWD = os.path.abspath(os.getcwd())

if __name__ == "__main__":
    main()
