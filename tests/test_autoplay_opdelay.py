"""--op-delay LO HI overrides MahjongCopilot's delay_random_lower/upper.

The AI's random hesitation between receiving an operation offer and clicking
defaults to (0.5, 1.0)s -- too short for the FrameSyncer's quiet capture
(0.30s) to reliably fire while the action buttons are still on screen.
--op-delay widens that window for button-frame harvest runs.

Plain-script style: PYTHONPATH=. <auto-python> tests/test_autoplay_opdelay.py
"""
import importlib.util
import os
import sys


def _load_autoplay():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "capture", "autoplay_ai.py")
    spec = importlib.util.spec_from_file_location("autoplay_ai_opdelay_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_op_delay_matches_current_values():
    mod = _load_autoplay()
    settings = mod.mjc_settings((0.5, 1.0))
    assert settings["delay_random_lower"] == 0.5
    assert settings["delay_random_upper"] == 1.0


def test_op_delay_overrides_lower_and_upper():
    mod = _load_autoplay()
    settings = mod.mjc_settings((1.5, 2.5))
    assert settings["delay_random_lower"] == 1.5
    assert settings["delay_random_upper"] == 2.5


def test_default_settings_dict_matches_original_literal():
    # Every OTHER key must stay byte-identical to the pre-refactor inline dict.
    mod = _load_autoplay()
    settings = mod.mjc_settings((0.5, 1.0))
    expected = {
        "update_url": "https://update.mjcopilot.com", "auto_launch_browser": False, "gui_set_dpi": True,
        "browser_width": 1280, "browser_height": 720, "ms_url": mod.SERVERS["jp"],
        "enable_chrome_ext": False, "mitm_port": 10999, "upstream_proxy": "", "enable_proxinject": False,
        "inject_process_name": "jantama_mahjongsoul", "language": "ZHS", "enable_overlay": False,
        "model_type": "Local", "model_file": "v4_js_09260526.pth", "model_file_3p": "",
        "akagi_ot_url": "", "akagi_ot_apikey": "", "mjapi_url": "https://mjai.7xcnnw11phu.eu.org",
        "mjapi_user": "", "mjapi_secret": "", "mjapi_models": [], "mjapi_model_select": "baseline",
        "enable_automation": False, "auto_idle_move": False, "auto_random_move": True,
        "auto_reply_emoji_rate": 0.0, "auto_emoji_intervel": 5.0, "auto_dahai_drag": False,
        "game_end_reminder": False, "ai_randomize_choice": 2,
        "delay_random_lower": 0.5, "delay_random_upper": 1.0, "auto_retry_interval": 1.5,
        "auto_join_game": False, "auto_join_level": 1, "auto_join_mode": "4E",
    }
    assert settings == expected


def test_op_delay_lo_greater_than_hi_exits_via_argparse():
    mod = _load_autoplay()
    real_argv = sys.argv
    sys.argv = ["autoplay_ai.py", "--op-delay", "2.5", "1.5"]
    try:
        try:
            mod.main()
            raised = False
        except SystemExit:
            raised = True
    finally:
        sys.argv = real_argv
    assert raised, "LO > HI must argparse-error (SystemExit) before any browser/mjc setup"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_autoplay_opdelay OK")
