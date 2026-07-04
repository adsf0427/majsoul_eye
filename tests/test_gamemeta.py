"""Dependency-light tests for per-game capture metadata (display language) — no browser.

Plain-script style (no pytest): run with
  PYTHONPATH=. <auto-python> tests/test_gamemeta.py
"""
import json
import os
import tempfile

from majsoul_eye.capture import gamemeta as gm  # noqa: E402


def test_server_coarse_language():
    # jp/en are unambiguous from the server; cn defaults to Simplified.
    assert gm.resolve_language("jp") == "ja"
    assert gm.resolve_language("en") == "en"
    assert gm.resolve_language("cn") == "zh-Hans"
    assert gm.resolve_language("xx") == "unknown"        # unknown server


def test_override_wins():
    # explicit --lang override beats both the probe and the server coarse map.
    assert gm.resolve_language("cn", override="cht") == "zh-Hant"
    assert gm.resolve_language("jp", override="en") == "en"
    assert gm.resolve_language("cn", override="zh-Hant", probe="chs") == "zh-Hant"


def test_probe_refines_over_server_but_under_override():
    assert gm.resolve_language("cn", probe="cht") == "zh-Hant"          # probe refines cn -> traditional
    assert gm.resolve_language("cn", probe="cht", override="chs") == "zh-Hans"  # override still wins
    assert gm.resolve_language("cn", probe=None) == "zh-Hans"           # no probe -> server coarse


def test_normalize_lang():
    assert gm.normalize_lang("chs") == "zh-Hans"
    assert gm.normalize_lang("CHT") == "zh-Hant"
    assert gm.normalize_lang("zh_Hant") == "zh-Hant"
    assert gm.normalize_lang("zh") == "zh-Hans"          # bare zh -> Simplified default
    assert gm.normalize_lang("jp") == "ja"
    assert gm.normalize_lang("en-US") == "en"
    assert gm.normalize_lang("") is None
    assert gm.normalize_lang(None) is None


def test_parse_probe_dump_conservative():
    # Trust a lang/locale KEY whose value is a recognizable code.
    dump = json.dumps({"localStorage": {"config.language": "chs"},
                       "navigatorLanguage": "en-US", "cookie": ""})
    assert gm.parse_probe_dump(dump) == "zh-Hans"
    assert gm.parse_probe_dump(json.dumps({"localStorage": {"account.locale": "cht"}})) == "zh-Hant"
    # A code-looking value under a NON-lang key is NOT trusted (avoid false positives).
    assert gm.parse_probe_dump(json.dumps({"localStorage": {"someBlob": "en"}})) is None
    # Garbage / empty.
    assert gm.parse_probe_dump("not json") is None
    assert gm.parse_probe_dump(json.dumps({"localStorage": {}})) is None
    assert gm.parse_probe_dump(None) is None


def test_probe_language_js_is_str():
    js = gm.probe_language_js()
    assert isinstance(js, str)
    assert "localStorage" in js and "navigator.language" in js


def test_write_metadata():
    with tempfile.TemporaryDirectory() as d:
        path = gm.write_metadata(d, "zh-Hant")
        assert path == os.path.join(d, "metadata.json")
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
        assert obj == {"language": "zh-Hant"}


def test_autoplay_ai_exposes_lang_flag():
    import argparse
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "capture", "autoplay_ai.py")
    spec = importlib.util.spec_from_file_location("autoplay_ai_lang_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seen = {}
    real_parse = argparse.ArgumentParser.parse_args
    def capture_parse(self, *a, **k):
        for act in self._actions:
            seen[tuple(act.option_strings)] = act
        raise SystemExit(0)                       # stop before it touches the network
    argparse.ArgumentParser.parse_args = capture_parse
    try:
        try:
            mod.main()
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = real_parse
    flat = {opt for opts in seen for opt in opts}
    assert "--lang" in flat
    assert {opts[0]: act.default for opts, act in seen.items()}["--lang"] is None   # no override by default


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_gamemeta OK")
