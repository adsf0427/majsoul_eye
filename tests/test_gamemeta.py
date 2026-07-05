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


def test_majsoul_real_codes():
    # MajSoul's own localStorage codes (chs=简体, chs_t=繁體); '_' is normalized to '-'.
    assert gm.normalize_lang("chs") == "zh-Hans"
    assert gm.normalize_lang("chs_t") == "zh-Hant"
    assert gm.normalize_lang("kr") == "ko"
    # the REAL probe dump observed on game.maj-soul.com (client set to 繁體)
    real = json.dumps({"localStorage": {"prefer_language": "chs_t", "language": "chs_t"},
                       "navigatorLanguage": "en-US", "cookie": "G_ENABLED_IDPS=google"})
    assert gm.parse_probe_dump(real) == "zh-Hant"
    assert gm.resolve_language("cn", probe=gm.parse_probe_dump(real)) == "zh-Hant"
    # same client set to 简体 would store "chs"
    simp = json.dumps({"localStorage": {"prefer_language": "chs", "language": "chs"}})
    assert gm.parse_probe_dump(simp) == "zh-Hans"


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


def test_extract_authgame_skins():
    # parsed authGame RES (MahjongCopilot camelCase): hero in players[] with views + skin, AI in robots[].
    data = {
        "players": [{"accountId": 123, "avatarId": 400421,
                     "character": {"charid": 200042, "skin": 400421},
                     "views": [{"slot": 7, "itemId": 305015}, {"slot": 6, "itemId": 305012},
                               {"slot": 8, "itemId": 307001}]}],
        "robots": [{"character": {"charid": 200005, "skin": 400501}},
                   {"character": {"charid": 200009, "skin": 400901}},
                   {"accountId": 0, "character": {"charid": 200013, "skin": 401301}}],
        "seatList": [123, 0, 0, 0],
    }
    out = gm.extract_authgame_skins(data, hero_account=123)
    assert out["table"] == {"7": 305015, "6": 305012, "8": 307001}   # hero views = table 牌背/桌布/场景
    assert out["hero_account_id"] == 123
    assert out["characters"][0] == {"account_id": 123, "charid": 200042, "skin": 400421, "robot": False}
    assert [c["charid"] for c in out["characters"]] == [200042, 200005, 200009, 200013]
    assert [c["robot"] for c in out["characters"]] == [False, True, True, True]
    # graceful on empties / missing character
    assert gm.extract_authgame_skins({}) == {"table": {}, "characters": []}
    assert gm.extract_authgame_skins({"players": [{}]})["characters"][0]["skin"] == 0


def test_extract_authgame_skins_hero_not_first():
    # REAL layout (ai_session2): players[] is account_id-sorted, NOT hero-first (seat order lives in
    # seatList). table must be the HERO's views — mod.py rewrites only the hero's — never players[0]'s.
    data = {
        "players": [
            {"accountId": 22641905, "character": {"charid": 200078, "skin": 407801},
             "views": [{"slot": 0, "itemId": 305624}, {"slot": 7, "itemId": 30570013}]},   # a stranger's own 装扮
            {"accountId": 23775192, "character": {"charid": 20000103, "skin": 40010301},
             "views": [{"slot": 7, "itemId": 308045}, {"slot": 6, "itemId": 30580023},
                       {"slot": 8, "itemId": 307008}]},                                    # the hero (randomized)
        ],
        "seatList": [23775192, 22641905],
    }
    out = gm.extract_authgame_skins(data, hero_account=23775192)
    assert out["table"] == {"7": 308045, "6": 30580023, "8": 307008}
    assert out["hero_account_id"] == 23775192
    # hero unknown / not found -> EMPTY table (a stranger's cosmetics must not pose as our swap),
    # and no hero_account_id key (nothing trustworthy to record).
    for out in (gm.extract_authgame_skins(data), gm.extract_authgame_skins(data, hero_account=999)):
        assert out["table"] == {}
        assert "hero_account_id" not in out


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


def test_write_metadata_extra():
    # extra (e.g. skin provenance) is merged alongside language; language stays first / present.
    with tempfile.TemporaryDirectory() as d:
        skins = {"enabled": True, "randomize": True, "slots": "7,6,8", "all_seats": False}
        gm.write_metadata(d, "ja", extra={"skins": skins})
        with open(os.path.join(d, "metadata.json"), encoding="utf-8") as fh:
            obj = json.load(fh)
        assert obj == {"language": "ja", "skins": skins}
        # falsy extra -> language only (backward compatible)
        gm.write_metadata(d, "ja", extra=None)
        with open(os.path.join(d, "metadata.json"), encoding="utf-8") as fh:
            assert json.load(fh) == {"language": "ja"}


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
