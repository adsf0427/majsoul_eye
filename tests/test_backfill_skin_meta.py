"""Tests for scripts/capture/backfill_skin_meta.py (one-shot skins-provenance backfill).

Dependency-light: exercises the pure merge + per-game processing via an injected decoder —
no protobuf / _external needed. Plain-script style (no pytest): run with
  PYTHONPATH=. <auto-python> tests/test_backfill_skin_meta.py
"""
import copy
import importlib.util
import json
import os
import tempfile

_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "capture", "backfill_skin_meta.py")
_spec = importlib.util.spec_from_file_location("backfill_skin_meta", _path)
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

ACTUAL = {"table": {"7": 308045, "6": 30580023, "8": 307008},
          "characters": [{"account_id": 23775192, "charid": 20000103, "skin": 40010301, "robot": False}],
          "hero_account_id": 23775192}
META = {"language": "zh-Hant",
        "skins": {"enabled": True, "randomize": True, "slots": "7,6,8", "all_seats": True,
                  "table": {"0": 305624},                      # the bug: a stranger's own 装扮
                  "characters": [{"account_id": 1, "charid": 2, "skin": 3, "robot": False}]}}


def test_merge_game_metadata():
    before = copy.deepcopy(META)
    out = bf.merge_game_metadata(META, ACTUAL)
    assert out["language"] == "zh-Hant"                        # non-skins keys preserved
    s = out["skins"]
    assert s["table"] == ACTUAL["table"]                       # wrong table replaced by the hero's
    assert s["hero_account_id"] == 23775192
    assert s["characters"] == ACTUAL["characters"]
    for k in ("enabled", "randomize", "slots", "all_seats"):   # randomization config kept
        assert s[k] == META["skins"][k]
    assert META == before                                      # inputs not mutated


def _game_dir(tmp, meta):
    d = os.path.join(tmp, "run_9", "game1")
    os.makedirs(d)
    with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return d


def test_process_game_dry_run_then_apply():
    with tempfile.TemporaryDirectory() as tmp:
        d = _game_dir(tmp, META)
        decode = lambda wire_path: (23775192, {"players": [
            {"accountId": 23775192, "character": {"charid": 20000103, "skin": 40010301},
             "views": [{"slot": 7, "itemId": 308045}, {"slot": 6, "itemId": 30580023},
                       {"slot": 8, "itemId": 307008}]}]})
        status = bf.process_game(d, decode=decode, apply=False)          # dry-run: report only
        assert status.startswith("would-fix")
        with open(os.path.join(d, "metadata.json"), encoding="utf-8") as fh:
            assert json.load(fh) == META                                 # untouched
        status = bf.process_game(d, decode=decode, apply=True)
        assert status.startswith("fixed")
        with open(os.path.join(d, "metadata.json"), encoding="utf-8") as fh:
            got = json.load(fh)
        assert got["skins"]["table"] == {"7": 308045, "6": 30580023, "8": 307008}
        assert got["skins"]["hero_account_id"] == 23775192
        assert got["skins"]["slots"] == "7,6,8"
        assert got["language"] == "zh-Hant"


def test_process_game_skips():
    with tempfile.TemporaryDirectory() as tmp:
        # no skins key (a non --skins run) -> untouched even with --apply
        d = _game_dir(tmp, {"language": "ja"})
        assert bf.process_game(d, decode=lambda p: (1, {}), apply=True) == "no-skins"
        with open(os.path.join(d, "metadata.json"), encoding="utf-8") as fh:
            assert json.load(fh) == {"language": "ja"}
    with tempfile.TemporaryDirectory() as tmp:
        # wire unusable (no authGame pair) -> reported, untouched
        d = _game_dir(tmp, META)
        assert bf.process_game(d, decode=lambda p: (None, None), apply=True) == "no-authgame"
        with open(os.path.join(d, "metadata.json"), encoding="utf-8") as fh:
            assert json.load(fh) == META


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_backfill_skin_meta OK")
