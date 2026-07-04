"""paths.ai_game_name / ai_captures.

Plain-script style: PYTHONPATH=. <auto-python> tests/test_paths.py
"""
import os
import tempfile

from majsoul_eye import paths


def test_ai_game_name_multi_game():
    assert paths.ai_game_name("captures/raw/ai_session/run_3/game1.jsonl") == "ai_run_3_game1"
    assert paths.ai_game_name("captures/raw/ai_session/run_8/game6.jsonl") == "ai_run_8_game6"
    # absolute + backslash variants resolve the same
    assert paths.ai_game_name(r"D:\x\captures\raw\ai_session\run_10\game2.jsonl") == "ai_run_10_game2"


def test_ai_game_name_single_game_run():
    assert paths.ai_game_name("captures/raw/ai_session/run_1.jsonl") == "ai_run_1"


def test_ai_game_name_fallback_for_manual():
    # manual sessions (or anything not matching run/game) fall back to the stem
    assert paths.ai_game_name("captures/raw/manual/session5.jsonl") == "session5"


def test_ai_captures_globs_both_shapes():
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "raw", "ai_session")
        os.makedirs(os.path.join(base, "run_3", "game1"))
        os.makedirs(os.path.join(base, "run_3", "game1", "frames"))
        open(os.path.join(base, "run_3", "game1.jsonl"), "w").close()
        open(os.path.join(base, "run_3", "game1", "liqi.jsonl"), "w").close()   # must NOT match
        open(os.path.join(base, "run_3", "game1", "frames.jsonl"), "w").close() # must NOT match
        open(os.path.join(base, "run_1.jsonl"), "w").close()                    # single-game run
        open(os.path.join(base, "run_3", "ai_settings.json"), "w").close()      # must NOT match
        found = paths._ai_captures_in(base)      # test-seam over the real glob
        got = sorted(os.path.relpath(p, base).replace(os.sep, "/") for p in found)
        assert got == ["run_1.jsonl", "run_3/game1.jsonl"], got


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_paths OK")
