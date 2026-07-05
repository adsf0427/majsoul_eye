"""paths.ai_game_name / frames_dir_for / capture_for_frames_dir / ai_captures.

Canonical AI layout is NESTED: the GT jsonl lives INSIDE its frames dir
(``run_N/gameM/gameM.jsonl``). The old sibling shape (``run_N/gameM.jsonl`` next
to ``gameM/``) is still resolved as legacy (manual sessions keep it).

Plain-script style: PYTHONPATH=. <auto-python> tests/test_paths.py
"""
import os
import tempfile

from majsoul_eye import paths


def test_ai_game_name_multi_game():
    # nested (new canon): jsonl inside its own frames dir
    assert paths.ai_game_name("captures/raw/ai_session/run_3/game1/game1.jsonl") == "ai_run_3_game1"
    # absolute + backslash variants resolve the same
    assert paths.ai_game_name(r"D:\x\captures\raw\ai_session\run_10\game2\game2.jsonl") == "ai_run_10_game2"
    # legacy sibling shape still resolves (un-migrated trees)
    assert paths.ai_game_name("captures/raw/ai_session/run_8/game6.jsonl") == "ai_run_8_game6"


def test_ai_game_name_single_game_run():
    assert paths.ai_game_name("captures/raw/ai_session/run_1/run_1.jsonl") == "ai_run_1"  # nested
    assert paths.ai_game_name("captures/raw/ai_session/run_1.jsonl") == "ai_run_1"        # legacy sibling


def test_ai_game_name_fallback_for_manual():
    # manual sessions (or anything not matching run/game) fall back to the stem
    assert paths.ai_game_name("captures/raw/manual/session5.jsonl") == "session5"


def test_frames_dir_for_both_shapes():
    # nested: frames dir IS the jsonl's parent dir
    got = paths.frames_dir_for("captures/raw/ai_session/run_3/game1/game1.jsonl")
    assert got.replace(os.sep, "/") == "captures/raw/ai_session/run_3/game1"
    # legacy sibling: same-stem dir next to the jsonl
    got = paths.frames_dir_for("captures/raw/manual/session5.jsonl")
    assert got.replace(os.sep, "/") == "captures/raw/manual/session5"


def test_capture_for_frames_dir():
    with tempfile.TemporaryDirectory() as d:
        gd = os.path.join(d, "run_3", "game1")
        os.makedirs(gd)
        # nothing on disk yet -> the nested canon
        assert paths.capture_for_frames_dir(gd) == os.path.join(gd, "game1.jsonl")
        # legacy sibling only -> picked up
        sib = os.path.join(d, "run_3", "game1.jsonl")
        open(sib, "w").close()
        assert paths.capture_for_frames_dir(gd) == sib
        # nested twin appears -> preferred over the sibling
        open(os.path.join(gd, "game1.jsonl"), "w").close()
        assert paths.capture_for_frames_dir(gd) == os.path.join(gd, "game1.jsonl")


def test_ai_captures_globs_all_shapes():
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "raw", "ai_session")
        # nested (new canon)
        os.makedirs(os.path.join(base, "run_3", "game1", "frames"))
        open(os.path.join(base, "run_3", "game1", "game1.jsonl"), "w").close()
        open(os.path.join(base, "run_3", "game1", "liqi.jsonl"), "w").close()   # must NOT match
        open(os.path.join(base, "run_3", "game1", "frames.jsonl"), "w").close() # must NOT match
        # legacy sibling with no nested twin -> still discovered
        os.makedirs(os.path.join(base, "run_4", "game1"))
        open(os.path.join(base, "run_4", "game1.jsonl"), "w").close()
        # mid-migration: BOTH shapes present -> only the nested one counts (no dup name)
        os.makedirs(os.path.join(base, "run_5", "game2"))
        open(os.path.join(base, "run_5", "game2.jsonl"), "w").close()
        open(os.path.join(base, "run_5", "game2", "game2.jsonl"), "w").close()
        open(os.path.join(base, "run_1.jsonl"), "w").close()                    # legacy single-game run
        open(os.path.join(base, "run_3", "ai_settings.json"), "w").close()      # must NOT match
        found = paths._ai_captures_in(base)      # test-seam over the real glob
        got = sorted(os.path.relpath(p, base).replace(os.sep, "/") for p in found)
        assert got == ["run_1.jsonl", "run_3/game1/game1.jsonl",
                       "run_4/game1.jsonl", "run_5/game2/game2.jsonl"], got


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_paths OK")
