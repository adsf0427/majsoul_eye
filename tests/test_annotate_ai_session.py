"""Tests for scripts/annotate/annotate_ai_session.py — the per-game worker's
degenerate-input handling (the heavy vision path is exercised end-to-end by the
dataset builds themselves)."""
import importlib.util
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_module():
    path = os.path.join(os.path.dirname(__file__), "..",
                        "scripts", "annotate", "annotate_ai_session.py")
    spec = importlib.util.spec_from_file_location("annotate_ai_session", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_process_capture_zero_annotatable_frames_is_skip_not_crash():
    """A capture with no annotatable frames (aborted run with an empty
    frames.jsonl — ai_session_3p run_2/game25 killed the whole v5_3p annotate
    pool via KeyError 'frames'; equally reachable when every frame falls in a
    deal/call window) must come back as a SKIP (name, None), not an exception,
    and must not leave a stray empty annotations jsonl behind."""
    mod = _load_module()
    with tempfile.TemporaryDirectory() as td:
        game = os.path.join(td, "game9")
        os.makedirs(game)
        cap = os.path.join(game, "game9.jsonl")
        with open(cap, "w", encoding="utf-8") as f:
            f.write('{"_schema": 1}\n')                       # header-only GT
        open(os.path.join(game, "frames.jsonl"), "w").close()  # 0 frames
        out = os.path.join(td, "out")
        os.makedirs(os.path.join(out, "overlays"))
        cfg = {"out": out, "frames_dir": game, "overlay_every": 0,
               "backs": False, "qa_classifier": False, "qa_per_game": 0}
        name, entry = mod._process_capture(cap, cfg)
        assert entry is None, entry
        assert not os.path.exists(os.path.join(out, f"{name}.jsonl")), \
            "skip must not leave an empty annotations jsonl"


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
    print("test_annotate_ai_session OK")
