"""autoplay_ai unified-GT plumbing: frame-index line shape + import/flag smoke.

Plain-script style: PYTHONPATH=. <auto-python> tests/test_autoplay_gt.py
"""
import argparse
import importlib.util
import os


def _load_autoplay():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "capture", "autoplay_ai.py")
    spec = importlib.util.spec_from_file_location("autoplay_ai_gt_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gt_jsonl_written_inside_game_dir():
    # nested layout contract: the GTRecord jsonl goes INSIDE game<M>/, not next to it
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "capture", "autoplay_ai.py")
    src = open(path, encoding="utf-8").read()
    assert 'GTWriter(os.path.join(game_dir, f"game{game_idx}.jsonl"))' in src
    assert 'GTWriter(os.path.join(out_dir' not in src


def test_frame_index_line_shape():
    mod = _load_autoplay()
    line = mod._frame_index_line(9, 123.5, 0.42)
    assert line == {"seq": 9, "file": "frames/000009.png", "status": "ok", "ts": 123.5, "dt": 0.42}


def test_autoplay_ai_still_imports_and_has_flags():
    mod = _load_autoplay()
    seen = {}
    real = argparse.ArgumentParser.parse_args

    def capture(self, *a, **k):
        for act in self._actions:
            seen[tuple(act.option_strings)] = act
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = capture
    try:
        try:
            mod.main()
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = real
    flat = {opt for opts in seen for opt in opts}
    assert "--out" in flat and "--server" in flat and "--dry-run" in flat


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_autoplay_gt OK")
