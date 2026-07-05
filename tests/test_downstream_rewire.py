"""Downstream rewire contracts: annotate name == build_dataset from-annotations
stem, and rebuild_datasets discovers via paths.ai_captures.

Plain-script style: PYTHONPATH=. <auto-python> tests/test_downstream_rewire.py
"""
import os

from majsoul_eye import paths


def test_annotate_and_build_agree_on_name():
    # Both derive the annotation filename the same way for an AI capture (nested layout).
    cap = "captures/raw/ai_session/run_3/game1/game1.jsonl"
    assert paths.ai_game_name(cap) == "ai_run_3_game1"


def test_build_dataset_uses_ai_game_name():
    src = open("scripts/train/build_dataset.py", encoding="utf-8").read()
    assert "paths.ai_game_name(args.capture)" in src
    # the old collision-prone stem line is gone
    assert "os.path.splitext(os.path.basename(args.capture))[0]" not in src


def test_annotate_uses_ai_game_name_and_ai_captures():
    src = open("scripts/annotate/annotate_ai_session.py", encoding="utf-8").read()
    assert "paths.ai_game_name(cap)" in src
    assert "paths.ai_captures()" in src


def test_rebuild_uses_ai_captures_and_frames_dir_for():
    src = open("scripts/data/rebuild_datasets.py", encoding="utf-8").read()
    assert "paths.ai_captures()" in src
    assert "paths.frames_dir_for" in src


def test_ingest_run_has_no_convert_step():
    src = open("scripts/data/ingest_run.py", encoding="utf-8").read()
    assert "convert_mjcopilot.py" not in src


def test_ingest_run_resolves_capture_via_paths():
    # the GT jsonl <-> frames dir coupling lives in paths, not re-derived inline
    src = open("scripts/data/ingest_run.py", encoding="utf-8").read()
    assert "paths.capture_for_frames_dir" in src
    assert '+ ".jsonl"' not in src


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_downstream_rewire OK")
