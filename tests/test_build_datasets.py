"""Tests for the versioned dataset builder (scripts/data/build_datasets.py) and the
--dataset manifest expansion in the training-side scripts. Pure parts only — the
subprocess stages are the same vetted per-script invocations exercised elsewhere.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "train"))
import build_datasets as bds  # noqa: E402
import build_detector_dataset as bdd  # noqa: E402
import train_classifier as tc  # noqa: E402


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{}\n")


def test_discover_games_shapes_and_kinds():
    """AI multi-game, AI single-game, and manual session shapes are all found."""
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "ai_session")
        _touch(os.path.join(root, "run_1.jsonl"))                       # legacy single-game shape
        _touch(os.path.join(root, "run_2", "game1", "game1.jsonl"))     # nested (new canon)
        _touch(os.path.join(root, "session9.jsonl"))
        games = bds.discover_games([root])
        by_name = {g["name"]: g for g in games}
        assert set(by_name) == {"ai_session_run_1", "ai_session_run_2_game1", "session9"}, by_name
        assert by_name["ai_session_run_1"]["kind"] == "ai"
        assert by_name["ai_session_run_2_game1"]["kind"] == "ai"
        assert by_name["session9"]["kind"] == "manual"
        # frames dir = capture with .jsonl stripped, POSIX-slashed
        assert by_name["ai_session_run_2_game1"]["frames_dir"].endswith("run_2/game1")
        assert "\\" not in by_name["ai_session_run_2_game1"]["frames_dir"]
        # dir defaults to the game name (no prefix)
        assert by_name["ai_session_run_1"]["dir"] == "ai_session_run_1"


def test_discover_games_source_qualified_and_empty():
    """Same run number in two roots must NOT collide now (names are source-root
    qualified); the SAME root listed twice is a real duplicate and aborts; an empty
    root aborts."""
    with tempfile.TemporaryDirectory() as td:
        r1 = os.path.join(td, "ai_session")
        r2 = os.path.join(td, "ai_session2")
        _touch(os.path.join(r1, "run_1", "game1", "game1.jsonl"))
        _touch(os.path.join(r2, "run_1", "game1", "game1.jsonl"))
        names = {g["name"] for g in bds.discover_games([r1, r2])}
        assert names == {"ai_session_run_1_game1", "ai_session2_run_1_game1"}, names
        # same root passed twice -> genuine duplicate -> abort
        try:
            bds.discover_games([r1, r1])
            raise AssertionError("duplicate not detected")
        except SystemExit as e:
            assert "duplicate" in str(e)
        # empty root -> abort
        try:
            bds.discover_games([os.path.join(td, "empty")])
            raise AssertionError("empty root not detected")
        except SystemExit as e:
            assert "no captures" in str(e)


def test_frames_override_applies():
    """The letterboxed run_5 games must point at the de-letterboxed derived frames."""
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "ai_session")
        _touch(os.path.join(root, "run_5", "game2", "game2.jsonl"))
        (g,) = bds.discover_games([root])
        assert g["name"] == "ai_session_run_5_game2"
        assert g["frames_dir"] == bds.FRAMES_OVERRIDE["ai_session_run_5_game2"].replace(os.sep, "/")


def test_manifest_roundtrip_and_training_expansion():
    """games.json written by the builder expands into the exact --data specs the
    classifier trainer and the detector assembler consume (crops vs yolo)."""
    with tempfile.TemporaryDirectory() as td:
        ds = os.path.join(td, "v9")
        os.makedirs(ds)
        games = [
            {"name": "ai_run_1", "dir": "ai_run_1", "kind": "ai",
             "capture": "captures/raw/ai_session/run_1.jsonl",
             "frames_dir": "captures/raw/ai_session/run_1"},
            {"name": "session5", "dir": "precise_session5", "kind": "manual",
             "capture": "captures/raw/manual/session5.jsonl",
             "frames_dir": "captures/raw/manual/session5"},
        ]
        bds.write_manifest(ds, games, "ai_run_1")
        m = json.load(open(os.path.join(ds, "games.json"), encoding="utf-8"))
        assert m["val"] == "ai_run_1" and len(m["games"]) == 2

        ds_posix = ds.replace(os.sep, "/")
        crops = tc.dataset_data_specs(ds, "crops")
        assert crops == [
            ("ai_run_1", f"{ds_posix}/ai_run_1/crops", "captures/raw/ai_session/run_1.jsonl"),
            ("session5", f"{ds_posix}/precise_session5/crops", "captures/raw/manual/session5.jsonl"),
        ], crops
        yolo = bdd.dataset_data_specs(ds)
        assert yolo[0] == ("ai_run_1", f"{ds_posix}/ai_run_1/yolo",
                           "captures/raw/ai_session/run_1.jsonl")
        # tuples on purpose: an absolute ds_dir has a Windows drive colon, which the
        # NAME=DIR:CAPTURE string form's ':' split would break on
        assert yolo[1][1].endswith("precise_session5/yolo")


def test_missing_manifest_is_a_clear_error():
    with tempfile.TemporaryDirectory() as td:
        for call in (lambda d: tc.dataset_data_specs(d, "crops"),
                     lambda d: bdd.dataset_data_specs(d)):
            try:
                call(os.path.join(td, "nope"))
                raise AssertionError("missing games.json not detected")
            except SystemExit as e:
                assert "games.json" in str(e)


def test_apply_existing_dirs_preserves_v1_style_prefixes():
    """Resuming into a dataset whose games.json maps names to prefixed dirs (v1's
    hand-moved precise_*) must keep those dirs; new games keep the plain name."""
    with tempfile.TemporaryDirectory() as td:
        bds.write_manifest(td, [{"name": "ai_run_1", "dir": "precise_ai_run_1", "kind": "ai",
                                 "capture": "c", "frames_dir": "f"}], "ai_run_1")
        games = [{"name": "ai_run_1", "dir": "ai_run_1", "kind": "ai",
                  "capture": "c", "frames_dir": "f"},
                 {"name": "ai_run_15_game1", "dir": "ai_run_15_game1", "kind": "ai",
                  "capture": "c2", "frames_dir": "f2"}]
        out = bds.apply_existing_dirs(games, td)
        assert out[0]["dir"] == "precise_ai_run_1"
        assert out[1]["dir"] == "ai_run_15_game1"


def test_dataset_root_naming():
    assert bds.dataset_root("v2") == os.path.join("datasets", "v2")
    assert bds.dataset_root("datasets/v2") == "datasets/v2"
    assert bds.dataset_root("scratch/x") == "scratch/x"


def test_resolve_parallelism_shared_and_overrides():
    """`-j/--parallel` is the shared default for both heavy stages; an explicit
    per-stage --workers/--jobs overrides it. workers stays None when nothing is
    given (annotate then uses its own cap); jobs always lands on a concrete int."""
    # nothing set: workers falls through to annotate's own default (None), jobs -> DEFAULT_JOBS
    assert bds.resolve_parallelism(None, None, None) == (None, bds.DEFAULT_JOBS)
    # -j alone drives BOTH stages
    assert bds.resolve_parallelism(12, None, None) == (12, 12)
    # explicit per-stage flags override -j on their stage only
    assert bds.resolve_parallelism(12, 16, 6) == (16, 6)
    assert bds.resolve_parallelism(12, None, 6) == (12, 6)
    # legacy invocation (no -j) still honours explicit --workers/--jobs
    assert bds.resolve_parallelism(None, 16, 12) == (16, 12)
    # a per-stage flag with no -j leaves the other stage on its own default
    assert bds.resolve_parallelism(None, 16, None) == (16, bds.DEFAULT_JOBS)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_build_datasets OK")
