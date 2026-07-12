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


def test_discover_games_skips_frameless_aborted_captures():
    """A capture whose frames.jsonl exists but holds ZERO entries is an aborted
    run (autoplay killed before any frame was recorded — e.g. ai_session_3p
    run_2/game25): it can never be annotated, and stage 3 would reject its empty
    yolo dir as a poisoned split, so discovery drops it up front. A capture with
    NO frames.jsonl at all stays discoverable (legacy shapes; later stages
    decide)."""
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "ai_session_3p")
        good = os.path.join(root, "run_2", "game24")
        _touch(os.path.join(good, "game24.jsonl"))
        with open(os.path.join(good, "frames.jsonl"), "w", encoding="utf-8") as f:
            f.write('{"seq": 1, "status": "ok", "file": "frames/000001.png"}\n')
        aborted = os.path.join(root, "run_2", "game25")
        _touch(os.path.join(aborted, "game25.jsonl"))
        open(os.path.join(aborted, "frames.jsonl"), "w").close()      # 0 entries
        no_index = os.path.join(root, "run_2", "game26")
        _touch(os.path.join(no_index, "game26.jsonl"))                # no frames.jsonl
        names = {g["name"] for g in bds.discover_games([root])}
        assert names == {"ai_session_3p_run_2_game24",
                         "ai_session_3p_run_2_game26"}, names


def test_discover_games_excludes_named_games():
    """``--exclude NAME`` drops a discoverable-but-unusable game for good: the capture
    stays on disk (and keeps its GT), but never enters the manifest, so a later
    ``--resume`` cannot silently pull it back in. Motivating case:
    ai_session_3p_run_3_game11, a game the client played disconnected — a modal covers
    the table, so GT-driven river boxes land on nothing (dropped as unreliable) while
    meld boxes land on the WALL and pass the fill gate (emitted as phantom labels).
    An unknown --exclude name is an error, not a no-op: a typo'd name would silently
    leave the poisoned game in the split."""
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "ai_session_3p")
        for g in ("game10", "game11"):
            _touch(os.path.join(root, "run_3", g, f"{g}.jsonl"))
        names = {g["name"] for g in bds.discover_games([root])}
        assert names == {"ai_session_3p_run_3_game10", "ai_session_3p_run_3_game11"}

        kept = bds.discover_games([root], exclude=["ai_session_3p_run_3_game11"])
        assert [g["name"] for g in kept] == ["ai_session_3p_run_3_game10"]

        try:
            bds.discover_games([root], exclude=["ai_session_3p_run_3_game99"])
        except SystemExit:
            pass
        else:
            raise AssertionError("an --exclude name matching no discovered game must fail")


def test_letterboxed_games_use_own_frames_now():
    """run_5 game2/game3 were de-letterboxed IN PLACE (2026-07-05, deletterbox_frames.py
    --inplace), so the FRAMES_OVERRIDE map is gone and they resolve to their own nested
    frames dir like every other game — no derived-frames special-casing."""
    assert not hasattr(bds, "FRAMES_OVERRIDE"), "override should be fully removed"
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "ai_session")
        _touch(os.path.join(root, "run_5", "game2", "game2.jsonl"))
        (g,) = bds.discover_games([root])
        assert g["name"] == "ai_session_run_5_game2"
        assert g["frames_dir"].endswith("run_5/game2")
        assert "derived" not in g["frames_dir"]


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
        bds.write_manifest(ds, games, ["ai_run_1"])
        m = json.load(open(os.path.join(ds, "games.json"), encoding="utf-8"))
        assert m["val"] == ["ai_run_1"] and len(m["games"]) == 2   # val is a LIST (>=1 held-out game)

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


def test_resolve_vals_default_single_and_multiple():
    """--val is repeatable. None falls back to [DEFAULT_VAL] when it's a discovered
    game; one or many explicit names are validated against the game set (order kept)."""
    names = ["ai_session_run_8_game1", "ai_session2_run_21_game1", "g3"]
    # no --val -> default held-out game (as a one-element list)
    assert bds.resolve_vals(None, names) == [bds.DEFAULT_VAL]
    # explicit single
    assert bds.resolve_vals(["g3"], names) == ["g3"]
    # explicit multiple (the ask: add a second whole-game val), order preserved
    assert bds.resolve_vals(["ai_session_run_8_game1", "ai_session2_run_21_game1"], names) \
        == ["ai_session_run_8_game1", "ai_session2_run_21_game1"]


def test_resolve_vals_rejects_unknown_and_missing_default():
    # an explicit name not among discovered games aborts (names them)
    try:
        bds.resolve_vals(["ai_session_run_8_game1", "typo_game"], ["ai_session_run_8_game1"])
        raise AssertionError("unknown val game not rejected")
    except SystemExit as e:
        assert "typo_game" in str(e)
    # no --val AND default absent from the game set -> abort (must pass one)
    try:
        bds.resolve_vals(None, ["g1", "g2"])
        raise AssertionError("missing default not rejected")
    except SystemExit as e:
        assert "--val" in str(e)


def test_classifier_parse_val_specs_multiple():
    """train_classifier grows the same repeatable --val -> {name: val_set} parser."""
    assert tc.parse_val_specs(["a:*", "b:E1.0"]) == {"a": "*", "b": {"E1.0"}}
    assert tc.parse_val_specs([]) == {}


def test_resolve_formats():
    """--hbb/--obb fold into an ordered format list. Backward compatible: neither flag
    -> ['hbb'] (historical default); --obb alone -> ['obb'] (historical OBB-only); both
    -> ['hbb','obb'] (one version carrying detector/ + detector_obb/)."""
    assert bds.resolve_formats(False, False) == ["hbb"]
    assert bds.resolve_formats(False, True) == ["obb"]
    assert bds.resolve_formats(True, False) == ["hbb"]
    assert bds.resolve_formats(True, True) == ["hbb", "obb"]


def test_game_yolo_dir_dual_vs_single():
    """OBB gets the sibling '<game>__obb' dir ONLY when it coexists with HBB (so both
    formats can live in one version); a single-format build keeps the plain '<game>'
    dir — preserving today's OBB-only layout (and its crops path for the classifier)."""
    j = os.path.join
    assert bds.game_yolo_dir("datasets/v2", "g", "hbb", ["hbb"]) == j("datasets/v2", "g", "yolo")
    assert bds.game_yolo_dir("datasets/v2", "g", "obb", ["obb"]) == j("datasets/v2", "g", "yolo")
    assert bds.game_yolo_dir("datasets/v2", "g", "hbb", ["hbb", "obb"]) == j("datasets/v2", "g", "yolo")
    assert bds.game_yolo_dir("datasets/v2", "g", "obb", ["hbb", "obb"]) == j("datasets/v2", "g__obb", "yolo")


def test_symlink_reuse_images_link_is_safe_and_traversable():
    """The dual-build OBB `images` link (Runner.symlink) must, on BOTH platforms:
    (1) resolve to the HBB frames, (2) survive a --resume re-run without ever deleting
    the shared HBB frames it points at, and (3) create nothing under --dry-run. On
    Windows-without-symlink-privilege this exercises the directory-junction fallback
    (STATUS §1.42); on POSIX it's a plain relative symlink."""
    import glob
    with tempfile.TemporaryDirectory() as ds:
        hbb = os.path.join(ds, "game", "yolo", "images")
        os.makedirs(hbb)
        frame = os.path.join(hbb, "000001.png")
        with open(frame, "w") as f:
            f.write("frame")
        link = os.path.join(ds, "game__obb", "yolo", "images")

        r = bds.Runner(execute=True)
        r.symlink(hbb, link)
        assert glob.glob(os.path.join(link, "*.png")), "link does not traverse to HBB frame"

        # --resume re-run: the link is replaced in place, HBB frame MUST survive
        r.symlink(hbb, link)
        assert os.path.exists(frame), "re-run destroyed the shared HBB frame!"
        assert glob.glob(os.path.join(link, "*.png"))

        # dry-run touches nothing
        link2 = os.path.join(ds, "game3__obb", "yolo", "images")
        bds.Runner(execute=False).symlink(hbb, link2)
        assert not os.path.exists(link2)


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
