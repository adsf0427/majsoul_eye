"""Dependency-light tests for the detector data-prep + wrapper (mirrors
test_classifier.py). The heavy pieces (ultralytics training, a trained weight)
are exercised end-to-end by the Milestone runs, not here; the model smoke test
below is skipped unless ultralytics AND a weight file are present.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "train"))
import build_detector_dataset as bdd  # noqa: E402

from majsoul_eye.tiles import TILE_NAMES, NUM_CLASSES  # noqa: E402


def _parse_names(yaml_text):
    """Tiny parser: pull the `  <i>: '<name>'` lines back into an index->name dict."""
    names, in_names = {}, False
    for line in yaml_text.splitlines():
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if not line.startswith("  "):
                break
            k, v = line.strip().split(":", 1)
            names[int(k)] = v.strip().strip("'")
    return names


def test_data_yaml_names_match_frozen_taxonomy():
    txt = bdd.build_data_yaml_text("datasets/detector")
    assert f"nc: {NUM_CLASSES}" in txt
    names = _parse_names(txt)
    assert len(names) == NUM_CLASSES == 38
    # order MUST match tiles.TILE_NAMES (== NAME_TO_ID used at label export)
    assert [names[i] for i in range(NUM_CLASSES)] == list(TILE_NAMES)


def _make_fake_game(root, name, seqs):
    """Create <root>/<name>/images/<seq>.png stubs; return the yolo dir."""
    ydir = os.path.join(root, name, "yolo")
    os.makedirs(os.path.join(ydir, "images"))
    for s in seqs:
        open(os.path.join(ydir, "images", f"{s:06d}.png"), "w").close()
    return ydir


def test_split_kyoku_holdout_no_leakage():
    with tempfile.TemporaryDirectory() as tmp:
        ydir = _make_fake_game(tmp, "g", [1, 2, 3, 4])
        # seq 1,2 -> kyoku E1.0 (held out); seq 3,4 -> S1.0 (train)
        stub = {1: "E1.0", 2: "E1.0", 3: "S1.0", 4: "S1.0"}
        sources = {"g": (ydir, "unused-capture")}
        train, val = bdd.split_images(sources, "g", {"E1.0"}, kyoku_fn=lambda c: stub)
        base = lambda ps: {os.path.basename(p) for p in ps}
        assert base(val) == {"000001.png", "000002.png"}
        assert base(train) == {"000003.png", "000004.png"}
        assert not (base(train) & base(val))          # disjoint


def test_split_whole_game_holdout():
    with tempfile.TemporaryDirectory() as tmp:
        g1 = _make_fake_game(tmp, "g1", [1, 2])
        g2 = _make_fake_game(tmp, "g2", [1, 2, 3])
        sources = {"g1": (g1, "cap1"), "g2": (g2, "cap2")}
        # val = whole g2; kyoku_fn must NOT be consulted for '*'
        def boom(_):
            raise AssertionError("kyoku_fn called for whole-game '*' holdout")
        train, val = bdd.split_images(sources, "g2", "*", kyoku_fn=boom)
        assert len(val) == 3 and len(train) == 2
        assert all("g2" in p for p in val)
        assert all("g1" in p for p in train)


def test_split_paths_are_absolute_native():
    with tempfile.TemporaryDirectory() as tmp:
        ydir = _make_fake_game(tmp, "g", [5])
        train, val = bdd.split_images({"g": (ydir, "c")}, "", set())
        assert len(train) == 1 and not val
        # ultralytics recovers labels by swapping the 'images' path segment, so the
        # path must be absolute and use the native separator.
        assert os.path.isabs(train[0])
        assert (os.sep + "images" + os.sep) in train[0]


def test_parse_data_arg():
    name, ydir, cap = bdd.parse_data_arg("g1=datasets/precise_ai_run_3_game1/yolo:cap.jsonl")
    assert name == "g1"
    assert ydir == "datasets/precise_ai_run_3_game1/yolo"
    assert cap == "cap.jsonl"


def test_detector_wrapper_smoke():
    """Only runs if ultralytics + a trained weight exist; otherwise skipped."""
    weights = "majsoul_eye/recognize/tile_detector.pt"
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("  (skip wrapper smoke: ultralytics not installed)")
        return
    if not os.path.exists(weights):
        print(f"  (skip wrapper smoke: {weights} not trained yet)")
        return
    import numpy as np
    from majsoul_eye.recognize.detector import TileDetector
    det = TileDetector(weights)
    dets = det.predict(np.zeros((1080, 1920, 3), np.uint8))
    assert isinstance(dets, list)
    for d in dets:
        assert d.tile in TILE_NAMES and 0.0 <= d.score <= 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_detector OK")
