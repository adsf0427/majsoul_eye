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
import train_detector as td  # noqa: E402
import build_dataset as bd  # noqa: E402

from majsoul_eye.tiles import TILE_NAMES, NUM_CLASSES  # noqa: E402
from majsoul_eye.hud import DET_NAMES  # noqa: E402


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
    # Task 9: the detector head grew from the frozen 38 tile classes to the full
    # 56-class hud.DET_NAMES (38 tiles + 17 HUD elements + 1 reach_stick, Task
    # 17a/17c -- reach stick revised to a single symmetric class, spec §10);
    # v1 (pre-HUD) label files only ever used ids 0-37, so the tile
    # prefix/order must stay frozen.
    txt = bdd.build_data_yaml_text("datasets/detector")
    assert f"nc: {len(DET_NAMES)}" in txt
    names = _parse_names(txt)
    assert len(names) == len(DET_NAMES) == 56
    assert [names[i] for i in range(len(DET_NAMES))] == list(DET_NAMES)
    # tile prefix (ids 0..37) MUST match tiles.TILE_NAMES (== NAME_TO_ID used at
    # label export) — this is what keeps old 38-class labels valid under the
    # 59-class head.
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
        train, val = bdd.split_images(sources, {"g": {"E1.0"}}, kyoku_fn=lambda c: stub)
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
        train, val = bdd.split_images(sources, {"g2": "*"}, kyoku_fn=boom)
        assert len(val) == 3 and len(train) == 2
        assert all("g2" in p for p in val)
        assert all("g1" in p for p in train)


def test_parse_val_specs_multiple():
    """--val is repeatable: each 'NAME:spec' folds into a {name: val_set} map.
    ':*' -> whole game ('*'); ':k1,k2' -> a kyoku set; '' / None -> empty map."""
    assert bdd.parse_val_specs(["g1:*", "g2:E1.0,S2.0"]) == {"g1": "*", "g2": {"E1.0", "S2.0"}}
    assert bdd.parse_val_specs(["g:*"]) == {"g": "*"}          # single still works
    assert bdd.parse_val_specs([]) == {}
    assert bdd.parse_val_specs(None) == {}


def test_split_multiple_whole_games():
    """Two whole games held out at once: BOTH go entirely to val, the rest to train,
    and kyoku_fn is never consulted for a '*' game."""
    with tempfile.TemporaryDirectory() as tmp:
        g1 = _make_fake_game(tmp, "g1", [1, 2])
        g2 = _make_fake_game(tmp, "g2", [1, 2, 3])
        g3 = _make_fake_game(tmp, "g3", [1, 2])
        sources = {"g1": (g1, "c1"), "g2": (g2, "c2"), "g3": (g3, "c3")}
        def boom(_):
            raise AssertionError("kyoku_fn called for a whole-game '*' holdout")
        train, val = bdd.split_images(sources, {"g1": "*", "g2": "*"}, kyoku_fn=boom)
        assert len(val) == 5 and len(train) == 2                 # g1(2)+g2(3) val, g3(2) train
        assert all(("g1" in p or "g2" in p) for p in val)
        assert all("g3" in p for p in train)


def test_split_paths_are_relative_posix():
    # create the fake game UNDER the repo (relative yolodir in) so the output is a
    # portable relative path; ultralytics recovers labels by swapping the '/images/'
    # segment (forward slashes; it normalizes to os.sep at load).
    d = "datasets/_bdd_test_tmp"
    try:
        ydir = _make_fake_game(d, "g", [5])
        train, val = bdd.split_images({"g": (ydir, "c")}, {})
        assert len(train) == 1 and not val
        assert not os.path.isabs(train[0])          # relative → tar-and-go portable
        assert "/images/" in train[0] and "\\" not in train[0]
    finally:
        import shutil
        shutil.rmtree("datasets/_bdd_test_tmp", ignore_errors=True)


def test_parse_data_arg():
    name, ydir, cap = bdd.parse_data_arg("g1=datasets/precise_ai_run_3_game1/yolo:cap.jsonl")
    assert name == "g1"
    assert ydir == "datasets/precise_ai_run_3_game1/yolo"
    assert cap == "cap.jsonl"


def test_resolve_device():
    # '' = historical auto: GPU 0 when CUDA present, else CPU
    assert td.resolve_device("", True) == 0
    assert td.resolve_device("", False) == "cpu"
    # explicit cpu (case-insensitive)
    assert td.resolve_device("cpu", True) == "cpu"
    assert td.resolve_device("CPU", True) == "cpu"
    # single GPU -> int (keeps the single-device semantics of the old default)
    assert td.resolve_device("0", True) == 0
    assert td.resolve_device("3", True) == 3
    # multi-GPU -> comma string for ultralytics DDP; whitespace tolerated
    assert td.resolve_device("0,1,2,3", True) == "0,1,2,3"
    assert td.resolve_device(" 0, 1 ", True) == "0,1"


def test_box_quad_and_label_lines():
    from types import SimpleNamespace
    # river/meld: ordered perspective quad [TL,TR,BR,BL] passes through verbatim
    qbox = SimpleNamespace(poly_original=[[10, 10], [30, 12], [28, 40], [8, 38]], px_box=None)
    q = bd.box_quad(qbox)
    assert q == [[10.0, 10.0], [30.0, 12.0], [28.0, 40.0], [8.0, 38.0]]
    # OBB = 8 normalized corner coords, same point order
    assert bd.obb_label_line(5, q, 100, 100) == \
        "5 0.100000 0.100000 0.300000 0.120000 0.280000 0.400000 0.080000 0.380000"
    # HBB = axis-aligned bbox of the quad (x:8..30, y:10..40) -> cx cy w h
    assert bd.hbb_label_line(5, q, 100, 100) == "5 0.190000 0.250000 0.220000 0.300000"

    # hand/dora: axis-aligned px_box expands to a rectangle quad (angle 0)
    rbox = SimpleNamespace(poly_original=None, px_box=[10, 20, 50, 60])
    assert bd.box_quad(rbox) == [[10.0, 20.0], [50.0, 20.0], [50.0, 60.0], [10.0, 60.0]]

    # out-of-frame coords clamp to [0,1]
    assert bd.obb_label_line(1, [[-5, 0], [110, 0], [110, 50], [-5, 50]], 100, 100) == \
        "1 0.000000 0.000000 1.000000 0.000000 1.000000 0.500000 0.000000 0.500000"


def test_parse_result_hbb_and_obb():
    import numpy as np
    from types import SimpleNamespace
    from majsoul_eye.recognize.detector import _parse_result

    # HBB model: detections live in res.boxes; no obb attr -> poly stays None
    b = SimpleNamespace(cls=np.array([5.]), xyxy=np.array([[1., 2., 3., 4.]]), conf=np.array([0.9]))
    dets = _parse_result(SimpleNamespace(boxes=[b]))
    assert len(dets) == 1
    d = dets[0]
    assert d.cls == 5 and d.tile == TILE_NAMES[5] and d.name == TILE_NAMES[5]
    assert d.xyxy == (1., 2., 3., 4.)
    assert d.score == 0.9 and d.poly is None

    # OBB model: detections live in res.obb; xyxy = enclosing bbox, poly = 4 corners
    o = SimpleNamespace(cls=np.array([7.]), conf=np.array([0.8]),
                        xyxy=np.array([[8., 10., 30., 40.]]),
                        xyxyxyxy=np.array([[[10., 10.], [30., 12.], [28., 40.], [8., 38.]]]))
    dets = _parse_result(SimpleNamespace(obb=[o], boxes=None))
    assert len(dets) == 1
    d = dets[0]
    assert d.cls == 7 and d.tile == TILE_NAMES[7] and d.name == TILE_NAMES[7]
    assert d.xyxy == (8., 10., 30., 40.)
    assert d.poly == ((10., 10.), (30., 12.), (28., 40.), (8., 38.))

    # OBB model with zero detections must NOT fall through to boxes=None (would crash)
    assert _parse_result(SimpleNamespace(obb=[], boxes=None)) == []


def test_parse_result_hud_classes_and_out_of_range_ids():
    """56-class detector head regression (majsoul_eye.hud.DET_NAMES = 38 tiles +
    18 HUD classes, ids 38-55): a HUD-class detection must NOT raise IndexError
    (the historical bug -- `.tile = TILE_NAMES[cls]` unconditionally) and must
    carry `tile=None` + a valid `.name`; a cls id past the end of DET_NAMES
    (unknown/future class) must be skipped, not crash or invent a name."""
    import numpy as np
    from types import SimpleNamespace

    from majsoul_eye.recognize.detector import _parse_result

    assert len(DET_NAMES) == 56
    tile_cls, hud_cls, oor_cls = 5, 40, 99   # 40 is a HUD id; 99 >= len(DET_NAMES)

    def _box(cls, xyxy, conf):
        return SimpleNamespace(cls=np.array([float(cls)]), xyxy=np.array([list(xyxy)]),
                               conf=np.array([conf]))

    res = SimpleNamespace(boxes=[
        _box(tile_cls, (1., 2., 3., 4.), 0.9),
        _box(hud_cls, (5., 6., 7., 8.), 0.8),
        _box(oor_cls, (9., 10., 11., 12.), 0.5),
    ])
    dets = _parse_result(res)

    assert len(dets) == 2, "the out-of-range cls must be skipped, not crash or appear"
    tile_det, hud_det = dets
    assert tile_det.cls == tile_cls and tile_det.tile == TILE_NAMES[tile_cls]
    assert tile_det.name == TILE_NAMES[tile_cls]

    assert hud_det.cls == hud_cls and hud_det.tile is None      # the historical crash site
    assert hud_det.name == DET_NAMES[hud_cls]
    assert all(d.cls != oor_cls for d in dets)

    # same behavior on the OBB path
    def _obb(cls, xyxy, conf):
        return SimpleNamespace(cls=np.array([float(cls)]), conf=np.array([conf]),
                               xyxy=np.array([list(xyxy)]),
                               xyxyxyxy=np.array([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]]))

    res_obb = SimpleNamespace(boxes=None, obb=[
        _obb(hud_cls, (5., 6., 7., 8.), 0.8),
        _obb(oor_cls, (9., 10., 11., 12.), 0.5),
    ])
    dets_obb = _parse_result(res_obb)
    assert len(dets_obb) == 1
    assert dets_obb[0].tile is None and dets_obb[0].name == DET_NAMES[hud_cls]


def test_detector_obb_wrapper_smoke():
    """End-to-end OBB path; skipped unless ultralytics + the OBB weight exist."""
    import glob
    weights = "weights/detector/tile_detector_obb.pt"
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("  (skip obb smoke: ultralytics not installed)")
        return
    if not os.path.exists(weights):
        print(f"  (skip obb smoke: {weights} not trained yet)")
        return
    import cv2
    from majsoul_eye.recognize.detector import TileDetector
    det = TileDetector(weights)
    imgs = glob.glob("datasets/obb_precise_ai_run_8_game1/yolo/images/*.png")
    if not imgs:
        print("  (skip obb smoke: no OBB frames on disk)")
        return
    dets = det.predict(cv2.imread(sorted(imgs)[300]))
    assert isinstance(dets, list) and dets, "OBB model returned no detections"
    for d in dets:
        assert d.tile in TILE_NAMES and 0.0 <= d.score <= 1.0
        assert d.poly is not None and len(d.poly) == 4        # oriented 4-corner box
    print(f"  (obb smoke: {len(dets)} oriented detections)")


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
