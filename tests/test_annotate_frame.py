"""Tests for per-frame annotation orchestration (majsoul_eye.annotate.frame).

annotate_frame's box GEOMETRY comes from GT + calibration (no image needed), so a
black frame exercises structure + the reliability gate (nothing renders -> every
box reliable=False). iter_tile_boxes / crop_box are the seam build_dataset uses.
"""
import numpy as np

from majsoul_eye.state.replay import BoardState, Meld, RiverTile
from majsoul_eye.annotate import build_homographies, annotate_frame, iter_tile_boxes, crop_box
from majsoul_eye.annotate.frame import AnnBox, crop_quad
from majsoul_eye.tiles import NAME_TO_ID

HOM = build_homographies(1920, 1080)


def _state():
    s = BoardState(hero_seat=0, bakaze="E", kyoku=1, honba=0, last_actor=1)
    s.rivers[0] = [RiverTile("1m"), RiverTile("2p")]
    s.melds[0] = [Meld("pon", 1, ["P", "P", "P"], called_pai="P")]
    s.hero_hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]
    s.dora_markers = ["E"]
    return s


def test_annotate_frame_shape_and_reliability():
    rec = annotate_frame(np.zeros((1080, 1920, 3), np.uint8), _state(), HOM)
    assert set(rec) >= {"hero_seat", "kyoku", "discard_slots", "meld_boxes", "hand_boxes", "dora_boxes", "flags"}
    assert rec["hero_seat"] == 0 and rec["kyoku"] == "E1"
    assert set(rec["discard_slots"]) == {"0", "1", "2", "3"}
    assert len(rec["discard_slots"]["0"]) == 2          # self river = rivers[0]
    assert len(rec["meld_boxes"]["0"]) == 3             # pon -> 3 display cells
    # black frame renders nothing -> reliability gate marks every box unreliable
    assert all(not s.get("reliable", True) for s in rec["discard_slots"]["0"])


def test_iter_tile_boxes_zones_and_types():
    rec = annotate_frame(np.zeros((1080, 1920, 3), np.uint8), _state(), HOM)
    boxes = list(iter_tile_boxes(rec))
    assert {"river", "meld", "hand"} <= {b.zone for b in boxes}
    for b in boxes:
        assert b.tile in NAME_TO_ID
        assert (b.poly_original is not None) == (b.zone in ("river", "meld"))
        assert (b.px_box is not None) == (b.zone in ("hand", "dora"))


def test_reliable_propagation():
    rec = {"discard_slots": {"0": [
                {"tile": "1m", "face_poly_original": [[0, 0], [10, 0], [10, 10], [0, 10]], "riichi": False},
                {"tile": "2p", "face_poly_original": [[0, 0], [10, 0], [10, 10], [0, 10]],
                 "riichi": False, "reliable": False}],
            "1": [], "2": [], "3": []},
           "meld_boxes": {"0": [], "1": [], "2": [], "3": []}, "hand_boxes": [], "dora_boxes": []}
    a, b = list(iter_tile_boxes(rec))
    assert a.reliable is True and b.reliable is False
    assert a.sideways is False


def test_crop_box_sizes():
    img = np.zeros((1080, 1920, 3), np.uint8)
    quad = AnnBox("river", "1m", "tile", [[100, 100], [160, 100], [160, 180], [100, 180]], None, False, True)
    px = AnnBox("hand", "1m", "tile", None, [100, 100, 160, 180], False, True)
    assert crop_box(img, quad).shape == (64, 64, 3)
    assert crop_box(img, px).shape == (64, 64, 3)
    assert crop_box(img, quad, size=96).shape == (96, 96, 3)
    assert crop_quad(img, quad.poly_original, 48).shape == (48, 48, 3)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_annotate_frame OK")
