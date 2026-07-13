"""The nukidora pile: FORWARD annotate geometry -> detections -> assemble must
invert it exactly. No detector, no images.

Why this needs its own guard: the pulled norths are ordinary `N` detections (the
detector has no separate class for them), and their lane is adjacent to the meld
strip BY CONSTRUCTION — sanma's SELF meld corner sits 46px further inward than 4P's
precisely because the north lane took the 4P corner. So an N that gets routed one
zone over does not raise; it silently becomes a bogus meld cell and takes the whole
strip parse down with it.
"""
import numpy as np

from majsoul_eye.annotate import pipeline as P
from majsoul_eye.normalize import BoardRegion
from majsoul_eye.recognize.assemble import _fw_points, assemble
from majsoul_eye.recognize.detector import Detection
from majsoul_eye.tiles import NAME_TO_ID

H = P.build_homographies(1920, 1080)
REGION = BoardRegion(0, 0, 1920, 1080)
G3 = P.GEOMETRY_3P

# A sanma-legal 13-tile hand.
HAND = ["1m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "1s", "2s"]


def _det(poly, tile):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return Detection(xyxy=(min(xs), min(ys), max(xs), max(ys)), name=tile, tile=tile,
                     cls=NAME_TO_ID[tile], score=0.9,
                     poly=tuple((float(x), float(y)) for x, y in poly))


def _box_dets(box, tile):
    return _det([[box.x0 * 1920, box.y0 * 1080], [box.x1 * 1920, box.y0 * 1080],
                 [box.x1 * 1920, box.y1 * 1080], [box.x0 * 1920, box.y1 * 1080]], tile)


def _hand_dets(tiles, drawn=None):
    from majsoul_eye.coords import HAND as HAND_MODEL
    dets = [_box_dets(HAND_MODEL.slot_box(i), t) for i, t in enumerate(tiles)]
    if drawn:
        dets.append(_box_dets(HAND_MODEL.slot_box(len(tiles), is_tsumo=True), drawn))
    return dets


def _dora_dets(tiles):
    from majsoul_eye.coords import dora_slot
    return [_box_dets(dora_slot(i), t) for i, t in enumerate(tiles)]


def _nuki_dets(seat, count, along_offset=0.0):
    boxes = P.generate_nukidora_boxes(seat, count, H["H_full_inv"],
                                      along_offset=along_offset)
    return [_det(b["poly_original"], b["tile"]) for b in boxes]


def _board(nuki_by_seat, along_offset=0.0):
    dets = _hand_dets(HAND) + _dora_dets(["5s"])
    for seat, k in enumerate(nuki_by_seat):
        dets += _nuki_dets(seat, k, along_offset)
    return assemble(dets, REGION, geom=G3, phantom_rel=3)


def test_pile_sizes_recovered_for_every_seat():
    for seat in (0, 1, 2):
        for k in (1, 2, 3, 4):
            piles = [0, 0, 0, 0]
            piles[seat] = k
            o = _board(piles)
            assert o.violations == [], (seat, k, o.violations)
            assert o.nukidora == piles, (seat, k, o.nukidora)


def test_a_pulled_north_never_lands_in_the_meld_strip():
    # The failure this guards: an N absorbed by the adjacent meld strip becomes a
    # bogus cell, and the strip parse dies with "meld strip unparsable".
    # Only four norths exist, so all three live piles at once means 2+1+1.
    o = _board([2, 1, 1, 0])
    assert o.violations == [], o.violations
    assert [len(ms) for ms in o.melds] == [0, 0, 0, 0], "a north leaked into a meld"
    assert [len(r) for r in o.rivers] == [0, 0, 0, 0], "a north leaked into a river"
    assert o.nukidora == [2, 1, 1, 0]


def test_more_norths_than_exist_is_rejected_not_rendered():
    # 12 pulled norths is not a board; it is a mis-detection. The 4-copy budget
    # (which counts the piles) must catch it rather than hand back a confident,
    # coherent, impossible position.
    o = _board([4, 4, 4, 0])
    assert any("N seen 12>4" in v for v in o.violations), o.violations


def test_the_self_pile_floats_and_is_still_read():
    # NUKI_STRIP_3P documents the SELF pile drifting ±12px along its row per round.
    for offset in (-12.0, -6.0, 0.0, 6.0, 12.0):
        o = _board([3, 0, 0, 0], along_offset=offset)
        assert o.violations == [], (offset, o.violations)
        assert o.nukidora[0] == 3, (offset, o.nukidora)


def test_four_player_boards_have_no_lane_at_all():
    # With 4P geometry the strip is None, so an N is routed as an ordinary tile.
    # Nothing may end up in nukidora.
    dets = _hand_dets(HAND) + _dora_dets(["5s"])
    o = assemble(dets, REGION, geom=P.GEOMETRY_4P)
    assert o.nukidora == [0, 0, 0, 0]
    assert o.sanma is False and o.phantom_rel is None


def test_assemble_stamps_the_mode_onto_the_state():
    o = _board([1, 0, 0, 0])
    assert o.sanma is True and o.phantom_rel == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_nuki_lane OK")
