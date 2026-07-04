import tempfile

import numpy as np

from majsoul_eye.coords import HAND, NormBox, px_box, dora_slot, DORA_STRIP, MAX_DORA
from majsoul_eye.normalize import locate_fullscreen, locate_letterbox
from majsoul_eye.state.replay import BoardState
from majsoul_eye.label import label_frame, to_yolo_lines, save_classification_crops


def _state():
    s = BoardState(hero_seat=0)
    s.bakaze, s.kyoku, s.honba, s.kyotaku = "E", 1, 0, 0
    s.scores = [25000, 24000, 26000, 25000]
    s.dora_markers = ["1m", "E"]
    s.hero_hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]
    return s


def test_normbox_pixels_and_yolo():
    b = px_box(0, 0, 1920, 1080)
    assert b.to_pixels(1920, 1080) == (0, 0, 1920, 1080)
    half = NormBox(0.25, 0.5, 0.75, 1.0)
    assert half.to_pixels(100, 100) == (25, 50, 75, 100)
    assert half.yolo() == (0.5, 0.75, 0.5, 0.5)


def test_hand_slots_monotonic_inbounds():
    xs = [HAND.slot_box(i).x0 for i in range(13)]
    assert all(xs[i] < xs[i + 1] for i in range(12))
    for i in range(13):
        b = HAND.slot_box(i)
        assert 0 <= b.x0 < b.x1 <= 1 and 0 <= b.y0 < b.y1 <= 1


def test_dora_slots_partition_strip():
    assert abs(dora_slot(0).x0 - DORA_STRIP.x0) < 1e-9
    assert abs(dora_slot(MAX_DORA - 1).x1 - DORA_STRIP.x1) < 1e-6


ALL_ZONES = frozenset({"hand", "dora", "score", "meta"})


def test_label_frame_easy_zones():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    region = locate_fullscreen(frame)
    s = _state()
    samples = label_frame(frame, s, region, zones=ALL_ZONES)
    hand = [x for x in samples if x.zone == "hand"]
    dora = [x for x in samples if x.zone == "dora"]
    score = [x for x in samples if x.zone == "score"]
    assert len(hand) == 13 and [x.label for x in hand] == s.hero_hand
    assert all(x.tile_class is not None for x in hand)
    assert len(dora) == 2 and len(score) == 3
    for x in samples:                       # all boxes inside the frame
        x0, y0, x1, y1 = x.px_box
        assert 0 <= x0 < x1 <= 1920 and 0 <= y0 < y1 <= 1080
    yolo = to_yolo_lines(samples)
    assert len(yolo) == len(hand) + len(dora)
    assert all(len(line.split()) == 5 for line in yolo)


def test_label_skips_hand_with_untracked_draw():
    # 14 tiles but no tracked drawn tile (e.g. a hand-built state): geometry of the
    # separated slot is unknown, so the hand is still skipped rather than mislabeled.
    s = _state()
    s.hero_hand = s.hero_hand + ["5p"]      # 14 tiles, drawn_tile stays None
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    samples = label_frame(frame, s, locate_fullscreen(frame))
    assert not any(x.zone == "hand" for x in samples)


def test_label_hand_with_tracked_tsumo():
    # Hero's own turn: the tracked drawn tile is labeled in the separated tsumo slot
    # (gapped, right of the 13 sorted concealed tiles). This is the fix for the
    # detector suppressing the hero hand on the player's own turn.
    from majsoul_eye.state.replay import _sort_hand
    s = _state()                                     # 13-tile sorted concealed hand
    concealed = list(s.hero_hand)
    s.hero_hand = _sort_hand(s.hero_hand + ["5p"])   # 14 sorted (drawn 5p merged in)
    s.drawn_tile = "5p"
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    hand = [x for x in label_frame(frame, s, locate_fullscreen(frame)) if x.zone == "hand"]
    assert len(hand) == 14                            # 13 concealed + the drawn tile
    # 13 concealed occupy slots 0..12 in sorted order
    assert [x.label for x in hand[:13]] == concealed
    for i, x in enumerate(hand[:13]):
        assert abs(x.norm_box.x0 - HAND.slot_box(i).x0) < 1e-9
    # the drawn tile sits in the gapped tsumo slot, to the right of a normal slot 13
    drawn = hand[13]
    assert drawn.label == "5p"
    assert abs(drawn.norm_box.x0 - HAND.slot_box(13, is_tsumo=True).x0) < 1e-9
    assert drawn.norm_box.x0 > HAND.slot_box(13).x0   # the tsumo gap pushed it right
    assert all(x.tile_class is not None for x in hand)


def test_letterbox_trims_black_border():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[100:900, 200:1700] = 200
    r = locate_letterbox(frame)
    assert (r.ox, r.oy, r.bw, r.bh) == (200, 100, 1500, 800)


def test_save_classification_crops():
    frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    region = locate_fullscreen(frame)
    samples = label_frame(frame, _state(), region, zones=ALL_ZONES)
    with tempfile.TemporaryDirectory() as d:
        n = save_classification_crops(frame, region, samples, d)
        assert n == 15  # 13 hand + 2 dora tiles


def test_default_zone_is_hand_only():
    # 河/副露 moved to the precise fullwarp pipeline (majsoul_eye.annotate); this
    # legacy annotator now defaults to the hand zone only (dora/score/meta opt-in).
    from majsoul_eye.state.replay import RiverTile
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    s = _state()
    s.rivers[1] = [RiverTile("1m"), RiverTile("2m")]   # someone has discards
    samples = label_frame(frame, s, locate_fullscreen(frame))
    zones = {x.zone for x in samples}
    assert zones == {"hand"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_label OK")
