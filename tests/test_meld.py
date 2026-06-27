import numpy as np

from majsoul_eye.normalize import locate_fullscreen
from majsoul_eye.state.replay import BoardState, Meld
from majsoul_eye.label.meld import label_meld, _flatten


def test_flatten_reverse_vs_chrono():
    melds = [Meld("pon", 3, ["P", "P", "P"]), Meld("chi", 0, ["4s", "5sr", "6s"])]
    assert _flatten(melds, "reverse") == ["4s", "5sr", "6s", "P", "P", "P"]   # self: newest-first
    assert _flatten(melds, "chrono") == ["P", "P", "P", "4s", "5sr", "6s"]


def test_flatten_ankan_shows_backs():
    melds = [Meld("ankan", 3, ["2p", "2p", "2p", "2p"])]
    assert _flatten(melds, "chrono") == ["back", "2p", "2p", "back"]


def test_label_meld_self_inbounds():
    s = BoardState(hero_seat=1)
    s.melds[1] = [Meld("pon", 0, ["P", "P", "P"])]
    frame = np.zeros((2160, 3840, 3), np.uint8)
    samples, ok = label_meld(locate_fullscreen(frame), s, "self")
    assert ok and [x.label for x in samples] == ["P", "P", "P"]
    assert all(x.zone == "meld" and x.tile_class is not None for x in samples)
    for x in samples:
        x0, y0, x1, y1 = x.px_box
        assert 0 <= x0 < x1 <= 3840 and 0 <= y0 < y1 <= 2160


def test_label_meld_empty():
    s = BoardState(hero_seat=1)
    samples, ok = label_meld(locate_fullscreen(np.zeros((2160, 3840, 3), np.uint8)), s, "self")
    assert samples == [] and ok


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_meld OK")
