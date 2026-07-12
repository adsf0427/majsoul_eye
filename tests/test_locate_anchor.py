"""The landmark localizer: recover (scale, offset) from detections alone.

Synthetic landmarks, so the exact answer is known. The real-screenshot proof
lives in the samples/ golden (tests/test_samples_golden.py).
"""
import numpy as np

from majsoul_eye.coords import HAND, HUD_SEEDS
from majsoul_eye.normalize import (
    CANON_W, CANON_H, PANEL_LANDMARKS, BoardRegion, clipped_sides, locate_anchor,
)


class Det:
    def __init__(self, xyxy, name, tile=None, score=0.9):
        self.xyxy, self.name, self.tile, self.score = xyxy, name, tile, score


def place(box, k, ox, oy):
    """Canonical NormBox -> image-px xyxy under img = k * canon + (ox, oy)."""
    return (box.x0 * CANON_W * k + ox, box.y0 * CANON_H * k + oy,
            box.x1 * CANON_W * k + ox, box.y1 * CANON_H * k + oy)


def scene(k=1.0, ox=0.0, oy=0.0, *, concealed=13, drawn=True, panel=True):
    dets = []
    if panel:
        dets += [Det(place(HUD_SEEDS[name], k, ox, oy), name)
                 for name in PANEL_LANDMARKS]
    for i in range(concealed):
        dets.append(Det(place(HAND.slot_box(i), k, ox, oy), "1m", tile="1m"))
    if drawn:
        dets.append(Det(place(HAND.slot_box(concealed, is_tsumo=True), k, ox, oy),
                        "2p", tile="2p"))
    return dets


def frame(w, h):
    return np.zeros((h, w, 3), np.uint8)


def test_recovers_a_clean_fullscreen_board():
    got = locate_anchor(frame(1920, 1080), scene())
    assert got is not None
    assert (got.region.ox, got.region.oy) == (0, 0)
    assert (got.region.bw, got.region.bh) == (1920, 1080)
    assert got.residual < 1e-6


def test_recovers_a_wide_phone_board_the_aspect_gate_used_to_reject():
    # 2868x1320 phone: the 16:9 table is centered, HUD columns either side.
    k = 1320 / CANON_H
    dets = scene(k=k, ox=(2868 - k * CANON_W) / 2)
    got = locate_anchor(frame(2868, 1320), dets)
    assert got is not None
    assert abs(got.region.bw - round(k * CANON_W)) <= 1
    assert abs(got.region.ox - 260) <= 2
    assert got.hand_inliers >= 13


def test_recovers_a_board_inset_under_browser_chrome():
    got = locate_anchor(frame(1700, 1050), scene(k=0.75, ox=110, oy=140))
    assert got is not None
    assert (got.region.ox, got.region.oy) == (110, 140)
    assert (got.region.bw, got.region.bh) == (1440, 810)


def test_a_screen_anchored_stray_is_outvoted_not_averaged_in():
    """The dora strip does not move with the board on a phone. If such a landmark
    ever leaked into the fit it must be rejected as an outlier, not absorbed."""
    dets = scene(k=0.8, ox=200, oy=0)
    liar = Det((10.0, 10.0, 60.0, 60.0), "round_label")   # nowhere near the panel
    got = locate_anchor(frame(2200, 900), dets + [liar])
    assert got is not None
    assert (got.region.ox, got.region.oy) == (200, 0)
    assert got.inliers == len(dets)                        # the liar, and only it, is out


def test_hand_row_alone_still_pins_the_board():
    """The panel is a 270/1920 baseline; the hand is what carries the scale."""
    got = locate_anchor(frame(1920, 1080), scene(panel=False))
    assert got is not None
    assert (got.region.bw, got.region.bh) == (1920, 1080)
    assert got.panel_inliers == 0


def test_panel_alone_is_reported_as_having_no_hand_evidence():
    """Cropping the hand row away leaves a fit the runtime must NOT trust: the
    caller gates on hand_inliers, because panel-only scale is ill-conditioned."""
    got = locate_anchor(frame(1920, 1080), scene(concealed=0, drawn=False))
    assert got is not None
    assert got.hand_inliers == 0


def test_a_short_hand_still_anchors_left():
    """Melds shorten the hand from the right; slot 0 does not move."""
    got = locate_anchor(frame(1920, 1080), scene(concealed=7))
    assert got is not None
    assert (got.region.ox, got.region.bw) == (0, 1920)


def test_too_few_landmarks_is_a_refusal_not_a_guess():
    assert locate_anchor(frame(1920, 1080), []) is None
    assert locate_anchor(frame(1920, 1080), scene(concealed=1, drawn=False,
                                                  panel=False)) is None


def test_clipped_sides_names_every_cropped_edge():
    full = BoardRegion(0, 0, 1920, 1080, 1920, 1080)
    assert clipped_sides(full) == []
    assert clipped_sides(BoardRegion(-40, 0, 1920, 1080, 1920, 1080)) == ["left"]
    assert clipped_sides(BoardRegion(0, 0, 1920, 1080, 1700, 1080)) == ["right"]
    assert clipped_sides(BoardRegion(0, -30, 1920, 1080, 1920, 1080)) == ["top"]
    assert clipped_sides(BoardRegion(0, 0, 1920, 1080, 1920, 900)) == ["bottom"]
    # Sub-percent fit jitter is not a crop.
    assert clipped_sides(BoardRegion(-4, -4, 1920, 1080, 1920, 1080)) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_locate_anchor OK")
