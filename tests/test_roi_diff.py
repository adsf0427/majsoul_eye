import numpy as np
from majsoul_eye.capture.roi_diff import roi_diff, STABILITY_ROIS, TABLE_ROI


def test_identical_is_zero():
    a = np.random.RandomState(0).randint(0, 255, (90, 160, 3), np.uint8)
    assert roi_diff(a, a) == 0.0
    print("test_identical_is_zero OK")


def test_change_outside_all_rois_ignored():
    a = np.zeros((900, 1600, 3), np.uint8)
    b = a.copy()
    b[20:60, 1530:1590] = 255   # top-right HUD corner (?/gear), outside every rect
    assert roi_diff(a, b) == 0.0, roi_diff(a, b)
    # legacy single-rect call still supported
    b2 = a.copy()
    b2[:80, :] = 255            # top band: outside TABLE_ROI alone...
    assert roi_diff(a, b2, TABLE_ROI) == 0.0
    print("test_change_outside_all_rois_ignored OK")


def test_opponent_hand_row_motion_detected():
    # the 2026-07-07 fix: 理牌 compaction in the toimen hand row must block
    # stability (it was outside the old single TABLE_ROI).
    a = np.zeros((900, 1600, 3), np.uint8)
    b = a.copy()
    b[10:70, 600:1100] = 255    # toimen row region
    assert roi_diff(a, b) > 3.0, roi_diff(a, b)
    assert roi_diff(a, b, TABLE_ROI) == 0.0      # the old ROI really was blind here
    assert len(STABILITY_ROIS) == 5
    print("test_opponent_hand_row_motion_detected OK")


def test_change_inside_roi_detected():
    a = np.zeros((100, 100, 3), np.uint8)
    b = a.copy()
    b[40:60, 40:60] = 255     # center, inside ROI
    assert roi_diff(a, b) > 1.0
    print("test_change_inside_roi_detected OK")


def test_shape_mismatch_sentinel():
    a = np.zeros((100, 100, 3), np.uint8)
    b = np.zeros((90, 90, 3), np.uint8)
    assert roi_diff(a, b) >= 1e8
    print("test_shape_mismatch_sentinel OK")


if __name__ == "__main__":
    test_identical_is_zero(); test_change_outside_all_rois_ignored()
    test_opponent_hand_row_motion_detected()
    test_change_inside_roi_detected(); test_shape_mismatch_sentinel()
    print("ALL test_roi_diff OK")
