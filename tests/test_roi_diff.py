import numpy as np
from majsoul_eye.capture.roi_diff import roi_diff, TABLE_ROI


def test_identical_is_zero():
    a = np.random.RandomState(0).randint(0, 255, (90, 160, 3), np.uint8)
    assert roi_diff(a, a) == 0.0
    print("test_identical_is_zero OK")


def test_change_outside_roi_ignored():
    a = np.zeros((100, 100, 3), np.uint8)
    b = a.copy()
    b[:10, :] = 255           # top HUD band, outside TABLE_ROI y0=0.16
    assert roi_diff(a, b) == 0.0, roi_diff(a, b)
    print("test_change_outside_roi_ignored OK")


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
    test_identical_is_zero(); test_change_outside_roi_ignored()
    test_change_inside_roi_detected(); test_shape_mismatch_sentinel()
    print("ALL test_roi_diff OK")
