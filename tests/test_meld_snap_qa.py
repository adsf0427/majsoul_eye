"""Fast unit test for the meld_snap_qa clustering helper (no capture data)."""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "annotate"))
from meld_snap_qa import dominant  # noqa: E402


def test_dominant_majority_over_outliers():
    # 70 frames locked at +46, 30 mislocked negative -> center ~46, frac ~0.7
    c, f = dominant([46.0] * 70 + [-20.0] * 30)
    assert abs(c - 46.0) < 2.0, c
    assert 0.65 <= f <= 0.75, f


def test_dominant_tight_cluster():
    c, f = dominant([5.0, 5.5, 4.5, 5.2])
    assert abs(c - 5.0) < 1.0, c
    assert f == 1.0, f


def test_dominant_empty():
    assert dominant([]) == (0.0, 0.0)


if __name__ == "__main__":
    for _name, _fn in sorted(list(globals().items())):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(_name, "OK")
    print("all meld_snap_qa tests passed")
