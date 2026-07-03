"""AI-path stability: only save once two consecutive grabs match within the table ROI."""
import importlib.util
import numpy as np

_spec = importlib.util.spec_from_file_location("ap", "scripts/capture/autoplay_ai.py")
ap = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ap)


def test_waits_for_stability():
    a = np.zeros((100, 100, 3), np.uint8)
    moving = a.copy(); moving[40:60, 40:60] = 255      # tile mid-flight in ROI
    settled = a.copy()                                  # animation done

    st = {"ref": None}
    act, st = ap.stable_capture_step(st, moving, thresh=3.0)
    assert act == "wait"                                # first grab -> set ref, wait
    act, st = ap.stable_capture_step(st, settled, thresh=3.0)
    assert act == "wait"                                # differs from moving -> still wait
    act, st = ap.stable_capture_step(st, settled, thresh=3.0)
    assert act == "save"                                # two identical grabs -> save
    print("test_waits_for_stability OK")


if __name__ == "__main__":
    test_waits_for_stability()
    print("ALL test_autoplay_stability OK")
