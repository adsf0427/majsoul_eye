"""build_dataset.gate_frame: returns the indices of boxes to skip."""
import importlib.util
import numpy as np

_spec = importlib.util.spec_from_file_location("bd", "scripts/train/build_dataset.py")
bd = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bd)

from majsoul_eye.tiles import NAME_TO_ID


class _Box:
    def __init__(self, tile): self.tile = tile


class StubClf:
    def __init__(self, seen): self.seen = seen
    def predict_proba(self, crops):
        out = np.zeros((len(crops), 38), np.float32)
        for i in range(len(crops)):
            out[i, NAME_TO_ID[self.seen[i]]] = 1.0
        return out


def test_gate_frame_skips_bad_boxes():
    # frame with 3 boxes; middle one mislabeled -> skip index 1
    frame = np.full((100, 100, 3), 240, np.uint8)
    boxes = [_Box("8s"), _Box("3p"), _Box("S")]
    crops = [np.full((32, 32, 3), 240, np.uint8) for _ in range(3)]
    clf = StubClf(["8s", "9m", "S"])
    skip = bd.gate_frame(frame, boxes, crops, clf, tau=0.5, max_bad=2)
    assert skip == {1}, skip
    print("test_gate_frame_skips_bad_boxes OK")


if __name__ == "__main__":
    test_gate_frame_skips_bad_boxes()
    print("ALL test_build_gate OK")
