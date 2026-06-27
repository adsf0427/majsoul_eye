import numpy as np
import torch

from majsoul_eye.recognize.classifier import TileNet, preprocess, INPUT
from majsoul_eye.tiles import NUM_CLASSES


def test_tilenet_forward_shape():
    m = TileNet()
    out = m(torch.randn(4, 3, INPUT, INPUT))
    assert out.shape == (4, NUM_CLASSES)


def test_tilenet_input_size_agnostic():
    # AdaptiveAvgPool head → arbitrary input size works
    m = TileNet()
    assert m(torch.randn(1, 3, 48, 48)).shape == (1, NUM_CLASSES)
    assert m(torch.randn(1, 3, 96, 96)).shape == (1, NUM_CLASSES)


def test_preprocess_shape_and_norm():
    crop = np.full((40, 30, 3), 128, np.uint8)
    t = preprocess(crop)
    assert t.shape == (3, INPUT, INPUT)
    assert abs(float(t.mean())) < 0.2   # 128/255 normalized ~ 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_classifier OK")
