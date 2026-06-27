"""38-class tile classifier (small CNN) for the easy zones (hand/dora deterministic crops).

A clean rewrite of mycv's TileNet, trained on auto-labeled crops. 32×32 input,
38-class output (tiles.TILE_NAMES). CPU-friendly. The shipped recognizer imports
this; it never imports the Akagi-coupled capture/ package.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..tiles import TILE_NAMES, NUM_CLASSES

INPUT = 64   # fine-grained tile discrimination (dot/stroke counting) needs > 32px
_MEAN = np.array([0.5, 0.5, 0.5], np.float32)
_STD = np.array([0.5, 0.5, 0.5], np.float32)


class TileNet(nn.Module):
    def __init__(self, n_classes: int = NUM_CLASSES):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))

        # 64→32→16→8 ; AdaptiveAvgPool makes the head input-size-agnostic.
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True), nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def preprocess(bgr: np.ndarray) -> torch.Tensor:
    """cv2 BGR uint8 crop → normalized 3×32×32 float tensor."""
    import cv2
    img = cv2.resize(bgr, (INPUT, INPUT))[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB
    img = (img - _MEAN) / _STD
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


class TileClassifier:
    """Inference wrapper: list of BGR crops → list of tile names."""

    def __init__(self, weights: str, device: str = "cpu"):
        self.device = device
        self.model = TileNet().to(device)
        self.model.load_state_dict(torch.load(weights, map_location=device))
        self.model.eval()

    @torch.no_grad()
    def predict(self, crops: list[np.ndarray]) -> list[str]:
        if not crops:
            return []
        batch = torch.stack([preprocess(c) for c in crops]).to(self.device)
        idx = self.model(batch).argmax(1).cpu().numpy()
        return [TILE_NAMES[i] for i in idx]
