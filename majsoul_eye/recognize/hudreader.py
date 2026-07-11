"""HUD micro-readers (shipped product; capture-free).

DigitCTC — segmentation-free CRNN-CTC over a 32px-high strip (charset
hud.CTC_CHARSET; index+1, 0=blank). Round/wind heads reuse TileNet(n_classes).
HudReader wraps all three behind one read(crop, cls_name) call; rotation to
upright happens UPSTREAM (dataset crops are saved rotated; runtime rotates by
hud.FIELD_ROT before calling read)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from majsoul_eye.hud import (CTC_CHARSET, NUMERIC_FIELDS, ROUND_CLASSES,
                             WIND_CLASSES)
from majsoul_eye.recognize.classifier import TileNet, preprocess

N_CTC = len(CTC_CHARSET) + 1          # +blank at index 0


def encode_text(s: str) -> list[int]:
    return [CTC_CHARSET.index(c) + 1 for c in s]


class DigitCTC(nn.Module):
    """1x32xW -> (W/4) x N_CTC log-probs. Pools H 32->2 then collapses."""

    def __init__(self, n_out: int = N_CTC):
        super().__init__()

        def block(i, o, pool):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(pool))

        self.features = nn.Sequential(
            block(1, 32, (2, 2)), block(32, 64, (2, 2)),   # H 32->8, W /4
            block(64, 128, (2, 1)), block(128, 128, (2, 1)))  # H 8->2, W keeps /4
        self.head = nn.Linear(128 * 2, n_out)

    def forward(self, x):                       # B,1,32,W
        f = self.features(x)                    # B,128,2,W/4
        f = f.permute(0, 3, 1, 2).flatten(2)    # B,T,256
        return self.head(f).log_softmax(-1)     # B,T,N_CTC


def ctc_decode(logits: torch.Tensor) -> str:
    """Greedy best-path: argmax per step, collapse repeats, drop blank(0)."""
    out, prev = [], -1
    for i in logits.argmax(-1).tolist():
        if i != prev and i != 0:
            out.append(CTC_CHARSET[i - 1])
        prev = i
    return "".join(out)


def _strip(bgr: np.ndarray) -> torch.Tensor:
    """BGR crop -> 1x1x32xW normalized gray strip (W scaled with aspect, min 32)."""
    import cv2
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    nw = max(32, round(w * 32 / h))
    g = cv2.resize(g, (nw, 32)).astype(np.float32) / 255.0
    return torch.from_numpy(g)[None, None]


class HudReader:
    def __init__(self, path: str | None = None, device: str = "cpu"):
        import os
        path = path or os.path.join(os.path.dirname(__file__), "hud_reader.pt")
        ck = torch.load(path, map_location=device, weights_only=True)
        assert ck["charset"] == CTC_CHARSET, "charset drift vs weights"
        self.device = device
        self.ctc = DigitCTC().to(device).eval()
        self.ctc.load_state_dict(ck["ctc"])
        self.round = TileNet(n_classes=len(ROUND_CLASSES)).to(device).eval()
        self.round.load_state_dict(ck["round"])
        self.wind = TileNet(n_classes=len(WIND_CLASSES)).to(device).eval()
        self.wind.load_state_dict(ck["wind"])

    @torch.no_grad()
    def read(self, bgr_crop: np.ndarray, cls_name: str) -> str:
        if cls_name in NUMERIC_FIELDS:
            return ctc_decode(self.ctc(_strip(bgr_crop).to(self.device))[0])
        # round/wind reuse TileNet -> use classifier.py's canonical inference
        # preprocess (resize INPUT x INPUT, BGR->RGB, /255 then (x-0.5)/0.5) so
        # train_hudreader.py's CE datasets and this read() path agree bit-for-bit.
        t = preprocess(bgr_crop)[None].to(self.device)
        if cls_name == "round_label":
            return ROUND_CLASSES[int(self.round(t).argmax())]
        if cls_name == "seat_wind_self":
            return WIND_CLASSES[int(self.wind(t).argmax())]
        raise ValueError(f"not a readable field: {cls_name}")

    @torch.no_grad()
    def class_probabilities(self, bgr_crop: np.ndarray,
                            cls_name: str) -> tuple[list[str], np.ndarray]:
        tensor = preprocess(bgr_crop)[None].to(self.device)
        if cls_name == "round_label":
            probabilities = torch.softmax(self.round(tensor), dim=1)[0]
            return list(ROUND_CLASSES), probabilities.cpu().numpy()
        if cls_name == "seat_wind_self":
            probabilities = torch.softmax(self.wind(tensor), dim=1)[0]
            return list(WIND_CLASSES), probabilities.cpu().numpy()
        raise ValueError(f"no class distribution for {cls_name}")
