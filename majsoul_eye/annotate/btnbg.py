"""Per-game BTN_ZONE background model — the skin-agnostic prior that lets
annotate/hud.py segment action-button PLATES instead of thresholding brightness.

An action button is an overlay drawn onto an otherwise static patch of table, so
``|frame − background|`` isolates it. The background is that game's own median of
BTN_ZONE over the frames GT says carry NO button. Median, not mean: the zone also
holds animated tablecloth FX and breathing 立绘, and a median over ~31 frames
erases anything transient while a mean would smear it.

Why per game: the tablecloth is a per-game skin (126 distinct table skins across
datasets/v5), so there is no global background to subtract. Building it costs one
median over ~31 decoded frames, once per game.

Akagi-free and import-light like the rest of annotate/; the SHIPPED recognizer
never needs this (it is a labeling-time prior, not a runtime one).
"""
from __future__ import annotations

import cv2
import numpy as np

from majsoul_eye.coords import BTN_ZONE
from majsoul_eye.hud import buttons_for_ops
from majsoul_eye.normalize import locate_fullscreen

MAX_FRAMES = 31    # median sample size; 8+ already separates cleanly (measured)
MIN_FRAMES = 8     # below this the median is too noisy to threshold against


def zone_gray(img: np.ndarray, region) -> np.ndarray:
    """BTN_ZONE of `img` as gray float32."""
    x0, y0, x1, y1 = region.norm_to_px(BTN_ZONE)
    return cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)


def game_btn_background(seq_states, seq_frames, max_frames: int = MAX_FRAMES,
                        min_frames: int = MIN_FRAMES):
    """One game -> zone-shaped gray float32 median, or None if too few clean frames.

    Clean = GT offers no action button on that frame. Sampling is evenly spaced
    over the game (deterministic, and it spreads the sample across the animation
    cycle better than taking the first N)."""
    clean = [s for s in sorted(seq_frames)
             if s in seq_states and seq_states[s] is not None
             and not buttons_for_ops(getattr(seq_states[s], "pending_ops", None) or [])]
    if len(clean) < min_frames:
        return None
    if len(clean) > max_frames:
        step = len(clean) / max_frames
        clean = [clean[int(i * step)] for i in range(max_frames)]

    stack = []
    for seq in clean:
        img = cv2.imread(seq_frames[seq])
        if img is None:
            continue
        if img.shape[1] != 1920 or img.shape[0] != 1080:
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
        stack.append(zone_gray(img, locate_fullscreen(img)))
    if len(stack) < min_frames:
        return None
    return np.median(np.stack(stack), axis=0).astype(np.float32)
