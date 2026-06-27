"""Run mycv's REAL recognition pipeline as a measurement engine (approach B').

Rather than re-port mycv's algorithm by hand (risking a threshold/off-by-one
divergence), we import the actual ``myCV`` class from ``../auto/mycv`` and call
its real functions (``cutPic``, ``model``, ``getType``, ``getHandTiles``). This
is maximally faithful — it IS mycv's code — while the seat-driving and scoring
live here in our framework.

Findings that shape this adapter (validated on real session6 frames, see
docs/superpowers/specs/2026-06-27-mycv-baseline-reproduction-design.md):

* mycv works in a fixed 1080p screen space; we resize each frame to 1920×1080
  before handing it over (``cutPic`` and ``getHandTiles`` assume those coords).
* Seat masks ``m{1,2,3}`` cleanly isolate the three OPPONENT rivers; with the
  hero at screen-bottom, raw mask index k → absolute seat ``(hero+k) % 4``
  (k=1 shimocha/right, k=2 toimen/top, k=3 kamicha/left). Mask ``m0`` is a loose
  "all opponents" mask and is unused here.
* The hero's OWN river is masked out of every seat mask (mycv knows its own
  discards from its own play) → self-river is NOT a mycv vision target.
* Seat masks OVERLAP in meld zones, so a meld tile can appear under several
  masks. We therefore return meld-type detections as a de-duplicated global pool
  (keyed by rounded position) rather than trusting per-mask seat assignment.

This module is Akagi-free but mycv-coupled — it is a dev baseline, never imported
by ``recognize/``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

DEFAULT_MYCV_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "auto", "mycv")
)

# mycv's 1080p working canvas.
W1080, H1080 = 1920, 1080
# Opponent seat masks: raw mask index -> screen offset from hero.
OPPONENT_MASKS = (1, 2, 3)  # +1 right(shimocha), +2 across(toimen), +3 left(kamicha)


@dataclass
class Detection:
    name: str
    x: int
    y: int
    w: int = 0
    h: int = 0


@dataclass
class FrameResult:
    hand: list[str] = field(default_factory=list)               # self hand tile names (screen order)
    rivers: dict[int, list[Detection]] = field(default_factory=dict)  # abs seat -> river dets
    melds: list[Detection] = field(default_factory=list)        # global de-duped meld-type dets
    error: str | None = None


class MycvEngine:
    """Thin adapter over the real ``myCV`` instance."""

    def __init__(self, mycv_dir: str = DEFAULT_MYCV_DIR):
        self.mycv_dir = os.path.abspath(mycv_dir)
        if not os.path.isdir(self.mycv_dir):
            raise FileNotFoundError(f"mycv dir not found: {self.mycv_dir}")
        prev_cwd = os.getcwd()
        try:
            # myCV.__init__ loads assets via cwd-relative paths (myweight.pth, m/*.png).
            os.chdir(self.mycv_dir)
            if self.mycv_dir not in sys.path:
                sys.path.insert(0, self.mycv_dir)
            import main2  # noqa: WPS433 (deliberate sibling-repo import)
            self._main2 = main2
            self.cv = main2.myCV()
        finally:
            os.chdir(prev_cwd)
        self.river_meld_classes = dict(self.cv.dictjianxie)  # 37-class idx -> name
        self.type_of = dict(self.cv.dicttype)                # region code -> 0 river / 1-3 meld / 4 hand

    # --- low-level: one seat mask -> classified, type-routed detections --------

    def _segment_seat(self, f1080: np.ndarray, mask_idx: int):
        """Return (river_dets, meld_dets) for one seat mask, using mycv's real
        cutPic + ResNet + getType. river = type 0, meld = type 1/2/3."""
        import torch

        cv = self.cv
        masked = self._main2.cv.copyTo(f1080, cv.mask_list[mask_idx], cv.chun.copy())
        opt, ptn, lizhi, coord = cv.cutPic(masked)
        if not opt:
            return [], []
        res = cv.model(torch.tensor(np.array(opt, np.float32))).argmax(1).numpy()
        river, meld = [], []
        for i in range(len(opt)):
            code = cv.getType(ptn[i])
            t = self.type_of.get(code, -1)
            name = self.river_meld_classes.get(int(res[i]), f"?{int(res[i])}")
            x, y, w, h = (int(v) for v in coord[i])
            det = Detection(name, int(ptn[i][0]), int(ptn[i][1]), w, h)
            if t == 0:
                river.append(det)
            elif t in (1, 2, 3):
                meld.append(det)
        return river, meld

    # --- public: full frame ----------------------------------------------------

    def recognize(self, frame_bgr: np.ndarray, hero_seat: int) -> FrameResult:
        import cv2

        out = FrameResult()
        try:
            f1080 = cv2.resize(frame_bgr, (W1080, H1080))
        except Exception as e:  # pragma: no cover - defensive
            out.error = f"resize: {e!r}"
            return out

        # self hand via floodFill + tile.model (mycv's getHandTiles)
        try:
            hand = self.cv.getHandTiles(f1080)
            out.hand = list(hand) if hand else []
        except Exception as e:
            out.error = f"hand: {e!r}"

        # opponent rivers + meld pool via seat masks
        meld_pool: dict[tuple[int, int], Detection] = {}
        for k in OPPONENT_MASKS:
            seat = (hero_seat + k) % 4
            try:
                river, meld = self._segment_seat(f1080, k)
            except Exception as e:
                out.error = (out.error or "") + f"; seat{k}: {e!r}"
                continue
            out.rivers[seat] = river
            for d in meld:
                meld_pool[(round(d.x / 12), round(d.y / 12))] = d  # ~12px dedup grid
        out.melds = list(meld_pool.values())
        return out
