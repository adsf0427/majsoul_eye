"""Map an arbitrary screenshot onto the canonical 16:9 board frame.

This is what lets the fixed-slot ROIs in ``coords.py`` survive different
resolutions / windows / external screenshots (docs/DESIGN.md §3.5). A locator
returns a :class:`BoardRegion` (the board's pixel rect within the frame); ROIs
are then placed relative to that region.

- :func:`locate_anchor` — **the one the recognizer uses.** Fits the board rect from
  the DETECTIONS themselves (centre-panel HUD classes + the hero hand row), so it
  needs no assumption about the screenshot's aspect or framing: phone, windowed
  under browser chrome, cropped, downscaled. See its section below.
- :func:`locate_fullscreen` — assume the whole frame IS the 16:9 board. Correct for
  our own captures; a silent disaster on anything else (a mis-located board yields
  a full, plausible, WRONG board rather than an error), so it is not the default.
- :func:`locate_letterbox` — trim black bars (letterboxed captures).
- :func:`locate_wide` — wider-than-16:9 phone screenshots: centered 16:9 board rect.
- :func:`locate_auto` — dispatch between those three by aspect ratio. Superseded by
  :func:`locate_anchor`; kept for the offline/eval tooling that assumes clean frames.
- :func:`clipped_sides` — a correct fit on a CROPPED screenshot still returns a
  correct rect that simply sticks out of the image. This is what notices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .coords import MYCV_REF, NormBox


@dataclass(frozen=True)
class BoardRegion:
    """The board's pixel rect within a frame: offset (ox,oy) + size (bw,bh).

    ``fw``/``fh`` are the FULL frame dims (0 = unknown, treated as the board
    rect itself) — consumers that place screen-anchored 2D-HUD elements (which
    do NOT live inside the 16:9 board rect on wide phone screenshots) need
    them; pure board-relative ROIs ignore them."""
    ox: int
    oy: int
    bw: int
    bh: int
    fw: int = 0
    fh: int = 0

    @property
    def frame_w(self) -> int:
        return self.fw or self.ox + self.bw

    @property
    def frame_h(self) -> int:
        return self.fh or self.oy + self.bh

    def norm_to_px(self, box: NormBox) -> tuple[int, int, int, int]:
        return (self.ox + round(box.x0 * self.bw), self.oy + round(box.y0 * self.bh),
                self.ox + round(box.x1 * self.bw), self.oy + round(box.y1 * self.bh))

    def px_to_norm_box(self, x0: int, y0: int, x1: int, y1: int) -> NormBox:
        """Map a full-frame pixel box back to a normalized canonical box."""
        return NormBox((x0 - self.ox) / self.bw, (y0 - self.oy) / self.bh,
                       (x1 - self.ox) / self.bw, (y1 - self.oy) / self.bh)

    def crop(self, frame: np.ndarray, box: NormBox) -> np.ndarray:
        x0, y0, x1, y1 = self.norm_to_px(box)
        h, w = frame.shape[:2]
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        return frame[y0:y1, x0:x1]

    @property
    def aspect(self) -> float:
        return self.bw / self.bh if self.bh else 0.0


def locate_fullscreen(frame: np.ndarray) -> BoardRegion:
    """Treat the entire frame as the 16:9 board."""
    h, w = frame.shape[:2]
    return BoardRegion(0, 0, w, h, w, h)


def locate_wide(frame: np.ndarray) -> BoardRegion:
    """Wider-than-16:9 screenshot (phones, 2.17:1 etc.): the 3D table renders
    as a CENTERED 16:9 rect (verified on real 2.17/2.20 phone samples — hand /
    rivers / melds align once cropped); the extra width holds only 2D HUD.
    NOTE the dora indicator lives OUTSIDE this rect (screen-corner anchored,
    device-dependent inset) — assemble rescues it from stray detections."""
    h, w = frame.shape[:2]
    bw = round(h * 16 / 9)
    return BoardRegion((w - bw) // 2, 0, bw, h, w, h)


def locate_auto(frame: np.ndarray, tol: float = 0.02) -> BoardRegion:
    """Dispatch by aspect: ~16:9 -> fullscreen, wider -> wide (centered 16:9
    table), narrower -> letterbox (trim bars)."""
    h, w = frame.shape[:2]
    aspect = w / h if h else 0.0
    if aspect > (16 / 9) * (1 + tol):
        return locate_wide(frame)
    if aspect < (16 / 9) * (1 - tol):
        return locate_letterbox(frame)
    return locate_fullscreen(frame)


def locate_letterbox(frame: np.ndarray, black_thresh: int = 16) -> BoardRegion:
    """Trim near-black borders and return the content rect.

    Handles browser chrome only partially — use fullscreen capture when possible.
    """
    if frame.ndim == 3:
        gray = frame.max(axis=2)
    else:
        gray = frame
    cols = np.where(gray.max(axis=0) > black_thresh)[0]
    rows = np.where(gray.max(axis=1) > black_thresh)[0]
    if len(cols) == 0 or len(rows) == 0:
        return locate_fullscreen(frame)
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    h, w = frame.shape[:2]
    return BoardRegion(x0, y0, x1 - x0, y1 - y0, w, h)


# --- anchor localization: the detections ARE the landmarks --------------------
#
# The board is ALWAYS a 16:9 rect (centered on wide phones, inset in a window),
# so the image->canonical map has just 3 DOF: uniform scale k and offset
# (ox, oy), with  img = k * canon + (ox, oy).  Two correspondences determine it.
#
# Correspondences come from detections whose canonical position is
# state-independent:
#   * the 7 center-panel HUD classes — always rendered, board-anchored
#     ("center-anchored; identical on PC/mobile", hud.py), pixel-tight seeds.
#     Short x-baseline (270/1920) though, so scale from these ALONE is
#     ill-conditioned; they are redundancy, not the scale source.
#   * the hero hand row — the tallest tiles in any what-cut screenshot, laid out
#     on a known pitch from a known left anchor. Spans 69% of frame width, so it
#     is what actually pins the scale.
#
# Screen-anchored elements (the dora strip, the top-left kyotaku/honba panel) do
# NOT move with the board on off-16:9 frames. They are simply never fed in here,
# and if a stray one were, RANSAC would drop it as an outlier.

CANON_W, CANON_H = MYCV_REF                     # 1920 x 1080

PANEL_LANDMARKS: tuple[str, ...] = (
    "score_self", "score_right", "score_across", "score_left",
    "round_label", "wall_count", "seat_wind_self",
)

# Reprojection tolerance, in canonical px. A hand tile is 95 canon px wide, and
# zone routing rejects beyond 60 fullwarp px, so a third of a tile is a fair
# "this landmark agrees" bar while still admitting detector jitter.
ANCHOR_TOL_CANON = 30.0


@dataclass(frozen=True)
class Localization:
    """A fitted board rect plus the evidence for trusting it."""
    region: BoardRegion
    method: str                  # "anchor" | "aspect"
    inliers: int
    total: int
    residual: float              # median reprojection residual, canonical px
    panel_inliers: int
    hand_inliers: int

    @property
    def confidence(self) -> float:
        return self.inliers / self.total if self.total else 0.0


def _panel_pairs(dets) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Center-panel HUD detections -> (canonical center, image center) pairs.

    The detector is trained on exactly these seed boxes, so its box center is
    the seed's center. One detection per class (the best-scoring one).
    """
    from .coords import HUD_SEEDS                     # local: coords is heavier

    best: dict = {}
    for det in dets:
        if det.name not in PANEL_LANDMARKS:
            continue
        if det.name not in best or det.score > best[det.name].score:
            best[det.name] = det
    pairs = []
    for name, det in best.items():
        box = HUD_SEEDS[name]
        x0, y0, x1, y1 = det.xyxy
        pairs.append(((box.cx * CANON_W, box.cy * CANON_H),
                      ((x0 + x1) / 2, (y0 + y1) / 2)))
    return pairs


def _hand_row(dets) -> list:
    """The hero's concealed hand row, left to right — found scale-free.

    Hand tiles are the tallest tile faces in the frame (152 canon px vs ~92 for
    rivers/melds and ~81 for dora), so "tallest, and sharing a row" isolates
    them without knowing the scale yet.
    """
    tiles = [d for d in dets if d.tile is not None and d.tile != "back"]
    if len(tiles) < 4:
        return []
    heights = np.array([d.xyxy[3] - d.xyxy[1] for d in tiles], float)
    tall = [d for d, h in zip(tiles, heights)
            if h >= 0.75 * float(np.percentile(heights, 95))]
    if len(tall) < 4:
        return []
    # Group the tall tiles into rows by y-center; keep the biggest row.
    tall.sort(key=lambda d: (d.xyxy[1] + d.xyxy[3]) / 2)
    rows, current = [], [tall[0]]
    for det in tall[1:]:
        span = (det.xyxy[3] - det.xyxy[1])
        cy, prev_cy = ((det.xyxy[1] + det.xyxy[3]) / 2,
                       (current[-1].xyxy[1] + current[-1].xyxy[3]) / 2)
        if abs(cy - prev_cy) < 0.5 * span:
            current.append(det)
        else:
            rows.append(current)
            current = [det]
    rows.append(current)
    row = max(rows, key=len)
    row.sort(key=lambda d: d.xyxy[0])
    return row


def _hand_pairs(dets) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Hand-row detections -> (canonical center, image center) pairs.

    The concealed hand is LEFT-anchored (melds shorten it from the right), so the
    leftmost tile is always slot 0. The drawn tile sits after an extra gap; it is
    detected as such and mapped to its own (gapped) slot, which usefully extends
    the baseline further right.
    """
    from .coords import HAND

    row = _hand_row(dets)
    if len(row) < 4:
        return []
    lefts = np.array([d.xyxy[0] for d in row], float)
    gaps = np.diff(lefts)
    if len(gaps) == 0:
        return []
    pitch = float(np.median(gaps))
    if pitch <= 0:
        return []
    # A gap materially wider than the pitch means the tiles after it are the
    # drawn tile (there is at most one such break).
    concealed = len(row)
    for index, gap in enumerate(gaps):
        if gap > 1.25 * pitch:
            concealed = index + 1
            break

    pairs = []
    for index, det in enumerate(row):
        is_tsumo = index >= concealed
        slot = index if not is_tsumo else concealed
        box = HAND.slot_box(slot, is_tsumo=is_tsumo)
        x0, y0, x1, y1 = det.xyxy
        pairs.append(((box.cx * CANON_W, box.cy * CANON_H),
                      ((x0 + x1) / 2, (y0 + y1) / 2)))
    return pairs


def _fit_similarity(pairs) -> Optional[tuple[float, float, float]]:
    """Least-squares (k, ox, oy) for  img = k * canon + (ox, oy)."""
    if len(pairs) < 2:
        return None
    rows, rhs = [], []
    for (cx, cy), (ix, iy) in pairs:
        rows.append([cx, 1.0, 0.0])
        rhs.append(ix)
        rows.append([cy, 0.0, 1.0])
        rhs.append(iy)
    solution, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    k, ox, oy = (float(v) for v in solution)
    if k <= 0:
        return None
    return k, ox, oy


def _residuals(pairs, model) -> np.ndarray:
    """Per-pair reprojection error, expressed in CANONICAL px (scale-free)."""
    k, ox, oy = model
    canon = np.array([p[0] for p in pairs], float)
    image = np.array([p[1] for p in pairs], float)
    back = (image - np.array([ox, oy])) / k
    return np.hypot(*(back - canon).T)


def locate_anchor(frame: np.ndarray, dets,
                  tol: float = ANCHOR_TOL_CANON) -> Optional[Localization]:
    """Fit the board rect from landmark detections. None if it cannot be pinned.

    Deterministic RANSAC: with ~20 candidate correspondences, every 2-subset is
    enumerable, so there is no sampling and no seed.
    """
    panel = _panel_pairs(dets)
    hand = _hand_pairs(dets)
    pairs = panel + hand
    if len(pairs) < 2:
        return None

    best_inliers: Optional[np.ndarray] = None
    best_count = -1
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            model = _fit_similarity([pairs[i], pairs[j]])
            if model is None:
                continue
            mask = _residuals(pairs, model) <= tol
            count = int(mask.sum())
            if count > best_count:
                best_count, best_inliers = count, mask
    if best_inliers is None or best_count < 2:
        return None

    inlier_pairs = [p for p, keep in zip(pairs, best_inliers) if keep]
    model = _fit_similarity(inlier_pairs)          # refit on the consensus set
    if model is None:
        return None
    k, ox, oy = model
    residual = float(np.median(_residuals(inlier_pairs, model)))

    height, width = frame.shape[:2]
    region = BoardRegion(int(round(ox)), int(round(oy)),
                         int(round(k * CANON_W)), int(round(k * CANON_H)),
                         width, height)
    n_panel = len(panel)
    return Localization(region=region, method="anchor",
                        inliers=best_count, total=len(pairs), residual=residual,
                        panel_inliers=int(best_inliers[:n_panel].sum()),
                        hand_inliers=int(best_inliers[n_panel:].sum()))


def clipped_sides(region: BoardRegion, tol_frac: float = 0.005) -> list[str]:
    """Which sides of the board fall outside the frame (i.e. were cropped away).

    A correct fit on a cropped screenshot still returns a correct rect — it just
    sticks out of the image. So this, not the fit, is what tells us the board is
    incomplete. Tolerance absorbs fit jitter (measured <=0.3% of board width).
    """
    tol = tol_frac * region.bw
    sides = []
    if region.ox < -tol:
        sides.append("left")
    if region.oy < -tol:
        sides.append("top")
    if region.ox + region.bw > region.frame_w + tol:
        sides.append("right")
    if region.oy + region.bh > region.frame_h + tol:
        sides.append("bottom")
    return sides


def clipped_required_regions(region: BoardRegion, tol_frac: float = 0.005) -> list[str]:
    """Which REQUIRED canonical regions fall (partly) outside the frame.

    A clipped board edge is not, by itself, an unanswerable screenshot: on off-16:9
    devices (4:3 / 1.44:1 iPads, sub-16:9 windows) the game legitimately crops the
    canonical scene's margins while keeping every gameplay element on screen, so a
    correct fit routinely sticks out of the frame. What recognition cannot proceed
    without is (a) the hero hand row — the what-cut question itself — and (b) the
    center HUD panel (round/scores/seat) that pins the game state. Anything else a
    clipped edge may cost (an opponent's river tail, the screen-anchored dora
    strip) degrades to a correctable draft issue downstream, not a dead end.

    Returns a subset of ``["hand", "panel"]``; empty means every required region
    is inside the frame (within the same fit-jitter tolerance as
    :func:`clipped_sides`).
    """
    from .coords import HAND, HUD_SEEDS                 # local: coords is heavier

    hand_boxes = [HAND.slot_box(0), HAND.slot_box(13, is_tsumo=True)]
    panel_boxes = [HUD_SEEDS[name] for name in PANEL_LANDMARKS]
    tol = tol_frac * region.bw
    missing = []
    for name, boxes in (("hand", hand_boxes), ("panel", panel_boxes)):
        # Normalized canonical box -> image px: img_x = ox + fx * bw (the region
        # IS the canonical rect mapped into the image).
        x0 = region.ox + min(b.x0 for b in boxes) * region.bw
        y0 = region.oy + min(b.y0 for b in boxes) * region.bh
        x1 = region.ox + max(b.x1 for b in boxes) * region.bw
        y1 = region.oy + max(b.y1 for b in boxes) * region.bh
        if (x0 < -tol or y0 < -tol
                or x1 > region.frame_w + tol or y1 > region.frame_h + tol):
            missing.append(name)
    return missing


class AnchorLocator:
    """Landmark localizer for arbitrary screenshots (phone, windowed, cropped).

    Falls back to the aspect heuristics when the landmarks cannot pin a rect —
    a frame with too few detections is not one we can recognize anyway, so the
    fallback exists to keep the type total, not to rescue anything.
    """

    def __init__(self, tol: float = ANCHOR_TOL_CANON):
        self.tol = tol

    def locate(self, frame: np.ndarray, dets=()) -> BoardRegion:
        found = locate_anchor(frame, dets, tol=self.tol)
        return found.region if found is not None else locate_auto(frame)
