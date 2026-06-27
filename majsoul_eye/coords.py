"""Normalized coordinate model for Majsoul easy-zone ROIs.

All boxes are normalized to 0–1 against a canonical 16:9 board, so they apply at
any resolution once a frame has been mapped to the canonical frame (see
``normalize.py``). Seed values are ported from ``auto/mycv`` (the proven 1920×1080
*web-client* layout) and Akagi's normalized hand slots.

⚠️ CALIBRATE: these seeds are from the web client at 1920×1080. Verify/adjust them
against a real ``session2`` capture (and again per target client — the Win client
may differ slightly). Edit the pixel tables below; everything else derives.
"""

from __future__ import annotations

from dataclasses import dataclass

MYCV_REF = (1920, 1080)  # reference resolution the mycv pixel ROIs were measured at


@dataclass(frozen=True)
class NormBox:
    """Axis-aligned box in normalized [0,1] coords (x0,y0 top-left, x1,y1 bot-right)."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float: return self.x1 - self.x0
    @property
    def h(self) -> float: return self.y1 - self.y0
    @property
    def cx(self) -> float: return (self.x0 + self.x1) / 2
    @property
    def cy(self) -> float: return (self.y0 + self.y1) / 2

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Map to integer pixel box (x0,y0,x1,y1) on a width×height canonical frame."""
        return (round(self.x0 * width), round(self.y0 * height),
                round(self.x1 * width), round(self.y1 * height))

    def yolo(self) -> tuple[float, float, float, float]:
        """YOLO box (cx,cy,w,h), all normalized."""
        return (self.cx, self.cy, self.w, self.h)

    def erode(self, top: float = 0.0, bottom: float = 0.0,
              left: float = 0.0, right: float = 0.0) -> "NormBox":
        """Shrink the box by per-side fractions of its own width/height. Used to
        trim next-tile bleed off packed river cells (the next discard bleeds into
        the bottom of a cell — see scripts/build_dataset.py --river-erode-*)."""
        w, h = self.w, self.h
        return NormBox(self.x0 + left * w, self.y0 + top * h,
                       self.x1 - right * w, self.y1 - bottom * h)


def px_box(x0: int, y0: int, x1: int, y1: int, ref: tuple[int, int] = MYCV_REF) -> NormBox:
    """Build a NormBox from a pixel ROI measured at `ref` resolution."""
    W, H = ref
    return NormBox(x0 / W, y0 / H, x1 / W, y1 / H)


# --- easy-zone ROIs (ported from mycv main2.py get_* methods, 1920×1080) -----
# Format: name -> pixel ROI (x0, y0, x1, y1).  # CALIBRATE on session2.
_MYCV_ROIS_PX: dict[str, tuple[int, int, int, int]] = {
    "riichi_sticks": (110, 140, 140, 180),   # get_bangzi
    "honba":         (250, 140, 280, 180),   # get_benchang
    "self_wind":     (800, 500, 850, 530),   # get_zifeng
    "round_wind":    (920, 390, 950, 420),   # get_changfeng
    "round_number":  (950, 390, 970, 420),   # get_jushu
    # per-seat score readouts (union of mycv's per-digit ROIs)
    "score_self":    (915, 472, 1000, 492),  # get_f1  (bottom / self)
    "score_right":   (863, 396, 888, 428),   # get_f2  (right, stacked digits)
    "score_across":  (951, 355, 995, 373),   # get_f3  (top / across)
    # "score_left":  get_f4 is computed differently — CALIBRATE/derive on session2
}

REGIONS: dict[str, NormBox] = {name: px_box(*roi) for name, roi in _MYCV_ROIS_PX.items()}

# Dora indicators are NOT on the dead wall — they're in the TOP-LEFT HUD panel
# (calibrated T3 on session6: the single 'N' tile = (0.015,0.036,0.054,0.111),
# tile width ~0.0395), growing rightward as kan-dora reveal. NOTE: this is 2D HUD,
# so it's RESOLUTION-DEPENDENT (like scores) — fine at this 16:9 res; needs
# anchor-normalization for other resolutions. # CALIBRATE multi-dora spacing.
DORA_STRIP: NormBox = NormBox(0.015, 0.036, 0.2125, 0.111)
MAX_DORA = 5


def dora_slot(i: int, n: int = MAX_DORA) -> NormBox:
    """i-th dora indicator slot within DORA_STRIP (left→right)."""
    s = DORA_STRIP
    w = s.w / n
    return NormBox(s.x0 + i * w, s.y0, s.x0 + (i + 1) * w, s.y1)


# --- hero hand-tile slots ----------------------------------------------------
# Parametric model. Seeds: mycv flood-fill tiles are ~95×152 px at 1920×1080
# starting x≈235, y≈1002 (seed point inside tile); Akagi's normalized hand centers
# run x 0.139→0.788 at y 0.929. We model 13 concealed slots + a separated tsumo slot.
@dataclass(frozen=True)
class HandModel:
    x0: float = 235 / 1920          # left edge of first tile      # CALIBRATE
    slot_w: float = 95 / 1920       # tile width (also slot pitch)  # CALIBRATE
    y0: float = (1002 - 76) / 1080  # tile top (seed 1002 is mid; half-height ~76)
    tile_h: float = 152 / 1080      # tile height                   # CALIBRATE
    tsumo_gap: float = 0.015        # extra gap before the drawn tile (Akagi TSUMO_OFFSET_X)

    def slot_box(self, i: int, is_tsumo: bool = False) -> NormBox:
        """Box for the i-th hand tile (0-based). `is_tsumo` adds the drawn-tile gap."""
        x = self.x0 + i * self.slot_w + (self.tsumo_gap if is_tsumo else 0.0)
        return NormBox(x, self.y0, x + self.slot_w, self.y0 + self.tile_h)


HAND = HandModel()


# --- perspective river zones (screen-quadrant bounds, mycv lizhipai, 1080p) ---
# Coarse bounding boxes per opponent quadrant — used by P4 to route contour-detected
# tiles to a seat. Hero's own 河 (bottom) handled separately.  # CALIBRATE.
RIVER_ZONES_PX: dict[str, tuple[int, int, int, int]] = {
    "self":   (760, 600, 1180, 840),   # 自家 (bottom-center) — CALIBRATE (mycv read elsewhere)
    "across": (770, 130, 1140, 290),   # 对家 (top)
    "left":   (450, 265, 790, 650),    # 上家
    "right":  (1120, 265, 1430, 650),  # 下家
}
RIVER_ZONES: dict[str, NormBox] = {k: px_box(*v) for k, v in RIVER_ZONES_PX.items()}

# --- per-seat 副露 (meld) strips (for a 1-row RiverGrid) ----------------------
# Calibrated on session6 (T5). Each strip is a 1-row quad [g00,g10,g11,g01] bounding
# the `count` meld tiles observed, with a per-seat layout rule:
#   anchor: which end is FIXED as melds grow ('start'=g00 fixed, 'end'=g10 fixed)
#   order : meld-GROUP screen order ('chrono'=oldest-first, 'reverse'=newest-first)
# self melds sit bottom-right and grow LEFT (right-anchored, newest-first); the other
# seats grow OUTWARD from their avatar (start-anchored, chronological). The rotated
# called tile is ~1.3× wider (uniform cells are approximate); ankan shows 2 face-down
# 'back' tiles + 2 face-up. # CALIBRATE — approximate (partial); bootstrap refines.
MELD_STRIPS: dict[str, dict] = {
    "self":   {"quad": ((0.702, 0.861), (0.911, 0.873), (0.911, 0.960), (0.702, 0.950)), "count": 6, "anchor": "end",   "order": "reverse"},
    "right":  {"quad": ((0.863, 0.392), (0.926, 0.782), (0.964, 0.792), (0.901, 0.402)), "count": 6, "anchor": "start", "order": "chrono"},
    "across": {"quad": ((0.206, 0.022), (0.316, 0.032), (0.316, 0.092), (0.206, 0.082)), "count": 4, "anchor": "start", "order": "chrono"},
    "left":   {"quad": ((0.071, 0.695), (0.047, 0.875), (0.103, 0.875), (0.113, 0.695)), "count": 3, "anchor": "start", "order": "chrono"},
}

# --- per-seat discard GRID quads (for RiverGrid) -----------------------------
# Each quad is the FULL 6×3 discard grid as 4 normalized corners
# (g00=first-discard, g10=col-max/row0, g11=col-max/row-max, g01=col0/row-max).
# SEED values are axis-aligned bboxes auto-detected on a full-river frame
# (session6 seq 1458, 3840×2160); T2 calibration refines these into the true
# perspective trapezoids and fixes per-seat corner order (reading direction /
# rotation for side & top seats). # CALIBRATE (T2).
RIVER_QUADS: dict[str, tuple[tuple[float, float], ...]] = {
    # Calibrated on session6 seq 1458 (T2). Corner order [g00,g10,g11,g01] encodes
    # each seat's reading direction (verified against wind-tile faces):
    "self":   ((0.400, 0.498), (0.595, 0.497), (0.596, 0.696), (0.401, 0.698)),  # L→R, rows toward center
    "across": ((0.584, 0.276), (0.416, 0.276), (0.418, 0.1445), (0.581, 0.1445)),  # 180° mirror: g00=bottom-right, R→L
    "left":   ((0.402, 0.268), (0.395, 0.480), (0.300, 0.484), (0.292, 0.267)),  # 90° rot: g00=top-right, read down
    "right":  ((0.598, 0.500), (0.600, 0.268), (0.722, 0.273), (0.720, 0.500)),  # 90° rot: g00=bottom-left, read up
}
