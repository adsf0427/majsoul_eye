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
        the bottom of a cell — see scripts/train/build_dataset.py --river-erode-*)."""
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

# Dora indicators are NOT on the dead wall — they're in the TOP-LEFT HUD panel,
# growing rightward as kan-dora reveal. NOTE: this is 2D HUD, so it's
# RESOLUTION-DEPENDENT (like scores) — fine at this 16:9 res; needs
# anchor-normalization for other resolutions.
# x-extent data-calibrated: brightness-scan of the 5-slot strip on session6 puts
# tile seams at px 82/138/194/250/306 → pitch ~56 px (0.0292), first edge ~28 px.
# The earlier single-tile 0.0395 width was ~33% too wide (boxes bled into the next
# slot), so x1 pulled 0.2125 → 0.1608 (5 × 56 px).
DORA_STRIP: NormBox = NormBox(0.015, 0.036, 0.1608, 0.111)
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
# x0 data-calibrated: brightness-scan of the first-tile left edge over 132 rendered
# session6 hands gives 223 px (mycv's 235 seed sat +12 px right); slot width 94.5≈95.
@dataclass(frozen=True)
class HandModel:
    x0: float = 223 / 1920          # left edge of first tile (measured)
    slot_w: float = 95 / 1920       # tile width (also slot pitch)  # measured 94.5
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

# NOTE: the per-seat 河/副露 geometry (``RIVER_QUADS`` / ``MELD_STRIPS``, the
# equal-subdivision RiverGrid model) was removed here — it is superseded by the
# precise fullwarp annotator in ``majsoul_eye.annotate`` (data-calibrated
# DISCARD_GRID / composition-aware melds). See docs/STATUS.md §1.13.
