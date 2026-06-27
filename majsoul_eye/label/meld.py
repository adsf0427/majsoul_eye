"""副露 (meld) auto-labeling — a per-seat 1-row strip + GT order (reuses RiverGrid).

Melds render at each player's corner and grow as more are called. Layout differs
per seat (calibrated T5, coords.MELD_STRIPS):
  - self: bottom-right, UPRIGHT, right-anchored, melds newest→oldest (reverse).
  - right/left: along the side edge, 3D-angled, anchored at the avatar, chronological.
  - across: top-left, 180° reflection, anchored at corner, chronological.
The rotated *called* tile is ~1.3× wider, so uniform cells are APPROXIMATE (centers
land on tiles; boxes are a bit loose). Ankan shows 2 face-down 'back' + 2 face-up.
This is a pragmatic v0 — the bootstrap detector later refines meld boxes.
"""

from __future__ import annotations

import numpy as np

from ..coords import MELD_STRIPS
from ..normalize import BoardRegion
from ..tiles import NAME_TO_ID
from .autolabel import LabelSample
from .river import RiverGrid, _screen_to_seat


def _flatten(melds, order: str) -> list[str]:
    """Flatten chronological melds to the on-screen tile-label sequence."""
    groups = list(reversed(melds)) if order == "reverse" else list(melds)
    labels: list[str] = []
    for m in groups:
        if m.type == "ankan":
            t = m.tiles  # 4 identical; shown as back, face, face, back
            labels += ["back", t[1], t[2], "back"]
        else:
            labels += list(m.tiles)
    return labels


def _strip_grid(cfg: dict, k: int) -> RiverGrid:
    """Build a 1-row grid of k cells from a strip calibrated for cfg['count'] tiles,
    honoring the per-seat growth anchor."""
    g = [np.array(p, float) for p in cfg["quad"]]
    step = (g[1] - g[0]) / max(1, cfg["count"])
    h = g[3] - g[0]                       # tile-height direction (g00→g01)
    if cfg["anchor"] == "end":            # g10 fixed; tiles extend back toward g00
        g10 = g[1]
        g00 = g[1] - step * k
    else:                                 # g00 fixed; tiles extend toward g10
        g00 = g[0]
        g10 = g[0] + step * k
    return RiverGrid(tuple(g00), tuple(g10), tuple(g10 + h), tuple(g00 + h), cols=max(1, k), rows=1)


def label_meld(region: BoardRegion, state, seat_pos: str) -> tuple[list[LabelSample], bool]:
    """Auto-label one seat's 副露 from its calibrated strip + GT meld order."""
    cfg = MELD_STRIPS.get(seat_pos)
    if cfg is None:
        return [], True
    seat = _screen_to_seat(state.hero_seat, seat_pos)
    if seat is None or not state.melds[seat]:
        return [], True
    labels = _flatten(state.melds[seat], cfg["order"])
    grid = _strip_grid(cfg, len(labels))
    samples = [
        LabelSample("meld", lab, "tile", grid.cell_box(i), region.norm_to_px(grid.cell_box(i)), NAME_TO_ID.get(lab))
        for i, lab in enumerate(labels)
    ]
    return samples, True
