"""Auto-label the easy zones of a frame from the reconstructed BoardState.

Produces :class:`LabelSample`s for hero hand tiles, dora indicators, the four
score readouts, and round-meta text. Bridge tile names ('E','5mr',...) already
match ``tiles.TILE_NAMES`` so no conversion is needed.

Scope (skeleton): hand is labeled only on *settled* concealed states
(len % 3 == 1, i.e. no separated drawn tile) — the drawn-tile (14-tile) layout
puts the tsumo in a gapped slot and needs the unsorted draw tracked; TODO once
session2 frames confirm the gap geometry. Hard zones (河/副露) come in P4.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..coords import HAND, MAX_DORA, REGIONS, NormBox, dora_slot
from ..normalize import BoardRegion
from ..tiles import NAME_TO_ID


@dataclass
class LabelSample:
    zone: str                       # 'hand' | 'dora' | 'score' | 'meta'
    label: str                      # tile name, score string, or meta value
    kind: str                       # 'tile' (classify/detect) | 'text' (OCR/digits)
    norm_box: NormBox               # normalized box on the canonical board
    px_box: tuple[int, int, int, int]   # resolved pixel box on the frame
    tile_class: Optional[int] = None    # tiles.NAME_TO_ID for kind=='tile'


def _seat_screen_map(hero: int) -> dict[str, int]:
    """Map score-panel screen position -> absolute seat (counter-clockwise)."""
    return {
        "self": hero,
        "right": (hero + 1) % 4,   # shimocha 下家
        "across": (hero + 2) % 4,  # toimen 対面
        "left": (hero + 3) % 4,    # kamicha 上家
    }


# Zones whose normalized coords are calibrated and resolution-stable.
#   'hand'  : 3D-table element — scales linearly, calibrated (session2). ✓
#   'river' : 3D-table 河 — per-seat perspective grid, calibrated (session6 T2). ✓
#   'meld' : 3D-table 副露 — per-seat strip, calibrated (session6 T5). OPT-IN:
#     self/left are decent but right/across (3D-angled side seats + the rotated
#     called-tile gap) give loose boxes (~88-93% on-tile) — too noisy for default.
#     Refine via bootstrap (detector trained on hand+river relabels meld tiles).
#   'dora'  : 2D HUD top-left panel — calibrated (T3) but RESOLUTION-DEPENDENT. opt-in.
#   'score'/'meta': 2D HUD — does NOT scale with resolution; needs anchor-relative
#     placement; also exact GT, so low priority. opt-in.
DEFAULT_ZONES = frozenset({"hand", "river"})


def label_frame(frame: np.ndarray, state, region: BoardRegion,
                zones: frozenset[str] = DEFAULT_ZONES) -> list[LabelSample]:
    """Emit label samples for the selected zones. `state` is a BoardState; `region`
    locates the canonical board within `frame`. Only `zones` are emitted (default
    'hand' — the only fully-calibrated zone; see DEFAULT_ZONES)."""
    out: list[LabelSample] = []

    def add(zone, label, kind, box, tile_class=None):
        if zone in zones:
            out.append(LabelSample(zone, str(label), kind, box, region.norm_to_px(box), tile_class))

    # hero hand (settled concealed states only)
    hand = state.hero_hand
    if "hand" in zones and state.hero_seat >= 0 and hand and "?" not in hand and len(hand) % 3 == 1:
        for i, tile in enumerate(hand):
            add("hand", tile, "tile", HAND.slot_box(i), NAME_TO_ID.get(tile))

    # four discard rivers (per-seat perspective grid + GT order)
    if "river" in zones and state.hero_seat >= 0:
        from .river import RiverGrid, label_river   # lazy: avoids import cycle
        from ..coords import RIVER_QUADS
        for sp in ("self", "across", "left", "right"):
            if sp in RIVER_QUADS:
                rs, _ = label_river(region, state, sp, RiverGrid(*RIVER_QUADS[sp]))
                out.extend(rs)

    # four meld (副露) strips
    if "meld" in zones and state.hero_seat >= 0:
        from .meld import label_meld           # lazy: avoids import cycle
        for sp in ("self", "across", "left", "right"):
            ms, _ = label_meld(region, state, sp)
            out.extend(ms)

    # dora indicators
    for i, d in enumerate(state.dora_markers[:MAX_DORA]):
        add("dora", d, "tile", dora_slot(i), NAME_TO_ID.get(d))

    # scores (4 seats by screen quadrant)
    if state.hero_seat >= 0:
        smap = _seat_screen_map(state.hero_seat)
        for pos, key in (("self", "score_self"), ("right", "score_right"), ("across", "score_across")):
            if key in REGIONS:
                add("score", state.scores[smap[pos]], "text", REGIONS[key])

    # round-meta text
    if state.bakaze is not None:
        add("meta", state.bakaze, "text", REGIONS["round_wind"])
        add("meta", state.kyoku, "text", REGIONS["round_number"])
        add("meta", state.honba, "text", REGIONS["honba"])
        add("meta", state.kyotaku, "text", REGIONS["riichi_sticks"])

    return out


def save_classification_crops(frame: np.ndarray, region: BoardRegion,
                              samples: list[LabelSample], out_dir: str, prefix: str = "") -> int:
    """Write tile crops to ``out_dir/<label>/<prefix><idx>.png`` for the classifier
    dataset. Returns the number of crops written. Requires cv2 (falls back to .npy)."""
    n = 0
    for i, s in enumerate(samples):
        if s.kind != "tile":
            continue
        crop = region.crop(frame, s.norm_box)
        if crop.size == 0:
            continue
        d = os.path.join(out_dir, s.label)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{prefix}{i:02d}.png")
        try:
            import cv2  # type: ignore
            cv2.imwrite(path, crop)
        except Exception:
            np.save(path.replace(".png", ".npy"), crop)
        n += 1
    return n


def to_yolo_lines(samples: list[LabelSample]) -> list[str]:
    """YOLO label lines ('class cx cy w h', normalized) for tile samples."""
    lines = []
    for s in samples:
        if s.kind == "tile" and s.tile_class is not None:
            cx, cy, w, h = s.norm_box.yolo()
            lines.append(f"{s.tile_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines
