"""Auto-label the easy zones of a frame from the reconstructed BoardState.

Produces :class:`LabelSample`s for hero hand tiles, dora indicators, the four
score readouts, and round-meta text. Bridge tile names ('E','5mr',...) already
match ``tiles.TILE_NAMES`` so no conversion is needed.

Hand states: settled concealed hands (len % 3 == 1) fill the slots left-to-right;
the hero's-own-turn 14-tile state adds the freshly-drawn tile (``state.drawn_tile``,
tracked by the replayer) in a gapped tsumo slot. Labeling that state matters — the
detector otherwise trains on hero-turn frames with the hand UNLABELED and learns to
suppress it on the player's own turn. The hard 河/副露 zones now live in the precise
fullwarp pipeline (``majsoul_eye.annotate``); ``annotate.frame`` calls this module
only for the hand + dora zones.
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
#   'dora'  : 2D HUD top-left panel — calibrated (T3) but RESOLUTION-DEPENDENT. opt-in.
#   'score'/'meta': 2D HUD — does NOT scale with resolution; needs anchor-relative
#     placement; also exact GT, so low priority. opt-in.
# The hard 河/副露 zones moved to the precise fullwarp pipeline
# (``majsoul_eye.annotate``); this legacy annotator now supplies only hand + dora.
DEFAULT_ZONES = frozenset({"hand"})


def label_frame(frame: np.ndarray, state, region: BoardRegion,
                zones: frozenset[str] = DEFAULT_ZONES) -> list[LabelSample]:
    """Emit label samples for the selected zones. `state` is a BoardState; `region`
    locates the canonical board within `frame`. Only `zones` are emitted (default
    'hand' — the only fully-calibrated zone; see DEFAULT_ZONES)."""
    out: list[LabelSample] = []

    def add(zone, label, kind, box, tile_class=None):
        if zone in zones:
            out.append(LabelSample(zone, str(label), kind, box, region.norm_to_px(box), tile_class))

    # hero hand. Settled concealed states (len % 3 == 1: 13/10/7/…) fill slots
    # left-to-right. On the hero's own turn the freshly-drawn tile
    # (``state.drawn_tile``) renders in a SEPARATED slot to the right of the sorted
    # concealed run — label the concealed tiles in slots 0..n-1 and the draw in the
    # gapped tsumo slot. A 14-tile state WITHOUT a tracked draw (rare / hand-built)
    # is skipped, since the separated slot's identity is unknown.
    hand = state.hero_hand
    drawn = getattr(state, "drawn_tile", None)
    if "hand" in zones and state.hero_seat >= 0 and hand and "?" not in hand:
        if drawn is not None and drawn != "?" and drawn in hand:
            concealed = list(hand)
            concealed.remove(drawn)                 # sorted concealed run (drop one instance)
            for i, tile in enumerate(concealed):
                add("hand", tile, "tile", HAND.slot_box(i), NAME_TO_ID.get(tile))
            add("hand", drawn, "tile", HAND.slot_box(len(concealed), is_tsumo=True), NAME_TO_ID.get(drawn))
        elif len(hand) % 3 == 1:
            for i, tile in enumerate(hand):
                add("hand", tile, "tile", HAND.slot_box(i), NAME_TO_ID.get(tile))

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
