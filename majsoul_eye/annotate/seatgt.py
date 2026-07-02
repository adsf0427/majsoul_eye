"""Ground-truth plumbing for the precise annotator.

Map a screen position (self/right/across/left) to the absolute seat and pull that
seat's visible river / riichi-sideways index / melds from a replayed BoardState.

``seat_gt`` was moved verbatim out of ``scripts/annotate/calibrate_annotation_model.py``;
``_screen_to_seat`` / ``SEAT_POS`` were relocated here from the now-removed
``majsoul_eye.label.river`` so the package owns the seat mapping and neither the
package nor the scripts import from a deprecated module.
"""
from __future__ import annotations

from typing import Optional

from majsoul_eye.annotate import pipeline as P

SEAT_POS = ["self", "right", "across", "left"]


def _screen_to_seat(hero: int, seat_pos: str) -> Optional[int]:
    """Absolute seat of a screen position, counter-clockwise from the hero."""
    if hero < 0:
        return None
    return {"self": hero, "right": (hero + 1) % 4,
            "across": (hero + 2) % 4, "left": (hero + 3) % 4}.get(seat_pos)


def seat_gt(state, pos: int):
    """(visible_river, sideways_idx, melds, abs_seat) for screen position pos."""
    seat = _screen_to_seat(state.hero_seat, SEAT_POS[pos])
    river = [{"pai": t.pai, "tsumogiri": bool(t.tsumogiri), "riichi": bool(t.riichi)}
             for t in state.visible_river(seat)]
    full = [{"pai": t.pai, "riichi": bool(t.riichi), "called": bool(t.called)}
            for t in state.rivers[seat]]
    melds = [{"type": m.type, "tiles": list(m.tiles),
              "from_seat": (pos + ((m.from_seat - seat) % 4)) % 4,
              "called_pai": m.called_pai, "added_pai": m.added_pai}
             for m in state.melds[seat]]
    return river, P.river_sideways_index(full), melds, seat
