"""Ground-truth plumbing for the precise annotator.

Map a screen position (self/right/across/left) to the absolute seat and pull that
seat's visible river / riichi-sideways index / melds from a replayed BoardState.

Moved verbatim out of ``scripts/annotate/calibrate_annotation_model.py`` so the package
(``annotate.frame``) and both scripts share one definition instead of the script
importing from another script.
"""
from __future__ import annotations

from majsoul_eye.annotate import pipeline as P
from majsoul_eye.label.river import _screen_to_seat  # relocated in the deprecation PR

SEAT_POS = ["self", "right", "across", "left"]


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
