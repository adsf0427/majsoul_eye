"""Is this screenshot a 3-player board or a 4-player one, and which chair is empty?

THE PRIMARY SIGNAL IS FREE. Majsoul's centre panel carries four separate score
plates and the detector has a distinct class for each (`score_self`, `score_right`,
`score_across`, `score_left`). A sanma table renders only THREE: the phantom chair
has no plate. So "exactly one score-plate class is missing" is simultaneously the
mode signal AND the locator of the empty chair — with no new training data, no new
class, and no new geometry.

THE MODE IS A HYPOTHESIS; ``check_observed`` IS THE VERIFIER. Nothing here is
trusted downstream. A board read in the wrong mode fails the 105000 points
identity, the 55-tile wall identity, the no-chi rule and the tile-set rule — all
of them AFTER assembly. That layering is deliberate: a mis-detected mode must not
be able to hand back a confident, coherent, WRONG board, which is the one failure
this whole feature cannot tolerate (see docs; the same reasoning drives the
board-rect coverage check).

Corroboration is therefore a VETO, never a vote: a single contradicting piece of
evidence blocks, it does not get outvoted. When the evidence disagrees we refuse
and say so — we never pick the likelier story.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from majsoul_eye.annotate import pipeline as P
from majsoul_eye.recognize.assemble import (ZONE_RADIUS, _fw_points,
                                            _nuki_distance, zone_distances)
from majsoul_eye.tiles import red_to_normal

# Score-plate detector classes, in SCREEN-RELATIVE seat order.
_PLATES = ("score_self", "score_right", "score_across", "score_left")
# Tiles that cannot exist on a sanma table (5mr normalizes to 5m).
_NO_SANMA_TILES = {f"{rank}m" for rank in range(2, 9)}

AUTO, THREE_P, FOUR_P = "auto", "3p", "4p"
_VALID_OVERRIDES = (AUTO, THREE_P, FOUR_P)


@dataclass
class ModeDecision:
    sanma: bool = False
    phantom_rel: int | None = None
    source: str = "plates"                # plates | override
    issues: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i["severity"] == "blocking" for i in self.issues)


def _issue(code: str, message: str, **params) -> dict:
    return {"code": code, "severity": "blocking", "fieldPath": None,
            "evidenceIds": [], "messageKey": f"whatCut.issue.{code}",
            "params": {"message": message, **params}}


def _plate_seats(dets) -> set[int]:
    return {_PLATES.index(d.name) for d in dets if d.name in _PLATES}


def _tile_dets(dets):
    return [d for d in dets if getattr(d, "tile", None)
            and d.tile not in ("back",) and d.name not in _PLATES]


def _phantom_chair_is_occupied(dets, region, phantom_rel: int) -> bool:
    """Anything rendered in the would-be empty chair refutes sanma outright."""
    Hs = P.build_homographies(1920, 1080)
    geom = P.GEOMETRY_3P
    for det in _tile_dets(dets):
        pts = _fw_points(det, region, Hs["H_full"])
        centre = pts.mean(axis=0)
        d_river, d_meld = zone_distances(centre, phantom_rel, geom)
        if min(d_river, d_meld) <= ZONE_RADIUS:
            return True
        if det.tile == "N" and _nuki_distance(centre, phantom_rel, geom) <= ZONE_RADIUS:
            return True
    return False


def detect_mode(dets, region, override: str = AUTO) -> ModeDecision:
    """Detections -> (sanma?, which chair is empty), or a blocking issue."""
    if override not in _VALID_OVERRIDES:
        return ModeDecision(issues=[_issue(
            "MODE_OVERRIDE_INVALID",
            f"board mode must be one of {_VALID_OVERRIDES}, got {override!r}")])

    plates = _plate_seats(dets)
    has_babei = any(d.name == "btn_babei" for d in dets)
    illegal_tiles = sorted({d.tile for d in _tile_dets(dets)
                            if red_to_normal(d.tile) in _NO_SANMA_TILES})

    # --- hypothesis ---------------------------------------------------------
    if override == FOUR_P:
        hypothesis, phantom, source = False, None, "override"
    elif override == THREE_P:
        hypothesis, source = True, "override"
        phantom = next(iter({0, 1, 2, 3} - plates), None) if len(plates) == 3 else None
        if phantom is None or phantom == 0:
            return ModeDecision(issues=[_issue(
                "PHANTOM_SEAT_UNKNOWN",
                "3-player mode was forced but the empty chair cannot be located: "
                f"score plates were read for seats {sorted(plates)}")])
    else:
        source = "plates"
        if len(plates) == 4:
            hypothesis, phantom = False, None
        elif len(plates) == 3 and 0 in plates:
            hypothesis = True
            phantom = next(iter({0, 1, 2, 3} - plates))
        elif len(plates) == 3:
            # The hero is never the phantom: their own plate is always on screen.
            return ModeDecision(issues=[_issue(
                "SCORE_PLATE_SELF_MISSING",
                "the hero's own score plate was not found, so the board cannot be "
                "read; try a less cropped screenshot")])
        else:
            return ModeDecision(issues=[_issue(
                "SCORE_PLATES_UNREADABLE",
                f"only {len(plates)} of 4 score plates were found; the player count "
                f"cannot be determined from this screenshot")])

    # --- vetoes (each REFUTES; none of them votes) ---------------------------
    if hypothesis:
        if illegal_tiles:
            return ModeDecision(issues=[_issue(
                "SANMA_MODE_CONTRADICTED",
                f"this looks like a 3-player board, but {', '.join(illegal_tiles)} "
                f"was detected and those tiles do not exist in 3-player mahjong",
                tiles=illegal_tiles)])
        if phantom is not None and _phantom_chair_is_occupied(dets, region, phantom):
            return ModeDecision(issues=[_issue(
                "PHANTOM_SEAT_NOT_EMPTY",
                "this looks like a 3-player board, but tiles were detected in the "
                "chair that should be empty")])
    else:
        if has_babei:
            return ModeDecision(issues=[_issue(
                "SANMA_MODE_CONTRADICTED",
                "this looks like a 4-player board, but the 北抜き (north pull) "
                "button was detected, and that only exists in 3-player mahjong")])

    return ModeDecision(sanma=hypothesis, phantom_rel=phantom, source=source)
