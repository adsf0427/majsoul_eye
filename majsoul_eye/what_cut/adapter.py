from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from majsoul_eye.state.history import ReconstructionOverrides, UserTsumogiriOverride
from majsoul_eye.state.observe import ObservedMeld, ObservedRiverTile, ObservedState
from majsoul_eye.tiles import red_to_normal
from majsoul_eye.what_cut.schema import WhatCutDraftV1, WhatCutIssueV1


@dataclass
class DraftAdapterResult:
    observed: ObservedState | None
    overrides: ReconstructionOverrides
    issues: list[WhatCutIssueV1]


def _issue(code: str, field_path: str, **params) -> WhatCutIssueV1:
    return {"code": code, "severity": "blocking", "fieldPath": field_path,
            "evidenceIds": [], "messageKey": f"whatCut.issue.{code}",
            "params": params}


def _valid_meld(meld: dict, owner: int, path: str) -> list[WhatCutIssueV1]:
    tiles = meld["tiles"]
    norm = [red_to_normal(t) for t in tiles]
    kind = meld["type"]
    issues: list[WhatCutIssueV1] = []
    if kind == "chi":
        ranks = sorted(int(t[0]) for t in norm if len(t) == 2 and t[0].isdigit())
        suits = {t[1] for t in norm if len(t) == 2}
        if (len(tiles) != 3 or len(ranks) != 3 or len(suits) != 1
                or ranks[1] != ranks[0] + 1 or ranks[2] != ranks[1] + 1):
            issues.append(_issue("INVALID_CHI", path))
        if meld["fromOffset"] != 3:
            issues.append(_issue("INVALID_CHI_SOURCE", f"{path}.fromOffset"))
    elif kind == "pon" and (len(tiles) != 3 or len(set(norm)) != 1):
        issues.append(_issue("INVALID_PON", path))
    elif kind in ("daiminkan", "ankan", "kakan") and (
            len(tiles) != 4 or len(set(norm)) != 1):
        issues.append(_issue("INVALID_KAN", path))
    if kind == "ankan" and meld["fromOffset"] != 0:
        issues.append(_issue("INVALID_ANKAN_SOURCE", f"{path}.fromOffset"))
    if (kind in ("pon", "daiminkan", "kakan")
            and meld["fromOffset"] not in (1, 2, 3)):
        issues.append(_issue("INVALID_MELD_SOURCE", f"{path}.fromOffset"))
    if (kind in ("chi", "pon", "daiminkan", "kakan")
            and meld["calledPai"] not in tiles):
        issues.append(_issue("CALLED_TILE_NOT_IN_MELD", f"{path}.calledPai"))
    if kind == "ankan" and meld["calledPai"] is not None:
        issues.append(_issue("UNEXPECTED_CALLED_TILE", f"{path}.calledPai"))
    if kind == "kakan":
        if meld["addedPai"] not in tiles:
            issues.append(_issue("ADDED_TILE_NOT_IN_MELD", f"{path}.addedPai"))
    elif meld["addedPai"] is not None:
        issues.append(_issue("UNEXPECTED_ADDED_TILE", f"{path}.addedPai"))
    if Counter(norm).most_common(1) and Counter(norm).most_common(1)[0][1] > 4:
        issues.append(_issue("INVALID_MELD_TILE_COUNT", path, owner=owner))
    return issues


def draft_to_observed(draft: WhatCutDraftV1) -> DraftAdapterResult:
    issues: list[WhatCutIssueV1] = []
    overrides = ReconstructionOverrides()
    observed = ObservedState()
    round_ = draft["round"]
    # v1 drafts carry neither key; they are four-player by construction.
    # Game-logic violations (occupied phantom, 2m-8m, chi, …) are check_observed's
    # job downstream — this projection only carries the declared mode.
    observed.sanma = draft["nPlayers"] == 3
    observed.phantom_rel = round_.get("phantomRelSeat") if observed.sanma else None
    observed.bakaze = round_["bakaze"]
    observed.kyoku = round_["kyoku"]
    observed.honba = round_["honba"]
    observed.kyotaku = round_["kyotaku"]
    observed.left_tile_count = round_["leftTileCount"]
    observed.seat_wind_self = round_["seatWindSelf"]
    scores = round_["scores"]
    if all(v is None for v in scores):
        observed.scores = None
    elif len(scores) == 4 and all(isinstance(v, int) for v in scores):
        observed.scores = [int(v) for v in scores]
    else:
        issues.append(_issue("PARTIAL_SCORES", "round.scores"))

    for tile in draft["doraMarkers"]:
        if tile["pai"] is None:
            issues.append(_issue("MISSING_TILE", f"doraMarkers.{tile['id']}.pai"))
        else:
            observed.dora_markers.append(tile["pai"])

    meld_by_id: dict[str, tuple[int, int, dict]] = {}
    for seat, player in enumerate(draft["players"]):
        observed.reach[seat] = player["reach"]
        observed.concealed_counts[seat] = player["concealedCount"]
        observed.nukidora[seat] = player.get("nukiCount", 0)
        if seat == 0:
            if player["hand"] is None:
                issues.append(_issue("MISSING_HERO_HAND", "players.0.hand"))
            else:
                for tile in player["hand"]:
                    if tile["pai"] is None:
                        issues.append(_issue(
                            "MISSING_TILE", f"players.0.hand.{tile['id']}.pai"))
                    else:
                        observed.hero_hand.append(tile["pai"])
            draw = player["drawnTile"]
            if draw is not None:
                if draw["pai"] is None:
                    issues.append(_issue(
                        "MISSING_TILE", f"players.0.drawnTile.{draw['id']}.pai"))
                else:
                    observed.drawn_tile = draw["pai"]
        for ri, discard in enumerate(player["rivers"]):
            overrides.river_ids[(seat, ri)] = discard["id"]
            if discard["tsumogiri"]["source"] == "user":
                overrides.user_visible[(seat, ri)] = UserTsumogiriOverride(
                    discard["tsumogiri"]["value"], discard["id"],
                    f"players.{seat}.rivers.{discard['id']}.tsumogiri")
            if discard["pai"] is None:
                issues.append(_issue(
                    "MISSING_TILE", f"players.{seat}.rivers.{discard['id']}.pai"))
            else:
                observed.rivers[seat].append(ObservedRiverTile(
                    discard["pai"], sideways=discard["sideways"]))
        for mi, meld in enumerate(player["melds"]):
            path = f"players.{seat}.melds.{meld['id']}"
            issues.extend(_valid_meld(meld, seat, path))
            meld_by_id[meld["id"]] = (seat, mi, meld)
            observed.melds[seat].append(ObservedMeld(
                meld["type"], list(meld["tiles"]), meld["calledPai"] or "",
                meld["addedPai"] or "", meld["fromOffset"]))

    seen_ghosts = set()
    for ghost in draft["historyOverrides"]["ghostDiscards"]:
        linked = meld_by_id.get(ghost["beforeMeldId"])
        path = f"historyOverrides.ghostDiscards.{ghost['id']}"
        if linked is None:
            issues.append(_issue(
                "GHOST_MELD_NOT_FOUND", f"{path}.beforeMeldId"))
            continue
        caller, meld_index, meld = linked
        expected_owner = (caller + meld["fromOffset"]) % 4
        if ghost["ownerRelSeat"] != expected_owner or ghost["pai"] != meld["calledPai"]:
            issues.append(_issue(
                "GHOST_MELD_MISMATCH", path, expectedOwner=expected_owner))
            continue
        key = (caller, meld_index)
        if key in seen_ghosts:
            issues.append(_issue(
                "GHOST_DUPLICATE", f"{path}.beforeMeldId"))
            continue
        seen_ghosts.add(key)
        overrides.ghost_ids[key] = ghost["id"]
        overrides.ghost_order.append(key)
        if ghost["tsumogiri"]["source"] == "user":
            overrides.user_ghosts[key] = UserTsumogiriOverride(
                ghost["tsumogiri"]["value"], ghost["id"],
                f"{path}.tsumogiri")

    required_ghosts = {
        (seat, index)
        for seat in range(4)
        for index, meld in enumerate(observed.melds[seat])
        if meld.type in ("chi", "pon", "daiminkan", "kakan")
    }
    for caller, meld_index in sorted(required_ghosts - seen_ghosts):
        meld_id = draft["players"][caller]["melds"][meld_index]["id"]
        issues.append(_issue(
            "GHOST_REQUIRED", f"players.{caller}.melds.{meld_id}"))

    return DraftAdapterResult(
        None if any(i["severity"] == "blocking" for i in issues) else observed,
        overrides,
        issues,
    )
