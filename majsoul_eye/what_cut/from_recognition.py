from __future__ import annotations

from dataclasses import dataclass
import re

from majsoul_eye.recognize.evidence import AssemblyResult
from majsoul_eye.what_cut.schema import (
    HistoryBaselineItemV1, WhatCutDraftV1, WhatCutRecognizerV1,
)


@dataclass(frozen=True)
class DraftBuildContext:
    draft_id: str
    image_ref: str | None
    image_hash: str
    width: int
    height: int


_ITEM_KEY_PATTERNS = (
    (re.compile(r"hand:(0|[1-9][0-9]*)$"),
     lambda match: f"players.0.hand.hand:0:{match.group(1)}.pai"),
    (re.compile(r"dora:(0|[1-9][0-9]*)$"),
     lambda match: f"doraMarkers.dora:{match.group(1)}.pai"),
    (re.compile(r"river:([0-3]):(0|[1-9][0-9]*)$"),
     lambda match: (f"players.{match.group(1)}.rivers."
                    f"river:{match.group(1)}:{match.group(2)}.pai")),
    (re.compile(r"meld:([0-3]):(0|[1-9][0-9]*)$"),
     lambda match: (f"players.{match.group(1)}.melds."
                    f"meld:{match.group(1)}:{match.group(2)}")),
)
_HUD_FIELD_PATHS = {
    "round.bakazeKyoku": ("round.bakaze", "round.kyoku"),
    "round.leftTileCount": ("round.leftTileCount",),
    "round.kyotaku": ("round.kyotaku",),
    "round.honba": ("round.honba",),
    "round.seatWindSelf": ("round.seatWindSelf",),
}


def recognized_field_paths(field_key: str) -> tuple[str, ...]:
    if field_key == "drawn:0":
        return ("players.0.drawnTile.drawn:0.pai",)
    if field_key in _HUD_FIELD_PATHS:
        return _HUD_FIELD_PATHS[field_key]
    score = re.fullmatch(r"round\.scores\.([0-3])", field_key)
    if score is not None:
        return (f"round.scores.{score.group(1)}",)
    nuki = re.fullmatch(r"nuki:([0-3])", field_key)
    if nuki is not None:
        return (f"players.{nuki.group(1)}.nukiCount",)
    for pattern, render in _ITEM_KEY_PATTERNS:
        match = pattern.fullmatch(field_key)
        if match is not None:
            return (render(match),)
    raise ValueError(f"unmapped recognition field_key: {field_key}")


def _draft_editable_field_paths(draft: WhatCutDraftV1) -> set[str]:
    paths = {f"round.{key}" for key in
             ("gameLength", "bakaze", "kyoku", "honba", "kyotaku",
              "leftTileCount", "seatWindSelf")}
    paths.update(f"round.scores.{seat}" for seat in range(4))
    if draft["nPlayers"] == 3:
        paths.update(f"players.{seat}.nukiCount" for seat in range(4))
    paths.update(f"doraMarkers.{item['id']}.pai" for item in draft["doraMarkers"])
    for player in draft["players"]:
        if player["hand"] is not None:
            paths.update(f"players.0.hand.{item['id']}.pai" for item in player["hand"])
        if player["drawnTile"] is not None:
            paths.add(f"players.0.drawnTile.{player['drawnTile']['id']}.pai")
        paths.update(f"players.{player['relSeat']}.rivers.{item['id']}.pai"
                     for item in player["rivers"])
        paths.update(f"players.{player['relSeat']}.melds.{item['id']}"
                     for item in player["melds"])
    return paths


def _mark(value: bool, source: str = "inferred"):
    return {"value": value, "source": source,
            "baselineValue": value, "baselineSource": source}


def _annotation(field, evidence_ids):
    return {"source": "recognized", "confidence": field.confidence,
            "candidates": [{"value": c.value, "confidence": c.confidence}
                           for c in field.candidates],
            "evidenceIds": evidence_ids, "confirmedRevision": None}


def build_recognized_draft(assembly: AssemblyResult, context: DraftBuildContext,
                           recognizer: WhatCutRecognizerV1) -> WhatCutDraftV1:
    observed = assembly.observed
    players = []
    for seat in range(4):
        hand = None
        drawn = None
        if seat == 0:
            hand = [{"id": f"hand:0:{i}", "pai": tile}
                    for i, tile in enumerate(observed.hero_hand)]
            if observed.drawn_tile is not None:
                drawn = {"id": "drawn:0", "pai": observed.drawn_tile}
        rivers = []
        for index, tile in enumerate(observed.rivers[seat]):
            default = True if seat == 0 else False
            rivers.append({"id": f"river:{seat}:{index}", "pai": tile.pai,
                           "sideways": tile.sideways,
                           "tsumogiri": _mark(default)})
        melds = []
        for index, meld in enumerate(observed.melds[seat]):
            melds.append({"id": f"meld:{seat}:{index}", "type": meld.type,
                          "tiles": list(meld.tiles),
                          "calledPai": meld.called_pai or None,
                          "addedPai": meld.added_pai or None,
                          "fromOffset": meld.from_rel})
        players.append({"relSeat": seat, "hand": hand, "drawnTile": drawn,
                        "concealedCount": observed.concealed_counts[seat],
                        "reach": observed.reach[seat], "rivers": rivers,
                        "melds": melds,
                        "nukiCount": observed.nukidora[seat]})

    ghosts = []
    for caller in range(4):
        for meld_index, meld in enumerate(observed.melds[caller]):
            if meld.type not in ("chi", "pon", "daiminkan", "kakan"):
                continue
            ghosts.append({"id": f"ghost:{caller}:{meld_index}",
                           "ownerRelSeat": (caller + meld.from_rel) % 4,
                           "pai": meld.called_pai,
                           "beforeMeldId": f"meld:{caller}:{meld_index}",
                           "tsumogiri": _mark(False)})

    draft: WhatCutDraftV1 = {
            "schemaVersion": 2, "draftId": context.draft_id, "revision": 0,
            "nPlayers": 3 if observed.sanma else 4,
            "seatFrame": "screen-relative",
            "source": {"kind": "screenshot", "imageRef": context.image_ref,
                       "imageHash": context.image_hash, "width": context.width,
                       "height": context.height},
            "recognizer": recognizer,
            "round": {"gameLength": "hanchan", "bakaze": observed.bakaze,
                      "kyoku": observed.kyoku, "honba": observed.honba,
                      "kyotaku": observed.kyotaku,
                      "leftTileCount": observed.left_tile_count,
                      "seatWindSelf": observed.seat_wind_self,
                      "scores": list(observed.scores) if observed.scores is not None
                                else [None, None, None, None],
                      "phantomRelSeat": observed.phantom_rel
                                        if observed.sanma else None},
            "doraMarkers": [{"id": f"dora:{i}", "pai": tile}
                            for i, tile in enumerate(observed.dora_markers)],
            "players": players, "annotations": {},
            "evidence": [],
            "historyOverrides": {"ghostDiscards": ghosts}}

    editable_paths = _draft_editable_field_paths(draft)
    for field in assembly.fields:
        field_paths = recognized_field_paths(field.field_key)
        missing = [path for path in field_paths if path not in editable_paths]
        if missing:
            raise ValueError(
                f"recognition field_key {field.field_key} maps to missing draft paths: {missing}")
        evidence_ids = []
        for index, detection in enumerate(field.detections):
            # Compound HUD fields share this primary stable-path evidence ID.
            evidence_id = f"e:{field_paths[0]}:{index}"
            evidence_ids.append(evidence_id)
            draft["evidence"].append({
                "id": evidence_id,
                "bbox": [float(v) for v in detection.xyxy],
                "polygon": ([[float(x), float(y)] for x, y in detection.poly]
                            if detection.poly is not None else None),
                "zone": field.field_key.split(":", 1)[0].split(".", 1)[0],
            })
        for field_path in field_paths:
            if field_path in draft["annotations"]:
                raise ValueError(f"duplicate recognition fieldPath: {field_path}")
            draft["annotations"][field_path] = _annotation(field, evidence_ids)
    return draft


def apply_history_baseline(draft: WhatCutDraftV1,
                           baseline: list[HistoryBaselineItemV1]) -> None:
    marks = {}
    expected_order = []
    for player in draft["players"]:
        for discard in player["rivers"]:
            marks[("river", discard["id"])] = discard["tsumogiri"]
            expected_order.append(("river", discard["id"]))
    for ghost in draft["historyOverrides"]["ghostDiscards"]:
        marks[("ghost", ghost["id"])] = ghost["tsumogiri"]
        expected_order.append(("ghost", ghost["id"]))
    actual_order = [(item["itemKind"], item["itemId"]) for item in baseline]
    if actual_order != expected_order or len(set(actual_order)) != len(expected_order):
        raise ValueError(f"historyBaseline cardinality/order mismatch: expected {expected_order}, got {actual_order}")
    for item in baseline:
        mark = marks[(item["itemKind"], item["itemId"])]
        mark["baselineValue"] = item["baselineValue"]
        mark["baselineSource"] = item["baselineSource"]
        if mark["source"] != "user":
            mark["value"] = item["baselineValue"]
            mark["source"] = item["baselineSource"]
