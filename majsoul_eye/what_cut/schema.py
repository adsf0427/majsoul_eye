from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Literal, Mapping, TypedDict, cast

from majsoul_eye.tiles import TILE_NAMES

SCHEMA_VERSION: Literal[1] = 1
_TILES = set(TILE_NAMES) - {"back"}
CurrentSource = Literal["forced", "inferred", "user"]
BaselineSource = Literal["forced", "inferred"]


class DraftSchemaError(ValueError):
    def __init__(self, code: str, path: str, message: str):
        super().__init__(message)
        self.code, self.path = code, path


class WhatCutCandidateV1(TypedDict):
    value: Any
    confidence: float


class WhatCutFieldAnnotationV1(TypedDict):
    source: Literal["recognized", "user", "default", "inferred", "forced"]
    confidence: float | None
    candidates: list[WhatCutCandidateV1]
    evidenceIds: list[str]
    confirmedRevision: int | None


class WhatCutEvidenceV1(TypedDict):
    id: str
    bbox: list[float]
    polygon: list[list[float]] | None
    zone: str


class WhatCutTsumogiriV1(TypedDict):
    value: bool
    source: CurrentSource
    baselineValue: bool
    baselineSource: BaselineSource


class WhatCutTileV1(TypedDict):
    id: str
    pai: str | None


class WhatCutDiscardV1(TypedDict):
    id: str
    pai: str | None
    sideways: bool
    tsumogiri: WhatCutTsumogiriV1


class WhatCutMeldV1(TypedDict):
    id: str
    type: Literal["chi", "pon", "daiminkan", "ankan", "kakan"]
    tiles: list[str]
    calledPai: str | None
    addedPai: str | None
    fromOffset: Literal[0, 1, 2, 3]


class WhatCutGhostDiscardV1(TypedDict):
    id: str
    ownerRelSeat: Literal[0, 1, 2, 3]
    pai: str
    beforeMeldId: str
    tsumogiri: WhatCutTsumogiriV1


class WhatCutHistoryOverridesV1(TypedDict):
    ghostDiscards: list[WhatCutGhostDiscardV1]


class WhatCutPlayerV1(TypedDict):
    relSeat: Literal[0, 1, 2, 3]
    hand: list[WhatCutTileV1] | None
    drawnTile: WhatCutTileV1 | None
    concealedCount: int | None
    reach: bool
    rivers: list[WhatCutDiscardV1]
    melds: list[WhatCutMeldV1]


class WhatCutRoundV1(TypedDict):
    gameLength: Literal["hanchan", "tonpu"] | None
    bakaze: Literal["E", "S", "W", "N"] | None
    kyoku: int | None
    honba: int | None
    kyotaku: int | None
    leftTileCount: int | None
    seatWindSelf: Literal["E", "S", "W", "N"] | None
    scores: list[int | None]


class WhatCutSourceV1(TypedDict):
    kind: Literal["screenshot", "manual"]
    imageRef: str | None
    imageHash: str | None
    width: int | None
    height: int | None


class WhatCutRecognizerV1(TypedDict):
    manifestVersion: str
    layoutId: str
    detectorSha: str
    classifierSha: str | None
    hudReaderSha: str | None
    eyeRevision: str
    supportStatus: Literal["experimental", "supported"]


class WhatCutIssueV1(TypedDict):
    code: str
    severity: Literal["blocking", "warning", "uncertain"]
    fieldPath: str | None
    evidenceIds: list[str]
    messageKey: str
    params: dict[str, str | int | float]


class WhatCutDraftV1(TypedDict):
    schemaVersion: Literal[1]
    draftId: str
    revision: int
    nPlayers: Literal[4]
    seatFrame: Literal["screen-relative"]
    source: WhatCutSourceV1
    recognizer: WhatCutRecognizerV1 | None
    round: WhatCutRoundV1
    doraMarkers: list[WhatCutTileV1]
    players: list[WhatCutPlayerV1]
    annotations: dict[str, WhatCutFieldAnnotationV1]
    evidence: list[WhatCutEvidenceV1]
    historyOverrides: WhatCutHistoryOverridesV1


class RecognizeWhatCutData(TypedDict):
    schemaVersion: Literal[1]
    draft: WhatCutDraftV1
    issues: list[WhatCutIssueV1]
    recognizer: WhatCutRecognizerV1


class HistoryBaselineItemV1(TypedDict):
    itemKind: Literal["river", "ghost"]
    itemId: str
    baselineValue: bool
    baselineSource: BaselineSource


class SelectedHistoryOpV1(TypedDict):
    kind: Literal["draw", "river", "ghost", "call", "ankan", "kakan"]
    actorRelSeat: Literal[0, 1, 2, 3]
    targetRelSeat: Literal[0, 1, 2, 3] | None
    riverIndex: int | None
    meldIndex: int | None
    pai: str | None
    tsumogiri: bool | None
    reach: bool


class SelectedHistoryV1(TypedDict):
    solverVersion: Literal["hidden-history-v1"]
    oyaRelSeat: Literal[0, 1, 2, 3]
    pendingReachRelSeat: Literal[0, 1, 2, 3] | None
    heroHaipai: list[str]
    operations: list[SelectedHistoryOpV1]


class FabricatedHistoryV1(TypedDict):
    defaultedRoundFields: list[str]
    heroHiddenDrawCount: int
    opponentUnknownDrawCount: int
    inferredRiverCount: int
    inferredGhostCount: int


class WhatCutDecisionV1(TypedDict):
    actorRelSeat: Literal[0]
    kind: Literal["action"]
    legalDiscards: list[str]
    legalActions: list[str]
    candidateCount: int


class ReconstructWhatCutData(TypedDict):
    schemaVersion: Literal[1]
    revision: int
    ok: bool
    issues: list[WhatCutIssueV1]
    mjai: list[dict[str, Any]] | None
    heroSeatAbs: int | None
    fabricated: FabricatedHistoryV1 | None
    historyBaseline: list[HistoryBaselineItemV1]
    selectedHistory: SelectedHistoryV1 | None
    decision: WhatCutDecisionV1 | None


class WorkerErrorBodyV1(TypedDict):
    code: str
    message: str
    requestId: str


class WorkerErrorV1(TypedDict):
    schemaVersion: Literal[1]
    error: WorkerErrorBodyV1


def _schema_error(code: str, path: str, message: str) -> DraftSchemaError:
    return DraftSchemaError(code, path, message)


def _require_dict(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _schema_error("INVALID_DRAFT", path, "expected object")
    return value


def _require_keys(value: Mapping[str, Any], path: str, expected: set[str]) -> None:
    actual = set(value)
    if actual != expected:
        raise _schema_error(
            "INVALID_DRAFT", path,
            f"expected keys {sorted(expected)}, got {sorted(actual)}",
        )


def _require_id(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _schema_error("INVALID_DRAFT", path, "expected non-empty id")
    return value


def _validate_mark(value: Any, path: str) -> None:
    mark = _require_dict(value, path)
    _require_keys(
        mark, path,
        {"value", "source", "baselineValue", "baselineSource"},
    )
    if not isinstance(mark.get("value"), bool) or not isinstance(
            mark.get("baselineValue"), bool):
        raise _schema_error("INVALID_DRAFT", path, "history values must be boolean")
    if mark.get("source") not in ("forced", "inferred", "user"):
        raise _schema_error("INVALID_DRAFT", f"{path}.source", "invalid current source")
    if mark.get("baselineSource") not in ("forced", "inferred"):
        raise _schema_error(
            "INVALID_DRAFT", f"{path}.baselineSource", "invalid baseline source",
        )


def _validate_tile(value: Any, path: str, *, nullable: bool) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str) or value not in _TILES:
        raise _schema_error("INVALID_DRAFT", path, f"invalid tile {value!r}")


def _is_finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def parse_what_cut_draft(payload: Mapping[str, Any]) -> WhatCutDraftV1:
    root = _require_dict(payload, "draft")
    _require_keys(root, "draft", {"schemaVersion", "draftId", "revision",
        "nPlayers", "seatFrame", "source", "recognizer", "round",
        "doraMarkers", "players", "annotations", "evidence",
        "historyOverrides"})
    if root.get("schemaVersion") != SCHEMA_VERSION:
        raise _schema_error(
            "UNSUPPORTED_SCHEMA", "schemaVersion",
            "only schemaVersion 1 is supported",
        )
    if root.get("nPlayers") != 4 or root.get("seatFrame") != "screen-relative":
        raise _schema_error(
            "INVALID_DRAFT", "nPlayers",
            "draft must be screen-relative four-player",
        )
    _require_id(root.get("draftId"), "draftId")
    if type(root.get("revision")) is not int or root["revision"] < 0:
        raise _schema_error(
            "INVALID_DRAFT", "revision",
            "revision must be a non-negative integer",
        )
    source = _require_dict(root.get("source"), "source")
    _require_keys(
        source, "source", {"kind", "imageRef", "imageHash", "width", "height"},
    )
    if source["kind"] not in ("screenshot", "manual"):
        raise _schema_error("INVALID_DRAFT", "source.kind", "invalid source kind")
    for key in ("imageRef", "imageHash"):
        if source[key] is not None and not isinstance(source[key], str):
            raise _schema_error(
                "INVALID_DRAFT", f"source.{key}", "expected string or null",
            )
    for key in ("width", "height"):
        if source[key] is not None and (
                type(source[key]) is not int or source[key] <= 0):
            raise _schema_error(
                "INVALID_DRAFT", f"source.{key}",
                "expected positive integer or null",
            )
    recognizer = root.get("recognizer")
    if recognizer is not None:
        recognizer = _require_dict(recognizer, "recognizer")
        _require_keys(recognizer, "recognizer", {"manifestVersion", "layoutId",
            "detectorSha", "classifierSha", "hudReaderSha", "eyeRevision",
            "supportStatus"})
        for key in ("manifestVersion", "layoutId", "detectorSha", "eyeRevision"):
            if not isinstance(recognizer[key], str):
                raise _schema_error(
                    "INVALID_DRAFT", f"recognizer.{key}", "expected string",
                )
        for key in ("classifierSha", "hudReaderSha"):
            if recognizer[key] is not None and not isinstance(recognizer[key], str):
                raise _schema_error(
                    "INVALID_DRAFT", f"recognizer.{key}",
                    "expected string or null",
                )
        if recognizer["supportStatus"] not in ("experimental", "supported"):
            raise _schema_error(
                "INVALID_DRAFT", "recognizer.supportStatus",
                "invalid support status",
            )
    round_ = _require_dict(root.get("round"), "round")
    _require_keys(round_, "round", {"gameLength", "bakaze", "kyoku", "honba",
        "kyotaku", "leftTileCount", "seatWindSelf", "scores"})
    if not isinstance(round_["scores"], list) or len(round_["scores"]) != 4:
        raise _schema_error(
            "INVALID_DRAFT", "round.scores", "scores must have four entries",
        )
    if round_["gameLength"] not in (None, "hanchan", "tonpu"):
        raise _schema_error(
            "INVALID_DRAFT", "round.gameLength", "invalid game length",
        )
    for key in ("bakaze", "seatWindSelf"):
        if round_[key] not in (None, "E", "S", "W", "N"):
            raise _schema_error("INVALID_DRAFT", f"round.{key}", "invalid wind")
    for key in ("kyoku", "honba", "kyotaku", "leftTileCount"):
        if round_[key] is not None and (
                type(round_[key]) is not int or round_[key] < 0):
            raise _schema_error(
                "INVALID_DRAFT", f"round.{key}",
                "expected non-negative integer or null",
            )
    if any(value is not None and type(value) is not int for value in round_["scores"]):
        raise _schema_error(
            "INVALID_DRAFT", "round.scores",
            "score entries must be integer or null",
        )
    annotations = root.get("annotations")
    if not isinstance(annotations, Mapping):
        raise _schema_error(
            "INVALID_DRAFT", "annotations", "annotations must be an object",
        )
    for field_path, annotation in annotations.items():
        if not isinstance(field_path, str) or not field_path:
            raise _schema_error(
                "INVALID_DRAFT", "annotations",
                "annotation key must be a field path",
            )
        a = _require_dict(annotation, f"annotations.{field_path}")
        _require_keys(a, f"annotations.{field_path}", {"source", "confidence",
            "candidates", "evidenceIds", "confirmedRevision"})
        if (a["source"] not in ("recognized", "user", "default", "inferred", "forced")
                or not isinstance(a["candidates"], list)
                or not isinstance(a["evidenceIds"], list)):
            raise _schema_error(
                "INVALID_DRAFT", f"annotations.{field_path}",
                "invalid annotation shape",
            )
        if a["confidence"] is not None and not _is_finite_number(a["confidence"]):
            raise _schema_error(
                "INVALID_DRAFT", f"annotations.{field_path}.confidence",
                "confidence must be a finite number or null",
            )
        if (a["confirmedRevision"] is not None
                and type(a["confirmedRevision"]) is not int):
            raise _schema_error(
                "INVALID_DRAFT", f"annotations.{field_path}.confirmedRevision",
                "confirmedRevision must be an integer or null",
            )
        for ci, candidate in enumerate(a["candidates"]):
            candidate_path = f"annotations.{field_path}.candidates.{ci}"
            c = _require_dict(candidate, candidate_path)
            _require_keys(c, candidate_path, {"value", "confidence"})
            if not _is_finite_number(c["confidence"]):
                raise _schema_error(
                    "INVALID_DRAFT", f"{candidate_path}.confidence",
                    "candidate confidence must be a finite number",
                )
        for ei, evidence_id in enumerate(a["evidenceIds"]):
            if not isinstance(evidence_id, str):
                raise _schema_error(
                    "INVALID_DRAFT",
                    f"annotations.{field_path}.evidenceIds.{ei}",
                    "evidence id must be a string",
                )
    players = root.get("players")
    if (not isinstance(players, list)
            or [p.get("relSeat") for p in players if isinstance(p, Mapping)]
            != [0, 1, 2, 3]):
        raise _schema_error(
            "INVALID_DRAFT", "players", "players must be ordered relSeat 0..3",
        )
    ids: set[str] = set()

    def take_id(value: Any, path: str) -> None:
        item_id = _require_id(value, path)
        if item_id in ids:
            raise _schema_error(
                "DUPLICATE_ITEM_ID", path, f"duplicate id {item_id}",
            )
        ids.add(item_id)

    for pi, player in enumerate(players):
        p = _require_dict(player, f"players.{pi}")
        _require_keys(p, f"players.{pi}", {"relSeat", "hand", "drawnTile",
            "concealedCount", "reach", "rivers", "melds"})
        if not isinstance(p["reach"], bool):
            raise _schema_error(
                "INVALID_DRAFT", f"players.{pi}.reach", "reach must be boolean",
            )
        if (p["concealedCount"] is not None
                and (type(p["concealedCount"]) is not int
                     or p["concealedCount"] < 0)):
            raise _schema_error(
                "INVALID_DRAFT", f"players.{pi}.concealedCount",
                "concealedCount must be non-negative integer or null",
            )
        if not isinstance(p["rivers"], list) or not isinstance(p["melds"], list):
            raise _schema_error(
                "INVALID_DRAFT", f"players.{pi}",
                "rivers and melds must be lists",
            )
        hand = p.get("hand")
        if hand is not None:
            if not isinstance(hand, list):
                raise _schema_error(
                    "INVALID_DRAFT", f"players.{pi}.hand",
                    "hand must be a list or null",
                )
            for ti, tile in enumerate(hand):
                t = _require_dict(tile, f"players.{pi}.hand.{ti}")
                _require_keys(t, f"players.{pi}.hand.{ti}", {"id", "pai"})
                take_id(t.get("id"), f"players.{pi}.hand.{ti}.id")
                _validate_tile(
                    t.get("pai"), f"players.{pi}.hand.{ti}.pai", nullable=True,
                )
        draw = p.get("drawnTile")
        if draw is not None:
            d = _require_dict(draw, f"players.{pi}.drawnTile")
            _require_keys(d, f"players.{pi}.drawnTile", {"id", "pai"})
            take_id(d.get("id"), f"players.{pi}.drawnTile.id")
            _validate_tile(
                d.get("pai"), f"players.{pi}.drawnTile.pai", nullable=True,
            )
        for ri, discard in enumerate(p.get("rivers") or []):
            d = _require_dict(discard, f"players.{pi}.rivers.{ri}")
            _require_keys(
                d, f"players.{pi}.rivers.{ri}",
                {"id", "pai", "sideways", "tsumogiri"},
            )
            if not isinstance(d["sideways"], bool):
                raise _schema_error(
                    "INVALID_DRAFT", f"players.{pi}.rivers.{ri}.sideways",
                    "sideways must be boolean",
                )
            take_id(d.get("id"), f"players.{pi}.rivers.{ri}.id")
            _validate_tile(
                d.get("pai"), f"players.{pi}.rivers.{ri}.pai", nullable=True,
            )
            _validate_mark(
                d.get("tsumogiri"), f"players.{pi}.rivers.{ri}.tsumogiri",
            )
        for mi, meld in enumerate(p.get("melds") or []):
            m = _require_dict(meld, f"players.{pi}.melds.{mi}")
            _require_keys(m, f"players.{pi}.melds.{mi}", {"id", "type", "tiles",
                "calledPai", "addedPai", "fromOffset"})
            if (m["type"] not in ("chi", "pon", "daiminkan", "ankan", "kakan")
                    or m["fromOffset"] not in (0, 1, 2, 3)
                    or not isinstance(m["tiles"], list)):
                raise _schema_error(
                    "INVALID_DRAFT", f"players.{pi}.melds.{mi}",
                    "invalid meld shape",
                )
            _validate_tile(
                m["calledPai"], f"players.{pi}.melds.{mi}.calledPai",
                nullable=True,
            )
            _validate_tile(
                m["addedPai"], f"players.{pi}.melds.{mi}.addedPai",
                nullable=True,
            )
            take_id(m.get("id"), f"players.{pi}.melds.{mi}.id")
            for ti, tile in enumerate(m.get("tiles") or []):
                _validate_tile(
                    tile, f"players.{pi}.melds.{mi}.tiles.{ti}", nullable=False,
                )
    dora_markers = root.get("doraMarkers")
    if not isinstance(dora_markers, list):
        raise _schema_error(
            "INVALID_DRAFT", "doraMarkers", "doraMarkers must be a list",
        )
    for di, dora in enumerate(dora_markers):
        d = _require_dict(dora, f"doraMarkers.{di}")
        _require_keys(d, f"doraMarkers.{di}", {"id", "pai"})
        take_id(d.get("id"), f"doraMarkers.{di}.id")
        _validate_tile(d.get("pai"), f"doraMarkers.{di}.pai", nullable=True)
    history = _require_dict(root.get("historyOverrides"), "historyOverrides")
    _require_keys(history, "historyOverrides", {"ghostDiscards"})
    ghosts = history.get("ghostDiscards")
    if not isinstance(ghosts, list):
        raise _schema_error(
            "INVALID_DRAFT", "historyOverrides.ghostDiscards",
            "ghostDiscards must be a list",
        )
    for gi, ghost in enumerate(ghosts):
        g = _require_dict(ghost, f"historyOverrides.ghostDiscards.{gi}")
        _require_keys(
            g, f"historyOverrides.ghostDiscards.{gi}",
            {"id", "ownerRelSeat", "pai", "beforeMeldId", "tsumogiri"},
        )
        if (g["ownerRelSeat"] not in (0, 1, 2, 3)
                or not isinstance(g["beforeMeldId"], str)
                or not g["beforeMeldId"]):
            raise _schema_error(
                "INVALID_DRAFT", f"historyOverrides.ghostDiscards.{gi}",
                "invalid ghost owner or meld reference",
            )
        take_id(g.get("id"), f"historyOverrides.ghostDiscards.{gi}.id")
        _validate_tile(
            g.get("pai"), f"historyOverrides.ghostDiscards.{gi}.pai",
            nullable=False,
        )
        _validate_mark(
            g.get("tsumogiri"),
            f"historyOverrides.ghostDiscards.{gi}.tsumogiri",
        )
    evidence = root.get("evidence")
    if not isinstance(evidence, list):
        raise _schema_error(
            "INVALID_DRAFT", "evidence", "evidence must be a list",
        )
    for ei, item in enumerate(evidence):
        e = _require_dict(item, f"evidence.{ei}")
        _require_keys(e, f"evidence.{ei}", {"id", "bbox", "polygon", "zone"})
        take_id(e.get("id"), f"evidence.{ei}.id")
        if not isinstance(e["zone"], str) or not e["zone"]:
            raise _schema_error(
                "INVALID_EVIDENCE", f"evidence.{ei}.zone",
                "zone must be a non-empty string",
            )
        bbox = e.get("bbox")
        polygon = e.get("polygon")
        if (not isinstance(bbox, list) or len(bbox) != 4
                or not all(isinstance(v, (int, float)) and math.isfinite(v)
                           for v in bbox)):
            raise _schema_error(
                "INVALID_EVIDENCE", f"evidence.{ei}.bbox",
                "bbox must contain four finite numbers",
            )
        if polygon is not None and (
                not isinstance(polygon, list) or len(polygon) != 4
                or any(not isinstance(point, list) or len(point) != 2
                       or not all(isinstance(v, (int, float)) and math.isfinite(v)
                                  for v in point) for point in polygon)):
            raise _schema_error(
                "INVALID_EVIDENCE", f"evidence.{ei}.polygon",
                "polygon must contain four finite [x,y] points",
            )
    return cast(WhatCutDraftV1, deepcopy(dict(root)))


def copy_what_cut_draft(draft: WhatCutDraftV1) -> WhatCutDraftV1:
    return parse_what_cut_draft(deepcopy(draft))


def restore_tsumogiri(mark: WhatCutTsumogiriV1) -> WhatCutTsumogiriV1:
    return {"value": mark["baselineValue"], "source": mark["baselineSource"],
            "baselineValue": mark["baselineValue"],
            "baselineSource": mark["baselineSource"]}
