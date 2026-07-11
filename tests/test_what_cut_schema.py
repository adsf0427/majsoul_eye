from majsoul_eye.what_cut.schema import (
    DraftSchemaError, copy_what_cut_draft, parse_what_cut_draft,
    restore_tsumogiri,
)


def minimal_draft():
    return {
        "schemaVersion": 1,
        "draftId": "draft-1",
        "revision": 7,
        "nPlayers": 4,
        "seatFrame": "screen-relative",
        "source": {"kind": "manual", "imageRef": None, "imageHash": None,
                   "width": None, "height": None},
        "recognizer": None,
        "round": {"gameLength": "hanchan", "bakaze": "E", "kyoku": 1,
                  "honba": 0, "kyotaku": 0, "leftTileCount": 70,
                  "seatWindSelf": "E", "scores": [25000, 25000, 25000, 25000]},
        "doraMarkers": [{"id": "dora-0", "pai": "5s"}],
        "players": [
            {"relSeat": 0,
             "hand": [{"id": f"hand-{i}", "pai": p} for i, p in enumerate(
                 ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m",
                  "9m", "1p", "2p", "3p", "4p"])],
             "drawnTile": {"id": "draw-0", "pai": "5p"},
             "concealedCount": None, "reach": False,
             "rivers": [{"id": "river-0-0", "pai": "9p", "sideways": False,
                         "tsumogiri": {"value": False, "source": "user",
                                        "baselineValue": True,
                                        "baselineSource": "inferred"}}],
             "melds": []},
            {"relSeat": 1, "hand": None, "drawnTile": None,
             "concealedCount": 13, "reach": False, "rivers": [], "melds": []},
            {"relSeat": 2, "hand": None, "drawnTile": None,
             "concealedCount": 13, "reach": False, "rivers": [], "melds": []},
            {"relSeat": 3, "hand": None, "drawnTile": None,
             "concealedCount": 13, "reach": False, "rivers": [], "melds": []},
        ],
        "annotations": {},
        "evidence": [],
        "historyOverrides": {"ghostDiscards": [{
            "id": "ghost-0", "ownerRelSeat": 3, "pai": "7s",
            "beforeMeldId": "meld-0-0",
            "tsumogiri": {"value": True, "source": "user",
                           "baselineValue": False,
                           "baselineSource": "inferred"},
        }]},
    }


def valid_recognizer():
    return {
        "manifestVersion": "manifest-v1",
        "layoutId": "desktop-16x9",
        "detectorSha": "detector-sha",
        "classifierSha": None,
        "hudReaderSha": "hud-reader-sha",
        "eyeRevision": "eye-revision",
        "supportStatus": "experimental",
    }


def valid_annotation():
    return {
        "source": "recognized",
        "confidence": 0.9,
        "candidates": [{"value": "5m", "confidence": 0.8}],
        "evidenceIds": ["e-1"],
        "confirmedRevision": 7,
    }


def _assert_invalid(raw, path, code="INVALID_DRAFT"):
    try:
        parse_what_cut_draft(raw)
    except DraftSchemaError as exc:
        assert exc.code == code
        assert exc.path == path
    else:
        raise AssertionError(f"{path} must be rejected")


def test_schema_round_trip_preserves_editing_metadata():
    raw = minimal_draft()
    parsed = parse_what_cut_draft(raw)
    copied = copy_what_cut_draft(parsed)
    assert copied == raw
    assert copied is not raw
    assert copied["players"][0]["rivers"][0]["tsumogiri"]["baselineValue"] is True
    assert copied["historyOverrides"]["ghostDiscards"][0]["tsumogiri"]["source"] == "user"


def test_restore_uses_latest_baseline():
    mark = {"value": False, "source": "user", "baselineValue": True,
            "baselineSource": "forced"}
    assert restore_tsumogiri(mark) == {
        "value": True, "source": "forced", "baselineValue": True,
        "baselineSource": "forced",
    }


def test_wrong_version_is_rejected():
    raw = minimal_draft()
    raw["schemaVersion"] = 2
    try:
        parse_what_cut_draft(raw)
    except DraftSchemaError as exc:
        assert exc.code == "UNSUPPORTED_SCHEMA"
        assert exc.path == "schemaVersion"
    else:
        raise AssertionError("schema version 2 must be rejected")


def test_evidence_requires_finite_fixed_length_geometry():
    raw = minimal_draft()
    raw["evidence"] = [{"id": "e-1", "bbox": [0.0, 1.0, 2.0],
                        "polygon": None, "zone": "hand"}]
    try:
        parse_what_cut_draft(raw)
    except DraftSchemaError as exc:
        assert exc.code == "INVALID_EVIDENCE"
        assert exc.path == "evidence.0.bbox"
    else:
        raise AssertionError("three-value bbox must be rejected")

    raw["evidence"][0] = {"id": "e-1", "bbox": [0.0, 1.0, 2.0, 3.0],
                           "polygon": [[0.0, 0.0], [1.0, 0.0],
                                       [1.0, float("inf")], [0.0, 1.0]],
                           "zone": "hand"}
    try:
        parse_what_cut_draft(raw)
    except DraftSchemaError as exc:
        assert exc.code == "INVALID_EVIDENCE"
        assert exc.path == "evidence.0.polygon"
    else:
        raise AssertionError("non-finite polygon coordinate must be rejected")


def test_history_override_and_mark_keys_are_exact():
    raw = minimal_draft()
    raw["historyOverrides"]["legacy"] = []
    try:
        parse_what_cut_draft(raw)
    except DraftSchemaError as exc:
        assert exc.code == "INVALID_DRAFT"
        assert exc.path == "historyOverrides"
    else:
        raise AssertionError("unknown historyOverrides key must be rejected")

    raw = minimal_draft()
    raw["players"][0]["rivers"][0]["tsumogiri"]["confidence"] = 0.8
    try:
        parse_what_cut_draft(raw)
    except DraftSchemaError as exc:
        assert exc.path == "players.0.rivers.0.tsumogiri"
    else:
        raise AssertionError("history mark wire keys must be exact")


def test_recognizer_metadata_value_types_are_exact():
    invalid_values = {
        "manifestVersion": None,
        "layoutId": 1,
        "detectorSha": False,
        "classifierSha": [],
        "hudReaderSha": {},
        "eyeRevision": 2.0,
    }
    for key, value in invalid_values.items():
        raw = minimal_draft()
        raw["recognizer"] = valid_recognizer()
        raw["recognizer"][key] = value
        _assert_invalid(raw, f"recognizer.{key}")


def test_annotation_candidates_require_exact_objects():
    for candidate in (None, {"value": "5m"},
                      {"value": "5m", "confidence": 0.8, "legacy": True}):
        raw = minimal_draft()
        raw["annotations"]["round.kyoku"] = valid_annotation()
        raw["annotations"]["round.kyoku"]["candidates"] = [candidate]
        _assert_invalid(raw, "annotations.round.kyoku.candidates.0")


def test_annotation_confidences_are_finite_numbers():
    for confidence in ("0.9", True, float("nan"),
                       float("inf"), float("-inf")):
        raw = minimal_draft()
        raw["annotations"]["round.kyoku"] = valid_annotation()
        raw["annotations"]["round.kyoku"]["confidence"] = confidence
        _assert_invalid(raw, "annotations.round.kyoku.confidence")

        raw = minimal_draft()
        raw["annotations"]["round.kyoku"] = valid_annotation()
        raw["annotations"]["round.kyoku"]["candidates"][0]["confidence"] = confidence
        _assert_invalid(raw, "annotations.round.kyoku.candidates.0.confidence")


def test_annotation_evidence_ids_are_strings():
    raw = minimal_draft()
    raw["annotations"]["round.kyoku"] = valid_annotation()
    raw["annotations"]["round.kyoku"]["evidenceIds"] = ["e-1", 7]
    _assert_invalid(raw, "annotations.round.kyoku.evidenceIds.1")


def test_annotation_confirmed_revision_is_integer_or_null():
    for revision in ("7", 7.0, True):
        raw = minimal_draft()
        raw["annotations"]["round.kyoku"] = valid_annotation()
        raw["annotations"]["round.kyoku"]["confirmedRevision"] = revision
        _assert_invalid(raw, "annotations.round.kyoku.confirmedRevision")


def test_valid_nested_metadata_round_trips():
    raw = minimal_draft()
    raw["recognizer"] = valid_recognizer()
    raw["annotations"]["round.kyoku"] = valid_annotation()
    raw["evidence"] = [{"id": "e-1", "bbox": [0.0, 0.0, 1.0, 1.0],
                        "polygon": None, "zone": "round"}]
    assert parse_what_cut_draft(raw) == raw


def test_huge_integer_confidences_round_trip():
    huge = 10 ** 1000
    raw = minimal_draft()
    raw["annotations"]["round.kyoku"] = valid_annotation()
    raw["annotations"]["round.kyoku"]["confidence"] = huge
    raw["annotations"]["round.kyoku"]["candidates"][0]["confidence"] = huge

    parsed = parse_what_cut_draft(raw)

    assert parsed["annotations"]["round.kyoku"]["confidence"] == huge
    assert parsed["annotations"]["round.kyoku"]["candidates"][0]["confidence"] == huge


def test_huge_integer_evidence_geometry_round_trip():
    huge = 10 ** 1000
    raw = minimal_draft()
    raw["evidence"] = [{
        "id": "e-huge", "bbox": [huge, 0.0, 1.0, 1.0],
        "polygon": [[huge, 0.0], [1.0, 0.0],
                    [1.0, 1.0], [0.0, 1.0]],
        "zone": "hand",
    }]

    parsed = parse_what_cut_draft(raw)

    assert parsed["evidence"][0]["bbox"][0] == huge
    assert parsed["evidence"][0]["polygon"][0][0] == huge


def test_dora_markers_container_must_be_list():
    for value in (None, {}, "", False):
        raw = minimal_draft()
        raw["doraMarkers"] = value
        _assert_invalid(raw, "doraMarkers")


def test_integer_root_source_and_round_fields_reject_numeric_impostors():
    cases = [
        ("root", "schemaVersion", True, "schemaVersion", "UNSUPPORTED_SCHEMA"),
        ("root", "schemaVersion", 1.0, "schemaVersion", "UNSUPPORTED_SCHEMA"),
        ("root", "nPlayers", 4.0, "nPlayers", "INVALID_DRAFT"),
        ("root", "revision", True, "revision", "INVALID_DRAFT"),
        ("root", "revision", 7.0, "revision", "INVALID_DRAFT"),
        ("source", "width", 1920.0, "source.width", "INVALID_DRAFT"),
        ("source", "height", True, "source.height", "INVALID_DRAFT"),
        ("round", "kyoku", 1.0, "round.kyoku", "INVALID_DRAFT"),
        ("round", "honba", False, "round.honba", "INVALID_DRAFT"),
        ("round", "kyotaku", 0.0, "round.kyotaku", "INVALID_DRAFT"),
        ("round", "leftTileCount", True, "round.leftTileCount", "INVALID_DRAFT"),
    ]
    for container, key, value, path, code in cases:
        raw = minimal_draft()
        target = raw if container == "root" else raw[container]
        target[key] = value
        _assert_invalid(raw, path, code)

    for value in (25000.0, True):
        raw = minimal_draft()
        raw["round"]["scores"][0] = value
        _assert_invalid(raw, "round.scores")


def test_player_seat_and_count_fields_require_exact_integers():
    for player_index, value in ((0, False), (1, 1.0)):
        raw = minimal_draft()
        raw["players"][player_index]["relSeat"] = value
        _assert_invalid(raw, "players")

    for value in (13.0, False):
        raw = minimal_draft()
        raw["players"][1]["concealedCount"] = value
        _assert_invalid(raw, "players.1.concealedCount")


def test_meld_and_ghost_seat_literals_require_exact_integers():
    for value in (False, 1.0):
        raw = minimal_draft()
        raw["players"][0]["melds"] = [{
            "id": "meld-0-0", "type": "pon",
            "tiles": ["1m", "1m", "1m"],
            "calledPai": "1m", "addedPai": None, "fromOffset": value,
        }]
        _assert_invalid(raw, "players.0.melds.0")

    for value in (True, 3.0):
        raw = minimal_draft()
        raw["historyOverrides"]["ghostDiscards"][0]["ownerRelSeat"] = value
        _assert_invalid(raw, "historyOverrides.ghostDiscards.0")


def test_evidence_coordinates_reject_booleans():
    raw = minimal_draft()
    raw["evidence"] = [{"id": "e-bbox", "bbox": [False, 0.0, 1.0, 1.0],
                        "polygon": None, "zone": "hand"}]
    _assert_invalid(raw, "evidence.0.bbox", "INVALID_EVIDENCE")

    raw = minimal_draft()
    raw["evidence"] = [{
        "id": "e-polygon", "bbox": [0.0, 0.0, 1.0, 1.0],
        "polygon": [[0.0, 0.0], [True, 0.0],
                    [1.0, 1.0], [0.0, 1.0]],
        "zone": "hand",
    }]
    _assert_invalid(raw, "evidence.0.polygon", "INVALID_EVIDENCE")


def test_evidence_coordinates_reject_non_finite_floats():
    for value in (float("nan"), float("inf"), float("-inf")):
        raw = minimal_draft()
        raw["evidence"] = [{"id": "e-bbox", "bbox": [value, 0.0, 1.0, 1.0],
                            "polygon": None, "zone": "hand"}]
        _assert_invalid(raw, "evidence.0.bbox", "INVALID_EVIDENCE")

        raw = minimal_draft()
        raw["evidence"] = [{
            "id": "e-polygon", "bbox": [0.0, 0.0, 1.0, 1.0],
            "polygon": [[value, 0.0], [1.0, 0.0],
                        [1.0, 1.0], [0.0, 1.0]],
            "zone": "hand",
        }]
        _assert_invalid(raw, "evidence.0.polygon", "INVALID_EVIDENCE")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_what_cut_schema OK")
