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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_what_cut_schema OK")
