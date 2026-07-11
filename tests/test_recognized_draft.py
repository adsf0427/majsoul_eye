from copy import deepcopy
from types import SimpleNamespace

from majsoul_eye.recognize.evidence import AssemblyResult, FieldObservation
from majsoul_eye.state.observe import ObservedMeld, ObservedRiverTile, ObservedState
from majsoul_eye.what_cut.from_recognition import (
    DraftBuildContext, apply_history_baseline, build_recognized_draft,
)


def recognizer():
    return {"manifestVersion": "internal-v1", "layoutId": "majsoul-desktop-16x9-v1",
            "detectorSha": "a" * 64, "classifierSha": "b" * 64,
            "hudReaderSha": "c" * 64, "eyeRevision": "rev",
            "supportStatus": "experimental"}


def assembly():
    o = ObservedState(hero_hand=["1m"] * 3 + ["2m"] * 3 + ["3m"] * 3 + ["4m"] * 3 + ["9m"],
                      drawn_tile="5p", dora_markers=["5s"])
    o.bakaze = "E"; o.kyoku = 2; o.honba = 1; o.kyotaku = 1
    o.left_tile_count = 42; o.seat_wind_self = "S"
    o.scores = [25000, 26000, 24000, 25000]
    o.rivers[0] = [ObservedRiverTile("9p")]
    o.rivers[2] = [ObservedRiverTile("P")]
    o.melds[1] = [ObservedMeld("pon", ["P", "P", "P"], "P", "", 1)]
    round_det = SimpleNamespace(xyxy=(10, 20, 30, 40), poly=None)
    fields = [
        FieldObservation("hand:0", "1m", 0.9, []),
        FieldObservation("drawn:0", "5p", 0.9, []),
        FieldObservation("dora:0", "5s", 0.9, []),
        FieldObservation("river:0:0", "9p", 0.8, []),
        FieldObservation("river:2:0", "P", 0.8, []),
        FieldObservation("meld:1:0", o.melds[1][0], 0.7, []),
        FieldObservation("round.bakazeKyoku", ("E", 2), 0.95, [round_det]),
        FieldObservation("round.leftTileCount", 42, 0.95, []),
        FieldObservation("round.kyotaku", 1, 0.95, []),
        FieldObservation("round.honba", 1, 0.95, []),
        FieldObservation("round.seatWindSelf", "S", 0.95, []),
    ]
    fields.extend(FieldObservation(f"round.scores.{seat}", o.scores[seat], 0.95, [])
                  for seat in range(4))
    return AssemblyResult(o, fields, [])


def editable_field_paths(draft):
    # Literal parity with Plan 2 frontend-react/src/utils/whatCutDraft.ts fieldPath.
    paths = {f"round.{key}" for key in
             ("gameLength", "bakaze", "kyoku", "honba", "kyotaku",
              "leftTileCount", "seatWindSelf")}
    paths.update(f"round.scores.{seat}" for seat in range(4))
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


def test_builder_assigns_deterministic_ids_and_ghost():
    draft = build_recognized_draft(
        assembly(), DraftBuildContext("d-1", "img-ref", "f" * 64, 1920, 1080),
        recognizer())
    assert draft["players"][0]["rivers"][0]["id"] == "river:0:0"
    assert draft["players"][1]["melds"][0]["id"] == "meld:1:0"
    ghost = draft["historyOverrides"]["ghostDiscards"][0]
    assert ghost["id"] == "ghost:1:0"
    assert ghost["ownerRelSeat"] == 2 and ghost["beforeMeldId"] == "meld:1:0"


def test_annotations_use_plan2_stable_field_paths_and_resolve_to_draft_items():
    draft = build_recognized_draft(
        assembly(), DraftBuildContext("d-1", "img-ref", "f" * 64, 1920, 1080),
        recognizer())
    expected = {
        "players.0.hand.hand:0:0.pai", "players.0.drawnTile.drawn:0.pai",
        "doraMarkers.dora:0.pai", "players.0.rivers.river:0:0.pai",
        "players.2.rivers.river:2:0.pai", "players.1.melds.meld:1:0",
        "round.bakaze", "round.kyoku", "round.leftTileCount",
        "round.kyotaku", "round.honba", "round.seatWindSelf",
        "round.scores.0", "round.scores.1", "round.scores.2", "round.scores.3",
    }
    assert set(draft["annotations"]) == expected
    assert set(draft["annotations"]) <= editable_field_paths(draft)
    assert draft["annotations"]["round.bakaze"]["evidenceIds"] == ["e:round.bakaze:0"]
    assert (draft["annotations"]["round.kyoku"]["evidenceIds"] ==
            draft["annotations"]["round.bakaze"]["evidenceIds"])


def test_unmapped_recognition_field_key_fails_closed():
    bad = assembly()
    bad.fields.append(FieldObservation("new_detector_zone:0", "1m", 0.5, []))
    try:
        build_recognized_draft(
            bad, DraftBuildContext("d-1", "img-ref", "f" * 64, 1920, 1080),
            recognizer())
    except ValueError as exc:
        assert str(exc) == "unmapped recognition field_key: new_detector_zone:0"
    else:
        raise AssertionError("unmapped recognition keys must fail closed")


def test_baseline_sync_preserves_user_current_and_updates_every_baseline():
    draft = build_recognized_draft(
        assembly(), DraftBuildContext("d-1", "img-ref", "f" * 64, 1920, 1080),
        recognizer())
    river = draft["players"][0]["rivers"][0]
    river["tsumogiri"].update({"value": False, "source": "user"})
    baseline = [
        {"itemKind": "river", "itemId": "river:0:0", "baselineValue": True,
         "baselineSource": "forced"},
        {"itemKind": "river", "itemId": "river:2:0", "baselineValue": False,
         "baselineSource": "inferred"},
        {"itemKind": "ghost", "itemId": "ghost:1:0", "baselineValue": True,
         "baselineSource": "inferred"},
    ]
    apply_history_baseline(draft, baseline)
    assert river["tsumogiri"]["value"] is False
    assert river["tsumogiri"]["source"] == "user"
    assert river["tsumogiri"]["baselineValue"] is True
    ghost = draft["historyOverrides"]["ghostDiscards"][0]
    assert ghost["tsumogiri"] == {"value": True, "source": "inferred",
                                   "baselineValue": True,
                                   "baselineSource": "inferred"}


def test_missing_or_duplicate_baseline_item_is_rejected():
    draft = build_recognized_draft(
        assembly(), DraftBuildContext("d-1", "img-ref", "f" * 64, 1920, 1080),
        recognizer())
    try:
        apply_history_baseline(deepcopy(draft), [])
    except ValueError as exc:
        assert "historyBaseline cardinality" in str(exc)
    else:
        raise AssertionError("missing baseline items must fail")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_recognized_draft OK")
