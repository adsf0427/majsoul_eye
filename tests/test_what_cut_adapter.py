from copy import deepcopy

from majsoul_eye.what_cut.adapter import draft_to_observed
from test_what_cut_schema import minimal_draft


def _draft_with_meld(meld, *, ghost_source="inferred"):
    draft = minimal_draft()
    draft["players"][1]["melds"] = [meld]
    ghosts = []
    if meld["type"] != "ankan" and meld["calledPai"] is not None:
        ghosts.append({
            "id": "ghost-1-0",
            "ownerRelSeat": (1 + meld["fromOffset"]) % 4,
            "pai": meld["calledPai"],
            "beforeMeldId": meld["id"],
            "tsumogiri": {
                "value": False,
                "source": ghost_source,
                "baselineValue": False,
                "baselineSource": "inferred",
            },
        })
    draft["historyOverrides"]["ghostDiscards"] = ghosts
    return draft


def test_complete_draft_maps_screen_relative_state():
    draft = minimal_draft()
    draft["historyOverrides"]["ghostDiscards"] = []
    result = draft_to_observed(draft)
    assert result.issues == []
    assert result.observed is not None
    assert result.observed.hero_hand[0] == "1m"
    assert result.observed.drawn_tile == "5p"
    assert result.observed.rivers[0][0].pai == "9p"
    assert result.observed.scores == [25000, 25000, 25000, 25000]
    assert result.overrides.user_visible[(0, 0)].value is False
    assert result.overrides.river_ids[(0, 0)] == "river-0-0"


def test_non_user_current_value_is_not_a_solver_constraint():
    draft = minimal_draft()
    draft["historyOverrides"]["ghostDiscards"] = []
    mark = draft["players"][0]["rivers"][0]["tsumogiri"]
    mark.update({"value": True, "source": "inferred",
                 "baselineValue": False, "baselineSource": "inferred"})
    result = draft_to_observed(draft)
    assert result.overrides.user_visible == {}
    assert result.overrides.river_ids[(0, 0)] == "river-0-0"


def test_null_tile_returns_field_addressed_blocker_without_observed_state():
    draft = minimal_draft()
    draft["historyOverrides"]["ghostDiscards"] = []
    draft["players"][0]["hand"][3]["pai"] = None
    result = draft_to_observed(draft)
    assert result.observed is None
    issue = next(i for i in result.issues if i["code"] == "MISSING_TILE")
    assert issue["fieldPath"] == "players.0.hand.hand-3.pai"


def test_ghost_owner_and_meld_reference_are_validated():
    draft = minimal_draft()
    bad = deepcopy(draft)
    result = draft_to_observed(bad)
    assert result.observed is None
    assert any(i["code"] == "GHOST_MELD_NOT_FOUND" for i in result.issues)


def test_each_called_meld_requires_exactly_one_ghost():
    draft = minimal_draft()
    draft["historyOverrides"]["ghostDiscards"] = []
    draft["players"][1]["melds"] = [{
        "id": "meld-1-0", "type": "pon", "tiles": ["P", "P", "P"],
        "calledPai": "P", "addedPai": None, "fromOffset": 1,
    }]
    missing = draft_to_observed(draft)
    assert any(i["code"] == "GHOST_REQUIRED" for i in missing.issues)

    ghost = {"id": "ghost-1", "ownerRelSeat": 2, "pai": "P",
             "beforeMeldId": "meld-1-0",
             "tsumogiri": {"value": False, "source": "inferred",
                            "baselineValue": False, "baselineSource": "inferred"}}
    draft["historyOverrides"]["ghostDiscards"] = [ghost, {**ghost, "id": "ghost-2"}]
    duplicate = draft_to_observed(draft)
    assert any(i["code"] == "GHOST_DUPLICATE" for i in duplicate.issues)


def test_called_meld_offsets_reject_self_calls():
    cases = [
        {"id": "meld-1-pon", "type": "pon", "tiles": ["P"] * 3,
         "calledPai": "P", "addedPai": None, "fromOffset": 0},
        {"id": "meld-1-daiminkan", "type": "daiminkan", "tiles": ["C"] * 4,
         "calledPai": "C", "addedPai": None, "fromOffset": 0},
        {"id": "meld-1-kakan", "type": "kakan", "tiles": ["F"] * 4,
         "calledPai": "F", "addedPai": "F", "fromOffset": 0},
    ]
    for meld in cases:
        result = draft_to_observed(_draft_with_meld(meld))
        issue = next(
            (i for i in result.issues if i["code"] == "INVALID_MELD_SOURCE"),
            None,
        )
        assert issue is not None, meld["type"]
        assert issue["fieldPath"] == f"players.1.melds.{meld['id']}.fromOffset"
        assert issue["severity"] == "blocking"
        assert result.observed is None


def test_every_called_meld_requires_called_tile_membership():
    cases = [
        {"id": "meld-1-chi", "type": "chi", "tiles": ["1m", "2m", "3m"],
         "calledPai": "9p", "addedPai": None, "fromOffset": 3},
        {"id": "meld-1-pon", "type": "pon", "tiles": ["P"] * 3,
         "calledPai": "9p", "addedPai": None, "fromOffset": 1},
        {"id": "meld-1-daiminkan", "type": "daiminkan", "tiles": ["C"] * 4,
         "calledPai": "9p", "addedPai": None, "fromOffset": 2},
        {"id": "meld-1-kakan", "type": "kakan", "tiles": ["F"] * 4,
         "calledPai": "9p", "addedPai": "F", "fromOffset": 1},
    ]
    for meld in cases:
        result = draft_to_observed(_draft_with_meld(meld))
        issue = next(
            (i for i in result.issues
             if i["code"] == "CALLED_TILE_NOT_IN_MELD"),
            None,
        )
        assert issue is not None, meld["type"]
        assert issue["fieldPath"] == f"players.1.melds.{meld['id']}.calledPai"
        assert issue["severity"] == "blocking"
        assert result.observed is None


def test_meld_types_reject_inappropriate_called_and_added_fields():
    cases = [
        ({"id": "meld-1-chi", "type": "chi", "tiles": ["1m", "2m", "3m"],
          "calledPai": "1m", "addedPai": "1m", "fromOffset": 3},
         "UNEXPECTED_ADDED_TILE", "addedPai"),
        ({"id": "meld-1-pon", "type": "pon", "tiles": ["P"] * 3,
          "calledPai": "P", "addedPai": "P", "fromOffset": 1},
         "UNEXPECTED_ADDED_TILE", "addedPai"),
        ({"id": "meld-1-daiminkan", "type": "daiminkan", "tiles": ["C"] * 4,
          "calledPai": "C", "addedPai": "C", "fromOffset": 2},
         "UNEXPECTED_ADDED_TILE", "addedPai"),
        ({"id": "meld-1-ankan-called", "type": "ankan", "tiles": ["F"] * 4,
          "calledPai": "F", "addedPai": None, "fromOffset": 0},
         "UNEXPECTED_CALLED_TILE", "calledPai"),
        ({"id": "meld-1-ankan-added", "type": "ankan", "tiles": ["F"] * 4,
          "calledPai": None, "addedPai": "F", "fromOffset": 0},
         "UNEXPECTED_ADDED_TILE", "addedPai"),
        ({"id": "meld-1-kakan", "type": "kakan", "tiles": ["P"] * 4,
          "calledPai": "P", "addedPai": None, "fromOffset": 1},
         "ADDED_TILE_NOT_IN_MELD", "addedPai"),
    ]
    for meld, code, field in cases:
        result = draft_to_observed(_draft_with_meld(meld))
        issue = next((i for i in result.issues if i["code"] == code), None)
        assert issue is not None, f"{meld['type']} {field}"
        assert issue["fieldPath"] == f"players.1.melds.{meld['id']}.{field}"
        assert issue["severity"] == "blocking"
        assert result.observed is None


def test_type_appropriate_meld_fields_are_accepted():
    cases = [
        {"id": "meld-1-chi", "type": "chi", "tiles": ["1m", "2m", "3m"],
         "calledPai": "1m", "addedPai": None, "fromOffset": 3},
        {"id": "meld-1-pon", "type": "pon", "tiles": ["P"] * 3,
         "calledPai": "P", "addedPai": None, "fromOffset": 1},
        {"id": "meld-1-daiminkan", "type": "daiminkan", "tiles": ["C"] * 4,
         "calledPai": "C", "addedPai": None, "fromOffset": 2},
        {"id": "meld-1-ankan", "type": "ankan", "tiles": ["F"] * 4,
         "calledPai": None, "addedPai": None, "fromOffset": 0},
        {"id": "meld-1-kakan", "type": "kakan", "tiles": ["P"] * 4,
         "calledPai": "P", "addedPai": "P", "fromOffset": 1},
    ]
    for meld in cases:
        result = draft_to_observed(_draft_with_meld(meld))
        assert result.issues == [], meld["type"]
        assert result.observed is not None


def test_ghost_owner_mismatch_is_field_addressed_blocker():
    meld = {"id": "meld-1-0", "type": "pon", "tiles": ["P"] * 3,
            "calledPai": "P", "addedPai": None, "fromOffset": 1}
    draft = _draft_with_meld(meld)
    draft["historyOverrides"]["ghostDiscards"][0]["ownerRelSeat"] = 3
    result = draft_to_observed(draft)
    issue = next(i for i in result.issues if i["code"] == "GHOST_MELD_MISMATCH")
    assert issue["fieldPath"] == "historyOverrides.ghostDiscards.ghost-1-0"
    assert issue["severity"] == "blocking"
    assert result.observed is None


def test_user_ghost_override_uses_meld_position_and_stable_id():
    meld = {"id": "meld-1-0", "type": "pon", "tiles": ["P"] * 3,
            "calledPai": "P", "addedPai": None, "fromOffset": 1}
    result = draft_to_observed(_draft_with_meld(meld, ghost_source="user"))
    assert result.issues == []
    assert result.observed is not None
    override = result.overrides.user_ghosts[(1, 0)]
    assert override.value is False
    assert override.item_id == "ghost-1-0"
    assert override.field_path == (
        "historyOverrides.ghostDiscards.ghost-1-0.tsumogiri"
    )
    assert result.overrides.ghost_ids == {(1, 0): "ghost-1-0"}
    assert result.overrides.ghost_order == [(1, 0)]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_what_cut_adapter OK")


def test_v1_draft_projects_four_player_defaults():
    draft = minimal_draft()
    draft["historyOverrides"]["ghostDiscards"] = []
    result = draft_to_observed(draft)
    assert result.observed is not None
    assert result.observed.sanma is False
    assert result.observed.phantom_rel is None
    assert result.observed.nukidora == [0, 0, 0, 0]


def test_v2_sanma_draft_projects_mode_and_nuki_counts():
    from test_what_cut_schema import minimal_draft_v2

    draft = minimal_draft_v2(3)
    draft["players"][0]["nukiCount"] = 1
    draft["players"][2]["nukiCount"] = 2
    result = draft_to_observed(draft)
    assert result.observed is not None
    assert result.observed.sanma is True
    assert result.observed.phantom_rel == 3
    assert result.observed.nukidora == [1, 0, 2, 0]
