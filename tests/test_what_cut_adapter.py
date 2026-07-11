from copy import deepcopy

from majsoul_eye.what_cut.adapter import draft_to_observed
from test_what_cut_schema import minimal_draft


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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_what_cut_adapter OK")
