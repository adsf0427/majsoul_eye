from majsoul_eye.state.history import ReconstructionOverrides, UserTsumogiriOverride
from majsoul_eye.state.observe import ObservedMeld, ObservedRiverTile, ObservedState
from majsoul_eye.state.reconstruct import reconstruct

H13 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
       "1p", "2p", "3p", "4p"]


def rotation_obs():
    return ObservedState(
        hero_hand=list(H13), drawn_tile="5p", dora_markers=["5s"],
        rivers=[[ObservedRiverTile("9p")], [ObservedRiverTile("E")],
                [ObservedRiverTile("S")], [ObservedRiverTile("W")]])


def test_hero_tedashi_rebuilds_draws_and_final_hand():
    o = rotation_obs()
    overrides = ReconstructionOverrides(
        user_visible={(0, 0): UserTsumogiriOverride(False, "r-0-0",
                       "players.0.rivers.r-0-0.tsumogiri")},
        river_ids={(s, i): f"r-{s}-{i}" for s, river in enumerate(o.rivers)
                   for i, _ in enumerate(river)})
    result = reconstruct(o, overrides)
    assert result.ok, result.reason
    dahai = next(e for e in result.events if e["type"] == "dahai" and e["actor"] == 0)
    assert dahai["pai"] == "9p" and dahai["tsumogiri"] is False
    assert result.selected_history["solverVersion"] == "hidden-history-v1"
    assert len(result.selected_history["heroHaipai"]) == 13


def test_post_reach_tedashi_override_is_field_addressed_conflict():
    # Counts [1, 2, 2, 2] with oya seat 1: the rotation's last discard is
    # seat 3's, so hero can legally hold the pending draw. (A [1, 2, 1, 1]
    # board forces oya=1 and ends on seat 1 — hero could never be drawing.)
    o = rotation_obs()
    o.rivers[1] = [ObservedRiverTile("E", sideways=True), ObservedRiverTile("N")]
    o.rivers[2] = [ObservedRiverTile("S"), ObservedRiverTile("1s")]
    o.rivers[3] = [ObservedRiverTile("W"), ObservedRiverTile("2s")]
    o.reach[1] = True
    overrides = ReconstructionOverrides(
        user_visible={(1, 1): UserTsumogiriOverride(False, "r-1-1",
                       "players.1.rivers.r-1-1.tsumogiri")},
        river_ids={(s, i): f"r-{s}-{i}" for s, river in enumerate(o.rivers)
                   for i, _ in enumerate(river)})
    result = reconstruct(o, overrides)
    assert not result.ok
    issue = next(i for i in result.issues if i["code"] == "TSUMOGIRI_RULE_CONFLICT")
    assert issue["fieldPath"] == "players.1.rivers.r-1-1.tsumogiri"


def test_called_ghost_override_sets_matching_draw_when_legal():
    o = ObservedState(hero_hand=list(H13), drawn_tile="5p", dora_markers=["5s"])
    o.rivers[0] = [ObservedRiverTile("9p")]
    o.rivers[2] = [ObservedRiverTile("S")]
    # seat 3 must discard once so the rotation can return to hero's draw
    o.rivers[3] = [ObservedRiverTile("W")]
    o.melds[2] = [ObservedMeld("pon", ["P", "P", "P"], "P", "", 3)]
    overrides = ReconstructionOverrides(
        user_ghosts={(2, 0): UserTsumogiriOverride(True, "g-0",
                     "historyOverrides.ghostDiscards.g-0.tsumogiri")},
        river_ids={(0, 0): "r-0-0", (2, 0): "r-2-0"},
        ghost_ids={(2, 0): "g-0"}, ghost_order=[(2, 0)])
    result = reconstruct(o, overrides)
    assert result.ok, result.reason
    events = result.events
    pon_index = next(i for i, e in enumerate(events) if e["type"] == "pon")
    assert events[pon_index - 1]["type"] == "dahai"
    assert events[pon_index - 1]["pai"] == "P"
    assert events[pon_index - 1]["tsumogiri"] is True
    assert events[pon_index - 2] == {"type": "tsumo", "actor": events[pon_index - 1]["actor"], "pai": "P"}


def test_same_semantics_with_different_ids_select_same_history():
    o = rotation_obs()
    a = ReconstructionOverrides(river_ids={(s, i): f"a-{s}-{i}"
        for s, river in enumerate(o.rivers) for i, _ in enumerate(river)})
    b = ReconstructionOverrides(river_ids={(s, i): f"b-{s}-{i}"
        for s, river in enumerate(o.rivers) for i, _ in enumerate(river)})
    ra, rb = reconstruct(o, a), reconstruct(o, b)
    assert ra.events == rb.events
    assert ra.selected_history == rb.selected_history
    assert ra.history_baseline != rb.history_baseline


def test_skeleton_budget_exhaustion_is_not_reported_as_user_conflict():
    from majsoul_eye.state.reconstruct import SkeletonBudget, _iter_skeletons
    budget = SkeletonBudget(limit=0)
    assert list(_iter_skeletons(rotation_obs(), 0, budget=budget)) == []
    assert budget.exhausted is True


def test_history_node_budget_exhaustion_aborts_search_without_hanging():
    # Plan 2 precondition: reverse() inside solve_hidden_history had no node
    # budget of its own -- only the skeleton COUNT was capped. A shared
    # HistoryNodeBudget (mirroring SkeletonBudget's injectable `limit`) must
    # cut the reverse-search short and (a) abort rather than hang, while
    # (b) surfacing the EXISTING HISTORY_SEARCH_LIMIT code -- never a
    # HIDDEN_HISTORY_CONFLICT/TSUMOGIRI_RULE_CONFLICT "user conflict", which
    # would misreport a budget cutoff as an unsolvable/contradictory board.
    from majsoul_eye.state.history import HistoryNodeBudget
    o = rotation_obs()
    overrides = ReconstructionOverrides(
        river_ids={(s, i): f"r-{s}-{i}" for s, river in enumerate(o.rivers)
                   for i, _ in enumerate(river)})
    budget = HistoryNodeBudget(limit=1)
    result = reconstruct(o, overrides, node_budget=budget)
    assert budget.exhausted is True                     # (a) aborted, not hung
    assert not result.ok
    assert result.issues[0]["code"] == "HISTORY_SEARCH_LIMIT"       # (b) reused
    assert result.issues[0]["code"] not in (
        "HIDDEN_HISTORY_CONFLICT", "TSUMOGIRI_RULE_CONFLICT", "NO_LEGAL_TURN_ORDER")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_history_solver OK")
