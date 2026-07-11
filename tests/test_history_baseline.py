from majsoul_eye.state.history import ReconstructionOverrides
from majsoul_eye.state.observe import ObservedMeld, ObservedRiverTile, ObservedState
from majsoul_eye.state.reconstruct import reconstruct


H13 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
       "1p", "2p", "3p", "4p"]


def obs():
    # Reachable turn order: oya is seat 1; counts [2, 3, 3, 3] end with hero
    # (seat 0) holding the pending draw. A no-meld board can never have river
    # counts differing by more than 1 — turns are strictly round-robin.
    return ObservedState(hero_hand=list(H13), drawn_tile="5p",
                         rivers=[[ObservedRiverTile("9p"), ObservedRiverTile("1s")],
                                 [ObservedRiverTile("E"), ObservedRiverTile("W", sideways=True),
                                  ObservedRiverTile("N")],
                                 [ObservedRiverTile("S"), ObservedRiverTile("2s"),
                                  ObservedRiverTile("4s")],
                                 [ObservedRiverTile("C"), ObservedRiverTile("3s"),
                                  ObservedRiverTile("6s")]],
                         dora_markers=["5s"], reach=[False, True, False, False])


def ids_for(o):
    overrides = ReconstructionOverrides()
    for seat, river in enumerate(o.rivers):
        for index, _ in enumerate(river):
            overrides.river_ids[(seat, index)] = f"r-{seat}-{index}"
    return overrides


def test_baseline_has_each_river_once_in_screen_order():
    o = obs()
    result = reconstruct(o, ids_for(o))
    assert result.ok, result.reason
    assert [item["itemId"] for item in result.history_baseline] == [
        "r-0-0", "r-0-1", "r-1-0", "r-1-1", "r-1-2",
        "r-2-0", "r-2-1", "r-2-2", "r-3-0", "r-3-1", "r-3-2"]
    post_reach = next(item for item in result.history_baseline if item["itemId"] == "r-1-2")
    assert post_reach == {"itemKind": "river", "itemId": "r-1-2",
                          "baselineValue": True, "baselineSource": "forced"}


def test_editing_reach_rederives_baseline_instead_of_reusing_request_value():
    o = obs()
    first = reconstruct(o, ids_for(o))
    o.reach[1] = False
    o.rivers[1][1].sideways = False
    second = reconstruct(o, ids_for(o))
    assert first.ok and second.ok
    item = next(x for x in second.history_baseline if x["itemId"] == "r-1-2")
    assert item["baselineValue"] is False and item["baselineSource"] == "inferred"


def test_ghosts_follow_rivers_and_keep_draft_order():
    o = ObservedState(hero_hand=H13[:10], drawn_tile="5p", dora_markers=["5s"])
    o.melds[0] = [ObservedMeld("pon", ["P", "P", "P"], "P", "", 2)]
    o.rivers[0] = [ObservedRiverTile("9p")]
    o.rivers[1] = [ObservedRiverTile("E")]
    o.rivers[2] = [ObservedRiverTile("S")]
    # seat 3 must discard once: a zero-river, never-called seat dead-ends every
    # rotation when hero also holds a pending draw
    o.rivers[3] = [ObservedRiverTile("7s")]
    overrides = ids_for(o)
    overrides.ghost_ids[(0, 0)] = "g-0"
    overrides.ghost_order = [(0, 0)]
    result = reconstruct(o, overrides)
    assert result.ok, result.reason
    assert result.history_baseline[-1]["itemKind"] == "ghost"
    assert result.history_baseline[-1]["itemId"] == "g-0"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_history_baseline OK")
