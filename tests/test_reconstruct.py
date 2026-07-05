"""ObservedState -> legal mjai sequence. Every case round-trips through the
Replayer and must project back to the exact same ObservedState."""
from majsoul_eye.state.observe import ObservedMeld, ObservedRiverTile, ObservedState
from majsoul_eye.state.reconstruct import reconstruct

H13 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]


def _obs(**kw):
    o = ObservedState(hero_hand=list(H13), dora_markers=["5s"],
                      rivers=[[], [], [], []], melds=[[], [], [], []])
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def _roundtrip(obs):
    from majsoul_eye.state.observe import observed_from_board
    from majsoul_eye.state.replay import Replayer, check_invariants
    r = reconstruct(obs)
    assert r.ok, r.reason
    rp = Replayer()
    for ev in r.events:
        rp.apply(ev)
    assert check_invariants(rp.state) == []
    back = observed_from_board(rp.state, include_hud=False)
    assert [[t.pai for t in riv] for riv in back.rivers] == \
           [[t.pai for t in riv] for riv in obs.rivers]
    assert [[t.sideways for t in riv] for riv in back.rivers] == \
           [[t.sideways for t in riv] for riv in obs.rivers]
    assert [[(m.type, m.from_rel, sorted(m.tiles)) for m in ms] for ms in back.melds] == \
           [[(m.type, m.from_rel, sorted(m.tiles)) for m in ms] for ms in obs.melds]
    assert sorted(back.hero_hand) == sorted(obs.hero_hand)
    assert back.drawn_tile == obs.drawn_tile
    assert back.dora_markers == obs.dora_markers
    return r


def test_empty_board_start_of_kyoku():
    r = _roundtrip(_obs())
    assert [e["type"] for e in r.events] == ["start_game", "start_kyoku"]


def test_rotation_only():
    o = _obs(rivers=[[ObservedRiverTile("9p")], [ObservedRiverTile("E")],
                     [ObservedRiverTile("S")], []])
    r = _roundtrip(o)
    kinds = [e["type"] for e in r.events]
    assert kinds == ["start_game", "start_kyoku",
                     "tsumo", "dahai", "tsumo", "dahai", "tsumo", "dahai"]
    # oya inferred = rel 0 (hero discarded first); hero_abs defaults to 0
    sk = r.events[1]
    assert sk["oya"] == 0 and sk["kyoku"] == 1 and sk["bakaze"] == "E"
    assert sorted(sk["tehais"][0]) == sorted(H13)           # all-tsumogiri: haipai == hand
    assert sk["tehais"][1] == ["?"] * 13
    # hero discard is tsumo-then-cut of the same tile
    assert r.events[2] == {"type": "tsumo", "actor": 0, "pai": "9p"}
    assert r.events[3]["tsumogiri"] is True


def test_oya_inferred_uniquely_from_river_lengths():
    o = _obs(rivers=[[ObservedRiverTile("9p")], [ObservedRiverTile("E")],
                     [ObservedRiverTile("S")], []])
    r = reconstruct(o)
    assert r.diagnostics["feasible_oya_rel"] == [0]


def test_hero_holding_draw():
    o = _obs(rivers=[[ObservedRiverTile("9p")], [ObservedRiverTile("E")],
                     [ObservedRiverTile("S")], [ObservedRiverTile("W")]],
             drawn_tile="6s")
    r = _roundtrip(o)
    assert r.events[-1] == {"type": "tsumo", "actor": 0, "pai": "6s"}


def test_hud_fields_flow_into_start_kyoku():
    o = _obs(rivers=[[], [ObservedRiverTile("E")], [ObservedRiverTile("S")],
                     [ObservedRiverTile("W")]],
             bakaze="S", kyoku=3, honba=2, kyotaku=1, seat_wind_self="N",
             scores=[24000, 26000, 25000, 25000])
    # seat_wind N -> oya_rel = (4 - 3) % 4 = 1; kyoku 3 -> oya_abs 2 -> hero_abs 1
    r = _roundtrip(o)
    sk = r.events[1]
    assert sk["bakaze"] == "S" and sk["kyoku"] == 3 and sk["oya"] == 2
    assert sk["honba"] == 2 and sk["kyotaku"] == 1
    assert r.events[0] == {"type": "start_game", "id": 1}
    assert sk["scores"][1] == 24000                          # hero_abs=1 slot = rel0 score
    assert sorted(sk["tehais"][1]) == sorted(H13)


def test_infeasible_reports_reason():
    # rel3 discarded but rel2 hasn't and nothing explains the skip -> no legal order
    o = _obs(rivers=[[ObservedRiverTile("9p")], [ObservedRiverTile("E")],
                     [], [ObservedRiverTile("W"), ObservedRiverTile("N")]])
    r = reconstruct(o)
    assert not r.ok and r.reason


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_reconstruct OK")
