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


def test_pon_ghost_discard_and_forced_tedashi():
    # hero(rel0) pon P from rel2 (across). hero hand 10 + pon = 13.
    o = _obs(hero_hand=H13[:10],
             melds=[[ObservedMeld("pon", ["P", "P", "P"], called_pai="P", from_rel=2)],
                    [], [], []],
             rivers=[[ObservedRiverTile("9p")],          # hero's forced tedashi after pon
                     [ObservedRiverTile("E")],
                     [ObservedRiverTile("S")],           # across also has a VISIBLE discard
                     []])
    r = _roundtrip(o)
    evs = r.events
    pon = next(e for e in evs if e["type"] == "pon")
    assert pon["actor"] == 0 and pon["target"] == 2 and pon["consumed"] == ["P", "P"]
    # ghost dahai P by seat 2 immediately precedes the pon
    ghost = evs[evs.index(pon) - 1]
    assert ghost == {"type": "dahai", "actor": 2, "pai": "P", "tsumogiri": False}
    # hero's discard right after the pon is tedashi and lands in haipai
    after = evs[evs.index(pon) + 1]
    assert after["type"] == "dahai" and after["actor"] == 0 and after["tsumogiri"] is False
    sk = evs[1]
    assert sorted(sk["tehais"][0]) == sorted(H13[:10] + ["P", "P", "9p"])


def test_chi_only_from_kamicha():
    # NOTE: rel3 (kamicha) must NOT also carry a second, later visible discard
    # here (e.g. an extra "W") the way the pon test gives its target seat one:
    # chi's target is always adjacent to the caller (kamicha -> hero), so the
    # call never skips a seat the way pon/kan can. With rel1/rel2 both empty,
    # rotation can only ever reach rel3 once (as oya's first turn); asking it
    # for a second discard with no way to lap back through 1/2 is an
    # unreachable board state, not a search-algorithm limitation.
    o = _obs(hero_hand=H13[:10],
             melds=[[ObservedMeld("chi", ["4s", "5s", "6s"], called_pai="5s", from_rel=3)],
                    [], [], []],
             rivers=[[ObservedRiverTile("9p")], [], [], []])
    r = _roundtrip(o)
    chi = next(e for e in r.events if e["type"] == "chi")
    assert chi["target"] == 3 and sorted(chi["consumed"]) == ["4s", "6s"]


def test_call_timing_needs_backtracking():
    # Late-call-first fails: 0:A -> 1:C -> 2 has nothing => backtrack to the
    # ghost branch (1 discards ghost P before its visible C).
    o = _obs(hero_hand=H13[:10],
             melds=[[ObservedMeld("pon", ["P", "P", "P"], called_pai="P", from_rel=1)],
                    [], [], []],
             rivers=[[ObservedRiverTile("1m"), ObservedRiverTile("9p")],
                     [ObservedRiverTile("2m")], [], []])
    _roundtrip(o)


def test_daiminkan_by_opponent():
    # rel1 daiminkan's C from hero (from_rel=3 -> target rel0): hero discards a
    # ghost C, rel1 kans + rinshan-draws + discards F. dora NOT yet flipped
    # (1 marker for 1 kan): allowed (daiminkan flip is delayed).
    o = _obs(melds=[[], [ObservedMeld("daiminkan", ["C", "C", "C", "C"],
                                      called_pai="C", from_rel=3)], [], []],
             rivers=[[ObservedRiverTile("9p")],
                     [ObservedRiverTile("E"), ObservedRiverTile("F")],
                     [ObservedRiverTile("S")], [ObservedRiverTile("W")]],
             dora_markers=["5s"])
    r = _roundtrip(o)
    kan = next(e for e in r.events if e["type"] == "daiminkan")
    assert kan["actor"] == 1 and kan["target"] == 0 and kan["consumed"] == ["C", "C", "C"]
    # the ghost C came from the hero: it was that turn's fabricated draw
    ghost = r.events[[i for i, e in enumerate(r.events)
                      if e["type"] == "dahai" and e["pai"] == "C"][0]]
    assert ghost["actor"] == 0 and ghost["tsumogiri"] is True


def test_opponent_riichi_reach_events_and_tsumogiri():
    o = _obs(rivers=[[ObservedRiverTile("9p"), ObservedRiverTile("1p")],
                     [ObservedRiverTile("E"), ObservedRiverTile("W", sideways=True),
                      ObservedRiverTile("N")],
                     [ObservedRiverTile("S"), ObservedRiverTile("S")],
                     [ObservedRiverTile("C"), ObservedRiverTile("F")]],
             reach=[False, True, False, False])
    r = _roundtrip(o)
    evs = r.events
    ri = next(i for i, e in enumerate(evs) if e["type"] == "reach")
    assert evs[ri]["actor"] == 1
    assert evs[ri + 1]["type"] == "dahai" and evs[ri + 1]["pai"] == "W"
    assert evs[ri + 2] == {"type": "reach_accepted", "actor": 1}
    # post-riichi discards by seat 1 are forced tsumogiri
    later = [e for e in evs[ri + 3:] if e["type"] == "dahai" and e["actor"] == 1]
    assert later and all(e["tsumogiri"] for e in later)
    # backfill: kyotaku defaults to observed riichi count -> start 0; score +1000
    sk = evs[1]
    assert sk["kyotaku"] == 0 and sk["scores"][1] == 26000


def test_hero_ankan_with_kandora():
    o = _obs(hero_hand=H13[:9] + ["C"],                     # 10 concealed
             melds=[[ObservedMeld("ankan", ["F", "F", "F", "F"])], [], [], []],
             rivers=[[ObservedRiverTile("9p")], [ObservedRiverTile("E")],
                     [ObservedRiverTile("S")], [ObservedRiverTile("W")]],
             dora_markers=["5s", "6s"])
    r = _roundtrip(o)
    evs = r.events
    ak = next(i for i, e in enumerate(evs) if e["type"] == "ankan")
    assert evs[ak]["consumed"] == ["F", "F", "F", "F"]
    assert evs[ak + 1] == {"type": "dora", "dora_marker": "6s"}
    # the 4th F was that turn's draw: haipai holds only 3 F
    assert evs[1]["tehais"][0].count("F") == 3


def test_kakan_pon_then_upgrade():
    # On-screen kakan = TWO events at different times: its pon (needs rel1's
    # ghost P) and the own-turn upgrade. The frame ends with hero holding the
    # rinshan draw, so the search must finish 'kakan -> rinshan tsumo'.
    o = _obs(hero_hand=H13[:10],
             melds=[[ObservedMeld("kakan", ["P", "P", "P", "P"],
                                  called_pai="P", added_pai="P", from_rel=1)],
                    [], [], []],
             rivers=[[ObservedRiverTile("9p")], [ObservedRiverTile("E")],
                     [ObservedRiverTile("S")], [ObservedRiverTile("W")]],
             drawn_tile="6s", dora_markers=["5s", "6s"])
    r = _roundtrip(o)
    kinds = [e["type"] for e in r.events]
    assert kinds.index("pon") < kinds.index("kakan")
    kk = next(e for e in r.events if e["type"] == "kakan")
    assert kk["pai"] == "P" and kk["consumed"] == ["P", "P", "P"]
    ki = r.events.index(kk)
    assert r.events[ki + 1] == {"type": "dora", "dora_marker": "6s"}
    assert r.events[-1] == {"type": "tsumo", "actor": 0, "pai": "6s"}
    # haipai: 10 concealed + pon's [P,P] + forced tedashi 9p = 13
    assert sorted(r.events[1]["tehais"][0]) == sorted(H13[:10] + ["P", "P", "9p"])


def test_riichi_tile_claimed_ghost_reach():
    # rel2's riichi declaration tile was ponned by rel1 -> rel2's NEXT discard
    # renders sideways. Search must bind reach to the ghost.
    o = _obs(hero_hand=H13[:13],
             melds=[[], [ObservedMeld("pon", ["W", "W", "W"], called_pai="W",
                                      from_rel=1)], [], []],
             rivers=[[ObservedRiverTile("9p"), ObservedRiverTile("1p")],
                     [ObservedRiverTile("E"), ObservedRiverTile("F")],
                     [ObservedRiverTile("S"), ObservedRiverTile("N", sideways=True)],
                     [ObservedRiverTile("C"), ObservedRiverTile("P")]],
             reach=[False, False, True, False])
    r = _roundtrip(o)
    evs = r.events
    ri = next(i for i, e in enumerate(evs) if e["type"] == "reach")
    assert evs[ri]["actor"] == 2
    nxt = evs[ri + 1]
    assert nxt["type"] == "dahai" and nxt["actor"] == 2
    # either binding is legal; the DECLARATION discard may be W (ghost) or N
    assert nxt["pai"] in ("W", "N")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_reconstruct OK")
