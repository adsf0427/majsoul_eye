"""Sanma (3P) ObservedState -> legal mjai. Every claim here is measured against
the 3P GT corpus by scripts/eval/verify_sanma_rules.py; these are the unit-level
guards that keep the code honest between corpus runs.

Sanma model: the 4-chair SCREEN ring survives; absolute chair 3 is a phantom that
renders empty. Its SCREEN slot rotates with the hero's chair
(phantom_rel = (3 - hero_abs) % 4), so nothing may key off index 3.
"""
from majsoul_eye.state.decision import analyze_hero_decision
from majsoul_eye.state.observe import (ObservedMeld, ObservedRiverTile,
                                       ObservedState, check_observed,
                                       observed_from_board)
from majsoul_eye.state.replay import Replayer, check_invariants
from majsoul_eye.state.reconstruct import reconstruct

# A sanma-legal 13-tile hand: no 2m-8m anywhere.
H13 = ["1m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "1s", "2s"]


def _obs(**kw):
    o = ObservedState(hero_hand=list(H13), dora_markers=["5s"],
                      rivers=[[], [], [], []], melds=[[], [], [], []],
                      sanma=True, phantom_rel=3)
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def _roundtrip(obs):
    """Reconstruct, replay, and project back: the board must be identical."""
    assert check_observed(obs) == [], check_observed(obs)
    r = reconstruct(obs)
    assert r.ok, r.reason
    rp = Replayer()
    for ev in r.events:
        rp.apply(ev)
    assert check_invariants(rp.state) == []
    back = observed_from_board(rp.state, include_hud=False)
    assert [[t.pai for t in riv] for riv in back.rivers] == \
           [[t.pai for t in riv] for riv in obs.rivers]
    assert [[(m.type, m.from_rel, sorted(m.tiles)) for m in ms] for ms in back.melds] == \
           [[(m.type, m.from_rel, sorted(m.tiles)) for m in ms] for ms in obs.melds]
    assert sorted(back.hero_hand) == sorted(obs.hero_hand)
    assert back.drawn_tile == obs.drawn_tile
    # The zone sanma adds. Without this the round-trip is blind to the whole feature.
    assert back.nukidora == obs.nukidora, (back.nukidora, obs.nukidora)
    return r


def _actors(events, *types):
    return [e["actor"] for e in events if e["type"] in types]


# --- the turn ring ----------------------------------------------------------

def test_turn_ring_skips_the_phantom_from_every_hero_chair():
    # phantom_rel p <=> hero_abs (3 - p) % 4. Absolute chair 3 must NEVER act.
    for phantom_rel, hero_abs in ((3, 0), (2, 1), (1, 2)):
        live = [r for r in range(4) if r != phantom_rel]
        rivers = [[] for _ in range(4)]
        for r in live:
            rivers[r] = [ObservedRiverTile("9p" if r == 0 else "E")]
        o = _obs(phantom_rel=phantom_rel, rivers=rivers)
        r = _roundtrip(o)
        assert r.events[0] == {"type": "start_game", "id": hero_abs}
        acted = {e["actor"] for e in r.events if "actor" in e}
        assert 3 not in acted, f"phantom acted (phantom_rel={phantom_rel})"
        assert acted == {0, 1, 2}


def test_turn_order_is_plus_one_over_live_chairs():
    # Hero at chair 1 (phantom on screen slot 2). Absolute turn order 0->1->2.
    o = _obs(phantom_rel=2, kyoku=1, seat_wind_self="S",
             rivers=[[ObservedRiverTile("9p")], [ObservedRiverTile("E")],
                     [], [ObservedRiverTile("S")]])
    r = _roundtrip(o)
    # kyoku 1 => oya == chair 0; hero wind S => hero is 1 after the oya => chair 1.
    assert r.events[1]["oya"] == 0
    assert r.events[0]["id"] == 1
    assert _actors(r.events, "dahai") == [0, 1, 2]


# --- nukidora ---------------------------------------------------------------

def test_nukidora_does_not_flip_a_dora():
    # 1 ankan + 2 dora markers => exactly ONE dora flip, and it belongs to the KAN.
    # Position is the assertion: flip_dora() is capped by len(dora_markers), so a
    # bogus flip on the pull would not change the COUNT -- only which event it
    # trails. Guards against "a nuki takes a dead-wall tile like a kan, so surely
    # it flips like a kan too" (it does not: 0/472 measured).
    o = _obs(hero_hand=["1m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p"],
             melds=[[ObservedMeld("ankan", ["9s"] * 4, "", "", 0)], [], [], []],
             dora_markers=["5s", "1s"], nukidora=[2, 0, 0, 0], drawn_tile="9p")
    r = _roundtrip(o)
    kinds = [e["type"] for e in r.events]
    assert kinds.count("dora") == 1
    assert kinds.count("nukidora") == 2
    dora_at = kinds.index("dora")
    assert kinds[dora_at - 1] == "ankan", "the dora flip must trail the KAN"
    for i, k in enumerate(kinds):
        if k == "nukidora":
            assert kinds[i + 1] != "dora", "a north pull must not flip a dora"


def test_nukidora_does_not_end_the_turn():
    # Measured shape: tsumo -> nukidora -> tsumo -> ... -> dahai.
    o = _obs(nukidora=[0, 2, 0, 0], rivers=[[ObservedRiverTile("9p")],
                                            [ObservedRiverTile("E")], [], []])
    r = _roundtrip(o)
    kinds = [e["type"] for e in r.events]
    for i, e in enumerate(r.events):
        if e["type"] != "nukidora":
            continue
        assert e["pai"] == "N"
        assert kinds[i - 1] == "tsumo" and r.events[i - 1]["actor"] == e["actor"]
        assert kinds[i + 1] == "tsumo" and r.events[i + 1]["actor"] == e["actor"], \
            "the replacement draw is what makes a pull not end the turn"


def test_hero_pull_survives_the_backward_hidden_history_solve():
    # The hero's pulled norths are gone from the hand and sitting in the pile; the
    # backward solve has to put them back to fabricate a legal 13-tile haipai.
    # The hero holds a fresh draw, so the ring must have come back around: the two
    # live opponents each took a turn.
    o = _obs(nukidora=[2, 0, 0, 0], drawn_tile="9s",
             rivers=[[ObservedRiverTile("9p")], [ObservedRiverTile("E")],
                     [ObservedRiverTile("S")], []])
    r = _roundtrip(o)
    haipai = r.events[1]["tehais"][0]
    assert len(haipai) == 13
    assert haipai.count("N") <= 4
    assert [e["type"] for e in r.events].count("nukidora") == 2


def test_four_norths_pulled_is_the_ceiling_and_still_solves():
    o = _obs(nukidora=[4, 0, 0, 0], drawn_tile="9s",
             rivers=[[ObservedRiverTile("9p")], [ObservedRiverTile("E")],
                     [ObservedRiverTile("S")], []])
    r = _roundtrip(o)
    assert [e["type"] for e in r.events].count("nukidora") == 4


def test_north_pulls_cost_ZERO_extra_skeletons():
    """The pull placement is CHOSEN, not searched — that is the whole reason the
    feature fits. Searching WHEN each seat pulled is C(turns, k) per seat (~1e10
    across three seats) against a 4096-skeleton budget, so the forced, unbranching
    prefix is not an optimisation but the only way this terminates.

    Guard: on the same board, the skeleton count must not move when piles appear.
    A future 'let's also try pulling on turn 2' would blow up here, loudly,
    instead of quietly turning every call-heavy sanma board into a search-limit
    failure in production.
    """
    import copy
    from majsoul_eye.state.reconstruct import (SkeletonBudget, _iter_skeletons,
                                               _oya_candidates)

    def count(o):
        total = 0
        for oya_rel in _oya_candidates(o):
            budget = SkeletonBudget(limit=50_000)
            for _ in _iter_skeletons(o, oya_rel, pending_reach=None, budget=budget):
                total += 1
        return total

    # A deliberately call-heavy board: call timing is what the DFS actually
    # enumerates, so this is where an extra branch would show up.
    rivers = [[ObservedRiverTile(t) for t in ("9p", "1s", "2s")],
              [ObservedRiverTile(t) for t in ("E", "S", "W")],
              [ObservedRiverTile(t) for t in ("P", "F", "C")],
              []]
    melds = [[], [ObservedMeld("pon", ["9s"] * 3, "9s", "", 2)],
             [ObservedMeld("pon", ["1p"] * 3, "1p", "", 1)], []]
    base = _obs(hero_hand=list(H13), rivers=rivers, melds=melds)
    with_piles = copy.deepcopy(base)
    with_piles.nukidora = [1, 2, 1, 0]
    assert count(base) == count(with_piles), \
        "a north pile must not multiply the skeleton search"


def test_a_pull_costs_a_wall_tile_like_a_kan():
    # 55 live wall; each pull -1, exactly like a kan (MEASURED V1).
    o = _obs(nukidora=[2, 1, 0, 0], left_tile_count=55 - 3)
    assert check_observed(o) == []
    o.left_tile_count = 55            # as if the pulls had cost nothing
    assert any("wall count" in v for v in check_observed(o))


def test_the_four_copy_budget_counts_the_piles():
    # 3 norths on the table + 2 in hand = 5 > 4. Without counting the piles this
    # board sails into the solver and dies as an inscrutable history conflict.
    hand = ["N", "N", "1m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p"]
    o = _obs(hero_hand=hand, nukidora=[3, 0, 0, 0])
    assert any("N seen 5>4" in v for v in check_observed(o))


# --- sanma structure --------------------------------------------------------

def test_start_kyoku_carries_the_measured_3p_shape():
    o = _obs()
    r = reconstruct(o)
    assert r.ok
    sk = r.events[1]
    assert sk["scores"] == [35000, 35000, 35000, 0]     # phantom padded with 0
    assert len(sk["tehais"]) == 4                        # 4 arrays, phantom included
    assert sk["tehais"][3] == ["?"] * 13


def test_dealer_is_kyoku_minus_one_mod_three_from_every_chair():
    for hero_abs in (0, 1, 2):
        for oya_abs in (0, 1, 2):
            phantom_rel = (3 - hero_abs) % 4
            wind = "ESW"[(hero_abs - oya_abs) % 3]
            o = _obs(phantom_rel=phantom_rel, kyoku=oya_abs + 1,
                     seat_wind_self=wind, bakaze="E")
            r = reconstruct(o)
            assert r.ok, (hero_abs, oya_abs, r.reason)
            assert r.events[0]["id"] == hero_abs, (hero_abs, oya_abs)
            assert r.events[1]["oya"] == oya_abs, (hero_abs, oya_abs)


def test_sanma_has_no_two_through_eight_man():
    o = _obs(hero_hand=["3m"] + H13[1:])
    assert any("sanma wall" in v for v in check_observed(o))
    o = _obs(dora_markers=["5mr"])
    assert any("sanma wall" in v for v in check_observed(o))


def test_sanma_has_no_chi():
    o = _obs(hero_hand=H13[:10],
             melds=[[ObservedMeld("chi", ["1p", "2p", "3p"], "1p", "", 3)], [], [], []])
    assert any("no chi" in v for v in check_observed(o))


def test_the_phantom_chair_must_be_empty():
    o = _obs(rivers=[[], [], [], [ObservedRiverTile("E")]])   # phantom_rel is 3
    assert any("phantom" in v for v in check_observed(o))


def test_points_conserve_to_105000():
    o = _obs(scores=[35000, 35000, 35000, 0], kyotaku=0)
    assert check_observed(o) == []
    o = _obs(scores=[25000, 25000, 25000, 25000], kyotaku=0)   # the 4P total
    assert any("105000" in v for v in check_observed(o))


# --- decision ---------------------------------------------------------------

def _dstate(hand, draw=None, **kw):
    o = ObservedState(hero_hand=list(hand), drawn_tile=draw, dora_markers=["5s"],
                      scores=[35000, 35000, 35000, 0], kyotaku=0,
                      left_tile_count=40, sanma=True, phantom_rel=3)
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def test_holding_a_north_offers_the_pull():
    o = _dstate(["N", "1m", "9m", "1p", "2p", "3p", "5p", "6p", "7p", "9p", "1s", "2s", "3s"], "9s")
    assert "nukidora:N" in analyze_hero_decision(o).decision["legalActions"]


def test_no_north_no_pull():
    o = _dstate(H13, "9s")
    assert "nukidora:N" not in analyze_hero_decision(o).decision["legalActions"]


def test_riichi_freezes_the_hand_so_only_the_drawn_north_may_be_pulled():
    hand = ["N", "1m", "9m", "1p", "2p", "3p", "5p", "6p", "7p", "9p", "1s", "2s", "3s"]
    held = _dstate(hand, "9s", reach=[True, False, False, False])
    assert "nukidora:N" not in analyze_hero_decision(held).decision["legalActions"], \
        "pulling a north out of a riichi hand would change the hand"
    drawn = _dstate(hand[1:], "N", reach=[True, False, False, False])
    assert "nukidora:N" in analyze_hero_decision(drawn).decision["legalActions"]


def test_four_player_boards_never_offer_a_pull():
    o = _dstate(["N", "1m", "9m", "1p", "2p", "3p", "5p", "6p", "7p", "9p", "1s", "2s", "3s"],
                "9s", sanma=False, phantom_rel=None,
                scores=[25000] * 4)
    assert "nukidora:N" not in analyze_hero_decision(o).decision["legalActions"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_sanma_reconstruct OK")
