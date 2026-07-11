from majsoul_eye.state.decision import analyze_hero_decision
from majsoul_eye.state.observe import ObservedMeld, ObservedRiverTile, ObservedState


def state(hand, draw=None):
    return ObservedState(hero_hand=list(hand), drawn_tile=draw,
                         dora_markers=["5s"], scores=[25000] * 4,
                         left_tile_count=50)


def test_normal_draw_lists_canonical_distinct_discards():
    o = state(["1m", "1m", "2m", "3m", "4m", "5mr", "5m",
               "6p", "7p", "8p", "E", "E", "E"], "9s")
    result = analyze_hero_decision(o)
    assert result.decision["actorRelSeat"] == 0
    assert result.decision["legalDiscards"] == [
        "1m", "2m", "3m", "4m", "5mr", "5m", "6p", "7p", "8p", "9s", "E"]
    assert result.decision["candidateCount"] == len(result.decision["legalActions"])


def test_tsumo_agari_is_explicit_blocking_action():
    o = state(["1m", "1m", "1m", "2m", "3m", "4m", "2p", "3p", "4p",
               "2s", "3s", "4s", "E"], "E")
    result = analyze_hero_decision(o)
    assert "hora:tsumo" in result.decision["legalActions"]
    assert any(i["code"] == "AGARI_AVAILABLE" and i["severity"] == "blocking"
               for i in result.issues)


def test_open_complete_shape_without_yaku_must_still_discard():
    o = state(["1p", "2p", "3p", "4s", "5s", "6s", "7m", "8m", "9m", "E"], "E")
    o.melds[0] = [ObservedMeld("pon", ["2m", "2m", "2m"], "2m", "", 1)]
    result = analyze_hero_decision(o)
    assert "hora:tsumo" not in result.decision["legalActions"]
    assert not any(i["code"] == "AGARI_AVAILABLE" for i in result.issues)


def test_open_yakuhai_shape_can_tsumo():
    o = state(["1p", "2p", "3p", "4s", "5s", "6s", "7m", "8m", "9m", "E"], "E")
    o.melds[0] = [ObservedMeld("pon", ["P", "P", "P"], "P", "", 1)]
    assert "hora:tsumo" in analyze_hero_decision(o).decision["legalActions"]


def test_open_tanyao_shape_can_tsumo_under_majsoul_rules():
    o = state(["2p", "3p", "4p", "4s", "5s", "6s", "6m", "7m", "8m", "5p"], "5p")
    o.melds[0] = [ObservedMeld("pon", ["2m", "2m", "2m"], "2m", "", 1)]
    assert "hora:tsumo" in analyze_hero_decision(o).decision["legalActions"]


def test_call_pending_only_allows_discards():
    o = state(["1m", "2m", "3m", "4m", "5m", "6m", "7p", "8p", "9p", "E", "E"])
    o.melds[0] = [ObservedMeld("pon", ["P", "P", "P"], "P", "", 2)]
    result = analyze_hero_decision(o)
    assert result.decision is not None
    assert all(action.startswith("dahai:") for action in result.decision["legalActions"])


def test_ts_kuikae_vector_filters_called_tile_and_same_sequence_swap():
    # Parity with rulesUpdater.ts chi forbiddenTiles and discardMask.ts case 3.
    o = state(["3m", "6m", "1p", "2p", "3p", "4p", "5p", "6p", "7s", "8s", "9s"])
    o.melds[0] = [ObservedMeld("chi", ["3m", "4m", "5m"], "3m", "", 3)]
    result = analyze_hero_decision(o)
    assert "3m" not in result.decision["legalDiscards"]
    assert "6m" not in result.decision["legalDiscards"]
    assert "1p" in result.decision["legalDiscards"]


def test_ts_pon_kuikae_vector_filters_same_tile():
    o = state(["P", "1m", "2m", "3m", "4m", "5m", "6m", "7p", "8p", "9p", "E"])
    o.melds[0] = [ObservedMeld("pon", ["P", "P", "P"], "P", "", 2)]
    result = analyze_hero_decision(o)
    assert "P" not in result.decision["legalDiscards"]
    assert "1m" in result.decision["legalDiscards"]


def test_reach_and_kan_action_grammar_and_order():
    o = state(["1m", "1m", "1m", "1m", "2m", "3m", "4m", "5p", "6p",
               "7p", "8s", "8s", "E"], "E")
    result = analyze_hero_decision(o)
    assert "ankan:1m" in result.decision["legalActions"]
    assert result.decision["legalActions"] == sorted(
        result.decision["legalActions"], key=result.action_sort_key)


def test_reach_candidate_names_the_declaring_discard():
    o = state(["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s",
               "7s", "8s", "9s", "E"], "9p")
    result = analyze_hero_decision(o)
    assert "reach:9p" in result.decision["legalActions"]


def test_ts_golden_post_riichi_honor_ankan_preserves_wait():
    # Literal parity vector from frontend-react/src/utils/mahjong/rules/
    # __tests__/editorIssues.spec.ts, “post-riichi ankan + dora reveal”.
    o = state(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
               "W", "W", "W", "1p"], "W")
    o.reach[0] = True
    result = analyze_hero_decision(o)
    assert "ankan:W" in result.decision["legalActions"]


def test_post_riichi_suited_ankan_that_loses_a_wait_is_rejected():
    # Before draw waits are 1m/4m/8m; ankan 1m would lose the 8m interpretation.
    o = state(["1m", "1m", "1m", "2m", "3m", "8m", "8m",
               "3p", "4p", "5p", "3s", "3s", "3s"], "1m")
    o.reach[0] = True
    result = analyze_hero_decision(o)
    assert "ankan:1m" not in result.decision["legalActions"]


def test_one_legal_discard_plus_ankan_is_still_not_meaningful():
    # Before and after ankan 1m, the sole wait remains 7p.
    o = state(["1m", "1m", "1m", "3m", "3m", "3m", "6m", "7m", "8m",
               "4p", "4p", "8p", "9p"], "1m")
    o.reach[0] = True
    result = analyze_hero_decision(o)
    assert result.decision["legalDiscards"] == ["1m"]
    assert "ankan:1m" in result.decision["legalActions"]
    assert result.decision["candidateCount"] == len(result.decision["legalActions"])
    assert any(issue["code"] == "NO_MEANINGFUL_CHOICE"
               for issue in result.issues)


def test_ts_golden_matching_pon_exposes_only_matching_kakan():
    # Literal parity vector from frontend-react/src/utils/mahjong/shorthand.spec.ts,
    # “kanSelf: kakan when a matching pon meld exists”.
    o = state(["7p", "1m", "2m", "4s", "4s", "6s", "7s", "8s",
               "9m", "9m"], "4m")
    o.melds[0] = [ObservedMeld("pon", ["7p", "7p", "7p"], "7p", "", 1)]
    result = analyze_hero_decision(o)
    assert "kakan:7p" in result.decision["legalActions"]
    assert "kakan:9s" not in result.decision["legalActions"]


def test_first_uninterrupted_draw_exposes_kyushukyuhai_but_keeps_discards():
    o = state(["1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W",
               "2m", "3m", "4m", "5m"], "N")
    result = analyze_hero_decision(o)
    assert "ryukyoku:kyushukyuhai" in result.decision["legalActions"]
    assert any(action.startswith("dahai:") for action in result.decision["legalActions"])
    assert any(i["code"] == "ABORTIVE_DRAW_AVAILABLE" for i in result.issues)

    o.rivers[0].append(ObservedRiverTile("2p"))
    assert "ryukyoku:kyushukyuhai" not in analyze_hero_decision(o).decision["legalActions"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_what_cut_decision OK")
