from __future__ import annotations

from dataclasses import dataclass

from mahjong.agari import Agari
from mahjong.constants import TERMINAL_AND_HONOR_INDICES
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld
from mahjong.shanten import Shanten

from majsoul_eye.state.observe import ObservedState
from majsoul_eye.state.reconstruct import _hero_call_pending
from majsoul_eye.state.replay import _PAI_RANK
from majsoul_eye.tiles import TILE34_TO_ID, from_mjai, red_to_normal
from majsoul_eye.what_cut.schema import (
    SelectedHistoryV1, WhatCutDecisionV1, WhatCutIssueV1,
)

_GROUP = {"dahai": 0, "reach": 1, "ankan": 2, "kakan": 3, "nukidora": 4,
          "hora": 5, "ryukyoku": 6}
_RED_136 = {"5mr": 16, "5pr": 52, "5sr": 88}
_RED_IDS = set(_RED_136.values())


@dataclass
class DecisionAnalysis:
    decision: WhatCutDecisionV1 | None
    issues: list[WhatCutIssueV1]

    @staticmethod
    def action_sort_key(action: str):
        kind, value = action.split(":", 1)
        tile = red_to_normal(from_mjai(value)) if kind == "ankan" else from_mjai(value)
        return (_GROUP[kind], _PAI_RANK.get(tile, 10_000), value)


def _issue(code: str, severity: str = "blocking") -> WhatCutIssueV1:
    return {"code": code, "severity": severity, "fieldPath": None,
            "evidenceIds": [], "messageKey": f"whatCut.issue.{code}", "params": {}}


def _canonical(tile: str) -> str:
    tile = from_mjai(tile)
    return {"0m": "5mr", "0p": "5pr", "0s": "5sr"}.get(tile, tile)


def _counts(tiles: list[str]) -> list[int]:
    result = [0] * 34
    for tile in tiles:
        result[TILE34_TO_ID[red_to_normal(_canonical(tile))]] += 1
    return result


def _fixed_sets(obs: ObservedState) -> list[list[int]]:
    return [[TILE34_TO_ID[red_to_normal(_canonical(tile))] for tile in meld.tiles]
            for meld in obs.melds[0]]


def _waits(counts: list[int], fixed_sets: list[list[int]] | None = None) -> tuple[int, ...]:
    fixed_sets = fixed_sets or []
    full = list(counts)
    fixed_counts = [0] * 34
    for meld in fixed_sets:
        for tile_id in meld:
            full[tile_id] += 1
            fixed_counts[tile_id] += 1
    waits = []
    for index in range(34):
        if counts[index] + fixed_counts[index] >= 4:
            continue
        candidate = list(full); candidate[index] += 1
        if Agari.is_agari(candidate, open_sets_34=fixed_sets):
            waits.append(index)
    return tuple(waits)


def _closed(obs: ObservedState) -> bool:
    return all(meld.type == "ankan" for meld in obs.melds[0])


def _kuikae_forbidden(obs: ObservedState) -> set[str]:
    if not _hero_call_pending(obs):
        return set()
    meld = obs.melds[0][-1]
    called = TILE34_TO_ID[red_to_normal(_canonical(meld.called_pai))]
    forbidden = {red_to_normal(_canonical(meld.called_pai))}
    if meld.type != "chi":
        return forbidden
    consumed = list(meld.tiles)
    for index, tile in enumerate(consumed):
        if _canonical(tile) == _canonical(meld.called_pai):
            consumed.pop(index)
            break
    else:
        for index, tile in enumerate(consumed):
            if red_to_normal(_canonical(tile)) == red_to_normal(_canonical(meld.called_pai)):
                consumed.pop(index)
                break
    consumed_ids = sorted(TILE34_TO_ID[red_to_normal(_canonical(tile))]
                          for tile in consumed)
    low, high = consumed_ids
    if called < low and high % 9 < 8:
        forbidden.add(next(tile for tile, tile_id in TILE34_TO_ID.items()
                           if tile_id == high + 1))
    elif called > high and low % 9 > 0:
        forbidden.add(next(tile for tile, tile_id in TILE34_TO_ID.items()
                           if tile_id == low - 1))
    return forbidden


def _first_uninterrupted_draw(obs: ObservedState,
                              selected: SelectedHistoryV1 | None) -> bool:
    if obs.drawn_tile is None or any(obs.melds) or obs.rivers[0]:
        return False
    if selected is None:
        return all(len(obs.rivers[seat]) <= 1 for seat in (1, 2, 3))
    operations = selected["operations"]
    if not operations or operations[-1]["kind"] != "draw" \
            or operations[-1]["actorRelSeat"] != 0:
        return False
    if any(op["kind"] in ("call", "ankan", "kakan", "ghost") for op in operations):
        return False
    prior = [0, 0, 0, 0]
    for op in operations[:-1]:
        if op["kind"] == "river":
            prior[op["actorRelSeat"]] += 1
    return prior[0] == 0 and all(count <= 1 for count in prior[1:])


def _is_rinshan(obs: ObservedState, selected: SelectedHistoryV1 | None) -> bool:
    if selected is None or len(selected["operations"]) < 2:
        return False
    previous, current = selected["operations"][-2:]
    if current["kind"] != "draw" or current["actorRelSeat"] != 0:
        return False
    if previous["kind"] in ("ankan", "kakan") and previous["actorRelSeat"] == 0:
        return True
    if previous["kind"] == "call" and previous["actorRelSeat"] == 0:
        meld_index = previous["meldIndex"]
        return meld_index is not None and obs.melds[0][meld_index].type == "daiminkan"
    return False


def _allocate_136(groups: list[list[str]]) -> list[list[int]]:
    used: set[int] = set()
    allocated = []
    for group in groups:
        out = []
        for raw in group:
            tile = _canonical(raw)
            if tile in _RED_136:
                candidates = [_RED_136[tile]]
            else:
                base = TILE34_TO_ID[red_to_normal(tile)] * 4
                candidates = [value for value in range(base, base + 4)
                              if value not in _RED_IDS]
            chosen = next((value for value in candidates if value not in used), None)
            if chosen is None:
                raise ValueError(f"cannot allocate physical copy for {tile}")
            used.add(chosen); out.append(chosen)
        allocated.append(out)
    return allocated


def _calculator_input(obs: ObservedState, current: list[str]):
    groups = [current] + [list(meld.tiles) for meld in obs.melds[0]]
    allocated = _allocate_136(groups)
    calculator_melds = []
    kind = {"chi": Meld.CHI, "pon": Meld.PON, "daiminkan": Meld.KAN,
            "ankan": Meld.KAN, "kakan": Meld.SHOUMINKAN}
    for observed_meld, ids in zip(obs.melds[0], allocated[1:]):
        calculator_melds.append(Meld(
            meld_type=kind[observed_meld.type], tiles=ids,
            opened=observed_meld.type != "ankan", called_tile=ids[0],
            who=0, from_who=observed_meld.from_rel))
    return [tile for group in allocated for tile in group], allocated[0][-1], calculator_melds


def _has_tsumo_yaku(obs: ObservedState, current: list[str],
                    selected: SelectedHistoryV1 | None) -> bool:
    try:
        tiles, win_tile, melds = _calculator_input(obs, current)
    except ValueError:
        return False
    config = HandConfig(
        is_tsumo=True, is_riichi=obs.reach[0],
        is_rinshan=_is_rinshan(obs, selected),
        is_haitei=obs.left_tile_count == 0,
        player_wind=(TILE34_TO_ID[obs.seat_wind_self]
                     if obs.seat_wind_self is not None else None),
        round_wind=(TILE34_TO_ID[obs.bakaze]
                    if obs.bakaze is not None else None),
        options=OptionalRules(has_open_tanyao=True, has_aka_dora=False),
    )
    return HandCalculator.estimate_hand_value(
        tiles, win_tile, melds=melds, config=config).error is None


def _riichi_ankan_allowed(obs: ObservedState, tile: str, current: list[str]) -> bool:
    if obs.drawn_tile is None:
        return False
    if red_to_normal(_canonical(obs.drawn_tile)) != red_to_normal(tile):
        return False
    before = list(current); before.remove(_canonical(obs.drawn_tile))
    after = list(current)
    base = red_to_normal(tile)
    removed = 0
    for candidate in list(after):
        if red_to_normal(candidate) == base and removed < 4:
            after.remove(candidate); removed += 1
    if removed != 4:
        return False
    tile_id = TILE34_TO_ID[base]
    if tile_id >= 27:  # exact non-strict rule parity: honor ankan cannot alter shape
        return True
    existing_sets = _fixed_sets(obs)
    before_waits = set(_waits(_counts(before), existing_sets))
    after_waits = set(_waits(_counts(after), existing_sets + [[tile_id] * 4]))
    return tile_id not in before_waits and before_waits <= after_waits


def analyze_hero_decision(obs: ObservedState,
                          selected_history: SelectedHistoryV1 | None = None) -> DecisionAnalysis:
    call_pending = _hero_call_pending(obs)
    if obs.drawn_tile is None and not call_pending:
        return DecisionAnalysis(None, [_issue("NOT_HERO_DECISION")])
    current = [_canonical(tile) for tile in obs.hero_hand]
    if obs.drawn_tile is not None:
        current.append(_canonical(obs.drawn_tile))
    if obs.reach[0] and obs.drawn_tile is not None:
        discards = [_canonical(obs.drawn_tile)]
    else:
        forbidden = _kuikae_forbidden(obs)
        discards = sorted({tile for tile in current if red_to_normal(tile) not in forbidden},
                          key=lambda tile: (_PAI_RANK.get(tile, 10_000), tile))
    actions = [f"dahai:{tile}" for tile in discards]
    issues = []

    if obs.drawn_tile is not None and _has_tsumo_yaku(obs, current, selected_history):
        actions.append("hora:tsumo")
        issues.append(_issue("AGARI_AVAILABLE"))

    yaochu = {TILE34_TO_ID[red_to_normal(tile)] for tile in current
              if TILE34_TO_ID[red_to_normal(tile)] in TERMINAL_AND_HONOR_INDICES}
    if _first_uninterrupted_draw(obs, selected_history) and len(yaochu) >= 9:
        actions.append("ryukyoku:kyushukyuhai")
        issues.append(_issue("ABORTIVE_DRAW_AVAILABLE"))

    if not call_pending and obs.drawn_tile is not None:
        for tile in discards:
            hand_after = list(current); hand_after.remove(tile)
            may_reach = (not obs.reach[0] and _closed(obs)
                         and (obs.scores is None or obs.scores[0] >= 1000)
                         and (obs.left_tile_count is None or obs.left_tile_count >= 4)
                         and Shanten.calculate_shanten(_counts(hand_after)) == 0)
            if may_reach:
                actions.append(f"reach:{tile}")
        kan_allowed = ((obs.left_tile_count is None or obs.left_tile_count > 0)
                       and obs.n_kans() < 4)
        if kan_allowed:
            for base in sorted({red_to_normal(tile) for tile in current},
                               key=lambda tile: (_PAI_RANK.get(tile, 10_000), tile)):
                if sum(1 for tile in current if red_to_normal(tile) == base) == 4:
                    if not obs.reach[0] or _riichi_ankan_allowed(obs, base, current):
                        actions.append(f"ankan:{base}")
        if kan_allowed and not obs.reach[0]:
            pon_bases = {red_to_normal(meld.tiles[0]) for meld in obs.melds[0]
                         if meld.type == "pon"}
            for tile in discards:
                if red_to_normal(tile) in pon_bases:
                    actions.append(f"kakan:{tile}")
        if obs.sanma and any(red_to_normal(tile) == "N" for tile in current):
            # Under an ACCEPTED riichi the hand is frozen, so only the north you
            # just drew may be pulled — measured 13/13 with zero counterexamples
            # among the observable riichi pulls (verify_sanma_rules V10). Outside
            # riichi, pulling from hand is ordinary (87 observed).
            if not obs.reach[0] or red_to_normal(obs.drawn_tile) == "N":
                actions.append("nukidora:N")

    actions = sorted(set(actions), key=DecisionAnalysis.action_sort_key)
    decision = {"actorRelSeat": 0, "kind": "action",
                "legalDiscards": discards, "legalActions": actions,
                "candidateCount": len(actions)}
    # Deliberately keyed on DISCARDS, not on len(actions): the product question is
    # 何切 ("which tile to cut"), so a lone cuttable tile is no question even when
    # an ankan or a north pull is also on offer. Pinned by test_what_cut_decision.
    if len(discards) < 2:
        issues.append(_issue("NO_MEANINGFUL_CHOICE"))
    return DecisionAnalysis(decision, issues)
