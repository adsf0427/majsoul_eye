from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from majsoul_eye.what_cut.schema import (
    HistoryBaselineItemV1, SelectedHistoryV1, WhatCutIssueV1,
)


@dataclass(frozen=True)
class UserTsumogiriOverride:
    value: bool
    item_id: str
    field_path: str


@dataclass
class ReconstructionOverrides:
    user_visible: dict[tuple[int, int], UserTsumogiriOverride] = field(default_factory=dict)
    user_ghosts: dict[tuple[int, int], UserTsumogiriOverride] = field(default_factory=dict)
    river_ids: dict[tuple[int, int], str] = field(default_factory=dict)
    ghost_ids: dict[tuple[int, int], str] = field(default_factory=dict)
    ghost_order: list[tuple[int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class DiscardSite:
    op_index: int
    item_kind: Literal["river", "ghost"]
    actor: int
    pai: str
    had_draw: bool
    river_index: int | None
    ghost_key: tuple[int, int] | None
    reach_here: bool
    post_reach: bool


@dataclass(frozen=True)
class HistoryBaseline:
    value: bool
    source: Literal["forced", "inferred"]


def discard_sites(obs, ops: list, pending_reach: int | None = None) -> list[DiscardSite]:
    declared = [False] * 4
    sites = []
    for op_index, op in enumerate(ops):
        kind = op[0]
        if kind == "discard":
            actor, river_index, had_draw = op[1], op[2], op[3]
            sideways = obs.rivers[actor][river_index].sideways
            reach_here = sideways and not declared[actor]
            sites.append(DiscardSite(op_index, "river", actor,
                                     obs.rivers[actor][river_index].pai,
                                     had_draw, river_index, None, reach_here,
                                     declared[actor]))
            if reach_here and actor != pending_reach:
                declared[actor] = True
        elif kind == "ghost":
            actor, pai, reach_here, had_draw, caller, meld_index = op[1:7]
            sites.append(DiscardSite(op_index, "ghost", actor, pai, had_draw,
                                     None, (caller, meld_index), reach_here,
                                     declared[actor]))
            if reach_here and actor != pending_reach:
                declared[actor] = True
    return sites


def baseline_for_site(site: DiscardSite) -> HistoryBaseline:
    if not site.had_draw:
        return HistoryBaseline(False, "forced")
    if site.post_reach:
        return HistoryBaseline(True, "forced")
    return HistoryBaseline(True if site.actor == 0 else False, "inferred")


def derive_history_baseline(obs, ops: list, overrides: ReconstructionOverrides,
                            pending_reach: int | None = None) -> tuple[list[HistoryBaselineItemV1],
                                                                       dict[int, HistoryBaseline]]:
    sites = discard_sites(obs, ops, pending_reach)
    by_op = {site.op_index: baseline_for_site(site) for site in sites}
    by_river = {(site.actor, site.river_index): by_op[site.op_index]
                for site in sites if site.item_kind == "river"}
    by_ghost = {site.ghost_key: by_op[site.op_index]
                for site in sites if site.item_kind == "ghost"}
    ordered = []
    for seat in range(4):
        for index in range(len(obs.rivers[seat])):
            item_id = overrides.river_ids.get((seat, index), f"river:{seat}:{index}")
            baseline = by_river[(seat, index)]
            ordered.append({"itemKind": "river", "itemId": item_id,
                            "baselineValue": baseline.value,
                            "baselineSource": baseline.source})
    for key in overrides.ghost_order:
        baseline = by_ghost[key]
        ordered.append({"itemKind": "ghost", "itemId": overrides.ghost_ids[key],
                        "baselineValue": baseline.value,
                        "baselineSource": baseline.source})
    return ordered, by_op


from collections import Counter

from majsoul_eye.state.replay import _PAI_RANK
from majsoul_eye.tiles import red_to_normal
from majsoul_eye.what_cut.schema import SelectedHistoryV1


@dataclass
class HistoryConflict:
    code: str
    field_path: str | None
    params: dict


@dataclass
class HistorySolution:
    hero_haipai: list[str]
    draw_pai: dict[int, str]
    tsumogiri: dict[int, bool]
    selected_history: SelectedHistoryV1


def _hand_valid(hand: list[str]) -> bool:
    normalized = Counter(red_to_normal(tile) for tile in hand)
    red = Counter(tile for tile in hand if tile.endswith("r"))
    return all(count <= 4 for count in normalized.values()) and all(count <= 1 for count in red.values())


def _tile_order(tile: str) -> tuple[int, str]:
    return (_PAI_RANK.get(tile, 10_000), tile)


def validate_history_semantics(events: list[dict]) -> list[str]:
    last_draw = {}
    reached = [False] * 4
    awaiting_discard = -1
    violations = []
    for index, event in enumerate(events):
        kind = event["type"]
        if kind == "tsumo":
            last_draw[event["actor"]] = event["pai"]
        elif kind == "dahai":
            actor = event["actor"]
            if event["tsumogiri"] and last_draw.get(actor) not in (event["pai"], "?"):
                violations.append(f"event {index}: tsumogiri {event['pai']} != draw {last_draw.get(actor)}")
            if awaiting_discard == actor and event["tsumogiri"]:
                violations.append(f"event {index}: post-call discard cannot be tsumogiri")
            if reached[actor] and not event["tsumogiri"]:
                violations.append(f"event {index}: post-reach discard must be tsumogiri")
            last_draw.pop(actor, None)
            awaiting_discard = -1 if awaiting_discard == actor else awaiting_discard
        elif kind in ("chi", "pon"):
            awaiting_discard = event["actor"]
            last_draw.pop(event["actor"], None)
        elif kind == "reach_accepted":
            reached[event["actor"]] = True
    return violations


def solve_hidden_history(obs, ops: list, overrides: ReconstructionOverrides,
                         oya_rel: int, pending_reach: int | None) -> HistorySolution | HistoryConflict:
    sites = discard_sites(obs, ops, pending_reach)
    baseline_items, baseline_by_op = derive_history_baseline(
        obs, ops, overrides, pending_reach=pending_reach)
    site_by_op = {site.op_index: site for site in sites}
    effective = {op_index: baseline.value for op_index, baseline in baseline_by_op.items()}

    for site in sites:
        if site.item_kind == "river":
            override = overrides.user_visible.get((site.actor, site.river_index))
        else:
            override = overrides.user_ghosts.get(site.ghost_key)
        if override is None:
            continue
        baseline = baseline_by_op[site.op_index]
        if baseline.source == "forced" and override.value != baseline.value:
            return HistoryConflict("TSUMOGIRI_RULE_CONFLICT", override.field_path,
                                   {"required": int(baseline.value)})
        effective[site.op_index] = override.value

    last_draw = [None] * 4
    draw_for_site = {}
    for op_index, op in enumerate(ops):
        if op[0] == "draw":
            last_draw[op[1]] = op_index
        elif op[0] in ("discard", "ghost"):
            site = site_by_op[op_index]
            if site.had_draw:
                draw_for_site[op_index] = last_draw[site.actor]
            last_draw[site.actor] = None
        elif op[0] in ("call", "ankan", "kakan"):
            last_draw[op[1].owner] = None
    site_for_draw = {draw_index: site_by_op[site_index]
                     for site_index, draw_index in draw_for_site.items()}

    final_hand = list(obs.hero_hand)
    if obs.drawn_tile is not None:
        final_hand.append(obs.drawn_tile)
    assignments = {}

    def reverse(index: int, hand: list[str]):
        if not _hand_valid(hand):
            return None
        if index < 0:
            return (sorted(hand, key=_tile_order), dict(assignments)) if len(hand) == 13 else None
        op = ops[index]
        kind = op[0]
        if kind in ("discard", "ghost") and op[1] == 0:
            pai = obs.rivers[0][op[2]].pai if kind == "discard" else op[2]
            return reverse(index - 1, hand + [pai])
        if kind == "call" and op[1].owner == 0:
            return reverse(index - 1, hand + list(op[1].consumed))
        if kind == "ankan" and op[1].owner == 0:
            return reverse(index - 1, hand + list(op[1].consumed))
        if kind == "kakan" and op[1].owner == 0:
            return reverse(index - 1, hand + [op[1].pai])
        if kind == "draw" and op[1] == 0:
            site = site_for_draw.get(index)
            if site is not None and effective[site.op_index]:
                candidates = [site.pai]
            elif index == len(ops) - 1 and obs.drawn_tile is not None:
                candidates = [obs.drawn_tile]
            else:
                candidates = sorted(set(hand), key=_tile_order)
            for pai in candidates:
                if pai not in hand:
                    continue
                before = list(hand)
                before.remove(pai)
                assignments[index] = pai
                solved = reverse(index - 1, before)
                if solved is not None:
                    return solved
                assignments.pop(index, None)
            return None
        return reverse(index - 1, hand)

    solved = reverse(len(ops) - 1, final_hand)
    if solved is None:
        user_paths = [x.field_path for x in overrides.user_visible.values()] + [
            x.field_path for x in overrides.user_ghosts.values()]
        return HistoryConflict("HIDDEN_HISTORY_CONFLICT",
                               user_paths[0] if len(user_paths) == 1 else None,
                               {"overrideCount": len(user_paths)})
    hero_haipai, hero_draws = solved

    draw_pai = {}
    for op_index, op in enumerate(ops):
        if op[0] != "draw":
            continue
        actor = op[1]
        site = site_for_draw.get(op_index)
        if actor == 0:
            draw_pai[op_index] = hero_draws.get(op_index, obs.drawn_tile or "?")
        elif site is not None and effective[site.op_index]:
            draw_pai[op_index] = site.pai
        else:
            draw_pai[op_index] = "?"

    operations = []
    for op_index, op in enumerate(ops):
        if op[0] == "draw":
            operations.append({"kind": "draw", "actorRelSeat": op[1],
                               "targetRelSeat": None, "riverIndex": None,
                               "meldIndex": None, "pai": draw_pai[op_index],
                               "tsumogiri": None, "reach": False})
        elif op[0] in ("discard", "ghost"):
            site = site_by_op[op_index]
            operations.append({"kind": "river" if site.item_kind == "river" else "ghost",
                               "actorRelSeat": site.actor, "targetRelSeat": None,
                               "riverIndex": site.river_index,
                               "meldIndex": site.ghost_key[1] if site.ghost_key else None,
                               "pai": site.pai, "tsumogiri": effective[op_index],
                               "reach": site.reach_here})
        elif op[0] == "call":
            item = op[1]
            operations.append({"kind": "call", "actorRelSeat": item.owner,
                               "targetRelSeat": item.target, "riverIndex": None,
                               "meldIndex": item.mi, "pai": item.pai,
                               "tsumogiri": None, "reach": False})
        elif op[0] in ("ankan", "kakan"):
            item = op[1]
            operations.append({"kind": op[0], "actorRelSeat": item.owner,
                               "targetRelSeat": None, "riverIndex": None,
                               "meldIndex": item.mi, "pai": item.pai or None,
                               "tsumogiri": None, "reach": False})

    selected = {"solverVersion": "hidden-history-v1", "oyaRelSeat": oya_rel,
                "pendingReachRelSeat": pending_reach,
                "heroHaipai": hero_haipai, "operations": operations}
    return HistorySolution(hero_haipai, draw_pai, effective, selected)
