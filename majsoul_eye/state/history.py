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
