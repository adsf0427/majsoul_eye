"""ObservedState -> legal hero-perspective MJAI sequence (spec 2026-07-05 §4).

Turn-machine simulation with backtracking DFS over call timing, then a
deterministic emission pass: hero draws are fabricated "all-tsumogiri" (every
hero discard = tsumo X, dahai X tsumogiri) so the fabricated haipai is exactly
hero_hand + meld-consumed + forced post-call tedashi. Opponents draw "?".
Canonical solution: plain discards preferred over calls (= calls as late as
feasible). Pure logic — no vision/Akagi imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from majsoul_eye.state.history import (
    HistoryConflict, HistorySolution, ReconstructionOverrides,
    derive_history_baseline, solve_hidden_history, validate_history_semantics,
)
from majsoul_eye.state.observe import ObservedState, check_observed
from majsoul_eye.tiles import red_to_normal
from majsoul_eye.what_cut.schema import HistoryBaselineItemV1, SelectedHistoryV1, WhatCutIssueV1

WINDS = ["E", "S", "W", "N"]


@dataclass
class ReconstructionResult:
    ok: bool
    events: list = field(default_factory=list)
    reason: str = ""
    fabricated: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    issues: list[WhatCutIssueV1] = field(default_factory=list)
    history_baseline: list[HistoryBaselineItemV1] = field(default_factory=list)
    selected_history: SelectedHistoryV1 | None = None


# --- search (Task 3: rotation only; Task 4 adds calls; Task 5 adds kans/riichi) ---

def _minus(tiles: list, remove: list) -> list:
    """Multiset removal with red-five fallback both ways."""
    out = list(tiles)
    for x in remove:
        if x in out:
            out.remove(x)
            continue
        for t in list(out):
            if red_to_normal(t) == red_to_normal(x):
                out.remove(t)
                break
    return out


@dataclass(frozen=True)
class _Item:
    kind: str        # chi | pon | daiminkan | ankan | kakan
    owner: int
    target: int      # rel seat whose discard is claimed (== owner for ankan/kakan)
    pai: str         # claimed tile | kakan's added tile | "" for ankan
    consumed: tuple  # tiles leaving owner's HAND
    mi: int          # on-screen meld index (owner chronology)


def _items_for(obs: ObservedState):
    """(per-owner creation list in screen order, kakan own-turn parts)."""
    creation: list[list[_Item]] = [[] for _ in range(4)]
    kakans: list[_Item] = []
    for o in range(4):
        for mi, m in enumerate(obs.melds[o]):
            t = (o + m.from_rel) % 4
            if m.type in ("chi", "pon", "daiminkan"):
                creation[o].append(_Item(m.type, o, t, m.called_pai,
                                         tuple(_minus(m.tiles, [m.called_pai])), mi))
            elif m.type == "kakan":
                pon_cons = tuple(_minus(m.tiles, [m.called_pai, m.added_pai]))
                creation[o].append(_Item("pon", o, t, m.called_pai, pon_cons, mi))
                kakans.append(_Item("kakan", o, o, m.added_pai,
                                    tuple(_minus(m.tiles, [m.added_pai])), mi))
            elif m.type == "ankan":
                creation[o].append(_Item("ankan", o, o, "", tuple(m.tiles), mi))
    return creation, kakans


def _hero_call_pending(obs: ObservedState) -> bool:
    """The call -> mandatory-discard gap, HERO side (check_observed's twin):
    hand + 3*melds == 14 with no drawn slot and a trailing hero chi/pon.
    Unlike an opponent's gap (their withheld discard is invisible — gated by
    replay.is_call_pending), hero's side is fully visible: the legal sequence
    simply ENDS at the call, which is a genuine decision point."""
    return (obs.drawn_tile is None and bool(obs.melds[0])
            and obs.melds[0][-1].type in ("chi", "pon")
            and len(obs.hero_hand) + 3 * len(obs.melds[0]) == 14)


def _pending_reach_seats(obs: ObservedState) -> list[int]:
    """Seats whose riichi may still be inside the declaration animation
    (stick/score/kyotaku settle only at reach_accepted): the HUD counter is
    short by exactly one and the seat's declaring tile is still the NEWEST
    discard of its river. Empty list = no pending declaration to model."""
    if obs.kyotaku is None:
        return []
    n_reach = sum(1 for x in obs.reach if x)
    if obs.kyotaku != n_reach - 1:
        return []
    return [r for r in range(4)
            if obs.reach[r] and obs.rivers[r] and obs.rivers[r][-1].sideways]


@dataclass
class SkeletonBudget:
    limit: int = 4096
    yielded: int = 0
    exhausted: bool = False


def _iter_skeletons(obs: ObservedState, oya_rel: int,
                    pending_reach: Optional[int] = None,
                    budget: SkeletonBudget | None = None):
    """Ops: ("draw",rel) ("discard",rel,idx,drew) ("ghost",rel,pai,reach,drew,owner,mi)
    ("call",_Item) ("ankan",_Item) ("kakan",_Item). Canonical branch order:
    visible discard > own-turn kan > ghost/call (calls as late as feasible).
    Yields every complete operation list (not just the first). pending_reach:
    seat whose riichi declaration is still unaccepted — its declaring dahai must
    be the sequence's FINAL event. `active` is recursion-cycle protection only;
    NO prefix-insensitive failed-state memo (two prefixes reaching the same turn
    state can differ in hidden-hand feasibility)."""
    budget = budget or SkeletonBudget()
    rivers = obs.rivers
    counts = [len(river) for river in rivers]
    creation, kakans = _items_for(obs)
    creation_counts = [len(items) for items in creation]
    active = set()
    pending_item = creation[0][-1] if _hero_call_pending(obs) else None
    sideways = [next((i for i, tile in enumerate(river) if tile.sideways), None)
                for river in rivers]
    must_reach = [bool(obs.reach[seat]) and sideways[seat] is None
                  for seat in range(4)]

    def declared(seat, cursors, reach_ghost_mask):
        if sideways[seat] is None:
            return bool(reach_ghost_mask >> seat & 1)
        return cursors[seat] > sideways[seat] or bool(reach_ghost_mask >> seat & 1)

    def all_done(cursors, creation_index, kakan_mask, reach_ghost_mask):
        return (list(cursors) == counts
                and list(creation_index) == creation_counts
                and kakan_mask == (1 << len(kakans)) - 1
                and all(reach_ghost_mask >> seat & 1 for seat in range(4)
                        if must_reach[seat]))

    def go(cursors, creation_index, kakan_mask, reach_ghost_mask, actor, prefix):
        if budget.yielded >= budget.limit:
            budget.exhausted = True
            return
        state = (cursors, creation_index, kakan_mask, reach_ghost_mask, actor)
        if state in active:
            return
        active.add(state)
        if all_done(cursors, creation_index, kakan_mask, reach_ghost_mask):
            tail = [("draw", 0)] if obs.drawn_tile is not None and actor == 0 else []
            if obs.drawn_tile is None or tail:
                budget.yielded += 1
                yield prefix + tail
            active.remove(state)
            return
        yield from decide(cursors, creation_index, kakan_mask, reach_ghost_mask,
                          actor, True, prefix + [("draw", actor)])
        active.remove(state)

    def decide(cursors, creation_index, kakan_mask, reach_ghost_mask,
               actor, drew, prefix):
        if (drew and actor == 0 and obs.drawn_tile is not None
                and all_done(cursors, creation_index, kakan_mask, reach_ghost_mask)):
            if budget.yielded < budget.limit:
                budget.yielded += 1
                yield prefix
            else:
                budget.exhausted = True
            return

        if cursors[actor] < counts[actor]:
            next_cursors = list(cursors)
            next_cursors[actor] += 1
            discard = ("discard", actor, cursors[actor], drew)
            is_pending_declaration = (actor == pending_reach
                                      and cursors[actor] == sideways[actor])
            if is_pending_declaration:
                if (obs.drawn_tile is None and pending_item is None
                        and all_done(tuple(next_cursors), creation_index,
                                     kakan_mask, reach_ghost_mask)
                        and budget.yielded < budget.limit):
                    budget.yielded += 1
                    yield prefix + [discard]
                elif (obs.drawn_tile is None and pending_item is None
                      and all_done(tuple(next_cursors), creation_index,
                                   kakan_mask, reach_ghost_mask)):
                    budget.exhausted = True
            else:
                yield from go(tuple(next_cursors), creation_index, kakan_mask,
                              reach_ghost_mask, (actor + 1) % 4,
                              prefix + [discard])

        if drew:
            if (creation_index[actor] < creation_counts[actor]
                    and creation[actor][creation_index[actor]].kind == "ankan"):
                item = creation[actor][creation_index[actor]]
                next_creation = list(creation_index)
                next_creation[actor] += 1
                yield from decide(cursors, tuple(next_creation), kakan_mask,
                                  reach_ghost_mask, actor, True,
                                  prefix + [("ankan", item), ("draw", actor)])
            if not declared(actor, cursors, reach_ghost_mask):
                for kakan_index, item in enumerate(kakans):
                    if kakan_mask >> kakan_index & 1 or item.owner != actor:
                        continue
                    pon_position = next(index for index, created in enumerate(creation[actor])
                                        if created.mi == item.mi)
                    if creation_index[actor] <= pon_position:
                        continue
                    yield from decide(cursors, creation_index,
                                      kakan_mask | (1 << kakan_index),
                                      reach_ghost_mask, actor, True,
                                      prefix + [("kakan", item), ("draw", actor)])

        for caller in range(4):
            if caller == actor or creation_index[caller] >= creation_counts[caller]:
                continue
            item = creation[caller][creation_index[caller]]
            if item.kind not in ("chi", "pon", "daiminkan") or item.target != actor:
                continue
            if declared(caller, cursors, reach_ghost_mask):
                continue
            next_creation = list(creation_index)
            next_creation[caller] += 1
            reach_variants = [False]
            if not declared(actor, cursors, reach_ghost_mask) and actor != pending_reach:
                if sideways[actor] is not None and cursors[actor] == sideways[actor]:
                    reach_variants.append(True)
                elif must_reach[actor]:
                    reach_variants.append(True)
            for reach_here in reach_variants:
                next_reach_mask = (reach_ghost_mask | (1 << actor)
                                   if reach_here else reach_ghost_mask)
                ghost = ("ghost", actor, item.pai, reach_here, drew,
                         item.owner, item.mi)
                called = [ghost, ("call", item)]
                if item is pending_item:
                    if (all_done(cursors, tuple(next_creation), kakan_mask,
                                 next_reach_mask)
                            and budget.yielded < budget.limit):
                        budget.yielded += 1
                        yield prefix + called
                    elif all_done(cursors, tuple(next_creation), kakan_mask,
                                  next_reach_mask):
                        budget.exhausted = True
                    continue
                if item.kind == "daiminkan":
                    yield from decide(cursors, tuple(next_creation), kakan_mask,
                                      next_reach_mask, caller, True,
                                      prefix + called + [("draw", caller)])
                else:
                    yield from decide(cursors, tuple(next_creation), kakan_mask,
                                      next_reach_mask, caller, False,
                                      prefix + called)

    yield from go((0, 0, 0, 0), (0, 0, 0, 0), 0, 0, oya_rel, [])


# --- emission -----------------------------------------------------------------

def _emit(obs: ObservedState, ops: list, oya_rel: int,
          solution: HistorySolution, pending_reach: Optional[int] = None):
    events: list[dict] = []
    haipai = list(solution.hero_haipai)
    reach_count = [0] * 4
    declared = [False] * 4
    side_idx = [next((i for i, tile in enumerate(river) if tile.sideways), None)
                for river in obs.rivers]
    dora_next = 1

    def flip_dora():
        nonlocal dora_next
        if dora_next < len(obs.dora_markers):
            events.append({"type": "dora",
                           "dora_marker": obs.dora_markers[dora_next]})
            dora_next += 1

    for op_index, op in enumerate(ops):
        kind = op[0]
        if kind == "draw":
            events.append({"type": "tsumo", "actor": op[1],
                           "pai": solution.draw_pai[op_index]})
        elif kind in ("discard", "ghost"):
            actor = op[1]
            if kind == "discard":
                river_index = op[2]
                pai = obs.rivers[actor][river_index].pai
                is_reach = river_index == side_idx[actor] and not declared[actor]
            else:
                pai = op[2]
                is_reach = op[3]
            if is_reach:
                events.append({"type": "reach", "actor": actor})
            events.append({"type": "dahai", "actor": actor, "pai": pai,
                           "tsumogiri": solution.tsumogiri[op_index]})
            if is_reach and actor != pending_reach:
                events.append({"type": "reach_accepted", "actor": actor})
                declared[actor] = True
                reach_count[actor] = 1
        elif kind == "call":
            item = op[1]
            events.append({"type": item.kind, "actor": item.owner,
                           "target": item.target, "pai": item.pai,
                           "consumed": list(item.consumed)})
            if item.kind == "daiminkan":
                flip_dora()
        elif kind == "ankan":
            item = op[1]
            events.append({"type": "ankan", "actor": item.owner,
                           "consumed": list(item.consumed)})
            flip_dora()
        elif kind == "kakan":
            item = op[1]
            events.append({"type": "kakan", "actor": item.owner,
                           "pai": item.pai, "consumed": list(item.consumed)})
            flip_dora()
    return events, {"haipai": haipai, "reach_count": reach_count}


# --- absolute-seat mapping + start_kyoku backfill -------------------------------

def _abs_map(obs: ObservedState, oya_rel: int):
    """(hero_abs, oya_abs, kyoku). Without HUD: hero_abs=0, kyoku=oya_rel+1."""
    if obs.kyoku is not None:
        oya_abs = obs.kyoku - 1
        return (oya_abs - oya_rel) % 4, oya_abs, obs.kyoku
    return 0, oya_rel, oya_rel + 1


def _relabel(events: list, hero_abs: int) -> list:
    out = []
    for ev in events:
        ev = dict(ev)
        for k in ("actor", "target"):
            if k in ev:
                ev[k] = (hero_abs + ev[k]) % 4
        out.append(ev)
    return out


def _reconstruction_failure(code: str, message: str,
                            field_path: str | None = None,
                            **params) -> ReconstructionResult:
    issue = {"code": code, "severity": "blocking", "fieldPath": field_path,
             "evidenceIds": [], "messageKey": f"whatCut.issue.{code}",
             "params": {"message": message, **params}}
    return ReconstructionResult(False, reason=message, issues=[issue])


def reconstruct(obs: ObservedState,
                overrides: ReconstructionOverrides | None = None) -> ReconstructionResult:
    overrides = overrides or ReconstructionOverrides()
    violations = list(obs.violations) + check_observed(obs)
    if violations:
        return _reconstruction_failure("OBSERVED_STATE_INVALID",
                                       "; ".join(violations))
    if obs.seat_wind_self is not None:
        cand = [(4 - WINDS.index(obs.seat_wind_self)) % 4]
    else:
        cand = [0, 1, 2, 3]
    # kyotaku short by one => exactly one riichi is mid-declaration; its seat
    # must be one whose sideways tile is still its newest discard (candidates
    # from _pending_reach_seats; check_observed already rejected other deficits)
    pend_cand = _pending_reach_seats(obs) or [None]
    first_conflict = None
    chosen = ops = solution = pending = None
    feasible = []
    search_exhausted = False
    for oya_rel in cand:
        for pending_candidate in pend_cand:
            budget = SkeletonBudget()
            for candidate_ops in _iter_skeletons(
                    obs, oya_rel, pending_reach=pending_candidate, budget=budget):
                if oya_rel not in feasible:
                    feasible.append(oya_rel)
                if chosen is not None:
                    break  # survey later dealer feasibility without replacing selection
                candidate_solution = solve_hidden_history(
                    obs, candidate_ops, overrides, oya_rel, pending_candidate)
                if isinstance(candidate_solution, HistoryConflict):
                    first_conflict = first_conflict or candidate_solution
                    continue
                chosen, ops, solution, pending = (
                    oya_rel, candidate_ops, candidate_solution, pending_candidate)
                break
            search_exhausted = search_exhausted or budget.exhausted
    if chosen is None:
        conflict = (HistoryConflict("HISTORY_SEARCH_LIMIT", None,
                                    {"skeletonLimit": SkeletonBudget().limit})
                    if search_exhausted else
                    first_conflict or HistoryConflict("NO_LEGAL_TURN_ORDER", None, {}))
        issue = {"code": conflict.code, "severity": "blocking",
                 "fieldPath": conflict.field_path, "evidenceIds": [],
                 "messageKey": f"whatCut.issue.{conflict.code}",
                 "params": conflict.params}
        return ReconstructionResult(False, reason=conflict.code, issues=[issue])
    history_baseline, _ = derive_history_baseline(
        obs, ops, overrides, pending_reach=pending)
    body, info = _emit(obs, ops, chosen, solution, pending_reach=pending)
    if len(info["haipai"]) != 13:
        return _reconstruction_failure(
            "INTERNAL_HAIPAI_INVALID",
            f"internal: fabricated haipai {len(info['haipai'])} != 13",
            actualCount=len(info["haipai"]))
    hero_abs, oya_abs, kyoku = _abs_map(obs, chosen)
    n_reach = sum(info["reach_count"])
    scores_rel = list(obs.scores) if obs.scores is not None else [25000] * 4
    scores_abs = [25000] * 4
    for r in range(4):
        scores_abs[(hero_abs + r) % 4] = scores_rel[r] + 1000 * info["reach_count"][r]
    kyotaku = (obs.kyotaku if obs.kyotaku is not None else n_reach) - n_reach
    tehais: list = [["?"] * 13 for _ in range(4)]
    tehais[hero_abs] = sorted(info["haipai"])
    sk = {"type": "start_kyoku", "bakaze": obs.bakaze or "E", "kyoku": kyoku,
          "honba": obs.honba or 0, "kyotaku": max(0, kyotaku), "oya": oya_abs,
          "dora_marker": obs.dora_markers[0], "scores": scores_abs, "tehais": tehais}
    fabricated = {"haipai": tehais[hero_abs],
                  "defaults": [k for k in ("scores", "bakaze", "kyoku", "honba", "kyotaku")
                               if getattr(obs, k) is None]}
    events = [{"type": "start_game", "id": hero_abs}, sk] + _relabel(body, hero_abs)
    semantic_violations = validate_history_semantics(events)
    if semantic_violations:
        issue = {"code": "INTERNAL_HISTORY_INVALID", "severity": "blocking",
                 "fieldPath": None, "evidenceIds": [],
                 "messageKey": "whatCut.issue.INTERNAL_HISTORY_INVALID",
                 "params": {"message": "; ".join(semantic_violations)}}
        return ReconstructionResult(False, reason=issue["code"], issues=[issue])
    diagnostics = {"feasible_oya_rel": feasible, "oya_rel": chosen}
    if _hero_call_pending(obs):
        diagnostics["hero_call_pending"] = True
    if pending is not None:
        diagnostics["pending_reach_seat"] = pending
    return ReconstructionResult(
        True, events=events, fabricated=fabricated, diagnostics=diagnostics,
        history_baseline=history_baseline,
        selected_history=solution.selected_history,
    )
