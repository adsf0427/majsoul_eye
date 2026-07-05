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

from majsoul_eye.state.observe import ObservedState, check_observed

WINDS = ["E", "S", "W", "N"]


@dataclass
class ReconstructionResult:
    ok: bool
    events: list = field(default_factory=list)
    reason: str = ""
    fabricated: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


# --- search (Task 3: rotation only; Task 4 adds calls; Task 5 adds kans/riichi) ---

def _search(obs: ObservedState, oya_rel: int) -> Optional[list]:
    """Return op list or None. Ops:
    ("draw", rel) | ("discard", rel, idx) — extended by later tasks."""
    rivers = obs.rivers
    n = [len(r) for r in rivers]

    def go(cursors: tuple, actor: int) -> Optional[list]:
        if list(cursors) == n:
            if obs.drawn_tile is not None:
                return [("draw", 0)] if actor == 0 else None
            return []
        if cursors[actor] < n[actor]:
            nxt = list(cursors)
            nxt[actor] += 1
            rest = go(tuple(nxt), (actor + 1) % 4)
            if rest is not None:
                return [("draw", actor), ("discard", actor, cursors[actor])] + rest
        return None

    return go((0, 0, 0, 0), oya_rel)


# --- emission -----------------------------------------------------------------

def _emit(obs: ObservedState, ops: list, oya_rel: int):
    """ops -> (mjai events after start_kyoku, info dict for backfill)."""
    events: list = []
    haipai = list(obs.hero_hand)
    reach_count = [0] * 4
    for i, op in enumerate(ops):
        kind = op[0]
        if kind == "draw":
            r = op[1]
            if r != 0:
                events.append({"type": "tsumo", "actor": r, "pai": "?"})
                continue
            pai = obs.drawn_tile
            if i + 1 < len(ops):
                nxt = ops[i + 1]
                if nxt[0] == "discard":
                    pai = obs.rivers[0][nxt[2]].pai
            events.append({"type": "tsumo", "actor": 0, "pai": pai})
        elif kind == "discard":
            r, idx = op[1], op[2]
            pai = obs.rivers[r][idx].pai
            events.append({"type": "dahai", "actor": r, "pai": pai,
                           "tsumogiri": r == 0})
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


def reconstruct(obs: ObservedState) -> ReconstructionResult:
    viol = check_observed(obs)
    if viol:
        return ReconstructionResult(False, reason="; ".join(viol))
    if obs.seat_wind_self is not None:
        cand = [(4 - WINDS.index(obs.seat_wind_self)) % 4]
    else:
        cand = [0, 1, 2, 3]
    feasible, chosen, ops = [], None, None
    for oya_rel in cand:
        got = _search(obs, oya_rel)
        if got is not None:
            feasible.append(oya_rel)
            if chosen is None:
                chosen, ops = oya_rel, got
    if chosen is None:
        return ReconstructionResult(
            False, reason=f"no legal turn order for any oya in {cand}",
            diagnostics={"feasible_oya_rel": []})
    body, info = _emit(obs, ops, chosen)
    if len(info["haipai"]) != 13:
        return ReconstructionResult(
            False, reason=f"internal: fabricated haipai {len(info['haipai'])} != 13")
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
    events = [{"type": "start_game", "id": hero_abs}, sk] + _relabel(body, hero_abs)
    fabricated = {"haipai": tehais[hero_abs],
                  "defaults": [k for k in ("scores", "bakaze", "kyoku", "honba", "kyotaku")
                               if getattr(obs, k) is None]}
    return ReconstructionResult(True, events=events, fabricated=fabricated,
                                diagnostics={"feasible_oya_rel": feasible,
                                             "oya_rel": chosen})
