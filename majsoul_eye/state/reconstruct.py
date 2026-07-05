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
from majsoul_eye.tiles import red_to_normal

WINDS = ["E", "S", "W", "N"]


@dataclass
class ReconstructionResult:
    ok: bool
    events: list = field(default_factory=list)
    reason: str = ""
    fabricated: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


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


def _search(obs: ObservedState, oya_rel: int) -> Optional[list]:
    """Ops: ("draw",rel) ("discard",rel,idx) ("ghost",rel,pai,reach)
    ("call",_Item) ("ankan",_Item) ("kakan",_Item). Canonical branch order:
    visible discard > own-turn kan > ghost/call (calls as late as feasible)."""
    rivers = obs.rivers
    n = [len(r) for r in rivers]
    creation, kakans = _items_for(obs)
    ncre = [len(c) for c in creation]
    failed: set = set()

    # precompute per-seat sideways visible index (None if no riichi shown)
    side_idx = [next((i for i, t in enumerate(rivers[r]) if t.sideways), None)
                for r in range(4)]

    def declared(r, cur, rghost):
        if side_idx[r] is None:
            return bool(rghost >> r & 1)
        return cur[r] > side_idx[r] or bool(rghost >> r & 1)

    def all_done(cur, cidx, kkmask):
        return list(cur) == n and list(cidx) == ncre and kkmask == (1 << len(kakans)) - 1

    def go(cur, cidx, kkmask, rghost, actor):
        key = (cur, cidx, kkmask, rghost, actor)
        if key in failed:
            return None
        if all_done(cur, cidx, kkmask):
            if obs.drawn_tile is not None:
                return [("draw", 0)] if actor == 0 else None
            return []
        rest = decide(cur, cidx, kkmask, rghost, actor, drew=True)
        if rest is not None:
            return [("draw", actor)] + rest
        failed.add(key)
        return None

    def decide(cur, cidx, kkmask, rghost, actor, drew):
        if (drew and actor == 0 and obs.drawn_tile is not None
                and all_done(cur, cidx, kkmask)):
            return []
        # (a) plain visible discard
        if cur[actor] < n[actor]:
            nxt = list(cur)
            nxt[actor] += 1
            rest = go(tuple(nxt), cidx, kkmask, rghost, (actor + 1) % 4)
            if rest is not None:
                return [("discard", actor, cur[actor])] + rest
        # (b) own-turn kans (need a fresh draw; kakan forbidden after riichi)
        if drew:
            if cidx[actor] < ncre[actor] and creation[actor][cidx[actor]].kind == "ankan":
                it = creation[actor][cidx[actor]]
                ncidx = list(cidx)
                ncidx[actor] += 1
                rest = decide(cur, tuple(ncidx), kkmask, rghost, actor, drew=True)
                if rest is not None:
                    return [("ankan", it), ("draw", actor)] + rest
            if not declared(actor, cur, rghost):
                for ki, it in enumerate(kakans):
                    if kkmask >> ki & 1 or it.owner != actor:
                        continue
                    # its pon-part must already be triggered
                    pon_pos = next(j for j, c in enumerate(creation[actor])
                                   if c.mi == it.mi)
                    if cidx[actor] <= pon_pos:
                        continue
                    rest = decide(cur, cidx, kkmask | (1 << ki), rghost, actor, drew=True)
                    if rest is not None:
                        return [("kakan", it), ("draw", actor)] + rest
        # (c) ghost discard + call (a riichi'd owner cannot call)
        for o in range(4):
            if o == actor or cidx[o] >= ncre[o]:
                continue
            it = creation[o][cidx[o]]
            if it.kind not in ("chi", "pon", "daiminkan") or it.target != actor:
                continue
            if declared(o, cur, rghost):
                continue
            ncidx = list(cidx)
            ncidx[o] += 1
            variants = [False]
            if side_idx[actor] is not None and cur[actor] == side_idx[actor] \
                    and not declared(actor, cur, rghost):
                variants.append(True)          # bind the reach to this ghost
            for reach_here in variants:
                nrg = rghost | (1 << actor) if reach_here else rghost
                pre = [("ghost", actor, it.pai, reach_here), ("call", it)]
                if it.kind == "daiminkan":
                    rest = decide(cur, tuple(ncidx), kkmask, nrg, o, drew=True)
                    if rest is not None:
                        return pre + [("draw", o)] + rest
                else:
                    rest = decide(cur, tuple(ncidx), kkmask, nrg, o, drew=False)
                    if rest is not None:
                        return pre + rest
        return None

    return go((0, 0, 0, 0), (0, 0, 0, 0), 0, 0, oya_rel)


# --- emission -----------------------------------------------------------------

def _emit(obs: ObservedState, ops: list, oya_rel: int):
    """ops -> (mjai events after start_kyoku, info dict for backfill)."""
    events: list = []
    haipai = list(obs.hero_hand)
    reach_count = [0] * 4
    declared = [False] * 4
    just_called_hero = False
    side_idx = [next((i for i, t in enumerate(r) if t.sideways), None)
                for r in obs.rivers]
    dora_next = 1                      # markers[0] went into start_kyoku

    def flip_dora():
        nonlocal dora_next
        if dora_next < len(obs.dora_markers):
            events.append({"type": "dora", "dora_marker": obs.dora_markers[dora_next]})
            dora_next += 1

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
                elif nxt[0] == "ghost" and nxt[1] == 0:
                    pai = nxt[2]
                elif nxt[0] == "ankan" and nxt[1].owner == 0:
                    pai = nxt[1].consumed[0]
                elif nxt[0] == "kakan" and nxt[1].owner == 0:
                    pai = nxt[1].pai
            events.append({"type": "tsumo", "actor": 0, "pai": pai})
        elif kind in ("discard", "ghost"):
            r = op[1]
            if kind == "discard":
                idx, pai = op[2], obs.rivers[r][op[2]].pai
                is_reach = (idx == side_idx[r]) and not declared[r]
            else:
                pai = op[2]
                is_reach = op[3]
            if is_reach:
                events.append({"type": "reach", "actor": r})
            if r == 0:
                tsumogiri = not just_called_hero
                if just_called_hero:
                    haipai.append(pai)
                just_called_hero = False
            else:
                tsumogiri = declared[r]        # post-riichi discards are forced tsumogiri
            events.append({"type": "dahai", "actor": r, "pai": pai,
                           "tsumogiri": tsumogiri})
            if is_reach:
                events.append({"type": "reach_accepted", "actor": r})
                declared[r] = True
                reach_count[r] = 1
        elif kind == "call":
            it = op[1]
            events.append({"type": it.kind, "actor": it.owner, "target": it.target,
                           "pai": it.pai, "consumed": list(it.consumed)})
            if it.owner == 0:
                haipai.extend(it.consumed)
                if it.kind in ("chi", "pon"):
                    just_called_hero = True
            if it.kind == "daiminkan":
                flip_dora()
        elif kind == "ankan":
            it = op[1]
            events.append({"type": "ankan", "actor": it.owner,
                           "consumed": list(it.consumed)})
            if it.owner == 0:
                haipai.extend(it.consumed[1:])   # 4th copy was that turn's draw
            flip_dora()
        elif kind == "kakan":
            it = op[1]
            events.append({"type": "kakan", "actor": it.owner, "pai": it.pai,
                           "consumed": list(it.consumed)})
            flip_dora()                          # added tile was the draw: nothing to haipai
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
    viol = list(obs.violations) + check_observed(obs)
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
