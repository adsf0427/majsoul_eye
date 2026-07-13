"""Single-frame OBSERVED board state — the vision-side mirror of replay.BoardState.

ObservedState is what one screenshot shows a human (spec 2026-07-05 §3.1):
tiles zones are recognizable today; 2D-HUD fields are Optional slots filled by
the HUD micro-readers when available. Seats are SCREEN-RELATIVE
(0=self 1=right 2=across 3=left, counter-clockwise = turn order).
hero_hand EXCLUDES drawn_tile (the separated tsumo slot is its own field).
Pure data + checks; no cv2/numpy/Akagi imports at module level.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from majsoul_eye.tiles import TILE_NAMES, red_to_normal

_VALID = set(TILE_NAMES)

# Tiles that do not exist in a sanma wall. 5mr normalizes to 5m, so this set
# catches the red five too. (Measured: 0 occurrences in 6,782 sanma discards —
# scripts/eval/verify_sanma_rules.py V4.)
_SANMA_ABSENT = {f"{rank}m" for rank in range(2, 9)}

# Live wall at the deal, per mode. 4P: 136 - 14 dead - 4*13 = 70.
# Sanma: 108 - 14 dead - 3*13 = 55. Both MEASURED, not derived — see V1.
LIVE_WALL = {False: 70, True: 55}


@dataclass
class ObservedRiverTile:
    pai: str
    sideways: bool = False           # rendered sideways (riichi declaration slot)


@dataclass
class ObservedMeld:
    type: str                        # chi | pon | daiminkan | ankan | kakan
    tiles: list[str] = field(default_factory=list)   # full composition, reds exact
    called_pai: str = ""             # sideways tile ("" for ankan)
    added_pai: str = ""              # kakan's stacked tile
    from_rel: int = 0                # (target - owner) % 4: 1=shimocha 2=toimen 3=kamicha; 0=ankan


@dataclass
class ObservedState:
    hero_hand: list[str] = field(default_factory=list)
    drawn_tile: Optional[str] = None
    rivers: list[list[ObservedRiverTile]] = field(default_factory=lambda: [[] for _ in range(4)])
    melds: list[list[ObservedMeld]] = field(default_factory=lambda: [[] for _ in range(4)])
    dora_markers: list[str] = field(default_factory=list)
    concealed_counts: list[Optional[int]] = field(default_factory=lambda: [None] * 4)
    reach: list[bool] = field(default_factory=lambda: [False] * 4)
    # --- 2D HUD slots (None until the HUD readers land; relative seat order) ---
    scores: Optional[list[int]] = None
    bakaze: Optional[str] = None
    kyoku: Optional[int] = None
    honba: Optional[int] = None
    kyotaku: Optional[int] = None
    left_tile_count: Optional[int] = None
    seat_wind_self: Optional[str] = None
    pending_buttons: Optional[list[str]] = None
    # --- 3-player (sanma) ---
    # Sanma keeps the 4P screen ring; absolute chair 3 is a phantom that renders
    # EMPTY all game. Which SCREEN slot that is depends on where the hero sits:
    # phantom_rel = (3 - hero_abs) % 4 — so it ROTATES, and rivers[3] is a real
    # opponent two thirds of the time. Never key off index 3; key off phantom_rel.
    sanma: bool = False
    phantom_rel: Optional[int] = None
    # Nukidora (拔北) is a per-seat COUNT, not a meld: the pulled north sits in its
    # own on-table pile. Modelled exactly as replay.BoardState.nukidora, which is
    # why ObservedMeld.type needs no new arm.
    nukidora: list[int] = field(default_factory=lambda: [0] * 4)
    # --- meta ---
    violations: list[str] = field(default_factory=list)
    zone_confidence: dict[str, float] = field(default_factory=dict)

    def n_kans(self) -> int:
        return sum(1 for ms in self.melds for m in ms
                   if m.type in ("daiminkan", "ankan", "kakan"))


def check_observed(o: ObservedState) -> list[str]:
    """Single-frame consistency checks (replay.check_invariants' vision twin)."""
    v: list[str] = []
    counts: dict[str, int] = {}

    def bump(pai: str) -> None:
        if pai and pai != "back":
            if pai not in _VALID:
                v.append(f"unknown tile name {pai!r}")
                return
            k = red_to_normal(pai)
            if o.sanma and k in _SANMA_ABSENT:
                v.append(f"tile {pai!r} does not exist in a sanma wall (no 2m-8m)")
                return
            counts[k] = counts.get(k, 0) + 1

    for p in o.hero_hand:
        bump(p)
    if o.drawn_tile:
        bump(o.drawn_tile)
    for r in range(4):
        for t in o.rivers[r]:
            bump(t.pai)
        for m in o.melds[r]:
            for p in m.tiles:
                bump(p)
    for d in o.dora_markers:
        bump(d)
    # The nukidora piles are face-up N tiles ON THE TABLE, so they spend from the
    # same 4-copies-per-kind budget as everything else (MEASURED: V8). This is
    # load-bearing for the hidden-history solve, not bookkeeping: history's
    # _hand_valid bounds only the HAND, and the backward solve transiently puts
    # the hero's pulled norths back INTO the hand. Without this check a 5-north
    # board reaches the solver and dies as an inscrutable HIDDEN_HISTORY_CONFLICT
    # instead of the truthful "N seen 5>4 times". The two checks are coupled.
    for r in range(4):
        for _ in range(o.nukidora[r]):
            bump("N")
    for kind, n in counts.items():
        if n > 4:
            v.append(f"tile {kind} seen {n}>4 times across visible zones")

    n_melds = len(o.melds[0])
    total = len(o.hero_hand) + 3 * n_melds
    # total == 14 with no drawn slot and a trailing hero chi/pon = the frame
    # sits in the call -> mandatory-discard gap. Hero's side of that gap is
    # fully visible (unlike an opponent's), so it is a VALID single-frame
    # state — reconstruct ends the sequence at the call (hero_call_pending).
    call_pending = (total == 14 and o.drawn_tile is None and n_melds > 0
                    and o.melds[0][-1].type in ("chi", "pon"))
    if total != 13 and not call_pending:
        v.append(f"hero hand {len(o.hero_hand)} + 3*{n_melds} melds != 13")

    if not o.dora_markers:
        v.append("no dora marker visible")
    elif len(o.dora_markers) - 1 > o.n_kans():
        v.append(f"{len(o.dora_markers)} dora markers but only {o.n_kans()} kans")

    # Hand size 13 - 3*melds is MODE-INVARIANT: a nukidora removes the north AND
    # draws a replacement, so it does not shrink the hand (V3). Do not subtract
    # nukidora[r] here.
    for r in range(1, 4):
        if r == o.phantom_rel:
            continue                           # covered by the phantom block below
        c = o.concealed_counts[r]
        if c is None:
            continue
        expect = 13 - 3 * len(o.melds[r])
        if c not in (expect, expect + 1):      # +1: mid-draw, or just chi/pon'd and
                                                # hasn't discarded yet (state.is_call_pending)
            v.append(f"seat {r} concealed {c} != {expect}(+1) for {len(o.melds[r])} melds")

    for r in range(4):
        for m in o.melds[r]:
            if m.type in ("chi", "pon", "daiminkan") and m.from_rel not in (1, 2, 3):
                v.append(f"seat {r} {m.type} from_rel {m.from_rel} invalid")
            if m.type == "chi" and m.from_rel != 3:
                v.append(f"seat {r} chi from_rel {m.from_rel} != 3 (kamicha only)")
            if o.sanma and m.type == "chi":
                v.append(f"seat {r} chi — sanma has no chi")
            if (o.sanma and o.phantom_rel is not None
                    and m.type in ("chi", "pon", "daiminkan")
                    and (r + m.from_rel) % 4 == o.phantom_rel):
                v.append(f"seat {r} {m.type} called from the phantom chair "
                         f"(from_rel {m.from_rel})")

    # --- sanma structure ------------------------------------------------------
    if o.sanma:
        if o.phantom_rel is None:
            v.append("sanma board with no phantom seat identified")
        elif o.phantom_rel not in (1, 2, 3):
            # The hero is never the phantom (V7), so rel 0 is never empty.
            v.append(f"phantom_rel {o.phantom_rel} invalid (the hero is never the phantom)")
        else:
            p = o.phantom_rel
            if o.rivers[p] or o.melds[p] or o.nukidora[p] or o.reach[p]:
                v.append(f"phantom chair (rel {p}) is not empty")
            if o.concealed_counts[p] not in (None, 0):
                v.append(f"phantom chair (rel {p}) holds {o.concealed_counts[p]} concealed tiles")
            if o.scores is not None and o.scores[p] not in (None, 0):
                v.append(f"phantom chair (rel {p}) has a score of {o.scores[p]}")
        if o.seat_wind_self is not None and o.seat_wind_self not in ("E", "S", "W"):
            v.append(f"seat wind {o.seat_wind_self!r} impossible in sanma (3 seats)")
        if o.kyoku is not None and o.kyoku not in (1, 2, 3):
            v.append(f"kyoku {o.kyoku} impossible in sanma (3 dealers)")
    for r in range(4):
        if o.nukidora[r] and not o.sanma:
            v.append(f"seat {r} has {o.nukidora[r]} nukidora in a 4-player game")
        if o.nukidora[r] > 4:
            v.append(f"seat {r} nukidora {o.nukidora[r]} > 4 (only 4 norths exist)")

    # --- HUD x vision cross-checks (fields are None unless a HUD reader ran) --
    n_reach = sum(1 for x in o.reach if x)
    if o.kyotaku is not None and o.kyotaku < n_reach:
        # Every ACCEPTED riichi put a stick on the table, but stick/score/
        # counter all settle only at reach_accepted. A deficit of exactly one
        # whose declaring tile is still the NEWEST discard of its river is the
        # declaration window — a legal decision point (others may still ron/
        # call); reconstruct must end the sequence at that dahai
        # (pending_reach). Any other deficit is detector noise.
        pending_ok = (o.kyotaku == n_reach - 1
                      and any(o.reach[r] and o.rivers[r]
                              and o.rivers[r][-1].sideways for r in range(4)))
        if not pending_ok:
            v.append(f"kyotaku {o.kyotaku} < visible riichi count {n_reach}")
    # Points conservation: 4P is 25000x4, sanma is 35000x3 (V6).
    total_points = 105000 if o.sanma else 100000
    if o.scores is not None and o.kyotaku is not None \
            and sum(o.scores) + 1000 * o.kyotaku != total_points:
        v.append(f"scores sum {sum(o.scores)} + 1000*{o.kyotaku} kyotaku != {total_points}")
    if o.left_tile_count is not None:
        # Conservation: each discard implies one draw except the post-chi/pon
        # forced one; a called-away discard's draw cancels against exactly that
        # exemption; each kan nets -1 (replacement). A nukidora ALSO nets -1: it
        # takes a dead-wall replacement exactly like a kan (MEASURED: V1 — the
        # identity lands in {54,55} with this term and scatters over {50..55}
        # without it). It does NOT, however, flip a dora (V2) — so the term
        # belongs here and NOT in the dora budget above. +-1 absorbs an
        # opponent's in-flight draw and the §1.53 pixel=GT-1 counter timing.
        pred = LIVE_WALL[o.sanma] - sum(len(r) for r in o.rivers) - o.n_kans() \
            - sum(o.nukidora) - (1 if o.drawn_tile else 0)
        if abs(pred - o.left_tile_count) > 1:
            v.append(f"wall count {o.left_tile_count} vs predicted {pred} (>1 off)")

    # Riichi requires a closed hand (ankan is the only meld type allowed under
    # riichi). A reach flag on a seat holding open melds (chi/pon/daiminkan/
    # kakan) is detector noise (phantom reach stick / misread sideways tile),
    # not a legal state — reject rather than emit illegal mjai. Not HUD-gated:
    # fires whenever reach[r] is True regardless of source (sideways or stick).
    for r in range(4):
        if o.reach[r] and any(m.type != "ankan" for m in o.melds[r]):
            v.append(f"seat {r} riichi with open melds")
    return v


WINDS = ["E", "S", "W", "N"]


def observed_from_board(state, include_hud: bool = True) -> ObservedState:
    """Project a replayed BoardState to what the SCREEN shows (eval oracle).

    Relative seats: rel r == abs (hero+r)%4. reach[] = VISIBLE sideways tile OR
    state.reach — a riichi whose declaration tile was called away with no later
    discard has no sideways tile in a single frame, but its on-table stick is
    still visible, so state.reach[a] (accepted riichi) also counts (matches
    what the vision side sees via the detected reach stick; spec 2026-07-09 §4).
    Lazy-imports annotate.pipeline for river_sideways_index (keeps this module
    cv2-free unless projecting).
    """
    from majsoul_eye.annotate.pipeline import river_sideways_index

    hero = state.hero_seat
    o = ObservedState()
    o.sanma = bool(getattr(state, "sanma", False))
    if o.sanma and hero >= 0:
        # Absolute chair 3 is the phantom; where it lands on screen depends on
        # where the hero sits. Derived, not assumed to be rel 3.
        o.phantom_rel = (3 - hero) % 4
    o.drawn_tile = state.drawn_tile
    hand = list(state.hero_hand)
    if state.drawn_tile:
        if state.drawn_tile in hand:
            hand.remove(state.drawn_tile)
        elif red_to_normal(state.drawn_tile) in hand:
            hand.remove(red_to_normal(state.drawn_tile))
    o.hero_hand = hand
    o.dora_markers = list(state.dora_markers)
    for r in range(4):
        a = (hero + r) % 4
        vis = state.visible_river(a)
        side = river_sideways_index(
            [{"riichi": t.riichi, "called": t.called} for t in state.rivers[a]])
        o.rivers[r] = [ObservedRiverTile(t.pai, sideways=(i == side))
                       for i, t in enumerate(vis)]
        o.melds[r] = [ObservedMeld(m.type, list(m.tiles), m.called_pai, m.added_pai,
                                   from_rel=((m.from_seat - a) % 4))
                      for m in state.melds[a] if m.type != "nukidora"]
        # sideways tile OR the on-table stick: an accepted riichi whose
        # declaration tile was called away is still visible via its stick.
        o.reach[r] = any(t.sideways for t in o.rivers[r]) or state.reach[a]
        o.nukidora[r] = getattr(state, "nukidora", [0] * 4)[a]
        if r == o.phantom_rel:
            # The phantom never draws, so BoardState.concealed_counts[3] sits at
            # its untouched 13. On screen that chair shows nothing at all.
            o.concealed_counts[r] = 0
        else:
            o.concealed_counts[r] = None if r == 0 else state.concealed_counts[a]
    if include_hud:
        o.scores = [state.scores[(hero + r) % 4] for r in range(4)]
        o.bakaze, o.kyoku = state.bakaze, state.kyoku
        o.honba, o.kyotaku = state.honba, state.kyotaku
        o.left_tile_count = state.left_tile_count
        if state.oya >= 0 and hero >= 0:
            # Seat winds rotate mod the number of SEATS, not mod 4. Reading this
            # %4 in sanma yields "N" — which is not a seat wind at all, and would
            # silently poison every machine-generated 3P golden.
            o.seat_wind_self = ("ESW"[(hero - state.oya) % 3] if o.sanma
                                else WINDS[(hero - state.oya) % 4])
    return o
