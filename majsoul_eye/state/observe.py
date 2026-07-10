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

    for r in range(1, 4):
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
    if o.scores is not None and o.kyotaku is not None \
            and sum(o.scores) + 1000 * o.kyotaku != 100000:
        v.append(f"scores sum {sum(o.scores)} + 1000*{o.kyotaku} kyotaku != 100000")
    if o.left_tile_count is not None:
        # Conservation: each discard implies one draw except the post-chi/pon
        # forced one; a called-away discard's draw cancels against exactly that
        # exemption; each kan nets -1 (replacement). +-1 absorbs an opponent's
        # in-flight draw and the §1.53 pixel=GT-1 counter timing.
        pred = 70 - sum(len(r) for r in o.rivers) - o.n_kans() \
            - (1 if o.drawn_tile else 0)
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
        o.concealed_counts[r] = None if r == 0 else state.concealed_counts[a]
    if include_hud:
        o.scores = [state.scores[(hero + r) % 4] for r in range(4)]
        o.bakaze, o.kyoku = state.bakaze, state.kyoku
        o.honba, o.kyotaku = state.honba, state.kyotaku
        o.left_tile_count = state.left_tile_count
        if state.oya >= 0 and hero >= 0:
            o.seat_wind_self = WINDS[(hero - state.oya) % 4]
    return o
