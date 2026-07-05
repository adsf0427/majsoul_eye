"""Single-frame OBSERVED board state — the vision-side mirror of replay.BoardState.

ObservedState is what one screenshot shows a human (spec 2026-07-05 §3.1):
tiles zones are recognizable today; 2D-HUD fields are Optional slots filled once
the HUD micro-readers (spec 2026-07-04) land. Seats are SCREEN-RELATIVE
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
    if len(o.hero_hand) + 3 * n_melds != 13:
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
        if c not in (expect, expect + 1):      # +1: that seat may be mid-draw
            v.append(f"seat {r} concealed {c} != {expect}(+1) for {len(o.melds[r])} melds")

    for r in range(4):
        for m in o.melds[r]:
            if m.type in ("chi", "pon", "daiminkan") and m.from_rel not in (1, 2, 3):
                v.append(f"seat {r} {m.type} from_rel {m.from_rel} invalid")
            if m.type == "chi" and m.from_rel != 3:
                v.append(f"seat {r} chi from_rel {m.from_rel} != 3 (kamicha only)")
    return v
