"""Reconstruct full 4-player board state from captured MJAI events.

Drives off Akagi's MJAI event stream (the normalized common case) and pulls a
few superset fields (``left_tile_count``) from the raw liqi message. Produces a
:class:`BoardState` per tick — seat-absolute, with everything the labeler needs
to emit boxes/classes: rivers (河, ordered, with tsumogiri / riichi-sideways /
called flags), melds (副露), dora indicators, hero hand, concealed counts (for
opponents' back×N), scores, and round meta.

Validated-on-synthetic-data first; edge cases flagged ``# VALIDATE`` need a real
capture (see docs/DESIGN.md §3.3, §8). Pure / Akagi-free.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from ..tiles import red_to_normal

# Canonical ordering for a tidy hero hand (red five sits just before its 5).
_PAI_ORDER = [
    "1m", "2m", "3m", "4m", "5mr", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5pr", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5sr", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C", "?",
]
_PAI_RANK = {p: i for i, p in enumerate(_PAI_ORDER)}

NUM_PLAYERS = 4


@dataclass
class RiverTile:
    pai: str
    tsumogiri: bool = False          # drawn-and-discarded (摸切) vs from hand (手切)
    riichi: bool = False             # this discard declared riichi → rendered sideways
    called: bool = False             # taken by another player → moved out of the visible 河

    def copy(self) -> "RiverTile":
        return RiverTile(self.pai, self.tsumogiri, self.riichi, self.called)


@dataclass
class Meld:
    type: str                        # chi | pon | daiminkan | ankan | kakan | nukidora
    from_seat: int                   # who the called tile came from (== actor for ankan/kakan/nukidora)
    tiles: list[str] = field(default_factory=list)
    # Geometry needs the exact identity of the specially-rendered tiles, which the
    # sorted `tiles` list loses (matters for chi ordering and red-5 vs normal-5):
    called_pai: str = ""             # the claimed tile — rendered SIDEWAYS in the meld
    added_pai: str = ""              # kakan's added tile — stacked beside the sideways one

    def copy(self) -> "Meld":
        return Meld(self.type, self.from_seat, list(self.tiles), self.called_pai, self.added_pai)


@dataclass
class BoardState:
    hero_seat: int = -1
    # round meta
    bakaze: Optional[str] = None
    kyoku: int = 0
    honba: int = 0
    kyotaku: int = 0
    oya: int = -1
    scores: list[int] = field(default_factory=lambda: [25000] * NUM_PLAYERS)
    dora_markers: list[str] = field(default_factory=list)
    left_tile_count: Optional[int] = None
    # per-player
    rivers: list[list[RiverTile]] = field(default_factory=lambda: [[] for _ in range(NUM_PLAYERS)])
    melds: list[list[Meld]] = field(default_factory=lambda: [[] for _ in range(NUM_PLAYERS)])
    reach: list[bool] = field(default_factory=lambda: [False] * NUM_PLAYERS)
    nukidora: list[int] = field(default_factory=lambda: [0] * NUM_PLAYERS)
    concealed_counts: list[int] = field(default_factory=lambda: [13] * NUM_PLAYERS)
    # hero only (others are face-down)
    hero_hand: list[str] = field(default_factory=list)
    # The tile the hero just drew (tsumo), shown in a SEPARATED slot on screen until
    # the hero acts. hero_hand is kept sorted (the draw is merged), so this is the
    # only record of which of the 14 is the gapped one — the labeler needs it to
    # place the tsumo box. None on every other seat's turn / after the hero acts.
    drawn_tile: Optional[str] = None
    # bookkeeping
    last_actor: int = -1
    last_event: Optional[str] = None
    # The full set of MJAI event types applied by the LATEST record (empty if the
    # latest record applied none). Real records can BUNDLE several events (e.g. a
    # riichi declaration ships as [reach, dahai] in one record) — last_event only
    # keeps the final one, silently losing the earlier ones. last_event_types
    # keeps the whole set so window predicates below can't false-negative on a
    # bundled record. frozenset is immutable, so copy() can share it by reference.
    last_event_types: frozenset = frozenset()
    # liqi op types offered to the hero by the LATEST applied record (None if the
    # latest record carries no offer) — drives button auto-labels; see state/ops.py.
    pending_ops: Optional[list[int]] = None
    in_round: bool = False
    ended: bool = False
    # Seat that owes the MANDATORY immediate discard following a chi/pon/daiminkan/
    # ankan/kakan, until their next dahai clears it (-1 = nobody owes one). Drives
    # is_call_pending — see its docstring.
    awaiting_discard: int = -1

    def copy(self) -> "BoardState":
        s = copy.copy(self)
        s.scores = list(self.scores)
        s.dora_markers = list(self.dora_markers)
        s.rivers = [[t.copy() for t in r] for r in self.rivers]
        s.melds = [[m.copy() for m in ms] for ms in self.melds]
        s.reach = list(self.reach)
        s.nukidora = list(self.nukidora)
        s.concealed_counts = list(self.concealed_counts)
        s.hero_hand = list(self.hero_hand)
        s.pending_ops = list(self.pending_ops) if self.pending_ops else None
        return s

    def visible_river(self, seat: int) -> list[RiverTile]:
        """The 河 as drawn on screen: called-away tiles are excluded."""
        return [t for t in self.rivers[seat] if not t.called]

    def num_melds(self, seat: int) -> int:
        """Meld *sets* (kakan upgrades a pon in place, so it is not double-counted)."""
        return len(self.melds[seat])


class Replayer:
    """Apply MJAI events to evolve a :class:`BoardState`."""

    def __init__(self, hero_seat: int = -1):
        self.state = BoardState(hero_seat=hero_seat)
        self._pending_reach = [False] * NUM_PLAYERS

    # --- driving from a capture --------------------------------------------

    def apply_record(self, record: Any) -> None:
        """Apply one GTRecord: its MJAI events + superset extras (left_tile_count)."""
        # left_tile_count only lives in the raw liqi ActionDealTile.
        raw = getattr(record, "raw_liqi", None)
        if isinstance(raw, dict):
            data = raw.get("data", {})
            inner = data.get("data", {}) if isinstance(data, dict) else {}
            if isinstance(inner, dict):
                # Majsoul uses camelCase 'leftTileCount' (on ActionDealTile).
                ltc = inner.get("leftTileCount", inner.get("left_tile_count"))
                if ltc is not None:
                    self.state.left_tile_count = ltc
        applied_types = set()
        for ev in (getattr(record, "mjai", None) or []):
            self.apply(ev)
            applied_types.add(ev.get("type"))
        # Set pending_ops/last_event_types at the end: start_kyoku replaces self.state
        # with a fresh BoardState, so setting it at the start would be silently
        # dropped. Setting it here (after all MJAI events) ensures it survives the
        # replacement and captures the "latest record" semantics (bundling-proof:
        # a [reach, dahai] record's last_event_types == {"reach", "dahai"}, not
        # just whichever type happened to apply last).
        from majsoul_eye.state.ops import ops_from_record
        self.state.pending_ops = ops_from_record(record)
        self.state.last_event_types = frozenset(applied_types)

    def apply(self, ev: dict) -> None:
        etype = ev.get("type")
        handler = getattr(self, f"_on_{etype}", None)
        if handler is not None:
            handler(ev)
        self.state.last_event = etype

    # --- event handlers -----------------------------------------------------

    def _on_start_game(self, ev: dict) -> None:
        self.state.hero_seat = ev.get("id", self.state.hero_seat)

    def _on_start_kyoku(self, ev: dict) -> None:
        s = self.state
        hero = s.hero_seat
        self.state = BoardState(
            hero_seat=hero,
            bakaze=ev["bakaze"],
            kyoku=ev["kyoku"],
            honba=ev["honba"],
            kyotaku=ev["kyotaku"],
            oya=ev["oya"],
            scores=list(ev["scores"]),
            dora_markers=[ev["dora_marker"]],
            in_round=True,
        )
        self._pending_reach = [False] * NUM_PLAYERS
        tehais = ev.get("tehais") or []
        if 0 <= hero < len(tehais):
            self.state.hero_hand = _sort_hand([t for t in tehais[hero] if t != "?"])

    def _on_tsumo(self, ev: dict) -> None:
        actor, pai = ev["actor"], ev["pai"]
        self.state.concealed_counts[actor] += 1
        if actor == self.state.hero_seat and pai != "?":
            self.state.hero_hand = _sort_hand(self.state.hero_hand + [pai])
            self.state.drawn_tile = pai
        self.state.last_actor = actor

    def _on_dahai(self, ev: dict) -> None:
        actor, pai = ev["actor"], ev["pai"]
        riichi_flag = self._pending_reach[actor]
        if actor == self.state.hero_seat:
            _remove_one(self.state.hero_hand, pai)
            self.state.drawn_tile = None
        self.state.concealed_counts[actor] -= 1
        self.state.rivers[actor].append(
            RiverTile(pai=pai, tsumogiri=bool(ev.get("tsumogiri")), riichi=riichi_flag)
        )
        if riichi_flag:
            self._pending_reach[actor] = False
        if self.state.awaiting_discard == actor:
            self.state.awaiting_discard = -1
        self.state.last_actor = actor

    def _on_reach(self, ev: dict) -> None:
        # Emitted just before the declaring dahai; stick/score change at accept.
        self._pending_reach[ev["actor"]] = True

    def _on_reach_accepted(self, ev: dict) -> None:
        actor = ev["actor"]
        self.state.reach[actor] = True
        self.state.scores[actor] -= 1000
        self.state.kyotaku += 1

    def _on_dora(self, ev: dict) -> None:
        self.state.dora_markers.append(ev["dora_marker"])

    def _on_chi(self, ev: dict) -> None:
        self._open_meld("chi", ev)

    def _on_pon(self, ev: dict) -> None:
        self._open_meld("pon", ev)

    def _on_daiminkan(self, ev: dict) -> None:
        self._open_meld("daiminkan", ev)

    def _open_meld(self, mtype: str, ev: dict) -> None:
        actor, target = ev["actor"], ev["target"]
        consumed = list(ev["consumed"])
        pai = ev["pai"]
        # The called tile leaves the target's visible 河.
        for t in reversed(self.state.rivers[target]):
            if not t.called:
                t.called = True
                break
        self.state.melds[actor].append(
            Meld(type=mtype, from_seat=target, tiles=_sort_hand([pai] + consumed), called_pai=pai)
        )
        self.state.concealed_counts[actor] -= len(consumed)
        if actor == self.state.hero_seat:
            self.state.drawn_tile = None          # chi/pon/daiminkan claim another's discard
            for c in consumed:
                _remove_one(self.state.hero_hand, c)
        self.state.awaiting_discard = actor       # mandatory discard follows (rinshan first for kan)
        self.state.last_actor = actor

    def _on_ankan(self, ev: dict) -> None:
        actor = ev["actor"]
        consumed = list(ev["consumed"])
        self.state.melds[actor].append(Meld(type="ankan", from_seat=actor, tiles=_sort_hand(consumed)))
        self.state.concealed_counts[actor] -= len(consumed)
        if actor == self.state.hero_seat:
            self.state.drawn_tile = None          # declared instead of discarding; rinshan re-sets it
            for c in consumed:
                _remove_one(self.state.hero_hand, c)
        self.state.awaiting_discard = actor       # mandatory discard follows the rinshan draw
        self.state.last_actor = actor

    def _on_kakan(self, ev: dict) -> None:
        actor, pai = ev["actor"], ev["pai"]
        # Upgrade the matching pon in place (no new meld set).
        base = red_to_normal(pai)
        target_meld = None
        for m in self.state.melds[actor]:
            if m.type == "pon" and red_to_normal(m.tiles[0]) == base:
                target_meld = m
                break
        if target_meld is not None:
            target_meld.type = "kakan"
            target_meld.tiles = _sort_hand(target_meld.tiles + [pai])
            target_meld.added_pai = pai
        else:  # VALIDATE: pon not found (e.g. reconnect mid-state) — record anyway
            self.state.melds[actor].append(
                Meld(type="kakan", from_seat=actor, tiles=list(ev.get("consumed", [pai])), added_pai=pai)
            )
        self.state.concealed_counts[actor] -= 1
        if actor == self.state.hero_seat:
            self.state.drawn_tile = None          # declared instead of discarding; rinshan re-sets it
            _remove_one(self.state.hero_hand, pai)
        self.state.awaiting_discard = actor       # mandatory discard follows the rinshan draw
        self.state.last_actor = actor

    def _on_nukidora(self, ev: dict) -> None:  # 3P only
        actor = ev["actor"]
        self.state.nukidora[actor] += 1
        self.state.concealed_counts[actor] -= 1
        if actor == self.state.hero_seat:
            self.state.drawn_tile = None
            _remove_one(self.state.hero_hand, "N")
        self.state.last_actor = actor

    def _on_end_kyoku(self, ev: dict) -> None:
        self.state.in_round = False

    def _on_end_game(self, ev: dict) -> None:
        self.state.ended = True


# --- frame-quality predicates -----------------------------------------------

def is_deal_window(s: BoardState) -> bool:
    """True during the deal-in animation window: a kyoku has started but no discard
    has happened yet (``rivers`` all empty).

    Such frames show the hero hand still dealing/sorting (unsorted tiles, some slots
    still empty), so GT hand boxes are placed at sorted positions that don't match
    the pixels — DROP them from training crops / detector labels (and don't bother
    capturing them). ``rivers``-empty is mode-agnostic (3P/4P, no wall-size magic
    number) and robust to the bridge bundling ``start_kyoku`` + the dealer's first
    ``tsumo`` into ONE record (which makes ``last_event == 'start_kyoku'`` miss the
    deal frame entirely). It flips to False on the very first ``dahai`` of the kyoku.
    """
    return s is not None and s.in_round and sum(len(r) for r in s.rivers) == 0


_CALL_EVENT_TYPES = ("chi", "pon", "daiminkan", "ankan", "kakan", "nukidora")
_REACH_EVENT_TYPES = ("reach", "reach_accepted")


def is_call_window(state) -> bool:
    """A call event (chi/pon/kan/nukidora) has arrived but the forced follow-up
    dahai hasn't: the call animation is mid-flight, so GT leads the pixels
    (river already shrunk, meld already added) — frame-level drop, same policy
    as is_deal_window. Real records can BUNDLE a call with another event in one
    record (e.g. [pon, dora]), which overwrites last_event to "dora" and would
    false-negative — check last_event_types (the full set applied by the latest
    record) first, falling back to last_event for duck-typed states that don't
    carry it. ⚠️ Residual: the falsy check can't tell "field absent" from "latest
    record applied zero events", so the fallback may consult a STALE last_event
    in the latter case — harmless today because every production consumer
    (gtframes.build_seq_state) only snapshots records with non-empty mjai, but a
    future per-tick live consumer must not rely on it. Recovery from multi-shot
    extras is a planned follow-up."""
    types = getattr(state, "last_event_types", None) or frozenset()
    if types & frozenset(_CALL_EVENT_TYPES):
        return True
    return getattr(state, "last_event", None) in _CALL_EVENT_TYPES


def is_score_anim_window(state) -> bool:
    """Riichi declaration/stick animation + score-roll window: HUD numeric
    fields on screen lag/animate right after a reach event, so HUD labels from
    these frames are unreliable (tile labels are unaffected). Real records
    BUNDLE a riichi declaration with its forced dahai ([reach, dahai]) — or
    reach_accepted with the next actor's tsumo — into ONE record, overwriting
    last_event to "dahai"/"tsumo" and making the old last_event-only check
    false-negative on every real reach frame — check last_event_types (the
    full set applied by the latest record) first, falling back to last_event
    for duck-typed states that don't carry it (same stale-fallback residual as
    is_call_window — see its docstring). Spec'd out of recognition scope by the
    user."""
    types = getattr(state, "last_event_types", None) or frozenset()
    if types & frozenset(_REACH_EVENT_TYPES):
        return True
    return getattr(state, "last_event", None) in _REACH_EVENT_TYPES


def is_call_pending(s: BoardState) -> bool:
    """True in the split-second between a chi/pon/daiminkan/ankan/kakan and the
    caller's MANDATORY immediate discard: mahjong rules force a discard right
    after any of these (kan via an interposed rinshan draw), so a frame captured
    in that gap (the ``quiet`` debounce firing before the forced ``dahai`` event
    arrives) shows a meld already updated with no matching discard yet — that
    seat's "float" tile has no on-screen slot the way hero's own ``drawn_tile``
    does, so it is genuinely un-reconstructable from this single frame. Mirrors
    ``is_deal_window``'s role for the very first turn of a kyoku (found via real
    capture: ~2% of oracle-eval frames — 209/10330, of which 11 were outright
    infeasible — see ``scripts/eval/eval_reconstruction.py``).
    """
    return s is not None and s.in_round and s.awaiting_discard >= 0


# --- invariants -------------------------------------------------------------

def check_invariants(s: BoardState) -> list[str]:
    """Return a list of violation messages (empty == consistent).

    Only checks what is *visible* (hero hand + rivers + melds + dora); opponents'
    concealed tiles are unknown. Drop / human-review frames with violations
    (they usually mean an image/GT desync — docs/DESIGN.md §3.4).
    """
    violations: list[str] = []

    # 1) No normalized tile kind appears more than 4 times among visible tiles.
    counts: dict[str, int] = {}
    def bump(pai: str) -> None:
        if pai and pai != "?":
            counts[red_to_normal(pai)] = counts.get(red_to_normal(pai), 0) + 1
    for pai in s.hero_hand:
        bump(pai)
    for seat in range(NUM_PLAYERS):
        # Count the *visible* 河 only: a called tile is physically moved into
        # the caller's meld, so counting it in both river and meld is a double-count.
        for t in s.visible_river(seat):
            bump(t.pai)
        for m in s.melds[seat]:
            for pai in m.tiles:
                bump(pai)
    for d in s.dora_markers:
        bump(d)
    for kind, n in counts.items():
        if n > 4:
            violations.append(f"tile {kind} seen {n}>4 times across visible zones")

    # 2) Hero hand size consistent with melds: hand + 3*melds in {13,14}.
    if s.hero_seat >= 0 and s.hero_hand:
        total = len(s.hero_hand) + 3 * s.num_melds(s.hero_seat)
        if total not in (13, 14):
            violations.append(f"hero hand {len(s.hero_hand)} + 3*{s.num_melds(s.hero_seat)} melds = {total} not in 13/14")

    # 3) Concealed counts non-negative.
    for seat in range(NUM_PLAYERS):
        if s.concealed_counts[seat] < 0:
            violations.append(f"seat {seat} concealed count {s.concealed_counts[seat]} < 0")

    return violations


# --- helpers ----------------------------------------------------------------

def _sort_hand(tiles: list[str]) -> list[str]:
    return sorted(tiles, key=lambda p: _PAI_RANK.get(p, 999))


def _remove_one(hand: list[str], pai: str) -> bool:
    """Remove one instance of ``pai`` from ``hand``; if absent, fall back to the
    plain-five form (Majsoul sometimes reports a meld's red five inexactly)."""
    if pai in hand:
        hand.remove(pai)
        return True
    alt = red_to_normal(pai)
    if alt in hand:
        hand.remove(alt)
        return True
    return False


def replay_capture(records: Iterator[Any]) -> Iterator[tuple[Any, BoardState]]:
    """Yield (record, board_state_snapshot) for each record in a capture."""
    rp = Replayer()
    for rec in records:
        rp.apply_record(rec)
        yield rec, rp.state.copy()
