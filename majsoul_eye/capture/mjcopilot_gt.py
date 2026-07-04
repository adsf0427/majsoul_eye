"""Shared MJAI-extraction for the MahjongCopilot GT paths (DEV-ONLY).

MahjongCopilot's ``GameState.input(msg)`` derives MJAI events into
``self.mjai_pending_input_msgs`` and batches them to the bot. To record those
events we wrap GameState so every appended event is ``deepcopy``'d and queued —
GameState MUTATES the AI hand list in place as the game proceeds, so an
un-copied reference to ``start_kyoku.tehais`` gets overwritten later.

``make_capturing_game_state`` is MahjongCopilot-agnostic at import time: the
caller passes in whichever ``GameState`` class its import context resolved, so
this module never imports MahjongCopilot itself. Used by both the live capture
(``scripts/capture/autoplay_ai``) and the offline converter/migration
(``scripts/data/convert_mjcopilot``) — one derivation, no drift.
"""
from __future__ import annotations

import copy
from typing import Any, Callable


def make_capturing_game_state(game_state_cls, bot) -> tuple[Any, Callable[[], list]]:
    """Return ``(gs, drain_mjai)``.

    ``gs`` is a ``game_state_cls`` instance (drop-in for the real GameState);
    ``drain_mjai()`` returns the deep-copied MJAI events derived since the
    previous call, in append order.
    """
    events: list = []
    read = [0]

    class _CapList(list):
        def append(self, x):
            events.append(copy.deepcopy(x))
            list.append(self, x)

        def extend(self, xs):
            for x in xs:
                self.append(x)

    class _Traced(game_state_cls):
        def __setattr__(self, k, v):
            # Always install an EMPTY CapList for the pending list (ignore v):
            # GameState resets it to [] then populates via append/extend, so an
            # empty tracked list captures every subsequent event. Matches the
            # original convert_mjcopilot TracedGameState behavior verbatim.
            object.__setattr__(self, k, _CapList() if k == "mjai_pending_input_msgs" else v)

    gs = _Traced(bot)

    def drain_mjai() -> list:
        new = events[read[0]:]
        read[0] = len(events)
        return new

    return gs, drain_mjai


def gt_fields(msg) -> tuple:
    """``(method, action_name)`` for a parsed liqi message, matching GTRecord.

    ``action_name`` is the ActionPrototype's inner ``data.name`` (e.g.
    ``ActionDiscardTile``) and is None for any other method.
    """
    if not isinstance(msg, dict):
        return None, None
    method = msg.get("method")
    data = msg.get("data")
    action_name = data.get("name") if method == ".lq.ActionPrototype" and isinstance(data, dict) else None
    return method, action_name
