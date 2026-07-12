"""3-player (sanma) Mortal bot for the AI-autoplay capture path (DEV-ONLY).

MahjongCopilot's stock 3P route (``bot.local.engine3p``) expects the standard
Mortal architecture and a ``libriichi3p`` binary in its own package dir —
neither matches the assets that actually exist on this machine. The proven 3P
stack lives in the sibling Akagi repo (``Akagi/mjai_bot/mortal3p/``):

- ``default.pth`` — the "new architecture" sanma checkpoint (622-channel obs,
  compressed 27-tile / 36-action space, DQN head 37);
- its bundled ``.libriichi`` featurizer pyd — obs_shape(4)=(622,34),
  ACTION_SPACE=44, i.e. the 4p-shaped representation that ``local_engine``
  slices down internally (NOT the shinkuan ``libriichi3p`` release, whose
  775-channel obs pairs with the legacy ``mortal.pth`` instead);
- ``local_engine.build_engine`` — loads the checkpoint and applies the
  default.pth-specific rule fixes.

``make_sanma_bot`` grafts that stack onto MahjongCopilot's ``BotMortalLocal``
so one bot instance serves both modes: 4P games keep the stock engine, and a
3P game's ``init_bot(seat, MJ3P)`` gets an Akagi-backed ``mjai.Bot``. Like
``mjcopilot_gt``, this module is MahjongCopilot-agnostic at import time — the
caller passes the classes its own import context resolved; Akagi is imported
lazily inside the factory (its ``mjai_bot`` packages are namespace packages
with no side effects, and its pyd is cp312 — verified importable in the
``auto`` env alongside the site-packages 4p libriichi).
"""
from __future__ import annotations

import os
import sys
from typing import Any

AKAGI_DIR = r"D:/code/phoenix/Akagi"    # default sibling checkout (like autoplay_ai.MJC)


def make_sanma_bot(bot_mortal_local_cls, game_mode_cls, model_files: dict,
                   akagi_dir: str = AKAGI_DIR, model_3p: str = "default.pth") -> Any:
    """Build a ``BotMortalLocal`` that additionally supports ``GameMode.MJ3P``.

    ``model_files`` is the stock 4P mapping (``{GameMode.MJ4P: path}``);
    ``model_3p`` is a checkpoint filename resolved by Akagi's ``build_engine``
    relative to ``Akagi/mjai_bot/mortal3p/``. Raises on any missing 3P asset —
    the caller decides whether to fall back to a plain 4P-only bot.
    """
    akagi_dir = os.path.abspath(akagi_dir)
    if not os.path.isdir(os.path.join(akagi_dir, "mjai_bot", "mortal3p")):
        raise FileNotFoundError(f"Akagi mortal3p stack not found under {akagi_dir}")
    if akagi_dir not in sys.path:
        sys.path.append(akagi_dir)          # append AFTER MJC so MJC wins any name clash

    from mjai_bot.mortal3p.local_engine import build_engine
    from mjai_bot.mortal3p.libriichi import mjai as ak_mjai

    class _SanmaBotMortalLocal(bot_mortal_local_cls):
        """BotMortalLocal + MJ3P via the Akagi mortal3p engine."""

        def _init_bot_impl(self, mode=game_mode_cls.MJ4P):
            if mode == game_mode_cls.MJ3P:
                # Akagi's featurizer Bot, not MJC's `import libriichi3p`
                # (that package dir holds no binary here, and the shinkuan
                # build would be the wrong obs shape for default.pth anyway).
                self.mjai_bot = ak_mjai.Bot(self._get_engine(mode), self.seat)
            else:
                super()._init_bot_impl(mode)

    bot = _SanmaBotMortalLocal(model_files)
    bot._engines[game_mode_cls.MJ3P] = build_engine(model_3p)
    bot._supported_modes = list(bot._engines.keys())
    bot.model_files[game_mode_cls.MJ3P] = os.path.join(
        akagi_dir, "mjai_bot", "mortal3p", model_3p)      # truthful info_str
    return bot
