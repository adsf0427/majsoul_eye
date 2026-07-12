"""One-shot verification tool: the sanma (3P) bot stack behind autoplay_ai.

NOT a pipeline stage — run it whenever the 3P engine assets change (Akagi
checkout, default.pth, MJC upgrade) to confirm AI-autoplay's sanma
prerequisites still hold, without launching a browser:

1. builds the exact bot autoplay_ai builds (`capture.bot3p.make_sanma_bot`:
   MJC BotMortalLocal + the Akagi mortal3p engine), 4P engine included, so
   both rust featurizer pyds coexist in one process like a real run;
2. MJC-interface smoke: `init_bot(seat, mode)` for MJ3P seats 0-2 and MJ4P,
   first-reaction sanity through MJC's `Bot.react` glue;
3. full-log replay: every event of a real sanma mjai log through the raw
   3P bot from each seat — asserts legal reaction types throughout (expect
   dahai/nukidora/reach/pon/... and no exceptions).

Run (auto env):  PYTHONPATH=. python scripts/capture/verify_bot3p.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from majsoul_eye.capture.bot3p import make_sanma_bot, AKAGI_DIR  # noqa: E402

MJC_DEFAULT = r"D:/code/phoenix/MahjongCopilot"
SMOKE_LOG = r"D:/code/phoenix/mortal3p/log-viewer/game_records/smoke.json"

VALID_REACTIONS = {"dahai", "reach", "pon", "chi", "kan", "daiminkan", "ankan", "kakan",
                   "nukidora", "hora", "ryukyoku", "none"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mjc", default=MJC_DEFAULT)
    ap.add_argument("--akagi", default=AKAGI_DIR)
    ap.add_argument("--model", default="v4_js_09260526.pth", help="4P model under MJC/models/.")
    ap.add_argument("--model-3p", default="default.pth", help="3P model under Akagi/mjai_bot/mortal3p/.")
    ap.add_argument("--log", default=SMOKE_LOG, help="Sanma mjai log (JSONL) to replay.")
    args = ap.parse_args()

    # mirror autoplay_ai's env setup: chdir into MJC, drop CWD from sys.path
    mjc = os.path.abspath(args.mjc)
    os.chdir(mjc)
    sys.path[:] = [p for p in sys.path if p not in ("", ".")]
    sys.path.append(mjc)
    import libriichi  # noqa: F401  (sanity; site-packages cp312 4P build)
    assert "site-packages" in (libriichi.__file__ or ""), libriichi.__file__
    from common.utils import GameMode
    from bot.local.bot_local import BotMortalLocal

    model_path = os.path.join(mjc, "models", args.model)
    bot = make_sanma_bot(BotMortalLocal, GameMode, {GameMode.MJ4P: model_path},
                         akagi_dir=args.akagi, model_3p=args.model_3p)
    print("bot:", bot.info_str)
    assert GameMode.MJ3P in bot.supported_modes and GameMode.MJ4P in bot.supported_modes, \
        bot.supported_modes

    events = [json.loads(l) for l in Path(args.log).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"log: {args.log} ({len(events)} events)")

    # --- MJC-interface smoke: init_bot + first reaction through Bot.react glue ---
    for seat in (0, 1, 2):
        bot.init_bot(seat, GameMode.MJ3P)               # the call that used to raise BotNotSupportingMode
        first = None
        for ev in events:
            first = bot.react(dict(ev))
            if first is not None:
                break
        assert isinstance(first, dict) and first["type"] in VALID_REACTIONS, first
        print(f"  init_bot(seat={seat}, MJ3P) OK; first reaction: {first['type']}")
    bot.init_bot(0, GameMode.MJ4P)                      # 4P path still intact
    print("  init_bot(seat=0, MJ4P) OK")

    # --- full-log replay per seat (raw 3P bot; MJC's reach glue would commit
    #     unplayed reach declarations and desync a fixed log) ---
    for seat in (0, 1, 2):
        bot.init_bot(seat, GameMode.MJ3P)
        kinds: Counter = Counter()
        for i, ev in enumerate(events):
            try:
                r = bot.mjai_bot.react(json.dumps(ev))
            except Exception:
                print(f"  seat {seat}: FAILED at event {i}: {ev}")
                raise
            if r is not None:
                rd = json.loads(r)
                assert rd["type"] in VALID_REACTIONS, rd
                kinds[rd["type"]] += 1
        print(f"  seat {seat}: {sum(kinds.values())} reactions  {dict(kinds)}")
        assert kinds.get("dahai", 0) > 0, "no dahai reactions - engine not reacting"

    print("PASS: sanma bot stack verified (engine + MJC glue + full-log replay)")


if __name__ == "__main__":
    main()
