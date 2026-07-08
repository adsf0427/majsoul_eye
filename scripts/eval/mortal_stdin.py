"""mjai stdin/stdout wrapper around ../auto/mycv's Mortal — QA helper for
``eval_reconstruction.py --level engine`` (PIPELINE.md §4 one-shot tool; NOT a
pipeline stage and NOT part of the shipped recognizer, hence free to reach
into the sibling repo).

Reads mjai events as JSON lines on stdin, prints each reaction as a JSON line
to stdout (the LAST line is the reaction to the final event — exactly what
``ask_engine`` consumes). Model/构造 copied from mycv's qidong.py (version=4,
b24c512, ``mortal.pth`` at the mycv root, cpu).

Usage:
  python scripts/eval/mortal_stdin.py <player_id 0-3>
  PYTHONPATH=. python scripts/eval/eval_reconstruction.py --captures ... \
      --level engine --engine-cmd "python scripts/eval/mortal_stdin.py {seat}"
"""
import os
import sys

MYCV = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    os.pardir, os.pardir, os.pardir,
                                    "auto", "mycv"))


def main():
    player_id = int(sys.argv[1])
    sys.path.insert(0, MYCV)
    import torch
    from mortal.model import Brain, DQN
    from mortal.engine import MortalEngine
    try:
        from mortal.libriichi.mjai import Bot   # mycv's own pyd (what the bot ships)
    except ImportError:
        from libriichi.mjai import Bot          # auto-env fallback

    device = torch.device("cpu")
    brain = Brain(version=4, num_blocks=24, conv_channels=512).eval()
    dqn = DQN(version=4).eval()
    state = torch.load(os.path.join(MYCV, "mortal.pth"),
                       map_location=device, weights_only=False)
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    engine = MortalEngine(brain, dqn, version=4, is_oracle=False, device=device,
                          enable_amp=False, enable_quick_eval=True,
                          enable_rule_based_agari_guard=False, name="mortal")
    bot = Bot(engine, player_id)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        reaction = bot.react(line)
        if reaction:
            print(reaction, flush=True)


if __name__ == "__main__":
    main()
