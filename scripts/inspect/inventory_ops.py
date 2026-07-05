"""Count button-visible frames across captures: records offering non-dapai ops,
and how many of those seqs actually have a saved frame. Decides how much
--op-delay harvest capture (Task 4) is needed.

Usage:  PYTHONPATH=. python scripts/inspect/inventory_ops.py [--sources captures/raw/ai_session]
"""
from __future__ import annotations

import argparse
import os

from majsoul_eye import paths
from majsoul_eye.capture.gtframes import load_frames
from majsoul_eye.capture.schema import read_records
from majsoul_eye.hud import buttons_for_ops
from majsoul_eye.state.ops import ops_from_record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=[paths.RAW_AI_SESSION])
    args = ap.parse_args()

    tot_off = tot_btn = tot_btn_framed = 0
    per_btn: dict[str, int] = {}
    for root in args.sources:
        for cap in paths._ai_captures_in(root):
            frames = {}
            fdir = paths.frames_dir_for(cap)
            if os.path.exists(os.path.join(fdir, "frames.jsonl")):
                frames = load_frames(fdir, statuses=("ok", "timeout"))
            n_off = n_btn = n_btn_framed = 0
            for r in read_records(cap):
                ops = ops_from_record(r)
                btns = buttons_for_ops(ops or [])
                if ops:
                    n_off += 1
                if btns:
                    n_btn += 1
                    if r.seq in frames:
                        n_btn_framed += 1
                        for b in btns:
                            per_btn[b] = per_btn.get(b, 0) + 1
            print(f"{paths.ai_game_name(cap):24s} offers={n_off:4d} "
                  f"button-records={n_btn:3d} with-frame={n_btn_framed:3d}")
            tot_off += n_off; tot_btn += n_btn; tot_btn_framed += n_btn_framed
    print(f"\nTOTAL offers={tot_off} button-records={tot_btn} with-frame={tot_btn_framed}")
    print("per-button (framed):", dict(sorted(per_btn.items())))
    print("\nNOTE: with-frame counts frames captured at the offering seq; whether the"
          "\nbuttons are still rendered in the pixel is verified later (Task 7 count-check).")


if __name__ == "__main__":
    main()
