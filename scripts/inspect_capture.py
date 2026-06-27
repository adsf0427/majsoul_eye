"""Join a GT capture with its screenshot frames and report sync quality.

Run after a `record_gt.py --screenshots` session:
    python scripts/inspect_capture.py captures/session1.jsonl captures/session1/

Reports: frame status counts, how many board-changing steps got an 'ok' frame,
and (with --step N) the reconstructed BoardState at a step beside its screenshot
path — so you can eyeball whether the animation had settled. Offline / no client.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from majsoul_eye.capture.schema import read_records
from majsoul_eye.state.replay import Replayer, check_invariants


def load_frames(frames_dir: str) -> dict[int, dict]:
    """seq -> frame record (file, status) from frames.jsonl."""
    idx_path = os.path.join(frames_dir, "frames.jsonl")
    frames: dict[int, dict] = {}
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    frames[r.get("seq", r.get("step"))] = r
    return frames


def state_summary(s) -> str:
    return (f"{s.bakaze}{s.kyoku} honba={s.honba} ltc={s.left_tile_count} "
            f"scores={s.scores} dora={s.dora_markers} "
            f"rivers={[len(s.visible_river(k)) for k in range(4)]} "
            f"melds={[s.num_melds(k) for k in range(4)]} hero={len(s.hero_hand)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", help="GT capture .jsonl")
    ap.add_argument("frames_dir", nargs="?", help="frames dir (with frames.jsonl)")
    ap.add_argument("--step", type=int, default=None, help="Show state+frame for this step.")
    args = ap.parse_args()

    recs = list(read_records(args.capture))
    frames = load_frames(args.frames_dir) if args.frames_dir else {}

    # frame status counts
    status_counts = Counter(r["status"] for r in frames.values())
    saved = {st for st, r in frames.items() if r.get("file")}

    # board-changing records from GT (those that should have a frame), keyed by seq
    rp = Replayer()
    seq_state: dict[int, object] = {}
    changing_seqs: set[int] = set()
    from majsoul_eye.capture.sync import RELEVANT_EVENTS
    for r in recs:
        rp.apply_record(r)
        if r.mjai and any(ev.get("type") in RELEVANT_EVENTS for ev in r.mjai):
            changing_seqs.add(r.seq)
            seq_state[r.seq] = rp.state.copy()

    covered = changing_seqs & saved
    print(f"records: {len(recs)}  frame-records: {len(frames)}  unique-seqs-with-saved-frame: {len(saved)}")
    print(f"frame status: {dict(status_counts)}")
    if changing_seqs:
        print(f"board-changing records: {len(changing_seqs)}  with saved frame: {len(covered)} "
              f"({100*len(covered)//max(1,len(changing_seqs))}%)")

    if args.step is not None:
        s = seq_state.get(args.step)
        fr = frames.get(args.step)
        print(f"\n[seq {args.step}]")
        if s is None:
            ex = sorted(seq_state)[:12]
            print(f"  (no such seq; e.g. valid seqs: {ex} ...)")
        print(f"  frame: {fr}")
        if s is not None:
            print(f"  state: {state_summary(s)}")
            viol = check_invariants(s)
            print(f"  invariants: {'OK' if not viol else viol}")
            if s.hero_hand:
                print(f"  hero_hand: {s.hero_hand}")


if __name__ == "__main__":
    main()
