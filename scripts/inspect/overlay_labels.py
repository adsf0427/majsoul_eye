"""Draw the seeded ROIs/labels on a real frame to calibrate coords.py by eye.

  python scripts/inspect/overlay_labels.py captures/session2.jsonl captures/session2/ \
         --out /tmp/overlay.png [--step N] [--max-width 1920]
"""
from __future__ import annotations
import argparse, glob, json, os
import cv2

from majsoul_eye.capture.schema import read_records
from majsoul_eye.capture.sync import RELEVANT_EVENTS
from majsoul_eye.state.replay import Replayer
from majsoul_eye.coords import HAND, REGIONS, RIVER_ZONES, DORA_STRIP, dora_slot, MAX_DORA
from majsoul_eye.normalize import locate_fullscreen
from majsoul_eye.label import label_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture"); ap.add_argument("frames_dir")
    ap.add_argument("--out", required=True); ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--max-width", type=int, default=1920)
    args = ap.parse_args()

    rp = Replayer(); seq_state = {}
    for r in read_records(args.capture):
        rp.apply_record(r)
        if r.mjai and any(ev.get("type") in RELEVANT_EVENTS for ev in r.mjai):
            seq_state[r.seq] = rp.state.copy()

    frames = {}
    with open(os.path.join(args.frames_dir, "frames.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if d.get("status") == "ok" and d.get("file"):
                    frames[d.get("seq", d.get("step"))] = d["file"]

    # choose a 16:9 frame whose state has a settled 13-tile hand
    step = args.step
    if step is None:
        for st in sorted(frames):
            s = seq_state.get(st)
            if not s or not os.path.exists(frames[st]):
                continue
            im = cv2.imread(frames[st]); h, w = im.shape[:2]
            if abs(w / h - 16 / 9) < 0.02 and s.hero_hand and "?" not in s.hero_hand and len(s.hero_hand) % 3 == 1:
                step = st; break
    if step is None:
        raise SystemExit("no suitable 16:9 frame with a settled hero hand found")

    s = seq_state[step]; frame = cv2.imread(frames[step]); region = locate_fullscreen(frame)
    print(f"seq {step}: {s.bakaze}{s.kyoku} hero_seat={s.hero_seat} hand={s.hero_hand} dora={s.dora_markers} scores={s.scores}")

    def box(nb, color, label=None, t=3):
        x0, y0, x1, y1 = region.norm_to_px(nb)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, t)
        if label:
            cv2.putText(frame, str(label), (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    # hand slots (green), dora (cyan), scores/meta (blue), river zones (yellow, thin)
    n = len(s.hero_hand)
    for i in range(n):
        box(HAND.slot_box(i), (0, 255, 0), s.hero_hand[i] if i < n else None)
    box(DORA_STRIP, (255, 255, 0), "dora-strip", 2)
    for i in range(MAX_DORA):
        box(dora_slot(i), (255, 255, 0), None, 1)
    for k, nb in REGIONS.items():
        box(nb, (255, 128, 0), k, 2)
    for k, nb in RIVER_ZONES.items():
        box(nb, (0, 255, 255), k, 2)

    h, w = frame.shape[:2]
    if w > args.max_width:
        frame = cv2.resize(frame, (args.max_width, round(h * args.max_width / w)))
    cv2.imwrite(args.out, frame)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
