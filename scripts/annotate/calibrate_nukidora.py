"""One-shot: measure the sanma nukidora-pile geometry (per-seat anchor + step).

Nukidora (拔北) tiles render face-up in each seat's meld zone, laid in a row
(STATUS §1.59). GT gives the exact count per seat (``BoardState.nukidora``);
this tool finds the tiles as tile-face-mask connected components inside a
generous search window around the seat's meld corner, keeps only frames whose
component count MATCHES the GT count (occlusion/animation/contamination frames
self-exclude), and fits per seat:

    anchor = centre of nukidora #0 (fullwarp px), step = advance per tile,
    foot   = median component (w, h).

Merged runs (adjacent faces bridging the crevice) are accepted when a single
component's along-extent is ~n*w. Frames inside is_call_window are skipped
(GT leads the render). Prints a suggested ``NUKI_STRIP_3P`` for
``majsoul_eye.annotate.pipeline``.

Run (conda `auto` env, repo root):
  PYTHONPATH=. python scripts/annotate/calibrate_nukidora.py \
      --captures captures/raw/ai_session_3p/run_*/game*/game*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np

from majsoul_eye.annotate import pipeline as P
from majsoul_eye.annotate.seatgt import seat_gt
from majsoul_eye import paths
from majsoul_eye.capture.gtframes import build_seq_state, load_frames
from majsoul_eye.state.replay import is_call_window


# Per-pos absolute search windows (fullwarp px), derived from gridded fullwarp
# dumps of real nuki frames + the pos2 auto-fit (dump_nuki_views; the pile sits
# in its own lane between the meld strip and the river, NOT at the meld corner).
# along = pile growth direction; foot = expected face box (upright for 0/2,
# sideways for 1/3). Windows are deliberately tight so hand/UI warps can't leak in.
WINDOWS = {
    0: {"rect": (1850, 1470, 2350, 1680), "along": (-1.0, 0.0), "foot": (73.0, 92.0)},  # self: grows leftward
    1: {"rect": (2040, 430, 2380, 950),   "along": (0.0, 1.0),  "foot": (92.0, 73.0)},  # right: grows down
    2: {"rect": (830, 250, 1350, 460),    "along": (1.0, 0.0),  "foot": (73.0, 92.0)},  # across: grows right
    3: {"rect": (760, 1250, 980, 1620),   "along": (0.0, -1.0), "foot": (85.0, 72.0)},  # left: grows up
}


def components_in_window(mask: np.ndarray, pos: int) -> list[dict]:
    """Tile-face components inside the seat's nuki search window (fullwarp)."""
    x1, y1, x2, y2 = WINDOWS[pos]["rect"]
    fw, fh = WINDOWS[pos]["foot"]
    roi = mask[y1:y2, x1:x2]
    n, _, stats, cents = cv2.connectedComponentsWithStats((roi > 0).astype(np.uint8), 8)
    along = np.array(WINDOWS[pos]["along"], float)
    horiz = abs(along[0]) > abs(along[1])           # pile grows along x?
    out = []
    for i in range(1, n):
        bw, bh, area = stats[i, 2], stats[i, 3], stats[i, 4]
        if area < 0.45 * fw * fh:                   # specks / clipped edges
            continue
        # size gate: the non-growth dimension must look like one tile face;
        # the growth dimension may span a merged run of up to 4 faces.
        grow, keep = (bw, bh) if horiz else (bh, bw)
        gf, kf = (fw, fh) if horiz else (fh, fw)
        if not (0.62 * kf <= keep <= 1.38 * kf and 0.62 * gf <= grow <= 4.6 * gf):
            continue
        out.append({"c": (cents[i][0] + x1, cents[i][1] + y1),
                    "wh": (float(bw), float(bh)), "area": float(area),
                    "grow": float(grow), "gf": gf})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", nargs="*",
                    default=sorted(glob.glob("captures/raw/ai_session_3p/run_*/game*/game*.jsonl")))
    ap.add_argument("--per-game", type=int, default=120)
    args = ap.parse_args()

    hom = P.build_homographies(1920, 1080)
    samples: dict[int, list] = {0: [], 1: [], 2: [], 3: []}   # pos -> (n, centers, whs)

    for cap in args.captures:
        try:
            seq_state = build_seq_state(cap)
            frames = load_frames(paths.frames_dir_for(cap))
        except Exception as e:
            print(f"  {cap}: SKIP ({e})")
            continue
        seqs = [s for s in sorted(seq_state) if s in frames]
        if len(seqs) > args.per_game:
            idx = np.linspace(0, len(seqs) - 1, args.per_game).astype(int)
            seqs = [seqs[i] for i in sorted(set(idx))]
        got = 0
        for seq in seqs:
            state = seq_state[seq]
            if not getattr(state, "sanma", False) or is_call_window(state):
                continue
            P.set_sanma(True)
            wanted = [(pos, seat_gt(state, pos)[3]) for pos in range(4)]
            wanted = [(pos, seat) for pos, seat in wanted
                      if seat is not None and seat < len(state.nukidora)
                      and state.nukidora[seat] > 0 and not state.melds[seat]]
            if not wanted:
                continue
            img = cv2.imread(frames[seq])
            if img is None:
                continue
            if img.shape[1] != 1920:
                img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
            full = P.warp_to_full(img, hom["H_full"], hom["full_size"])
            mask = P.tile_face_mask(hsv=cv2.cvtColor(full, cv2.COLOR_BGR2HSV))
            for pos, seat in wanted:
                nuki = state.nukidora[seat]
                comps = components_in_window(mask, pos)
                along = np.array(WINDOWS[pos]["along"], float)
                for c in comps:
                    c["t"] = float(np.dot(np.array(c["c"]), along))
                comps.sort(key=lambda c: c["t"])
                if len(comps) == nuki:                            # separated faces
                    centers = [c["c"] for c in comps]
                elif len(comps) == 1 and nuki > 1:                # merged run
                    ext, gf = comps[0]["grow"], comps[0]["gf"]
                    if not (0.75 * nuki * gf <= ext <= 1.30 * nuki * gf):
                        continue
                    step_len = ext / nuki
                    start = np.array(comps[0]["c"], float) - (ext / 2 - step_len / 2) * along
                    centers = [tuple(start + k * step_len * along) for k in range(nuki)]
                else:
                    continue
                samples[pos].append((nuki, centers, [c["wh"] for c in comps]))
                got += 1
        print(f"  {cap}: {got} seat-samples", flush=True)

    print("\n=== nukidora pile fit (fullwarp px) ===")
    print("NUKI_STRIP_3P = {")
    for pos in range(4):
        ss = samples[pos]
        if not ss:
            print(f"    # pos{pos}: NO SAMPLES")
            continue
        along = np.array(WINDOWS[pos]["along"], float)
        anchors = [np.array(cs[0], float) for _, cs, _ in ss]
        steps = [np.array(cs[k + 1], float) - np.array(cs[k], float)
                 for _, cs, _ in ss for k in range(len(cs) - 1)]
        whs = [wh for _, _, whl in ss for wh in whl]
        am = np.median(np.stack(anchors), axis=0)
        asd = np.std(np.stack(anchors), axis=0)
        if steps:
            sm = np.median(np.stack(steps), axis=0)
            ssd = np.std(np.stack(steps), axis=0)
        else:
            sm, ssd = along * WINDOWS[pos]["foot"][0], (0.0, 0.0)
        wm = np.median(np.stack([np.array(x) for x in whs]), axis=0)
        print(f'    {pos}: {{"anchor": ({am[0]:.1f}, {am[1]:.1f}), '
              f'"step": ({sm[0]:.2f}, {sm[1]:.2f}), "foot": ({wm[0]:.1f}, {wm[1]:.1f})}},'
              f'  # n={len(ss)} σa=({asd[0]:.1f},{asd[1]:.1f}) σs=({ssd[0]:.1f},{ssd[1]:.1f})')
    print("}")


if __name__ == "__main__":
    main()
