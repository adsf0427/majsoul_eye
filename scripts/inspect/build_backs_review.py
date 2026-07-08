"""Build a state-DIVERSE, multi-game backs REVIEW dataset for FiftyOne (QA tool).

A single game is mostly near-duplicate consecutive frames with few melds. This
scans every AI game (replay only — fast, no image decode), buckets frames by a
label-relevant STATE SIGNATURE (per-opponent meld count + holding, reach, hero
melds/reach, dora count), then greedily picks a diverse subset — rare/interesting
states first (melds per seat, holding, riichi), spread across many games so
skins vary. Only the selected frames are then annotated (``backs=True``) and
emitted as a flat ``<game>__<seq>`` YOLO dataset (HBB + OBB), reusing
build_dataset's label helpers and the same drop policy (deal/call window,
invariants, backs_sorting whole-frame).

One-shot QA tool (NOT a pipeline stage; sibling of fiftyone_view/overlay_labels).

Run (conda ``auto``, repo root):
  PYTHONPATH=. python scripts/inspect/build_backs_review.py            # -> datasets/backs_review
  PYTHONPATH=. python scripts/inspect/build_backs_review.py --count 500 --sources ai_session3 ai_session4
Then browse:
  PYTHONPATH=. python scripts/inspect/fiftyone_view.py --data datasets/backs_review/obb/data.yaml --name backs_review
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import cv2

from majsoul_eye import paths
from majsoul_eye.annotate import annotate_frame, build_homographies, iter_tile_boxes
from majsoul_eye.annotate import meldsnap as _meldsnap
from majsoul_eye.annotate.seatgt import SEAT_POS, _screen_to_seat
from majsoul_eye.capture.gtframes import build_seq_state, load_frames
from majsoul_eye.hud import DET_NAMES
from majsoul_eye.state.replay import (check_invariants, is_call_window,
                                      is_deal_window, is_score_anim_window)
from majsoul_eye.tiles import NAME_TO_ID

sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "train"))
from build_dataset import box_quad, hbb_label_line, hud_emit, obb_label_line  # noqa: E402

PER_SIG_TOTAL = 4          # at most N frames of one signature (drawn from distinct games)


def sig_of(st):
    opp_melds, opp_hold, opp_reach = [], [], []
    for pos in (1, 2, 3):
        seat = _screen_to_seat(st.hero_seat, SEAT_POS[pos])
        opp_melds.append(len(st.melds[seat]))
        opp_hold.append(st.concealed_counts[seat] % 3 == 2)
        opp_reach.append(bool(st.reach[seat]) if st.reach else False)
    hero = st.hero_seat
    return (tuple(opp_melds), tuple(opp_hold), tuple(opp_reach),
            len(st.melds[hero]), bool(st.reach[hero]) if st.reach else False,
            min(len(st.dora_markers), 5))


def interest(sig) -> int:
    opp_melds, opp_hold, opp_reach, hero_melds, hero_reach, ndora = sig
    return (sum(opp_melds) * 3 + hero_melds * 2 + sum(opp_hold) + sum(opp_reach)
            + (2 if hero_reach else 0) + max(0, ndora - 1))


def select(sources, total_cap):
    frames_by_sig = defaultdict(list)
    for root in sources:
        for cap in sorted(glob.glob(os.path.join(root, "run_*", "game*", "game*.jsonl"))):
            try:
                ss = build_seq_state(cap)
                fr = load_frames(os.path.dirname(cap))
            except Exception:
                continue
            for s in sorted(ss):
                if s not in fr:
                    continue
                st = ss[s]
                if is_deal_window(st) or is_call_window(st):
                    continue
                frames_by_sig[sig_of(st)].append((cap, s))
    chosen = defaultdict(list)
    picked = 0
    for sig in sorted(frames_by_sig, key=lambda g: -interest(g)):
        if picked >= total_cap:
            break
        by_game = defaultdict(list)
        for cap, s in frames_by_sig[sig]:
            by_game[cap].append(s)
        # one frame per distinct game (skin variety), the median (settled) one
        games = sorted(by_game, key=lambda c: len(chosen[c]))   # prefer under-used games
        for cap in games[:PER_SIG_TOTAL]:
            pool = sorted(x for x in by_game[cap] if x not in chosen[cap])
            if pool:
                chosen[cap].append(pool[len(pool) // 2])
                picked += 1
                if picked >= total_cap:
                    break
    return {cap: sorted(v) for cap, v in chosen.items() if v}, len(frames_by_sig), picked


def build(sel, out_root):
    hom = build_homographies(1920, 1080)
    dirs = {}
    for fmt in ("obb", "hbb"):
        for sub in ("images", "labels"):
            d = os.path.join(out_root, fmt, "yolo", sub)
            os.makedirs(d, exist_ok=True)
            dirs[(fmt, sub)] = d
    kept = {"obb": [], "hbb": []}
    dropped = 0
    for cap, seqs in sel.items():
        gname = paths.ai_game_name(cap)
        fr = load_frames(os.path.dirname(cap))
        ss = build_seq_state(cap)
        overrides = _meldsnap.game_meld_overrides(ss, fr, hom)
        for seq in seqs:
            st = ss.get(seq)
            if seq not in fr or st is None or is_deal_window(st) or is_call_window(st) \
                    or check_invariants(st):
                dropped += 1
                continue
            img = cv2.imread(fr[seq])
            if img is None:
                dropped += 1
                continue
            if img.shape[1] != 1920:
                img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            rec = annotate_frame(img, st, hom, backs=True, meld_snap_override=overrides.get(seq))
            rec["_seq"] = seq
            if any(f.endswith(("backs_sorting", "backs_holding")) for f in rec.get("flags", [])):
                dropped += 1
                continue
            reliable = [b for b in iter_tile_boxes(rec) if b.reliable]
            base = f"{gname}__{seq:06d}"
            for fmt in ("obb", "hbb"):
                obb = fmt == "obb"
                lines = [obb_label_line(NAME_TO_ID[b.tile], box_quad(b), w, h) if obb
                         else hbb_label_line(NAME_TO_ID[b.tile], box_quad(b), w, h)
                         for b in reliable if b.tile in NAME_TO_ID]
                if not is_score_anim_window(st):
                    lines += hud_emit(rec, None, w, h, obb)[0]
                if not lines:
                    continue
                cv2.imwrite(os.path.join(dirs[(fmt, "images")], base + ".png"), img)
                with open(os.path.join(dirs[(fmt, "labels")], base + ".txt"), "w") as f:
                    f.write("\n".join(lines) + "\n")
                kept[fmt].append(f"{out_root}/{fmt}/yolo/images/{base}.png".replace(os.sep, "/"))
    names = "\n".join(f"  {i}: '{n}'" for i, n in enumerate(DET_NAMES))
    for fmt in ("obb", "hbb"):
        open(os.path.join(out_root, fmt, "val.txt"), "w", encoding="utf-8").write(
            "\n".join(sorted(kept[fmt])) + "\n")
        open(os.path.join(out_root, fmt, "train.txt"), "w").write("")
        open(os.path.join(out_root, fmt, "data.yaml"), "w", encoding="utf-8").write(
            f"path: {out_root}/{fmt}\ntrain: train.txt\nval: val.txt\n"
            f"nc: {len(DET_NAMES)}\nnames:\n{names}\n")
    return kept, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+",
                    default=["captures/raw/ai_session", "captures/raw/ai_session2",
                             "captures/raw/ai_session3", "captures/raw/ai_session4"])
    ap.add_argument("--count", type=int, default=360, help="target frame count")
    ap.add_argument("--out", default="datasets/backs_review")
    args = ap.parse_args()

    sel, n_sig, picked = select(args.sources, args.count)
    print(f"{n_sig} distinct signatures; selected {picked} frames from {len(sel)} games")
    kept, dropped = build(sel, args.out)
    print(f"built obb={len(kept['obb'])} hbb={len(kept['hbb'])} frames (dropped {dropped}) -> {args.out}")
    print(f"browse: PYTHONPATH=. python scripts/inspect/fiftyone_view.py "
          f"--data {args.out}/obb/data.yaml --name backs_review")


if __name__ == "__main__":
    main()
