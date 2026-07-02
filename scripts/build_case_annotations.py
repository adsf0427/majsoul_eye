"""Build corrected relative-seat annotations for the topdown_demo case_frames.

Ties the two halves of the annotation thesis together for the AB case set:
  * WHAT  = Akagi/MahjongCopilot ground truth (river tile ids, riichi index, meld
            composition), read from the captures via scripts.spike_topdown.
  * WHERE = the fixed-camera fullwarp geometry calibrated in
            ``mahjong_relative_annotation_pipeline`` (generate_discard_slots /
            generate_meld_boxes).

Output = out/mahjong_AB_relative_data_with_reliability.json (all 11 cases,
GT-labeled discards + reliable-but-approximate melds), plus optional overlay PNGs.

Discards are geometry-solid (grid + per-seat reading order + riichi sideways +
deep-river 4th-row overflow, verified on ai_run_3_game1 & ai_run_3_game3). Melds are corner-anchored
strips with ~half-tile per-round tolerance (Majsoul places them relative to the
hand; the periphery homography also drifts slightly cross-session).

Run from repo root with PYTHONPATH=. and the conda `auto` python:
    PYTHONPATH=. $PY scripts/build_case_annotations.py                 # write JSON
    PYTHONPATH=. $PY scripts/build_case_annotations.py --overlays out/ # + PNGs
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

import mahjong_relative_annotation_pipeline as P
from scripts.spike_topdown import CASES, load_pair, _screen_to_seat, SEAT_POS

FRAMES_DIR = "fails/topdown_demo/case_frames"
DEFAULT_JSON = os.path.join("out", "mahjong_AB_relative_data_with_reliability.json")

# cache the per-capture replay so 11 cases don't each re-parse the jsonl
import scripts.spike_topdown as _spike
_SS: dict = {}
_orig_bss = _spike.build_seq_state
_spike.build_seq_state = lambda cap: _SS.setdefault(cap, _orig_bss(cap))


def gt_for(case: str) -> dict:
    """Ground truth per seat (relative position) for a case: river tile list + melds."""
    cfg = CASES[case]
    _, state, _ = load_pair(cfg["capture"], cfg["seq"], None)
    seats = []
    for pos, name in enumerate(SEAT_POS):
        seat = _screen_to_seat(state.hero_seat, name)
        river = [{"pai": t.pai, "tsumogiri": bool(t.tsumogiri), "riichi": bool(t.riichi)}
                 for t in state.visible_river(seat)]
        full = [{"pai": t.pai, "riichi": bool(t.riichi), "called": bool(t.called)}
                for t in state.rivers[seat]]
        # from_seat remapped to the SCREEN-relative frame (generate_meld_boxes_v2
        # computes the sideways position as (from_seat - pos) % 4)
        melds = [{"type": m.type, "tiles": list(m.tiles),
                  "from_seat": (pos + ((m.from_seat - seat) % 4)) % 4,
                  "called_pai": m.called_pai, "added_pai": m.added_pai}
                 for m in state.melds[seat]]
        seats.append({"pos": pos, "abs_seat": seat, "river": river, "melds": melds,
                      "sideways_idx": P.river_sideways_index(full)})
    return {"hero_seat": state.hero_seat, "source": {"capture": cfg["capture"], "seq": cfg["seq"]},
            "seats": seats}


def build_item(case: str, hom: dict) -> dict:
    Hinv = hom["H_full_inv"]
    gt = gt_for(case)
    item = {
        "coordinate_system": {
            "seat_convention": "relative",
            "seat_mapping": {"0": "self/自家/bottom", "1": "shimocha/下家/right",
                             "2": "toimen/对家/top", "3": "kamicha/上家/left"},
            "space": "fullwarp (relative pipeline, 1920x1080 calibration)",
            "discard_bbox_model": "fixed_axis_aligned_rect_in_fullwarp",
            "meld_bbox_model": "axis_aligned_rect_in_fullwarp (corner-anchored strip, approximate)",
        },
        "hero_seat": gt["hero_seat"],
        "source": gt["source"],
        "discard_slots": {"0": [], "1": [], "2": [], "3": []},
        "meld_boxes": {"0": [], "1": [], "2": [], "3": []},
    }
    # per-frame image snap for the meld strips (they float a few px per round)
    masks = None
    frame = cv2.imread(os.path.join(FRAMES_DIR, f"{case}.png"))
    if frame is not None:
        full = P.warp_to_full(frame, hom["H_full"], hom["full_size"])
        masks = (P.tile_face_mask(full), P.tile_back_mask(full))
    for s in gt["seats"]:
        pos = s["pos"]
        item["discard_slots"][str(pos)] = P.generate_discard_slots(
            pos, s["river"], Hinv, sideways_idx=s.get("sideways_idx"))
        boxes = P.generate_meld_boxes_v2(pos, s["melds"], Hinv)
        if boxes and masks is not None:
            da, dc, diag = P.snap_meld_strip(masks[0], masks[1], boxes, pos)
            if diag["n_along"] + diag["n_cross"] >= 2:
                boxes = P.shift_boxes(boxes, pos, float(np.clip(da, -70, 70)),
                                      float(np.clip(dc, -16, 16)), Hinv)
        item["meld_boxes"][str(pos)] = boxes
    return item


SEAT_COLORS = {0: (220, 80, 220), 1: (70, 180, 70), 2: (70, 70, 255), 3: (80, 170, 255)}


def render_overlay(case: str, item: dict, hom: dict, out_dir: str) -> None:
    img = cv2.imread(os.path.join(FRAMES_DIR, f"{case}.png"))
    if img is None:
        print(f"  (no frame for {case}, skipping overlay)")
        return
    for pos_key, slots in item["discard_slots"].items():
        color = SEAT_COLORS[int(pos_key)]
        for sl in slots:
            cv2.polylines(img, [np.int32(sl["poly_original"])], True, color, 2, cv2.LINE_AA)
    for pos_key, boxes in item["meld_boxes"].items():
        for b in boxes:
            c = (255, 0, 255) if b["is_kan_tile"] else (0, 200, 255)
            cv2.polylines(img, [np.int32(b["poly_original"])], True, c, 2, cv2.LINE_AA)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{case}_annot.png"), img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_JSON, help="output JSON path")
    ap.add_argument("--overlays", default=None, help="dir to also write annotated PNG overlays")
    ap.add_argument("--cases", nargs="*", default=None, help="subset of case names")
    args = ap.parse_args()

    hom = P.build_homographies(1920, 1080)
    cases = args.cases or list(CASES)
    out = {}
    for case in cases:
        item = build_item(case, hom)
        out[case] = item
        nd = sum(len(v) for v in item["discard_slots"].values())
        nm = sum(len(v) for v in item["meld_boxes"].values())
        print(f"  {case:16} discards={nd:3} melds={nm:3}")
        if args.overlays:
            render_overlay(case, item, hom, args.overlays)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}  ({len(out)} cases)")
    if args.overlays:
        print(f"wrote overlays to {args.overlays}/")


if __name__ == "__main__":
    main()
