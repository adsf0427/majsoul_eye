"""Per-frame precise annotation.

Assemble river / meld / hero-hand / dora tile boxes for one (frame, GT
``BoardState``) pair. River/meld boxes come from the fullwarp geometry
(``majsoul_eye.annotate.pipeline``) as 4-point perspective quads in ORIGINAL
1920x1080 px; hero hand + dora come from the calibrated NormBox model
(``majsoul_eye.label.autolabel``) as axis-aligned px boxes. Both meet in original
image pixels.

``annotate_frame`` returns the full JSON record (moved verbatim out of
``scripts/annotate/annotate_ai_session.py`` — that script and ``build_dataset`` both call
it). ``iter_tile_boxes`` flattens a record into typed :class:`AnnBox` items and
``crop_box`` cuts each box (perspective-warp for quads, resize for axis-aligned)
so downstream classifier-crop / YOLO consumers share one path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np

from majsoul_eye.annotate import pipeline as P
from majsoul_eye.annotate.seatgt import seat_gt
from majsoul_eye.coords import MAX_DORA, dora_slot
from majsoul_eye.label.autolabel import label_frame
from majsoul_eye.normalize import locate_fullscreen

FILL_OK = 0.25          # face-mask coverage below this = not rendered / occluded
SNAP_MAX_ALONG = 70.0   # clamp for the rigid meld snap (the strip floats per round)
SNAP_MAX_CROSS = 70.0   # the SELF strip also floats vertically (up to ~1/2 tile)


def _fill(ii: np.ndarray, poly) -> float:
    p = np.float32(poly)
    return P._box_fill(ii, p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())


def annotate_frame(img: np.ndarray, state, hom: dict, hand_suspect: bool = False) -> dict:
    """Full annotation record for one frame. `hand_suspect` marks frames right
    after a kyoku start, where the deal/sort animation may not match GT order."""
    Hinv = hom["H_full_inv"]
    full = P.warp_to_full(img, hom["H_full"], hom["full_size"])
    mw = P.tile_face_mask(full)
    mb = P.tile_back_mask(full)                 # face/back discrimination for snap
    ii_w = cv2.integral(mw)
    ii_l = cv2.integral(P.tile_live_mask(full)) # skin-agnostic liveness for back-cell fill

    rec = {"hero_seat": state.hero_seat,
           "kyoku": f"{state.bakaze}{state.kyoku}",
           "discard_slots": {}, "meld_boxes": {}, "hand_boxes": [], "dora_boxes": [],
           "flags": []}

    for pos in range(4):
        river, sideways_idx, melds, seat = seat_gt(state, pos)
        slots = P.generate_discard_slots(pos, river, Hinv, sideways_idx=sideways_idx)
        newest = len(slots) - 1 if state.last_actor == seat else -1
        for i, s in enumerate(slots):
            f = _fill(ii_w, s["face_poly_fullwarp"])
            s["fill"] = round(f, 3)
            if f < FILL_OK:
                s["reliable"] = False
                if i == newest:
                    s["unrendered"] = True      # GT leads the render by ~1 action
                    rec["flags"].append(f"pos{pos}:river[{i}]:unrendered")
                else:
                    s["low_conf"] = True
                    rec["flags"].append(f"pos{pos}:river[{i}]:low_fill={f:.2f}")
        rec["discard_slots"][str(pos)] = slots

        boxes = P.generate_meld_boxes_v2(pos, melds, Hinv)
        if boxes:
            da, dc, diag = P.snap_meld_strip(mw, mb, boxes, pos)
            if diag["n_along"] + diag["n_cross"] >= 2:
                da = float(np.clip(da, -SNAP_MAX_ALONG, SNAP_MAX_ALONG))
                dc = float(np.clip(dc, -SNAP_MAX_CROSS, SNAP_MAX_CROSS))
                boxes = P.shift_boxes(boxes, pos, da, dc, Hinv)
            for b in boxes:
                ii = ii_l if b["tile"] == "back" else ii_w
                f = _fill(ii, b["poly_fullwarp"])
                b["fill"] = round(f, 3)
                b["snap"] = (round(da, 1), round(dc, 1))
                if f < FILL_OK:
                    b["reliable"] = False
                    b["low_conf"] = True
                    rec["flags"].append(f"pos{pos}:meld[{b['tile']}]:low_fill={f:.2f}")
        rec["meld_boxes"][str(pos)] = boxes

    # hero hand via the calibrated HandModel (settled 13-tile states only)
    try:
        region = locate_fullscreen(img)
        hb = []
        for s in label_frame(img, state, region, zones=frozenset({"hand"})):
            x1, y1, x2, y2 = s.px_box
            f = 0.0
            if x2 > x1 and y2 > y1:
                hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
                f = float(((hsv[..., 1] < 70) & (hsv[..., 2] > 165)).mean())
            hb.append({"tile": s.label, "px_box": list(s.px_box), "fill": round(f, 3)})
        if hb and float(np.median([h["fill"] for h in hb])) < 0.30:
            # deal/draw animation still playing — GT leads the render
            for h in hb:
                h["reliable"] = False
            rec["flags"].append("hand:unrendered")
        elif hb and hand_suspect:
            # kyoku just started: tiles may be rendered but not yet GT-sorted
            for h in hb:
                h["reliable"] = False
            rec["flags"].append("hand:deal_unsorted")
        rec["hand_boxes"] = hb
    except Exception as e:                       # hand layout is best-effort
        rec["flags"].append(f"hand:error:{e}")

    # dora indicators: top-left 2D-HUD strip, MAX_DORA (=5) fixed calibrated slots
    # (coords.DORA_STRIP) — resolution-stable at 16:9. The first len(dora_markers)
    # slots are REVEALED (face-up, GT tile class); the rest are face-DOWN backs
    # (tile="back", flagged `back`). Kan-dora reveal left→right, so a revealed slot
    # turns from back to face as the game progresses. `fill` = white-face coverage
    # for revealed / skin-agnostic content coverage for backs; a rendered slot reads high, a
    # not-yet-rendered one (GT leads the client) is flagged not-reliable.
    try:
        region = locate_fullscreen(img)
        n_rev = len(state.dora_markers)
        db = []
        for i in range(MAX_DORA):
            is_back = i >= n_rev
            tile = "back" if is_back else state.dora_markers[i]
            x1, y1, x2, y2 = region.norm_to_px(dora_slot(i))
            f = 0.0
            if x2 > x1 and y2 > y1:
                if is_back:                          # skin-agnostic: any rendered tile back
                    f = float(P.tile_live_mask(img[y1:y2, x1:x2]).mean())
                else:                                # white tile-face
                    hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
                    f = float(((hsv[..., 1] < 70) & (hsv[..., 2] > 165)).mean())
            d = {"tile": tile, "px_box": [x1, y1, x2, y2], "fill": round(f, 3), "back": is_back}
            if f < FILL_OK:
                d["reliable"] = False               # slot not rendered yet (GT leads) / occluded
                rec["flags"].append(f"dora[{i}:{tile}]:low_fill={f:.2f}")
            db.append(d)
        rec["dora_boxes"] = db
    except Exception as e:                       # dora strip is best-effort
        rec["flags"].append(f"dora:error:{e}")

    # HUD fields + action buttons (GT text/ops drive labels; see annotate/hud.py)
    try:
        from majsoul_eye.annotate import hud as HUD
        from majsoul_eye.state.replay import is_score_anim_window
        region = locate_fullscreen(img)
        boxes = HUD.hud_field_boxes(img, state, region)
        if is_score_anim_window(state):
            for b in boxes:
                b["reliable"] = False
            rec["flags"].append("hud:score_anim")
        rec["hud_boxes"] = boxes + HUD.button_boxes(img, state, region)
    except Exception as e:                       # HUD is best-effort like dora
        rec["flags"].append(f"hud:error:{e}")
        rec["hud_boxes"] = []
    return rec


def crop_quad(img: np.ndarray, poly, size: int = 64) -> np.ndarray:
    """Perspective-warp a 4-point original-px quad to a size×size upright patch."""
    src = np.float32(poly)
    dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (size, size))


@dataclass
class AnnBox:
    """One tile box flattened from an annotate_frame record, in ORIGINAL px.

    River/meld carry a 4-point perspective quad (``poly_original``); hero hand /
    dora carry an axis-aligned ``px_box``. ``sideways`` flags rotated tiles
    (riichi discard, called meld tile) whose upright orientation is not
    recoverable from geometry alone.
    """
    zone: str                       # 'river' | 'meld' | 'hand' | 'dora'
    tile: str                       # tiles.TILE_NAMES member ('back' allowed for melds)
    kind: str                       # always 'tile'
    poly_original: Optional[list]   # 4x2 px quad (river/meld); None for hand/dora
    px_box: Optional[list]          # (x0,y0,x1,y1) px (hand/dora); None for river/meld
    sideways: bool
    reliable: bool


def iter_tile_boxes(rec: dict) -> Iterator[AnnBox]:
    """Flatten an annotate_frame record into typed tile boxes (all zones).

    River tiles use the inset ``face_poly_original`` (the QA/crop box); melds use
    the full ``poly_original``. ``reliable`` defaults True (the pipeline only ever
    *sets* it False), matching the annotator's ``.get('reliable', True)`` reads.
    """
    for slots in rec["discard_slots"].values():
        for s in slots:
            yield AnnBox("river", s["tile"], "tile", s["face_poly_original"], None,
                         bool(s.get("riichi")), bool(s.get("reliable", True)))
    for boxes in rec["meld_boxes"].values():
        for b in boxes:
            yield AnnBox("meld", b["tile"], "tile", b["poly_original"], None,
                         bool(b.get("sideways")), bool(b.get("reliable", True)))
    for h in rec["hand_boxes"]:
        yield AnnBox("hand", h["tile"], "tile", None, list(h["px_box"]),
                     False, bool(h.get("reliable", True)))
    for d in rec.get("dora_boxes", []):
        yield AnnBox("dora", d["tile"], "tile", None, list(d["px_box"]),
                     False, bool(d.get("reliable", True)))


def crop_box(img: np.ndarray, box: AnnBox, size: int = 64) -> np.ndarray:
    """Cut one AnnBox to a size×size patch: perspective-warp for a quad, resize
    for an axis-aligned px_box (matches the annotator's per-zone crops)."""
    if box.poly_original is not None:
        return crop_quad(img, box.poly_original, size)
    x1, y1, x2, y2 = box.px_box
    return cv2.resize(img[y1:y2, x1:x2], (size, size))


@dataclass
class HudBox:
    """One HUD box from an annotate_frame record. `text` is the exact string a
    micro-reader must output (None for buttons — class IS the label)."""
    name: str
    px_box: list
    text: Optional[str]
    reliable: bool


def iter_hud_boxes(rec: dict) -> Iterator[HudBox]:
    for d in rec.get("hud_boxes", []):
        if d.get("px_box") is None:
            continue
        yield HudBox(d["name"], list(d["px_box"]), d.get("text"),
                     bool(d.get("reliable", True)))
