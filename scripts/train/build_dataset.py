"""Build an auto-labeled dataset from a synced capture (GT + screenshots), using
the PRECISE fullwarp annotation pipeline (majsoul_eye.annotate).

For every 'ok'/'timeout' screenshot, reconstruct the BoardState at its record,
run the precise annotator, and emit — from ONE calibration — both:
  <out>/crops/<tile>/<seq>_<i>.png   classifier dataset (hand + 4 rivers + melds
                                     + dora; perspective-deskewed 96px face crops)
  <out>/yolo/images/<seq>.png        detector images (the RESIZED 1920x1080 frame)
  <out>/yolo/labels/<seq>.txt        detector labels (YOLO: class cx cy w h, axis-
                                     aligned bbox of each box's original-px quad;
                                     classes 0-37 = tiles, 38-54 = HUD — see below)
  <out>/hud/<field>/<seq>.png        HUD micro-reader crops (15% padded, rotated
                                     upright per majsoul_eye.hud.FIELD_ROT)
  <out>/hud/labels.jsonl             one {"file","name","text","pad"} row per crop

Only boxes the annotator marks ``reliable`` are emitted (drops unrendered newest
discards + low-fill/occluded cells). ``sideways`` tiles (riichi discard, called
meld tile) still go to YOLO but are EXCLUDED from classifier crops — their upright
glyph orientation is not recoverable from geometry, so an upright-only crop set
stays clean (runtime classifies both rotations for these).

HUD fields/buttons (``rec["hud_boxes"]``, from ``annotate_frame`` — see
``majsoul_eye.hud`` for the 17-class taxonomy) are emitted the same way: reliable
boxes become extra YOLO lines (classes 38-54); boxes that also carry ``text``
(numeric/round fields, not buttons) additionally get a padded, upright-rotated
reader crop under ``hud/``. Score-roll/riichi-stick animation frames
(``is_score_anim_window``) are skipped for HUD entirely — belt-and-suspenders on
top of Task 8's per-box ``reliable`` flag. Old (pre-HUD) ``--from-annotations``
records simply lack ``hud_boxes`` and silently emit no HUD labels/crops.

The precise geometry is calibrated at 1920x1080 fullscreen 16:9; frames are
resized to that. Non-16:9 / letterboxed frames are skipped with a warning (their
river/meld boxes would be garbage — see session4).

``--from-annotations DIR`` REUSES the records ``annotate_ai_session.py`` already
wrote (DIR/<capture-stem>.jsonl) instead of re-running ``annotate_frame`` — the
expensive warp/mask/snap runs ONCE (in that script, which parallelizes it), and
this step only cuts crops from the stored polys. Same crop/YOLO output, no double
compute. Assumes the annotations were generated at the frame's native resolution
(true for the 1080p AI games; the records store native-px polys, so this mode does
NOT resize under them).

Usage (conda `auto` env, repo root, PYTHONPATH=.):
  # self-contained (re-annotates):
  python scripts/train/build_dataset.py captures/raw/ai_session/run_3/game1/game1.jsonl \
         captures/raw/ai_session/run_3/game1 --out datasets/ai_g_run3_1
  # reuse annotate_ai_session output (no re-annotation):
  python scripts/train/build_dataset.py captures/raw/ai_session/run_3/game1/game1.jsonl \
         captures/raw/ai_session/run_3/game1 --out datasets/precise_ai_run_3_game1 \
         --from-annotations out/ai_session_annotations --no-yolo
"""

from __future__ import annotations

import argparse
import os
import shutil


def gate_frame(frame, boxes, crops, clf, tau, max_bad):
    """Return the set of box indices to SKIP for occlusion/mislabel. `boxes` and
    `crops` are aligned; a whole-frame drop returns the indices of every
    NON-sideways box.

    `box.sideways` boxes (riichi discards, called-meld tiles — rendered rotated
    90 degrees) are NEVER judged: the classifier is upright-trained and can't
    read them, so a mismatch there is an orientation artifact, not evidence of
    occlusion/mislabel. They are always kept and never count against the
    per-frame `max_bad` budget.
    """
    from majsoul_eye.annotate.consistency import score_frame, frame_decision

    if clf is None or not boxes:
        return set()
    upright_idx = [i for i, b in enumerate(boxes) if not b.sideways]
    if not upright_idx:
        return set()
    upright_crops = [crops[i] for i in upright_idx]
    upright_gts = [boxes[i].tile for i in upright_idx]
    decision, bad = frame_decision(
        score_frame(upright_crops, upright_gts, clf, tau=tau), max_bad=max_bad
    )
    if decision == "keep":
        return set()
    if decision == "drop_frame":
        return {upright_idx[i] for i in range(len(upright_idx))}
    return {upright_idx[i] for i in bad}


def box_quad(box):
    """The box's 4 corner points ``[TL, TR, BR, BL]`` in ORIGINAL px.

    River/meld boxes carry an ordered perspective quad (``poly_original``, already
    in getPerspectiveTransform corner order — see annotate.frame.crop_quad); hand/
    dora carry an axis-aligned ``px_box`` we expand to a rectangle (angle 0). This
    is the single source of geometry for both the OBB and HBB label formats.
    """
    if box.poly_original is not None:
        return [[float(x), float(y)] for x, y in box.poly_original]
    x0, y0, x1, y1 = (float(v) for v in box.px_box)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _clip01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def obb_label_line(cls, quad, w, h):
    """Ultralytics OBB label: ``cls x1 y1 x2 y2 x3 y3 x4 y4`` (normalized, clipped
    to [0,1]), preserving the quad's corner order — keeps the tile's real rotation
    (riichi-sideways discards, called melds, far seats)."""
    coords = " ".join(f"{_clip01(x / w):.6f} {_clip01(y / h):.6f}" for x, y in quad)
    return f"{cls} {coords}"


def hbb_label_line(cls, quad, w, h):
    """Ultralytics HBB label: ``cls cx cy bw bh`` (normalized), the axis-aligned
    bbox of the quad (the historical detector format)."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    bw, bh = (x1 - x0) / w, (y1 - y0) / h
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def hud_emit(rec, frame, w, h, obb):
    """Reliable hud_boxes -> (yolo label lines, reader crops). Crops are padded
    15% per side (jitter headroom for the trainer) and rotated upright per
    hud.FIELD_ROT; buttons contribute a label line only (class IS the label)."""
    import cv2

    from majsoul_eye.hud import HUD_NAME_TO_ID, FIELD_ROT

    ROT_CODE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    PAD = 0.15
    lines, crops = [], []
    for d in rec.get("hud_boxes", []):
        if not d.get("reliable", True) or d.get("px_box") is None:
            continue
        cls = HUD_NAME_TO_ID.get(d["name"])
        if cls is None:
            continue
        x0, y0, x1, y1 = d["px_box"]
        quad = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        lines.append(obb_label_line(cls, quad, w, h) if obb
                     else hbb_label_line(cls, quad, w, h))
        text = d.get("text")
        if text is None or frame is None:
            # frame is None under --reuse-images (label-only, no pixels loaded) — the
            # YOLO line above still stands, but there's nothing to crop from.
            continue
        px, py = int((x1 - x0) * PAD), int((y1 - y0) * PAD)
        cy0, cy1 = max(0, y0 - py), min(h, y1 + py)
        cx0, cx1 = max(0, x0 - px), min(w, x1 + px)
        crop = frame[cy0:cy1, cx0:cx1]
        rot = FIELD_ROT.get(d["name"], 0)
        if rot in ROT_CODE:
            crop = cv2.rotate(crop, ROT_CODE[rot])
        rel = f"{d['name']}/{rec.get('_seq', 0):06d}.png"
        crops.append((rel, crop,
                      {"file": rel, "name": d["name"], "text": text, "pad": PAD}))
    return lines, crops


def main() -> None:
    # Lightweight (numpy-only) import, needed early for the --occ-tau/--occ-max-bad
    # argparse defaults; the heavier cv2/torch imports stay deferred past parse_args().
    from majsoul_eye.annotate.consistency import TAU as OCC_TAU, MAX_BAD as OCC_MAX_BAD

    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("frames_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-violations", action="store_true",
                    help="Skip frames whose reconstructed state fails invariants.")
    ap.add_argument("--crop-size", type=int, default=96,
                    help="Saved classifier crop size (px). preprocess() resizes to 64 at "
                         "train/inference; a larger saved crop gives augmentation headroom.")
    ap.add_argument("--no-crops", action="store_true", help="Skip classifier crops.")
    ap.add_argument("--no-yolo", action="store_true", help="Skip YOLO detector labels.")
    ap.add_argument("--obb", action="store_true",
                    help="Emit ORIENTED 8-point labels (cls x1 y1..x4 y4) from the perspective "
                         "quads instead of axis-aligned HBB (cls cx cy w h). Train a *-obb model.")
    ap.add_argument("--from-annotations", metavar="DIR", default=None,
                    help="Reuse annotate_ai_session records from DIR/<capture-stem>.jsonl "
                         "instead of re-running annotate_frame (no warp/mask recompute).")
    ap.add_argument("--backs", action="store_true",
                    help="EXPERIMENTAL (default OFF): annotate opponent hand-row tile backs "
                         "on the direct (non --from-annotations) path. Records that carry "
                         "back_boxes — from this flag or from a --backs annotate run — emit "
                         "them as YOLO 'back' labels (no classifier crops), and any frame "
                         "with a backs_holding flag is dropped whole for label consistency "
                         "(an unlabeled rendered row would teach the detector to suppress backs).")
    ap.add_argument("--reuse-images", metavar="DIR", default=None,
                    help="Label-only mode: take the frame SET + pixel dims from DIR/<seq>.png "
                         "(an already-built yolo/images dir, e.g. the HBB precise_<game>/yolo/images) "
                         "via a header-only read — no source-PNG decode, and DO NOT re-write images. "
                         "Use to build OBB labels off a finished HBB build (symlink the OBB images dir "
                         "at DIR) without re-encoding the identical frames. Requires --no-crops and no "
                         "--occlusion-gate (both need pixels this mode never loads).")
    ap.add_argument("--occlusion-gate", dest="occlusion_gate", action="store_true",
                    help="Opt-in GT-consistency occlusion/mislabel gate (DEFAULT OFF). Runs a "
                         "classifier over every box and DELETES mismatches. Prefer capture-time "
                         "ROI-stability (see capture/roi_diff.py) to avoid occlusion at the source; "
                         "measured residual is ~0.4%, so this heavier delete-on-build gate is opt-in.")
    ap.set_defaults(occlusion_gate=False)
    ap.add_argument("--occ-tau", type=float, default=OCC_TAU,
                    help="Min P(gt_cls) for a top1-mismatch box to still pass the gate.")
    ap.add_argument("--occ-max-bad", type=int, default=OCC_MAX_BAD,
                    help="Per-frame bad-box budget before dropping the whole frame.")
    args = ap.parse_args()

    if args.reuse_images and (not args.no_crops or args.occlusion_gate):
        ap.error("--reuse-images is label-only (loads no pixels): pass --no-crops and drop --occlusion-gate")

    import cv2  # auto env
    from PIL import Image  # header-only reads: frame sizes (--reuse-images) + copy fast-path

    def plain_rgb(path: str) -> bool:
        """True when the PNG decodes to exactly the 8-bit 3-channel array
        ``cv2.imwrite`` would re-emit (no alpha/16-bit/palette) — the
        precondition for copying the source file instead of re-encoding."""
        try:
            with Image.open(path) as im:      # header only, no pixel decode
                return im.mode == "RGB"
        except Exception:
            return False

    from majsoul_eye import paths
    from majsoul_eye.tiles import NAME_TO_ID
    from majsoul_eye.state.replay import check_invariants, is_deal_window, is_call_window, is_score_anim_window
    from majsoul_eye.capture.gtframes import build_seq_state, load_frames
    from majsoul_eye.annotate import build_homographies, annotate_frame, iter_tile_boxes, crop_box
    from majsoul_eye.annotate import meldsnap as _meldsnap

    # seq -> frame path (keep 'timeout' frames as before)
    frames = load_frames(args.frames_dir, statuses=("ok", "timeout"))

    if args.from_annotations:
        import json
        stem = paths.ai_game_name(args.capture)
        ann_path = os.path.join(args.from_annotations, f"{stem}.jsonl")
        recs = {}
        with open(ann_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    recs[r["seq"]] = r
        # cheap replay (only annotate_frame is skipped): feeds the deal-window drop
        # and, under --drop-violations, the invariant filter.
        seq_state = build_seq_state(args.capture)
        hom = None
        seqs = sorted(recs)
        meld_overrides = {}
        print(f"reuse: {len(recs)} records <- {ann_path}")
    else:
        seq_state = build_seq_state(args.capture)
        hom = build_homographies(1920, 1080)
        seqs = sorted(seq_state)
        meld_overrides = _meldsnap.game_meld_overrides(seq_state, frames, hom)

    crops_dir = os.path.join(args.out, "crops")
    img_dir = os.path.join(args.out, "yolo", "images")
    lbl_dir = os.path.join(args.out, "yolo", "labels")
    for d in (crops_dir, img_dir, lbl_dir):
        os.makedirs(d, exist_ok=True)

    n_frames = n_crops = n_yolo = n_skip = n_letterbox = n_deal = n_call = 0
    n_occ_box = n_occ_frame = n_backs_hold = 0
    hud_meta = []
    occ_clf = None
    if args.occlusion_gate:
        from majsoul_eye.recognize.classifier import TileClassifier
        occ_clf = TileClassifier()

    for seq in seqs:
        if seq not in frames:
            continue
        state = seq_state.get(seq)
        # Deal-in animation frame (hand still dealing/sorting, GT boxes don't match
        # the pixels) — drop from crops AND YOLO. See state.replay.is_deal_window.
        if is_deal_window(state):
            n_deal += 1
            continue
        # Call-window frame (meld animation mid-flight, GT updated but pixels lag)
        # — drop from crops AND YOLO. See state.replay.is_call_window.
        if is_call_window(state):
            n_call += 1
            continue
        if args.drop_violations:
            if state is None or check_invariants(state):
                n_skip += 1
                continue
        if args.reuse_images:
            # Label-only: frame SET + dims come from an already-built yolo/images dir (the
            # reference build already applied the letterbox/resize/imread filters, so a seq
            # missing there is one it dropped). No decode, no re-encode.
            imgp = os.path.join(args.reuse_images, f"{seq:06d}.png")
            try:
                with Image.open(imgp) as im:     # header only, no pixel decode
                    w, h = im.size
            except (FileNotFoundError, OSError):
                n_skip += 1
                continue
            frame = None
        else:
            frame = cv2.imread(frames[seq])
            if frame is None:
                n_skip += 1
                continue
            h, w = frame.shape[:2]
            if abs(w / h - 16 / 9) > 0.02:       # letterboxed / non-16:9 → precise geom invalid
                n_letterbox += 1
                continue
            # default path calibrates at 1920x1080 and resizes to match; reuse path keeps the
            # native frame (the stored polys are native-px — don't rescale the frame under them).
            resized = (w, h) != (1920, 1080) and not args.from_annotations
            if resized:
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
                h, w = 1080, 1920

        rec = (recs[seq] if args.from_annotations
               else annotate_frame(frame, seq_state[seq], hom, backs=args.backs,
                                   meld_snap_override=meld_overrides.get(seq)))
        rec["_seq"] = seq

        # Backs experiment: a post-tedashi 理牌 reflow row (backs_sorting, pixel-
        # gated) has its templates misaligned mid-animation (see annotate/backs.py),
        # so that row goes unlabeled — drop the WHOLE frame to keep the 'back'
        # training signal consistent rather than teach suppression. (Holding rows
        # ARE labeled now — static row + drawn tile — so they are NOT dropped;
        # backs_holding stays matched only for backward-compat with pre-STATUS-1.46
        # annotation JSONs that still carry it.)
        if rec.get("back_boxes") is not None and any(
                f.endswith(("backs_holding", "backs_sorting")) for f in rec.get("flags", [])):
            n_backs_hold += 1
            continue

        reliable = [b for b in iter_tile_boxes(rec) if b.reliable]
        skip = set()
        if occ_clf is not None and reliable:
            gate_crops = [crop_box(frame, b, size=args.crop_size) for b in reliable]
            skip = gate_frame(frame, reliable, gate_crops, occ_clf,
                              args.occ_tau, args.occ_max_bad)
            # gate_frame never judges sideways boxes, so a whole-frame drop shows up
            # as "every non-sideways reliable box is skipped", not the full range.
            upright_reliable = {i for i, b in enumerate(reliable) if not b.sideways}
            if upright_reliable and skip == upright_reliable:
                n_occ_frame += 1
            else:
                n_occ_box += len(skip)

        yolo_lines = []
        ci = 0
        for bi, box in enumerate(reliable):
            if bi in skip:
                continue
            cls = NAME_TO_ID.get(box.tile)
            if cls is None:
                continue
            if not args.no_yolo:
                # OBB keeps the perspective quad's rotation; HBB collapses it to an
                # axis-aligned bbox (both from the same box_quad — river/meld quad or
                # hand/dora rectangle).
                quad = box_quad(box)
                yolo_lines.append(obb_label_line(cls, quad, w, h) if args.obb
                                  else hbb_label_line(cls, quad, w, h))
            # classifier crop: skip sideways (upright orientation not geometry-recoverable)
            # and opponent hand backs (detector-only labels — 39 near-identical crops per
            # frame would just flood the classifier's 'back' class)
            if not args.no_crops and not box.sideways and box.zone != "oppback":
                crop = crop_box(frame, box, size=args.crop_size)
                if crop.size:
                    cdir = os.path.join(crops_dir, box.tile)
                    os.makedirs(cdir, exist_ok=True)
                    cv2.imwrite(os.path.join(cdir, f"{seq:06d}_{ci:03d}.png"), crop)
                    ci += 1
                    n_crops += 1

        # HUD fields/buttons -> 55-class YOLO lines + reader crops (hud/<field>/<seq>.png).
        # belt-and-suspenders with Task 8's per-box `reliable` flag: is_score_anim_window
        # tolerates state=None (annotations-reuse path can have seqs with missing state).
        if not is_score_anim_window(state):
            hlines, hcrops = hud_emit(rec, frame, w, h, args.obb)
            if not args.no_yolo:
                yolo_lines += hlines
            if not args.no_crops:
                for rel, crop, meta in hcrops:
                    p = os.path.join(args.out, "hud", rel)
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    cv2.imwrite(p, crop)
                    hud_meta.append(meta)

        if yolo_lines and not args.no_yolo:
            if frame is not None:                                        # reuse-images: image already on disk (symlinked)
                dst = os.path.join(img_dir, f"{seq:06d}.png")
                if not resized and plain_rgb(frames[seq]):
                    shutil.copyfile(frames[seq], dst)   # pixel-identical to the source — skip the ~100ms PNG re-encode
                else:
                    cv2.imwrite(dst, frame)             # RESIZED (or non-plain-RGB source) frame
            with open(os.path.join(lbl_dir, f"{seq:06d}.txt"), "w") as lf:
                lf.write("\n".join(yolo_lines) + "\n")
            n_yolo += 1
        n_frames += 1

    if hud_meta:
        import json
        with open(os.path.join(args.out, "hud", "labels.jsonl"), "w", encoding="utf-8") as f:
            for meta in hud_meta:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"frames labeled: {n_frames}  crops: {n_crops}  yolo-imgs: {n_yolo}  "
          f"skipped: {n_skip}  deal-skipped: {n_deal}  call-skipped: {n_call}  letterbox-skipped: {n_letterbox}  "
          f"occ-box-skipped: {n_occ_box}  occ-frame-dropped: {n_occ_frame}  "
          f"backs-holding-dropped: {n_backs_hold}  hud-crops: {len(hud_meta)}")
    print(f"dataset -> {args.out}")


if __name__ == "__main__":
    main()
