"""End-to-end HUD accuracy vs GT on a held-out game: detector v2 + HudReader
-> assemble_hud, compared field-by-field to the replayed BoardState.

Iterates every 'ok' frame of one capture (skips `is_deal_window` /
`is_score_anim_window` seqs -- same drop policy build_dataset/annotate_frame use
for HUD, see state/replay.py), runs the 55-class tile+HUD detector, keeps only
the HUD detections (ids 38-54, `hud.DET_NAMES`), assembles them via
`recognize.hudstate.assemble_hud`, and compares field-by-field against GT
derived from the replayed BoardState (`annotate.hud.field_texts` for the
numeric/round/wind fields, `hud.buttons_for_ops` for the button set). Prints
per-field exact-match rates + the whole-frame all-fields-correct rate (one
wrong field/button = the whole frame is wrong; spec
docs/superpowers/specs/2026-07-04-hud-detection-design.md §6).

Usage (real weights, once v2 exists):
  PYTHONPATH=. python scripts/inspect/qa_hud.py \
      captures/raw/ai_session/run_8/game1/game1.jsonl

Usage (--selftest -- proves the comparison/aggregation wiring without any
trained weights; see _FakeDetector/_FakeReader below):
  PYTHONPATH=. python scripts/inspect/qa_hud.py \
      captures/raw/ai_session/run_13/game1/game1.jsonl --selftest --limit 200

(Previously a real `TileDetector.predict()` crashed with `IndexError` on any
HUD-class box, since `_parse_result` set `.tile = TILE_NAMES[cls]`
unconditionally. Fixed: `Detection` now carries `.name` (valid for all 55
classes) alongside `.tile` (None for HUD-class ids). This script reads `.name`
via `det_to_hud_pairs` below, never `.tile`.)
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import cv2

from majsoul_eye import paths
from majsoul_eye.annotate.hud import field_texts
from majsoul_eye.capture.gtframes import build_seq_state, load_frames
from majsoul_eye.hud import NUMERIC_FIELDS, buttons_for_ops
from majsoul_eye.recognize.hudstate import _to_int, assemble_hud
from majsoul_eye.state.replay import is_deal_window, is_score_anim_window
from majsoul_eye.tiles import NUM_CLASSES

DEFAULT_WEIGHTS = "majsoul_eye/recognize/tile_detector.pt"
DEFAULT_READER = "majsoul_eye/recognize/hud_reader.pt"

# Stable print/iteration order; keys match both field_texts()'s output names and
# flat_from_hud()'s output names below (so gt/pred compare like-for-like).
FIELDS = ["score_self", "score_right", "score_across", "score_left",
          "round_label", "wall_count", "riichi_stick_count", "honba_count",
          "seat_wind_self"]
_PREFIX = {"wall_count": "余", "riichi_stick_count": "x", "honba_count": "x"}

# _to_int is imported from majsoul_eye.recognize.hudstate (not re-implemented here)
# so this GT-side normalization can never drift from assemble_hud's own parsing --
# used to fold the GT side (field_texts emits reader-target strings like '余64'/
# 'x1') into the same int shape assemble_hud's parsed output uses, so the two
# sides of the comparison are like-for-like (brief requirement).


def gt_fields(state) -> dict:
    """BoardState -> {field_name: value} in the SAME shape assemble_hud emits
    (ints for numeric fields, str for round_label/seat_wind_self). A field
    field_texts() omits (GT genuinely unknown, e.g. before the first kyoku)
    is simply absent here -> not scored either way."""
    out = {}
    for name, text in field_texts(state).items():
        out[name] = _to_int(text, _PREFIX.get(name, "")) if name in NUMERIC_FIELDS else text
    return out


def flat_from_hud(hud: dict) -> dict:
    """assemble_hud's nested dict -> the flat {field_name: value} shape gt_fields
    uses (buttons are compared separately -- a set, not a scalar field)."""
    return {
        "score_self": hud["scores"]["self"], "score_right": hud["scores"]["right"],
        "score_across": hud["scores"]["across"], "score_left": hud["scores"]["left"],
        "round_label": hud["round"], "wall_count": hud["wall"],
        "riichi_stick_count": hud["kyotaku"], "honba_count": hud["honba"],
        "seat_wind_self": hud["seat_wind"],
    }


def det_to_hud_pairs(dets) -> list:
    """Detector output -> assemble_hud's `dets` arg: (cls_name, px_box) pairs,
    HUD classes only (id >= tiles.NUM_CLASSES). Uses `.name` (valid for all 55
    classes, incl. HUD ids), NOT `.tile` (None for HUD-class detections --
    see `recognize.detector.Detection`)."""
    out = []
    for d in dets:
        if d.cls < NUM_CLASSES:
            continue
        out.append((d.name, tuple(int(v) for v in d.xyxy)))
    return out


# --- --selftest oracle stand-ins --------------------------------------------
# Neither trained weights nor a GPU exist yet for v2 (brief). These are NOT
# meant to measure real detector/reader accuracy -- they prove the comparison/
# aggregation logic (gt_fields/flat_from_hud/the run() loop/the printed rates)
# is wired correctly by feeding it GT-derived answers with a known, injected
# error rate. If the pipeline were silently broken (e.g. comparing the wrong
# keys, or a shape mismatch that always vacuously "matches"), the printed rates
# would land at 100% or 0% instead of tracking (1 - drop_rate/err_rate) -- that
# is exactly what running --selftest is meant to catch.
class _FakeReader:
    """Oracle reader: returns the true text for the field it's asked to read
    (set per-frame by _FakeDetector.predict), corrupted with probability
    err_rate so int()/string comparisons in gt_fields vs flat_from_hud actually
    exercise the MISMATCH path too (an oracle that is never wrong would make a
    broken comparison look identical to a correct one)."""

    def __init__(self, err_rate: float = 0.08, seed: int = 0):
        self.answers: dict[str, str] = {}
        self.err_rate = err_rate
        self.rng = random.Random(seed)

    def read(self, crop, cls: str) -> str:
        text = self.answers.get(cls, "")
        if self.rng.random() < self.err_rate:
            return text + "?"          # guaranteed int()/string mismatch
        return text


class _FakeDetector:
    """Oracle detector: for the CURRENT frame (set_frame), emits one Detection
    per GT-known field (real on-frame px boxes from HUD_SEEDS, so crops are
    non-empty) + one per expected button, each dropped with probability
    drop_rate (simulates a missed detection), plus one bogus tile detection
    (id < NUM_CLASSES) to prove det_to_hud_pairs filters tiles out."""

    def __init__(self, reader: _FakeReader, drop_rate: float = 0.05, seed: int = 1):
        self.reader = reader
        self.drop_rate = drop_rate
        self.rng = random.Random(seed)
        self._state = None
        self._region = None

    def set_frame(self, state, region) -> None:
        self._state = state
        self._region = region

    def predict(self, frame_bgr):
        from majsoul_eye.coords import HUD_SEEDS
        from majsoul_eye.hud import HUD_NAME_TO_ID
        from majsoul_eye.recognize.detector import Detection

        texts = field_texts(self._state)
        self.reader.answers = dict(texts)      # oracle "sees" this frame's GT
        # name=/tile= mirror a real Detection (recognize.detector.Detection): tile
        # ids carry tile==name, HUD ids carry tile=None (see det_to_hud_pairs, which
        # must key off .name, not .tile).
        dets = [Detection(xyxy=(100, 900, 190, 1050), name="1m", tile="1m", cls=0, score=0.99)]
        for name in texts:
            if self.rng.random() < self.drop_rate:
                continue
            x0, y0, x1, y1 = (int(v) for v in self._region.norm_to_px(HUD_SEEDS[name]))
            dets.append(Detection(xyxy=(x0, y0, x1, y1), name=name, tile=None,
                                  cls=HUD_NAME_TO_ID[name], score=0.99))
        for name in buttons_for_ops(self._state.pending_ops or []):
            if self.rng.random() < self.drop_rate:
                continue
            dets.append(Detection(xyxy=(1200, 740, 1360, 790), name=name, tile=None,
                                  cls=HUD_NAME_TO_ID[name], score=0.9))
        return dets


def _load_detector(weights_path: str):
    if not os.path.exists(weights_path):
        sys.exit(f"qa_hud: detector weights not found at {weights_path!r}. Train v2 first "
                  f"(scripts/train/train_detector.py --data datasets/v2/detector/data.yaml), "
                  f"pass --weights, or use --selftest to exercise the comparison logic "
                  f"without trained weights.")
    try:
        from majsoul_eye.recognize.detector import TileDetector
        return TileDetector(weights_path)
    except Exception as e:
        sys.exit(f"qa_hud: failed to load detector from {weights_path!r}: {e}")


def _load_reader(reader_path: str):
    if not os.path.exists(reader_path):
        sys.exit(f"qa_hud: HUD reader weights not found at {reader_path!r}. Train first "
                  f"(scripts/train/train_hudreader.py), pass --reader, or use --selftest "
                  f"to exercise the comparison logic without trained weights.")
    try:
        from majsoul_eye.recognize.hudreader import HudReader
        return HudReader(reader_path)
    except Exception as e:
        sys.exit(f"qa_hud: failed to load HUD reader from {reader_path!r}: {e}")


def run(capture: str, frames_dir: str, detector, reader, limit: int | None = None) -> dict:
    from majsoul_eye.normalize import locate_fullscreen

    seq_state = build_seq_state(capture)
    frames = load_frames(frames_dir)
    seqs = sorted(s for s in seq_state if s in frames)

    n_deal = n_score_anim = n_imread_fail = n_eval = 0
    field_hit = {f: 0 for f in FIELDS}
    field_total = {f: 0 for f in FIELDS}
    btn_exact_hit = btn_exact_total = 0
    btn_recall_hit = btn_recall_total = 0
    frame_hit = frame_total = 0

    for seq in seqs:
        if limit is not None and n_eval >= limit:
            break
        state = seq_state[seq]
        if is_deal_window(state):
            n_deal += 1
            continue
        if is_score_anim_window(state):
            n_score_anim += 1
            continue
        frame = cv2.imread(frames[seq])
        if frame is None:
            n_imread_fail += 1
            continue
        n_eval += 1

        if hasattr(detector, "set_frame"):          # oracle path (--selftest)
            detector.set_frame(state, locate_fullscreen(frame))
        dets = detector.predict(frame)
        hud = assemble_hud(det_to_hud_pairs(dets), reader, frame)
        pred = flat_from_hud(hud)
        gt = gt_fields(state)
        gt_btns = buttons_for_ops(state.pending_ops or [])

        frame_total += 1
        frame_ok = True
        for f in FIELDS:
            if f not in gt:
                continue
            field_total[f] += 1
            if pred.get(f) == gt[f]:
                field_hit[f] += 1
            else:
                frame_ok = False

        btn_exact_total += 1
        btns_match = hud["buttons"] == gt_btns
        if btns_match:
            btn_exact_hit += 1
        else:
            frame_ok = False
        if gt_btns:                                  # recall: button-frames only (§7: data is scarce)
            btn_recall_total += 1
            if btns_match:
                btn_recall_hit += 1

        if frame_ok:
            frame_hit += 1

    return {
        "n_seqs": len(seqs), "n_deal": n_deal, "n_score_anim": n_score_anim,
        "n_imread_fail": n_imread_fail, "n_eval": n_eval,
        "field_hit": field_hit, "field_total": field_total,
        "btn_exact_hit": btn_exact_hit, "btn_exact_total": btn_exact_total,
        "btn_recall_hit": btn_recall_hit, "btn_recall_total": btn_recall_total,
        "frame_hit": frame_hit, "frame_total": frame_total,
    }


def _rate(hit: int, total: int) -> str:
    return f"{hit / total:6.2%}  ({hit}/{total})" if total else f"{'n/a':>8s}  (0/0)"


def print_report(r: dict) -> None:
    print(f"seqs: {r['n_seqs']} board-changing | skipped: deal-window={r['n_deal']} "
          f"score-anim={r['n_score_anim']} imread-fail={r['n_imread_fail']} "
          f"| evaluated={r['n_eval']}")
    print("\nper-field exact-match rate:")
    for f in FIELDS:
        print(f"  {f:24s} {_rate(r['field_hit'][f], r['field_total'][f])}")
    print(f"  {'buttons (exact set)':24s} {_rate(r['btn_exact_hit'], r['btn_exact_total'])}")
    print(f"  {'buttons (recall, btn-frames)':24s} {_rate(r['btn_recall_hit'], r['btn_recall_total'])}")
    print(f"\nwhole-frame all-correct: {_rate(r['frame_hit'], r['frame_total'])}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", help="GT capture jsonl, e.g. captures/raw/ai_session/run_8/game1/game1.jsonl")
    ap.add_argument("--frames-dir", default=None,
                    help="default: majsoul_eye.paths.frames_dir_for(capture)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="tile+HUD detector weights (v2, 55-class)")
    ap.add_argument("--reader", default=DEFAULT_READER, help="HudReader checkpoint")
    ap.add_argument("--limit", type=int, default=None, help="only score the first N eligible frames (smoke)")
    ap.add_argument("--selftest", action="store_true",
                    help="use a FAKE GT-oracle detector+reader (with injected errors) instead of "
                         "loading real weights, to exercise the comparison/aggregation logic before "
                         "trained v2 weights exist. Ignores --weights/--reader.")
    args = ap.parse_args()

    frames_dir = args.frames_dir or paths.frames_dir_for(args.capture)

    if args.selftest:
        reader = _FakeReader()
        detector = _FakeDetector(reader)
    else:
        detector = _load_detector(args.weights)
        reader = _load_reader(args.reader)

    print_report(run(args.capture, frames_dir, detector, reader, limit=args.limit))


if __name__ == "__main__":
    main()
