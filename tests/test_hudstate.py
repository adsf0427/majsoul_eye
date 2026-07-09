"""assemble_hud: routes fields through a stub reader, strips 余/x, collects
buttons; unparseable -> None. Also: reach_stick seat attribution (spec §10) --
the detector emits one SYMMETRIC `reach_stick` class with no seat info baked
in; assemble_hud recovers the seat from detection-relative geometry (vector
from the round_label/wall_count anchor to the stick's center)."""
import numpy as np

from majsoul_eye.hud import DET_NAMES
from majsoul_eye.recognize.detector import Detection
from majsoul_eye.recognize.hudstate import assemble_hud
from majsoul_eye.tiles import TILE_NAMES


def D(name, box):
    """Test helper: a Detection as the 56-class detector would emit it."""
    cls = DET_NAMES.index(name)
    return Detection(xyxy=tuple(float(v) for v in box), name=name,
                     tile=name if cls < len(TILE_NAMES) else None,
                     cls=cls, score=0.9)


class StubReader:
    def __init__(self, answers): self.answers = answers
    def read(self, crop, cls): return self.answers[cls]


frame = np.zeros((1080, 1920, 3), np.uint8)
dets = [D("score_self", (900, 460, 1000, 500)), D("wall_count", (925, 385, 995, 415)),
        D("honba_count", (235, 135, 315, 185)), D("round_label", (905, 350, 1015, 385)),
        D("btn_pon", (1200, 740, 1360, 790)), D("btn_skip", (1400, 740, 1560, 790)),
        D("1m", (100, 900, 190, 1050))]          # tile det must be ignored
r = StubReader({"score_self": "25000", "wall_count": "余64",
                "honba_count": "x1", "round_label": "E3"})
h = assemble_hud(dets, r, frame)
assert h["scores"]["self"] == 25000 and h["scores"]["across"] is None
assert h["wall"] == 64 and h["honba"] == 1 and h["round"] == "E3"
assert h["buttons"] == ["btn_pon", "btn_skip"]

bad = StubReader({"score_self": "2x500", "wall_count": "余",
                  "honba_count": "x1", "round_label": "E3"})
h2 = assemble_hud(dets, bad, frame)
assert h2["scores"]["self"] is None and h2["wall"] is None

# baseline (no reach-stick detections) -> all False
assert h["riichi"] == {"self": False, "right": False, "across": False, "left": False}

# --- reach_stick seat attribution -------------------------------------------
# round_label det above is D("round_label", (905, 350, 1015, 385)) -> anchor
# center = (960, 367.5). All boxes below are single `reach_stick`-class dets
# (no per-seat name baked in) placed at slot-plausible offsets from that
# anchor, exercising the actual dx/dy attribution rule (not bypassed).

# stick BELOW anchor (dy > 0, |dy| >= |dx|) -> "self"
below = D("reach_stick", (900, 500, 1020, 530))
h_self = assemble_hud(dets + [below], r, frame)
assert h_self["riichi"] == {"self": True, "right": False, "across": False, "left": False}

# stick LEFT of anchor (dx < 0, |dx| > |dy|) -> "left"
left = D("reach_stick", (700, 350, 800, 385))
h_left = assemble_hud(dets + [left], r, frame)
assert h_left["riichi"] == {"self": False, "right": False, "across": False, "left": True}
assert "reach_stick" not in h_left["buttons"]           # never conflated with buttons

# two sticks at once -> both attributed independently
h_two = assemble_hud(dets + [below, left], r, frame)
assert h_two["riichi"] == {"self": True, "right": False, "across": False, "left": True}

# NO anchor detection present (no round_label, no wall_count) -> all False,
# even though a reach_stick detection exists -- "leave riichi all False" per spec.
no_anchor_dets = [D("score_self", (900, 460, 1000, 500)), D("1m", (100, 900, 190, 1050)),
                  D("reach_stick", (900, 500, 1020, 530))]
h_no_anchor = assemble_hud(no_anchor_dets, r, frame)
assert h_no_anchor["riichi"] == {"self": False, "right": False, "across": False, "left": False}

# anchor fallback: no round_label, but wall_count present -> still attributes.
# wall_count det = (925, 385, 995, 415) -> center (960, 400); stick well below it -> "self"
fallback_dets = [D("wall_count", (925, 385, 995, 415)), D("1m", (100, 900, 190, 1050)),
                  D("reach_stick", (900, 550, 1020, 580))]
h_fallback = assemble_hud(fallback_dets, r, frame)
assert h_fallback["riichi"]["self"] is True
assert h_fallback["riichi"]["right"] is False and h_fallback["riichi"]["across"] is False
assert h_fallback["riichi"]["left"] is False

print("test_hudstate OK")
