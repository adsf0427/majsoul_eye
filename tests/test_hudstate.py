"""assemble_hud: routes fields through a stub reader, strips 余/x, collects
buttons; unparseable -> None."""
import numpy as np

from majsoul_eye.recognize.hudstate import assemble_hud


class StubReader:
    def __init__(self, answers): self.answers = answers
    def read(self, crop, cls): return self.answers[cls]


frame = np.zeros((1080, 1920, 3), np.uint8)
dets = [("score_self", (900, 460, 1000, 500)), ("wall_count", (925, 385, 995, 415)),
        ("honba_count", (235, 135, 315, 185)), ("round_label", (905, 350, 1015, 385)),
        ("btn_pon", (1200, 740, 1360, 790)), ("btn_skip", (1400, 740, 1560, 790)),
        ("1m", (100, 900, 190, 1050))]          # tile det must be ignored
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
print("test_hudstate OK")
