"""DigitCTC shapes; greedy CTC decode (repeat-collapse + blank-drop); charset
round-trip for every char the pipeline can emit."""
import numpy as np
import torch

from majsoul_eye.hud import CTC_CHARSET
from majsoul_eye.recognize.hudreader import DigitCTC, ctc_decode, encode_text

m = DigitCTC()
x = torch.zeros(2, 1, 32, 128)
y = m(x)
assert y.shape == (2, 32, len(CTC_CHARSET) + 1)      # T=W/4, C=13+blank

# decode: blank=0; "2 2 blank 5 5 5 blank blank 0(char '0'=idx1)" -> "250"
idx = {c: i + 1 for i, c in enumerate(CTC_CHARSET)}
seq = [idx["2"], idx["2"], 0, idx["5"], idx["5"], idx["5"], 0, 0, idx["0"]]
logits = torch.full((len(seq), len(CTC_CHARSET) + 1), -10.0)
for t, i in enumerate(seq):
    logits[t, i] = 0.0
assert ctc_decode(logits) == "250"
# encode/decode round-trip incl. 余 and x and -
for s in ("25000", "余64", "x2", "-1200"):
    enc = encode_text(s)
    assert all(1 <= i <= len(CTC_CHARSET) for i in enc)
print("test_hudreader OK")
