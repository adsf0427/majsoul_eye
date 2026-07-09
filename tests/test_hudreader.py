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

# manifest expansion: build_datasets.write_manifest stores "val" as a LIST of
# held-out game names (multi-val convention); older hand-written manifests may
# still carry a scalar. dataset_hud_specs must yield the val names either way.
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "train"))
from train_hudreader import dataset_hud_specs, load_rows  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    for val_field, want in ((["g2"], ["g2"]), ("g2", ["g2"]), (["g1", "g2"], ["g1", "g2"])):
        with open(os.path.join(td, "games.json"), "w", encoding="utf-8") as f:
            json.dump({"val": val_field,
                       "games": [{"name": "g1", "dir": "g1"}, {"name": "g2", "dir": "g2"}]}, f)
        vals, specs = dataset_hud_specs(td)
        assert vals == want, (val_field, vals)
        assert [n for n, _ in specs] == ["g1", "g2"]
    # split is by val NAME LIST (missing labels.jsonl -> both empty, but no crash)
    train, val = load_rows(specs, ["g2"])
    assert train == [] and val == []
print("test_hudreader OK")
