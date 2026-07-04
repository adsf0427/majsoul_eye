"""GTWriter round-trip: header + one line per record, readable by read_records.

Plain-script style: PYTHONPATH=. <auto-python> tests/test_schema_writer.py
"""
import os
import tempfile

from majsoul_eye.capture.schema import GTRecord, GTWriter, read_records


def _rec(seq):
    return GTRecord(seq=seq, ts=1.0, flow_id="", seat=0, last_op_step=0,
                    syncing=False, method=".lq.ActionPrototype",
                    action_name="ActionDiscardTile",
                    raw_liqi={"method": ".lq.ActionPrototype", "data": {"name": "ActionDiscardTile"}},
                    mjai=[{"type": "dahai", "pai": "5m"}])


def test_gtwriter_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "game1.jsonl")
        w = GTWriter(path)
        w.put(_rec(0))
        w.put(_rec(1))
        w.close()
        recs = list(read_records(path))
        assert [r.seq for r in recs] == [0, 1]
        assert recs[0].mjai == [{"type": "dahai", "pai": "5m"}]
        # header line present and skipped by read_records
        with open(path, encoding="utf-8") as fh:
            first = fh.readline()
        assert first.strip() == '{"_schema": 1}'


def test_gtwriter_next_seq_monotonic():
    with tempfile.TemporaryDirectory() as d:
        w = GTWriter(os.path.join(d, "g.jsonl"))
        assert [w.next_seq() for _ in range(3)] == [0, 1, 2]
        w.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_schema_writer OK")
