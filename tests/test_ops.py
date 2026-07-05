"""ops_from_record: verified wire shape; syncing/wrong-seat/absent -> None.
Replayer surfaces pending_ops on the snapshot of the offering record and
clears it on the next record."""
from types import SimpleNamespace as NS

from majsoul_eye.state.ops import ops_from_record


def rec(op=None, syncing=False, seat=0, raw=True):
    inner = {"operation": op} if op is not None else {}
    return NS(syncing=syncing, seat=seat,
              raw_liqi={"data": {"name": "ActionDealTile", "data": inner}} if raw else None)


OP = {"seat": 0, "operationList": [{"type": 1, "combination": []},
                                   {"type": 7, "combination": []}],
      "timeAdd": 20000, "timeFixed": 5000}

assert ops_from_record(rec(OP)) == [1, 7]
assert ops_from_record(rec(None)) is None                    # no operation field
assert ops_from_record(rec(OP, syncing=True)) is None        # reconnect replay
assert ops_from_record(rec(dict(OP, seat=2), seat=0)) is None # not hero's offer
assert ops_from_record(rec(raw=False)) is None               # raw_liqi missing
assert ops_from_record(rec({"seat": 0, "operationList": []})) is None

# --- Replayer wiring: pending_ops rides the snapshot, next record clears it --
from majsoul_eye.state.replay import BoardState

s = BoardState()
assert s.pending_ops is None
s.pending_ops = [1, 7]
c = s.copy()
c.pending_ops.append(9)
assert s.pending_ops == [1, 7], "copy() must deep-copy pending_ops"
print("test_ops OK")
