"""Spike #3b — FULL decode of the captured GT stream (inner action fields:
which tile, which seat, tsumogiri, etc.), using the generated liqi protobuf.

Builds on scripts/spike_decode_frames.py (framing walk + XOR) but additionally
protobuf-decodes the inner ActionPrototype body, so we see the actual game truth:
`ActionDealTile seat=2 tile=5p`, `ActionDiscardTile seat=0 tile=1m moqie=True`...

It needs ONLY the generated message classes (liqi_pb2) — NOT MahjongCopilot's
liqi.py parser, common.utils, or liqi.json (those are only for REQ/RES typing).
So the GT decode is clean-room: our framing + the public XOR + the schema's
generated classes. (Dev-time: liqi_pb2 is reused locally from MahjongCopilot;
swap to the MIT MahjongRepository proto before any release — docs/DATA_AUTOMATION.md §4.)

Usage:
    & $PY scripts/spike_decode_full.py captures/wsdump
    & $PY scripts/spike_decode_full.py captures/wsdump --proto-dir ../MahjongCopilot
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import spike_decode_frames as F  # framing walk + xor_decode  # noqa: E402


def message_to_dict(msg):
    """MessageToDict across protobuf 3/4 (including_default_value_fields) and
    5+ (always_print_fields_with_no_presence)."""
    from google.protobuf.json_format import MessageToDict
    try:
        return MessageToDict(msg, including_default_value_fields=True, preserving_proto_field_name=True)
    except TypeError:
        return MessageToDict(msg, always_print_fields_with_no_presence=True, preserving_proto_field_name=True)


# compact fields worth echoing per action, if present
_INTERESTING = ("seat", "tile", "pai", "doras", "dora", "moqie", "tsumogiri",
                "is_liqi", "zhenting", "type", "froms", "tiles", "operation")


def summarize(name: str, d: dict) -> str:
    bits = []
    for k in _INTERESTING:
        if k in d and d[k] not in ([], "", None):
            v = d[k]
            if isinstance(v, (list, dict)):
                v = json.dumps(v, ensure_ascii=False)
                if len(v) > 60:
                    v = v[:57] + "..."
            bits.append(f"{k}={v}")
    return f"{name:<20} " + " ".join(bits)


def main() -> None:
    ap = argparse.ArgumentParser(description="Full protobuf decode of the captured GT action stream.")
    ap.add_argument("path", help="frames.jsonl, or a dir containing it.")
    ap.add_argument("--proto-dir", default="../MahjongCopilot",
                    help="Dir containing liqi_proto/liqi_pb2.py (default ../MahjongCopilot).")
    ap.add_argument("--limit", type=int, default=0, help="Only decode the first N ActionPrototype frames (0=all).")
    args = ap.parse_args()

    proto_dir = os.path.abspath(args.proto_dir)
    sys.path.insert(0, proto_dir)
    try:
        from liqi_proto import liqi_pb2 as pb
    except Exception as e:
        sys.exit(f"could not import liqi_pb2 from {proto_dir}\n  {type(e).__name__}: {e}\n"
                 f"  (need `pip install protobuf` and liqi_proto/liqi_pb2.py present)")

    path = args.path
    if os.path.isdir(path):
        path = os.path.join(path, "frames.jsonl")
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")

    actions = Counter()
    seen_first = set()
    n = ok = bad = 0

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or json.loads(line).get("opcode") != "bin":
                continue
            rec = json.loads(line)
            raw = base64.b64decode(rec["b64"])
            if not raw or raw[0] != 1:                 # only NOTIFY here
                continue
            blocks = F.walk_protobuf(raw[1:])
            method = (F._first_string(blocks) or b"").decode("ascii", "ignore")
            if method != ".lq.ActionPrototype" or len(blocks) < 2:
                continue
            ap_fields = F.walk_protobuf(blocks[1]["data"])
            name_b = F._first_string(ap_fields, lambda d: d[:6] == b"Action")
            body = F._first_string(ap_fields, lambda d: d[:6] != b"Action")
            if not name_b:
                continue
            name = name_b.decode("ascii", "ignore")
            n += 1
            if args.limit and n > args.limit:
                break
            try:
                cls = getattr(pb, name)
                inner = cls.FromString(F.xor_decode(body)) if body else cls()
                d = message_to_dict(inner)
                actions[name] += 1
                ok += 1
                print(f"  [{rec['t']:7.1f}s] {summarize(name, d)}")
                if name not in seen_first:              # dump the full first instance of each type
                    seen_first.add(name)
                    print(f"          FULL {name}: {json.dumps(d, ensure_ascii=False)[:400]}")
            except Exception as e:
                bad += 1
                print(f"  [{rec['t']:7.1f}s] {name}: DECODE FAIL {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print(f"ActionPrototype decoded : {ok} ok / {bad} fail (of {n} seen)")
    print("action breakdown        : " + ", ".join(f"{a}:{c}" for a, c in actions.most_common()))
    print("=" * 60)
    print(f"VERDICT: {'PASS — full GT fields decoded from real captured frames.' if ok and not bad else 'CHECK — see failures above.'}")


if __name__ == "__main__":
    main()
