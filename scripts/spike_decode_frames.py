"""Spike #3 — decode the liqi framing of captured frames into the ordered action
stream, with ZERO external deps (no protobuf, no GPL code).

Input: a frames.jsonl produced by `scripts/spike_ws_tap.py --dump DIR`.

It walks the liqi wire format ourselves (the framing is generic protobuf
wire-format parsing — a technique, not Majsoul IP):

  byte[0] = msg type   1=NOTIFY, 2=REQ, 3=RES
  NOTIFY  : 0x01 + Wrapper{ name=1:string, data=2:bytes }
            for name=='.lq.ActionPrototype', data is itself
            ActionPrototype{ step=1:varint, name=2:string ('ActionDiscardTile'...),
                             data=N:bytes }  ← inner action body is base64+XOR
  REQ     : 0x02 + msg_id(u16 LE) + Wrapper{ method=1:string, data=2:bytes }
  RES     : 0x03 + msg_id(u16 LE) + Wrapper{ (empty)=1, data=2:bytes }  (type by req)

We recover msg-type, method, and for ActionPrototype the inner action NAME + step
WITHOUT protobuf — proving the full ordered GT sequence is recoverable from the
captured bytes. (Tile-level fields like *which* tile need the proto schema; that
arrives with the vendored MIT liqi proto — see docs/DATA_AUTOMATION.md §2/§5.)

This script optionally XOR-decodes the inner action body just to confirm it is
well-formed protobuf (the well-known public liqi XOR keys; same in every MIT
majsoul tool). It does not interpret it.

Usage:
    PYTHONPATH=. $PY scripts/spike_decode_frames.py captures/wsdump
    PYTHONPATH=. $PY scripts/spike_decode_frames.py captures/wsdump/frames.jsonl --all
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
from collections import Counter

# Public liqi obfuscation keys (identical across majsoul_wrapper / MahjongRepository / etc.)
_XOR_KEYS = [0x84, 0x5e, 0x4e, 0x42, 0x39, 0xa2, 0x1f, 0x60, 0x1c]


def xor_decode(data: bytes) -> bytes:
    """Reverse the inner-ActionPrototype.data obfuscation."""
    out = bytearray(data)
    n = len(out)
    for i in range(n):
        u = (23 ^ n) + 5 * i + _XOR_KEYS[i % len(_XOR_KEYS)] & 255
        out[i] ^= u
    return bytes(out)


def parse_varint(buf: bytes, p: int) -> tuple[int, int]:
    val = shift = 0
    while p < len(buf):
        b = buf[p]
        val |= (b & 0x7F) << shift
        shift += 7
        p += 1
        if not (b & 0x80):
            break
    return val, p


def walk_protobuf(buf: bytes) -> list[dict]:
    """Generic protobuf wire-walk: returns [{id, wire, data}]. Tolerant: stops on
    anything it can't read rather than raising (captured frames may include fields
    we don't model)."""
    out, p = [], 0
    while p < len(buf):
        try:
            key, p = parse_varint(buf, p)
            field_id, wire = key >> 3, key & 7
            if wire == 0:                       # varint
                data, p = parse_varint(buf, p)
            elif wire == 2:                     # length-delimited (string/bytes/message)
                ln, p = parse_varint(buf, p)
                data = buf[p:p + ln]
                p += ln
            elif wire == 5:                     # 32-bit
                data = buf[p:p + 4]; p += 4
            elif wire == 1:                     # 64-bit
                data = buf[p:p + 8]; p += 8
            else:
                break
            out.append({"id": field_id, "wire": wire, "data": data})
        except Exception:
            break
    return out


def _first_string(fields: list[dict], pred=None) -> bytes | None:
    for f in fields:
        if f["wire"] == 2 and (pred is None or pred(f["data"])):
            return f["data"]
    return None


def _first_varint(fields: list[dict]) -> int | None:
    for f in fields:
        if f["wire"] == 0:
            return f["data"]
    return None


def decode_frame(raw: bytes) -> dict:
    """Return {type, method, action, step, inner_ok} for one liqi frame."""
    if not raw:
        return {"type": "empty"}
    mtype = raw[0]
    if mtype == 1:  # NOTIFY
        blocks = walk_protobuf(raw[1:])
        method = (_first_string(blocks) or b"").decode("ascii", "ignore")
        res = {"type": "NOTIFY", "method": method}
        if method == ".lq.ActionPrototype" and len(blocks) >= 2:
            ap = walk_protobuf(blocks[1]["data"])
            step = _first_varint(ap)
            name = _first_string(ap, lambda d: d[:6] == b"Action")
            body = _first_string(ap, lambda d: not d[:6] == b"Action")
            res["action"] = name.decode("ascii", "ignore") if name else "?"
            res["step"] = step
            if body:                      # confirm inner body is XOR-decodable protobuf
                try:                      # body is raw bytes off the wire (XOR'd), not base64
                    res["inner_ok"] = len(walk_protobuf(xor_decode(body))) > 0
                except Exception:
                    res["inner_ok"] = False
        return res
    elif mtype in (2, 3):  # REQ / RES
        msg_id = struct.unpack("<H", raw[1:3])[0]
        blocks = walk_protobuf(raw[3:])
        method = (_first_string(blocks) or b"").decode("ascii", "ignore")
        return {"type": "REQ" if mtype == 2 else "RES", "id": msg_id, "method": method}
    return {"type": f"raw{mtype}"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode liqi framing of captured WS frames (no protobuf).")
    ap.add_argument("path", help="frames.jsonl, or a dir containing it.")
    ap.add_argument("--all", action="store_true", help="Print every frame (default: skip heartbeats).")
    args = ap.parse_args()

    path = args.path
    if os.path.isdir(path):
        path = os.path.join(path, "frames.jsonl")
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")

    methods = Counter()
    actions = Counter()
    n = action_n = inner_ok = inner_bad = 0
    SKIP = (".lq.Lobby.heatbeat", ".lq.FastTest.checkNetworkDelay", "")

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("opcode") != "bin":
                continue
            raw = base64.b64decode(rec["b64"])
            d = decode_frame(raw)
            n += 1
            methods[d.get("method", d["type"])] += 1
            if d.get("type") == "NOTIFY" and d.get("method") == ".lq.ActionPrototype":
                action_n += 1
                actions[d.get("action", "?")] += 1
                if d.get("inner_ok") is True:
                    inner_ok += 1
                elif d.get("inner_ok") is False:
                    inner_bad += 1
                print(f"  [{rec['t']:7.1f}s] {rec.get('dir',''):4} "
                      f"ActionPrototype -> {d.get('action','?'):<20} step={d.get('step')}"
                      f"  inner_ok={d.get('inner_ok')}")
            elif args.all and d.get("method") not in SKIP:
                print(f"  [{rec['t']:7.1f}s] {rec.get('dir',''):4} {d['type']:6} {d.get('method','')}")

    print("\n" + "=" * 60)
    print(f"binary frames decoded : {n}")
    print(f"ActionPrototype (GT)  : {action_n}   inner XOR-decodable: {inner_ok} ok / {inner_bad} bad")
    print("action breakdown      : " + ", ".join(f"{a}×{c}" for a, c in actions.most_common()))
    print("top methods           : " + ", ".join(f"{m}×{c}" for m, c in methods.most_common(10)))
    print("=" * 60)
    ok = action_n > 0 and inner_bad == 0
    print(f"VERDICT: {'PASS — full ordered action stream recovered from captured bytes.' if ok else 'CHECK — see above (0 actions or XOR mismatch).'}")


if __name__ == "__main__":
    main()
