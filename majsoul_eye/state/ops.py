"""Pending-operation extraction from a GTRecord's raw liqi message.

Wire shape (verified on captures/raw/ai_session/run_13/game1.jsonl):
    raw_liqi["data"]["data"]["operation"] =
        {"seat": <hero>, "operationList": [{"type": N, "combination": [...], ...}],
         "timeAdd": ..., "timeFixed": ...}
type codes: 1=dapai(no button) 2=chi 3=pon 4/5/6=kan 7=riichi 8=tsumo 9=ron
10=kyushukyuhai 11=babei(3p). Mapping to button classes lives in majsoul_eye.hud.
Semantics: a record carrying operationList OFFERS ops to the hero; any later
record supersedes it — so BoardState.pending_ops (set by Replayer.apply_record
from the LATEST record) is exactly "ops pending at this snapshot".
"""
from __future__ import annotations

from typing import Optional


def ops_from_record(r) -> Optional[list[int]]:
    """liqi op type codes offered to the hero by this record, else None.

    None for: syncing records (reconnect replays re-send stale offers), records
    without raw_liqi, offers addressed to another seat (defensive; each client
    normally only receives its own), and empty operationList.
    """
    if getattr(r, "syncing", False) or not getattr(r, "raw_liqi", None):
        return None
    try:
        op = ((r.raw_liqi.get("data") or {}).get("data") or {}).get("operation") or {}
        ol = op.get("operationList") or []
        if not ol:
            return None
        seat = getattr(r, "seat", None)
        if "seat" in op and seat is not None and op["seat"] != seat:
            return None
        types = [int(o["type"]) for o in ol if "type" in o]
        return types or None
    except Exception:
        return None
