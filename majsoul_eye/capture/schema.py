"""Capture record schema + JSONL I/O.

A capture is a JSONL file: one :class:`GTRecord` per line, in arrival order.
Each record pairs Akagi's raw-liqi message (the *superset*) with the MJAI
events it derived, plus the bridge sync key (``last_op_step``) needed to align
a screenshot to this game tick (see ``docs/DESIGN.md`` §3.2).
"""

from __future__ import annotations

import base64
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Iterator


SCHEMA_VERSION = 1


@dataclass
class GTRecord:
    """One ground-truth tick captured from Akagi's MajsoulBridge.

    Attributes:
        seq: Monotonic counter within a capture session (0-based).
        ts: Wall-clock time of capture (``time.time()``).
        flow_id: MITM WebSocket flow id (one game = one flow; changes on reconnect).
        seat: Hero's absolute seat (0-3) as known to the bridge.
        last_op_step: Majsoul's monotonic step counter — the screenshot sync key.
        syncing: True if this came from a syncGame/enterGame reconnection replay
            (events here are a re-send of history; do NOT double-count).
        method: liqi method, e.g. '.lq.ActionPrototype', '.lq.FastTest.authGame'.
        action_name: ActionPrototype name if any, e.g. 'ActionDiscardTile'.
        raw_liqi: the full parsed liqi message dict (superset GT).
        mjai: the list of MJAI event dicts the bridge derived from this message.
    """

    seq: int
    ts: float
    flow_id: str
    seat: int
    last_op_step: int
    syncing: bool
    method: str | None
    action_name: str | None
    raw_liqi: dict[str, Any] | None
    mjai: list[dict[str, Any]]
    # Optional: filename of the screenshot aligned to this tick (filled in P2).
    screenshot: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json_line(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False, default=_json_default)

    @classmethod
    def from_json_line(cls, line: str) -> "GTRecord":
        d = json.loads(line)
        d.pop("_schema", None)
        # tolerate older/newer records missing optional fields
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def _json_default(o: Any) -> Any:
    """Make liqi messages JSON-safe (protobuf dicts may hold bytes)."""
    if isinstance(o, (bytes, bytearray)):
        return {"__bytes_b64__": base64.b64encode(bytes(o)).decode("ascii")}
    try:
        return str(o)
    except Exception:
        return None


def write_records(path: str, records: list[GTRecord]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_schema": SCHEMA_VERSION}) + "\n")
        for r in records:
            f.write(r.to_json_line() + "\n")


def read_records(path: str) -> Iterator[GTRecord]:
    """Yield records from a capture JSONL, skipping the schema header line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_schema" in obj and len(obj) == 1:
                continue  # header
            yield GTRecord.from_json_line(line)
