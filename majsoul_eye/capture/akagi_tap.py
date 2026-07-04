"""Non-invasive GT recorder: tee Akagi's MajsoulBridge stream to a JSONL file.

How it works
------------
``MajsoulBridge.parse_liqi(self, liqi_message)`` is the one place that receives
the already-parsed *raw liqi dict* (the superset) and returns the derived MJAI
events, with ``self.seat`` / ``self.last_op_step`` / ``self.syncing`` already
updated. We monkeypatch it (the class method, before any flow opens) to append a
:class:`GTRecord` per call. We do NOT call ``liqi_proto.parse`` ourselves —
that parser is stateful (req/res id matching), so re-parsing would corrupt it.

Records are written from a background thread so file I/O never stalls the MITM
proxy thread (``parse_liqi`` runs under Akagi's ``bridge_lock``).

Activation
----------
1. Launcher (recommended): ``python scripts/capture/record_gt.py`` — sets up sys.path,
   installs the tap, then runs Akagi's ``main()``.
2. Or add two lines to Akagi's ``run_akagi.py`` before ``main()``::

       from majsoul_eye.capture.akagi_tap import install
       install("captures/session.jsonl")

Then play / 观战 a game (autoplay OFF) with the client routed through Akagi's
MITM. Call :func:`uninstall` (or just exit) to flush and close.
"""

from __future__ import annotations

import atexit
import time
from typing import Any, Optional

from .schema import GTRecord, GTWriter


# --- monkeypatch state ------------------------------------------------------

_writer: GTWriter | None = None
_orig_parse_liqi = None
_recorded_count = 0
_syncer = None  # optional FrameSyncer (screenshots); only set when --screenshots


def _extract_method_and_name(liqi_message: dict[str, Any] | None) -> tuple[str | None, str | None]:
    if not isinstance(liqi_message, dict):
        return None, None
    method = liqi_message.get("method")
    action_name = None
    data = liqi_message.get("data")
    if isinstance(data, dict):
        action_name = data.get("name")  # e.g. 'ActionDiscardTile' for ActionPrototype
    return method, action_name


def install(path: str, akagi_dir: str | None = None, syncer=None) -> GTWriter:
    """Patch ``MajsoulBridge.parse_liqi`` to record GT to ``path``. Idempotent.

    Args:
        path: output JSONL path.
        akagi_dir: optional path to the Akagi repo to add to ``sys.path``. If
            None, assumes Akagi is already importable.
        syncer: optional ``FrameSyncer`` — if given, each board-changing message
            triggers an asynchronous screenshot tagged with ``last_op_step``.
    """
    global _writer, _orig_parse_liqi, _syncer
    _syncer = syncer

    if akagi_dir:
        import sys
        if akagi_dir not in sys.path:
            sys.path.insert(0, akagi_dir)

    from mitm.bridge.majsoul.bridge import MajsoulBridge  # type: ignore

    if _orig_parse_liqi is not None:
        # already installed; just (re)point the writer
        if _writer is not None:
            _writer.close()
        _writer = GTWriter(path)
        return _writer

    _writer = GTWriter(path)
    _orig_parse_liqi = MajsoulBridge.parse_liqi

    def patched_parse_liqi(self, liqi_message):  # noqa: ANN001
        global _recorded_count
        result = _orig_parse_liqi(self, liqi_message)
        try:
            w = _writer
            if w is not None and liqi_message is not None:
                method, action_name = _extract_method_and_name(liqi_message)
                rec = GTRecord(
                    seq=w.next_seq(),
                    ts=time.time(),
                    flow_id=getattr(self, "_me_flow_id", ""),
                    seat=getattr(self, "seat", -1),
                    last_op_step=getattr(self, "last_op_step", -1),
                    syncing=getattr(self, "syncing", False),
                    method=method,
                    action_name=action_name,
                    raw_liqi=liqi_message,
                    mjai=list(result) if result else [],
                )
                w.put(rec)
                _recorded_count += 1
                if _syncer is not None:
                    # Tag screenshots with the GLOBAL record seq (unique across
                    # kyoku), NOT last_op_step (which resets each kyoku → frame
                    # filename collisions / wrong GT joins).
                    _syncer.note(result, rec.seq)
        except Exception:
            # never let recording break the bridge
            pass
        return result

    MajsoulBridge.parse_liqi = patched_parse_liqi  # type: ignore[assignment]
    atexit.register(uninstall)
    return _writer


def uninstall() -> None:
    """Restore the original method and flush the writer."""
    global _writer, _orig_parse_liqi
    if _orig_parse_liqi is not None:
        try:
            from mitm.bridge.majsoul.bridge import MajsoulBridge  # type: ignore
            MajsoulBridge.parse_liqi = _orig_parse_liqi  # type: ignore[assignment]
        except Exception:
            pass
        _orig_parse_liqi = None
    if _writer is not None:
        _writer.close()
        _writer = None


def recorded_count() -> int:
    return _recorded_count
