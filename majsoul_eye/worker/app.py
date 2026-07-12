from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from majsoul_eye.recognize.runtime import RecognitionContext, RuntimeFailure
from majsoul_eye.what_cut.schema import DraftSchemaError

logger = logging.getLogger(__name__)


class WorkerBusy(RuntimeError):
    pass


class CapacityGate:
    def __init__(self, max_pending: int, concurrency: int):
        self.max_pending = max_pending
        self.pending = 0
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(concurrency)

    @asynccontextmanager
    async def slot(self):
        async with self.lock:
            if self.pending >= self.max_pending:
                raise WorkerBusy("recognition queue is full")
            self.pending += 1
        try:
            async with self.semaphore:
                yield
        finally:
            async with self.lock:
                self.pending -= 1


def _error(status: int, code: str, message: str, request_id: str):
    return JSONResponse(status_code=status, content={"schemaVersion": 1,
        "error": {"code": code, "message": message, "requestId": request_id}})


_STATUS = {"INVALID_IMAGE": 422, "INVALID_IMAGE_DIGEST": 422,
           "INVALID_DRAFT": 422, "UNSUPPORTED_SCHEMA": 422,
           "UNSUPPORTED_LAYOUT": 422,
           # Localization outcomes. All 422: the request was well-formed, the
           # screenshot just isn't one we can read. Each is a DISTINCT code so
           # the caller can tell the user what to do about it.
           "IMAGE_TOO_SMALL": 422, "LOCALIZATION_FAILED": 422,
           "HAND_NOT_VISIBLE": 422, "BOARD_TOO_SMALL": 422,
           "BOARD_CLIPPED": 422,
           "MODEL_MANIFEST_MISMATCH": 503, "MODEL_UNAVAILABLE": 503}


def _parse_board_rect(raw: str):
    """``X-Board-Rect: ox,oy,bw,bh`` (source-image px) -> tuple, or None.

    Carried as a header, not a draft field: the draft schema is key-exact on
    three sides (worker, API, browser) and the client mails the whole draft back
    on every reconstruct, so a new draft field would 422 the EDIT path against an
    older worker. Reconstruct never needs the image, so the rect never needs to
    outlive recognition.
    """
    if not raw:
        return None
    parts = raw.split(",")
    if len(parts) != 4:
        raise ValueError("board rect must be ox,oy,bw,bh")
    ox, oy, bw, bh = (int(part) for part in parts)
    if bw <= 0 or bh <= 0:
        raise ValueError("board rect must have positive extent")
    return ox, oy, bw, bh


def _consume_background(task: asyncio.Task) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        pass


async def _run_bounded(gate: CapacityGate, function, *args):
    async with gate.slot():
        return await run_in_threadpool(function, *args)


async def _invoke(gate: CapacityGate, timeout: float, function, *args):
    task = asyncio.create_task(_run_bounded(gate, function, *args))
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except asyncio.TimeoutError:
        # Native Torch/OpenCV work cannot be cancelled safely. The background
        # task retains its gate slot until the thread really exits.
        task.add_done_callback(_consume_background)
        raise


def create_app(runtime, *, max_pending: int = 8, inference_concurrency: int = 1,
               max_reconstruct_pending: int = 32, reconstruct_concurrency: int = 4,
               request_timeout_seconds: float = 30.0) -> FastAPI:
    app = FastAPI(title="majsoul-eye-what-cut-worker", docs_url=None,
                  redoc_url=None, openapi_url=None)
    if (max_pending < 1 or inference_concurrency < 1
            or max_reconstruct_pending < 1 or reconstruct_concurrency < 1
            or request_timeout_seconds <= 0):
        raise ValueError("capacity and timeout settings must be positive")
    recognition_gate = CapacityGate(max_pending, inference_concurrency)
    reconstruct_gate = CapacityGate(max_reconstruct_pending, reconstruct_concurrency)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/readyz")
    async def readyz():
        return {"ready": True, "recognizer": runtime.metadata()}

    @app.post("/v1/recognize")
    async def recognize_endpoint(request: Request):
        request_id = request.headers.get("X-Request-ID", "")
        required = {name: request.headers.get(name, "") for name in
                    ("X-Draft-ID", "X-Image-SHA256", "X-Layout-ID")}
        if not request_id or not all(required.values()):
            return _error(400, "INVALID_REQUEST", "missing worker context header",
                          request_id)
        content_type = request.headers.get("content-type", "").split(";", 1)[0]
        if content_type not in ("image/png", "image/jpeg", "image/webp",
                                "application/octet-stream"):
            return _error(400, "INVALID_REQUEST", "unsupported Content-Type",
                          request_id)
        allow = request.headers.get("X-Allow-Experimental", "0")
        digest = required["X-Image-SHA256"]
        if (allow not in ("0", "1") or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)):
            return _error(400, "INVALID_REQUEST", "invalid worker context header",
                          request_id)
        try:
            board_rect = _parse_board_rect(request.headers.get("X-Board-Rect", ""))
        except ValueError:
            return _error(400, "INVALID_REQUEST", "invalid X-Board-Rect", request_id)
        context = RecognitionContext(
            request_id, required["X-Draft-ID"], required["X-Image-SHA256"],
            required["X-Layout-ID"],
            allow == "1", None, board_rect)
        body = await request.body()
        try:
            return await _invoke(recognition_gate, request_timeout_seconds,
                                 runtime.recognize_bytes, body, context)
        except WorkerBusy as exc:
            return _error(429, "WORKER_BUSY", str(exc), request_id)
        except asyncio.TimeoutError:
            return _error(504, "RECOGNITION_TIMEOUT", "worker request timed out",
                          request_id)
        except RuntimeFailure as exc:
            return _error(_STATUS.get(exc.code, 500), exc.code, str(exc), request_id)
        except Exception:
            logger.exception("unhandled recognize failure request_id=%s", request_id)
            return _error(500, "INTERNAL_ERROR", "internal worker error", request_id)

    @app.post("/v1/reconstruct")
    async def reconstruct_endpoint(request: Request):
        request_id = request.headers.get("X-Request-ID", "")
        if not request_id:
            return _error(400, "INVALID_REQUEST", "missing X-Request-ID", "")
        try:
            payload = await request.json()
            if not isinstance(payload, dict) or set(payload) != {"draft", "revision"}:
                raise TypeError("body keys must be exact")
            draft, revision = payload["draft"], payload["revision"]
            if type(revision) is not int:
                raise TypeError("revision must be integer")
        except (KeyError, TypeError, ValueError):
            return _error(400, "INVALID_REQUEST", "body must contain draft and integer revision",
                          request_id)
        try:
            return await _invoke(reconstruct_gate, request_timeout_seconds,
                                 runtime.reconstruct_draft, draft, revision)
        except WorkerBusy as exc:
            return _error(429, "WORKER_BUSY", str(exc), request_id)
        except asyncio.TimeoutError:
            return _error(504, "RECONSTRUCTION_TIMEOUT", "worker request timed out",
                          request_id)
        except RuntimeFailure as exc:
            return _error(_STATUS.get(exc.code, 500), exc.code, str(exc), request_id)
        except DraftSchemaError as exc:
            return _error(422, exc.code, str(exc), request_id)
        except Exception:
            logger.exception("unhandled reconstruct failure request_id=%s", request_id)
            return _error(500, "INTERNAL_ERROR", "internal worker error", request_id)

    return app
