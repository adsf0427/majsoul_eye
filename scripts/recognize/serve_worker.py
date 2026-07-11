from __future__ import annotations

import argparse
import os

from majsoul_eye.recognize.runtime import RecognitionRuntime
from majsoul_eye.worker import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eye-revision", default=os.environ.get("EYE_REVISION", ""))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-pending", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-reconstruct-pending", type=int, default=32)
    parser.add_argument("--reconstruct-concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.eye_revision:
        parser.error("--eye-revision or EYE_REVISION is required")
    runtime = RecognitionRuntime.from_manifest(
        args.manifest, device=args.device, eye_revision=args.eye_revision,
        evaluation_mode=False)
    runtime.warmup()
    print(runtime.metadata(), flush=True)
    if args.check_only:
        return
    import uvicorn
    app = create_app(runtime, max_pending=args.max_pending,
                     inference_concurrency=args.concurrency,
                     max_reconstruct_pending=args.max_reconstruct_pending,
                     reconstruct_concurrency=args.reconstruct_concurrency,
                     request_timeout_seconds=args.timeout)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
