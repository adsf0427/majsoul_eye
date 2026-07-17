"""Dev wrapper for the worker HTTP service.

The implementation lives in the package (``majsoul_eye/worker/serve.py``) so a
deployment that ships only ``majsoul_eye/`` — like the self-contained release
payload, which prunes ``scripts/`` — can start the worker with
``python -m majsoul_eye.worker``. This wrapper keeps the historical dev
invocation working unchanged.
"""
from majsoul_eye.worker.serve import main

if __name__ == "__main__":
    main()
