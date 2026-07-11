import threading

from fastapi.testclient import TestClient

from majsoul_eye.recognize.runtime import RuntimeFailure
from majsoul_eye.worker.app import create_app


class FakeRuntime:
    def metadata(self):
        return {"manifestVersion": "test-v1", "layoutId": "majsoul-desktop-16x9-v1",
                "detectorSha": "a" * 64, "classifierSha": "b" * 64,
                "hudReaderSha": "c" * 64, "eyeRevision": "rev",
                "supportStatus": "experimental"}

    def recognize_bytes(self, body, context):
        assert context.image_ref is None
        if body == b"bad":
            raise RuntimeFailure("INVALID_IMAGE", "cannot decode image")
        return {"schemaVersion": 1, "draft": {"draftId": context.draft_id},
                "issues": [], "recognizer": self.metadata()}

    def reconstruct_draft(self, draft, revision):
        return {"schemaVersion": 1, "revision": revision, "ok": False,
                "issues": [], "mjai": None, "heroSeatAbs": None,
                "fabricated": None, "historyBaseline": [],
                "selectedHistory": None, "decision": None}


def headers(allow="1"):
    return {"X-Request-ID": "req-1", "X-Draft-ID": "draft-1",
            "X-Image-SHA256": "f" * 64,
            "X-Layout-ID": "majsoul-desktop-16x9-v1",
            "X-Allow-Experimental": allow,
            "Content-Type": "application/octet-stream"}


def test_health_and_ready_are_separate():
    client = TestClient(create_app(FakeRuntime()))
    assert client.get("/healthz").json() == {"ok": True}
    ready = client.get("/readyz").json()
    assert ready["ready"] is True
    assert ready["recognizer"]["supportStatus"] == "experimental"


def test_recognize_forwards_raw_bytes_and_context():
    client = TestClient(create_app(FakeRuntime()))
    response = client.post("/v1/recognize", content=b"png", headers=headers())
    assert response.status_code == 200
    assert response.json()["draft"]["draftId"] == "draft-1"


def test_reconstruct_uses_exact_body_shape():
    client = TestClient(create_app(FakeRuntime()))
    response = client.post("/v1/reconstruct", headers={"X-Request-ID": "req-2"},
                           json={"draft": {"schemaVersion": 1}, "revision": 9})
    assert response.status_code == 200
    assert response.json()["revision"] == 9


def test_reconstruct_rejects_extra_body_keys():
    client = TestClient(create_app(FakeRuntime()))
    response = client.post("/v1/reconstruct", headers={"X-Request-ID": "req-2"},
                           json={"draft": {}, "revision": 9, "legacy": True})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_timed_out_inference_keeps_capacity_until_native_work_finishes():
    class BlockingRuntime(FakeRuntime):
        started = threading.Event()
        release = threading.Event()

        def recognize_bytes(self, body, context):
            self.started.set()
            self.release.wait(2.0)
            return super().recognize_bytes(body, context)

    runtime = BlockingRuntime()
    with TestClient(create_app(runtime, max_pending=1,
                               request_timeout_seconds=0.01)) as client:
        first = client.post("/v1/recognize", content=b"png", headers=headers())
        assert first.status_code == 504 and runtime.started.is_set()
        second = client.post("/v1/recognize", content=b"png", headers=headers())
        assert second.status_code == 429
        correction = client.post("/v1/reconstruct", headers={"X-Request-ID": "req-edit"},
                                 json={"draft": {}, "revision": 3})
        assert correction.status_code == 200  # edits are not queued behind GPU work
        runtime.release.set()


def test_runtime_failure_has_strict_worker_error_body():
    client = TestClient(create_app(FakeRuntime()))
    response = client.post("/v1/recognize", content=b"bad", headers=headers())
    assert response.status_code == 422
    assert response.json() == {"schemaVersion": 1, "error": {
        "code": "INVALID_IMAGE", "message": "cannot decode image",
        "requestId": "req-1"}}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_worker_api OK")
