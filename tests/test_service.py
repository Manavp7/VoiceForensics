"""Tests for the production service layer (DB, storage, security, queue, webhooks)."""

from __future__ import annotations

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from voiceforensics.config import Settings
from voiceforensics.service.db import Job, create_api_key, get_sessionmaker, init_db
from voiceforensics.service.queue import ThreadJobQueue
from voiceforensics.service.security import RateLimiter, is_safe_url, validate_api_key
from voiceforensics.service.storage import LocalStorage
from voiceforensics.service.webhooks import deliver, sign_payload

# --- SSRF guard ---------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",
        "http://localhost/x",
        "http://10.0.0.5/x",
        "http://169.254.169.254/latest/meta-data",
        "ftp://example.com/x",
        "file:///etc/passwd",
    ],
)
def test_ssrf_blocks_internal(url):
    assert is_safe_url(url, settings=Settings(allow_private_url_fetch=False)) is False


def test_ssrf_allows_public_ip_literal():
    # 8.8.8.8 is a public literal (no DNS / network needed) → allowed.
    assert is_safe_url("http://8.8.8.8/audio.wav", settings=Settings()) is True


def test_ssrf_override_allows_private():
    assert is_safe_url("http://127.0.0.1/x", settings=Settings(allow_private_url_fetch=True)) is True


# --- rate limiter -------------------------------------------------------------


def test_rate_limiter_blocks_after_capacity():
    rl = RateLimiter(per_minute=2)
    assert rl.allow("k") is True
    assert rl.allow("k") is True
    assert rl.allow("k") is False
    assert rl.allow("other") is True  # independent bucket


# --- storage ------------------------------------------------------------------


def test_local_storage_roundtrip(tmp_path):
    s = LocalStorage(tmp_path / "store")
    s.put("reports/a.pdf", b"%PDF-1.4 hi")
    assert s.exists("reports/a.pdf")
    assert s.get("reports/a.pdf") == b"%PDF-1.4 hi"


def test_local_storage_blocks_traversal(tmp_path):
    s = LocalStorage(tmp_path / "store")
    with pytest.raises(ValueError):
        s.put("../escape.txt", b"nope")


# --- api keys -----------------------------------------------------------------


def test_api_key_create_and_validate():
    raw = create_api_key("test")
    assert raw.startswith("vfk_")
    assert validate_api_key(raw) is not None
    assert validate_api_key("vfk_invalid") is None


# --- webhook signing + delivery ----------------------------------------------


def test_sign_payload_deterministic():
    sig = sign_payload(b"{}", "secret")
    assert sig.startswith("sha256=")
    assert sig == sign_payload(b"{}", "secret")


class _CaptureHandler(BaseHTTPRequestHandler):
    received: dict = {}

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _CaptureHandler.received = {
            "body": body,
            "signature": self.headers.get("X-VoiceForensics-Signature"),
        }
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):  # silence
        pass


def test_webhook_delivery_to_local_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        ok = deliver(
            f"http://127.0.0.1:{port}/hook",
            {"job_id": "x", "status": "completed"},
            settings=Settings(allow_private_url_fetch=True, webhook_secret="s3cr3t"),
        )
        assert ok is True
        assert _CaptureHandler.received["signature"].startswith("sha256=")
        assert json.loads(_CaptureHandler.received["body"])["status"] == "completed"
    finally:
        server.shutdown()


def test_webhook_refuses_unsafe_url():
    assert deliver("http://127.0.0.1/x", {}, settings=Settings(allow_private_url_fetch=False)) is False


# --- async job lifecycle ------------------------------------------------------


def test_thread_queue_job_completes(genuine_wav):
    init_db()
    job_id = "job_" + uuid.uuid4().hex[:8]
    Session = get_sessionmaker()
    with Session() as session:
        session.add(Job(id=job_id, status="queued", analysis_type="quick"))
        session.commit()

    q = ThreadJobQueue()
    q.run_sync(job_id, genuine_wav.read_bytes(), "quick", ".wav")

    with Session() as session:
        job = session.get(Job, job_id)
        assert job.status == "completed"
        result = json.loads(job.result_json)
        assert 0.0 <= result["result"]["deepfake_probability"] <= 1.0


def test_job_failure_recorded():
    init_db()
    job_id = "job_" + uuid.uuid4().hex[:8]
    Session = get_sessionmaker()
    with Session() as session:
        session.add(Job(id=job_id, status="queued", analysis_type="quick"))
        session.commit()
    ThreadJobQueue().run_sync(job_id, b"not audio at all", "quick", ".wav")
    with Session() as session:
        job = session.get(Job, job_id)
        assert job.status == "failed"
        assert job.error
