"""API tests using FastAPI's TestClient."""

from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from voiceforensics.api.app import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "heuristic" in body["active_detectors"]
    assert body["baseline_only"] is True


def test_analyze_base64(synthetic_wav):
    b64 = base64.b64encode(synthetic_wav.read_bytes()).decode()
    resp = client.post(
        "/v1/analyze",
        json={"audio_base64": b64, "analysis_type": "quick"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["analysis_id"].startswith("vf_")
    assert 0.0 <= body["result"]["deepfake_probability"] <= 1.0
    assert body["result"]["metadata"]["file_hash_sha256"]


def test_analyze_upload_full(genuine_wav):
    with open(genuine_wav, "rb") as fh:
        resp = client.post(
            "/v1/analyze/upload",
            files={"file": ("genuine.wav", fh, "audio/wav")},
            data={"analysis_type": "full"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["result"]["segments"]
    assert body["result"]["fingerprint"] is not None


def test_analyze_requires_input():
    resp = client.post("/v1/analyze", json={"analysis_type": "quick"})
    assert resp.status_code == 400


def test_analyze_invalid_base64():
    resp = client.post("/v1/analyze", json={"audio_base64": "!!!not-base64!!!"})
    assert resp.status_code == 400


def test_analyze_bad_audio_returns_400():
    b64 = base64.b64encode(b"this is not audio").decode()
    resp = client.post("/v1/analyze", json={"audio_base64": b64, "analysis_type": "quick"})
    assert resp.status_code == 400


def test_analyze_silence_returns_422(silence_wav):
    b64 = base64.b64encode(silence_wav.read_bytes()).decode()
    resp = client.post("/v1/analyze", json={"audio_base64": b64, "analysis_type": "quick"})
    assert resp.status_code == 422


def test_webhook_noted_as_noop_in_sync(synthetic_wav):
    b64 = base64.b64encode(synthetic_wav.read_bytes()).decode()
    resp = client.post(
        "/v1/analyze",
        json={
            "audio_base64": b64,
            "analysis_type": "quick",
            "webhook_url": "https://example.com/callback",
        },
    )
    assert resp.status_code == 200
    notes = resp.json()["provenance"]["notes"]
    assert any("webhook" in n for n in notes)


def test_async_job_lifecycle(genuine_wav):
    # Force inline job execution for determinism by patching the queue to run_sync.
    import voiceforensics.api.app as appmod
    from voiceforensics.service.queue import ThreadJobQueue

    class _InlineQueue(ThreadJobQueue):
        def enqueue(self, job_id, audio_bytes, analysis_type, suffix):
            self.run_sync(job_id, audio_bytes, analysis_type, suffix)

    appmod._queue = _InlineQueue()
    try:
        b64 = base64.b64encode(genuine_wav.read_bytes()).decode()
        resp = client.post("/v1/analyze", json={"audio_base64": b64, "analysis_type": "quick", "mode": "async"})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        assert resp.json()["status"] == "queued"

        job = client.get(f"/v1/jobs/{job_id}")
        assert job.status_code == 200
        body = job.json()
        assert body["status"] == "completed"
        assert 0.0 <= body["result"]["result"]["deepfake_probability"] <= 1.0
    finally:
        appmod._queue = None


def test_job_not_found():
    assert client.get("/v1/jobs/job_doesnotexist").status_code == 404


def test_report_endpoint_serves_pdf(synthetic_wav):
    import voiceforensics.api.app as appmod
    from voiceforensics.service.queue import ThreadJobQueue

    class _InlineQueue(ThreadJobQueue):
        def enqueue(self, job_id, audio_bytes, analysis_type, suffix):
            self.run_sync(job_id, audio_bytes, analysis_type, suffix)

    appmod._queue = _InlineQueue()
    try:
        b64 = base64.b64encode(synthetic_wav.read_bytes()).decode()
        resp = client.post("/v1/analyze", json={"audio_base64": b64, "analysis_type": "legal", "mode": "async"})
        job_id = resp.json()["job_id"]
        result = client.get(f"/v1/jobs/{job_id}").json()
        analysis_id = result["result"]["analysis_id"]
        pdf = client.get(f"/v1/reports/{analysis_id}.pdf")
        assert pdf.status_code == 200
        assert pdf.content[:5] == b"%PDF-"
    finally:
        appmod._queue = None


def test_report_not_found():
    assert client.get("/v1/reports/vf_nope.pdf").status_code == 404


def test_auth_required_when_enabled(monkeypatch, genuine_wav):
    import voiceforensics.api.app as appmod
    from voiceforensics.config import Settings
    from voiceforensics.service.db import create_api_key

    monkeypatch.setattr(appmod, "get_settings", lambda: Settings(require_auth=True))
    b64 = base64.b64encode(genuine_wav.read_bytes()).decode()
    # No key → 401.
    r = client.post("/v1/analyze", json={"audio_base64": b64, "analysis_type": "quick"})
    assert r.status_code == 401
    # Valid key → 200.
    key = create_api_key("apitest")
    r2 = client.post(
        "/v1/analyze",
        json={"audio_base64": b64, "analysis_type": "quick"},
        headers={"x-api-key": key},
    )
    assert r2.status_code == 200


def test_rate_limit_returns_429(genuine_wav):
    import voiceforensics.api.app as appmod
    from voiceforensics.service.security import RateLimiter

    appmod._rate_limiter = RateLimiter(per_minute=1)
    try:
        b64 = base64.b64encode(genuine_wav.read_bytes()).decode()
        first = client.post("/v1/analyze", json={"audio_base64": b64, "analysis_type": "quick"})
        second = client.post("/v1/analyze", json={"audio_base64": b64, "analysis_type": "quick"})
        assert first.status_code == 200
        assert second.status_code == 429
    finally:
        appmod._rate_limiter = None


def test_ssrf_rejected_via_api():
    r = client.post("/v1/analyze", json={"audio_url": "http://169.254.169.254/latest/meta-data"})
    assert r.status_code == 400
