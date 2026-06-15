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


def test_webhook_noted_as_noop(synthetic_wav):
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
