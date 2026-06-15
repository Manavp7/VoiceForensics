"""Fuzz / robustness tests: malformed input must fail gracefully (typed errors,
no uncaught crashes, and API 4xx rather than 500 where avoidable).
"""

from __future__ import annotations

import base64
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voiceforensics.api.app import app
from voiceforensics.audio.io import AudioDecodeError, decode_to_waveform
from voiceforensics.audio.preprocess import QualityGateError
from voiceforensics.pipeline import Engine

client = TestClient(app)
SR = 16_000


def test_random_bytes_rejected(tmp_path):
    p = tmp_path / "rand.wav"
    p.write_bytes(np.random.default_rng(0).bytes(4096))
    with pytest.raises(AudioDecodeError):
        decode_to_waveform(p, SR)


def test_truncated_wav_header(tmp_path):
    p = tmp_path / "trunc.wav"
    p.write_bytes(b"RIFF" + struct.pack("<I", 1000) + b"WAVE")  # header only, no data
    with pytest.raises((AudioDecodeError, QualityGateError)):
        Engine().analyze(p, analysis_type="quick")


def test_empty_file(tmp_path):
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    with pytest.raises(AudioDecodeError):
        decode_to_waveform(p, SR)


def test_too_short_rejected(tmp_path):
    import soundfile as sf

    p = tmp_path / "short.wav"
    sf.write(str(p), np.zeros(100, dtype=np.float32), SR)
    with pytest.raises(QualityGateError):
        Engine().analyze(p, analysis_type="quick")


def test_api_random_bytes_returns_400():
    b64 = base64.b64encode(np.random.default_rng(1).bytes(2048)).decode()
    resp = client.post("/v1/analyze", json={"audio_base64": b64, "analysis_type": "quick"})
    assert resp.status_code == 400


def test_api_huge_base64_rejected_or_handled():
    # A large but invalid payload should not 500.
    b64 = base64.b64encode(b"\x00" * 100_000).decode()
    resp = client.post("/v1/analyze", json={"audio_base64": b64, "analysis_type": "quick"})
    assert resp.status_code in (400, 422)


def test_visualize_invalid_kind(genuine_wav):
    with open(genuine_wav, "rb") as fh:
        resp = client.post("/v1/visualize/upload", files={"file": ("g.wav", fh, "audio/wav")},
                           data={"kind": "bogus"})
    assert resp.status_code == 400


def test_visualize_mel_returns_png(genuine_wav):
    with open(genuine_wav, "rb") as fh:
        resp = client.post("/v1/visualize/upload", files={"file": ("g.wav", fh, "audio/wav")},
                           data={"kind": "mel"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_nan_audio_handled(tmp_path):
    import soundfile as sf

    y = np.full(SR * 2, np.nan, dtype=np.float32)
    p = tmp_path / "nan.wav"
    sf.write(str(p), y, SR)
    # Decoding sanitises NaNs; the (silent) result is then quality-gated out.
    with pytest.raises(QualityGateError):
        Engine().analyze(p, analysis_type="quick")
