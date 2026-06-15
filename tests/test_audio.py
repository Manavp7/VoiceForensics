"""Tests for audio decoding, preprocessing, and segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from voiceforensics.audio.io import AudioDecodeError, decode_to_waveform
from voiceforensics.audio.preprocess import (
    QualityGateError,
    analyze_quality,
    peak_normalize,
    preprocess,
)
from voiceforensics.audio.segment import energy_segments, frame_windows

SR = 16_000


def test_decode_wav(genuine_wav):
    y, sr = decode_to_waveform(genuine_wav, SR)
    assert sr == SR
    assert y.ndim == 1
    assert y.dtype == np.float32
    assert 2.5 * SR < len(y) < 3.5 * SR


def test_decode_missing_file(tmp_path):
    with pytest.raises(AudioDecodeError):
        decode_to_waveform(tmp_path / "nope.wav", SR)


def test_decode_all_formats(encoded_files):
    if not encoded_files:
        pytest.skip("ffmpeg not available")
    for ext, path in encoded_files.items():
        y, sr = decode_to_waveform(path, SR)
        assert sr == SR, ext
        assert len(y) > SR, ext  # at least ~1s decoded


def test_peak_normalize():
    y = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    out = peak_normalize(y, peak=0.5)
    assert np.isclose(np.max(np.abs(out)), 0.5, atol=1e-6)


def test_snr_ordering(genuine_signal):
    clean = analyze_quality(genuine_signal, SR)
    noisy = analyze_quality(
        genuine_signal + np.random.default_rng(0).normal(0, 0.3, len(genuine_signal)).astype(
            np.float32
        ),
        SR,
    )
    assert clean.snr_db > noisy.snr_db


def test_quality_gate_passes_genuine(genuine_signal):
    y, report = preprocess(
        genuine_signal, SR, min_snr_db=5.0, min_duration_s=0.4, max_duration_s=1800.0
    )
    assert report.duration_s > 2.5
    assert report.snr_db > 5.0


def test_quality_gate_rejects_silence():
    silent = np.zeros(SR, dtype=np.float32)
    with pytest.raises(QualityGateError):
        preprocess(silent, SR, min_snr_db=5.0, min_duration_s=0.4, max_duration_s=1800.0)


def test_quality_gate_rejects_too_short(genuine_signal):
    with pytest.raises(QualityGateError):
        preprocess(
            genuine_signal[: int(0.1 * SR)],
            SR,
            min_snr_db=5.0,
            min_duration_s=0.4,
            max_duration_s=1800.0,
        )


def test_frame_windows(genuine_signal):
    wins = frame_windows(genuine_signal, SR, window_ms=500, hop_ms=250)
    assert len(wins) > 1
    assert wins[0].start_ms == 0
    # Windows are ordered and within bounds.
    for w in wins:
        assert w.end_sample <= len(genuine_signal)
        assert w.start_sample < w.end_sample


def test_energy_segments_detects_activity(genuine_signal):
    segs = energy_segments(genuine_signal, SR)
    assert len(segs) >= 1
    for s in segs:
        assert s.end_sample > s.start_sample
