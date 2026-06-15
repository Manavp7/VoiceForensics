"""Tests for segment-level localization and naturalness."""

from __future__ import annotations

import numpy as np

from voiceforensics.config import Settings
from voiceforensics.features.spectral import log_mel_spectrogram
from voiceforensics.localization.segment_scoring import localize, naturalness_score
from voiceforensics.models import build_detectors

SR = 16_000


def test_localize_flags_synthetic_half(spliced_signal):
    # spliced = genuine first half, synthetic second half.
    settings = Settings()
    detectors = build_detectors(settings)
    mel = log_mel_spectrogram(spliced_signal, SR)
    result = localize(detectors, spliced_signal, SR, mel, settings)

    assert len(result.segments) > 2
    mid_ms = len(spliced_signal) / SR * 1000 / 2
    first = [s.score for s in result.segments if s.end_ms <= mid_ms]
    second = [s.score for s in result.segments if s.start_ms >= mid_ms]
    assert first and second
    # The synthetic (second) half should score higher on average.
    assert np.mean(second) > np.mean(first)


def test_heatmap_shape(genuine_signal):
    settings = Settings()
    detectors = build_detectors(settings)
    mel = log_mel_spectrogram(genuine_signal, SR)
    result = localize(detectors, genuine_signal, SR, mel, settings)
    assert result.heatmap.shape[0] == len(result.segments)
    assert result.heatmap.shape[1] == mel.shape[0]
    assert np.all(result.heatmap >= 0.0)


def test_naturalness_genuine_higher_than_synthetic(genuine_signal, synthetic_signal):
    g = naturalness_score(genuine_signal, SR)
    s = naturalness_score(synthetic_signal, SR)
    assert 0.0 <= s <= 1.0
    assert 0.0 <= g <= 1.0
    assert g > s
