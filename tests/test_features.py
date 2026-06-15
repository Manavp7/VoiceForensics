"""Tests for feature extraction."""

from __future__ import annotations

import numpy as np

from voiceforensics.features.codec import codec_features
from voiceforensics.features.extractor import SCALAR_KEYS, extract
from voiceforensics.features.formants import formant_stats, formant_tracks
from voiceforensics.features.pitch import f0_contour, jitter_stats
from voiceforensics.features.spectral import log_mel_spectrogram, mfcc_features

SR = 16_000


def test_mel_shape(genuine_signal):
    mel = log_mel_spectrogram(genuine_signal, SR)
    assert mel.shape[0] == 128
    assert mel.shape[1] > 10


def test_mfcc_shape(genuine_signal):
    mfcc = mfcc_features(genuine_signal, SR)
    assert mfcc.shape[0] == 39  # 13 + delta + delta2


def test_f0_detects_fundamental(genuine_signal):
    pitch = f0_contour(genuine_signal, SR)
    stats = jitter_stats(pitch)
    # genuine_like synthesised around 120 Hz.
    assert 90 < stats["f0_mean"] < 160
    assert stats["voiced_ratio"] > 0.2


def test_jitter_genuine_higher_than_synthetic(genuine_signal, synthetic_signal):
    g = jitter_stats(f0_contour(genuine_signal, SR))
    s = jitter_stats(f0_contour(synthetic_signal, SR))
    # The genuine proxy has more pitch micro-variation than the synthetic proxy.
    assert g["jitter_local"] >= s["jitter_local"]


def test_formants_ordered(genuine_signal):
    tracks = formant_tracks(genuine_signal, SR)
    stats = formant_stats(tracks)
    means = [stats["f1_mean"], stats["f2_mean"], stats["f3_mean"]]
    means = [m for m in means if m > 0]
    assert means == sorted(means)


def test_codec_bandlimit_detected(genuine_signal, synthetic_signal):
    g = codec_features(genuine_signal, SR)
    s = codec_features(synthetic_signal, SR)
    # synthetic_like is hard band-limited near 7 kHz → lower effective bandwidth
    # and less high-frequency energy.
    assert s["effective_bandwidth_hz"] <= g["effective_bandwidth_hz"]
    assert s["hf_energy_ratio_7k"] <= g["hf_energy_ratio_7k"]


def test_extract_bundle_complete(genuine_signal):
    bundle = extract(genuine_signal, SR)
    assert bundle.mel.shape[0] == 128
    assert set(SCALAR_KEYS).issubset(bundle.scalar_features.keys())
    vec = bundle.vector(SCALAR_KEYS)
    assert vec.shape == (len(SCALAR_KEYS),)
    assert np.all(np.isfinite(vec))


def test_extract_deterministic(genuine_signal):
    a = extract(genuine_signal, SR).scalar_features
    b = extract(genuine_signal, SR).scalar_features
    for k in SCALAR_KEYS:
        assert a[k] == b[k], k
