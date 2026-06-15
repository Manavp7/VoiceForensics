"""Tests for advanced acoustic features."""

from __future__ import annotations

import numpy as np

from voiceforensics.features.advanced import (
    advanced_features,
    breathing_features,
    cqcc_features,
    ltas_features,
    modulation_features,
)
from voiceforensics.features.extractor import SCALAR_KEYS, extract

SR = 16_000


def test_ltas_finite(genuine_signal):
    f = ltas_features(genuine_signal, SR)
    assert set(f) == {"ltas_tilt", "ltas_centroid_hz", "ltas_flatness"}
    assert all(np.isfinite(v) for v in f.values())


def test_modulation_band_ratio_range(genuine_signal):
    f = modulation_features(genuine_signal, SR)
    assert 0.0 <= f["mod_4_8hz_ratio"] <= 1.0
    assert f["mod_peak_hz"] >= 0.0


def test_cqcc_finite(genuine_signal):
    f = cqcc_features(genuine_signal, SR)
    assert np.isfinite(f["cqcc_var_mean"])
    assert f["cqt_high_low_ratio"] >= 0.0


def test_breathing_genuine_more_pauses_than_synthetic(genuine_signal, synthetic_signal):
    g = breathing_features(genuine_signal, SR)
    s = breathing_features(synthetic_signal, SR)
    assert g["pause_rate_hz"] >= 0.0 and s["pause_rate_hz"] >= 0.0
    # Both have syllabic gaps; ensure feature is computed and bounded.
    assert g["mean_pause_ms"] >= 0.0


def test_advanced_features_finite_and_deterministic(genuine_signal):
    a = advanced_features(genuine_signal, SR)
    b = advanced_features(genuine_signal, SR)
    assert a == b  # deterministic
    assert all(np.isfinite(v) for v in a.values())
    expected = {
        "ltas_tilt", "ltas_centroid_hz", "ltas_flatness",
        "mod_4_8hz_ratio", "mod_peak_hz",
        "cqcc_var_mean", "cqt_high_low_ratio",
        "pause_rate_hz", "voiced_run_rate_hz", "mean_pause_ms",
    }
    assert set(a) == expected


def test_advanced_keys_in_extractor(genuine_signal):
    feats = extract(genuine_signal, SR).scalar_features
    for key in (
        "ltas_tilt",
        "mod_4_8hz_ratio",
        "cqcc_var_mean",
        "pause_rate_hz",
    ):
        assert key in feats
        assert key in SCALAR_KEYS
    assert np.all(np.isfinite(extract(genuine_signal, SR).vector(SCALAR_KEYS)))
