"""Tests for the model-fingerprint matcher."""

from __future__ import annotations

from voiceforensics.config import Settings
from voiceforensics.features.extractor import FeatureBundle, extract
from voiceforensics.fingerprint.matcher import match
from voiceforensics.fingerprint.signatures import load_signature_db

SR = 16_000


def _bundle_from_scalars(scalars: dict) -> FeatureBundle:
    import numpy as np

    return FeatureBundle(
        sample_rate=SR,
        waveform=np.zeros(1, dtype="float32"),
        mel=np.zeros((128, 1), dtype="float32"),
        mfcc=np.zeros((39, 1), dtype="float32"),
        formants=np.zeros((4, 1), dtype="float32"),
        scalar_features=scalars,
    )


def test_signature_db_loads():
    settings = Settings()
    db = load_signature_db(settings.signature_db_path)
    names = [t.source_name for t in db.templates]
    assert "GENUINE" in names
    assert "ElevenLabs_v3" in names
    assert len(db.feature_keys) >= 5


def test_distribution_sums_to_one():
    settings = Settings()
    db = load_signature_db(settings.signature_db_path)
    # Use the ElevenLabs centroid as the query → should attribute to itself.
    eleven = next(t for t in db.templates if t.source_name == "ElevenLabs_v3")
    fp = match(_bundle_from_scalars(dict(eleven.centroid)), settings)
    assert abs(sum(fp.distribution.values()) - 1.0) < 1e-3
    assert fp.probable_source == "ElevenLabs_v3"


def test_synthetic_signal_attributed_to_a_synthetic_source(synthetic_signal):
    settings = Settings()
    fp = match(extract(synthetic_signal, SR), settings)
    assert fp.probable_source != "GENUINE"
    assert fp.probable_source != "UNKNOWN"
    assert 0.0 <= fp.confidence <= 1.0


def test_genuine_centroid_attributed_to_genuine():
    settings = Settings()
    db = load_signature_db(settings.signature_db_path)
    genuine = next(t for t in db.templates if t.source_name == "GENUINE")
    fp = match(_bundle_from_scalars(dict(genuine.centroid)), settings)
    assert fp.probable_source == "GENUINE"


def test_far_out_input_is_unknown():
    settings = Settings()
    weird = {
        "bandlimit_ratio": 50.0,
        "hf_energy_ratio_4k": 50.0,
        "jitter_local": 50.0,
        "f0_cv": 50.0,
        "formant_transition_rate": 100000.0,
        "phase_regularity": 50.0,
        "spectral_rolloff_hz": 100000.0,
    }
    fp = match(_bundle_from_scalars(weird), settings)
    assert fp.probable_source == "UNKNOWN"
