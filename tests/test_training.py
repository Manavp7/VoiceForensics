"""Tests for the training harness (datasets, train loop, calibration)."""

from __future__ import annotations

import numpy as np

from tests import synth
from voiceforensics.features.extractor import extract
from voiceforensics.models.aasist import AASISTDetector
from voiceforensics.models.rawnet2 import RawNet2Detector
from voiceforensics.training.calibrate import (
    fit_ensemble_weights,
    fit_platt,
    improvement,
)
from voiceforensics.training.datasets import SpoofDataset, asvspoof_protocol
from voiceforensics.training.train import train_model

SR = synth.DEFAULT_SR


def _make_dataset(root, n=4):
    (root / "real").mkdir(parents=True)
    (root / "fake").mkdir(parents=True)
    for i in range(n):
        synth.write_wav(root / "real" / f"r{i}.wav", synth.genuine_like(2.0, seed=i), SR)
        synth.write_wav(root / "fake" / f"f{i}.wav", synth.synthetic_like(2.0, seed=99 + i), SR)
    return root


def test_dataset_waveform_and_mel(tmp_path):
    _make_dataset(tmp_path, n=2)
    ds_w = SpoofDataset(tmp_path, representation="waveform", duration_s=2.0)
    x, label = ds_w[0]
    assert x.ndim == 1 and x.shape[0] == int(2.0 * SR)
    assert label in (0, 1)
    assert sorted(ds_w.labels) == [0, 0, 1, 1]

    ds_m = SpoofDataset(tmp_path, representation="mel", duration_s=2.0)
    mel, _ = ds_m[0]
    assert mel.shape[0] == 128


def test_asvspoof_protocol_parse(tmp_path):
    proto = tmp_path / "proto.txt"
    proto.write_text(
        "LA_0079 LA_T_1 - - bonafide\n"
        "LA_0079 LA_T_2 - A01 spoof\n"
        "garbage line\n"
    )
    parsed = asvspoof_protocol(proto)
    assert parsed == [("LA_T_1", 0), ("LA_T_2", 1)]


def test_train_rawnet2_smoke_and_load(tmp_path):
    _make_dataset(tmp_path, n=4)
    out = tmp_path / "rawnet2.pth"
    result = train_model(tmp_path, "rawnet2", out, epochs=2, batch_size=2, device="cpu")
    assert out.exists()
    assert 0.0 <= result.best_val_eer <= 1.0
    # Closing the loop: the detector can load the produced checkpoint and score.
    det = RawNet2Detector(out, SR)
    assert det.is_available() is True
    score = det.score(extract(synth.synthetic_like(2.0, seed=7), SR))
    assert score.available and 0.0 <= score.prob_fake <= 1.0


def test_train_aasist_smoke_and_load(tmp_path):
    _make_dataset(tmp_path, n=4)
    out = tmp_path / "aasist.pth"
    train_model(tmp_path, "aasist", out, epochs=2, batch_size=2, device="cpu")
    det = AASISTDetector(out, 128)
    assert det.is_available() is True


def test_fit_platt_improves_calibration():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 50 + [1] * 50)
    # Overconfident-but-ordered probabilities → Platt should reduce log loss.
    raw = np.concatenate([rng.uniform(0.3, 0.5, 50), rng.uniform(0.5, 0.7, 50)])
    a, b = fit_platt(raw, labels)
    assert improvement(a, b, raw, labels) >= -1e-6


def test_fit_ensemble_weights_normalised():
    labels = [0, 0, 1, 1]
    probs = {"heuristic": [0.1, 0.2, 0.8, 0.9], "rawnet2": [0.4, 0.5, 0.55, 0.6]}
    w = fit_ensemble_weights(probs, labels)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(v >= 0 for v in w.values())
