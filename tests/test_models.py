"""Tests for detectors and the registry."""

from __future__ import annotations

import torch

from voiceforensics.config import Settings
from voiceforensics.features.extractor import extract
from voiceforensics.models import available_detectors, build_detectors
from voiceforensics.models.aasist import AASISTDetector, AASISTLite
from voiceforensics.models.heuristic import HeuristicDetector
from voiceforensics.models.rawnet2 import RawNet2, RawNet2Detector

SR = 16_000


def test_heuristic_always_available():
    assert HeuristicDetector().is_available() is True


def test_heuristic_ranks_synthetic_above_genuine(genuine_signal, synthetic_signal):
    det = HeuristicDetector()
    g = det.score(extract(genuine_signal, SR))
    s = det.score(extract(synthetic_signal, SR))
    assert g.available and s.available
    assert s.prob_fake > g.prob_fake
    # Sanity: scores land on opposite sides of 0.5 for these clear proxies.
    assert g.prob_fake < 0.5 < s.prob_fake
    # Explainability: per-feature contributions are reported.
    assert any(k.startswith("suspicion_") for k in s.raw)


def test_neural_backends_unavailable_without_weights(genuine_signal):
    settings = Settings(rawnet2_weights_path=None, aasist_weights_path=None)
    dets = build_detectors(settings)
    assert [d.name for d in dets] == ["heuristic"]
    assert [d.name for d in available_detectors(settings)] == ["heuristic"]


def test_neural_backends_added_when_path_set_but_gate_on_missing_file(tmp_path):
    missing = tmp_path / "nope.pth"
    settings = Settings(rawnet2_weights_path=missing, aasist_weights_path=missing)
    dets = build_detectors(settings)
    names = [d.name for d in dets]
    assert "rawnet2" in names and "aasist" in names
    # But they are NOT available because the file does not load.
    assert [d.name for d in available_detectors(settings)] == ["heuristic"]


def test_rawnet2_loads_real_weights_and_scores(tmp_path, genuine_signal):
    model = RawNet2(SR)
    weights = tmp_path / "rawnet2.pth"
    torch.save(model.state_dict(), weights)
    det = RawNet2Detector(weights, SR)
    assert det.is_available() is True
    score = det.score(extract(genuine_signal, SR))
    assert score.available is True
    assert 0.0 <= score.prob_fake <= 1.0


def test_aasist_loads_real_weights_and_scores(tmp_path, genuine_signal):
    model = AASISTLite(128)
    weights = tmp_path / "aasist.pth"
    torch.save(model.state_dict(), weights)
    det = AASISTDetector(weights, 128)
    assert det.is_available() is True
    score = det.score(extract(genuine_signal, SR))
    assert score.available is True
    assert 0.0 <= score.prob_fake <= 1.0
