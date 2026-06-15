"""Tests for ensemble fusion and calibration."""

from __future__ import annotations

import torch

from voiceforensics.config import Settings
from voiceforensics.ensemble.calibration import (
    confidence_interval,
    platt_scale,
    to_verdict,
)
from voiceforensics.ensemble.fusion import fuse_scores, run_fusion
from voiceforensics.features.extractor import extract
from voiceforensics.models import build_detectors
from voiceforensics.models.heuristic import HeuristicDetector
from voiceforensics.models.rawnet2 import RawNet2, RawNet2Detector
from voiceforensics.schemas import ModelScore, Verdict

SR = 16_000


def test_fuse_single_detector_identity():
    settings = Settings()
    scores = [ModelScore(name="heuristic", prob_fake=0.73, available=True)]
    fused, contrib = fuse_scores(scores, settings)
    assert abs(fused - 0.73) < 1e-9
    assert abs(contrib["heuristic"] - 0.73) < 1e-3


def test_fuse_ignores_unavailable():
    settings = Settings()
    scores = [
        ModelScore(name="heuristic", prob_fake=0.8, available=True),
        ModelScore(name="rawnet2", prob_fake=0.0, available=False),
    ]
    fused, _ = fuse_scores(scores, settings)
    assert abs(fused - 0.8) < 1e-9


def test_fuse_weighted_mean():
    settings = Settings(ensemble_weights={"heuristic": 1.0, "rawnet2": 3.0})
    scores = [
        ModelScore(name="heuristic", prob_fake=0.0, available=True),
        ModelScore(name="rawnet2", prob_fake=1.0, available=True),
    ]
    fused, _ = fuse_scores(scores, settings)
    assert abs(fused - 0.75) < 1e-9  # (1*0 + 3*1) / 4


def test_platt_identity_default():
    for p in (0.1, 0.5, 0.9):
        assert abs(platt_scale(p, 1.0, 0.0) - p) < 1e-6


def test_platt_monotonic():
    a, b = 1.5, -0.3
    assert platt_scale(0.2, a, b) < platt_scale(0.5, a, b) < platt_scale(0.8, a, b)


def test_confidence_interval_brackets_point():
    lo, hi = confidence_interval(0.7, 0.2)
    assert 0.0 <= lo <= 0.7 <= hi <= 1.0


def test_confidence_interval_zero_std_is_tight():
    lo, hi = confidence_interval(0.7, 0.0)
    assert abs(hi - lo) < 1e-3


def test_verdict_mapping_boundaries():
    s = Settings()
    assert to_verdict(0.05, s) == Verdict.AUTHENTIC
    assert to_verdict(0.30, s) == Verdict.LEANING_AUTHENTIC
    assert to_verdict(0.50, s) == Verdict.UNCERTAIN
    assert to_verdict(0.70, s) == Verdict.LIKELY_SYNTHETIC
    assert to_verdict(0.95, s) == Verdict.HIGH_CONFIDENCE_SYNTHETIC


def test_run_fusion_heuristic_only(genuine_signal):
    settings = Settings()
    detectors = build_detectors(settings)
    out = run_fusion(detectors, extract(genuine_signal, SR), settings)
    assert 0.0 <= out.fused_prob <= 1.0
    assert out.uncertainty >= 0.0
    assert [s.name for s in out.scores if s.available] == ["heuristic"]


def test_run_fusion_with_neural_uncertainty(tmp_path, genuine_signal):
    weights = tmp_path / "rawnet2.pth"
    torch.save(RawNet2(SR).state_dict(), weights)
    settings = Settings(rawnet2_weights_path=weights)
    detectors = [HeuristicDetector(), RawNet2Detector(weights, SR)]
    out = run_fusion(detectors, extract(genuine_signal, SR), settings)
    available = [s.name for s in out.scores if s.available]
    assert "rawnet2" in available and "heuristic" in available
    assert out.uncertainty >= 0.0
