"""Ensemble fusion + uncertainty quantification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from voiceforensics.config import Settings
from voiceforensics.features.extractor import FeatureBundle
from voiceforensics.models.base import Detector
from voiceforensics.schemas import ModelScore


@dataclass
class FusionOutput:
    fused_prob: float
    uncertainty: float
    scores: list[ModelScore]
    contributions: dict[str, float]


def _weight_for(name: str, settings: Settings) -> float:
    return float(settings.ensemble_weights.get(name, 1.0))


def fuse_scores(scores: list[ModelScore], settings: Settings) -> tuple[float, dict[str, float]]:
    """Weighted mean over *available* detectors, with weights renormalised."""
    available = [s for s in scores if s.available]
    if not available:
        return 0.0, {}
    weights = np.array([_weight_for(s.name, settings) for s in available], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    probs = np.array([s.prob_fake for s in available], dtype=np.float64)
    fused = float(np.dot(weights, probs))
    contributions = {
        s.name: round(float(w * p), 4)
        for s, w, p in zip(available, weights, probs, strict=True)
    }
    return fused, contributions


def _mc_dropout_std(detector: Detector, bundle: FeatureBundle, passes: int) -> float | None:
    """Estimate predictive std via Monte-Carlo dropout for a torch-backed detector."""
    model = getattr(detector, "_model", None)
    if model is None or not isinstance(model, torch.nn.Module):
        return None
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    if not has_dropout:
        return None

    input_attr = "mel" if detector.name == "aasist" else "waveform"
    tensor = torch.tensor(getattr(bundle, input_attr), dtype=torch.float32).unsqueeze(0)

    model.train()  # enable dropout
    probs: list[float] = []
    try:
        with torch.no_grad():
            for _ in range(passes):
                logits = model(tensor)
                probs.append(float(torch.softmax(logits, dim=-1)[0, 1]))
    finally:
        model.eval()
    return float(np.std(probs)) if probs else None


def estimate_uncertainty(
    detectors: list[Detector],
    bundle: FeatureBundle,
    scores: list[ModelScore],
    settings: Settings,
) -> float:
    """Combine cross-detector disagreement with MC-dropout variance (when available)."""
    available = [s for s in scores if s.available]
    components: list[float] = []

    # Disagreement across detectors (std of their probabilities).
    if len(available) >= 2:
        components.append(float(np.std([s.prob_fake for s in available])))

    # MC-dropout variance from any neural detector that supports it.
    for det in detectors:
        if det.is_available():
            std = _mc_dropout_std(det, bundle, settings.mc_dropout_passes)
            if std is not None:
                components.append(std)

    if not components:
        # Heuristic-only fallback: derive a modest uncertainty from how close the
        # fused score sits to the decision boundary (max near 0.5).
        fused, _ = fuse_scores(scores, settings)
        return float(0.25 * (1.0 - abs(fused - 0.5) * 2.0) + 0.05)
    return float(np.mean(components))


def run_fusion(
    detectors: list[Detector],
    bundle: FeatureBundle,
    settings: Settings,
) -> FusionOutput:
    scores = [d.score(bundle) for d in detectors]
    for s in scores:
        s.weight = _weight_for(s.name, settings) if s.available else 0.0
    fused, contributions = fuse_scores(scores, settings)
    uncertainty = estimate_uncertainty(detectors, bundle, scores, settings)
    return FusionOutput(
        fused_prob=fused,
        uncertainty=uncertainty,
        scores=scores,
        contributions=contributions,
    )
