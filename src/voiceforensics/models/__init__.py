"""Detector registry.

Builds the set of detectors for an analysis. The heuristic baseline is always
present; neural backends are appended only when weights are configured (and are
themselves weight-gated, so a misconfigured path simply yields an unavailable
detector rather than a crash).
"""

from __future__ import annotations

from voiceforensics.config import Settings
from voiceforensics.models.base import Detector
from voiceforensics.models.heuristic import HeuristicDetector


def build_detectors(settings: Settings) -> list[Detector]:
    """Instantiate all configured detectors (available or not)."""
    detectors: list[Detector] = [HeuristicDetector()]

    if settings.rawnet2_weights_path is not None:
        from voiceforensics.models.rawnet2 import RawNet2Detector

        detectors.append(
            RawNet2Detector(settings.rawnet2_weights_path, settings.target_sample_rate)
        )

    if settings.aasist_weights_path is not None:
        from voiceforensics.models.aasist import AASISTDetector

        detectors.append(AASISTDetector(settings.aasist_weights_path))

    if settings.whisper_weights_path is not None:
        try:
            from voiceforensics.models.whisper_head import WhisperHeadDetector

            detectors.append(WhisperHeadDetector(settings.whisper_weights_path))
        except ImportError:
            pass  # optional [whisper] extra not installed

    return detectors


def available_detectors(settings: Settings) -> list[Detector]:
    """Return only detectors that can produce meaningful scores in this environment."""
    return [d for d in build_detectors(settings) if d.is_available()]


__all__ = ["Detector", "HeuristicDetector", "build_detectors", "available_detectors"]
