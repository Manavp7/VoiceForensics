"""Transparent DSP-feature heuristic detector — the shipped default baseline.

This detector is intentionally *explainable*: it converts a handful of forensic
cues into per-feature "suspicion" sub-scores in [0, 1] and blends them with fixed,
documented weights. It is NOT a trained SOTA model; it is a transparent baseline
whose reasoning can be inspected (and defended) feature-by-feature.

Cues (a synthetic/cloned voice tends to show):
  * abnormally low pitch jitter / F0 variation (over-stable pitch)
  * unnaturally smooth formant transitions
  * suppressed high-frequency energy / a hard band-limit wall (TTS/codec)
  * over-regular phase structure

Real telephony/codec audio can legitimately be band-limited, so band-limit cues
are weighted alongside (not above) pitch/formant cues to limit false positives.
"""

from __future__ import annotations

import math

from voiceforensics.features.extractor import FeatureBundle
from voiceforensics.models.base import Detector
from voiceforensics.schemas import ModelScore


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# (feature_key, threshold, scale, direction, weight)
# direction = -1 → suspicion rises as the feature falls below the threshold.
# direction = +1 → suspicion rises as the feature exceeds the threshold.
_RULES: tuple[tuple[str, float, float, int, float], ...] = (
    ("jitter_local", 0.0020, 0.0010, -1, 0.22),
    ("f0_cv", 0.0120, 0.0060, -1, 0.18),
    ("formant_transition_rate", 100.0, 40.0, -1, 0.15),
    ("hf_energy_ratio_4k", 0.0200, 0.0150, -1, 0.15),
    ("bandlimit_ratio", 0.7000, 0.2000, -1, 0.20),
    ("phase_regularity", 0.4500, 0.0700, +1, 0.10),
)


class HeuristicDetector(Detector):
    name = "heuristic"
    requires_weights = False

    def is_available(self) -> bool:
        return True

    def score(self, bundle: FeatureBundle) -> ModelScore:
        feats = bundle.scalar_features
        contributions: dict[str, float] = {}
        total_weight = 0.0
        weighted_sum = 0.0
        for key, thr, scale, direction, weight in _RULES:
            value = float(feats.get(key, 0.0))
            suspicion = _sigmoid(direction * (value - thr) / scale)
            contributions[f"suspicion_{key}"] = round(suspicion, 4)
            weighted_sum += weight * suspicion
            total_weight += weight
        prob = weighted_sum / total_weight if total_weight else 0.0
        return ModelScore(
            name=self.name,
            prob_fake=float(prob),
            available=True,
            raw=contributions,
        )
