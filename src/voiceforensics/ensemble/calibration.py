"""Score calibration, confidence intervals, and verdict mapping."""

from __future__ import annotations

import math

from voiceforensics.config import Settings
from voiceforensics.schemas import Verdict


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def platt_scale(prob: float, a: float, b: float) -> float:
    """Apply Platt scaling on the logit of ``prob``.

    With ``a=1, b=0`` this is the identity, so the default configuration leaves
    scores unchanged until calibration parameters are fit on labelled data.
    """
    return _sigmoid(a * _logit(prob) + b)


def confidence_interval(prob: float, std: float, z: float = 1.96) -> list[float]:
    """A logit-space interval mapped back to probability space, clamped to [0, 1]."""
    center = _logit(prob)
    lo = _sigmoid(center - z * std)
    hi = _sigmoid(center + z * std)
    lo, hi = min(lo, hi), max(lo, hi)
    return [round(max(0.0, lo), 4), round(min(1.0, hi), 4)]


def to_verdict(prob: float, settings: Settings) -> Verdict:
    if prob < settings.verdict_authentic_max:
        return Verdict.AUTHENTIC
    if prob < settings.verdict_leaning_authentic_max:
        return Verdict.LEANING_AUTHENTIC
    if prob < settings.verdict_uncertain_max:
        return Verdict.UNCERTAIN
    if prob < settings.verdict_likely_max:
        return Verdict.LIKELY_SYNTHETIC
    return Verdict.HIGH_CONFIDENCE_SYNTHETIC
