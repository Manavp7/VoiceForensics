"""Fit calibration (Platt scaling) and ensemble weights on a validation split.

Outputs values ready to drop into a ``.env`` (``VF_PLATT_A`` etc.). Pure numpy.
"""

from __future__ import annotations

import math

import numpy as np


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def fit_platt(
    probs,
    labels,
    *,
    iterations: int = 500,
    lr: float = 0.1,
) -> tuple[float, float]:
    """Fit ``a, b`` so that ``sigmoid(a * logit(p) + b)`` minimises log loss."""
    z = _logit(np.asarray(probs, dtype=np.float64))
    y = np.asarray(labels, dtype=np.float64)
    a, b = 1.0, 0.0
    n = max(len(y), 1)
    for _ in range(iterations):
        pred = _sigmoid(a * z + b)
        grad = pred - y
        ga = float(np.dot(grad, z) / n)
        gb = float(np.sum(grad) / n)
        a -= lr * ga
        b -= lr * gb
    return float(a), float(b)


def fit_ensemble_weights(
    detector_probs: dict[str, list[float]],
    labels,
    *,
    iterations: int = 1000,
    lr: float = 0.1,
) -> dict[str, float]:
    """Fit non-negative, normalised fusion weights via logistic regression.

    ``detector_probs`` maps detector name → per-sample P(fake). Returns weights
    summing to 1 (negative coefficients are clipped to 0).
    """
    names = list(detector_probs)
    if not names:
        return {}
    X = np.column_stack([_logit(np.asarray(detector_probs[n], dtype=np.float64)) for n in names])
    y = np.asarray(labels, dtype=np.float64)
    w = np.ones(len(names)) / len(names)
    b = 0.0
    n = max(len(y), 1)
    for _ in range(iterations):
        pred = _sigmoid(X @ w + b)
        grad = pred - y
        gw = X.T @ grad / n
        gb = float(np.sum(grad) / n)
        w -= lr * gw
        b -= lr * gb
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total <= 0:
        w = np.ones(len(names))
        total = float(w.sum())
    return {name: float(weight / total) for name, weight in zip(names, w, strict=True)}


def as_env_snippet(a: float, b: float, weights: dict[str, float] | None = None) -> str:
    lines = [f"VF_PLATT_A={a:.6f}", f"VF_PLATT_B={b:.6f}"]
    if weights:
        # ensemble_weights is a dict setting; pydantic-settings reads it as JSON.
        import json

        lines.append(f"VF_ENSEMBLE_WEIGHTS={json.dumps(weights)}")
    return "\n".join(lines)


def platt_logloss(a: float, b: float, probs, labels) -> float:
    z = _logit(np.asarray(probs, dtype=np.float64))
    y = np.asarray(labels, dtype=np.float64)
    pred = _sigmoid(a * z + b)
    pred = np.clip(pred, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred)))


def _uncalibrated_logloss(probs, labels) -> float:
    return platt_logloss(1.0, 0.0, probs, labels)


def improvement(a: float, b: float, probs, labels) -> float:
    """Return log-loss reduction vs. the uncalibrated baseline (>= ~0 when fit helps)."""
    return _uncalibrated_logloss(probs, labels) - platt_logloss(a, b, probs, labels)


def _safe(x: float) -> float:
    return x if math.isfinite(x) else 0.0
