"""Detection metrics shared by the benchmark and training tools.

Implements EER, a simplified minimum t-DCF, accuracy, and ROC-AUC with no heavy
dependencies (numpy only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionMetrics:
    eer: float
    eer_threshold: float
    auc: float
    accuracy: float
    threshold: float
    n_real: int
    n_fake: int

    def as_dict(self) -> dict[str, float]:
        return {
            "eer": round(self.eer, 4),
            "eer_threshold": round(self.eer_threshold, 4),
            "auc": round(self.auc, 4),
            "accuracy": round(self.accuracy, 4),
            "threshold": round(self.threshold, 4),
            "n_real": self.n_real,
            "n_fake": self.n_fake,
        }


def _as_arrays(scores, labels) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if s.shape != y.shape:
        raise ValueError("scores and labels must have the same shape")
    return s, y


def compute_eer(scores, labels) -> tuple[float, float]:
    """Equal Error Rate. ``scores`` = P(fake); ``labels`` = 1 for fake, 0 for real.

    Returns ``(eer, threshold)``.
    """
    s, y = _as_arrays(scores, labels)
    if len(np.unique(y)) < 2:
        return 0.0, 0.5
    thresholds = np.unique(np.concatenate([[0.0], s, [1.0]]))
    best_eer, best_thr, best_gap = 1.0, 0.5, np.inf
    n_fake = max(int(np.sum(y == 1)), 1)
    n_real = max(int(np.sum(y == 0)), 1)
    for thr in thresholds:
        # Predict fake when score >= thr.
        far = np.sum((s >= thr) & (y == 0)) / n_real  # false accept (real->fake)
        frr = np.sum((s < thr) & (y == 1)) / n_fake   # false reject (fake->real)
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap, best_eer, best_thr = gap, (far + frr) / 2.0, float(thr)
    return float(best_eer), best_thr


def compute_auc(scores, labels) -> float:
    """ROC-AUC via the rank (Mann-Whitney U) statistic."""
    s, y = _as_arrays(scores, labels)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks for ties.
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros(len(counts))
    np.add.at(sum_ranks, inv, ranks)
    avg = sum_ranks / counts
    ranks = avg[inv]
    r_pos = np.sum(ranks[y == 1])
    n_pos, n_neg = len(pos), len(neg)
    auc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_min_tdcf(
    scores,
    labels,
    *,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """A simplified normalised minimum tandem Detection Cost Function.

    This is the standalone spoof-detection approximation (no ASV errors): it sweeps
    the threshold and returns the minimum normalised cost. Useful as a relative
    metric; not the full ASVspoof t-DCF.
    """
    s, y = _as_arrays(scores, labels)
    if len(np.unique(y)) < 2:
        return 0.0
    n_fake = max(int(np.sum(y == 1)), 1)
    n_real = max(int(np.sum(y == 0)), 1)
    thresholds = np.unique(np.concatenate([[0.0], s, [1.0]]))
    norm = min(c_miss * p_target, c_fa * (1 - p_target))
    best = np.inf
    for thr in thresholds:
        p_miss = np.sum((s < thr) & (y == 1)) / n_fake
        p_fa = np.sum((s >= thr) & (y == 0)) / n_real
        cost = c_miss * p_target * p_miss + c_fa * (1 - p_target) * p_fa
        best = min(best, cost)
    return float(best / norm) if norm > 0 else float(best)


def evaluate(scores, labels, *, threshold: float = 0.5) -> DetectionMetrics:
    s, y = _as_arrays(scores, labels)
    eer, eer_thr = compute_eer(s, y)
    auc = compute_auc(s, y)
    preds = (s >= threshold).astype(np.int64)
    accuracy = float(np.mean(preds == y)) if len(y) else 0.0
    return DetectionMetrics(
        eer=eer,
        eer_threshold=eer_thr,
        auc=auc,
        accuracy=accuracy,
        threshold=threshold,
        n_real=int(np.sum(y == 0)),
        n_fake=int(np.sum(y == 1)),
    )
