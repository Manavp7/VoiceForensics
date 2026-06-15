"""Tests for detection metrics."""

from __future__ import annotations

from voiceforensics.metrics import (
    compute_auc,
    compute_eer,
    compute_min_tdcf,
    evaluate,
)


def test_perfect_separation():
    scores = [0.1, 0.2, 0.15, 0.9, 0.85, 0.95]
    labels = [0, 0, 0, 1, 1, 1]
    eer, _ = compute_eer(scores, labels)
    assert eer == 0.0
    assert compute_auc(scores, labels) == 1.0
    assert compute_min_tdcf(scores, labels) == 0.0
    m = evaluate(scores, labels)
    assert m.accuracy == 1.0


def test_random_scores_auc_near_half():
    # Interleaved identical-ish scores → AUC ~ 0.5.
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [0, 1, 0, 1]
    assert abs(compute_auc(scores, labels) - 0.5) < 1e-9


def test_eer_threshold_in_range():
    scores = [0.2, 0.4, 0.6, 0.8]
    labels = [0, 0, 1, 1]
    eer, thr = compute_eer(scores, labels)
    assert 0.0 <= eer <= 1.0
    assert 0.0 <= thr <= 1.0


def test_single_class_safe():
    eer, _ = compute_eer([0.3, 0.4], [0, 0])
    assert eer == 0.0
    assert compute_auc([0.3, 0.4], [0, 0]) == 0.5


def test_evaluate_dict_shape():
    m = evaluate([0.1, 0.9], [0, 1]).as_dict()
    assert {"eer", "auc", "accuracy", "n_real", "n_fake"}.issubset(m)
