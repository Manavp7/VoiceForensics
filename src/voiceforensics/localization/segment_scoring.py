"""Segment-level localization, spectrogram anomaly heatmap, and naturalness score."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voiceforensics.audio.segment import Window, frame_windows
from voiceforensics.config import Settings
from voiceforensics.features.extractor import extract
from voiceforensics.features.pitch import f0_contour, jitter_stats
from voiceforensics.models.base import Detector
from voiceforensics.schemas import Segment

# Minimum samples for a window to be worth scoring (~0.2 s at 16 kHz).
_MIN_WINDOW_SAMPLES = 3200


@dataclass
class LocalizationResult:
    segments: list[Segment]
    heatmap: np.ndarray  # (n_windows, n_mel_bands), anomaly in [0, 1]
    naturalness_score: float


def _score_window(detectors: list[Detector], y: np.ndarray, sr: int) -> float:
    """Fused suspicion for one window using the fast (YIN) feature path."""
    bundle = extract(y, sr, pitch_backend="yin")
    probs = [d.score(bundle).prob_fake for d in detectors if d.is_available()]
    return float(np.mean(probs)) if probs else 0.0


def _mel_anomaly_row(mel_col: np.ndarray) -> np.ndarray:
    """Per-band anomaly proxy: deviation from the window's own spectral envelope."""
    smooth = np.convolve(mel_col, np.ones(5) / 5, mode="same")
    dev = np.abs(mel_col - smooth)
    rng = float(dev.max() - dev.min()) or 1.0
    return (dev - dev.min()) / rng


def naturalness_score(y: np.ndarray, sr: int) -> float:
    """A 0-1 'human-likeness' score from prosody variation and pause presence.

    Higher = more natural. Driven by pitch micro-variation (jitter), F0 dynamic
    range, and the presence of low-energy gaps (breaths/pauses).
    """
    stats = jitter_stats(f0_contour(y, sr, backend="yin"))
    jitter = stats["jitter_local"]
    f0_cv = stats["f0_cv"]

    # Pause presence from frame-energy distribution.
    frame = max(1, int(sr * 0.025))
    hop = max(1, int(sr * 0.010))
    if len(y) >= frame:
        n = 1 + (len(y) - frame) // hop
        rms = np.array(
            [np.sqrt(np.mean(y[i * hop : i * hop + frame] ** 2) + 1e-12) for i in range(n)]
        )
        gap_ratio = float(np.mean(rms < 0.2 * (np.percentile(rms, 95) + 1e-9)))
    else:
        gap_ratio = 0.0

    jitter_term = min(1.0, jitter / 0.004)
    f0_term = min(1.0, f0_cv / 0.02)
    gap_term = min(1.0, gap_ratio / 0.3)
    score = 0.45 * jitter_term + 0.30 * f0_term + 0.25 * gap_term
    return float(round(min(1.0, max(0.0, score)), 4))


def localize(
    detectors: list[Detector],
    y: np.ndarray,
    sr: int,
    mel: np.ndarray,
    settings: Settings,
) -> LocalizationResult:
    windows: list[Window] = frame_windows(y, sr, settings.window_ms, settings.hop_ms)
    segments: list[Segment] = []
    heatmap_rows: list[np.ndarray] = []

    n_frames = mel.shape[1]
    total_ms = max(1, int(len(y) / sr * 1000))

    for w in windows:
        seg = w.slice(y)
        if len(seg) < _MIN_WINDOW_SAMPLES and len(windows) > 1:
            score = 0.0
        else:
            score = _score_window(detectors, seg, sr)
        segments.append(Segment(start_ms=w.start_ms, end_ms=w.end_ms, score=round(score, 4)))

        # Map window time-span onto mel frames for the heatmap row.
        f0 = int(w.start_ms / total_ms * n_frames)
        f1 = max(f0 + 1, int(w.end_ms / total_ms * n_frames))
        mel_col = mel[:, f0:f1].mean(axis=1) if f1 > f0 else mel[:, -1]
        heatmap_rows.append(_mel_anomaly_row(mel_col) * score)

    heatmap = np.vstack(heatmap_rows) if heatmap_rows else np.zeros((0, mel.shape[0]))
    return LocalizationResult(
        segments=segments,
        heatmap=heatmap.astype(np.float32),
        naturalness_score=naturalness_score(y, sr),
    )
