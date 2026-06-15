"""Preprocessing: noise-floor / SNR estimation, normalisation, quality gating."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class QualityGateError(Exception):
    """Raised when audio is too short / too noisy / silent to analyse reliably."""


@dataclass
class QualityReport:
    duration_s: float
    rms: float
    peak: float
    noise_floor_rms: float
    snr_db: float

    def as_dict(self) -> dict[str, float]:
        return {
            "duration_s": round(self.duration_s, 3),
            "rms": round(self.rms, 6),
            "peak": round(self.peak, 6),
            "noise_floor_rms": round(self.noise_floor_rms, 6),
            "snr_db": round(self.snr_db, 2),
        }


def _frame_rms(y: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(y) < frame:
        return np.array([float(np.sqrt(np.mean(y**2) + 1e-12))])
    n = 1 + (len(y) - frame) // hop
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        seg = y[i * hop : i * hop + frame]
        out[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
    return out


def estimate_snr_db(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Estimate (snr_db, noise_floor_rms) using a percentile split of frame energy."""
    rms = _frame_rms(y, sr)
    noise_floor = float(np.percentile(rms, 5))
    signal_level = float(np.percentile(rms, 95))
    noise_floor = max(noise_floor, 1e-7)
    snr_db = 20.0 * np.log10(max(signal_level, 1e-7) / noise_floor)
    return float(snr_db), noise_floor


def peak_normalize(y: np.ndarray, peak: float = 0.97) -> np.ndarray:
    m = float(np.max(np.abs(y))) if y.size else 0.0
    if m <= 1e-9:
        return y
    return (y / m) * peak


def analyze_quality(y: np.ndarray, sr: int) -> QualityReport:
    duration_s = len(y) / sr if sr else 0.0
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-12)) if y.size else 0.0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    snr_db, noise_floor = estimate_snr_db(y, sr) if y.size else (0.0, 0.0)
    return QualityReport(
        duration_s=duration_s, rms=rms, peak=peak, noise_floor_rms=noise_floor, snr_db=snr_db
    )


def quality_gate(
    report: QualityReport,
    *,
    min_snr_db: float,
    min_duration_s: float,
    max_duration_s: float,
) -> None:
    """Raise :class:`QualityGateError` if the audio fails admissibility checks."""
    if report.duration_s < min_duration_s:
        raise QualityGateError(
            f"audio too short: {report.duration_s:.2f}s < {min_duration_s:.2f}s"
        )
    if report.duration_s > max_duration_s:
        raise QualityGateError(
            f"audio too long: {report.duration_s:.1f}s > {max_duration_s:.1f}s"
        )
    if report.peak < 1e-4 or report.rms < 1e-5:
        raise QualityGateError("audio appears to be silent")
    if report.snr_db < min_snr_db:
        raise QualityGateError(
            f"SNR too low: {report.snr_db:.1f} dB < {min_snr_db:.1f} dB"
        )


def preprocess(
    y: np.ndarray,
    sr: int,
    *,
    min_snr_db: float,
    min_duration_s: float,
    max_duration_s: float,
) -> tuple[np.ndarray, QualityReport]:
    """Run quality analysis + gate, then peak-normalise. Returns (waveform, report)."""
    report = analyze_quality(y, sr)
    quality_gate(
        report,
        min_snr_db=min_snr_db,
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
    )
    return peak_normalize(y), report
