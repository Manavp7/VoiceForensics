"""Pitch (F0) contour and jitter analysis.

Default backend is ``librosa.pyin`` (probabilistic YIN) which is dependency-light
and CPU-friendly. An optional ``torchcrepe`` backend can be enabled if installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FMIN = 65.0
FMAX = 500.0


@dataclass
class PitchResult:
    f0: np.ndarray  # NaN where unvoiced
    voiced_flag: np.ndarray
    times: np.ndarray

    @property
    def voiced_f0(self) -> np.ndarray:
        return self.f0[~np.isnan(self.f0)]


def f0_contour(y: np.ndarray, sr: int, *, backend: str = "pyin") -> PitchResult:
    if backend == "torchcrepe":
        try:
            return _f0_torchcrepe(y, sr)
        except Exception:  # noqa: BLE001 - fall back if torchcrepe unavailable
            pass
    return _f0_pyin(y, sr)


def _f0_pyin(y: np.ndarray, sr: int) -> PitchResult:
    import librosa

    hop = max(1, int(sr * 0.010))
    f0, voiced_flag, _ = librosa.pyin(
        y.astype(np.float32), fmin=FMIN, fmax=FMAX, sr=sr, hop_length=hop
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    return PitchResult(f0=f0, voiced_flag=voiced_flag, times=times)


def _f0_torchcrepe(y: np.ndarray, sr: int) -> PitchResult:
    import torch
    import torchcrepe

    audio = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    hop = max(1, int(sr * 0.010))
    f0 = torchcrepe.predict(
        audio, sr, hop_length=hop, fmin=FMIN, fmax=FMAX, model="tiny", batch_size=512
    )
    f0 = f0.squeeze(0).cpu().numpy()
    voiced = f0 > 0
    f0 = np.where(voiced, f0, np.nan)
    times = np.arange(len(f0)) * hop / sr
    return PitchResult(f0=f0, voiced_flag=voiced, times=times)


def jitter_stats(pitch: PitchResult) -> dict[str, float]:
    """Compute pitch-stability statistics (low jitter ~ synthetic over-regularity)."""
    f0 = pitch.voiced_f0
    if f0.size < 3:
        return {
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "f0_cv": 0.0,
            "jitter_local": 0.0,
            "voiced_ratio": float(np.mean(pitch.voiced_flag)) if pitch.voiced_flag.size else 0.0,
        }
    mean = float(np.mean(f0))
    std = float(np.std(f0))
    diffs = np.abs(np.diff(f0))
    jitter_local = float(np.mean(diffs) / (mean + 1e-9))
    return {
        "f0_mean": mean,
        "f0_std": std,
        "f0_cv": float(std / (mean + 1e-9)),
        "jitter_local": jitter_local,
        "voiced_ratio": float(np.mean(pitch.voiced_flag)),
    }
