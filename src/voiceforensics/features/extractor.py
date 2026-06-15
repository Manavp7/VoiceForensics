"""Feature extraction orchestration → :class:`FeatureBundle`.

The bundle carries both dense arrays (for neural backends / heatmaps) and a flat
``scalar_features`` mapping (consumed by the heuristic detector and the
fingerprint matcher). Extraction is deterministic for a given input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from voiceforensics.features.codec import codec_features
from voiceforensics.features.formants import formant_stats, formant_tracks
from voiceforensics.features.phase import phase_features
from voiceforensics.features.pitch import f0_contour, jitter_stats
from voiceforensics.features.spectral import log_mel_spectrogram, mfcc_features


@dataclass
class FeatureBundle:
    sample_rate: int
    waveform: np.ndarray
    mel: np.ndarray
    mfcc: np.ndarray
    formants: np.ndarray
    scalar_features: dict[str, float] = field(default_factory=dict)

    def vector(self, keys: list[str]) -> np.ndarray:
        return np.array([self.scalar_features.get(k, 0.0) for k in keys], dtype=np.float64)


# Canonical ordering of scalar features used by downstream models / fingerprints.
SCALAR_KEYS: list[str] = [
    "f0_mean",
    "f0_std",
    "f0_cv",
    "jitter_local",
    "voiced_ratio",
    "f1_mean",
    "f2_mean",
    "f3_mean",
    "f4_mean",
    "formant_transition_rate",
    "group_delay_std",
    "phase_diff_std",
    "phase_regularity",
    "hf_energy_ratio_4k",
    "hf_energy_ratio_6k",
    "hf_energy_ratio_7k",
    "effective_bandwidth_hz",
    "spectral_rolloff_hz",
    "bandlimit_ratio",
    "mfcc_var_mean",
]


def extract(y: np.ndarray, sr: int, *, pitch_backend: str = "pyin") -> FeatureBundle:
    """Extract all features from a preprocessed mono waveform."""
    mel = log_mel_spectrogram(y, sr)
    mfcc = mfcc_features(y, sr)
    formants = formant_tracks(y, sr)

    scalars: dict[str, float] = {}
    scalars.update(jitter_stats(f0_contour(y, sr, backend=pitch_backend)))
    scalars.update(formant_stats(formants))
    scalars.update(phase_features(y, sr))
    scalars.update(codec_features(y, sr))
    # Temporal variability of MFCCs (synthetic speech tends to be flatter).
    scalars["mfcc_var_mean"] = float(np.mean(np.var(mfcc[: mfcc.shape[0] // 3], axis=1)))

    # Ensure all canonical keys exist.
    for k in SCALAR_KEYS:
        scalars.setdefault(k, 0.0)

    return FeatureBundle(
        sample_rate=sr,
        waveform=y.astype(np.float32),
        mel=mel,
        mfcc=mfcc,
        formants=formants,
        scalar_features=scalars,
    )
