"""Spectral features: log-mel spectrogram and MFCC (+ delta, delta-delta)."""

from __future__ import annotations

import numpy as np

N_MELS = 128
N_MFCC = 13
_WIN_MS = 25.0
_HOP_MS = 10.0


def _frame_params(sr: int) -> tuple[int, int]:
    n_fft = 1 << int(np.ceil(np.log2(sr * _WIN_MS / 1000)))
    hop = max(1, int(sr * _HOP_MS / 1000))
    return n_fft, hop


def log_mel_spectrogram(y: np.ndarray, sr: int, n_mels: int = N_MELS) -> np.ndarray:
    """Return a log-power mel spectrogram, shape ``(n_mels, T)`` (dB)."""
    import librosa

    n_fft, hop = _frame_params(sr)
    mel = librosa.feature.melspectrogram(
        y=y.astype(np.float32), sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0
    )
    return librosa.power_to_db(mel + 1e-10, ref=np.max).astype(np.float32)


def mfcc_features(y: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Return MFCC stacked with delta and delta-delta, shape ``(3*n_mfcc, T)``."""
    import librosa

    n_fft, hop = _frame_params(sr)
    mfcc = librosa.feature.mfcc(
        y=y.astype(np.float32), sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop
    )
    width = 9 if mfcc.shape[1] >= 9 else max(3, (mfcc.shape[1] // 2) * 2 + 1)
    if mfcc.shape[1] >= 3:
        d1 = librosa.feature.delta(mfcc, width=width)
        d2 = librosa.feature.delta(mfcc, order=2, width=width)
    else:
        d1 = np.zeros_like(mfcc)
        d2 = np.zeros_like(mfcc)
    return np.vstack([mfcc, d1, d2]).astype(np.float32)
