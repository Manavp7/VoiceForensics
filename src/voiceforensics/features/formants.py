"""Formant (F1-F4) estimation via LPC, and formant-transition statistics."""

from __future__ import annotations

import numpy as np

N_FORMANTS = 4


def _lpc_formants_frame(frame: np.ndarray, sr: int, order: int) -> list[float]:
    import librosa

    if np.max(np.abs(frame)) < 1e-6:
        return []
    win = frame * np.hamming(len(frame))
    try:
        a = librosa.lpc(win.astype(np.float64), order=order)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return []
    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0]
    roots = roots[np.abs(roots) < 1.0]  # stable poles inside the unit circle
    if roots.size == 0:
        return []
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angles * (sr / (2 * np.pi))
    freqs = np.sort(freqs[(freqs > 90) & (freqs < sr / 2 - 100)])
    return freqs.tolist()


def formant_tracks(y: np.ndarray, sr: int) -> np.ndarray:
    """Estimate F1-F4 per frame. Returns ``(N_FORMANTS, T)`` array (NaN if missing)."""
    frame_len = int(sr * 0.025)
    hop = int(sr * 0.010)
    order = int(2 + sr / 1000)  # rule of thumb: 2 + Fs(kHz)
    if len(y) < frame_len:
        return np.full((N_FORMANTS, 1), np.nan, dtype=np.float32)

    n_frames = 1 + (len(y) - frame_len) // hop
    tracks = np.full((N_FORMANTS, n_frames), np.nan, dtype=np.float64)
    for i in range(n_frames):
        frame = y[i * hop : i * hop + frame_len]
        fs = _lpc_formants_frame(frame, sr, order)
        for j in range(min(N_FORMANTS, len(fs))):
            tracks[j, i] = fs[j]
    return tracks.astype(np.float32)


def formant_stats(tracks: np.ndarray) -> dict[str, float]:
    """Summarise formant means and transition smoothness."""
    stats: dict[str, float] = {}
    transition_rates: list[float] = []
    for j in range(tracks.shape[0]):
        row = tracks[j]
        valid = row[~np.isnan(row)]
        if valid.size >= 2:
            stats[f"f{j + 1}_mean"] = float(np.mean(valid))
            stats[f"f{j + 1}_std"] = float(np.std(valid))
            transition_rates.append(float(np.mean(np.abs(np.diff(valid)))))
        else:
            stats[f"f{j + 1}_mean"] = 0.0
            stats[f"f{j + 1}_std"] = 0.0
    # Mean absolute frame-to-frame movement. Very low → unnaturally smooth.
    stats["formant_transition_rate"] = (
        float(np.mean(transition_rates)) if transition_rates else 0.0
    )
    return stats
