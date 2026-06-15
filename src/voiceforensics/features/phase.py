"""Phase / group-delay artefact features.

Many synthesis and vocoder pipelines produce unnaturally regular phase structure.
We summarise the dispersion of the group delay and the variability of phase
differences across frames as cues for over-regularity.
"""

from __future__ import annotations

import numpy as np


def phase_features(y: np.ndarray, sr: int) -> dict[str, float]:
    n_fft = 1 << int(np.ceil(np.log2(max(256, sr * 0.025))))
    hop = max(1, int(sr * 0.010))
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))

    window = np.hanning(n_fft)
    frames = []
    for start in range(0, len(y) - n_fft + 1, hop):
        seg = y[start : start + n_fft] * window
        frames.append(np.fft.rfft(seg))
    if len(frames) < 2:
        return {"group_delay_std": 0.0, "phase_diff_std": 0.0, "phase_regularity": 0.0}

    spec = np.stack(frames, axis=1)  # (freq, time)
    phase = np.angle(spec)

    # Group delay proxy: negative derivative of phase along frequency.
    gd = -np.diff(np.unwrap(phase, axis=0), axis=0)
    group_delay_std = float(np.std(gd))

    # Frame-to-frame phase change variability (low → over-regular synthesis).
    phase_diff = np.diff(np.unwrap(phase, axis=1), axis=1)
    phase_diff_std = float(np.std(phase_diff))

    # Combined regularity score in [0,1]: higher = more regular = more suspicious.
    phase_regularity = float(1.0 / (1.0 + phase_diff_std))
    return {
        "group_delay_std": group_delay_std,
        "phase_diff_std": phase_diff_std,
        "phase_regularity": phase_regularity,
    }
