"""Codec / band-limit artefact features.

TTS systems and low-bitrate codecs often impose a hard high-frequency wall and
suppress high-frequency energy. We estimate the effective bandwidth and the
fraction of energy above several cut-off bands.
"""

from __future__ import annotations

import numpy as np


def codec_features(y: np.ndarray, sr: int) -> dict[str, float]:
    n_fft = 1 << int(np.ceil(np.log2(max(512, sr * 0.032))))
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    power = spec**2
    total = float(np.sum(power)) + 1e-12

    # Fraction of energy in high bands.
    hf_4k = float(np.sum(power[freqs >= 4000]) / total)
    hf_6k = float(np.sum(power[freqs >= 6000]) / total)
    hf_7k = float(np.sum(power[freqs >= 7000]) / total)

    # Estimate effective bandwidth: highest freq holding >0.1% of peak-band power.
    band_power = _band_energy(power, freqs, band_hz=250.0, sr=sr)
    peak = float(np.max(band_power[1])) + 1e-12
    significant = band_power[0][band_power[1] > 0.001 * peak]
    effective_bandwidth = float(significant.max()) if significant.size else float(sr / 2)

    # Spectral rolloff (95% energy frequency).
    cumsum = np.cumsum(power)
    rolloff_idx = int(np.searchsorted(cumsum, 0.95 * cumsum[-1]))
    rolloff_hz = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    return {
        "hf_energy_ratio_4k": hf_4k,
        "hf_energy_ratio_6k": hf_6k,
        "hf_energy_ratio_7k": hf_7k,
        "effective_bandwidth_hz": effective_bandwidth,
        "spectral_rolloff_hz": rolloff_hz,
        "bandlimit_ratio": float(effective_bandwidth / (sr / 2)),
    }


def _band_energy(
    power: np.ndarray, freqs: np.ndarray, band_hz: float, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(0, sr / 2 + band_hz, band_hz)
    centres = edges[:-1] + band_hz / 2
    energies = np.empty(len(centres))
    for i in range(len(centres)):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        energies[i] = float(np.sum(power[mask]))
    return centres, energies
