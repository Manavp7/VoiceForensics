"""Advanced acoustic features for spoof detection.

- Long-Term Average Spectrum (LTAS): spectral tilt, flatness, centroid.
- Modulation spectrum: energy in the syllabic (4-8 Hz) envelope band.
- CQCC-style cepstral statistics (constant-Q transform → log → DCT).
- Breathing / pause statistics (pause rate, voiced-run rate).

All values are scalar and deterministic.
"""

from __future__ import annotations

import numpy as np


def ltas_features(y: np.ndarray, sr: int) -> dict[str, float]:
    import librosa

    n_fft = 1024
    hop = 256
    S = np.abs(librosa.stft(y.astype(np.float32), n_fft=n_fft, hop_length=hop)) ** 2
    ltas = S.mean(axis=1) + 1e-12
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # Spectral tilt: slope of log-power vs log-frequency (negative for natural speech).
    valid = freqs > 0
    log_f = np.log(freqs[valid])
    log_p = np.log(ltas[valid])
    tilt = float(np.polyfit(log_f, log_p, 1)[0])

    centroid = float(np.sum(freqs * ltas) / np.sum(ltas))
    gmean = float(np.exp(np.mean(np.log(ltas))))
    amean = float(np.mean(ltas))
    flatness = float(gmean / amean)
    return {
        "ltas_tilt": tilt,
        "ltas_centroid_hz": centroid,
        "ltas_flatness": flatness,
    }


def modulation_features(y: np.ndarray, sr: int) -> dict[str, float]:
    import librosa

    hop = 256
    env = librosa.feature.rms(y=y.astype(np.float32), hop_length=hop)[0]
    env = env - np.mean(env)
    env_sr = sr / hop
    if len(env) < 8:
        return {"mod_4_8hz_ratio": 0.0, "mod_peak_hz": 0.0}
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    mod_freqs = np.fft.rfftfreq(len(env), d=1.0 / env_sr)
    total = float(np.sum(spec)) + 1e-12
    band = (mod_freqs >= 4.0) & (mod_freqs <= 8.0)
    ratio = float(np.sum(spec[band]) / total)
    peak_hz = float(mod_freqs[int(np.argmax(spec))])
    return {"mod_4_8hz_ratio": ratio, "mod_peak_hz": peak_hz}


def cqcc_features(y: np.ndarray, sr: int) -> dict[str, float]:
    import librosa
    from scipy.fftpack import dct

    try:
        cqt = np.abs(librosa.cqt(y.astype(np.float32), sr=sr, n_bins=72, bins_per_octave=12))
    except Exception:  # noqa: BLE001 - very short signals can fail CQT
        return {"cqcc_var_mean": 0.0, "cqt_high_low_ratio": 0.0}
    log_cqt = np.log(cqt + 1e-9)
    cqcc = dct(log_cqt, axis=0, norm="ortho")[:20]
    cqcc_var_mean = float(np.mean(np.var(cqcc, axis=1)))

    power = cqt**2
    n = power.shape[0]
    low = float(np.sum(power[: n // 2])) + 1e-12
    high = float(np.sum(power[n // 2 :]))
    return {"cqcc_var_mean": cqcc_var_mean, "cqt_high_low_ratio": float(high / low)}


def breathing_features(y: np.ndarray, sr: int) -> dict[str, float]:
    frame = max(1, int(sr * 0.025))
    hop = max(1, int(sr * 0.010))
    if len(y) < frame:
        return {"pause_rate_hz": 0.0, "voiced_run_rate_hz": 0.0, "mean_pause_ms": 0.0}
    n = 1 + (len(y) - frame) // hop
    rms = np.array([np.sqrt(np.mean(y[i * hop : i * hop + frame] ** 2) + 1e-12) for i in range(n)])
    thr = 0.2 * (np.percentile(rms, 95) + 1e-9)
    silent = rms < thr

    # Count silent runs (pauses) and voiced runs.
    pauses = 0
    pause_lengths: list[int] = []
    voiced_runs = 0
    i = 0
    while i < n:
        run = 1
        while i + run < n and silent[i + run] == silent[i]:
            run += 1
        if silent[i]:
            pauses += 1
            pause_lengths.append(run)
        else:
            voiced_runs += 1
        i += run

    duration_s = len(y) / sr
    mean_pause_ms = float(np.mean(pause_lengths) * 10.0) if pause_lengths else 0.0
    return {
        "pause_rate_hz": float(pauses / duration_s) if duration_s else 0.0,
        "voiced_run_rate_hz": float(voiced_runs / duration_s) if duration_s else 0.0,
        "mean_pause_ms": mean_pause_ms,
    }


def advanced_features(y: np.ndarray, sr: int) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(ltas_features(y, sr))
    out.update(modulation_features(y, sr))
    out.update(cqcc_features(y, sr))
    out.update(breathing_features(y, sr))
    return out
