"""Deterministic synthetic-audio generators for offline tests.

These are crude *proxies*, not real speech. The goal is only to exercise the DSP
and scoring pipeline with two signals whose artefact profiles differ in the same
direction real genuine vs. synthetic speech tends to:

- ``genuine_like``: natural micro-variation in pitch (jitter) and amplitude
  (shimmer), breath noise, short pauses, and full-band content.
- ``synthetic_like``: over-regular pitch/phase, near-silent noise floor, no
  pauses, and a hard ~8 kHz band-limit wall with reduced high-frequency energy
  (mimicking TTS / codec artefacts).

Used by both ``conftest.py`` fixtures and ``scripts/make_sample_audio.py``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_SR = 16_000


def _formant_envelope(freqs: np.ndarray) -> np.ndarray:
    """A simple vowel-like spectral envelope (resonances near 700/1200/2600 Hz)."""
    env = np.zeros_like(freqs, dtype=np.float64)
    for centre, bw, gain in ((700.0, 130.0, 1.0), (1220.0, 180.0, 0.6), (2600.0, 250.0, 0.35)):
        env += gain * np.exp(-0.5 * ((freqs - centre) / bw) ** 2)
    return env + 1e-3


def genuine_like(duration_s: float = 3.0, sr: int = DEFAULT_SR, *, seed: int = 0) -> np.ndarray:
    """Generate a genuine-like voiced signal with natural jitter/shimmer + breath."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    t = np.arange(n) / sr

    base_f0 = 120.0
    # Slow vibrato + per-sample random-walk jitter → natural pitch micro-variation.
    vibrato = 3.0 * np.sin(2 * np.pi * 5.0 * t)
    jitter = np.cumsum(rng.normal(0, 0.7, n)) * 0.05
    f0 = base_f0 + vibrato + jitter
    phase = 2 * np.pi * np.cumsum(f0) / sr

    sig = np.zeros(n, dtype=np.float64)
    n_harm = 40
    for k in range(1, n_harm + 1):
        fk = k * base_f0
        if fk >= sr / 2:
            break
        amp = _formant_envelope(np.array([fk]))[0] / k
        shimmer = 1.0 + 0.08 * np.sin(2 * np.pi * 4.0 * t + k) + rng.normal(0, 0.03, n)
        sig += amp * shimmer * np.sin(k * phase + rng.uniform(0, 2 * np.pi))

    # Breath / aspiration noise (broadband) and a touch of room hiss.
    breath = rng.normal(0, 1.0, n)
    breath = _highpass(breath, sr, 2000.0) * 0.06
    sig = sig + breath + rng.normal(0, 0.002, n)

    # Insert a couple of short pauses (natural speech is not continuous).
    sig = _apply_pauses(sig, sr, rng, n_pauses=2)

    return _normalise(sig).astype(np.float32)


def synthetic_like(duration_s: float = 3.0, sr: int = DEFAULT_SR, *, seed: int = 1) -> np.ndarray:
    """Generate a synthetic-like signal: over-regular, band-limited, low-noise."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    t = np.arange(n) / sr

    base_f0 = 130.0
    # Almost no jitter → unnaturally stable pitch.
    f0 = base_f0 + 0.2 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr

    sig = np.zeros(n, dtype=np.float64)
    n_harm = 40
    for k in range(1, n_harm + 1):
        fk = k * base_f0
        if fk >= sr / 2:
            break
        amp = _formant_envelope(np.array([fk]))[0] / k
        # Fixed phase, fixed amplitude → over-regular structure.
        sig += amp * np.sin(k * phase)

    sig = sig + rng.normal(0, 0.0005, n)  # near-silent noise floor

    # Hard band-limit around 7 kHz to mimic a TTS / codec HF wall.
    sig = _lowpass(sig, sr, 7000.0)

    return _normalise(sig).astype(np.float32)


def spliced(duration_s: float = 4.0, sr: int = DEFAULT_SR, *, seed: int = 7) -> np.ndarray:
    """First half genuine-like, second half synthetic-like (for localization tests)."""
    half = duration_s / 2
    a = genuine_like(half, sr, seed=seed)
    b = synthetic_like(half, sr, seed=seed + 1)
    return np.concatenate([a, b]).astype(np.float32)


def silence(duration_s: float = 1.0, sr: int = DEFAULT_SR) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


# --- small DSP helpers --------------------------------------------------------


def _normalise(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) or 1.0
    return (x / m) * peak


def _biquad_butter(x: np.ndarray, sr: int, cutoff: float, *, btype: str) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, cutoff / (sr / 2), btype=btype, output="sos")
    return sosfiltfilt(sos, x)


def _lowpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    return _biquad_butter(x, sr, cutoff, btype="low")


def _highpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    return _biquad_butter(x, sr, cutoff, btype="high")


def _apply_pauses(x: np.ndarray, sr: int, rng: np.random.Generator, n_pauses: int) -> np.ndarray:
    out = x.copy()
    pause_len = int(0.12 * sr)
    for _ in range(n_pauses):
        if len(out) <= pause_len * 3:
            break
        start = rng.integers(pause_len, len(out) - 2 * pause_len)
        ramp = np.ones(pause_len)
        ramp[: pause_len // 4] = np.linspace(1, 0, pause_len // 4)
        ramp[-pause_len // 4 :] = np.linspace(0, 1, pause_len // 4)
        ramp[pause_len // 4 : -pause_len // 4] = 0.0
        out[start : start + pause_len] *= ramp
    return out


# --- writing / encoding -------------------------------------------------------


def write_wav(path: str | Path, y: np.ndarray, sr: int = DEFAULT_SR) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="PCM_16")
    return path


def encode_with_ffmpeg(wav_path: str | Path, out_path: str | Path) -> Path:
    """Transcode a WAV into another container/codec using ffmpeg."""
    wav_path, out_path = Path(wav_path), Path(out_path)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not available")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), str(out_path)],
        check=True,
    )
    return out_path
