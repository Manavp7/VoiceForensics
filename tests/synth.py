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


def _syllabic_envelope(
    n: int, sr: int, *, rate: float, jitter: float, rng: np.random.Generator
) -> np.ndarray:
    """A 0..1 amplitude envelope at syllable rate that dips to ~0 between syllables.

    The periodic near-silent troughs give the engine's percentile-based SNR
    estimator a real noise floor to find (as natural speech pauses would).
    """
    t = np.arange(n) / sr
    phase_wobble = jitter * np.cumsum(rng.normal(0, 1.0, n)) / sr if jitter else 0.0
    raw = 0.5 + 0.5 * np.sin(2 * np.pi * rate * t + 2 * np.pi * phase_wobble)
    env = raw**2.0  # sharpen so troughs approach zero
    return env


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

    # Syllabic envelope with natural rate jitter → periodic near-silent gaps.
    env = _syllabic_envelope(n, sr, rate=3.2, jitter=0.4, rng=rng)
    sig = sig * env

    # Breath / aspiration noise (broadband), gated by the same envelope.
    breath = rng.normal(0, 1.0, n)
    breath = _highpass(breath, sr, 2000.0) * 0.06 * env
    sig = sig + breath

    # Constant low room hiss sets the noise floor (well below voiced level).
    sig = sig + rng.normal(0, 0.0015, n)

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

    # Over-regular syllabic envelope (fixed rate, no jitter).
    env = _syllabic_envelope(n, sr, rate=3.0, jitter=0.0, rng=rng)
    sig = sig * env

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
