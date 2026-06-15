"""Audio decoding/loading.

All input formats (MP3, WAV, OGG, M4A, phone recordings, …) are normalised to a
mono float32 waveform at the engine's target sample rate via ffmpeg, which gives
broad codec coverage and consistent resampling.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioDecodeError(Exception):
    """Raised when an input cannot be decoded into a usable waveform."""


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def decode_to_waveform(src: str | Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Decode ``src`` to a mono float32 waveform at ``target_sr``.

    Uses ffmpeg when available (any format). Falls back to ``soundfile`` for
    formats it can read natively (WAV/FLAC/OGG) if ffmpeg is missing.
    """
    src = Path(src)
    if not src.exists():
        raise AudioDecodeError(f"file not found: {src}")
    if src.stat().st_size == 0:
        raise AudioDecodeError(f"empty file: {src}")

    if _ffmpeg_available():
        y = _decode_with_ffmpeg(src, target_sr)
    else:
        y = _decode_with_soundfile(src, target_sr)

    if y.size == 0:
        raise AudioDecodeError(f"decoded waveform is empty: {src}")
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y.astype(np.float32, copy=False), target_sr


def _decode_with_ffmpeg(src: Path, target_sr: int) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "-f",
            "wav",
            tmp.name,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise AudioDecodeError(f"ffmpeg failed for {src.name}: {proc.stderr.strip()[:300]}")
        try:
            y, _ = sf.read(tmp.name, dtype="float32", always_2d=False)
        except Exception as exc:  # noqa: BLE001
            raise AudioDecodeError(f"could not read decoded wav: {exc}") from exc
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y


def _decode_with_soundfile(src: Path, target_sr: int) -> np.ndarray:
    try:
        y, sr = sf.read(str(src), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001
        raise AudioDecodeError(
            f"ffmpeg unavailable and soundfile could not read {src.name}: {exc}"
        ) from exc
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y
