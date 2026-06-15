"""Dataset loading for training the neural backends.

- :class:`SpoofDataset` reads a labelled folder (``real/`` + ``fake/``) and yields
  either raw waveforms (for RawNet2) or log-mel spectrograms (for AASIST).
- :func:`asvspoof_protocol` parses an ASVspoof-style protocol file into
  ``(utt_id, label)`` pairs (no audio is downloaded).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from voiceforensics.audio.io import decode_to_waveform
from voiceforensics.features.spectral import log_mel_spectrogram

_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".aac", ".opus"}


@dataclass
class Sample:
    path: Path
    label: int  # 1 = fake/spoof, 0 = real/bonafide


def _scan(folder: Path, label: int) -> list[Sample]:
    if not folder.is_dir():
        return []
    return [
        Sample(p, label)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in _AUDIO_EXTS
    ]


def _fixed_length(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


class SpoofDataset(Dataset):
    """A ``real/`` + ``fake/`` folder dataset.

    ``representation`` is ``"waveform"`` (RawNet2) or ``"mel"`` (AASIST).
    """

    def __init__(
        self,
        root: str | Path,
        *,
        representation: str = "waveform",
        sample_rate: int = 16_000,
        duration_s: float = 4.0,
    ):
        root = Path(root)
        self.samples = _scan(root / "real", 0) + _scan(root / "fake", 1)
        if not self.samples:
            raise FileNotFoundError(f"no audio found under {root}/real or {root}/fake")
        self.representation = representation
        self.sample_rate = sample_rate
        self.n_samples = int(duration_s * sample_rate)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        y, sr = decode_to_waveform(sample.path, self.sample_rate)
        y = _fixed_length(y.astype(np.float32), self.n_samples)
        if self.representation == "mel":
            mel = log_mel_spectrogram(y, sr)
            return torch.tensor(mel, dtype=torch.float32), sample.label
        return torch.tensor(y, dtype=torch.float32), sample.label

    @property
    def labels(self) -> list[int]:
        return [s.label for s in self.samples]


def asvspoof_protocol(protocol_path: str | Path) -> list[tuple[str, int]]:
    """Parse an ASVspoof LA protocol file.

    Lines look like: ``LA_0079 LA_T_1138215 - - bonafide`` (label is the last token,
    ``bonafide`` → 0, ``spoof`` → 1). Returns ``[(utt_id, label), ...]``.
    """
    out: list[tuple[str, int]] = []
    for line in Path(protocol_path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        utt_id = parts[1]
        label_token = parts[-1].lower()
        if label_token not in {"bonafide", "spoof"}:
            continue
        out.append((utt_id, 0 if label_token == "bonafide" else 1))
    return out
