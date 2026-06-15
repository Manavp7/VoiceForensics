"""AASIST-lite spectro-temporal detector (weight-gated).

A simplified variant of AASIST: a small 2-D CNN encoder over the log-mel
spectrogram followed by a self-attention ("graph-attention-lite") pooling layer
and a binary classifier. The full AASIST uses heterogeneous spectral/temporal
graph attention; this compact version keeps the spectro-temporal-attention spirit
while remaining easy to train on CPU later. Weight-gated like all neural backends.

Reference: Jung et al., "AASIST" (ICASSP 2022).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from voiceforensics.features.extractor import FeatureBundle
from voiceforensics.models.base import Detector
from voiceforensics.schemas import ModelScore


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class AASISTLite(nn.Module):
    def __init__(self, n_mels: int = 128):
        super().__init__()
        self.enc1 = _ConvBlock(1, 16)
        self.enc2 = _ConvBlock(16, 32)
        self.enc3 = _ConvBlock(32, 64)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(64)
        self.fc = nn.Linear(64, 32)
        self.out = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, n_mels, T) → (B, 1, n_mels, T)
        x = mel.unsqueeze(1)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        b, c, freq, time = x.shape
        # Flatten freq/time into a token sequence for self-attention pooling.
        tokens = x.view(b, c, freq * time).transpose(1, 2)  # (B, N, C)
        attended, _ = self.attn(tokens, tokens, tokens)
        pooled = self.norm(attended.mean(dim=1))
        h = self.dropout(F.relu(self.fc(pooled)))
        return self.out(h)


class AASISTDetector(Detector):
    name = "aasist"
    requires_weights = True

    def __init__(self, weights_path: Path | None, n_mels: int = 128):
        self.weights_path = weights_path
        self.n_mels = n_mels
        self._model: AASISTLite | None = None
        self._load_error: str | None = None

    def is_available(self) -> bool:
        if self.weights_path is None:
            return False
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        return self._try_load()

    def _try_load(self) -> bool:
        try:
            if not Path(self.weights_path).exists():  # type: ignore[arg-type]
                self._load_error = "weights file not found"
                return False
            model = AASISTLite(self.n_mels)
            state = torch.load(self.weights_path, map_location="cpu")
            state = state.get("state_dict", state) if isinstance(state, dict) else state
            model.load_state_dict(state)
            model.eval()
            self._model = model
            return True
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            return False

    def score(self, bundle: FeatureBundle) -> ModelScore:
        if not self.is_available() or self._model is None:
            return self._unavailable_score()
        with torch.no_grad():
            mel = torch.tensor(bundle.mel, dtype=torch.float32).unsqueeze(0)
            logits = self._model(mel)
            prob = float(F.softmax(logits, dim=-1)[0, 1])
        return ModelScore(name=self.name, prob_fake=prob, available=True, raw={})
