"""RawNet2 end-to-end anti-spoofing model (weight-gated).

A compact, faithful RawNet2: a parameterised Sinc convolution front-end on the raw
waveform, residual blocks with filter-wise feature-map scaling (FMS), a GRU, and a
binary classifier head. The architecture exists so real pretrained weights can be
loaded later; without weights the wrapper reports itself unavailable.

Reference: Tak et al., "End-to-End anti-spoofing with RawNet2" (ICASSP 2021).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from voiceforensics.features.extractor import FeatureBundle
from voiceforensics.models.base import Detector
from voiceforensics.schemas import ModelScore


class SincConv(nn.Module):
    """Parameterised sinc band-pass filter bank (mel-spaced initial cut-offs)."""

    def __init__(self, out_channels: int = 20, kernel_size: int = 1024, sample_rate: int = 16000):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        low_hz = 30.0
        high_hz = sample_rate / 2 - (low_hz + 100.0)
        mel = np.linspace(self._to_mel(low_hz), self._to_mel(high_hz), out_channels + 1)
        hz = self._to_hz(mel)
        self.low_hz_ = nn.Parameter(torch.tensor(hz[:-1], dtype=torch.float32).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.tensor(np.diff(hz), dtype=torch.float32).view(-1, 1))

        n = (kernel_size - 1) / 2.0
        self.register_buffer("n_", 2 * math.pi * torch.arange(-n, n + 1).view(1, -1) / sample_rate)
        self.register_buffer(
            "window_", torch.hamming_window(kernel_size).view(1, -1)
        )

    @staticmethod
    def _to_mel(hz: float) -> float:
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def _to_hz(mel: np.ndarray) -> np.ndarray:
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = 30.0 + torch.abs(self.low_hz_)
        high = torch.clamp(low + torch.abs(self.band_hz_), 30.0, self.sample_rate / 2)
        band = (high - low)[:, 0]
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        bpl = (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2 + 1e-9)
        bpl = bpl * self.window_
        center = 2 * band.view(-1, 1)
        filters = torch.cat(
            [bpl[:, : self.kernel_size // 2], center, bpl[:, self.kernel_size // 2 + 1 :]], dim=1
        )
        filters = filters / (2 * band[:, None] + 1e-9)
        filters = filters.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.pool = nn.MaxPool1d(3)
        # Filter-wise feature-map scaling (FMS).
        self.fms = nn.Linear(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv1(F.leaky_relu(self.bn1(x), 0.3))
        out = self.conv2(F.leaky_relu(self.bn2(out), 0.3))
        out = out + identity
        out = self.pool(out)
        w = torch.sigmoid(self.fms(F.adaptive_avg_pool1d(out, 1).squeeze(-1)))
        return out * w.unsqueeze(-1)


class RawNet2(nn.Module):
    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sinc = SincConv(out_channels=20, kernel_size=1024, sample_rate=sample_rate)
        self.bn = nn.BatchNorm1d(20)
        self.block1 = _ResBlock(20, 32)
        self.block2 = _ResBlock(32, 32)
        self.block3 = _ResBlock(32, 64)
        self.block4 = _ResBlock(64, 64)
        self.gru = nn.GRU(64, 128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 64)
        self.out = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = torch.abs(self.sinc(x))
        x = F.max_pool1d(x, 3)
        x = F.leaky_relu(self.bn(x), 0.3)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.dropout(F.leaky_relu(self.fc(x), 0.3))
        return self.out(x)


class RawNet2Detector(Detector):
    name = "rawnet2"
    requires_weights = True

    def __init__(self, weights_path: Path | None, sample_rate: int = 16000):
        self.weights_path = weights_path
        self.sample_rate = sample_rate
        self._model: RawNet2 | None = None
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
            model = RawNet2(self.sample_rate)
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
            x = torch.tensor(bundle.waveform, dtype=torch.float32).unsqueeze(0)
            logits = self._model(x)
            prob = float(F.softmax(logits, dim=-1)[0, 1])
        return ModelScore(name=self.name, prob_fake=prob, available=True, raw={})
