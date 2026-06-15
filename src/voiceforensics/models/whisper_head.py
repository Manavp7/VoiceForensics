"""Optional Whisper-encoder + binary classifier head detector (weight-gated).

Requires the ``[whisper]`` extra (``transformers``) AND a trained classifier-head
weights file. Off by default. Importing this module without ``transformers``
installed raises ImportError, which the registry treats as "backend unavailable".
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel  # noqa: F401 - import gates availability

from voiceforensics.features.extractor import FeatureBundle
from voiceforensics.models.base import Detector
from voiceforensics.schemas import ModelScore


class _ClassifierHead(nn.Module):
    def __init__(self, in_dim: int = 384):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class WhisperHeadDetector(Detector):
    name = "whisper_head"
    requires_weights = True

    def __init__(self, weights_path: Path | None, model_name: str = "openai/whisper-tiny"):
        self.weights_path = weights_path
        self.model_name = model_name
        self._head: _ClassifierHead | None = None
        self._encoder = None
        self._load_error: str | None = None

    def is_available(self) -> bool:
        if self.weights_path is None:
            return False
        if self._head is not None:
            return True
        if self._load_error is not None:
            return False
        return self._try_load()

    def _try_load(self) -> bool:
        try:
            if not Path(self.weights_path).exists():  # type: ignore[arg-type]
                self._load_error = "weights file not found"
                return False
            from transformers import WhisperModel as _WM

            encoder = _WM.from_pretrained(self.model_name).encoder
            encoder.eval()
            head = _ClassifierHead(encoder.config.d_model)
            head.load_state_dict(torch.load(self.weights_path, map_location="cpu"))
            head.eval()
            self._encoder = encoder
            self._head = head
            return True
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            return False

    def score(self, bundle: FeatureBundle) -> ModelScore:
        if not self.is_available() or self._head is None:
            return self._unavailable_score()
        with torch.no_grad():
            from transformers import WhisperFeatureExtractor

            fe = WhisperFeatureExtractor.from_pretrained(self.model_name)
            inputs = fe(
                bundle.waveform, sampling_rate=bundle.sample_rate, return_tensors="pt"
            )
            feats = self._encoder(inputs.input_features).last_hidden_state.mean(dim=1)
            logits = self._head(feats)
            prob = float(F.softmax(logits, dim=-1)[0, 1])
        return ModelScore(name=self.name, prob_fake=prob, available=True, raw={})
