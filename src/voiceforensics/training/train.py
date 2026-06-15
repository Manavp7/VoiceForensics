"""Training loop for the weight-gated neural backends.

Trains RawNet2 or AASIST-lite on a labelled folder and saves a ``state_dict``
checkpoint that the corresponding detector can load. CPU-capable for smoke tests;
real training should run on GPU (see scripts/DATASETS.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from voiceforensics.metrics import compute_eer
from voiceforensics.models.aasist import AASISTLite
from voiceforensics.models.rawnet2 import RawNet2
from voiceforensics.training.datasets import SpoofDataset

_MODELS = {"rawnet2": ("waveform", RawNet2), "aasist": ("mel", AASISTLite)}


@dataclass
class TrainResult:
    checkpoint_path: Path
    best_val_eer: float
    epochs_run: int
    train_size: int
    val_size: int


def _split_indices(n: int, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    return idx[n_val:].tolist(), idx[:n_val].tolist()


def train_model(
    dataset_dir: str | Path,
    model_name: str,
    out_path: str | Path,
    *,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    val_frac: float = 0.3,
    device: str = "cpu",
    seed: int = 0,
    patience: int = 4,
) -> TrainResult:
    if model_name not in _MODELS:
        raise ValueError(f"unknown model '{model_name}', choose from {list(_MODELS)}")
    representation, model_cls = _MODELS[model_name]

    torch.manual_seed(seed)
    full = SpoofDataset(dataset_dir, representation=representation)
    train_idx, val_idx = _split_indices(len(full), val_frac, seed)
    train_dl = DataLoader(Subset(full, train_idx), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(Subset(full, val_idx), batch_size=batch_size)

    model = model_cls().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_eer = 1.0
    best_state = model.state_dict()
    no_improve = 0
    epochs_run = 0

    for epoch in range(epochs):
        epochs_run = epoch + 1
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()

        # Validation EER.
        model.eval()
        scores: list[float] = []
        labels: list[int] = []
        with torch.no_grad():
            for x, y in val_dl:
                probs = torch.softmax(model(x.to(device)), dim=-1)[:, 1]
                scores.extend(probs.cpu().tolist())
                labels.extend(y.tolist())
        eer, _ = compute_eer(scores, labels)
        if eer < best_eer:
            best_eer = eer
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    torch.save(best_state, out_path)
    return TrainResult(
        checkpoint_path=out_path,
        best_val_eer=float(best_eer),
        epochs_run=epochs_run,
        train_size=len(train_idx),
        val_size=len(val_idx),
    )
