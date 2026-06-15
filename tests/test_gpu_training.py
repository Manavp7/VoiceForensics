"""Tests for the GPU-training package: device resolution, orchestrator, notebook."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from tests import synth
from voiceforensics.training.train import resolve_device

SR = synth.DEFAULT_SR
_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "train_asvspoof.py"
_NOTEBOOK = _REPO / "notebooks" / "train_voiceforensics.ipynb"


def test_resolve_device_explicit():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_matches_availability():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto") == expected


def _make_dataset(root, n=4):
    (root / "real").mkdir(parents=True)
    (root / "fake").mkdir(parents=True)
    for i in range(n):
        synth.write_wav(root / "real" / f"r{i}.wav", synth.genuine_like(2.0, seed=i), SR)
        synth.write_wav(root / "fake" / f"f{i}.wav", synth.synthetic_like(2.0, seed=70 + i), SR)
    return root


def test_orchestrator_end_to_end(tmp_path):
    ds = _make_dataset(tmp_path / "ds")
    out = tmp_path / "weights"
    # Run in a subprocess: --calibrate mutates env, so isolate from the pytest process.
    proc = subprocess.run(
        [
            sys.executable, str(_SCRIPT), str(ds),
            "--models", "rawnet2",
            "--epochs", "1", "--batch-size", "2",
            "--out", str(out), "--calibrate", "--device", "cpu",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr

    ckpt = out / "rawnet2.pth"
    manifest = json.loads((out / "manifest.json").read_text())
    env_text = (out / "trained.env").read_text()

    assert ckpt.exists()
    assert "rawnet2" in manifest["models"]
    assert manifest["device"] == "cpu"
    assert "calibration" in manifest
    assert "VF_RAWNET2_WEIGHTS_PATH" in env_text
    assert "VF_PLATT_A" in env_text

    # The produced checkpoint loads into the detector (closing the loop).
    from voiceforensics.models.rawnet2 import RawNet2Detector

    det = RawNet2Detector(ckpt, SR)
    assert det.is_available() is True


def test_orchestrator_help_without_running():
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0
    assert "dataset" in proc.stdout.lower()


def test_training_notebook_is_valid():
    nb = json.loads(_NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 6
    # Every code cell has the required nbformat fields.
    for cell in nb["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        if cell["cell_type"] == "code":
            assert "source" in cell and "outputs" in cell
    # The notebook references the training entrypoints.
    blob = json.dumps(nb)
    assert "voiceforensics" in blob and "train" in blob
