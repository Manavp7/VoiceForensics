# GPU training for VoiceForensics

The neural backends (RawNet2, AASIST) are **weight-gated**: the engine runs the
explainable heuristic baseline until you supply trained checkpoints. This folder + the
`scripts/train_asvspoof.py` orchestrator are everything needed to train them on a GPU.

## Quickest path — `train_voiceforensics.ipynb`

A turnkey notebook (Colab / Kaggle / any Jupyter+GPU): install → check GPU → prepare a
`real/`+`fake/` dataset → train both backends → calibrate → export a weights zip.

- **Colab**: upload the notebook, set Runtime ▸ GPU, Run all.
- **Kaggle**: New Notebook ▸ add the file ▸ Accelerator = GPU ▸ Run all.

## Scripted path — orchestrator

```bash
# On the GPU box, after `pip install -e ".[dev]"` + CUDA torch:
python scripts/train_asvspoof.py /data/asvspoof_la \
    --models rawnet2 aasist --epochs 30 --batch-size 32 --out weights/ --calibrate
source weights/trained.env          # sets VF_*_WEIGHTS_PATH (+ Platt params)
python -m voiceforensics info       # active detectors now include rawnet2/aasist
```

`--device auto` selects CUDA when available (falls back to CPU for smoke tests).

## Cloud GPU providers

### RunPod / Lambda / Vast.ai (rented GPU box)
```bash
# Pick a PyTorch CUDA image (e.g. runpod/pytorch). Then:
apt-get update && apt-get install -y ffmpeg git
pip install torch --index-url https://download.pytorch.org/whl/cu121
git clone <repo> && cd VoiceForensics && pip install -e ".[dev]"
python scripts/train_asvspoof.py /workspace/data --epochs 30 --out weights/ --calibrate
# Copy weights/ back via the provider's file browser / scp / S3.
```

### Datasets
See `scripts/DATASETS.md` for ASVspoof 2019/2021, In-the-Wild, and how to build the
India-specific corpus. Arrange any dataset as `real/` (bonafide) + `fake/` (spoof).

## After training
1. Move `rawnet2.pth` / `aasist.pth` to your server.
2. Set `VF_RAWNET2_WEIGHTS_PATH` / `VF_AASIST_WEIGHTS_PATH` and `source calibration.env`.
3. `GET /health` will report `baseline_only: false`.

> Honesty: report measured EER/AUC/min-tDCF (from `voiceforensics benchmark`) on a held-out
> set before claiming detector accuracy. Do not ship numbers you have not measured.
