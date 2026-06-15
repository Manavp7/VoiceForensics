# Datasets & training handoff

The Core Detection Engine ships with a transparent heuristic baseline and
**weight-gated** neural backends (RawNet2, AASIST, Whisper-head). To turn the
backends on you need trained weights, and to train them you need data. This file
documents the recommended corpora and the training/fingerprinting handoff.

> Nothing here is downloaded automatically. These datasets are access-gated and/or
> large; obtain them under their respective licences.

## Anti-spoofing corpora (real vs. spoofed labels)

| Dataset | What it is | Link |
|---|---|---|
| ASVspoof 2019 (LA) | 17 spoofing systems, the standard anti-spoofing benchmark | https://www.asvspoof.org |
| ASVspoof 2021 (LA/DF) | Adds codec/compression + deepfake track | https://www.asvspoof.org |
| In-the-Wild | Real-world deepfakes of public figures | https://deepfake-total.com/in_the_wild |
| FakeAVCeleb | Multimodal (audio+video) deepfakes | https://sites.google.com/view/fakeavcelebdash-lab |

## Building the India-specific corpus (the moat)

1. **Real speech** (label = genuine): MUCS, IndicSpeech, AIR/LS-TV public archives —
   Hindi, Tamil, Telugu, Bengali, Marathi.
2. **Synthetic speech** (label = fake): generate matched utterances with
   ElevenLabs, XTTS-v2, RVC, Coqui — store the generator name as the
   fingerprint label.
3. Hold out generators/speakers for a clean test split; never leak speakers across
   train/test.

## Training the backends

The training harness is built in (`voiceforensics.training`). Organise data as a
folder with `real/` and `fake/` subdirectories, then:

```bash
# Train a backend (CPU works for smoke tests; use a GPU box for real runs).
python -m voiceforensics train  /data/asvspoof_la --model rawnet2 -o weights/rawnet2.pth
python -m voiceforensics train  /data/asvspoof_la --model aasist  -o weights/aasist.pth

# Fit Platt scaling + ensemble weights on the same/held-out split.
python -m voiceforensics calibrate /data/val -o calibration.env

# Point the engine at the trained weights (baseline_only becomes false).
export VF_RAWNET2_WEIGHTS_PATH=$(pwd)/weights/rawnet2.pth
export VF_AASIST_WEIGHTS_PATH=$(pwd)/weights/aasist.pth
source calibration.env   # sets VF_PLATT_A / VF_PLATT_B / VF_ENSEMBLE_WEIGHTS
```

Checkpoints are plain `state_dict`s matching the architectures in
`src/voiceforensics/models/{rawnet2,aasist}.py`. An ASVspoof protocol parser is
available via `voiceforensics.training.datasets.asvspoof_protocol`.

Recommended recipe: pretrain on ASVspoof (broad coverage), then fine-tune on the
Indian corpus (domain adaptation), and re-run `calibrate`.

## Refining the fingerprint database

`data/signatures/signatures.json` currently contains **placeholder** centroids.
To make source attribution trustworthy:

1. Extract scalar features (see `voiceforensics.features.extractor.SCALAR_KEYS`)
   for many samples from each known generator.
2. Compute the per-generator mean (centroid) and a global mean/std for
   standardization.
3. Replace the `templates` and `standardization` blocks in the JSON. Keep the
   `feature_keys` aligned with `SCALAR_KEYS`.
