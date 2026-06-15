# VoiceForensics — Core Detection Engine

Forensic-grade detection of AI-generated / cloned ("deepfake") speech.
**Defensive security software** — same category as antivirus or fraud detection.

This repository (this branch) contains the **Core Detection Engine**: a Python
package implementing the full audio anti-spoofing pipeline end-to-end, plus a thin
synchronous HTTP API and a CLI.

---

## ⚠️ Honesty & limitations (read this first)

VoiceForensics is intended for legal/forensic contexts, so it must never overstate
confidence. A few deliberate, important design choices:

- **The shipped default model is a transparent DSP-feature heuristic baseline.** It
  converts explainable forensic cues (pitch jitter, formant-transition smoothness,
  high-frequency energy / band-limiting, phase regularity) into a score you can
  defend feature-by-feature. **It is a baseline, not a trained SOTA model.**
- **Neural backends (RawNet2, AASIST, Whisper-head) are weight-gated.** They are
  fully implemented PyTorch architectures, but they only join the ensemble when you
  supply *real pretrained weights* via configuration. Without weights they report
  themselves unavailable — an untrained network never emits "confident" scores.
- Every result includes **provenance**: which detectors were active and a
  `baseline_only` flag. When `baseline_only` is true, treat scores as an
  explainable prior, not a trained-model verdict.
- The **fingerprint signature database ships with clearly-labelled placeholder
  centroids.** Fit them on a real labelled corpus before relying on source
  attribution. See [`scripts/DATASETS.md`](scripts/DATASETS.md).

Training the models, building the Indian-language corpus, the async job queue,
database, object storage, webhooks, PDF court-report rendering, dashboard, and
billing are **Month 2–3 roadmap** items and are intentionally not implemented here.

---

## Architecture

```
Audio (mp3/wav/ogg/m4a/…)
   │
   ├─ Chain of custody ........ SHA-256 hash, ffprobe metadata, edit indicators
   ├─ Preprocess .............. ffmpeg → 16 kHz mono, SNR estimate + quality gate
   ├─ Segment ................. fixed analysis windows + energy-based regions
   ├─ Features (parallel) ..... log-mel(128), MFCC+Δ+ΔΔ(39), F0/jitter,
   │                            formants F1–F4, phase/group-delay, codec/band-limit
   ├─ Detectors (ensemble) .... heuristic baseline  [+ RawNet2 / AASIST / Whisper
   │                            when weights are configured]
   ├─ Fusion .................. weighted mean (renormalised) + MC-dropout / spread
   │                            uncertainty + Platt calibration → verdict + CI
   ├─ Localization ............ per-window scores, mel anomaly heatmap, naturalness
   └─ Fingerprint ............. nearest-neighbour source attribution (UNKNOWN-gated)
        │
        └─→ AnalysisResult (JSON)   [PDF court report = Month-2 scope]
```

Package layout: `src/voiceforensics/{audio,features,models,ensemble,localization,fingerprint,api}`,
plus `pipeline.py`, `schemas.py`, `config.py`, `hashing.py`.

---

## Install

Requires Python ≥ 3.11 and `ffmpeg`/`ffprobe` on the PATH.

```bash
pip install -e ".[dev]"          # editable install with dev tools
# CPU-only PyTorch (if not already present):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quick start

```bash
# Generate synthetic sample clips (NOT real speech — for smoke-testing only)
python scripts/make_sample_audio.py --out samples

# Engine / detector status
python -m voiceforensics info

# Analyze a file (full = score + localization + fingerprint)
python -m voiceforensics analyze samples/synthetic_like.wav --type full

# Run the API
uvicorn voiceforensics.api.app:app --reload
```

### Library

```python
from voiceforensics.pipeline import Engine

result = Engine().analyze("call_recording.m4a", analysis_type="full")
print(result.result.deepfake_probability, result.result.verdict)
print(result.provenance.baseline_only)   # True until neural weights are configured
```

### HTTP API

- `GET /health` → engine status + active detectors.
- `POST /v1/analyze` → JSON body with `audio_url` **or** `audio_base64`
  (`mode: "sync"` default, or `"async"` → returns a `job_id`).
- `POST /v1/analyze/upload` → multipart file upload (also supports `mode`).
- `GET /v1/jobs/{job_id}` → async job status + result.
- `GET /v1/reports/{analysis_id}.pdf` → download a generated legal report.
- `POST /v1/visualize/upload` (`kind=mel|heatmap|waveform`) → exhibit PNG.

```bash
curl -s -X POST localhost:8000/v1/analyze/upload \
  -F "file=@call.wav" -F "analysis_type=full"
```

Response (abridged):

```json
{
  "analysis_id": "vf_ab12cd34ef",
  "analysis_type": "full",
  "processing_time_ms": 1945,
  "result": {
    "deepfake_probability": 0.7575,
    "confidence_interval": [0.6907, 0.8138],
    "verdict": "LIKELY_SYNTHETIC",
    "uncertainty": 0.17,
    "naturalness_score": 0.2965,
    "segments": [{"start_ms": 0, "end_ms": 500, "score": 0.74}, "…"],
    "fingerprint": {"probable_source": "VALL-E", "confidence": 0.5, "alternative_sources": ["…"]},
    "metadata": {"file_hash_sha256": "…", "edit_indicators": {"edited": false, "reasons": []}}
  },
  "provenance": {"active_detectors": ["heuristic"], "baseline_only": true, "notes": ["…"]},
  "report_url": null
}
```

`analysis_type`: `quick` (score + verdict only), `full` (+ localization +
fingerprint), `legal` (full + forces chain-of-custody; PDF rendering is Month-2).

---

## Enabling the neural backends

Provide trained weights via environment variables (or a `.env` file):

```bash
export VF_RAWNET2_WEIGHTS_PATH=/path/to/rawnet2.pth
export VF_AASIST_WEIGHTS_PATH=/path/to/aasist.pth
```

The detectors will load on startup and join the ensemble; `baseline_only` becomes
`false`. Weights are expected to match the architectures in
`src/voiceforensics/models/{rawnet2,aasist}.py`. See `scripts/DATASETS.md` for how
to obtain training data.

---

## Legal Chain-of-Custody PDF report

`analysis_type="legal"` produces a court-oriented PDF (SHA-256 + metadata, findings
with plain-language interpretation, methodology for non-technical judges,
spectrogram/heatmap/waveform exhibits, a pre-filled expert-witness statement, and an
explicit limitations section). The content hash is embedded in the PDF metadata.

```bash
python -m voiceforensics report call.wav -o reports/
```

## Science tools

```bash
# Evaluate on a labelled dataset (real/ + fake/ subdirs) → EER, AUC, accuracy, min t-DCF
python -m voiceforensics benchmark path/to/dataset

# Build a measured fingerprint DB from labelled generator samples (one subdir per source)
python -m voiceforensics build-fingerprints samples/ -o signatures.json
```

Advanced features now include LTAS (spectral tilt/flatness), the modulation spectrum,
CQCC-style cepstral statistics, and breathing/pause statistics.

## Training the neural backends

The RawNet2 / AASIST architectures are weight-gated; train them and point the engine
at the checkpoints (real training needs a GPU — see `scripts/DATASETS.md`).

```bash
python -m voiceforensics train  path/to/dataset --model aasist -o weights/aasist.pth
python -m voiceforensics calibrate path/to/dataset -o calibration.env   # fits Platt + weights
export VF_AASIST_WEIGHTS_PATH=$(pwd)/weights/aasist.pth                  # baseline_only → false
```

## Productionized service

- **Persistence**: SQLAlchemy (`VF_DATABASE_URL`, SQLite default, Postgres optional).
- **Storage**: local (default) or S3/R2 (`VF_STORAGE_BACKEND=s3`, `[s3]` extra).
- **Async queue**: in-process thread pool (default) or Celery/Redis (`[celery]` extra).
- **Webhooks**: signed (HMAC) delivery with retries in async mode.
- **Security**: API-key auth (`VF_REQUIRE_AUTH=true`), per-key rate limiting, and an
  SSRF guard on `audio_url` fetches.

```python
from voiceforensics.service.db import create_api_key
print(create_api_key("my-client"))   # → vfk_...
```

## Dashboard (Next.js)

```bash
cd frontend
npm install
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev   # http://localhost:3000
```

## Docker

```bash
docker build -t voiceforensics .
docker run -p 8000:8000 voiceforensics
```

## Development

```bash
ruff check .
pytest -q          # offline, deterministic (synthesises its own audio)
cd frontend && npm run lint && npm run build
```

All Python tests run on CPU without any network access or dataset downloads. CI
(`.github/workflows/ci.yml`) runs ruff + pytest on Python 3.11 and 3.12.
