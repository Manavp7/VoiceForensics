# AGENTS.md — VoiceForensics

VoiceForensics is a **defensive** security product: it detects AI-generated / cloned
("deepfake") speech and produces forensic, court-oriented analysis. Treat it the same
category as antivirus or fraud detection.

## What this branch contains
The **Core Detection Engine** (Month 1 scope): a Python package implementing the full
detection pipeline end-to-end, plus a thin synchronous FastAPI wrapper and a CLI.

## Honesty / integrity rules (non-negotiable)
This is a legal-grade tool, so the code must never overstate confidence.

- The shipped default model is a **transparent DSP-feature heuristic detector**. It is a
  *baseline*, not a trained SOTA model. Outputs always carry provenance
  (`engine_provenance.baseline_only`, list of active detectors).
- Neural backends (RawNet2, AASIST, Whisper-head) are **weight-gated**: they only join the
  ensemble when real pretrained weights are supplied via config. Never enable an untrained
  network to emit "confident" scores.
- Do not hard-code or invent accuracy numbers. Verdicts derive only from calibrated scores.
- The fingerprint signature DB ships with clearly-labeled **placeholder** centroids; refine
  them from a real corpus before relying on attribution in production.

## Environment notes
- No GPU here (CPU only). Keep `torch` imports lazy so the heuristic path / API start fast.
- ffmpeg + ffprobe are available and used for decode/metadata.
- Tests must be **offline & deterministic** (synthesize audio; never download datasets).

## Layout
- `src/voiceforensics/` — package (`audio/`, `features/`, `models/`, `ensemble/`,
  `localization/`, `fingerprint/`, `api/`, `pipeline.py`, `schemas.py`, `config.py`,
  `hashing.py`).
- `tests/` — pytest, synthetic-audio fixtures in `conftest.py`.
- `scripts/` — `make_sample_audio.py`, `DATASETS.md`.
- `data/signatures/signatures.json` — seed fingerprint DB.

## Dev workflow
- Install: `pip install -e ".[dev]"`
- Lint: `ruff check .`
- Test: `pytest -q`
- CLI: `python -m voiceforensics analyze <audio_file>`
- API: `uvicorn voiceforensics.api.app:app --reload`

## Out of scope this branch (Month 2-3 roadmap)
Model training + Indian corpus, Celery/Redis async queue, PostgreSQL, R2/S3 storage,
webhook delivery, PDF court-report rendering, Next.js dashboard, billing/Razorpay,
SSRF-hardened URL fetching. Stubbed/documented only.
