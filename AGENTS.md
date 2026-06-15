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
- `src/voiceforensics/` — package:
  - core: `audio/`, `features/`, `models/`, `ensemble/`, `localization/`,
    `fingerprint/`, `pipeline.py`, `schemas.py`, `config.py`, `hashing.py`.
  - `viz/` — headless PNG exhibit rendering (matplotlib Agg).
  - `reporting/` — ReportLab legal PDF (`templates.py` wording reviewed by counsel).
  - `metrics.py` — EER / AUC / min t-DCF (shared by benchmark + training).
  - `tools/` — `benchmark.py`, `build_fingerprints.py`.
  - `training/` — `datasets.py`, `train.py`, `calibrate.py` (GPU off-box).
  - `service/` — `db.py`, `storage.py`, `queue.py`, `jobs.py`, `webhooks.py`,
    `security.py` (auth + rate limit + SSRF guard).
  - `api/app.py` — FastAPI (sync + async jobs, reports, visualize, CORS).
  - `logging_config.py` — JSON structured logging + request id.
- `frontend/` — Next.js 14 dashboard (TypeScript + Tailwind). Gate: `npm run build`.
- `tests/` — pytest, synthetic-audio fixtures in `conftest.py` (a session-autouse
  fixture isolates DB/storage under a temp dir).
- `scripts/` — `make_sample_audio.py`, `DATASETS.md`. `data/signatures/signatures.json`
  — seed fingerprint DB (placeholder centroids).

## Honesty note for reports
The PDF report and dashboard must surface `baseline_only` and the limitations text.
Never remove the "decision-support, not proof" framing or the placeholder-DB caveat.

## Default backends vs. optional adapters
The tested path is: heuristic detector, SQLite, local storage, thread queue. Celery/
Redis, Postgres, S3/R2, Docker builds, and GPU training are wired + documented but not
exercised in this sandbox (no services/GPU). Keep defaults working and offline.

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
