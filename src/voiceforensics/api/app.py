"""Thin synchronous FastAPI wrapper around the detection engine.

This is the Month-1 inference surface. Async job queue, webhook delivery, PDF
report rendering, auth/billing, and SSRF-hardened URL fetching are Month-2 scope
and are intentionally NOT implemented here (the endpoint documents the no-ops).
"""

from __future__ import annotations

import base64
import binascii
import tempfile
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from voiceforensics import __version__
from voiceforensics.audio.io import AudioDecodeError
from voiceforensics.audio.preprocess import QualityGateError
from voiceforensics.config import get_settings
from voiceforensics.pipeline import Engine
from voiceforensics.schemas import AnalysisType, AnalyzeRequest

app = FastAPI(
    title="VoiceForensics API",
    version=__version__,
    description="Forensic-grade audio deepfake / voice-spoof detection (defensive security).",
)

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


@app.get("/health")
def health() -> dict:
    engine = get_engine()
    return {
        "status": "ok",
        "version": __version__,
        "active_detectors": engine.active_detector_names,
        "baseline_only": engine.baseline_only,
    }


def _run(path: Path, analysis_type: AnalysisType, language_hint: str, chain_of_custody: bool):
    engine = get_engine()
    try:
        result = engine.analyze(
            path,
            analysis_type=analysis_type,
            language_hint=language_hint,
            chain_of_custody=chain_of_custody,
        )
    except QualityGateError as exc:
        raise HTTPException(status_code=422, detail=f"quality gate failed: {exc}") from exc
    except AudioDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"could not decode audio: {exc}") from exc
    return result.model_dump()


def _fetch_url(url: str) -> bytes:
    settings = get_settings()
    # NOTE: SSRF hardening (blocking private/link-local ranges) is Month-2 scope.
    try:
        with httpx.Client(timeout=settings.download_timeout_s, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.content
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"failed to fetch audio_url: {exc}") from exc
    if len(data) > settings.max_download_bytes:
        raise HTTPException(status_code=413, detail="audio_url content exceeds size limit")
    return data


@app.post("/v1/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    """Analyze audio provided as a URL or base64 payload (JSON body)."""
    if not req.audio_url and not req.audio_base64:
        raise HTTPException(status_code=400, detail="provide either audio_url or audio_base64")

    if req.audio_base64:
        try:
            data = base64.b64decode(req.audio_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid base64: {exc}") from exc
    else:
        data = _fetch_url(req.audio_url)  # type: ignore[arg-type]

    notes = []
    if req.webhook_url:
        notes.append("webhook_url accepted but ignored (async delivery is Month-2 scope).")

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        payload = _run(Path(tmp.name), req.analysis_type, req.language_hint, req.chain_of_custody)

    if notes:
        payload.setdefault("provenance", {}).setdefault("notes", []).extend(notes)
    return payload


@app.post("/v1/analyze/upload")
def analyze_upload(
    file: UploadFile = File(...),
    analysis_type: str = Form("full"),
    language_hint: str = Form("auto"),
    chain_of_custody: bool = Form(True),
) -> dict:
    """Analyze an uploaded audio file (multipart/form-data)."""
    try:
        atype = AnalysisType(analysis_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid analysis_type: {exc}") from exc

    suffix = Path(file.filename or "audio").suffix or ".audio"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(file.file.read())
        tmp.flush()
        return _run(Path(tmp.name), atype, language_hint, chain_of_custody)
