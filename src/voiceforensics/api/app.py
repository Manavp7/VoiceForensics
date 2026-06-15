"""FastAPI surface for the detection engine.

Supports synchronous analysis (default, backward-compatible) and an asynchronous
job mode backed by a pluggable queue + persistence layer, plus report serving,
API-key auth, rate limiting, usage metering, and an SSRF-guarded URL fetcher.
"""

from __future__ import annotations

import base64
import binascii
import tempfile
import time
import uuid
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile

from voiceforensics import __version__
from voiceforensics.audio.io import AudioDecodeError
from voiceforensics.audio.preprocess import QualityGateError
from voiceforensics.config import get_settings
from voiceforensics.logging_config import configure_logging, get_logger, request_id_var
from voiceforensics.pipeline import Engine
from voiceforensics.schemas import AnalysisType, AnalyzeRequest
from voiceforensics.service.db import Job, UsageRecord, get_sessionmaker, init_db
from voiceforensics.service.queue import get_queue
from voiceforensics.service.security import (
    RateLimiter,
    SSRFError,
    assert_safe_url,
    validate_api_key,
)

configure_logging()
_log = get_logger("voiceforensics.api")

app = FastAPI(
    title="VoiceForensics API",
    version=__version__,
    description="Forensic-grade audio deepfake / voice-spoof detection (defensive security).",
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or "req_" + uuid.uuid4().hex[:12]
    token = request_id_var.set(rid)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    response.headers["X-Request-ID"] = rid
    _log.info(
        "request",
        extra={"extra_fields": {
            "method": request.method, "path": request.url.path, "ms": elapsed_ms,
        }},
    )
    return response

_engine: Engine | None = None
_queue = None
_rate_limiter: RateLimiter | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


def _queue_singleton():
    global _queue
    if _queue is None:
        _queue = get_queue()
    return _queue


def _limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(get_settings().rate_limit_per_minute)
    return _rate_limiter


@app.on_event("startup")
def _startup() -> None:
    init_db()


# --- dependencies -------------------------------------------------------------


def auth_dependency(request: Request):
    """Validate the API key when auth is required; return the ApiKey row or None."""
    settings = get_settings()
    raw = request.headers.get("x-api-key")
    key = validate_api_key(raw, settings)
    if settings.require_auth and key is None:
        raise HTTPException(status_code=401, detail="missing or invalid API key")
    return key


def rate_limit_dependency(request: Request, api_key=Depends(auth_dependency)):
    identity = api_key.key if api_key is not None else (request.client.host if request.client else "anon")
    if not _limiter().allow(identity):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    return api_key


def _record_usage(api_key, endpoint: str, analysis_type: str) -> None:
    Session = get_sessionmaker()
    with Session() as session:
        session.add(
            UsageRecord(
                api_key_id=getattr(api_key, "id", None),
                endpoint=endpoint,
                analysis_type=analysis_type,
            )
        )
        session.commit()


# --- basic endpoints ----------------------------------------------------------


@app.get("/health")
def health() -> dict:
    engine = get_engine()
    return {
        "status": "ok",
        "version": __version__,
        "active_detectors": engine.active_detector_names,
        "baseline_only": engine.baseline_only,
    }


def _run_sync(path: Path, analysis_type: AnalysisType, language_hint: str, chain_of_custody: bool):
    engine = get_engine()
    try:
        if analysis_type is AnalysisType.LEGAL:
            result, _ = engine.analyze_to_report(path, language_hint=language_hint)
            from voiceforensics.service.jobs import report_key
            from voiceforensics.service.storage import get_storage

            pdf_path = Path(get_settings().reports_dir) / f"{result.analysis_id}.pdf"
            if pdf_path.exists():
                get_storage().put(report_key(result.analysis_id), pdf_path.read_bytes())
        else:
            result = engine.analyze(
                path, analysis_type=analysis_type, language_hint=language_hint,
                chain_of_custody=chain_of_custody,
            )
    except QualityGateError as exc:
        raise HTTPException(status_code=422, detail=f"quality gate failed: {exc}") from exc
    except AudioDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"could not decode audio: {exc}") from exc
    return result.model_dump()


def _fetch_url(url: str) -> bytes:
    settings = get_settings()
    try:
        assert_safe_url(url, settings=settings)
    except SSRFError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        with httpx.Client(timeout=settings.download_timeout_s, follow_redirects=False) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.content
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"failed to fetch audio_url: {exc}") from exc
    if len(data) > settings.max_download_bytes:
        raise HTTPException(status_code=413, detail="audio_url content exceeds size limit")
    return data


def _enqueue_job(data: bytes, analysis_type: AnalysisType, webhook_url: str | None, api_key, suffix: str) -> dict:
    job_id = "job_" + uuid.uuid4().hex[:12]
    Session = get_sessionmaker()
    with Session() as session:
        session.add(
            Job(
                id=job_id,
                status="queued",
                analysis_type=analysis_type.value,
                webhook_url=webhook_url,
                api_key_id=getattr(api_key, "id", None),
            )
        )
        session.commit()
    _queue_singleton().enqueue(job_id, data, analysis_type.value, suffix)
    return {"job_id": job_id, "status": "queued"}


# --- analyze ------------------------------------------------------------------


@app.post("/v1/analyze")
def analyze(req: AnalyzeRequest, api_key=Depends(rate_limit_dependency)) -> dict:
    if not req.audio_url and not req.audio_base64:
        raise HTTPException(status_code=400, detail="provide either audio_url or audio_base64")

    if req.audio_base64:
        try:
            data = base64.b64decode(req.audio_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid base64: {exc}") from exc
    else:
        data = _fetch_url(req.audio_url)  # type: ignore[arg-type]

    _record_usage(api_key, "analyze", req.analysis_type.value)

    if req.mode == "async":
        return _enqueue_job(data, req.analysis_type, req.webhook_url, api_key, ".audio")

    notes = []
    if req.webhook_url:
        notes.append("webhook_url ignored in sync mode; use mode=async for delivery.")
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        payload = _run_sync(Path(tmp.name), req.analysis_type, req.language_hint, req.chain_of_custody)
    if notes:
        payload.setdefault("provenance", {}).setdefault("notes", []).extend(notes)
    return payload


@app.post("/v1/analyze/upload")
def analyze_upload(
    file: UploadFile = File(...),
    analysis_type: str = Form("full"),
    language_hint: str = Form("auto"),
    chain_of_custody: bool = Form(True),
    mode: str = Form("sync"),
    api_key=Depends(rate_limit_dependency),
) -> dict:
    try:
        atype = AnalysisType(analysis_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid analysis_type: {exc}") from exc

    data = file.file.read()
    suffix = Path(file.filename or "audio").suffix or ".audio"
    _record_usage(api_key, "analyze_upload", atype.value)

    if mode == "async":
        return _enqueue_job(data, atype, None, api_key, suffix)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        return _run_sync(Path(tmp.name), atype, language_hint, chain_of_custody)


# --- jobs + reports -----------------------------------------------------------


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, api_key=Depends(auth_dependency)) -> dict:
    import json

    Session = get_sessionmaker()
    with Session() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        out = {"job_id": job.id, "status": job.status, "analysis_type": job.analysis_type}
        if job.error:
            out["error"] = job.error
        if job.result_json:
            out["result"] = json.loads(job.result_json)
        return out


@app.get("/v1/reports/{analysis_id}.pdf")
def get_report(analysis_id: str, api_key=Depends(auth_dependency)) -> Response:
    from voiceforensics.service.jobs import report_key
    from voiceforensics.service.storage import get_storage

    storage = get_storage()
    key = report_key(analysis_id)
    if not storage.exists(key):
        raise HTTPException(status_code=404, detail="report not found")
    return Response(content=storage.get(key), media_type="application/pdf")


@app.post("/v1/visualize/upload")
def visualize_upload(
    file: UploadFile = File(...),
    kind: str = Form("mel"),
    api_key=Depends(rate_limit_dependency),
) -> Response:
    """Render a forensic exhibit PNG (``kind`` = mel | heatmap | waveform) for audio."""
    if kind not in {"mel", "heatmap", "waveform"}:
        raise HTTPException(status_code=400, detail="kind must be mel|heatmap|waveform")

    data = file.file.read()
    suffix = Path(file.filename or "audio").suffix or ".audio"
    engine = get_engine()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            result, artifacts = engine.analyze_with_artifacts(tmp.name, analysis_type="full")
        except QualityGateError as exc:
            raise HTTPException(status_code=422, detail=f"quality gate failed: {exc}") from exc
        except AudioDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"could not decode audio: {exc}") from exc

    if artifacts is None:
        raise HTTPException(status_code=500, detail="no artifacts produced")

    from voiceforensics.viz.render import heatmap_png, mel_spectrogram_png, waveform_png

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "exhibit.png"
        if kind == "mel":
            mel_spectrogram_png(artifacts.mel, artifacts.sample_rate, out)
        elif kind == "heatmap":
            heatmap_png(artifacts.heatmap, result.result.segments, out)
        else:
            waveform_png(artifacts.waveform, artifacts.sample_rate, result.result.segments, out)
        png = out.read_bytes()
    return Response(content=png, media_type="image/png")
