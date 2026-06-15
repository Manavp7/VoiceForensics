"""Job processing: runs an analysis, persists the result, optionally renders a
report and delivers a webhook. Shared by all queue backends.
"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

from voiceforensics.config import Settings, get_settings
from voiceforensics.pipeline import Engine
from voiceforensics.schemas import AnalysisType


@lru_cache(maxsize=1)
def _engine() -> Engine:
    return Engine()


def report_key(analysis_id: str) -> str:
    return f"reports/{analysis_id}.pdf"


def run_analysis_job(
    job_id: str,
    audio_bytes: bytes,
    *,
    analysis_type: str = "full",
    suffix: str = ".audio",
    settings: Settings | None = None,
) -> None:
    """Execute one analysis job end-to-end, updating its DB row in place."""
    settings = settings or get_settings()
    from voiceforensics.service.db import Job, get_sessionmaker
    from voiceforensics.service.storage import get_storage
    from voiceforensics.service.webhooks import deliver

    Session = get_sessionmaker(settings)

    def _set_status(status: str) -> None:
        with Session() as session:
            job = session.get(Job, job_id)
            if job is not None:
                job.status = status
                session.commit()

    _set_status("running")
    atype = AnalysisType(analysis_type)
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            engine = _engine()
            if atype is AnalysisType.LEGAL:
                result, artifacts = engine.analyze_with_artifacts(
                    tmp.name, analysis_type=atype
                )
                from voiceforensics.reporting.pdf import generate_report

                pdf_dir = Path(tempfile.mkdtemp(prefix="vf_job_"))
                pdf_path = generate_report(result, artifacts, pdf_dir / f"{result.analysis_id}.pdf")
                key = report_key(result.analysis_id)
                get_storage(settings).put(key, pdf_path.read_bytes())
                result.report_url = f"{settings.report_base_url}/{result.analysis_id}.pdf"
            else:
                result = engine.analyze(tmp.name, analysis_type=atype)
                key = None

        payload = result.model_dump()
        with Session() as session:
            job = session.get(Job, job_id)
            if job is not None:
                import json

                job.status = "completed"
                job.result_json = json.dumps(payload, default=str)
                job.report_key = key
                webhook = job.webhook_url
                session.commit()
            else:
                webhook = None

        if webhook:
            deliver(webhook, {"job_id": job_id, "status": "completed", "result": payload},
                    settings=settings)

    except Exception as exc:  # noqa: BLE001 - record any failure on the job row
        with Session() as session:
            job = session.get(Job, job_id)
            if job is not None:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                webhook = job.webhook_url
                session.commit()
            else:
                webhook = None
        if webhook:
            deliver(webhook, {"job_id": job_id, "status": "failed", "error": str(exc)},
                    settings=settings)
