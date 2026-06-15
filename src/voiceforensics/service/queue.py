"""Job queue abstraction.

- :class:`ThreadJobQueue` (default) runs jobs on a background thread pool. It works
  everywhere with no external services and is the tested path.
- :class:`CeleryJobQueue` (optional) dispatches to Celery workers via a Redis broker
  for horizontal scaling in production. Import-guarded behind the ``[celery]`` extra.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from voiceforensics.config import Settings, get_settings


class JobQueue(ABC):
    @abstractmethod
    def enqueue(self, job_id: str, audio_bytes: bytes, analysis_type: str, suffix: str) -> None: ...

    def shutdown(self) -> None:  # noqa: B027 - optional lifecycle hook, no-op by default
        pass


class ThreadJobQueue(JobQueue):
    def __init__(self, settings: Settings | None = None, max_workers: int = 4):
        self.settings = settings or get_settings()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vf-job")

    def enqueue(self, job_id: str, audio_bytes: bytes, analysis_type: str, suffix: str) -> None:
        from voiceforensics.service.jobs import run_analysis_job

        self._pool.submit(
            run_analysis_job,
            job_id,
            audio_bytes,
            analysis_type=analysis_type,
            suffix=suffix,
            settings=self.settings,
        )

    def run_sync(self, job_id: str, audio_bytes: bytes, analysis_type: str, suffix: str) -> None:
        """Run inline (used by tests for deterministic completion)."""
        from voiceforensics.service.jobs import run_analysis_job

        run_analysis_job(
            job_id, audio_bytes, analysis_type=analysis_type, suffix=suffix, settings=self.settings
        )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)


class CeleryJobQueue(JobQueue):  # pragma: no cover - requires redis broker + workers
    def __init__(self, settings: Settings | None = None):
        from celery import Celery

        self.settings = settings or get_settings()
        self.app = Celery("voiceforensics", broker=self.settings.celery_broker_url)
        self._task = self.app.task(name="voiceforensics.run_analysis_job")(self._run)

    @staticmethod
    def _run(job_id: str, audio_bytes: bytes, analysis_type: str, suffix: str) -> None:
        from voiceforensics.service.jobs import run_analysis_job

        run_analysis_job(job_id, audio_bytes, analysis_type=analysis_type, suffix=suffix)

    def enqueue(self, job_id: str, audio_bytes: bytes, analysis_type: str, suffix: str) -> None:
        self._task.delay(job_id, audio_bytes, analysis_type, suffix)


def get_queue(settings: Settings | None = None) -> JobQueue:
    settings = settings or get_settings()
    if settings.queue_backend == "celery":
        return CeleryJobQueue(settings)
    return ThreadJobQueue(settings)
