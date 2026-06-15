"""Shared pytest fixtures: deterministic synthetic audio in several formats."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from tests import synth

SR = synth.DEFAULT_SR


@pytest.fixture(scope="session", autouse=True)
def _service_env(tmp_path_factory):
    """Isolate persistence/storage/reports under a temp dir for the whole session."""
    d = tmp_path_factory.mktemp("svc")
    os.environ["VF_DATABASE_URL"] = f"sqlite:///{d / 'vf.db'}"
    os.environ["VF_STORAGE_DIR"] = str(d / "store")
    os.environ["VF_REPORTS_DIR"] = str(d / "reports")
    from voiceforensics.config import get_settings

    get_settings.cache_clear()
    # Create the schema up front: TestClient(app) at import time does not trigger
    # the FastAPI startup event, so tables must exist before the first request.
    from voiceforensics.service.db import init_db

    init_db()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def genuine_signal() -> np.ndarray:
    return synth.genuine_like(3.0, SR, seed=0)


@pytest.fixture(scope="session")
def synthetic_signal() -> np.ndarray:
    return synth.synthetic_like(3.0, SR, seed=1)


@pytest.fixture(scope="session")
def spliced_signal() -> np.ndarray:
    return synth.spliced(4.0, SR, seed=7)


@pytest.fixture(scope="session")
def genuine_wav(tmp_path_factory, genuine_signal) -> Path:
    d = tmp_path_factory.mktemp("audio")
    return synth.write_wav(d / "genuine.wav", genuine_signal, SR)


@pytest.fixture(scope="session")
def synthetic_wav(tmp_path_factory, synthetic_signal) -> Path:
    d = tmp_path_factory.mktemp("audio")
    return synth.write_wav(d / "synthetic.wav", synthetic_signal, SR)


@pytest.fixture(scope="session")
def silence_wav(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("audio")
    return synth.write_wav(d / "silence.wav", synth.silence(1.0, SR), SR)


@pytest.fixture(scope="session")
def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture(scope="session")
def encoded_files(tmp_path_factory, genuine_wav, has_ffmpeg) -> dict[str, Path]:
    """Genuine signal encoded into mp3/ogg/m4a (empty dict if ffmpeg missing)."""
    if not has_ffmpeg:
        return {}
    d = tmp_path_factory.mktemp("encoded")
    out: dict[str, Path] = {}
    for ext in ("mp3", "ogg", "m4a"):
        try:
            out[ext] = synth.encode_with_ffmpeg(genuine_wav, d / f"genuine.{ext}")
        except Exception:  # noqa: BLE001 - skip codecs ffmpeg build lacks
            pass
    return out
