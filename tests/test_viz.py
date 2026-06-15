"""Tests for headless exhibit rendering and artifact threading."""

from __future__ import annotations

from voiceforensics.config import Settings
from voiceforensics.features.spectral import log_mel_spectrogram
from voiceforensics.localization.segment_scoring import localize
from voiceforensics.models import build_detectors
from voiceforensics.pipeline import Engine
from voiceforensics.viz.render import heatmap_png, mel_spectrogram_png, waveform_png

SR = 16_000
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_png(path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(8) == _PNG_MAGIC


def test_mel_png(tmp_path, genuine_signal):
    mel = log_mel_spectrogram(genuine_signal, SR)
    out = mel_spectrogram_png(mel, SR, tmp_path / "mel.png")
    assert out.exists() and _is_png(out) and out.stat().st_size > 1000


def test_heatmap_and_waveform_png(tmp_path, spliced_signal):
    settings = Settings()
    detectors = build_detectors(settings)
    mel = log_mel_spectrogram(spliced_signal, SR)
    loc = localize(detectors, spliced_signal, SR, mel, settings)
    h = heatmap_png(loc.heatmap, loc.segments, tmp_path / "heat.png")
    w = waveform_png(spliced_signal, SR, loc.segments, tmp_path / "wave.png")
    assert _is_png(h) and _is_png(w)
    assert h.stat().st_size > 1000 and w.stat().st_size > 1000


def test_artifacts_returned_for_full(genuine_wav):
    result, artifacts = Engine().analyze_with_artifacts(genuine_wav, analysis_type="full")
    assert artifacts is not None
    assert artifacts.mel.shape[0] == 128
    assert artifacts.heatmap.shape[0] == len(result.result.segments)
    assert artifacts.waveform.ndim == 1


def test_artifacts_none_for_quick(genuine_wav):
    _, artifacts = Engine().analyze_with_artifacts(genuine_wav, analysis_type="quick")
    assert artifacts is None
