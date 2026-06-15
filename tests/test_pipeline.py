"""End-to-end pipeline tests."""

from __future__ import annotations

import pytest

from voiceforensics.audio.preprocess import QualityGateError
from voiceforensics.pipeline import Engine
from voiceforensics.schemas import AnalysisResult, AnalysisType, Verdict


def test_pipeline_genuine_full(genuine_wav):
    result = Engine().analyze(genuine_wav, analysis_type="full")
    assert isinstance(result, AnalysisResult)
    assert result.analysis_id.startswith("vf_")
    assert result.status == "completed"
    assert 0.0 <= result.result.deepfake_probability <= 1.0
    assert result.result.segments  # localization present in full
    assert result.result.fingerprint is not None
    assert result.provenance.baseline_only is True  # no neural weights in CI
    assert result.result.metadata.file_hash_sha256
    assert result.report_url is None


def test_pipeline_verdicts_separate_genuine_and_synthetic(genuine_wav, synthetic_wav):
    g = Engine().analyze(genuine_wav, analysis_type="quick")
    s = Engine().analyze(synthetic_wav, analysis_type="quick")
    assert s.result.deepfake_probability > g.result.deepfake_probability
    assert g.result.verdict in (Verdict.AUTHENTIC, Verdict.LEANING_AUTHENTIC)
    assert s.result.verdict in (Verdict.LIKELY_SYNTHETIC, Verdict.HIGH_CONFIDENCE_SYNTHETIC)


def test_pipeline_quick_omits_localization_and_fingerprint(genuine_wav):
    result = Engine().analyze(genuine_wav, analysis_type="quick")
    assert result.analysis_type is AnalysisType.QUICK
    assert result.result.segments == []
    assert result.result.fingerprint is None
    assert result.result.naturalness_score is None


def test_pipeline_all_formats(encoded_files):
    if not encoded_files:
        pytest.skip("ffmpeg not available")
    engine = Engine()
    for ext, path in encoded_files.items():
        result = engine.analyze(path, analysis_type="quick")
        assert 0.0 <= result.result.deepfake_probability <= 1.0, ext


def test_pipeline_rejects_silence(silence_wav):
    with pytest.raises(QualityGateError):
        Engine().analyze(silence_wav, analysis_type="quick")


def test_pipeline_legal_forces_custody(genuine_wav):
    result = Engine().analyze(genuine_wav, analysis_type="legal", chain_of_custody=False)
    assert result.analysis_type is AnalysisType.LEGAL
    assert result.result.metadata.file_hash_sha256
