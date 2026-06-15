"""Tests for PDF report generation."""

from __future__ import annotations

from voiceforensics.pipeline import Engine
from voiceforensics.reporting.pdf import generate_report


def _read(path) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def test_generate_report_creates_valid_pdf(tmp_path, synthetic_wav):
    result, artifacts = Engine().analyze_with_artifacts(synthetic_wav, analysis_type="legal")
    out = generate_report(result, artifacts, tmp_path / "report.pdf")
    data = _read(out)
    assert data[:5] == b"%PDF-"
    assert len(data) > 20_000  # includes rendered exhibit images
    assert data.rstrip().endswith(b"%%EOF")


def test_report_embeds_hash_and_id(tmp_path, synthetic_wav):
    result, artifacts = Engine().analyze_with_artifacts(synthetic_wav, analysis_type="legal")
    out = generate_report(result, artifacts, tmp_path / "r.pdf")
    data = _read(out)
    # The analysis id and a prefix of the SHA-256 appear in the PDF text/metadata.
    assert result.analysis_id.encode() in data
    assert result.result.metadata.file_hash_sha256[:16].encode() in data


def test_report_without_artifacts_still_renders(tmp_path, genuine_wav):
    # quick analysis → no artifacts; report should still build (no exhibits).
    result = Engine().analyze(genuine_wav, analysis_type="quick")
    out = generate_report(result, None, tmp_path / "noart.pdf")
    assert _read(out)[:5] == b"%PDF-"


def test_engine_analyze_to_report_sets_url(tmp_path, synthetic_wav):
    result, pdf_path = Engine().analyze_to_report(synthetic_wav, out_dir=tmp_path)
    assert pdf_path.exists()
    assert result.report_url is not None
    assert result.report_url.endswith(f"{result.analysis_id}.pdf")
