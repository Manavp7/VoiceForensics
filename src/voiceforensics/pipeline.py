"""End-to-end detection pipeline orchestration.

Ties together chain-of-custody, decoding/preprocessing, feature extraction,
ensemble fusion + calibration, localization, and fingerprinting into a single
:class:`AnalysisResult`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voiceforensics import __version__
from voiceforensics.audio.io import decode_to_waveform
from voiceforensics.audio.preprocess import preprocess
from voiceforensics.config import Settings, get_settings
from voiceforensics.ensemble.calibration import (
    confidence_interval,
    platt_scale,
    to_verdict,
)
from voiceforensics.ensemble.fusion import run_fusion
from voiceforensics.features.extractor import extract
from voiceforensics.fingerprint.matcher import match
from voiceforensics.hashing import collect_metadata
from voiceforensics.localization.segment_scoring import localize
from voiceforensics.models import build_detectors
from voiceforensics.schemas import (
    AnalysisResult,
    AnalysisType,
    DetectionResult,
    EngineProvenance,
)


@dataclass
class AnalysisArtifacts:
    """Dense intermediate arrays for rendering exhibits (not part of the JSON result)."""

    sample_rate: int
    waveform: np.ndarray
    mel: np.ndarray
    heatmap: np.ndarray


class Engine:
    """Reusable detection engine. Construct once, call :meth:`analyze` many times."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.detectors = build_detectors(self.settings)

    @property
    def active_detector_names(self) -> list[str]:
        return [d.name for d in self.detectors if d.is_available()]

    @property
    def baseline_only(self) -> bool:
        return self.active_detector_names == ["heuristic"]

    def analyze_to_report(
        self,
        source: str | Path,
        *,
        out_dir: str | Path | None = None,
        language_hint: str = "auto",
    ) -> tuple[AnalysisResult, Path]:
        """Run a legal-grade analysis and render a PDF report.

        Returns ``(result, pdf_path)`` with ``result.report_url`` populated.
        """
        from voiceforensics.reporting.pdf import generate_report

        result, artifacts = self.analyze_with_artifacts(
            source,
            analysis_type=AnalysisType.LEGAL,
            language_hint=language_hint,
            chain_of_custody=True,
        )
        out_dir = Path(out_dir) if out_dir is not None else Path(self.settings.reports_dir)
        pdf_path = out_dir / f"{result.analysis_id}.pdf"
        generate_report(result, artifacts, pdf_path)
        result.report_url = f"{self.settings.report_base_url}/{result.analysis_id}.pdf"
        return result, pdf_path

    def analyze(
        self,
        source: str | Path,
        *,
        analysis_type: AnalysisType | str = AnalysisType.FULL,
        language_hint: str = "auto",
        chain_of_custody: bool = True,
    ) -> AnalysisResult:
        """Analyze ``source`` and return the JSON-serialisable result."""
        result, _ = self.analyze_with_artifacts(
            source,
            analysis_type=analysis_type,
            language_hint=language_hint,
            chain_of_custody=chain_of_custody,
        )
        return result

    def analyze_with_artifacts(
        self,
        source: str | Path,
        *,
        analysis_type: AnalysisType | str = AnalysisType.FULL,
        language_hint: str = "auto",
        chain_of_custody: bool = True,
    ) -> tuple[AnalysisResult, AnalysisArtifacts | None]:
        """Like :meth:`analyze` but also returns dense arrays for rendering exhibits.

        Artifacts are returned for ``full``/``legal`` analyses (which run
        localization); ``quick`` returns ``None`` for artifacts.
        """
        analysis_type = AnalysisType(analysis_type)
        started = time.perf_counter()
        analysis_id = "vf_" + uuid.uuid4().hex[:10]
        notes: list[str] = []

        # 1. Chain of custody (hash + metadata + edit indicators).
        metadata = collect_metadata(source)

        # 2. Decode + preprocess (raises on bad audio / failed quality gate).
        y, sr = decode_to_waveform(source, self.settings.target_sample_rate)
        y, quality = preprocess(
            y,
            sr,
            min_snr_db=self.settings.min_snr_db,
            min_duration_s=self.settings.min_duration_s,
            max_duration_s=self.settings.max_duration_s,
        )

        # 3. Features.
        bundle = extract(y, sr)

        # 4. Ensemble fusion + calibration.
        fusion = run_fusion(self.detectors, bundle, self.settings)
        calibrated = platt_scale(fusion.fused_prob, self.settings.platt_a, self.settings.platt_b)
        ci = confidence_interval(calibrated, fusion.uncertainty)
        verdict = to_verdict(calibrated, self.settings)

        if self.baseline_only:
            notes.append(
                "Heuristic baseline only: no trained neural weights configured. "
                "Scores are explainable DSP-feature priors, not a trained model output."
            )

        # 5. Localization + 6. Fingerprint (skipped for QUICK).
        segments = []
        fingerprint = None
        naturalness = None
        artifacts: AnalysisArtifacts | None = None
        if analysis_type in (AnalysisType.FULL, AnalysisType.LEGAL):
            loc = localize(self.detectors, y, sr, bundle.mel, self.settings)
            segments = loc.segments
            naturalness = loc.naturalness_score
            fingerprint = match(bundle, self.settings)
            artifacts = AnalysisArtifacts(
                sample_rate=sr, waveform=y, mel=bundle.mel, heatmap=loc.heatmap
            )

        if analysis_type is AnalysisType.LEGAL and not chain_of_custody:
            chain_of_custody = True
            notes.append("Chain-of-custody forced ON for legal analysis_type.")

        result = DetectionResult(
            deepfake_probability=round(calibrated, 4),
            confidence_interval=ci,
            verdict=verdict,
            uncertainty=round(fusion.uncertainty, 4),
            naturalness_score=naturalness,
            segments=segments,
            fingerprint=fingerprint,
            metadata=metadata,
            quality=quality.as_dict(),
        )

        provenance = EngineProvenance(
            engine_version=__version__,
            active_detectors=self.active_detector_names,
            baseline_only=self.baseline_only,
            detector_scores=fusion.scores,
            notes=notes,
        )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        report_url = None  # PDF report generation is Month-2 scope.

        analysis_result = AnalysisResult(
            analysis_id=analysis_id,
            status="completed",
            analysis_type=analysis_type,
            processing_time_ms=elapsed_ms,
            result=result,
            provenance=provenance,
            report_url=report_url,
        )
        return analysis_result, artifacts


def analyze_file(source: str | Path, **kwargs) -> AnalysisResult:
    """Convenience one-shot analysis using default settings."""
    return Engine().analyze(source, **kwargs)
