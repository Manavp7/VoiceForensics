"""Pydantic schemas for engine results and the API surface.

These mirror the product specification's response shape while adding explicit
*provenance* fields so the engine never overstates its confidence.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    AUTHENTIC = "AUTHENTIC"
    LEANING_AUTHENTIC = "LEANING_AUTHENTIC"
    UNCERTAIN = "UNCERTAIN"
    LIKELY_SYNTHETIC = "LIKELY_SYNTHETIC"
    HIGH_CONFIDENCE_SYNTHETIC = "HIGH_CONFIDENCE_SYNTHETIC"


class AnalysisType(str, Enum):
    QUICK = "quick"
    FULL = "full"
    LEGAL = "legal"


# --- Chain of custody ---------------------------------------------------------


class EditIndicators(BaseModel):
    edited: bool = False
    reasons: list[str] = Field(default_factory=list)


class FileMetadata(BaseModel):
    file_hash_sha256: str
    size_bytes: int
    duration_seconds: float | None = None
    format: str | None = None
    codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bitrate_bps: int | None = None
    container_tags: dict[str, str] = Field(default_factory=dict)
    edit_indicators: EditIndicators = Field(default_factory=EditIndicators)


# --- Detector / ensemble ------------------------------------------------------


class ModelScore(BaseModel):
    name: str
    prob_fake: float
    available: bool
    weight: float = 0.0
    raw: dict[str, float] = Field(default_factory=dict)


class Segment(BaseModel):
    start_ms: int
    end_ms: int
    score: float


class Fingerprint(BaseModel):
    probable_source: str
    confidence: float
    alternative_sources: list[str] = Field(default_factory=list)
    distribution: dict[str, float] = Field(default_factory=dict)


class EngineProvenance(BaseModel):
    engine_version: str
    active_detectors: list[str]
    baseline_only: bool
    detector_scores: list[ModelScore] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DetectionResult(BaseModel):
    deepfake_probability: float
    confidence_interval: list[float]
    verdict: Verdict
    uncertainty: float
    naturalness_score: float | None = None
    segments: list[Segment] = Field(default_factory=list)
    fingerprint: Fingerprint | None = None
    metadata: FileMetadata
    quality: dict[str, float] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    analysis_id: str
    status: str = "completed"
    analysis_type: AnalysisType
    processing_time_ms: int
    result: DetectionResult
    provenance: EngineProvenance
    report_url: str | None = None


# --- API request --------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    audio_url: str | None = None
    audio_base64: str | None = None
    analysis_type: AnalysisType = AnalysisType.FULL
    language_hint: str = "auto"
    chain_of_custody: bool = True
    webhook_url: str | None = None
    mode: str = "sync"  # "sync" (default, backward-compatible) | "async"
