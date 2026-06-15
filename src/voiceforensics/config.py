"""Runtime configuration for the VoiceForensics detection engine.

All settings can be overridden via environment variables prefixed with ``VF_``
(e.g. ``VF_TARGET_SAMPLE_RATE=8000``) or via a ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PACKAGE_ROOT = Path(__file__).resolve().parent
_DEFAULT_SIGNATURE_DB = _PACKAGE_ROOT.parent.parent / "data" / "signatures" / "signatures.json"


class Settings(BaseSettings):
    """Engine settings.

    Notes on integrity: neural backends are *weight-gated*. They only contribute to
    the ensemble when a real weights file is supplied via the corresponding
    ``*_weights_path``. With no weights, the engine runs the transparent heuristic
    baseline only and reports ``baseline_only=True`` in its provenance.
    """

    model_config = SettingsConfigDict(env_prefix="VF_", env_file=".env", extra="ignore")

    # --- Audio / preprocessing -------------------------------------------------
    target_sample_rate: int = 16_000
    min_snr_db: float = 5.0
    min_duration_s: float = 0.4
    max_duration_s: float = 1800.0

    # --- Segmentation / localization ------------------------------------------
    window_ms: int = 500
    hop_ms: int = 250
    vad_rms_percentile: float = 35.0

    # --- Ensemble fusion -------------------------------------------------------
    # Weights are keyed by detector name. Only *available* detectors are used; the
    # weights of the available subset are renormalised at fusion time.
    ensemble_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "heuristic": 1.0,
            "rawnet2": 1.4,
            "aasist": 1.6,
            "whisper_head": 1.2,
        }
    )

    # --- Calibration (Platt scaling on the logit of the fused probability) -----
    # Defaults are identity-ish; tune from a labelled validation set later.
    platt_a: float = 1.0
    platt_b: float = 0.0

    # --- Verdict thresholds (on the calibrated deepfake probability) -----------
    verdict_authentic_max: float = 0.20
    verdict_leaning_authentic_max: float = 0.40
    verdict_uncertain_max: float = 0.60
    verdict_likely_max: float = 0.80
    # >= verdict_likely_max → HIGH_CONFIDENCE_SYNTHETIC

    # --- Uncertainty -----------------------------------------------------------
    mc_dropout_passes: int = 16

    # --- Neural backend weights (None → backend disabled / weight-gated) -------
    rawnet2_weights_path: Path | None = None
    aasist_weights_path: Path | None = None
    whisper_weights_path: Path | None = None

    # --- Fingerprint -----------------------------------------------------------
    signature_db_path: Path = _DEFAULT_SIGNATURE_DB
    fingerprint_max_distance: float = 6.0

    # --- API -------------------------------------------------------------------
    max_download_bytes: int = 50 * 1024 * 1024
    download_timeout_s: float = 20.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
