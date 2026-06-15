"""Build a model-fingerprint signature DB from labelled generator samples.

Expected layout::

    samples/
      GENUINE/      *.wav
      ElevenLabs_v3/ *.wav
      RVC/          *.wav
      ...

Each subdirectory name becomes a source label. For each source we compute the
mean (centroid) of a chosen set of scalar features, and a global mean/std for
standardization. The output JSON matches ``data/signatures/signatures.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from voiceforensics.audio.io import AudioDecodeError, decode_to_waveform
from voiceforensics.audio.preprocess import QualityGateError, preprocess
from voiceforensics.config import get_settings
from voiceforensics.features.extractor import extract

_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".aac", ".opus"}

# Features used for source attribution (must exist in FeatureBundle.scalar_features).
DEFAULT_FEATURE_KEYS = [
    "bandlimit_ratio",
    "hf_energy_ratio_4k",
    "jitter_local",
    "f0_cv",
    "formant_transition_rate",
    "phase_regularity",
    "spectral_rolloff_hz",
]


def _scalar_vector(path: Path, sr: int, keys: list[str]) -> dict[str, float] | None:
    try:
        y, sr = decode_to_waveform(path, sr)
        y, _ = preprocess(y, sr, min_snr_db=0.0, min_duration_s=0.2, max_duration_s=3600.0)
    except (AudioDecodeError, QualityGateError):
        return None
    feats = extract(y, sr).scalar_features
    return {k: float(feats.get(k, 0.0)) for k in keys}


def build_signature_db(
    samples_dir: str | Path,
    *,
    feature_keys: list[str] | None = None,
) -> dict:
    samples_dir = Path(samples_dir)
    keys = feature_keys or DEFAULT_FEATURE_KEYS
    sr = get_settings().target_sample_rate

    per_source: dict[str, list[dict[str, float]]] = {}
    for source_dir in sorted(p for p in samples_dir.iterdir() if p.is_dir()):
        vectors = []
        for path in sorted(source_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in _AUDIO_EXTS:
                vec = _scalar_vector(path, sr, keys)
                if vec is not None:
                    vectors.append(vec)
        if vectors:
            per_source[source_dir.name] = vectors

    if not per_source:
        raise ValueError(f"no labelled audio found under {samples_dir}")

    # Global standardization stats across all samples.
    all_vecs = [v for vs in per_source.values() for v in vs]
    standardization = {}
    for k in keys:
        col = np.array([v[k] for v in all_vecs], dtype=np.float64)
        standardization[k] = {"mean": float(col.mean()), "std": float(col.std() or 1.0)}

    templates = []
    for source, vectors in per_source.items():
        centroid = {k: float(np.mean([v[k] for v in vectors])) for k in keys}
        templates.append(
            {
                "source_name": source,
                "version": "measured",
                "notes": f"Centroid from {len(vectors)} sample(s).",
                "centroid": centroid,
            }
        )

    return {
        "schema_version": 1,
        "description": f"Measured fingerprint DB built from {len(all_vecs)} samples.",
        "feature_keys": keys,
        "standardization": standardization,
        "templates": templates,
    }


def write_signature_db(samples_dir: str | Path, out_path: str | Path) -> Path:
    db = build_signature_db(samples_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(db, indent=2))
    return out_path
