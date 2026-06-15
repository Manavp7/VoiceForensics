"""Nearest-neighbour source attribution over the signature feature space.

Maps a feature bundle to a probability distribution over known generation models
(plus a GENUINE template). Distances are computed in a standardized (z-scored)
feature space; if the closest template is implausibly far, the source is reported
as ``UNKNOWN`` to avoid over-claiming.
"""

from __future__ import annotations

import math

from voiceforensics.config import Settings
from voiceforensics.features.extractor import FeatureBundle
from voiceforensics.fingerprint.signatures import SignatureDB, load_signature_db
from voiceforensics.schemas import Fingerprint

_TEMPERATURE = 1.0


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def match(bundle: FeatureBundle, settings: Settings) -> Fingerprint:
    db: SignatureDB = load_signature_db(settings.signature_db_path)
    query = db.standardize(bundle.scalar_features)

    distances: dict[str, float] = {}
    for tmpl in db.templates:
        distances[tmpl.source_name] = _euclidean(query, db.standardize(tmpl.centroid))

    if not distances:
        return Fingerprint(probable_source="UNKNOWN", confidence=0.0)

    nearest = min(distances, key=distances.get)  # type: ignore[arg-type]
    min_dist = distances[nearest]

    # Softmax over negative distances → a probability distribution.
    neg = {k: -d / _TEMPERATURE for k, d in distances.items()}
    m = max(neg.values())
    exp = {k: math.exp(v - m) for k, v in neg.items()}
    z = sum(exp.values()) or 1.0
    distribution = {k: round(v / z, 4) for k, v in exp.items()}

    if min_dist > settings.fingerprint_max_distance:
        return Fingerprint(
            probable_source="UNKNOWN",
            confidence=0.0,
            alternative_sources=[
                f"{k} ({distribution[k]:.2f})"
                for k in sorted(distribution, key=distribution.get, reverse=True)[:3]  # type: ignore[arg-type]
            ],
            distribution=distribution,
        )

    ordered = sorted(distribution, key=distribution.get, reverse=True)  # type: ignore[arg-type]
    alternatives = [f"{k} ({distribution[k]:.2f})" for k in ordered if k != nearest][:3]
    return Fingerprint(
        probable_source=nearest,
        confidence=float(distribution[nearest]),
        alternative_sources=alternatives,
        distribution=distribution,
    )
