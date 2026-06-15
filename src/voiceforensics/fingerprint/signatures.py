"""Loading and representation of the generation-model signature database."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class FeatureTemplate:
    source_name: str
    version: str
    notes: str
    centroid: dict[str, float]


@dataclass(frozen=True)
class SignatureDB:
    feature_keys: tuple[str, ...]
    standardization: dict[str, tuple[float, float]]  # key -> (mean, std)
    templates: tuple[FeatureTemplate, ...]
    description: str = ""

    def standardize(self, values: dict[str, float]) -> list[float]:
        out: list[float] = []
        for key in self.feature_keys:
            mean, std = self.standardization.get(key, (0.0, 1.0))
            std = std if std > 1e-12 else 1.0
            out.append((float(values.get(key, mean)) - mean) / std)
        return out


def _load(path: Path) -> SignatureDB:
    data = json.loads(Path(path).read_text())
    keys = tuple(data["feature_keys"])
    standardization = {
        k: (float(v["mean"]), float(v["std"])) for k, v in data["standardization"].items()
    }
    templates = tuple(
        FeatureTemplate(
            source_name=t["source_name"],
            version=t.get("version", ""),
            notes=t.get("notes", ""),
            centroid={k: float(v) for k, v in t["centroid"].items()},
        )
        for t in data["templates"]
    )
    return SignatureDB(
        feature_keys=keys,
        standardization=standardization,
        templates=templates,
        description=data.get("description", ""),
    )


@lru_cache(maxsize=8)
def load_signature_db(path: str | Path) -> SignatureDB:
    """Load (and cache) the signature database at ``path``."""
    return _load(Path(path))
