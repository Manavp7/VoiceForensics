"""Detector abstraction.

A :class:`Detector` maps a :class:`FeatureBundle` to a probability that the audio
is synthetic. Neural detectors are *weight-gated*: ``is_available()`` returns
False until real pretrained weights are loaded, so they never contribute
meaningless scores to the ensemble.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from voiceforensics.features.extractor import FeatureBundle
from voiceforensics.schemas import ModelScore


class Detector(ABC):
    name: str = "detector"
    requires_weights: bool = False

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this detector can produce meaningful scores in this environment."""

    @abstractmethod
    def score(self, bundle: FeatureBundle) -> ModelScore:
        """Return a :class:`ModelScore` with ``prob_fake`` in [0, 1]."""

    def _unavailable_score(self) -> ModelScore:
        return ModelScore(name=self.name, prob_fake=0.0, available=False, raw={})
