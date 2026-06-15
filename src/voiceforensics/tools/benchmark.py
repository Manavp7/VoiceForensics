"""Benchmark the engine on a labelled folder.

Expected layout::

    dataset/
      real/   *.wav|mp3|...
      fake/   *.wav|mp3|...

Reports EER, AUC, accuracy, a simplified min t-DCF, and a threshold sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from voiceforensics.audio.io import AudioDecodeError
from voiceforensics.audio.preprocess import QualityGateError
from voiceforensics.metrics import compute_min_tdcf, evaluate
from voiceforensics.pipeline import Engine

_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".aac", ".opus"}


@dataclass
class BenchmarkResult:
    metrics: dict
    min_tdcf: float
    threshold_sweep: list[dict]
    skipped: list[str] = field(default_factory=list)
    n_scored: int = 0


def _iter_audio(folder: Path):
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in _AUDIO_EXTS:
            yield p


def run_benchmark(dataset_dir: str | Path, *, engine: Engine | None = None) -> BenchmarkResult:
    dataset_dir = Path(dataset_dir)
    real_dir = dataset_dir / "real"
    fake_dir = dataset_dir / "fake"
    if not real_dir.is_dir() or not fake_dir.is_dir():
        raise FileNotFoundError(f"expected '{real_dir}' and '{fake_dir}' subdirectories")

    engine = engine or Engine()
    scores: list[float] = []
    labels: list[int] = []
    skipped: list[str] = []

    for label, folder in ((0, real_dir), (1, fake_dir)):
        for path in _iter_audio(folder):
            try:
                result = engine.analyze(path, analysis_type="quick")
            except (AudioDecodeError, QualityGateError) as exc:
                skipped.append(f"{path}: {exc}")
                continue
            scores.append(result.result.deepfake_probability)
            labels.append(label)

    if not scores:
        raise ValueError("no audio could be scored")

    metrics = evaluate(scores, labels)
    min_tdcf = compute_min_tdcf(scores, labels)

    sweep = []
    for t in [i / 10 for i in range(1, 10)]:
        m = evaluate(scores, labels, threshold=t)
        sweep.append({"threshold": t, "accuracy": round(m.accuracy, 4)})

    return BenchmarkResult(
        metrics=metrics.as_dict(),
        min_tdcf=round(min_tdcf, 4),
        threshold_sweep=sweep,
        skipped=skipped,
        n_scored=len(scores),
    )
