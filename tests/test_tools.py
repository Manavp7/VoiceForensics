"""Tests for the benchmark and fingerprint-builder tools."""

from __future__ import annotations

from tests import synth
from voiceforensics.config import Settings
from voiceforensics.features.extractor import FeatureBundle
from voiceforensics.fingerprint.matcher import match
from voiceforensics.fingerprint.signatures import load_signature_db
from voiceforensics.tools.benchmark import run_benchmark
from voiceforensics.tools.build_fingerprints import write_signature_db

SR = synth.DEFAULT_SR


def _make_dataset(root):
    real = root / "real"
    fake = root / "fake"
    real.mkdir(parents=True)
    fake.mkdir(parents=True)
    for i in range(3):
        synth.write_wav(real / f"r{i}.wav", synth.genuine_like(2.5, seed=i), SR)
        synth.write_wav(fake / f"f{i}.wav", synth.synthetic_like(2.5, seed=100 + i), SR)
    return root


def test_benchmark_runs_and_separates(tmp_path):
    _make_dataset(tmp_path)
    result = run_benchmark(tmp_path)
    assert result.n_scored == 6
    assert 0.0 <= result.metrics["eer"] <= 1.0
    assert 0.0 <= result.metrics["auc"] <= 1.0
    # Heuristic clearly separates the synthetic proxies → strong AUC.
    assert result.metrics["auc"] >= 0.9
    assert len(result.threshold_sweep) == 9


def test_build_fingerprints_roundtrip(tmp_path):
    samples = tmp_path / "samples"
    (samples / "GENUINE").mkdir(parents=True)
    (samples / "FakeTTS").mkdir(parents=True)
    for i in range(2):
        synth.write_wav(samples / "GENUINE" / f"g{i}.wav", synth.genuine_like(2.0, seed=i), SR)
        synth.write_wav(samples / "FakeTTS" / f"t{i}.wav", synth.synthetic_like(2.0, seed=50 + i), SR)

    out = write_signature_db(samples, tmp_path / "sig.json")
    assert out.exists()

    db = load_signature_db(out)
    names = {t.source_name for t in db.templates}
    assert names == {"GENUINE", "FakeTTS"}

    # A synthetic signal should attribute to FakeTTS under the measured DB.
    settings = Settings(signature_db_path=out)
    from voiceforensics.features.extractor import extract

    bundle: FeatureBundle = extract(synth.synthetic_like(2.0, seed=999), SR)
    fp = match(bundle, settings)
    assert fp.probable_source in {"FakeTTS", "GENUINE"}
    assert abs(sum(fp.distribution.values()) - 1.0) < 1e-3
