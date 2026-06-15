"""Command-line entry point: ``python -m voiceforensics analyze <file>``."""

from __future__ import annotations

import argparse
import sys

from voiceforensics.audio.io import AudioDecodeError
from voiceforensics.audio.preprocess import QualityGateError
from voiceforensics.schemas import AnalysisType


def _cmd_analyze(args: argparse.Namespace) -> int:
    from voiceforensics.pipeline import Engine

    try:
        result = Engine().analyze(
            args.file,
            analysis_type=AnalysisType(args.type),
            language_hint=args.language,
            chain_of_custody=not args.no_custody,
        )
    except QualityGateError as exc:
        print(f"quality gate failed: {exc}", file=sys.stderr)
        return 2
    except AudioDecodeError as exc:
        print(f"decode failed: {exc}", file=sys.stderr)
        return 3

    print(result.model_dump_json(indent=2))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from voiceforensics.pipeline import Engine

    try:
        result, pdf_path = Engine().analyze_to_report(
            args.file, out_dir=args.out_dir, language_hint=args.language
        )
    except QualityGateError as exc:
        print(f"quality gate failed: {exc}", file=sys.stderr)
        return 2
    except AudioDecodeError as exc:
        print(f"decode failed: {exc}", file=sys.stderr)
        return 3

    print(f"verdict: {result.result.verdict.value}")
    print(f"deepfake_probability: {result.result.deepfake_probability}")
    print(f"report written: {pdf_path}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    import json

    from voiceforensics.tools.benchmark import run_benchmark

    result = run_benchmark(args.dataset)
    print(
        json.dumps(
            {
                "n_scored": result.n_scored,
                "metrics": result.metrics,
                "min_tdcf": result.min_tdcf,
                "threshold_sweep": result.threshold_sweep,
                "skipped": result.skipped,
            },
            indent=2,
        )
    )
    return 0


def _cmd_build_fingerprints(args: argparse.Namespace) -> int:
    from voiceforensics.tools.build_fingerprints import write_signature_db

    out = write_signature_db(args.samples, args.out)
    print(f"signature DB written: {out}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from voiceforensics.training.train import train_model

    result = train_model(
        args.dataset,
        args.model,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    print(f"checkpoint: {result.checkpoint_path}")
    print(f"best val EER: {result.best_val_eer:.4f}  (epochs={result.epochs_run}, "
          f"train={result.train_size}, val={result.val_size})")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from voiceforensics.pipeline import Engine
    from voiceforensics.tools.benchmark import _iter_audio
    from voiceforensics.training.calibrate import (
        as_env_snippet,
        fit_ensemble_weights,
        fit_platt,
        improvement,
    )

    engine = Engine()
    fused: list[float] = []
    labels: list[int] = []
    det_probs: dict[str, list[float]] = {}
    from pathlib import Path

    for label, sub in ((0, "real"), (1, "fake")):
        for path in _iter_audio(Path(args.dataset) / sub):
            result = engine.analyze(path, analysis_type="quick")
            fused.append(result.result.deepfake_probability)
            labels.append(label)
            for s in result.provenance.detector_scores:
                if s.available:
                    det_probs.setdefault(s.name, []).append(s.prob_fake)

    a, b = fit_platt(fused, labels)
    weights = fit_ensemble_weights(det_probs, labels) if len(det_probs) > 1 else None
    snippet = as_env_snippet(a, b, weights)
    print(snippet)
    print(f"# log-loss improvement vs. uncalibrated: {improvement(a, b, fused, labels):+.4f}",
          file=sys.stderr)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(snippet + "\n")
        print(f"# written: {args.out}", file=sys.stderr)
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    from voiceforensics import __version__
    from voiceforensics.pipeline import Engine

    engine = Engine()
    print(f"VoiceForensics {__version__}")
    print(f"active detectors: {engine.active_detector_names}")
    print(f"baseline only:    {engine.baseline_only}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voiceforensics", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="analyze an audio file")
    p_analyze.add_argument("file", help="path to audio file (any ffmpeg-readable format)")
    p_analyze.add_argument(
        "--type", choices=[t.value for t in AnalysisType], default="full", help="analysis depth"
    )
    p_analyze.add_argument("--language", default="auto", help="language hint (advisory)")
    p_analyze.add_argument(
        "--no-custody", action="store_true", help="disable chain-of-custody fields"
    )
    p_analyze.set_defaults(func=_cmd_analyze)

    p_report = sub.add_parser("report", help="analyze and render a legal PDF report")
    p_report.add_argument("file", help="path to audio file")
    p_report.add_argument("-o", "--out-dir", default=None, help="output directory for the PDF")
    p_report.add_argument("--language", default="auto", help="language hint (advisory)")
    p_report.set_defaults(func=_cmd_report)

    p_bench = sub.add_parser("benchmark", help="evaluate on a labelled real/ + fake/ dataset")
    p_bench.add_argument("dataset", help="dataset dir containing real/ and fake/ subdirs")
    p_bench.set_defaults(func=_cmd_benchmark)

    p_fp = sub.add_parser("build-fingerprints", help="build a signature DB from labelled samples")
    p_fp.add_argument("samples", help="dir with one subdir per generator (label)")
    p_fp.add_argument("-o", "--out", default="signatures.json", help="output JSON path")
    p_fp.set_defaults(func=_cmd_build_fingerprints)

    p_train = sub.add_parser("train", help="train a neural backend on a labelled dataset")
    p_train.add_argument("dataset", help="dataset dir containing real/ and fake/ subdirs")
    p_train.add_argument("--model", choices=["rawnet2", "aasist"], default="rawnet2")
    p_train.add_argument("-o", "--out", default="weights.pth", help="checkpoint output path")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p_train.set_defaults(func=_cmd_train)

    p_cal = sub.add_parser("calibrate", help="fit Platt scaling + ensemble weights")
    p_cal.add_argument("dataset", help="dataset dir containing real/ and fake/ subdirs")
    p_cal.add_argument("-o", "--out", default=None, help="write .env snippet to this path")
    p_cal.set_defaults(func=_cmd_calibrate)

    p_info = sub.add_parser("info", help="show engine / detector status")
    p_info.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
