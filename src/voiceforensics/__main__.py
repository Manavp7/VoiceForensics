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

    p_info = sub.add_parser("info", help="show engine / detector status")
    p_info.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
