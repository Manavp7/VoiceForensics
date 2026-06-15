#!/usr/bin/env python3
"""Generate synthetic sample audio for local experimentation.

These are crude proxies (see ``tests/synth.py``), NOT real speech. They exist so
you can exercise the CLI/API end-to-end without any dataset downloads.

Usage:
    python scripts/make_sample_audio.py --out samples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable so we can reuse the test synthesisers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import synth  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic sample audio.")
    parser.add_argument("--out", default="samples", help="output directory")
    parser.add_argument("--duration", type=float, default=4.0, help="clip duration (s)")
    parser.add_argument("--encode", action="store_true", help="also write mp3/ogg/m4a via ffmpeg")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clips = {
        "genuine_like.wav": synth.genuine_like(args.duration, seed=0),
        "synthetic_like.wav": synth.synthetic_like(args.duration, seed=1),
        "spliced.wav": synth.spliced(args.duration, seed=7),
    }
    for name, y in clips.items():
        path = synth.write_wav(out / name, y)
        print(f"wrote {path}")
        if args.encode and name == "genuine_like.wav":
            for ext in ("mp3", "ogg", "m4a"):
                try:
                    enc = synth.encode_with_ffmpeg(path, out / f"genuine_like.{ext}")
                    print(f"wrote {enc}")
                except Exception as exc:  # noqa: BLE001
                    print(f"skip {ext}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
