#!/usr/bin/env python3
"""End-to-end GPU training orchestrator for VoiceForensics.

Trains the weight-gated neural backends on a labelled dataset, fits calibration,
and writes a manifest + a ready-to-source ``.env`` so the engine can immediately use
the trained weights.

Dataset layout (any ffmpeg-readable audio)::

    dataset/
      real/   *.flac|wav|...
      fake/   *.flac|wav|...

Typical GPU run (see notebooks/train_voiceforensics.ipynb for a turnkey Colab/Kaggle):

    python scripts/train_asvspoof.py /data/asvspoof_la \
        --models rawnet2 aasist --epochs 30 --batch-size 32 --out weights/

This script only depends on the installed package; it works on CPU too (for smoke
tests) and auto-selects CUDA when available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="dir with real/ and fake/ subdirectories")
    parser.add_argument(
        "--models", nargs="+", default=["rawnet2", "aasist"], choices=["rawnet2", "aasist"]
    )
    parser.add_argument("--out", default="weights", help="output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    parser.add_argument("--calibrate", action="store_true", help="also fit Platt + weights")
    args = parser.parse_args(argv)

    # Imported lazily so --help works without torch installed.
    from voiceforensics.training.train import resolve_device, train_model

    device = resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] device={device} dataset={args.dataset} models={args.models}", flush=True)

    manifest: dict = {"device": device, "dataset": str(args.dataset), "models": {}}
    for model_name in args.models:
        ckpt = out_dir / f"{model_name}.pth"
        print(f"[train] training {model_name} -> {ckpt}", flush=True)
        result = train_model(
            args.dataset,
            model_name,
            ckpt,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_frac=args.val_frac,
            device=device,
        )
        manifest["models"][model_name] = {
            "checkpoint": str(result.checkpoint_path),
            "best_val_eer": result.best_val_eer,
            "epochs_run": result.epochs_run,
            "train_size": result.train_size,
            "val_size": result.val_size,
        }
        print(
            f"[train] {model_name}: best val EER={result.best_val_eer:.4f} "
            f"(epochs={result.epochs_run})",
            flush=True,
        )

    # Build an .env that activates the trained backends.
    env_lines = []
    for model_name, info in manifest["models"].items():
        var = f"VF_{model_name.upper()}_WEIGHTS_PATH"
        env_lines.append(f"{var}={Path(info['checkpoint']).resolve()}")

    if args.calibrate:
        # Use the freshly trained weights when scoring for calibration.
        import os

        from voiceforensics.pipeline import Engine
        from voiceforensics.tools.benchmark import _iter_audio
        from voiceforensics.training.calibrate import as_env_snippet, fit_platt

        for line in env_lines:
            k, v = line.split("=", 1)
            os.environ[k] = v
        from voiceforensics.config import get_settings

        get_settings.cache_clear()

        engine = Engine()
        fused: list[float] = []
        labels: list[int] = []
        for label, sub in ((0, "real"), (1, "fake")):
            for path in _iter_audio(Path(args.dataset) / sub):
                res = engine.analyze(path, analysis_type="quick")
                fused.append(res.result.deepfake_probability)
                labels.append(label)
        a, b = fit_platt(fused, labels)
        manifest["calibration"] = {"platt_a": a, "platt_b": b}
        env_lines.append("")
        env_lines.append(as_env_snippet(a, b))

    env_path = out_dir / "trained.env"
    env_path.write_text("\n".join(env_lines) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[train] wrote {env_path} and manifest.json", flush=True)
    print(f"[train] activate with:  source {env_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
