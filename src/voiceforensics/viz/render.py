"""Headless rendering of forensic exhibits (mel spectrogram, anomaly heatmap, waveform).

Uses matplotlib's non-interactive Agg backend so images render in servers / CI with
no display. All functions write a PNG and return its path. Rendering is deterministic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import; headless rendering

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from voiceforensics.schemas import Segment  # noqa: E402

_DPI = 120


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def mel_spectrogram_png(mel: np.ndarray, sr: int, path: str | Path, *, title: str = "Mel spectrogram") -> Path:
    """Render a log-mel spectrogram (``mel`` shape: n_mels x T, in dB)."""
    path = _ensure_parent(path)
    n_mels, n_frames = mel.shape
    duration_s = max(n_frames, 1) * 0.010  # ~10 ms hop used in feature extraction
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(
        mel,
        origin="lower",
        aspect="auto",
        extent=(0.0, duration_s, 0.0, sr / 2.0),
        cmap="magma",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, format="%+2.0f dB", pad=0.01)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return path


def heatmap_png(
    heatmap: np.ndarray,
    segments: list[Segment],
    path: str | Path,
    *,
    title: str = "Synthetic-artifact heatmap",
) -> Path:
    """Render the per-window x mel-band anomaly heatmap with a suspicion strip.

    ``heatmap`` shape: (n_windows, n_mel_bands). ``segments`` align with windows.
    """
    path = _ensure_parent(path)
    if heatmap.size == 0:
        heatmap = np.zeros((1, 1), dtype=np.float32)
    # Orient as (bands, windows) for display.
    grid = heatmap.T
    n_windows = heatmap.shape[0]
    end_ms = segments[-1].end_ms if segments else max(1, n_windows)
    duration_s = end_ms / 1000.0

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 3.6), height_ratios=[1, 4], sharex=True
    )

    scores = [s.score for s in segments] if segments else [0.0]
    centers = (
        [(s.start_ms + s.end_ms) / 2000.0 for s in segments]
        if segments
        else [duration_s / 2]
    )
    ax_top.bar(centers, scores, width=(duration_s / max(len(scores), 1)) * 0.9, color="crimson")
    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel("susp.")
    ax_top.set_title(title)

    im = ax_bot.imshow(
        grid, origin="lower", aspect="auto", extent=(0.0, duration_s, 0, grid.shape[0]),
        cmap="inferno", vmin=0.0, vmax=max(float(grid.max()), 1e-6),
    )
    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_ylabel("Mel band")
    fig.colorbar(im, ax=[ax_top, ax_bot], pad=0.01)
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return path


def waveform_png(
    y: np.ndarray,
    sr: int,
    segments: list[Segment],
    path: str | Path,
    *,
    title: str = "Waveform with flagged segments",
    flag_threshold: float = 0.5,
) -> Path:
    """Render the waveform, shading windows whose suspicion exceeds ``flag_threshold``."""
    path = _ensure_parent(path)
    t = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.plot(t, y, linewidth=0.5, color="#1f3b73")
    for s in segments:
        if s.score >= flag_threshold:
            ax.axvspan(s.start_ms / 1000.0, s.end_ms / 1000.0, color="red", alpha=0.18)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return path
