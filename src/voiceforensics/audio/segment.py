"""Segmentation: fixed analysis windows and coarse speech-active regions.

True speaker diarization is out of scope for this branch; ``energy_segments``
provides a coarse RMS-based "speech-active region" proxy that is good enough to
drive segment-level localization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Window:
    start_ms: int
    end_ms: int
    start_sample: int
    end_sample: int

    def slice(self, y: np.ndarray) -> np.ndarray:
        return y[self.start_sample : self.end_sample]


def frame_windows(y: np.ndarray, sr: int, window_ms: int, hop_ms: int) -> list[Window]:
    """Split the waveform into overlapping fixed-length windows."""
    win = max(1, int(sr * window_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    n = len(y)
    if n == 0:
        return []
    if n <= win:
        return [Window(0, int(n / sr * 1000), 0, n)]
    windows: list[Window] = []
    start = 0
    while start < n:
        end = min(start + win, n)
        windows.append(
            Window(
                start_ms=int(start / sr * 1000),
                end_ms=int(end / sr * 1000),
                start_sample=start,
                end_sample=end,
            )
        )
        if end >= n:
            break
        start += hop
    return windows


def energy_segments(
    y: np.ndarray,
    sr: int,
    *,
    rms_percentile: float = 35.0,
    min_segment_ms: int = 200,
) -> list[Window]:
    """Return coarse speech-active regions via an adaptive RMS threshold."""
    frame = max(1, int(sr * 0.025))
    hop = max(1, int(sr * 0.010))
    if len(y) < frame:
        return [Window(0, int(len(y) / sr * 1000), 0, len(y))] if len(y) else []

    n_frames = 1 + (len(y) - frame) // hop
    rms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = y[i * hop : i * hop + frame]
        rms[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)

    threshold = float(np.percentile(rms, rms_percentile))
    active = rms > threshold

    segments: list[Window] = []
    min_frames = max(1, int(min_segment_ms / 10))
    i = 0
    while i < n_frames:
        if active[i]:
            j = i
            while j < n_frames and active[j]:
                j += 1
            if (j - i) >= min_frames:
                start_sample = i * hop
                end_sample = min(len(y), j * hop + frame)
                segments.append(
                    Window(
                        start_ms=int(start_sample / sr * 1000),
                        end_ms=int(end_sample / sr * 1000),
                        start_sample=start_sample,
                        end_sample=end_sample,
                    )
                )
            i = j
        else:
            i += 1

    if not segments:
        segments = [Window(0, int(len(y) / sr * 1000), 0, len(y))]
    return segments
