"""Chain-of-custody helpers: SHA-256 hashing, ffprobe metadata, edit indicators.

These functions provide the tamper-evidence foundation of a forensic report:
a stable content hash, objective container/codec metadata, and conservative
heuristics for whether a file shows signs of having been re-encoded or edited.

Edit indicators are intentionally cautious: they surface *reasons to look closer*,
never a definitive claim of tampering.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from voiceforensics.schemas import EditIndicators, FileMetadata

_CHUNK = 64 * 1024

# Encoder tags / software hints that imply the file was produced by an editor or a
# transcoding step rather than a raw capture device. Presence is suggestive, not proof.
_EDITOR_HINTS = (
    "lavf",          # ffmpeg/libavformat (transcode)
    "lavc",          # libavcodec
    "audacity",
    "adobe",
    "premiere",
    "audition",
    "ableton",
    "logic",
    "wavelab",
    "sound forge",
    "ocenaudio",
)


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    import hashlib

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file, read in streaming chunks."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _ffprobe(path: str | Path) -> dict:
    """Run ffprobe and return the parsed JSON (or ``{}`` if unavailable)."""
    if shutil.which("ffprobe") is None:
        return {}
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return {}
        return json.loads(proc.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return {}


def _first_audio_stream(probe: dict) -> dict:
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "audio":
            return stream
    return {}


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def detect_edit_indicators(probe: dict, tags: dict[str, str]) -> EditIndicators:
    """Conservatively flag signs of re-encoding / editing from container metadata."""
    reasons: list[str] = []
    lowered = {k.lower(): str(v) for k, v in tags.items()}

    blob = " ".join(lowered.values()).lower()
    for hint in _EDITOR_HINTS:
        if hint in blob:
            reasons.append(f"encoder/software tag suggests processing: '{hint}'")

    # Conflicting encoder tags across format/streams hint at a transcode chain.
    encoder_values = {
        v for k, v in lowered.items() if k in {"encoder", "encoded_by", "handler_name"} and v
    }
    if len(encoder_values) > 1:
        reasons.append(f"multiple distinct encoder tags present: {sorted(encoder_values)}")

    fmt = probe.get("format", {})
    audio = _first_audio_stream(probe)
    fmt_name = (fmt.get("format_name") or "").lower()
    codec = (audio.get("codec_name") or "").lower()

    # A lossy codec inside a nominally lossless/raw container is a re-wrap red flag.
    if codec in {"mp3", "aac"} and ("wav" in fmt_name or "pcm" in fmt_name):
        reasons.append(f"lossy codec '{codec}' inside container '{fmt_name}'")

    return EditIndicators(edited=bool(reasons), reasons=reasons)


def collect_metadata(path: str | Path) -> FileMetadata:
    """Build a :class:`FileMetadata` (hash + ffprobe details + edit indicators)."""
    path = Path(path)
    file_hash = sha256_file(path)
    size_bytes = path.stat().st_size

    probe = _ffprobe(path)
    fmt = probe.get("format", {})
    audio = _first_audio_stream(probe)

    tags: dict[str, str] = {}
    for source in (fmt.get("tags", {}), audio.get("tags", {})):
        if isinstance(source, dict):
            for k, v in source.items():
                tags[str(k)] = str(v)

    return FileMetadata(
        file_hash_sha256=file_hash,
        size_bytes=size_bytes,
        duration_seconds=_coerce_float(fmt.get("duration")),
        format=fmt.get("format_name"),
        codec=audio.get("codec_name"),
        sample_rate=_coerce_int(audio.get("sample_rate")),
        channels=_coerce_int(audio.get("channels")),
        bitrate_bps=_coerce_int(fmt.get("bit_rate")),
        container_tags=tags,
        edit_indicators=detect_edit_indicators(probe, tags),
    )
