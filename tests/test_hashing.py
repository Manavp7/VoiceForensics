"""Tests for chain-of-custody hashing and metadata."""

from __future__ import annotations

import hashlib
import subprocess

from voiceforensics.hashing import (
    collect_metadata,
    detect_edit_indicators,
    sha256_bytes,
    sha256_file,
)


def test_sha256_bytes_matches_hashlib():
    data = b"voiceforensics-test-payload"
    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_sha256_file_stable_and_matches(genuine_wav):
    digest = sha256_file(genuine_wav)
    # Stable across calls.
    assert digest == sha256_file(genuine_wav)
    # Matches a fresh full-file hash.
    expected = hashlib.sha256(genuine_wav.read_bytes()).hexdigest()
    assert digest == expected
    assert len(digest) == 64


def test_sha256_file_matches_sha256sum(genuine_wav):
    try:
        out = subprocess.run(
            ["sha256sum", str(genuine_wav)], capture_output=True, text=True, check=True
        ).stdout.split()[0]
    except (subprocess.SubprocessError, FileNotFoundError):
        return  # sha256sum not present; covered by hashlib comparison above
    assert out == sha256_file(genuine_wav)


def test_collect_metadata_basic(genuine_wav):
    meta = collect_metadata(genuine_wav)
    assert meta.file_hash_sha256 == sha256_file(genuine_wav)
    assert meta.size_bytes > 0


def test_collect_metadata_with_ffprobe(genuine_wav, has_ffmpeg):
    meta = collect_metadata(genuine_wav)
    if not has_ffmpeg:
        return
    assert meta.sample_rate == 16_000
    assert meta.channels == 1
    assert meta.duration_seconds is not None
    assert 2.5 < meta.duration_seconds < 3.5


def test_edit_indicators_clean_when_no_tags():
    ind = detect_edit_indicators({}, {})
    assert ind.edited is False
    assert ind.reasons == []


def test_edit_indicators_flags_editor_tag():
    ind = detect_edit_indicators({}, {"encoder": "Lavf60.16.100"})
    assert ind.edited is True
    assert any("processing" in r for r in ind.reasons)


def test_edit_indicators_flags_lossy_in_lossless_container():
    probe = {"format": {"format_name": "wav"}, "streams": [{"codec_type": "audio", "codec_name": "mp3"}]}
    ind = detect_edit_indicators(probe, {})
    assert ind.edited is True
    assert any("lossy codec" in r for r in ind.reasons)
