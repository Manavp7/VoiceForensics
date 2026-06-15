"""VoiceForensics — forensic-grade audio deepfake / voice-spoof detection.

Defensive security software. See AGENTS.md for integrity rules.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Public, lightweight re-exports. Heavy modules (pipeline/torch) are imported lazily
# by callers to keep import time and the API/CLI startup fast.
from voiceforensics.config import Settings, get_settings
from voiceforensics.schemas import AnalysisResult

__all__ = ["__version__", "Settings", "get_settings", "AnalysisResult"]
