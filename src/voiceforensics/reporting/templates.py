"""Plain-language text blocks for the forensic report.

Kept separate from layout so the wording (which matters for admissibility) can be
reviewed by counsel without touching rendering code. Verdict interpretations are
deliberately measured and never assert legal conclusions.
"""

from __future__ import annotations

from voiceforensics.schemas import Verdict

METHODOLOGY = (
    "VoiceForensics analyses the submitted audio with an ensemble of independent "
    "detectors. Each detector inspects the recording for traces commonly left by "
    "speech-synthesis and voice-conversion systems \u2014 for example, unnaturally "
    "stable pitch, over-smoothed formant transitions, suppressed high-frequency "
    "energy, hard band-limiting, and over-regular phase structure. The detectors' "
    "outputs are combined into a single probability, which is then statistically "
    "calibrated. A higher probability indicates stronger evidence that the audio is "
    "wholly or partly synthetic. The analysis is fully reproducible: the same file "
    "yields the same measurements."
)

LIMITATIONS = (
    "This report is decision-support evidence, not proof. An elevated probability "
    "indicates the presence of synthesis-consistent artefacts; it does not, on its "
    "own, establish that a recording is fabricated, nor identify any individual. "
    "Heavily compressed, noisy, telephone, or re-encoded audio can reduce accuracy "
    "and may raise or lower scores. Results should be weighed alongside other "
    "evidence and, where stakes are high, corroborated by an independent examiner. "
    "The findings reflect the state of the detection models at the time of analysis."
)

BASELINE_BANNER = (
    "NOTE ON MODEL PROVENANCE: This analysis was produced using the transparent "
    "heuristic baseline only (no trained neural detector weights were configured). "
    "The baseline reports explainable, signal-derived indicators rather than the "
    "output of a trained classifier. Scores should be interpreted accordingly."
)

EXPERT_STATEMENT = (
    "I, the undersigned, conducted a forensic analysis of the audio recording "
    "identified by the SHA-256 hash recorded in this report, using the VoiceForensics "
    "audio-authenticity analysis system. The methodology and findings are as stated "
    "herein. The content hash recorded at the time of submission establishes the "
    "integrity of the analysed file. The opinions expressed are based on the "
    "measurements produced by the system and my professional judgement."
)

GLOSSARY = {
    "Deepfake probability": "Calibrated likelihood (0\u20131) that the audio is synthetic.",
    "Confidence interval": "Plausible range for the probability given measured uncertainty.",
    "Verdict": "A categorical label derived from the calibrated probability.",
    "Formant": "A resonance of the vocal tract; their motion is hard for synthesis to mimic.",
    "Mel spectrogram": "A time\u2013frequency view of the audio on a perceptual frequency scale.",
    "F0 / pitch jitter": "Cycle-to-cycle variation in pitch; synthetic voices tend to be too stable.",
    "Band-limiting": "An abrupt high-frequency cut-off often introduced by synthesis or codecs.",
    "SHA-256": "A cryptographic fingerprint of the exact file bytes (tamper-evidence).",
}

_VERDICT_TEXT: dict[Verdict, str] = {
    Verdict.AUTHENTIC: (
        "The measurements are consistent with authentic human speech. No strong "
        "synthesis-consistent artefacts were detected."
    ),
    Verdict.LEANING_AUTHENTIC: (
        "The measurements lean towards authentic human speech, with only weak "
        "synthesis-consistent indicators present."
    ),
    Verdict.UNCERTAIN: (
        "The evidence is mixed. The analysis cannot confidently distinguish authentic "
        "from synthetic speech for this recording; further examination is advised."
    ),
    Verdict.LIKELY_SYNTHETIC: (
        "The measurements show several synthesis-consistent artefacts, indicating the "
        "audio is likely to be wholly or partly synthetic."
    ),
    Verdict.HIGH_CONFIDENCE_SYNTHETIC: (
        "The measurements show strong and consistent synthesis artefacts, indicating "
        "high confidence that the audio is wholly or partly synthetic."
    ),
}


def verdict_interpretation(verdict: Verdict) -> str:
    return _VERDICT_TEXT.get(verdict, "")
