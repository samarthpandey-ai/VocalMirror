"""
modules/fusion.py
─────────────────
VocalMirror Fusion Engine
Combines acoustic and NLP module outputs into 5 radar chart dimensions
plus the core Incongruence Score.

No external ML calls happen here — pure array arithmetic.
"""

from __future__ import annotations
import math
import numpy as np
from typing import TypedDict


# ──────────────────────────────────────────────────────────────────────────────
# Safe conversion helper
# ──────────────────────────────────────────────────────────────────────────────

def _safe_float(val: any, default: float = 0.0) -> float:
    """Safely convert a value to a float, falling back to a default if None or NaN."""
    if val is None:
        return default
    try:
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return default
        return f_val
    except (ValueError, TypeError):
        return default


# ──────────────────────────────────────────────────────────────────────────────
# Type definitions
# ──────────────────────────────────────────────────────────────────────────────

class AcousticScores(TypedDict):
    acoustic_nervousness: float
    vocal_confidence: float
    pitch_variance_norm: float
    speech_rate_norm: float
    pause_freq_norm: float


class NLPScores(TypedDict):
    linguistic_sentiment: float
    linguistic_confidence: float
    filler_word_density: float
    vocab_richness: float
    syntactic_fluency: float


class RadarDimensions(TypedDict):
    vocal_confidence: float       # [0,1]  higher = calmer voice
    clarity: float                # [0,1]  vocab richness × syntactic fluency
    filler_word_density: float    # [0,1]  inverted: higher = fewer fillers
    linguistic_sentiment: float   # [0,1]  positive sentiment
    incongruence_score: float     # [0,1]  gap between vocal & linguistic confidence
    congruence: float             # [0,1]  congruence = 1.0 - incongruence_score


class FusionResult(TypedDict):
    radar: RadarDimensions
    interpretation: str
    severity: str   # "low" | "medium" | "high"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float to [lo, hi]."""
    return float(np.clip(value, lo, hi))


def _interpret_incongruence(score: float) -> tuple[str, str]:
    """
    Map incongruence score to a human-readable message and severity level.

    Args:
        score: Incongruence score in [0, 1].

    Returns:
        (interpretation_text, severity)
    """
    if score < 0.15:
        return (
            "✅ Your voice and words are highly aligned. "
            "You sound as confident as you speak — authentic and authoritative.",
            "low",
        )
    elif score < 0.30:
        return (
            "🟡 Mild incongruence detected. "
            "Your words convey moderate confidence, but subtle vocal cues suggest "
            "some nervousness. Practice steady breathing and slower pacing.",
            "medium",
        )
    elif score < 0.50:
        return (
            "🟠 Moderate incongruence. "
            "There is a noticeable gap between your confident language and a "
            "nervous vocal pattern. Consider rehearsing with a lower pitch target "
            "and reducing pace.",
            "medium",
        )
    else:
        return (
            "🔴 High incongruence. "
            "Your words project confidence but your voice signals significant "
            "nervousness. Focus on diaphragmatic breathing, deliberate pausing, "
            "and pitch control exercises before your next presentation.",
            "high",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main fusion function
# ──────────────────────────────────────────────────────────────────────────────

def fuse_scores(
    acoustic: AcousticScores,
    nlp: NLPScores,
) -> FusionResult:
    """
    Fuse acoustic and NLP scores into the 5-dimension radar chart values
    and compute the Incongruence Score and Congruence Score.

    Dimensions
    ----------
    1. vocal_confidence      = 1 - acoustic_nervousness
    2. clarity               = vocab_richness × syntactic_fluency
    3. filler_word_density   = 1 - filler_density  (inverted so higher = better)
    4. linguistic_sentiment  = positive sentiment probability
    5. incongruence_score    = |vocal_confidence - linguistic_confidence|
    6. congruence            = 1 - incongruence_score

    Args:
        acoustic: Output of compute_acoustic_nervousness()
        nlp: Aggregated NLP module outputs.

    Returns:
        FusionResult with radar dimensions, interpretation text, and severity.
    """
    # ── Dimension 1: Vocal Confidence ──────────────────────────────────────
    vocal_confidence = _clip(_safe_float(acoustic.get("vocal_confidence"), 0.5))

    # ── Dimension 2: Clarity (vocab richness × syntactic fluency) ──────────
    vocab_richness = _safe_float(nlp.get("vocab_richness"), 0.5)
    syntactic_fluency = _safe_float(nlp.get("syntactic_fluency"), 0.5)
    clarity = _clip(vocab_richness * syntactic_fluency)

    # ── Dimension 3: Filler Word Density (inverted) ─────────────────────────
    filler_word_density = _safe_float(nlp.get("filler_word_density"), 0.0)
    filler_word_density_score = _clip(1.0 - filler_word_density)

    # ── Dimension 4: Linguistic Sentiment ───────────────────────────────────
    linguistic_sentiment = _clip(_safe_float(nlp.get("linguistic_sentiment"), 0.5))

    # ── Dimension 5: Incongruence Score & Congruence ────────────────────────
    # Weighted linguistic confidence: 60% emotion proxy + 40% sentiment
    linguistic_confidence_val = _safe_float(nlp.get("linguistic_confidence"), 0.5)
    linguistic_confidence = _clip(
        0.6 * linguistic_confidence_val
        + 0.4 * linguistic_sentiment
    )
    incongruence_score = _clip(abs(vocal_confidence - linguistic_confidence))
    congruence = _clip(1.0 - incongruence_score)

    radar: RadarDimensions = {
        "vocal_confidence": vocal_confidence,
        "clarity": clarity,
        "filler_word_density": filler_word_density_score,
        "linguistic_sentiment": linguistic_sentiment,
        "incongruence_score": incongruence_score,
        "congruence": congruence,
    }

    interpretation, severity = _interpret_incongruence(incongruence_score)

    return FusionResult(
        radar=radar,
        interpretation=interpretation,
        severity=severity,
    )


def aggregate_nlp_outputs(
    sentiment: dict,
    confidence: dict,
    fillers: dict,
    vocab: dict,
) -> NLPScores:
    """
    Flatten individual NLP module outputs into a single NLPScores dict
    for use with fuse_scores().

    Args:
        sentiment: Output of score_sentiment()
        confidence: Output of score_confidence()
        fillers: Output of detect_fillers()
        vocab: Output of analyze_vocabulary()

    Returns:
        NLPScores typed dict.
    """
    return NLPScores(
        linguistic_sentiment=_safe_float(sentiment.get("linguistic_sentiment"), 0.5),
        linguistic_confidence=_safe_float(confidence.get("linguistic_confidence"), 0.5),
        filler_word_density=_safe_float(fillers.get("filler_word_density"), 0.0),
        vocab_richness=_safe_float(vocab.get("vocab_richness"), 0.5),
        syntactic_fluency=_safe_float(vocab.get("syntactic_fluency"), 0.5),
    )
