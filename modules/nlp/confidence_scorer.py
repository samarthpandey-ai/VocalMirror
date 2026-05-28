"""
confidence_scorer.py
---------------------
Estimates speaker confidence from text using two complementary signals:

1. **Emotion model** (j-hartmann/emotion-english-distilroberta-base) —
   the combined probability mass on *neutral* + *joy* is used as a proxy
   for assertive, composed delivery.

2. **Confidence lexicon** (confidence_lexicon.json) — a curated word-list
   of confident and hedge words derived from CommonLit debate transcripts.

Both signals are lazy-loaded on first call.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
import logging
import torch
from transformers import pipeline  # type: ignore

logger = logging.getLogger(__name__)

# Force MPS if available, fallback to CPU
_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info("Using device for Confidence/Emotion Analysis: %s", _DEVICE)

# ---------------------------------------------------------------------------
# Module-level eager caches
# ---------------------------------------------------------------------------
_emotion_pipeline = pipeline(
    task="text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    truncation=True,
    max_length=512,
    device=_DEVICE,
)

# Eagerly load the lexicon
lexicon_path = Path(__file__).parent / "confidence_lexicon.json"
with lexicon_path.open("r", encoding="utf-8") as fh:
    _lexicon = json.load(fh)


def _get_emotion_pipeline():
    """Return the cached emotion pipeline."""
    return _emotion_pipeline


def _get_lexicon() -> dict:
    """Return the parsed confidence lexicon."""
    return _lexicon



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser (no external deps)."""
    return re.findall(r"\b\w+\b", text.lower())


def _compute_emotion_confidence(text: str) -> float:
    """Return the neutral+joy probability mass from the emotion model.

    Parameters
    ----------
    text:
        Raw transcript text (≤512 tokens; longer text is truncated by the
        pipeline).

    Returns
    -------
    float
        Sum of *neutral* and *joy* scores ∈ [0, 1].
    """
    pipe = _get_emotion_pipeline()
    raw: list[list[dict]] = pipe(text)  # type: ignore[operator]
    scores: dict[str, float] = {
        item["label"].lower(): float(item["score"])
        for item in raw[0]
    }
    return scores.get("neutral", 0.0) + scores.get("joy", 0.0)


def _compute_lexicon_confidence(text: str) -> float:
    """Return a lexicon-based confidence score ∈ [0, 1].

    Score = clip((confident_hits − 0.5 × hedge_hits) / max(total_words, 1) × 5, 0, 1)

    Multi-word phrases in the lexicon (e.g. "I think") are matched with a
    simple substring search; single words are checked against the token list.

    Parameters
    ----------
    text:
        Raw transcript text.

    Returns
    -------
    float
        Confidence score ∈ [0, 1].
    """
    lexicon = _get_lexicon()
    confident_words: list[str] = lexicon.get("confident_words", [])
    hedge_words: list[str] = lexicon.get("hedge_words", [])

    lowered = text.lower()
    tokens = _tokenize(text)
    total_words = max(len(tokens), 1)

    def _count_hits(word_list: list[str]) -> float:
        hits = 0.0
        for phrase in word_list:
            phrase_lower = phrase.lower()
            if " " in phrase_lower:
                # Multi-word — substring match on the lowercased text
                hits += lowered.count(phrase_lower)
            else:
                hits += tokens.count(phrase_lower)
        return hits

    confident_hits = _count_hits(confident_words)
    hedge_hits = _count_hits(hedge_words)

    raw_score = (confident_hits - 0.5 * hedge_hits) / total_words * 5.0
    return float(max(0.0, min(1.0, raw_score)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_confidence(text: str) -> dict:
    """Score the linguistic confidence of *text*.

    Uses a two-signal approach:

    * **emotion_confidence** — neutral+joy probability mass from
      ``j-hartmann/emotion-english-distilroberta-base``.
    * **lexicon_confidence** — confident vs. hedge word hit-rate from a
      curated debate-transcript lexicon.

    Parameters
    ----------
    text:
        Raw transcript text to analyse.

    Returns
    -------
    dict
        Keys:

        * ``"emotion_confidence"``   – neutral+joy mass            (float ∈ [0, 1])
        * ``"lexicon_confidence"``   – lexicon hit-rate score      (float ∈ [0, 1])
        * ``"linguistic_confidence"``– weighted combo              (float ∈ [0, 1])
          = 0.6 × emotion_confidence + 0.4 × lexicon_confidence
    """
    if not text or not text.strip():
        return {
            "emotion_confidence": 0.0,
            "lexicon_confidence": 0.0,
            "linguistic_confidence": 0.0,
        }

    emotion_confidence = _compute_emotion_confidence(text)
    lexicon_confidence = _compute_lexicon_confidence(text)
    linguistic_confidence = 0.6 * emotion_confidence + 0.4 * lexicon_confidence

    return {
        "emotion_confidence": float(emotion_confidence),
        "lexicon_confidence": float(lexicon_confidence),
        "linguistic_confidence": float(linguistic_confidence),
    }
