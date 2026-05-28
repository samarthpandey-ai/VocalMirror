"""
filler_detector.py
-------------------
Detects filler words and phrases in spoken-word transcripts using a
combination of regex (for multi-word fillers) and spaCy token matching
(for single-word fillers).

spaCy model ``en_core_web_sm`` is lazy-loaded on first call.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Filler word list
# ---------------------------------------------------------------------------
FILLER_WORDS: list[str] = [
    "um", "uh", "er", "ah", "like", "you know", "I mean",
    "basically", "literally", "actually", "sort of", "kind of",
    "right", "okay", "so", "well", "anyway", "hmm",
]

# Split into multi-word and single-word for efficient matching
_MULTI_WORD_FILLERS: list[str] = [f for f in FILLER_WORDS if " " in f]
_SINGLE_WORD_FILLERS: set[str] = {f.lower() for f in FILLER_WORDS if " " not in f}

# Pre-compile regex patterns for multi-word fillers (case-insensitive, word-boundary)
_MULTI_WORD_PATTERNS: list[tuple[str, re.Pattern]] = [
    (phrase, re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE))
    for phrase in _MULTI_WORD_FILLERS
]

# ---------------------------------------------------------------------------
# Module-level lazy cache
# ---------------------------------------------------------------------------
_nlp: Optional[object] = None  # type: ignore[assignment]


def _get_nlp():
    """Return the cached spaCy model, loading it on first call."""
    global _nlp
    if _nlp is None:
        import spacy  # type: ignore

        try:
            _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            # Fallback: download the model automatically if missing
            from spacy.cli import download  # type: ignore

            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _nlp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_fillers(text: str) -> dict:
    """Detect filler words and phrases in *text*.

    Uses regex for multi-word fillers (e.g. *"you know"*, *"I mean"*) and
    spaCy token-level matching for single-word fillers.

    Parameters
    ----------
    text:
        Raw transcript text to analyse.

    Returns
    -------
    dict
        Keys:

        * ``"filler_count"``      – total number of filler occurrences  (int)
        * ``"total_words"``       – total word count                    (int)
        * ``"filler_density"``    – fillers / total_words ∈ [0, 1]      (float)
        * ``"filler_word_density"``– alias for filler_density           (float)
        * ``"filler_positions"``  – token indices of detected fillers   (list[int])
        * ``"found_fillers"``     – actual filler strings found         (list[str])
    """
    if not text or not text.strip():
        return {
            "filler_count": 0,
            "total_words": 0,
            "filler_density": 0.0,
            "filler_word_density": 0.0,
            "filler_positions": [],
            "found_fillers": [],
        }

    nlp = _get_nlp()
    doc = nlp(text)

    tokens = list(doc)
    total_words = len(tokens)

    # -----------------------------------------------------------------------
    # Step 1: Multi-word fillers via regex
    # We track character spans that are already claimed to avoid double-counting
    # when a single-word in a multi-word filler is also in the single-word list.
    # -----------------------------------------------------------------------
    claimed_char_spans: list[tuple[int, int]] = []
    found_fillers: list[str] = []
    filler_positions: list[int] = []  # token indices

    for phrase, pattern in _MULTI_WORD_PATTERNS:
        for match in pattern.finditer(text):
            start_char, end_char = match.start(), match.end()
            # Skip if already claimed by a previous (longer) match
            if any(s <= start_char < e or s < end_char <= e for s, e in claimed_char_spans):
                continue
            claimed_char_spans.append((start_char, end_char))
            found_fillers.append(match.group())
            # Map char span → token index (first token that starts inside span)
            for i, tok in enumerate(tokens):
                if tok.idx >= start_char and tok.idx < end_char:
                    filler_positions.append(i)
                    break

    # -----------------------------------------------------------------------
    # Step 2: Single-word fillers via spaCy tokens
    # -----------------------------------------------------------------------
    for i, token in enumerate(tokens):
        if token.lower_ in _SINGLE_WORD_FILLERS:
            tok_start = token.idx
            tok_end = token.idx + len(token.text)
            # Skip if this token's character span is already covered
            if any(s <= tok_start < e for s, e in claimed_char_spans):
                continue
            claimed_char_spans.append((tok_start, tok_end))
            found_fillers.append(token.text)
            filler_positions.append(i)

    filler_count = len(found_fillers)
    filler_density = float(min(1.0, filler_count / max(total_words, 1)))

    return {
        "filler_count": filler_count,
        "total_words": total_words,
        "filler_density": filler_density,
        "filler_word_density": filler_density,  # alias
        "filler_positions": sorted(filler_positions),
        "found_fillers": found_fillers,
    }
