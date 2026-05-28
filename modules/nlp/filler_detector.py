"""
filler_detector.py
-------------------
Detects filler words and phrases in spoken-word transcripts natively in any language.

Uses regex for multi-word fillers (e.g. "you know", "I mean") and simple unicode
regex word tokenization for single-word fillers, removing the spaCy dependency.
"""

from __future__ import annotations

import re

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
# Public API
# ---------------------------------------------------------------------------

def detect_fillers(text: str) -> dict:
    """Detect filler words and phrases in *text* natively in any language.

    Uses regex for multi-word fillers and unicode-compliant regex tokenization
    for single-word fillers, maintaining compatibility without external NLP libraries.

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

    # Multilingual unicode tokenization
    tokens = []
    for match in re.finditer(r"\w+", text):
        tokens.append({
            "text": match.group(),
            "start": match.start(),
            "end": match.end()
        })
    total_words = len(tokens)

    # claimed_char_spans avoids double-counting single-word fillers that form part of multi-word fillers
    claimed_char_spans: list[tuple[int, int]] = []
    found_fillers: list[str] = []
    filler_positions: list[int] = []  # token indices

    # 1. Multi-word fillers via regex
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
                if tok["start"] >= start_char and tok["start"] < end_char:
                    filler_positions.append(i)
                    break

    # 2. Single-word fillers via regex tokens
    for i, tok in enumerate(tokens):
        if tok["text"].lower() in _SINGLE_WORD_FILLERS:
            tok_start = tok["start"]
            tok_end = tok["end"]
            # Skip if this token's character span is already covered
            if any(s <= tok_start < e for s, e in claimed_char_spans):
                continue
            claimed_char_spans.append((tok_start, tok_end))
            found_fillers.append(tok["text"])
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
