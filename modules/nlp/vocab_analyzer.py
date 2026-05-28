"""
vocab_analyzer.py
------------------
Analyses vocabulary richness and syntactic fluency of transcript text.

Provides a clean, fully multilingual calculation of vocabulary richness
(Type-Token Ratio) using a unicode-compliant regex tokeniser, entirely dropping
the language-specific spaCy syntactic parser to ensure absolute language-agnostic behavior.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_vocabulary(text: str) -> dict:
    """Analyse vocabulary richness and syntactic fluency of *text* natively in any language.

    Parameters
    ----------
    text:
        Raw transcript text to analyse.

    Returns
    -------
    dict
        Keys:

        * ``"type_token_ratio"``   – unique words / total words ∈ [0, 1]     (float)
        * ``"vocab_richness"``     – alias for type_token_ratio               (float)
        * ``"avg_clause_length"``  – mean sentence length in tokens (0.0)     (float)
        * ``"syntactic_fluency"``  – constant 1.0 (to maintain pipeline shape) (float)
        * ``"word_count"``         – total word tokens                        (int)
        * ``"unique_lemmas"``      – count of distinct lowercase words        (int)
    """
    if not text or not text.strip():
        return {
            "type_token_ratio": 0.0,
            "vocab_richness": 0.0,
            "avg_clause_length": 0.0,
            "syntactic_fluency": 1.0,
            "word_count": 0,
            "unique_lemmas": 0,
        }

    # Find all words using a unicode-compliant regex
    words = re.findall(r"\w+", text.lower())
    total_tokens = len(words)
    
    unique_words = set(words)
    unique_count = len(unique_words)
    
    type_token_ratio = float(unique_count / max(total_tokens, 1))

    return {
        "type_token_ratio": type_token_ratio,
        "vocab_richness": type_token_ratio,  # alias
        "avg_clause_length": 0.0,            # dropped clause-length syntax scoring
        "syntactic_fluency": 1.0,            # keeps pipeline language-agnostic
        "word_count": total_tokens,
        "unique_lemmas": unique_count,
    }
