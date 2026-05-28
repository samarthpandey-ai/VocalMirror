"""
vocab_analyzer.py
------------------
Analyses vocabulary richness and syntactic fluency of transcript text using
spaCy's ``en_core_web_sm`` model.

The module-level spaCy cache is compatible with ``filler_detector.py`` —
both modules import from the same global within the same process, so the
model is loaded only once.
"""

from __future__ import annotations

import math
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level lazy cache (compatible with filler_detector._nlp)
# ---------------------------------------------------------------------------
_nlp: Optional[object] = None  # type: ignore[assignment]


def _get_nlp():
    """Return the cached spaCy model, loading it on first call.

    The model is loaded with the *ner* component disabled for speed; the
    *tok2vec*, *tagger*, and *attribute_ruler* components are kept so that
    lemmatisation is available.
    """
    global _nlp
    if _nlp is None:
        import spacy  # type: ignore

        try:
            _nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except OSError:
            from spacy.cli import download  # type: ignore

            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm", disable=["ner"])
    return _nlp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_vocabulary(text: str) -> dict:
    """Analyse vocabulary richness and syntactic fluency of *text*.

    Parameters
    ----------
    text:
        Raw transcript text to analyse.

    Returns
    -------
    dict
        Keys:

        * ``"type_token_ratio"``   – unique lemmas / total tokens ∈ [0, 1]   (float)
        * ``"vocab_richness"``     – alias for type_token_ratio               (float)
        * ``"avg_clause_length"``  – mean sentence length in tokens           (float)
        * ``"syntactic_fluency"``  – tanh(avg_clause_length / 15.0) ∈ [0, 1] (float)
        * ``"word_count"``         – total alpha tokens                       (int)
        * ``"unique_lemmas"``      – count of distinct non-stop alpha lemmas  (int)

    Notes
    -----
    ``syntactic_fluency`` is computed as ``math.tanh(avg_clause_length / 15.0)``.
    This maps typical English sentences (10–20 words) to approximately 0.6–0.9,
    rewarding well-developed sentences without penalising concise ones heavily.
    """
    if not text or not text.strip():
        return {
            "type_token_ratio": 0.0,
            "vocab_richness": 0.0,
            "avg_clause_length": 0.0,
            "syntactic_fluency": 0.0,
            "word_count": 0,
            "unique_lemmas": 0,
        }

    nlp = _get_nlp()
    doc = nlp(text)

    # -----------------------------------------------------------------------
    # Collect alpha tokens (words, no punctuation or whitespace)
    # -----------------------------------------------------------------------
    all_tokens = [tok for tok in doc if tok.is_alpha]
    total_tokens = len(all_tokens)

    # -----------------------------------------------------------------------
    # Type-token ratio: unique lemmas over non-stop alpha tokens
    # -----------------------------------------------------------------------
    content_lemmas = [
        tok.lemma_.lower()
        for tok in all_tokens
        if not tok.is_stop
    ]
    unique_lemmas_set = set(content_lemmas)
    unique_lemmas = len(unique_lemmas_set)
    type_token_ratio = float(unique_lemmas / max(total_tokens, 1))

    # -----------------------------------------------------------------------
    # Average sentence (clause) length
    # -----------------------------------------------------------------------
    sentences = list(doc.sents)
    if sentences:
        sent_lengths = [
            len([tok for tok in sent if tok.is_alpha])
            for sent in sentences
        ]
        avg_clause_length = float(sum(sent_lengths) / len(sent_lengths))
    else:
        avg_clause_length = float(total_tokens)

    # -----------------------------------------------------------------------
    # Syntactic fluency via tanh normalisation
    # -----------------------------------------------------------------------
    syntactic_fluency = float(math.tanh(avg_clause_length / 15.0))

    return {
        "type_token_ratio": type_token_ratio,
        "vocab_richness": type_token_ratio,  # alias
        "avg_clause_length": avg_clause_length,
        "syntactic_fluency": syntactic_fluency,
        "word_count": total_tokens,
        "unique_lemmas": unique_lemmas,
    }
