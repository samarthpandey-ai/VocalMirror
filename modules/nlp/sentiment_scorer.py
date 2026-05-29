"""
sentiment_scorer.py
--------------------
Uses the cardiffnlp/twitter-xlm-roberta-base-sentiment model via
HuggingFace Transformers to score text sentiment.

Lazy-loads the pipeline on first call to avoid import-time overhead.
"""

from __future__ import annotations

import logging
import torch
from transformers import pipeline  # type: ignore

logger = logging.getLogger(__name__)

# Unified device selection: cuda -> mps -> cpu
if torch.cuda.is_available():
    _DEVICE = "cuda"
elif torch.backends.mps.is_available():
    _DEVICE = "mps"
else:
    _DEVICE = "cpu"
logger.info("Using device for Sentiment Analysis: %s", _DEVICE)

# Load in half-precision on CUDA for maximum performance
_torch_dtype = torch.float16 if _DEVICE == "cuda" else torch.float32

# ---------------------------------------------------------------------------
# Module-level eager cache
# ---------------------------------------------------------------------------
_pipeline = pipeline(
    task="text-classification",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    top_k=None,  # return all label scores
    truncation=True,
    max_length=512,
    device=_DEVICE,
    torch_dtype=_torch_dtype,
)


def _get_pipeline():
    """Return the cached sentiment pipeline."""
    return _pipeline


# ---------------------------------------------------------------------------
# Label mapping — the model may expose human-readable names directly or
# fall back to LABEL_0 / LABEL_1 / LABEL_2 style labels.
# ---------------------------------------------------------------------------
_LABEL_MAP: dict[str, str] = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


def score_sentiment(text: str) -> dict:
    """Score the linguistic sentiment of *text* using a RoBERTa-based model.

    Model: ``cardiffnlp/twitter-xlm-roberta-base-sentiment``

    Text longer than 512 tokens is automatically truncated by the pipeline.

    Parameters
    ----------
    text:
        Raw transcript text to analyse.

    Returns
    -------
    dict
        Keys:

        * ``"positive"``  – probability mass on the *positive* class  (float ∈ [0, 1])
        * ``"neutral"``   – probability mass on the *neutral*  class  (float ∈ [0, 1])
        * ``"negative"``  – probability mass on the *negative* class  (float ∈ [0, 1])
        * ``"linguistic_sentiment"`` – alias for the positive score   (float ∈ [0, 1])
    """
    if not text or not text.strip():
        return {
            "positive": 0.0,
            "neutral": 1.0,
            "negative": 0.0,
            "linguistic_sentiment": 0.0,
        }

    pipe = _get_pipeline()

    # The pipeline returns a list of lists when top_k=None:
    # [[{"label": ..., "score": ...}, ...]]
    with torch.no_grad():
        raw: list[list[dict]] = pipe(text)  # type: ignore[operator]
    scores_list: list[dict] = raw[0]

    scores: dict[str, float] = {}
    for item in scores_list:
        label: str = item["label"].lower()
        # Normalise LABEL_0/1/2 style labels to human-readable names
        if label.startswith("label_"):
            label = _LABEL_MAP.get(item["label"].upper(), label)
        scores[label] = float(item["score"])

    positive = scores.get("positive", 0.0)
    neutral = scores.get("neutral", 0.0)
    negative = scores.get("negative", 0.0)

    return {
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "linguistic_sentiment": positive,
    }
