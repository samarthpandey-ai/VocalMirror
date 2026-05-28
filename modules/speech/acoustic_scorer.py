"""
acoustic_scorer.py
------------------
Compute a normalised acoustic nervousness score [0, 1] for VocalMirror.

Reads RAVDESS-derived baselines from ``baselines.json`` (same directory),
z-scores each acoustic feature, maps through a sigmoid to [0, 1], and
combines via a weighted sum.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Path to the baselines file, resolved relative to this source file
_BASELINES_PATH: Path = Path(__file__).parent / "baselines.json"

# Cached baselines dict so the file is read only once per process
_baselines_cache: dict[str, Any] | None = None


def _load_baselines() -> dict[str, Any]:
    """
    Load and cache the RAVDESS baselines from ``baselines.json``.

    Returns:
        Parsed JSON as a Python dict.

    Raises:
        FileNotFoundError: If ``baselines.json`` is missing.
        ValueError:        If the JSON is malformed or missing required keys.
    """
    global _baselines_cache

    if _baselines_cache is not None:
        return _baselines_cache

    if not _BASELINES_PATH.is_file():
        raise FileNotFoundError(
            f"Baselines file not found at '{_BASELINES_PATH}'. "
            "Ensure 'baselines.json' is present in the speech module directory."
        )

    try:
        with _BASELINES_PATH.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"'baselines.json' contains invalid JSON: {exc}"
        ) from exc

    # Validate expected keys
    required_keys = {
        "calm": {
            "pitch_variance_mean", "pitch_variance_std",
            "speech_rate_mean", "speech_rate_std",
            "pause_freq_mean", "pause_freq_std",
        }
    }
    for section, keys in required_keys.items():
        if section not in data:
            raise ValueError(f"baselines.json missing section '{section}'.")
        missing = keys - data[section].keys()
        if missing:
            raise ValueError(
                f"baselines.json['{section}'] is missing keys: {missing}"
            )

    _baselines_cache = data
    logger.info("Loaded acoustic baselines from '%s'.", _BASELINES_PATH)
    return _baselines_cache


def _sigmoid(z: float) -> float:
    """
    Numerically stable sigmoid function.

    Args:
        z: Input value.

    Returns:
        Value in (0, 1).
    """
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    # Equivalent form that avoids exp overflow for large negative z
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def _z_score_sigmoid(value: float, mean: float, std: float) -> float:
    """
    Compute z-score of ``value`` and map through sigmoid to [0, 1].

    The sigmoid is centred at 0, so features at the baseline mean return 0.5.
    Features above the mean (e.g., elevated speech rate) return > 0.5.

    Args:
        value: The feature value to normalise.
        mean:  Baseline mean.
        std:   Baseline standard deviation.

    Returns:
        float in (0, 1).
    """
    z = (value - mean) / (std + 1e-9)
    return _sigmoid(z)


def compute_acoustic_nervousness(
    features: dict,
    model_size_hint: str = "small",
) -> dict:
    """
    Compute a normalised acoustic nervousness score from extracted features.

    Reads RAVDESS-calibrated baselines ('calm' class) and produces:
      - Per-feature normalised scores via z-score + sigmoid.
      - A weighted composite nervousness score.
      - A complementary vocal confidence score.

    Weighting rationale (sum = 1.0):
      0.4 × pitch_variance  — high F0 variability is a strong nervousness cue.
      0.3 × (1 - speech_rate) — slower-than-baseline speech often signals anxiety.
      0.3 × pause_frequency  — frequent pauses indicate disfluency / hesitation.

    Args:
        features: dict output of ``extract_acoustic_features()``.
            Required keys: 'pitch_variance', 'speech_rate', 'pause_frequency'.
        model_size_hint: Unused; retained for API compatibility with future
            model-dependent calibration (e.g., 'tiny' vs 'small').

    Returns:
        dict with keys:
            "acoustic_nervousness"  (float) — composite score [0, 1].
            "vocal_confidence"      (float) — 1 - acoustic_nervousness [0, 1].
            "pitch_variance_norm"   (float) — normalised pitch variance [0, 1].
            "speech_rate_norm"      (float) — normalised speech rate [0, 1].
            "pause_freq_norm"       (float) — normalised pause frequency [0, 1].

    Raises:
        KeyError:          If required feature keys are absent.
        FileNotFoundError: If baselines.json cannot be found.
        ValueError:        If baselines.json is malformed.

    Example:
        >>> from modules.speech import extract_acoustic_features, compute_acoustic_nervousness
        >>> feats = extract_acoustic_features("speech.wav")
        >>> scores = compute_acoustic_nervousness(feats)
        >>> print(scores["acoustic_nervousness"])
    """
    # --- Validate input features ---
    required_feature_keys = {"pitch_variance", "speech_rate", "pause_frequency"}
    missing = required_feature_keys - features.keys()
    if missing:
        raise KeyError(
            f"features dict is missing required keys: {missing}. "
            "Pass the output of extract_acoustic_features() directly."
        )

    # --- Load baselines ---
    baselines = _load_baselines()
    calm = baselines["calm"]

    pitch_var: float = float(features["pitch_variance"])
    speech_rate: float = float(features["speech_rate"])
    pause_freq: float = float(features["pause_frequency"])

    # --- Per-feature normalisation (z-score → sigmoid → [0, 1]) ---
    pitch_variance_norm = _z_score_sigmoid(
        pitch_var,
        mean=calm["pitch_variance_mean"],
        std=calm["pitch_variance_std"],
    )

    speech_rate_norm = _z_score_sigmoid(
        speech_rate,
        mean=calm["speech_rate_mean"],
        std=calm["speech_rate_std"],
    )

    pause_freq_norm = _z_score_sigmoid(
        pause_freq,
        mean=calm["pause_freq_mean"],
        std=calm["pause_freq_std"],
    )

    # --- Weighted composite nervousness ---
    # Higher pitch variance → more nervous  (direct)
    # Lower speech rate    → more nervous   (inverted: use 1 - norm)
    # Higher pause freq    → more nervous   (direct)
    acoustic_nervousness = (
        0.4 * pitch_variance_norm
        + 0.3 * (1.0 - speech_rate_norm)
        + 0.3 * pause_freq_norm
    )

    vocal_confidence = 1.0 - acoustic_nervousness

    # --- Clip all outputs to [0, 1] ---
    def _clip(v: float) -> float:
        return float(np.clip(v, 0.0, 1.0))

    return {
        "acoustic_nervousness": _clip(acoustic_nervousness),
        "vocal_confidence": _clip(vocal_confidence),
        "pitch_variance_norm": _clip(pitch_variance_norm),
        "speech_rate_norm": _clip(speech_rate_norm),
        "pause_freq_norm": _clip(pause_freq_norm),
    }
