"""
acoustic_scorer.py
------------------
Compute a normalised acoustic nervousness score [0, 1] for VocalMirror.

Integrates Facebook's Wav2Vec 2.0 (XLS-R) model ('facebook/wav2vec2-large-xlsr-53')
to compute vocal confidence by analyzing the stability and variance of deep
acoustic embeddings. Classical features (pitch variance, pause frequency)
are calculated as secondary indicators to preserve full backward compatibility.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Path to the baselines file, resolved relative to this source file
_BASELINES_PATH: Path = Path(__file__).parent / "baselines.json"

# Cached baselines dict so the file is read only once per process
_baselines_cache: dict[str, Any] | None = None

# Unified device selection: cuda -> mps -> cpu
if torch.cuda.is_available():
    _DEVICE = "cuda"
elif torch.backends.mps.is_available():
    _DEVICE = "mps"
else:
    _DEVICE = "cpu"
logger.info("Using device for XLS-R Wav2Vec2 Acoustic Analysis: %s", _DEVICE)

# Cache for XLS-R feature extractor and model
_feature_extractor: Any = None
_model: Any = None


def _load_baselines() -> dict[str, Any]:
    """Load and cache the RAVDESS baselines from ``baselines.json``."""
    global _baselines_cache

    if _baselines_cache is not None:
        return _baselines_cache

    if not _BASELINES_PATH.is_file():
        raise FileNotFoundError(
            f"Baselines file not found at '{_BASELINES_PATH}'."
        )

    try:
        with _BASELINES_PATH.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"'baselines.json' contains invalid JSON: {exc}"
        ) from exc

    _baselines_cache = data
    return _baselines_cache


def _get_wav2vec2() -> tuple[Any, Any]:
    """Lazy-load and cache Wav2Vec2 feature extractor and model."""
    global _feature_extractor, _model
    if _feature_extractor is None or _model is None:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        logger.info("Loading Wav2Vec2 (XLS-R) model 'facebook/wav2vec2-large-xlsr-53' on %s...", _DEVICE)
        _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        
        # Load in half-precision on CUDA for maximum performance
        torch_dtype = torch.float16 if _DEVICE == "cuda" else torch.float32
        _model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53",
            torch_dtype=torch_dtype
        ).to(_DEVICE)
        logger.info("Wav2Vec2 (XLS-R) model loaded successfully.")
    return _feature_extractor, _model


def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid function."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def _z_score_sigmoid(value: float, mean: float, std: float) -> float:
    """Compute z-score of value and map through sigmoid to [0, 1]."""
    z = (value - mean) / (std + 1e-9)
    return _sigmoid(z)


def compute_acoustic_nervousness(
    features: dict,
    model_size_hint: str = "small",
) -> dict:
    """
    Compute a normalised acoustic nervousness score from extracted features.

    Uses deep Wav2Vec2 sequence embeddings to analyze vocal stability and extract
    vocal confidence, and preserves traditional z-scored metrics for backward compatibility.

    Args:
        features: dict output of ``extract_acoustic_features()``.
            Required keys: 'pitch_variance', 'speech_rate', 'pause_frequency'.
            Optional key: 'raw_audio_16k'.
        model_size_hint: Unused.

    Returns:
        dict with keys:
            "acoustic_nervousness"  (float) — composite score [0, 1].
            "vocal_confidence"      (float) — 1 - acoustic_nervousness [0, 1].
            "pitch_variance_norm"   (float) — normalised pitch variance [0, 1].
            "speech_rate_norm"      (float) — normalised speech rate [0, 1].
            "pause_freq_norm"       (float) — normalised pause frequency [0, 1].
    """
    # --- Validate classical feature inputs ---
    required_feature_keys = {"pitch_variance", "speech_rate", "pause_frequency"}
    missing = required_feature_keys - features.keys()
    if missing:
        raise KeyError(
            f"features dict is missing required keys: {missing}."
        )

    # --- Load classical baselines ---
    baselines = _load_baselines()
    calm = baselines["calm"]

    pitch_var: float = float(features["pitch_variance"])
    speech_rate: float = float(features["speech_rate"])
    pause_freq: float = float(features["pause_frequency"])

    # Calculate standard classical features (for backward-compatible metrics dashboard)
    pitch_variance_norm = _z_score_sigmoid(pitch_var, mean=calm["pitch_variance_mean"], std=calm["pitch_variance_std"])
    speech_rate_norm = _z_score_sigmoid(speech_rate, mean=calm["speech_rate_mean"], std=calm["speech_rate_std"])
    pause_freq_norm = _z_score_sigmoid(pause_freq, mean=calm["pause_freq_mean"], std=calm["pause_freq_std"])

    # --- Deep Wav2Vec2 (XLS-R) Embedding Stability Scorer ---
    raw_audio_16k = features.get("raw_audio_16k")
    vocal_confidence = 0.8  # Default confident starting point

    if raw_audio_16k is not None and len(raw_audio_16k) > 0:
        try:
            feature_extractor, model = _get_wav2vec2()
            
            # Format inputs and cast to float16 if on CUDA
            inputs = feature_extractor(raw_audio_16k, return_tensors="pt", sampling_rate=16000)
            torch_dtype = torch.float16 if _DEVICE == "cuda" else torch.float32
            input_values = inputs.input_values.to(device=_DEVICE, dtype=torch_dtype)
            
            with torch.no_grad():
                outputs = model(input_values)
                # Shape: [seq_len, hidden_size]
                embeddings = outputs.last_hidden_state[0]
            
            if embeddings.size(0) > 1:
                # Compute adjacent frame similarities
                norm_embeddings = embeddings / (torch.norm(embeddings, dim=-1, keepdim=True) + 1e-9)
                similarities = torch.sum(norm_embeddings[:-1] * norm_embeddings[1:], dim=-1)
                
                stability = float(similarities.mean().cpu().item())
                # Normalize stability via tanh. Clean composure stays around 0.95 - 0.99.
                vocal_confidence = float(np.tanh((stability - 0.91) * 20.0))
                vocal_confidence = float(np.clip(vocal_confidence, 0.0, 1.0))
            else:
                vocal_confidence = 0.5
                
        except Exception as exc:
            logger.error("Wav2Vec2 deep speech scoring failed: %s", exc)
            vocal_confidence = 0.5

    acoustic_nervousness = 1.0 - vocal_confidence

    def _clip(v: float) -> float:
        return float(np.clip(v, 0.0, 1.0))

    return {
        "acoustic_nervousness": _clip(acoustic_nervousness),
        "vocal_confidence": _clip(vocal_confidence),
        "pitch_variance_norm": _clip(pitch_variance_norm),
        "speech_rate_norm": _clip(speech_rate_norm),
        "pause_freq_norm": _clip(pause_freq_norm),
    }
