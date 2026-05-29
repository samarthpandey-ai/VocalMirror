"""
transcriber.py
--------------
Whisper-based audio transcription for VocalMirror.

Uses openai-whisper for automatic speech recognition.
Model instances are cached at module level to avoid redundant loading.
"""

from __future__ import annotations

import gc
import logging
from typing import Any
import torch
import whisper

logger = logging.getLogger(__name__)

# Unified device selection: cuda -> mps -> cpu
if torch.cuda.is_available():
    _DEVICE = "cuda"
elif torch.backends.mps.is_available():
    _DEVICE = "mps"
else:
    _DEVICE = "cpu"
logger.info("Using device for Whisper transcription: %s", _DEVICE)

# Module-level model cache: {model_size: whisper.Whisper}
_model_cache: dict[str, Any] = {}

SUPPORTED_SIZES = frozenset({"tiny", "small"})

# Whisper models are loaded lazily and cached on first use to ensure fast import times.


def _load_model(model_size: str) -> Any:
    """
    Load and cache a Whisper model by size.

    Args:
        model_size: One of 'tiny' or 'small'.

    Returns:
        A loaded whisper.Whisper model instance.

    Raises:
        ValueError: If model_size is not supported.
        RuntimeError: If the model cannot be loaded.
    """
    if model_size not in SUPPORTED_SIZES:
        raise ValueError(
            f"Unsupported model size '{model_size}'. "
            f"Choose from: {sorted(SUPPORTED_SIZES)}"
        )

    if model_size not in _model_cache:
        try:
            logger.info("Loading Whisper model '%s' on %s.", model_size, _DEVICE)
            _model_cache[model_size] = whisper.load_model(model_size, device=_DEVICE)
            logger.info("Whisper model '%s' loaded and cached.", model_size)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Whisper model '{model_size}': {exc}"
            ) from exc

    return _model_cache[model_size]


def transcribe_audio(audio_path: str, model_size: str = "small") -> dict:
    """
    Transcribe an audio file using OpenAI Whisper.

    The model is loaded eagerly on startup and cached for subsequent calls.
    Supported sizes: 'tiny', 'small'.

    Args:
        audio_path:  Absolute or relative path to the audio file.
                     Supported formats: WAV, MP3, M4A, FLAC, OGG, etc.
        model_size:  Whisper model variant to use. Defaults to 'small'.
                     'tiny'  — fastest, lowest accuracy (~39 M params).
                     'small' — good accuracy / speed tradeoff (~244 M params).

    Returns:
        dict with keys:
            "text"     (str)       — full transcript string.
            "segments" (list[dict])— Whisper segment dicts, each containing
                                     'start', 'end', 'text', and more.
            "language" (str)       — detected language code (e.g. 'en').

    Raises:
        FileNotFoundError: If audio_path does not point to an existing file.
        ValueError:        If model_size is not in the supported set.
        RuntimeError:      If transcription fails for any other reason.

    Example:
        >>> result = transcribe_audio("speech.wav", model_size="small")
        >>> print(result["text"])
        'Hello, my name is ...'
    """
    import os

    # --- Validate input file ---
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: '{audio_path}'. "
            "Please provide a valid path to an audio file."
        )

    # --- Load (or retrieve cached) model ---
    model = _load_model(model_size)

    # --- Run transcription ---
    try:
        logger.info("Transcribing '%s' with Whisper '%s' on %s.", audio_path, model_size, _DEVICE)
        # Whisper fp16 parameter is True if CUDA is available, False otherwise
        use_fp16 = torch.cuda.is_available()
        result = model.transcribe(audio_path, fp16=use_fp16)
    except Exception as exc:
        raise RuntimeError(
            f"Whisper transcription failed for '{audio_path}': {exc}"
        ) from exc
    finally:
        # Aggressively clear memory immediately after audio processing
        gc.collect()
        if _DEVICE == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        elif _DEVICE == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    return {
        "text": result.get("text", "").strip(),
        "segments": result.get("segments", []),
        "language": result.get("language", "unknown"),
    }

