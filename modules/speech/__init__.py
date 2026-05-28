"""
VocalMirror Speech Module
-------------------------
Exports core speech analysis functions:
  - transcribe_audio: Whisper-based ASR transcription
  - extract_acoustic_features: librosa-based acoustic feature extraction
  - compute_acoustic_nervousness: normalized nervousness scoring
"""

from .transcriber import transcribe_audio
from .acoustic_analyzer import extract_acoustic_features
from .acoustic_scorer import compute_acoustic_nervousness

__all__ = [
    "transcribe_audio",
    "extract_acoustic_features",
    "compute_acoustic_nervousness",
]
