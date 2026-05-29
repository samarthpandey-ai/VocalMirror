"""
acoustic_analyzer.py
---------------------
librosa-based acoustic feature extraction for VocalMirror.

Extracts pitch variance, speech rate (onset proxy), pause frequency,
and raw F0 from a given audio file.  All numerical work uses numpy arrays;
no pandas, matplotlib, or database dependencies.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Constants
_TARGET_SR: int = 22050
_FRAME_LENGTH_S: float = 0.1       # seconds per energy frame
_SILENCE_RMS_THRESHOLD: float = 0.01
_MIN_PAUSE_DURATION_S: float = 0.3  # contiguous silent frames must exceed this
_MIN_AUDIO_DURATION_S: float = 1.0  # clips shorter than this get degraded results


def _to_mono(y: np.ndarray) -> np.ndarray:
    """
    Convert a stereo or multi-channel waveform to mono.

    Args:
        y: Audio array of shape (samples,) or (channels, samples).

    Returns:
        1-D mono float32 numpy array.
    """
    if y.ndim == 1:
        return y
    # librosa.to_mono expects (channels, samples)
    import librosa
    return librosa.to_mono(y)


def _compute_pitch_variance(
    f0: np.ndarray, voiced_flag: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    Compute normalised pitch (F0) variance restricted to voiced frames.

    Normalization via tanh keeps the result in [0, 1].

    Args:
        f0:          Raw F0 array in Hz (NaN for unvoiced frames by convention).
        voiced_flag: Boolean mask; True where a frame is voiced.

    Returns:
        Tuple of (pitch_variance_normalized: float, raw_f0: np.ndarray).
        pitch_variance_normalized is 0.0 when no voiced frames are found.
    """
    voiced_f0 = f0[voiced_flag]

    if voiced_f0.size == 0:
        logger.warning("No voiced frames detected; pitch_variance set to 0.0.")
        return 0.0, f0

    mean_f0 = np.mean(voiced_f0)
    if mean_f0 < 1e-9:
        return 0.0, f0

    variance = np.var(voiced_f0) / (mean_f0 ** 2 + 1e-9)
    normalized = float(np.tanh(variance * 0.5))
    return float(np.clip(normalized, 0.0, 1.0)), f0


def _compute_speech_rate(y: np.ndarray, sr: int, duration: float) -> float:
    """
    Estimate speech rate in words-per-minute using onset detection as a proxy.

    Uses librosa onset envelope to count acoustic onsets, then converts
    onset-rate to an approximate WPM (assuming each onset ≈ one syllable/word).

    Args:
        y:        Mono audio waveform.
        sr:       Sample rate in Hz.
        duration: Audio duration in seconds.

    Returns:
        Estimated speech rate in WPM (float). Returns 0.0 for silent / very
        short audio.
    """
    import librosa

    if duration < _MIN_AUDIO_DURATION_S:
        logger.warning("Audio too short (%.2fs) for reliable speech rate.", duration)
        return 0.0

    try:
        onset_times: np.ndarray = librosa.onset.onset_detect(
            y=y, sr=sr, units="time"
        )
        onset_rate_per_sec = len(onset_times) / duration
        return float(onset_rate_per_sec * 60.0)
    except Exception as exc:
        logger.error("Onset detection failed: %s", exc)
        return 0.0


def _compute_pause_frequency(y: np.ndarray, sr: int, duration: float) -> float:
    """
    Count pause events per second based on RMS energy thresholding.

    A 'pause' is a contiguous run of 0.1-second frames whose RMS energy is
    below ``_SILENCE_RMS_THRESHOLD`` and whose total duration exceeds
    ``_MIN_PAUSE_DURATION_S``.

    Args:
        y:        Mono audio waveform.
        sr:       Sample rate in Hz.
        duration: Audio duration in seconds.

    Returns:
        Number of pauses divided by audio duration (pauses per second).
        Returns 0.0 for very short or all-silent audio.
    """
    if duration < _MIN_AUDIO_DURATION_S:
        return 0.0

    frame_samples = int(_FRAME_LENGTH_S * sr)
    if frame_samples <= 0 or len(y) == 0:
        return 0.0

    # Compute per-frame RMS
    num_frames = len(y) // frame_samples
    if num_frames == 0:
        return 0.0

    frames = y[: num_frames * frame_samples].reshape(num_frames, frame_samples)
    rms_per_frame: np.ndarray = np.sqrt(np.mean(frames ** 2, axis=1))

    is_silent = rms_per_frame < _SILENCE_RMS_THRESHOLD

    # Count contiguous silent runs longer than _MIN_PAUSE_DURATION_S
    min_frames_for_pause = int(np.ceil(_MIN_PAUSE_DURATION_S / _FRAME_LENGTH_S))
    pause_count = 0
    run_length = 0

    for silent in is_silent:
        if silent:
            run_length += 1
        else:
            if run_length >= min_frames_for_pause:
                pause_count += 1
            run_length = 0

    # Close the last run
    if run_length >= min_frames_for_pause:
        pause_count += 1

    # Edge case: entirely silent audio
    total_silent_frames = int(np.sum(is_silent))
    if total_silent_frames == num_frames:
        logger.warning("Audio appears entirely silent.")
        return 0.0

    return float(pause_count / duration)


def extract_acoustic_features(audio_path: str) -> dict:
    """
    Extract acoustic features from an audio file for nervousness scoring.

    Uses librosa for signal processing; all arithmetic via numpy.
    No pandas, matplotlib, or database dependencies.

    Args:
        audio_path: Absolute or relative path to the audio file.
                    Supported formats: WAV, MP3, FLAC, OGG, M4A, etc.

    Returns:
        dict with keys:
            "pitch_variance"  (float)      — tanh-normalised F0 variance [0, 1].
            "speech_rate"     (float)      — onset-based WPM proxy.
            "pause_frequency" (float)      — pauses per second.
            "duration"        (float)      — audio duration in seconds.
            "raw_f0"          (np.ndarray) — raw F0 array (Hz), unvoiced = NaN.

    Raises:
        FileNotFoundError: If audio_path does not exist.
        ValueError:        If the loaded audio is empty or malformed.
        RuntimeError:      If librosa or pyin fails unexpectedly.

    Example:
        >>> feats = extract_acoustic_features("speech.wav")
        >>> print(feats["pitch_variance"], feats["speech_rate"])
    """
    import os
    import librosa

    # --- Validate input ---
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: '{audio_path}'."
        )

    # --- Load audio ---
    try:
        # Load directly at 16kHz for Wav2Vec2 and downstream metrics (highly standard for speech ML)
        y_16k, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as exc:
        raise RuntimeError(
            f"librosa failed to load '{audio_path}': {exc}"
        ) from exc

    if y_16k.size == 0:
        raise ValueError(f"Audio file '{audio_path}' loaded as empty array.")

    duration = float(len(y_16k)) / 16000.0
    logger.info("Loaded audio: %.2fs @ 16000Hz (%d samples).", duration, len(y_16k))

    # --- Edge case: very short clips ---
    if duration < _MIN_AUDIO_DURATION_S:
        logger.warning(
            "Audio duration %.2fs < %.1fs; some features will be unreliable.",
            duration, _MIN_AUDIO_DURATION_S,
        )

    # --- F0 extraction via pyin (optimized downsampled grid) ---
    try:
        # Downsample 16kHz audio to 8kHz for fast pyin pitch tracking
        y_8k = librosa.resample(y_16k, orig_sr=16000, target_sr=8000)
        
        # Run pyin with optimized settings (fmax=300 Hz is standard human limit, hop_length=256 at 8kHz)
        f0, voiced_flag, _voiced_probs = librosa.pyin(
            y_8k,
            fmin=75.0,
            fmax=300.0,
            sr=8000,
            hop_length=256,
        )
        # pyin returns NaN for unvoiced; voiced_flag is a boolean array
        voiced_flag = voiced_flag.astype(bool)
    except Exception as exc:
        logger.error("pyin failed: %s — defaulting F0 to zeros.", exc)
        f0 = np.zeros(1, dtype=np.float64)
        voiced_flag = np.zeros(1, dtype=bool)

    pitch_variance, raw_f0 = _compute_pitch_variance(f0, voiced_flag)
    speech_rate = _compute_speech_rate(y_16k, 16000, duration)
    pause_frequency = _compute_pause_frequency(y_16k, 16000, duration)

    return {
        "pitch_variance": pitch_variance,
        "speech_rate": speech_rate,
        "pause_frequency": pause_frequency,
        "duration": duration,
        "raw_f0": raw_f0,
        "raw_audio_16k": y_16k,
    }
