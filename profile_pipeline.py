import time
import numpy as np
import scipy.io.wavfile as wavfile
import os
import torch

# Unified device selection: cuda -> mps -> cpu
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Profiling using device: {device}")

# 1. Generate a dummy 15-second wav file (silence + 440Hz sine wave)
sr = 22050
duration = 15.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
audio_data = np.sin(2 * np.pi * 440 * t) * 0.5
audio_data = audio_data.astype(np.float32)

temp_audio_path = "temp_profile.wav"
wavfile.write(temp_audio_path, sr, audio_data)
print(f"Generated dummy audio file at {temp_audio_path}")

try:
    # Measure imports
    t0 = time.time()
    from modules.speech import transcribe_audio, extract_acoustic_features, compute_acoustic_nervousness
    from modules.nlp import score_sentiment, score_confidence, detect_fillers, analyze_vocabulary
    from modules.fusion import fuse_scores, aggregate_nlp_outputs
    print(f"Import time: {time.time() - t0:.3f} s")

    # Step 1: Whisper transcription (using "tiny" model)
    t0 = time.time()
    transcription = transcribe_audio(temp_audio_path, model_size="tiny")
    whisper_time_tiny = time.time() - t0
    print(f"Whisper transcription (tiny) took: {whisper_time_tiny:.3f} s")
    print(f"Transcript: {transcription.get('text', '')}")

    # Step 2: Acoustic features extraction (librosa pyin + f0)
    t0 = time.time()
    acoustic_features = extract_acoustic_features(temp_audio_path)
    acoustic_feat_time = time.time() - t0
    print(f"Acoustic features extraction (librosa pyin) took: {acoustic_feat_time:.3f} s")

    # Step 3: Acoustic nervousness scoring (Wav2Vec2 XLS-R)
    t0 = time.time()
    acoustic_scores = compute_acoustic_nervousness(acoustic_features)
    acoustic_score_time = time.time() - t0
    print(f"Acoustic nervousness scoring (Wav2Vec2 XLS-R) took: {acoustic_score_time:.3f} s")

    # Text for NLP tasks
    text = "Hello there. I am confident and ready to present. There are no fillers in my speech, um, actually, maybe one or two."

    # Step 4: NLP Sentiment scoring (RoBERTa)
    t0 = time.time()
    sentiment = score_sentiment(text)
    sentiment_time = time.time() - t0
    print(f"Sentiment scoring took: {sentiment_time:.3f} s")

    # Step 5: NLP Confidence scoring (Zero-shot RoBERTa large)
    t0 = time.time()
    confidence = score_confidence(text)
    confidence_time = time.time() - t0
    print(f"Confidence scoring took: {confidence_time:.3f} s")

    # Step 6: Filler & vocab
    t0 = time.time()
    fillers = detect_fillers(text)
    vocab = analyze_vocabulary(text)
    other_nlp_time = time.time() - t0
    print(f"Filler detection & vocab analysis took: {other_nlp_time:.3f} s")

    print("\n--- Pipeline Summary ---")
    total_execution = whisper_time_tiny + acoustic_feat_time + acoustic_score_time + sentiment_time + confidence_time + other_nlp_time
    print(f"Total Execution: {total_execution:.3f} s")

finally:
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
