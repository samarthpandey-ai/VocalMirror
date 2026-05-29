import time
import numpy as np
import librosa

print("Profiling librosa.pyin on CPU...")

# Generate a 15-second dummy audio signal (440Hz sine wave + noise)
sr_22k = 22050
duration = 15.0
t_22k = np.linspace(0, duration, int(sr_22k * duration), endpoint=False)
y_22k = np.sin(2 * np.pi * 440 * t_22k) * 0.5 + np.random.normal(0, 0.05, len(t_22k))
y_22k = y_22k.astype(np.float32)

# Test 1: Original pyin (22050 Hz, fmin=75, fmax=500, default hop_length=512)
print("\n--- Test 1: Original pyin (22050 Hz, hop_length=512) ---")
t0 = time.time()
f0_1, voiced_1, _ = librosa.pyin(y_22k, fmin=75.0, fmax=500.0, sr=sr_22k)
t_original = time.time() - t0
print(f"Time taken: {t_original:.3f} s")
print(f"Number of frames: {len(f0_1)}")
voiced_frames_1 = np.sum(voiced_1)
print(f"Voiced frames: {voiced_frames_1}")

# Test 2: Downsampled pyin (8000 Hz, hop_length=256)
print("\n--- Test 2: Downsampled pyin (8000 Hz, hop_length=256) ---")
t0 = time.time()
y_8k = librosa.resample(y_22k, orig_sr=sr_22k, target_sr=8000)
resample_time = time.time() - t0
print(f"Resampling took: {resample_time:.3f} s")

t0 = time.time()
f0_2, voiced_2, _ = librosa.pyin(y_8k, fmin=75.0, fmax=300.0, sr=8000, hop_length=256)
t_downsampled = time.time() - t0
print(f"pyin computation took: {t_downsampled:.3f} s")
print(f"Total time (resample + pyin): {resample_time + t_downsampled:.3f} s")
print(f"Number of frames: {len(f0_2)}")
voiced_frames_2 = np.sum(voiced_2)
print(f"Voiced frames: {voiced_frames_2}")
print(f"Speedup: {t_original / (resample_time + t_downsampled):.1f}x")

# Let's compare pitch variance
def get_variance(f0, voiced):
    voiced_f0 = f0[voiced]
    if voiced_f0.size == 0:
        return 0.0
    mean_f0 = np.mean(voiced_f0)
    variance = np.var(voiced_f0) / (mean_f0 ** 2 + 1e-9)
    return float(np.tanh(variance * 0.5))

print("\n--- Comparison of Results ---")
print(f"Original pitch variance: {get_variance(f0_1, voiced_1):.5f}")
print(f"Downsampled pitch variance: {get_variance(f0_2, voiced_2):.5f}")
