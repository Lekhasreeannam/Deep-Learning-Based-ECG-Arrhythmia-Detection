import numpy as np
import matplotlib.pyplot as plt
import wfdb
from skimage.restoration import denoise_wavelet

# --- 1. SETUP ---
record_name = '100'
path = 'data/'
window_size = 360  # Total width of the window (samples)
half_window = window_size // 2  # 180 samples before, 180 after

# --- 2. REUSE OUR CLEANING TOOLS (From Step 2) ---
def clean_signal(signal):
    # Apply Wavelet Denoising
    denoised = denoise_wavelet(signal, method='BayesShrink', mode='soft', 
                               wavelet_levels=3, wavelet='sym4', rescale_sigma='True')
    # Apply Robust Scaling
    median = np.nanmedian(denoised)
    iqr = np.percentile(denoised, 75) - np.percentile(denoised, 25)
    if iqr == 0: return denoised
    return (denoised - median) / iqr

# --- 3. LOAD DATA ---
print(f"Step 3: Loading Patient {record_name}...")
record = wfdb.rdrecord(path + record_name)
annotation = wfdb.rdann(path + record_name, 'atr')

raw_signal = record.p_signal[:, 0] # Get the full signal (Channel 0)
peaks = annotation.sample          # Get the list of R-peak locations
labels = annotation.symbol         # Get the doctor's label for each peak ('N', 'V', etc.)

print("   Cleaning the full signal (this might take a moment)...")
processed_signal = clean_signal(raw_signal)

# --- 4. THE COOKIE CUTTER (Segmentation) ---
print("   Cutting heartbeats...")

segmented_beats = []# <--- Add brackets
beat_labels =  [] # <--- Add brackets

# Loop through every peak the doctor marked
for i in range(len(peaks)):
    peak_location = peaks[i]
    label = labels[i]
    
    # Skip if the peak is too close to the start or end of the file
    if peak_location < half_window or peak_location > len(processed_signal) - half_window:
        continue
    
    # CUT: Take 180 samples before and 180 samples after the peak
    beat_window = processed_signal[peak_location - half_window : peak_location + half_window]
    
    segmented_beats.append(beat_window)
    beat_labels.append(label)

print(f"✅ Successfully cut {len(segmented_beats)} individual heartbeats!")

# --- 5. VISUALIZE THE "COOKIES" ---
# Let's plot 5 random heartbeats to see what they look like
plt.figure(figsize=(10, 6))

for i in range(1, 6): # Plot 5 beats
    plt.subplot(2, 3, i)
    plt.plot(segmented_beats[i * 10]) # Just picking arbitrary beats (10, 20, 30...)
    plt.title(f"Beat #{i*10} (Label: {beat_labels[i*10]})")
    plt.grid(True)

plt.tight_layout()
plt.show()