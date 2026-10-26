import numpy as np
import matplotlib.pyplot as plt
import wfdb
from skimage.restoration import denoise_wavelet

# --- SETTINGS ---
record_name = '100'
path = 'data/'

# --- 1. DEFINE OUR TOOLS ---

def ocan_denoise(signal):
    """
    Applies Wavelet Denoising using Symlet4 (sym4) and BayesShrink.
    This is exactly what the OCADN paper specifies.
    """
    # wavelet='sym4': The shape of the filter (looks like a heartbeat)
    # method='BayesShrink': Smart way to decide what is noise vs signal
    # mode='soft': Gently smooths the noise
    clean_signal = denoise_wavelet(signal, method='BayesShrink', mode='soft', 
                                   wavelet_levels=3, wavelet='sym4', rescale_sigma='True')
    return clean_signal

def robust_scaling(signal):
    """
    Normalizes the signal using Median and Interquartile Range (IQR).
    Formula: (Data - Median) / IQR
    """
    median = np.nanmedian(signal)
    q75, q25 = np.percentile(signal, [75 ,25])
    iqr = q75 - q25
    
    # Avoid division by zero if IQR is 0
    if iqr == 0:
        return signal
        
    normalized_signal = (signal - median) / iqr
    return normalized_signal

# --- 2. RUN THE PROCESS ---

print("Step 2: Loading data...")
record = wfdb.rdrecord(path + record_name)
raw_signal = record.p_signal[:1000, 0] # First 1000 samples, Lead II

print("   Applying Wavelet Denoising (Symlet4)...")
denoised_signal = ocan_denoise(raw_signal)

print("   Applying Robust Scaling...")
final_signal = robust_scaling(denoised_signal)

# --- 3. VISUALIZE THE DIFFERENCE ---
print("   Plotting results...")

plt.figure(figsize=(12, 8))

# Plot 1: Raw
plt.subplot(3, 1, 1)
plt.plot(raw_signal, color='black', alpha=0.7)
plt.title("1. Raw Signal (Noisy)")
plt.grid(True)

# Plot 2: Denoised
plt.subplot(3, 1, 2)
plt.plot(denoised_signal, color='blue')
plt.title("2. After Wavelet Denoising (Smoother)")
plt.grid(True)

# Plot 3: Normalized
plt.subplot(3, 1, 3)
plt.plot(final_signal, color='green')
plt.title("3. After Robust Scaling (Y-axis rescaled)")
plt.grid(True)

plt.tight_layout()
plt.show()