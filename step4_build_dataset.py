import numpy as np
import wfdb
from skimage.restoration import denoise_wavelet
from tqdm import tqdm
import os

# --- SETTINGS ---
patients = ['100', '101', '103', '105', '106', '109', '111', '112', '113', '114',
            '115', '116', '117', '118', '119', '121', '122', '123', '124', '200',
            '201', '202', '203', '205', '208', '209', '210', '212', '213', '214',
            '215', '219', '220', '221', '222', '223', '228', '230', '231', '232',
            '233', '234']

# --- THE FIX IS HERE ---
# Choose annotation symbols you want to keep. Example keeps common morphologies.
# Adjust this list to your use-case (MIT-BIH annotation symbols like 'N','L','R','V','A', etc.)
valid_classes = ['N', 'L', 'R', 'V', 'A', 'a', 'J', 'F']

window_size = 360
half_window = window_size // 2

# initialize containers (important!)
all_beats = []
all_labels = []

# --- HELPER FUNCTIONS ---
def clean_signal(signal):
    # Denoise
    denoised = denoise_wavelet(signal,
                               method='BayesShrink',
                               mode='soft',
                               wavelet_levels=3,
                               wavelet='sym4',
                               rescale_sigma=True)  # boolean, not string
    # Robust Scaling
    median = np.nanmedian(denoised)
    iqr = np.percentile(denoised, 75) - np.percentile(denoised, 25)
    if iqr == 0:
        return denoised
    return (denoised - median) / iqr

# --- MAIN LOOP ---
print(f"Step 4: Starting Factory Line. Processing {len(patients)} patients...")

for patient_id in tqdm(patients):
    try:
        # Download if missing
        if not os.path.exists(f'data/{patient_id}.dat'):
            wfdb.dl_database('mitdb', 'data/', records=[patient_id])

        record = wfdb.rdrecord(f'data/{patient_id}')
        annotation = wfdb.rdann(f'data/{patient_id}', 'atr')

        raw_signal = record.p_signal[:, 0]
        processed_signal = clean_signal(raw_signal)

        peaks = annotation.sample
        labels = annotation.symbol

        for peak, label in zip(peaks, labels):
            # label might be bytes depending on wfdb version; convert to str if necessary
            if isinstance(label, bytes):
                label = label.decode('utf-8', errors='ignore')

            if label in valid_classes:
                if peak < half_window or peak > len(processed_signal) - half_window:
                    continue

                beat = processed_signal[peak - half_window : peak + half_window]
                if len(beat) == window_size:  # extra safety
                    all_beats.append(beat)
                    all_labels.append(label)
    except Exception as e:
        print(f"⚠️ Error with Patient {patient_id}: {e}")

# --- SAVE RESULTS ---
print("✅ Processing Complete!")
print(f"   Total Heartbeats Collected: {len(all_beats)}")

if len(all_beats) == 0:
    print("❌ No beats collected. Possible causes:")
    print("   - valid_classes doesn't match annotation symbols in these records")
    print("   - peaks fall too close to signal edges and were skipped")
    print("   - annotations may be different (check annotation.symbol values)")
else:
    X = np.array(all_beats)  # shape (n_samples, window_size)
    y = np.array(all_labels)

    # Ensure shape is (samples, time, channels)
    if X.ndim == 2:
        X = np.expand_dims(X, axis=2)  # (n, time, 1)
    elif X.ndim == 3:
        pass  # already correct
    else:
        raise ValueError(f"Unexpected X ndim: {X.ndim}")

    print(f"   Data Shape: {X.shape}")
    print("   Saving to 'data/X_data.npy' and 'data/y_data.npy'...")

    os.makedirs('data', exist_ok=True)
    np.save('data/X_data.npy', X)
    np.save('data/y_data.npy', y)
    print("🎉 DONE! Dataset built.")
