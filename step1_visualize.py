import wfdb
import matplotlib.pyplot as plt
import numpy as np
import os

# --- SETTINGS ---
# We will download data for Patient "100" (a standard test file)
record_name = '100'
# We will save it in the 'data' folder we created
save_path = 'data/'

print("Step 1: Checking if data exists...")

# 1. Download the data if we don't have it
# This function checks if the file exists; if not, it downloads it from PhysioNet (MIT DB)
try:
    wfdb.dl_database('mitdb', save_path, records=[record_name])
    print(f"✅ Success! Patient {record_name} data downloaded to '{save_path}'")
except Exception as e:
    print(f"❌ Error downloading data: {e}")

# 2. Read the file
# We read the 'record' which contains the signal (voltage values)
# and 'annotation' which contains the doctor's labels (where the heartbeats are)
try:
    record = wfdb.rdrecord(save_path + record_name)
    annotation = wfdb.rdann(save_path + record_name, 'atr')
    
    print("✅ Data read successfully!")
    print(f"   Signal length: {record.p_signal.shape} samples")
    print(f"   Number of heartbeats marked: {len(annotation.sample)}")

    # 3. Plot the First 1000 Samples (approx 3 seconds of heartbeats)
    # The MIT-BIH dataset usually has 2 channels (Lead II and V5). 
    # We will look at Channel 0 (Lead II).
    
    signal_data = record.p_signal[:1000, 0] # Get first 1000 points, channel 0

    plt.figure(figsize=(10, 4))
    plt.plot(signal_data, color='blue')
    plt.title(f"First 3 Seconds of ECG - Patient {record_name} (Lead II)")
    plt.xlabel("Time (Samples)")
    plt.ylabel("Amplitude (Normalized Voltage)")
    plt.grid(True)
    
    # Show the plot
    print("   Opening plot window...")
    plt.show()

except Exception as e:
    print(f"❌ Error reading/plotting data: {e}")