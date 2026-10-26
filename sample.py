import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_ecg():
    # 1. Create a time axis (0 to 360 points)
    t = np.linspace(0, 1, 360)
    
    # 2. Mathematical approximation of ECG waves (P, QRS, T)
    # Baseline
    ecg = np.random.normal(0, 0.05, 360) # Minor noise
    
    # P-Wave (Atrial contraction) - small bump
    ecg += 0.15 * np.exp(-((t - 0.2) ** 2) / (2 * 0.02 ** 2))
    
    # Q-Wave (Start of ventricular contraction) - small dip
    ecg -= 0.15 * np.exp(-((t - 0.38) ** 2) / (2 * 0.005 ** 2))
    
    # R-Wave (Main spike) - tall peak
    ecg += 1.0 * np.exp(-((t - 0.4) ** 2) / (2 * 0.005 ** 2))
    
    # S-Wave (End of ventricular contraction) - small dip
    ecg -= 0.25 * np.exp(-((t - 0.42) ** 2) / (2 * 0.005 ** 2))
    
    # T-Wave (Recovery) - medium bump
    ecg += 0.3 * np.exp(-((t - 0.6) ** 2) / (2 * 0.05 ** 2))
    
    # 3. Save to CSV
    df = pd.DataFrame(ecg, columns=['amplitude'])
    df.to_csv('sample_normal_ecg.csv', index=False)
    
    print("✅ File 'sample_normal_ecg.csv' created successfully!")
    
    # Optional: Preview it
    plt.plot(ecg)
    plt.title("Generated Synthetic ECG")
    plt.show()

if __name__ == "__main__":
    generate_synthetic_ecg()