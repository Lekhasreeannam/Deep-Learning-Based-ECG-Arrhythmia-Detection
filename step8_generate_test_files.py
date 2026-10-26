import numpy as np
import pandas as pd
import os

# --- 1. SETUP ---
print("Step 8: Generating sample 'Patient Files' for the Website...")
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')
class_names = np.load('data/classes.npy')

# Create a folder to store these test files
output_folder = "sample_ecgs_for_upload"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- 2. PICK RANDOM BEATS ---
# We will pick 5 random beats to save as CSVs
# We try to pick different classes if possible
indices = np.random.choice(len(X_test), 10, replace=False)

print(f"   Saving files to folder: '{output_folder}/'...")

count = 0
for idx in indices:
    if count >= 10: break
    
    # Get the signal (360 numbers)
    signal = X_test[idx].flatten() 
    
    # Get the True Label (e.g., 'V')
    true_label_index = np.argmax(y_test[idx])
    true_label = class_names[true_label_index]
    
    # Create a DataFrame (Excel-like table)
    # We create a simple CSV with one column: "Amplitude"
    df = pd.DataFrame(signal, columns=['Amplitude'])
    
    # Save as CSV
    filename = f"{output_folder}/patient_{count+1}_label_{true_label}.csv"
    df.to_csv(filename, index=False)
    print(f"   ✅ Created: {filename}")
    count += 1

print("\n🎉 DONE! You can now use these CSV files to upload to your Website.")