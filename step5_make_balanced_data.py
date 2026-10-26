import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
from collections import Counter

# --- 1. LOAD DATA ---
print("Step 5: Loading dataset...")
X = np.load('data/X_data.npy')
y = np.load('data/y_data.npy')

print(f"   Original Data Shape: {X.shape}")
print(f"   Original Class Counts: {Counter(y)}")

# --- 2. ENCODE LABELS (Text -> Numbers) ---
# The computer needs numbers, not 'N' or 'V'
encoder = LabelEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y)

print(f"   Classes found: {encoder.classes_}")
print(f"   Mapped to: {encoder.transform(encoder.classes_)}")

# --- 3. SPLIT DATA (Train vs Test) ---
# We keep 20% aside for the final exam (Test)
# stratify=y_encoded ensures we split all classes evenly
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# --- 4. APPLY SMOTE (The Key Research Step) ---
# SMOTE only works on 2D data, but our X is 3D (Samples, 360, 1).
# We must flatten it -> SMOTE -> reshape back.

print("   Applying SMOTE to balance the training set (This takes time)...")

# Flatten: (N, 360, 1) -> (N, 360)
n_samples, time_steps, n_channels = X_train.shape
X_train_flat = X_train.reshape(n_samples, time_steps * n_channels)

# Create fake data for minority classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train)

# Reshape back: (N_new, 360) -> (N_new, 360, 1)
X_train_final = X_train_resampled.reshape(-1, time_steps, n_channels)
y_train_final = y_train_resampled

print(f"   ✅ SMOTE Complete!")
print(f"   New Training Shape: {X_train_final.shape}")
print(f"   New Class Counts: {Counter(y_train_final)}")

# --- 5. ONE-HOT ENCODING ---
# Convert 0, 1, 2 into , ,  for the AI
y_train_final = to_categorical(y_train_final)
y_test_final = to_categorical(y_test)

# --- 6. SAVE EVERYTHING ---
print("   Saving prepared data...")
np.save('data/X_train.npy', X_train_final)
np.save('data/y_train.npy', y_train_final)
np.save('data/X_test.npy', X_test)
np.save('data/y_test.npy', y_test_final)
np.save('data/classes.npy', encoder.classes_) # Save label names for later

print("🎉 DONE! Data is balanced and ready for the AI.")