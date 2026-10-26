import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, GlobalAveragePooling1D, BatchNormalization, Add, MultiHeadAttention
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# --- SETTINGS ---
# ⚠️ FIRST RUN: Set this to True to test if code works (fast). 
# ⚠️ SECOND RUN: Set to False to train the real brain (slow, takes hours).
TEST_MODE = False

# --- 1. LOAD DATA ---
print("Step 6: Loading data...")
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

# Get shapes dynamically
n_timesteps = X_train.shape[1]   # Should be 360
n_features = X_train.shape[2]    # Should be 1
n_classes = y_train.shape[1]     # Should be 8 based on your log

print(f"   Input Shape: {X_train.shape}")
print(f"   Classes to predict: {n_classes}")

# --- SUBSET FOR TESTING (If TEST_MODE is On) ---
if TEST_MODE:
    print("\n⚠️ TEST MODE ON: Using only 2000 samples for a quick check.")
    X_train = X_train[:2000]
    y_train = y_train[:2000]
    X_test = X_test[:500]
    y_test = y_test[:500]

# --- 2. BUILD THE OCADN MODEL (The Brain) ---
def build_ocadn_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Block 1: Standard Convolution (Finds sharp spikes)
    x = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Block 2: Dilated Convolution (Sees the rhythm/context)
    # dilation_rate=2 means it skips every other step to see "wider"
    x = Conv1D(filters=64, kernel_size=3, padding='same', dilation_rate=2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Block 3: Deep Features
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Block 4: ATTENTION MECHANISM (The Novelty)
    # This helps the model focus on the specific part of the signal that matters
    attention = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = Add()([x, attention]) # Skip connection (ResNet style)
    
    # Global Pooling (Squashes 360 points into a single summary vector)
    # This is crucial for Grad-CAM later!
    x = GlobalAveragePooling1D()(x)
    
    # Final Classification
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x) # Prevents memorization
    outputs = Dense(num_classes, activation='softmax')(x) # Probability for each class
    
    model = Model(inputs, outputs)
    return model

print("\n   Building OCADN Architecture...")
model = build_ocadn_model((n_timesteps, n_features), n_classes)
model.summary()

# --- 3. COMPILE & TRAIN ---
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Callbacks (Helpers to stop training if it gets good enough)
callbacks = [
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]


print("\n   Starting Training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=5 if TEST_MODE else 30,  # 5 epochs for test, 30 for real
    batch_size=32,
    callbacks=callbacks
)

# --- 4. SAVE & PLOT ---
print("   Saving final model...")
model.save('final_model.h5')

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Training History')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

print("🎉 Training Complete!")
if TEST_MODE:
    print("⚠️ NOTE: This was just a TEST RUN. Change TEST_MODE = False in the code to train on real data.")