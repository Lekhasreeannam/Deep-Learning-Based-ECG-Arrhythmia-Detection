import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
os.makedirs('explain_examples', exist_ok=True)
# --- 1. SETUP & ROBUST MODEL REBUILDING ---
print("Step 7: Loading and Rebuilding Model for Explanation...")

# A. Load the original model
try:
    orig_model = tf.keras.models.load_model('final_model.h5')
except Exception:
    print("❌ Error: 'final_model.h5' not found. Please run Step 6 first.")
    raise

# B. SURGICAL REPLACEMENT of the Final Layer (produce logits)
last_hidden_output = orig_model.layers[-2].output
final_layer_config = orig_model.layers[-1].get_config()
final_layer_config['activation'] = 'linear'  # Force linear (logits)
final_layer_config['name'] = 'logits_layer'
new_final_layer = tf.keras.layers.Dense.from_config(final_layer_config)
new_output = new_final_layer(last_hidden_output)
expl_model = tf.keras.models.Model(inputs=orig_model.input, outputs=new_output)
expl_model.layers[-1].set_weights(orig_model.layers[-1].get_weights())

print("   ✅ Model successfully rebuilt with Linear Output (Logits).")

# C. Load Data
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')
class_names = np.load('data/classes.npy')

# D. Find Last Conv Layer Name
last_conv_layer_name = ""
for layer in reversed(expl_model.layers):
    if 'conv1d' in layer.name:
        last_conv_layer_name = layer.name
        break
print(f"   Targeting Layer: {last_conv_layer_name}")


# --- 2. ROBUST GRAD-CAM FUNCTION ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Build grad model with single input (avoids structure warnings)
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_tensor)  # (1, T, C), (1, C)

        # choose scalar class index from first (and only) sample
        if pred_index is None:
            pred_index = tf.argmax(preds[0], axis=-1)  # scalar
        pred_index = tf.cast(pred_index, tf.int32)

        class_channel = preds[:, pred_index]  # shape (1,)

    # gradients of raw score wrt feature maps
    grads = tape.gradient(class_channel, last_conv_layer_output)  # (1, T, C)

    # pooled grads per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # (C,)

    # remove batch dim from feature maps -> (T, C)
    feature_maps = last_conv_layer_output[0]

    # weighted sum -> heatmap (T,)
    weighted_maps = feature_maps * pooled_grads[tf.newaxis, :]
    heatmap = tf.reduce_sum(weighted_maps, axis=-1)

    # relu + safe normalize
    heatmap = tf.nn.relu(heatmap)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros((heatmap.shape[0],), dtype=np.float32)
    heatmap = heatmap / (max_val + 1e-10)
    return heatmap.numpy().astype(np.float32)

def save_example(idx):
    sb = X_test[idx]
    sb_exp = np.expand_dims(sb, axis=0)
    rawp = expl_model.predict(sb_exp)
    probs = tf.nn.softmax(rawp, axis=-1).numpy()[0]
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    conf = probs[pred_idx] * 100.0

    hm = make_gradcam_heatmap(sb_exp, expl_model, last_conv_layer_name)
    print(f"Idx {idx}: True={class_names[np.argmax(y_test[idx])]}, Pred={pred_label}, conf={conf:.1f}%, hm_min={hm.min():.3e}, hm_max={hm.max():.3e}")

    # Plot & save
    plt.figure(figsize=(10,4))
    plt.plot(sb, color='black')
    hm_img = cv2.applyColorMap(np.uint8(255*cv2.resize(hm.reshape(1,-1), (360,1))), cv2.COLORMAP_JET)
    hm_img = cv2.cvtColor(hm_img, cv2.COLOR_BGR2RGB)
    extent = [0, 360, np.min(sb), np.max(sb)]
    plt.imshow(hm_img, aspect='auto', alpha=0.6, extent=extent)
    plt.title(f"Idx {idx} True:{class_names[np.argmax(y_test[idx])]} Pred:{pred_label} {conf:.1f}%")
    plt.tight_layout()
    plt.savefig(f"explain_examples/example_{idx}.png")
    plt.close()

for i in range(6):
    save_example(np.random.randint(0, len(X_test)))
    
# --- 3. RUN EXPLANATION ---
idx = np.random.randint(0, len(X_test))
sample_beat = X_test[idx]
sample_beat_expanded = np.expand_dims(sample_beat, axis=0)  # (1, 360, 1)

# Use expl_model for logits, then softmax for readable probs
raw_preds = expl_model.predict(sample_beat_expanded, verbose=0)  # logits shape (1, C)
probs = tf.nn.softmax(raw_preds, axis=-1).numpy()               # shape (1, C)
predicted_idx = int(np.argmax(probs[0]))                        # scalar index
predicted_label = class_names[predicted_idx]
true_label = class_names[np.argmax(y_test[idx])]
confidence = float(probs[0, predicted_idx] * 100.0)

print(f"   Explaining Beat #{idx}")
print(f"   True: {true_label} | Pred: {predicted_label} ({confidence:.1f}%)")

# Generate heatmap (from expl_model which outputs logits)
heatmap = make_gradcam_heatmap(sample_beat_expanded, expl_model, last_conv_layer_name)

# --- 4. VISUALIZE ---
plt.figure(figsize=(12, 6))

# Plot 1: Signal
ax1 = plt.subplot(2, 1, 1)
plt.plot(sample_beat, color='black')
plt.title(f"ECG Signal (True: {true_label}, Pred: {predicted_label})")
plt.grid(True, alpha=0.3)

# Plot 2: Heatmap
heatmap_img = heatmap.reshape(1, -1)                     # (1, Tconv)
heatmap_resized = cv2.resize(heatmap_img, (360, 1))      # (1,360)
heatmap_resized = np.uint8(255 * heatmap_resized)
heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(sample_beat, color='black', alpha=0.5, label='Signal')
extent = [0, 360, np.min(sample_beat), np.max(sample_beat)]
plt.imshow(heatmap_color, aspect='auto', alpha=0.6, extent=extent)
plt.title("Grad-CAM Heatmap (Red = Region AI focused on)")

plt.tight_layout()
plt.show()
print("✅ Success! Graph generated.")
