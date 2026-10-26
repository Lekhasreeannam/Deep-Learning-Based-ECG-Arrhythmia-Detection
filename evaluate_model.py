# evaluate_model.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns

# -------- CONFIG --------
DATA_DIR = "data"
OUT_DIR = "eval_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
WINDOW = 360

# -------- LOAD --------
X_test = np.load(f"{DATA_DIR}/X_test.npy")        # shape (N, 360, 1)
y_test = np.load(f"{DATA_DIR}/y_test.npy")        # one-hot (N, C)
class_names = np.load(f"{DATA_DIR}/classes.npy")  # (C,)

# load explanation model (logits) - same you used in app
expl_model = tf.keras.models.load_model("final_model.h5")
# If your saved model still has softmax, rebuild expl_model with linear logits as in your app.
# (Assume here expl_model returns logits; if not, follow the rebuild pattern used in app.py)

# Recreate helper functions if not already available (robust_scaling & make_gradcam_heatmap)
def robust_scaling(signal):
    signal = np.asarray(signal, dtype=np.float32)
    median = np.nanmedian(signal)
    q75, q25 = np.percentile(signal, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        std = np.std(signal)
        if std == 0:
            return signal.astype(np.float32)
        return ((signal - median) / std).astype(np.float32)
    return ((signal - median) / iqr).astype(np.float32)

# Minimal Grad-CAM wrapper (uses expl_model output logits)
def make_gradcam_heatmap(img_array, expl_model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(inputs=expl_model.input,
                                       outputs=[expl_model.get_layer(last_conv_layer_name).output,
                                                expl_model.output])
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_tensor)
        pred_index = tf.argmax(preds[0], axis=-1)
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    feature_maps = last_conv_layer_output[0]
    weighted_maps = feature_maps * pooled_grads[tf.newaxis, :]
    heatmap = tf.reduce_sum(weighted_maps, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    maxv = tf.reduce_max(heatmap)
    if maxv == 0:
        return np.zeros((heatmap.shape[0],), dtype=np.float32)
    heatmap = heatmap / (maxv + 1e-10)
    return heatmap.numpy().astype(np.float32)

# find last conv layer name
last_conv_layer_name = None
for layer in reversed(expl_model.layers):
    if 'conv1d' in layer.name:
        last_conv_layer_name = layer.name
        break
if last_conv_layer_name is None:
    raise RuntimeError("No conv1d layer found in expl_model")

# -------- PREDICT BATCH --------
logits = expl_model.predict(X_test, verbose=1)   # shape (N, C) logits
probs = tf.nn.softmax(logits, axis=-1).numpy()
y_pred_idx = np.argmax(probs, axis=1)
y_true_idx = np.argmax(y_test, axis=1)

# -------- METRICS --------
acc = np.mean(y_pred_idx == y_true_idx)
print(f"Overall accuracy: {acc*100:.3f}%")

print("\nClassification report (per-class):")
print(classification_report(y_true_idx, y_pred_idx, target_names=class_names, digits=4))

# Save detailed precision/recall/f1
prec, rec, f1, supp = precision_recall_fscore_support(y_true_idx, y_pred_idx)
for i, cname in enumerate(class_names):
    print(f"{cname}: precision {prec[i]:.3f}, recall {rec[i]:.3f}, f1 {f1[i]:.3f}, support {supp[i]}")

# Confusion matrix
cm = confusion_matrix(y_true_idx, y_pred_idx)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
plt.close()

# -------- SAVE MISCLASSIFIED EXAMPLES & VISUALIZE SOME --------
mis_idx = np.where(y_pred_idx != y_true_idx)[0]
print(f"Total misclassified: {len(mis_idx)} / {len(X_test)}")

# Save a CSV summary
import csv
with open(os.path.join(OUT_DIR, "misclassified_summary.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["index", "true_label", "pred_label", "pred_prob"])
    for i in mis_idx:
        writer.writerow([int(i), class_names[y_true_idx[i]], class_names[y_pred_idx[i]], float(probs[i, y_pred_idx[i]])])

# Generate plots (up to 50 misclassified)
max_save = 50
for i, idx in enumerate(mis_idx[:max_save]):
    beat = X_test[idx].squeeze()  # (360,)
    hm = make_gradcam_heatmap(X_test[idx:idx+1], expl_model, last_conv_layer_name)
    # resize heatmap to 360
    hm_img = hm.reshape(1, -1)
    hm_resized = cv2.resize(hm_img, (WINDOW, 1))
    hm_resized = np.uint8(255 * hm_resized)
    heatmap_color = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,3))
    plt.plot(beat, color='black')
    extent = [0, WINDOW, np.min(beat), np.max(beat)]
    plt.imshow(heatmap_color, aspect='auto', alpha=0.5, extent=extent)
    plt.title(f"idx={idx} true={class_names[y_true_idx[idx]]} pred={class_names[y_pred_idx[idx]]} p={probs[idx, y_pred_idx[idx]]:.3f}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"mis_{i}_idx{idx}.png"))
    plt.close()

print("Saved misclassified examples and confusion matrix to", OUT_DIR)
