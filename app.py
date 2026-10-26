# app.py (updated) — full file
import os
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import time
from skimage.restoration import denoise_wavelet
from scipy.signal import find_peaks, peak_prominences

# --- 0. Create eval_outputs folder to save examples for analysis (Option C) ---
os.makedirs("eval_outputs", exist_ok=True)

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="OCADN | Arrhythmia AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS (THE "PRO" LOOK") ---
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; margin: 0; font-size: 2.5rem; color: white !important; }
    .metric-card { background-color: white; padding: 20px; border-radius: 12px; border-left: 5px solid #1e3c72; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .card-success { border-left-color: #28a745; }
    .card-danger { border-left-color: #dc3545; }
    .big-font { font-size: 24px !important; font-weight: bold; }
    .med-font { font-size: 18px !important; color: #555; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD & PREPARE MODEL (cached) ---
@st.cache_resource
def load_and_prep_models(model_path='final_model.h5'):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Build explanation model (linear logits head) so grad-cam gradients exist
    last_hidden_output = model.layers[-2].output
    final_layer_config = model.layers[-1].get_config()
    final_layer_config['activation'] = 'linear'
    final_layer_config['name'] = 'logits_layer'
    new_final_layer = tf.keras.layers.Dense.from_config(final_layer_config)
    new_output = new_final_layer(last_hidden_output)
    expl_model = tf.keras.models.Model(inputs=model.input, outputs=new_output)
    # copy weights
    expl_model.layers[-1].set_weights(model.layers[-1].get_weights())

    return model, expl_model

# --- SIDEBAR (controls including new improvements) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=80)
    st.title("OCADN Control")
    st.info("System Ready")
    st.markdown("---")
    st.write("This tool uses Deep Learning (CNN) + Grad-CAM to detect and explain cardiac arrhythmias.")
    st.markdown("### 📋 Instructions")
    st.markdown("1. Upload a CSV file.")
    st.markdown("2. Wait for AI analysis.")
    st.markdown("3. Review heatmap and results.")
    st.markdown("---")
    # Option A: Confidence threshold
    confidence_threshold = st.slider("Confidence threshold (%) — predictions below this are marked 'Uncertain'", 50, 99, 70)
    # Option B: Multi-beat majority voting
    multi_beat_mode = st.checkbox("Enable multi-beat majority voting", value=True)
    n_beats_to_use = st.slider("Beats used for majority vote (odd number)", 1, 7, 3, step=2)
    # Option C: Save examples for later analysis
    save_examples = st.checkbox("Save explanation (segment + heatmap + preds) for analysis", value=False)
    # Option D: Quick test datasets (toggle)
    show_test_instructions = st.checkbox("Show quick test instructions & sample generator", value=False)

# --- MAIN HEADER ---
st.markdown("""
<div class="main-header">
    <h1>OCADN Arrhythmia Detector</h1>
    <p>Optimized Cardiac Arrhythmia Detection Network with Explainable AI</p>
</div>
""", unsafe_allow_html=True)

# Load models and class names
try:
    with st.spinner("Initializing AI Models..."):
        model, expl_model = load_and_prep_models()
        class_names = np.load('data/classes.npy')
        # find last conv layer name
        last_conv_layer_name = ""
        for layer in reversed(expl_model.layers):
            if 'conv1d' in layer.name:
                last_conv_layer_name = layer.name
                break
except Exception as e:
    st.error(f"Error loading model or class names: {e}")
    st.stop()

# --- Utility functions -----------------------------------------------------

def robust_scaling(signal):
    """Median / IQR scaling used in training; safe fallback to z-score."""
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

def make_gradcam_heatmap(img_array, expl_model, last_conv_layer_name):
    """Produces 1D heatmap aligned to last conv time-axis (normalized 0..1)."""
    grad_model = tf.keras.models.Model(
        inputs=expl_model.input,
        outputs=[expl_model.get_layer(last_conv_layer_name).output, expl_model.output]
    )
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_tensor)  # (1, T, C), (1, C)
        pred_index = tf.argmax(preds[0], axis=-1)
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)  # (1, T, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # (C,)
    feature_maps = last_conv_layer_output[0]  # (T, C)
    weighted_maps = feature_maps * pooled_grads[tf.newaxis, :]
    heatmap = tf.reduce_sum(weighted_maps, axis=-1)  # (T,)
    heatmap = tf.nn.relu(heatmap)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros((heatmap.shape[0],), dtype=np.float32)
    heatmap = heatmap / (max_val + 1e-10)
    return heatmap.numpy().astype(np.float32)

def predict_single_segment(seg, expl_model):
    """Return predicted index, label, confidence, raw preds for a single processed seg (1, T, 1)."""
    raw_preds = expl_model.predict(seg, verbose=0)  # logits (1, C)
    probs = tf.nn.softmax(raw_preds, axis=-1).numpy()  # (1, C)
    pred_idx = int(np.argmax(probs[0]))
    pred_label = class_names[pred_idx]
    confidence = float(probs[0, pred_idx] * 100.0)
    return pred_idx, pred_label, confidence, probs[0]

def save_example(seg_raw, seg_scaled, heatmap, probs, pred_label, idx_hint=0):
    """Save numpy arrays and a small JSON/CSV for offline analysis (Option C)."""
    t = int(time.time())
    base = f"eval_outputs/example_{t}_{idx_hint}"
    np.save(base + "_raw.npy", seg_raw.astype(np.float32))
    np.save(base + "_scaled.npy", seg_scaled.astype(np.float32))
    np.save(base + "_heatmap.npy", heatmap.astype(np.float32))
    pd.DataFrame(probs.reshape(1, -1), columns=class_names).to_csv(base + "_probs.csv", index=False)
    with open(base + "_meta.txt", "w") as f:
        f.write(f"pred_label={pred_label}\n")
    return base

def extract_beats_from_trace(trace, n_beats=3, distance_samples=80, prom_factor=0.5):
    """
    Returns up to n_beats center indices using peak detection with fallback.
    prom_factor is fraction of std used as default prominence.
    """
    if len(trace) == 0:
        return []
    stdv = np.std(trace)
    prom = stdv * prom_factor if stdv > 0 else 0.05
    peaks, props = find_peaks(trace, distance=distance_samples, prominence=prom)
    # fallback steps if none found
    if len(peaks) == 0:
        prom2 = stdv * (prom_factor / 2) if stdv > 0 else 0.02
        peaks, props = find_peaks(trace, distance=int(distance_samples * 0.75), prominence=prom2)
    if len(peaks) == 0:
        # try very permissive
        peaks, props = find_peaks(trace, distance=int(distance_samples * 0.5), prominence=0.01)
    if len(peaks) == 0:
        return []
    # prefer most-prominent peaks
    prominences = peak_prominences(trace, peaks)[0]
    order = np.argsort(prominences)[::-1]  # descending
    selected = peaks[order][:n_beats]
    # sort by time for nicer plotting
    return np.sort(selected)

# --- UI: FILE UPLOAD SECTION ---
st.markdown("### 📂 Patient Data Input")
uploaded_file = st.file_uploader("Drop patient ECG file here (CSV format). Single column or long trace accepted.", type=['csv'])

if show_test_instructions:
    st.markdown("""
    **Quick test instructions**
    - To reproduce training-like results, upload single-beat CSVs containing exactly 360 samples (one value per line).
    - Or upload a longer trace (many seconds). The app will detect peaks and explain the most-prominent beats.
    - Use the sidebar controls to toggle majority-vote, change the confidence threshold, and save examples for later analysis.
    """)

if uploaded_file is not None:
    # ---------- START robust read & preprocessing ----------
    try:
        df = pd.read_csv(uploaded_file, header=None)
    except Exception:
        df = pd.read_csv(uploaded_file)
    col0 = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values.astype(np.float32)
    if col0.size == 0:
        st.error("Uploaded CSV contains no numeric data in the first column.")
        st.stop()
    raw_trace = col0  # full trace

    # Denoise (Symlet4 + BayesShrink) — matches training
    denoised = denoise_wavelet(raw_trace, method='BayesShrink', mode='soft',
                               wavelet_levels=3, wavelet='sym4', rescale_sigma=True)

    # Decide extraction strategy: single-beat or multi-beat based on user choice
    if multi_beat_mode:
        # extract up to n_beats_to_use peaks and analyze them
        center_peaks = extract_beats_from_trace(denoised, n_beats=n_beats_to_use, distance_samples=80, prom_factor=0.5)
        if len(center_peaks) == 0:
            st.error("Could not detect any clear R-peaks in the uploaded signal. Upload a single-beat CSV or a longer trace containing beats.")
            st.stop()
    else:
        # pick the single most prominent peak
        centers = extract_beats_from_trace(denoised, n_beats=1, distance_samples=80, prom_factor=0.5)
        if len(centers) == 0:
            st.error("Could not detect any clear R-peaks in the uploaded signal.")
            st.stop()
        center_peaks = centers

    # For majority voting we will gather predictions
    per_beat_results = []
    processed_segments = []
    heatmaps = []

    # window extraction parameters
    window_size = 360
    half_w = window_size // 2

    for i, chosen_peak in enumerate(center_peaks):
        start = int(chosen_peak - half_w)
        end = int(chosen_peak + half_w)
        if start < 0:
            pad_left = -start
            seg = np.pad(denoised[:end], (pad_left, 0), 'constant')
        elif end > len(denoised):
            pad_right = end - len(denoised)
            seg = np.pad(denoised[start:], (0, pad_right), 'constant')
        else:
            seg = denoised[start:end]
        # if segment length differs fix it
        if seg.shape[0] != window_size:
            seg = np.resize(seg, (window_size,))

        seg_scaled = robust_scaling(seg)
        seg_reshaped = seg_scaled.reshape(1, window_size, 1)
        processed_segments.append((seg, seg_scaled, seg_reshaped))

    # Fake loading bar for UX
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.003)
        progress_bar.progress(i + 1)
    progress_bar.empty()

    # Predict each extracted beat
    for idx, (seg_raw, seg_scaled, seg_reshaped) in enumerate(processed_segments):
        pred_idx, pred_label, confidence, probs = predict_single_segment(seg_reshaped, expl_model)
        heatmap = make_gradcam_heatmap(seg_reshaped, expl_model, last_conv_layer_name)
        per_beat_results.append({
            "index": idx,
            "pred_idx": pred_idx,
            "pred_label": pred_label,
            "confidence": confidence,
            "probs": probs
        })
        heatmaps.append(heatmap)

    # If multi-beat and >1 beat selected, do majority voting (Option B)
    if multi_beat_mode and len(per_beat_results) > 1:
        votes = [r["pred_idx"] for r in per_beat_results]
        counts = np.bincount(votes)
        majority_idx = int(np.argmax(counts))
        majority_label = class_names[majority_idx]
        # compute average confidence for majority class across beats (where predicted == majority)
        maj_conf_list = [r["confidence"] for r in per_beat_results if r["pred_idx"] == majority_idx]
        majority_confidence = float(np.mean(maj_conf_list)) if len(maj_conf_list) > 0 else float(np.mean([r["confidence"] for r in per_beat_results]))
        final_pred_idx = majority_idx
        final_label = majority_label
        final_confidence = majority_confidence
    else:
        # single-beat mode or only one extracted: take the first
        final_pred_idx = per_beat_results[0]["pred_idx"]
        final_label = per_beat_results[0]["pred_label"]
        final_confidence = per_beat_results[0]["confidence"]

    # Apply confidence threshold (Option A)
    is_uncertain = final_confidence < confidence_threshold
    display_label = "Uncertain" if is_uncertain else final_label

    # Option C: Save examples if requested (save the first segment and its heatmap)
    saved_example_path = None
    if save_examples:
        # save the first segment and its metadata
        seg_raw, seg_scaled, _ = processed_segments[0]
        saved_example_path = save_example(seg_raw, seg_scaled, heatmaps[0], per_beat_results[0]["probs"], display_label, idx_hint=0)
        st.success(f"Saved example for analysis: {saved_example_path}_*")

    # --- D. VISUALIZATION (Professional Layout) ---
    st.markdown("---")
    col1, col2 = st.columns([1, 2], gap="large")

    # Left column: summary and per-beat details
    with col1:
        st.markdown("### 🏥 Diagnosis Report")
        card_class = "card-success" if (not is_uncertain and final_label == 'N') else "card-danger"
        text_color = "green" if (not is_uncertain and final_label == 'N') else "#d9534f"
        diagnosis_text = "Normal Sinus Rhythm" if (not is_uncertain and final_label == 'N') else (f"Arrhythmia Detected ({final_label})" if not is_uncertain else "Uncertain Prediction")

        st.markdown(f"""
        <div class="metric-card {card_class}">
            <p class="med-font">Prediction</p>
            <h2 style="color: {text_color}; margin-top: -10px;">{diagnosis_text}</h2>
            <p class="med-font">Confidence Score</p>
            <h2 style="color: #333; margin-top: -10px;">{final_confidence:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📊 View Detailed Per-beat Predictions"):
            # Build table of beat-level predictions
            rows = []
            for i, r in enumerate(per_beat_results):
                rows.append({
                    "Beat #": i + 1,
                    "Predicted": r["pred_label"],
                    "Confidence %": f"{r['confidence']:.1f}"
                })
            st.table(pd.DataFrame(rows))

        if save_examples and saved_example_path:
            st.info("Saved example for offline analysis in `eval_outputs/`.")

    # Right column: explainable plots (show first heatmap and aggregated view)
    with col2:
        st.markdown("### 👁️ Explainable AI Analysis")
        st.markdown("""<div style="background-color: white; padding: 10px; border-radius: 8px;"><small>The red zones indicate where the model focused. For multi-beat mode you can inspect each beat below.</small></div>""", unsafe_allow_html=True)

        # Plot 1: show the denoised long trace with detected peaks (for context)
        fig_ctx, ax_ctx = plt.subplots(figsize=(10, 2))
        ax_ctx.plot(denoised, color='#444', linewidth=1)
        ax_ctx.plot(center_peaks, denoised[center_peaks], "x", color='red')
        ax_ctx.set_title("Denoised Trace (detected peaks shown)")
        ax_ctx.set_xlabel("Samples")
        st.pyplot(fig_ctx)

        # For each extracted beat, plot scaled seg + its heatmap
        for i, (seg_raw, seg_scaled, seg_reshaped) in enumerate(processed_segments):
            hm = heatmaps[i]
            # create color map image
            hm_img = hm.reshape(1, -1)
            hm_resized = cv2.resize(hm_img, (360, 1))
            hm_resized = np.uint8(255 * hm_resized)
            heatmap_color = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            fig, ax = plt.subplots(figsize=(10, 3.2))
            ax.plot(seg_scaled, color='#2c3e50', linewidth=2)
            extent = [0, 360, np.min(seg_scaled), np.max(seg_scaled)]
            ax.imshow(heatmap_color, aspect='auto', alpha=0.5, extent=extent)
            beat_pred = per_beat_results[i]["pred_label"]
            beat_conf = per_beat_results[i]["confidence"]
            ax.set_title(f"Beat #{i+1} — Prediction: {beat_pred} ({beat_conf:.1f}%)")
            ax.set_xlabel("Samples")
            st.pyplot(fig)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 20px;">
        <small>⚠️ <b>Medical Disclaimer:</b> This tool is a prototype for research purposes only. 
        It is not a substitute for professional medical advice.</small>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👋 Waiting for ECG file upload...")

