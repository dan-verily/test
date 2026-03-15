"""
Streamlit UI: upload image + enter description -> urgency (1-5) + dog breed.
Run: streamlit run app.py
"""
import os
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import streamlit as st
import numpy as np
import tensorflow as tf
from pathlib import Path

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FUSION_PATH = os.path.join(SCRIPT_DIR, "multimodal_fusion_model.keras")
BREED_MODEL_PATH = os.path.join(SCRIPT_DIR, "dog_breed_model.keras")
BREED_TRAIN_DIR = os.path.join(SCRIPT_DIR, "dog_breed", "train")
IMG_SIZE = (160, 160)

PRIORITY_LABELS = {
    1: "Red / Immediate (most urgent)",
    2: "Orange / Emergent",
    3: "Yellow / Urgent",
    4: "Green / Semi-Urgent",
    5: "Blue / Non-Urgent (least urgent)",
}


@st.cache_resource
def load_fusion_model():
    """Load the saved fusion model once and cache it."""
    if not os.path.isfile(FUSION_PATH):
        return None
    tf.config.run_functions_eagerly(True)
    return tf.keras.models.load_model(FUSION_PATH)


@st.cache_resource
def load_breed_model_and_names():
    """Load dog breed model and class names (alphabetical order from dog_breed/train)."""
    if not os.path.isfile(BREED_MODEL_PATH):
        return None, []
    model = tf.keras.models.load_model(BREED_MODEL_PATH)
    class_names = []
    if os.path.isdir(BREED_TRAIN_DIR):
        class_names = sorted([d.name for d in Path(BREED_TRAIN_DIR).iterdir() if d.is_dir()])
    return model, class_names


def preprocess_image(uploaded_file):
    """Read uploaded image, resize to IMG_SIZE, return float32 tensor in [0, 255] for CNN."""
    raw = uploaded_file.read()
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    return tf.expand_dims(img, 0)  # (1, 160, 160, 3)


def predict_urgency(fusion_model, image_tensor, description: str):
    """Return (priority 1-5, label string, probs dict)."""
    text_tensor = tf.constant([[description.strip() or "No description provided."]])
    probs = fusion_model({"image": image_tensor, "text": text_tensor}, training=False)
    pred_class = int(tf.argmax(probs, axis=1).numpy()[0])  # 0-4
    priority = pred_class + 1  # 1-5
    prob_np = probs.numpy()[0]
    prob_dict = {i + 1: float(prob_np[i]) for i in range(5)}
    return priority, PRIORITY_LABELS.get(priority, "?"), prob_dict


def predict_breed(breed_model, image_tensor, class_names, top_k=5):
    """Return list of (breed_name, probability, relative_in_top5) for top_k."""
    logits = breed_model(image_tensor, training=False)
    probs = tf.nn.softmax(logits[0]).numpy()
    top_idx = np.argsort(probs)[::-1][:top_k]
    top_probs = [float(probs[i]) for i in top_idx]
    total_top = sum(top_probs)
    rel_top = [p / total_top if total_top > 0 else 1.0 / top_k for p in top_probs]
    names_list = [class_names[i] for i in top_idx] if class_names and len(class_names) == probs.shape[0] else [f"Class {i}" for i in top_idx]
    return list(zip(names_list, top_probs, rel_top))


# ---- UI ----
st.set_page_config(page_title="Pet Triage — Multimodal", layout="centered")
st.title("Pet Triage: Image + Description")
st.markdown("Upload a pet image and describe the situation. You get **urgency (priority 1–5)** and **dog breed** (if available).")

fusion_model = load_fusion_model()
breed_model, breed_class_names = load_breed_model_and_names()

if fusion_model is None:
    st.error(
        f"Fusion model not found at `{FUSION_PATH}`. "
        "Run `python MultimodalFusionLayer.py` first to train and save the model."
    )
    st.stop()

col1, col2 = st.columns(2)
with col1:
    image_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"], key="img")
with col2:
    description = st.text_area(
        "Description",
        placeholder="e.g. My dog is lethargic, not eating for two days, vomiting occasionally.",
        height=120,
        key="desc",
    )

if image_file is not None:
    st.image(image_file, caption="Uploaded image", use_container_width=True)

if st.button("Get priority & breed", type="primary"):
    if image_file is None:
        st.warning("Please upload an image.")
    elif not (description and description.strip()):
        st.warning("Please enter a description.")
    else:
        with st.spinner("Running models..."):
            image_tensor = preprocess_image(image_file)
            priority, label, prob_dict = predict_urgency(fusion_model, image_tensor, description)
            breed_results = predict_breed(breed_model, image_tensor, breed_class_names, top_k=5) if breed_model is not None else []
        st.success(f"**Priority: {priority}** — {label}")
        st.progress(prob_dict[priority], text=f"Urgency confidence: {prob_dict[priority]:.0%}")
        if breed_results:
            st.subheader("Dog breed")
            name1, prob1, rel1 = breed_results[0]
            st.write(f"**Predicted breed:** {name1}")
            st.caption(f"Raw confidence: {prob1:.1%} (over 120 breeds) · Among top 5: {rel1:.0%}")
            with st.expander("Top 5 breeds (raw %)"):
                for name, prob, rel in breed_results:
                    st.write(f"- **{name}**: {prob:.1%} (among top 5: {rel:.0%})")
            st.caption("With 120 breeds, raw probabilities are often low; the model still chooses the best match. Training on more data or fine-tuning can improve confidence.")
        else:
            st.info("Breed model not found. Add `dog_breed_model.keras` and `dog_breed/train` for breed prediction.")
        with st.expander("All urgency levels (probabilities)"):
            for p in range(1, 6):
                st.write(f"{p}. {PRIORITY_LABELS[p]}: {prob_dict[p]:.1%}")
