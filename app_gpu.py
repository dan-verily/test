"""
Streamlit UI: upload image + enter description -> urgency + NER + dog breed.
Uses separate models:
  - Image (Keras): EfficientNetB3 for breed classification
  - Text (PyTorch/BioBERT): urgency + NER

Run: streamlit run app_gpu.py
"""
import os
import numpy as np
import streamlit as st
import tensorflow as tf
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# ---- Paths ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BREED_MODEL_PATH = os.path.join(SCRIPT_DIR, "dog_breed_model_gpu_fine_tune_300_100_B3.keras")
BREED_TRAIN_DIR = os.path.join(SCRIPT_DIR, "dog_breed", "train")
NLP_MODEL_PATH = os.path.join(SCRIPT_DIR, "triage_multitask_model.pt")
IMG_SIZE = (300, 300)
NLP_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

PRIORITY_LABELS = {
    1: "Red / Immediate (most urgent)",
    2: "Orange / Emergent",
    3: "Yellow / Urgent",
    4: "Green / Semi-Urgent",
    5: "Blue / Non-Urgent (least urgent)",
}

# ---- NER config (must match nlp_gpu.py) ----
ENTITY_TYPES = ["AGE", "BREED", "DURATION", "EXPOSURE", "MEDICATION",
                "PRE_EXISTING", "SEX_STATUS", "SYMPTOM", "TOXIN"]

NER_LABELS = ["O"]
for etype in ENTITY_TYPES:
    NER_LABELS.append(f"B-{etype}")
    NER_LABELS.append(f"I-{etype}")
NUM_NER_LABELS = len(NER_LABELS)

NUM_URGENCY_CLASSES = 5


# ---- NLP Model (same as nlp_gpu.py) ----
class TriageMultiTaskModel(nn.Module):
    def __init__(self, model_name, num_urgency_classes, num_ner_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.urgency_head = nn.Linear(hidden_size, num_urgency_classes)
        self.ner_head = nn.Linear(hidden_size, num_ner_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        token_outputs = outputs.last_hidden_state
        urgency_logits = self.urgency_head(cls_output)
        ner_logits = self.ner_head(token_outputs)
        return urgency_logits, ner_logits


# ---- Load models ----
@st.cache_resource
def load_breed_model_and_names():
    if not os.path.isfile(BREED_MODEL_PATH):
        return None, []
    model = tf.keras.models.load_model(BREED_MODEL_PATH)
    class_names = []
    if os.path.isdir(BREED_TRAIN_DIR):
        class_names = sorted([d.name for d in Path(BREED_TRAIN_DIR).iterdir() if d.is_dir()])
    return model, class_names


@st.cache_resource
def load_nlp_model():
    if not os.path.isfile(NLP_MODEL_PATH):
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(NLP_MODEL_NAME)
    model = TriageMultiTaskModel(NLP_MODEL_NAME, NUM_URGENCY_CLASSES, NUM_NER_LABELS)
    model.load_state_dict(torch.load(NLP_MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, tokenizer


# ---- Image prediction ----
def preprocess_image(uploaded_file):
    raw = uploaded_file.read()
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    return tf.expand_dims(img, 0)


def predict_breed(breed_model, image_tensor, class_names, top_k=5):
    logits = breed_model(image_tensor, training=False)
    probs = tf.nn.softmax(logits[0]).numpy()
    top_idx = np.argsort(probs)[::-1][:top_k]
    top_probs = [float(probs[i]) for i in top_idx]
    total_top = sum(top_probs)
    rel_top = [p / total_top if total_top > 0 else 1.0 / top_k for p in top_probs]
    names_list = [class_names[i] for i in top_idx] if class_names and len(class_names) == probs.shape[0] else [f"Class {i}" for i in top_idx]
    return list(zip(names_list, top_probs, rel_top))


# ---- Text prediction ----
def predict_text(nlp_model, tokenizer, description):
    device = next(nlp_model.parameters()).device
    encoding = tokenizer(
        description, max_length=256, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        urg_logits, ner_logits = nlp_model(input_ids, attention_mask)

    # Urgency
    urg_probs = torch.softmax(urg_logits, dim=1)[0]
    urg_pred = urg_probs.argmax().item()
    priority = urg_pred + 1
    prob_dict = {i + 1: float(urg_probs[i]) for i in range(5)}

    # NER
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    ner_pred_ids = ner_logits.argmax(dim=2)[0].cpu().tolist()

    entities = []
    current_entity = None
    current_tokens = []

    for tok, pred_id, mask in zip(tokens, ner_pred_ids, attention_mask[0].tolist()):
        if mask == 0 or tok in ["[CLS]", "[SEP]", "[PAD]"]:
            if current_entity:
                entities.append((current_entity, current_tokens))
                current_entity = None
                current_tokens = []
            continue

        tag = NER_LABELS[pred_id]
        if tag.startswith("B-"):
            if current_entity:
                entities.append((current_entity, current_tokens))
            current_entity = tag[2:]
            current_tokens = [tok]
        elif tag.startswith("I-") and current_entity == tag[2:]:
            current_tokens.append(tok)
        else:
            if current_entity:
                entities.append((current_entity, current_tokens))
                current_entity = None
                current_tokens = []

    if current_entity:
        entities.append((current_entity, current_tokens))

    # Reconstruct entity text
    entity_list = []
    for etype, toks in entities:
        phrase = tokenizer.convert_tokens_to_string(toks)
        entity_list.append((etype, phrase))

    return priority, prob_dict, entity_list


# ---- UI ----
st.set_page_config(page_title="Pet Triage", layout="centered")
st.title("Pet Triage: Image + Description")
st.markdown(
    "Upload a pet image and describe the situation. "
    "Get **urgency**, **extracted entities**, and **breed prediction**."
)

breed_model, breed_class_names = load_breed_model_and_names()
nlp_model, tokenizer = load_nlp_model()

if nlp_model is None:
    st.error(
        f"NLP model not found at `{NLP_MODEL_PATH}`. "
        "Run `python nlp_gpu.py` first to train and save the model."
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

if st.button("Analyze", type="primary"):
    if not (description and description.strip()):
        st.warning("Please enter a description.")
    else:
        with st.spinner("Running models..."):
            # Text analysis (always)
            priority, prob_dict, entity_list = predict_text(
                nlp_model, tokenizer, description
            )
            # Image analysis (if uploaded)
            breed_results = []
            if image_file is not None and breed_model is not None:
                image_tensor = preprocess_image(image_file)
                breed_results = predict_breed(breed_model, image_tensor, breed_class_names, top_k=5)

        # ---- Urgency ----
        st.success(f"**Priority: {priority}** — {PRIORITY_LABELS.get(priority, '?')}")
        st.progress(prob_dict[priority], text=f"Urgency confidence: {prob_dict[priority]:.0%}")

        # ---- Entities (NER) ----
        if entity_list:
            st.subheader("Extracted Entities")
            for etype, phrase in entity_list:
                st.write(f"- **[{etype}]** {phrase}")
        else:
            st.info("No entities detected in the description.")

        # ---- Breed ----
        if breed_results:
            st.subheader("Dog Breed")
            name1, prob1, rel1 = breed_results[0]
            st.write(f"**Predicted breed:** {name1}")
            st.caption(f"Raw confidence: {prob1:.1%} (over 120 breeds) · Among top 5: {rel1:.0%}")
            with st.expander("Top 5 breeds"):
                for name, prob, rel in breed_results:
                    st.write(f"- **{name}**: {prob:.1%} (among top 5: {rel:.0%})")
        elif image_file is not None:
            st.info("Breed model not found. Add the trained model for breed prediction.")

        # ---- Details ----
        with st.expander("All urgency levels"):
            for p in range(1, 6):
                st.write(f"{p}. {PRIORITY_LABELS[p]}: {prob_dict[p]:.1%}")
