"""
Multi-task veterinary triage model (Keras/TF):
  - Urgency classification (1-5)
  - NER extraction (BIO tags for 9 entity types)

Uses BioBERT (via HuggingFace TFAutoModel) with Keras functional API.

Run: python nlp_gpu_kr.py
Requires: pip install transformers tensorflow scikit-learn
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from transformers import AutoTokenizer, TFAutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---- Config ----
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 15
LR = 2e-5

DATA_PATH = "triage_dataset_2500_clean.jsonl"

# ---- NER label scheme (BIO) ----
ENTITY_TYPES = ["AGE", "BREED", "DURATION", "EXPOSURE", "MEDICATION",
                "PRE_EXISTING", "SEX_STATUS", "SYMPTOM", "TOXIN"]

NER_LABELS = ["O"]
for etype in ENTITY_TYPES:
    NER_LABELS.append(f"B-{etype}")
    NER_LABELS.append(f"I-{etype}")
NER_LABEL2ID = {l: i for i, l in enumerate(NER_LABELS)}
NUM_NER_LABELS = len(NER_LABELS)

NUM_URGENCY_CLASSES = 5

URGENCY_NAMES = {
    0: "Red / Immediate",
    1: "Orange / Emergent",
    2: "Yellow / Urgent",
    3: "Green / Semi-Urgent",
    4: "Blue / Non-Urgent",
}


# ---- Data loading ----
def load_dataset(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            samples.append({
                "text": d["text"],
                "urgency": d["urgency_level"] - 1,
                "entities": d.get("entities", []),
            })
    return samples


def align_entities_to_tokens(text, entities, tokenizer, max_len):
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )

    offsets = encoding["offset_mapping"]
    ner_labels = [-100] * max_len

    for i, (start, end) in enumerate(offsets):
        if start == 0 and end == 0:
            continue
        ner_labels[i] = NER_LABEL2ID["O"]

    for ent in entities:
        phrase = ent["phrase"]
        label = ent["label"]
        if label not in ENTITY_TYPES:
            continue

        text_lower = text.lower()
        phrase_lower = phrase.lower()
        char_start = text_lower.find(phrase_lower)
        if char_start == -1:
            continue
        char_end = char_start + len(phrase)

        first = True
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end == 0:
                continue
            if tok_start >= char_start and tok_end <= char_end:
                tag = f"B-{label}" if first else f"I-{label}"
                ner_labels[i] = NER_LABEL2ID[tag]
                first = False

    encoding.pop("offset_mapping")
    return encoding, ner_labels


def prepare_data(samples, tokenizer, max_len):
    input_ids_list = []
    attention_mask_list = []
    urgency_labels = []
    ner_labels_list = []

    for s in samples:
        encoding, ner_labels = align_entities_to_tokens(
            s["text"], s["entities"], tokenizer, max_len
        )
        input_ids_list.append(encoding["input_ids"])
        attention_mask_list.append(encoding["attention_mask"])
        urgency_labels.append(s["urgency"])
        ner_labels_list.append(ner_labels)

    return {
        "input_ids": np.array(input_ids_list, dtype=np.int32),
        "attention_mask": np.array(attention_mask_list, dtype=np.int32),
        "urgency_labels": np.array(urgency_labels, dtype=np.int32),
        "ner_labels": np.array(ner_labels_list, dtype=np.int32),
    }


# ---- Model ----
def build_model(model_name, max_len, num_urgency_classes, num_ner_labels):
    # Inputs
    input_ids = keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = keras.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    # BioBERT backbone
    bert = TFAutoModel.from_pretrained(model_name, from_pt=True)
    bert_output = bert(input_ids, attention_mask=attention_mask)

    # Urgency head: [CLS] token -> Dense -> 5 classes
    cls_output = bert_output.last_hidden_state[:, 0, :]
    urgency_output = keras.layers.Dense(
        num_urgency_classes, activation="softmax", name="urgency"
    )(cls_output)

    # NER head: all tokens -> Dense -> BIO tags
    token_output = bert_output.last_hidden_state
    ner_output = keras.layers.Dense(
        num_ner_labels, activation="softmax", name="ner"
    )(token_output)

    model = keras.Model(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        outputs={"urgency": urgency_output, "ner": ner_output},
    )
    return model


# ---- Custom NER loss (ignores -100 labels) ----
class MaskedSparseCrossEntropy(keras.losses.Loss):
    def call(self, y_true, y_pred):
        mask = tf.not_equal(y_true, -100)
        y_true_masked = tf.where(mask, y_true, tf.zeros_like(y_true))
        loss = keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred)
        loss = tf.where(mask, loss, tf.zeros_like(loss))
        return tf.reduce_sum(loss) / (tf.reduce_sum(tf.cast(mask, tf.float32)) + 1e-8)


# ---- Main ----
if __name__ == "__main__":
    print("Loading data...")
    samples = load_dataset(DATA_PATH)
    print(f"Total samples: {len(samples)}")

    train_samples, temp_samples = train_test_split(samples, test_size=0.2, random_state=42)
    val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing data...")
    train_data = prepare_data(train_samples, tokenizer, MAX_LEN)
    val_data = prepare_data(val_samples, tokenizer, MAX_LEN)
    test_data = prepare_data(test_samples, tokenizer, MAX_LEN)

    print(f"Building model: {MODEL_NAME}")
    model = build_model(MODEL_NAME, MAX_LEN, NUM_URGENCY_CLASSES, NUM_NER_LABELS)

    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss={
            "urgency": "sparse_categorical_crossentropy",
            "ner": MaskedSparseCrossEntropy(),
        },
        loss_weights={"urgency": 1.0, "ner": 0.3},
        metrics={"urgency": "accuracy"},
    )

    model.summary(line_length=100)

    # Train
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="triage_multitask_kr.keras",
            save_best_only=True,
            monitor="val_loss",
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        x={"input_ids": train_data["input_ids"], "attention_mask": train_data["attention_mask"]},
        y={"urgency": train_data["urgency_labels"], "ner": train_data["ner_labels"]},
        validation_data=(
            {"input_ids": val_data["input_ids"], "attention_mask": val_data["attention_mask"]},
            {"urgency": val_data["urgency_labels"], "ner": val_data["ner_labels"]},
        ),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Evaluate on test
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    outputs = model.predict(
        {"input_ids": test_data["input_ids"], "attention_mask": test_data["attention_mask"]},
        batch_size=BATCH_SIZE,
    )

    # Urgency
    urg_preds = np.argmax(outputs["urgency"], axis=1)
    urg_labels = test_data["urgency_labels"]
    urgency_names = [URGENCY_NAMES[i] for i in range(NUM_URGENCY_CLASSES)]
    print("\n--- Urgency Classification ---")
    print(classification_report(urg_labels, urg_preds, target_names=urgency_names, digits=3))

    # NER
    ner_preds_flat = np.argmax(outputs["ner"], axis=2).flatten()
    ner_labels_flat = test_data["ner_labels"].flatten()
    mask = ner_labels_flat != -100
    ner_preds_masked = ner_preds_flat[mask]
    ner_labels_masked = ner_labels_flat[mask]

    ner_label_names_used = sorted(set(ner_labels_masked.tolist() + ner_preds_masked.tolist()))
    ner_target_names = [NER_LABELS[i] for i in ner_label_names_used]
    print("--- NER (token-level) ---")
    print(classification_report(
        ner_labels_masked, ner_preds_masked,
        labels=ner_label_names_used,
        target_names=ner_target_names,
        digits=3,
        zero_division=0,
    ))

    print("Done.")
