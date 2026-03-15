"""
Text models for veterinary triage urgency classification (S6_L1 style):
1) Bag-of-Words: Keras TextVectorization(multi_hot) + Dense linear classifier
2) Word Embedding + BiLSTM: Keras TextVectorization(int) + Embedding + BiLSTM
When run as main, loads triage_dataset_2500_clean.jsonl and trains both models.
"""
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Config (S6_L1 style)
# ---------------------------------------------------------------------------

BATCH_SIZE = 32
MAX_TOKENS = 20_000
MAX_LENGTH = 200   # sequence length for LSTM (S6_L1 uses 600 for IMDB)
HIDDEN_DIM = 64   # LSTM size and embedding dim, as in S6_L1
NUM_CLASSES = 5   # urgency levels 0-4


# ---------------------------------------------------------------------------
# Data loading and preprocessing (matches NLP_dog.ipynb)
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = "triage_dataset_2500_clean.jsonl"

URGENCY_MAPPING = {
    "Blue / Non-Urgent": 0,
    "Green / Semi-Urgent": 1,
    "Yellow / Urgent": 2,
    "Orange / Emergent": 3,
    "Red / Immediate": 4,
}


def load_and_prepare_data(data_path=DEFAULT_DATA_PATH):
    """Load JSONL, create clean_text and encoded_urgency, return train/val/test DataFrames."""
    df = pd.read_json(data_path, lines=True)

    df["medications"] = df["medications"].fillna("None")
    df["reproductive_status"] = df["reproductive_status"].fillna("Unknown")
    df["estimated_weight_lbs"] = df["estimated_weight_lbs"].fillna(df["estimated_weight_lbs"].median())

    df["clean_text"] = df["text"].str.lower()
    df["clean_text"] = df["clean_text"].apply(lambda x: re.sub(r"\s+", " ", x).strip())

    df["encoded_urgency"] = df["urgency_label"].map(URGENCY_MAPPING)
    df = df.dropna(subset=["encoded_urgency"]).reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["encoded_urgency"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["encoded_urgency"], random_state=42
    )

    return train_df, val_df, test_df


def _make_text_label_ds(texts, labels, shuffle=False):
    """Build tf.data.Dataset from (texts, labels) arrays."""
    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(texts), reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# 1) BAG-OF-WORDS (S6_L1: TextVectorization multi_hot + linear classifier)
# ---------------------------------------------------------------------------

def build_linear_classifier(max_tokens, num_classes, name="bag_of_words_classifier"):
    """Linear classifier on multi-hot bag-of-words (S6_L1 style)."""
    inputs = keras.Input(shape=(max_tokens,))
    outputs = layers.Dense(num_classes, activation="softmax")(inputs)
    model = keras.Model(inputs, outputs, name=name)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_and_train_bow_model(train_df, val_df, test_df, max_tokens=MAX_TOKENS, epochs=10, patience=2):
    """Train bag-of-words model: TextVectorization(multi_hot) + Dense (S6_L1 style)."""
    train_texts = train_df["clean_text"].values
    train_labels = train_df["encoded_urgency"].values
    val_texts = val_df["clean_text"].values
    val_labels = val_df["encoded_urgency"].values
    test_texts = test_df["clean_text"].values
    test_labels = test_df["encoded_urgency"].values

    text_vectorization = layers.TextVectorization(
        max_tokens=max_tokens,
        split="whitespace",
        output_mode="multi_hot",
    )
    text_vectorization.adapt(train_texts)

    def vectorize(x, y):
        return text_vectorization(x), y

    train_ds = _make_text_label_ds(train_texts, train_labels, shuffle=True)
    val_ds = _make_text_label_ds(val_texts, val_labels, shuffle=False)
    test_ds = _make_text_label_ds(test_texts, test_labels, shuffle=False)

    bag_of_words_train_ds = train_ds.map(vectorize, num_parallel_calls=tf.data.AUTOTUNE)
    bag_of_words_val_ds = val_ds.map(vectorize, num_parallel_calls=tf.data.AUTOTUNE)
    bag_of_words_test_ds = test_ds.map(vectorize, num_parallel_calls=tf.data.AUTOTUNE)

    model = build_linear_classifier(max_tokens, NUM_CLASSES, name="bag_of_words_classifier")
    model.summary(line_length=80)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        restore_best_weights=True,
        patience=patience,
    )
    history = model.fit(
        bag_of_words_train_ds,
        validation_data=bag_of_words_val_ds,
        epochs=epochs,
        callbacks=[early_stopping],
    )

    test_loss, test_acc = model.evaluate(bag_of_words_test_ds)
    print(f"\nBOW linear classifier - Test accuracy: {test_acc:.4f}")

    y_true = np.concatenate([y.numpy() for _, y in bag_of_words_test_ds], axis=0)
    y_pred = np.argmax(model.predict(bag_of_words_test_ds, verbose=0), axis=-1)
    print("\n=== Test set (BOW linear) ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(classification_report(y_true, y_pred, digits=4))

    return model, text_vectorization, history


# ---------------------------------------------------------------------------
# 2) WORD EMBEDDING + BiLSTM (S6_L1: Embedding + BiLSTM + Dense)
# ---------------------------------------------------------------------------

def build_and_train_embedding_lstm_model(
    train_df, val_df, test_df,
    max_tokens=MAX_TOKENS,
    max_length=MAX_LENGTH,
    hidden_dim=HIDDEN_DIM,
    epochs=10,
    patience=2,
):
    """Train word embedding + BiLSTM (S6_L1 style)."""
    train_texts = train_df["clean_text"].values
    train_labels = train_df["encoded_urgency"].values
    val_texts = val_df["clean_text"].values
    val_labels = val_df["encoded_urgency"].values
    test_texts = test_df["clean_text"].values
    test_labels = test_df["encoded_urgency"].values

    text_vectorization = layers.TextVectorization(
        max_tokens=max_tokens,
        split="whitespace",
        output_mode="int",
        output_sequence_length=max_length,
    )
    text_vectorization.adapt(train_texts)

    def vectorize(x, y):
        return text_vectorization(x), y

    train_ds = _make_text_label_ds(train_texts, train_labels, shuffle=True)
    val_ds = _make_text_label_ds(val_texts, val_labels, shuffle=False)
    test_ds = _make_text_label_ds(test_texts, test_labels, shuffle=False)

    sequence_train_ds = train_ds.map(vectorize, num_parallel_calls=tf.data.AUTOTUNE)
    sequence_val_ds = val_ds.map(vectorize, num_parallel_calls=tf.data.AUTOTUNE)
    sequence_test_ds = test_ds.map(vectorize, num_parallel_calls=tf.data.AUTOTUNE)

    inputs = keras.Input(shape=(max_length,), dtype="int32")
    x = layers.Embedding(
        input_dim=max_tokens,
        output_dim=hidden_dim,
        mask_zero=True,
    )(inputs)
    x = layers.Bidirectional(layers.LSTM(hidden_dim))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="lstm_with_embedding")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(line_length=80)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        restore_best_weights=True,
        patience=patience,
    )
    print("Training BiLSTM (may take several minutes per epoch on CPU)...")
    history = model.fit(
        sequence_train_ds,
        validation_data=sequence_val_ds,
        epochs=epochs,
        callbacks=[early_stopping],
    )

    test_loss, test_acc = model.evaluate(sequence_test_ds)
    print(f"\nBiLSTM Word-Embedding model - Test accuracy: {test_acc:.4f}")

    y_true = np.concatenate([y.numpy() for _, y in sequence_test_ds], axis=0)
    y_pred = np.argmax(model.predict(sequence_test_ds, verbose=0), axis=-1)
    print("\n=== Test set (WordEmbedding + BiLSTM) ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(classification_report(y_true, y_pred, digits=4))

    return model, text_vectorization, history


# ---------------------------------------------------------------------------
# Main: load data and train both models
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    data_path = os.path.join(os.path.dirname(__file__), DEFAULT_DATA_PATH)
    if not os.path.isfile(data_path):
        data_path = DEFAULT_DATA_PATH
    if not os.path.isfile(data_path):
        print("Data file not found. Expected:", DEFAULT_DATA_PATH)
        print("Load train_df, val_df, test_df (clean_text, encoded_urgency), then call:")
        print("  build_and_train_bow_model(train_df, val_df, test_df)")
        print("  build_and_train_embedding_lstm_model(train_df, val_df, test_df)")
        raise SystemExit(1)

    print("Loading and preparing data...")
    train_df, val_df, test_df = load_and_prepare_data(data_path)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("\n" + "=" * 60)
    print("1) BAG-OF-WORDS (multi_hot + linear classifier, S6_L1 style)")
    print("=" * 60)
    bow_model, bow_vec, bow_history = build_and_train_bow_model(
        train_df, val_df, test_df
    )

    print("\n" + "=" * 60)
    print("2) WORD EMBEDDING + BiLSTM (S6_L1 style)")
    print("=" * 60)
    lstm_model, lstm_vec, lstm_history = build_and_train_embedding_lstm_model(
        train_df, val_df, test_df
    )

    print("\nDone. Both models trained and evaluated.")
