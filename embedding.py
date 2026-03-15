import os
# Reduce threading to avoid hangs on some Mac/TF setups (must set before importing tf)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
import json
import warnings
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.config.run_functions_eagerly(True)
warnings.filterwarnings(
    "ignore",
    message=".*tf.config.experimental_run_functions_eagerly.*tf.data.*",
    category=UserWarning,
    module="tensorflow",
)
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
# ==========================================
# 1. LOAD AND PREPARE THE DATA
# ==========================================
# Priority scale: 1 = most urgent (worst), 5 = least urgent (routine)
PRIORITY_LABELS = {
    1: "Red / Immediate (most urgent)",
    2: "Orange / Emergent",
    3: "Yellow / Urgent",
    4: "Green / Semi-Urgent",
    5: "Blue / Non-Urgent (least urgent)",
}
file_path = 'triage_dataset_2500_clean.jsonl'

# ==========================================
# 1. LOAD AND PREPARE THE DATA
# ==========================================
def load_data(file_path):
    print("Loading and preparing data...")
    texts = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['text'])
            # Convert labels 1-5 to 0-4 for Keras
            labels.append(data['urgency_level'] - 1)

    return np.array(texts), np.array(labels)


# ==========================================
# 2. BAG-OF-WORDS MODEL (Fixed)
# ==========================================
def build_and_train_bow_model(train_x, train_y, val_x, val_y):
    print("\n" + "=" * 60)
    print("1) BAG-OF-WORDS MODEL (Fixed Input Size)")
    print("=" * 60)

    # TextVectorization for Bag of Words (multi-hot encoding)
    # We allow it to find up to 20000 words, but it will only use what it finds
    vectorizer = layers.TextVectorization(
        max_tokens=20000,
        output_mode='multi_hot'
    )

    print("Adapting BoW vectorizer...")
    vectorizer.adapt(train_x)

    # ------------------------------------------------------------------
    # THE FIX: We get the exact number of words it actually found
    # (In your dataset, this will be around 4,026 instead of 20,000)
    # ------------------------------------------------------------------
    actual_vocab_size = len(vectorizer.get_vocabulary())
    print(f"Actual vocabulary size found: {actual_vocab_size}")

    # Vectorize the text data before passing to the model.
    # Use .numpy() so fit() gets plain arrays and avoids first-batch graph-compilation hang.
    x_train_bow = vectorizer(train_x).numpy()
    x_val_bow = vectorizer(val_x).numpy()

    # Build the model using the EXACT vocab size it found
    model = models.Sequential([
        tf.keras.Input(shape=(actual_vocab_size,)),  # <--- FIXED HERE
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(5, activation='softmax')
    ], name="bag_of_words_classifier")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Pure GradientTape loop — no train_on_batch/evaluate (avoids first-call hang)
    batch_size = 32
    n_epochs = 10
    n_samples = len(x_train_bow)
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    print("\nTraining BoW Model...")
    # One-time forward pass so first batch in the loop doesn’t hang on graph/trace
    print("Warmup forward pass...", flush=True)
    _ = model(tf.constant(x_train_bow[:1].astype("float32")), training=True)
    print("Warmup done.", flush=True)
    for epoch in range(n_epochs):
        perm = np.random.permutation(n_samples)
        x_epoch = x_train_bow[perm].astype("float32")
        y_epoch = train_y[perm]
        epoch_loss, epoch_correct, epoch_n = [], 0, 0
        n_batches = (n_samples + batch_size - 1) // batch_size
        for b, start in enumerate(range(0, n_samples, batch_size)):
            end = min(start + batch_size, n_samples)
            bx = tf.constant(x_epoch[start:end])
            by = tf.constant(y_epoch[start:end])
            with tf.GradientTape() as tape:
                logits = model(bx, training=True)
                loss = loss_fn(by, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss.append(float(loss))
            preds = tf.argmax(logits, axis=1)
            epoch_correct += int(tf.reduce_sum(tf.cast(preds == by, tf.int32)))
            epoch_n += by.shape[0]
            if epoch == 0:
                print(f"  batch {b + 1}/{n_batches} done", flush=True)
        train_loss = np.mean(epoch_loss)
        train_acc = epoch_correct / epoch_n
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        # Validation (no gradient)
        val_logits = model(tf.constant(x_val_bow.astype("float32")), training=False)
        val_loss = float(loss_fn(tf.constant(val_y), val_logits))
        val_acc = float(tf.reduce_mean(tf.cast(tf.argmax(val_logits, axis=1) == val_y, tf.float32)))
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        print(f"Epoch {epoch + 1}/{n_epochs} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

    return model, vectorizer, history


# ==========================================
# 3. EMBEDDING + SEQUENCE MODEL (Phase 1)
# ==========================================
def build_and_train_embedding_model(train_x, train_y, val_x, val_y):
    print("\n" + "=" * 60)
    print("2) EMBEDDING MODEL (Understands word order)")
    print("=" * 60)

    max_vocab_size = 10000
    max_sequence_length = 150

    # TextVectorization for Sequences (integer encoding)
    vectorize_layer = layers.TextVectorization(
        max_tokens=max_vocab_size,
        output_mode='int',
        output_sequence_length=max_sequence_length
    )

    print("Adapting Embedding vectorizer...")
    vectorize_layer.adapt(train_x)

    model = models.Sequential([
        # Accept raw text
        tf.keras.Input(shape=(1,), dtype=tf.string),

        # Turn text into sequences
        vectorize_layer,

        # Learn word meanings (Embedding)
        layers.Embedding(input_dim=max_vocab_size, output_dim=64, mask_zero=True),

        # Read the sequence
        layers.GlobalAveragePooling1D(),

        # Hidden layer
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),

        # Output layer (5 urgency levels)
        layers.Dense(5, activation='softmax')
    ], name="embedding_classifier")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Pure GradientTape loop — no train_on_batch/evaluate (avoids first-call hang)
    batch_size = 32
    n_epochs = 10
    n_samples = len(train_x)
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    print("\nTraining Embedding Model...")
    for epoch in range(n_epochs):
        perm = np.random.permutation(n_samples)
        x_epoch = train_x[perm]
        y_epoch = train_y[perm]
        epoch_loss, epoch_correct, epoch_n = [], 0, 0
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            bx = tf.constant(x_epoch[start:end])
            by = tf.constant(y_epoch[start:end])
            with tf.GradientTape() as tape:
                logits = model(bx, training=True)
                loss = loss_fn(by, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss.append(float(loss))
            preds = tf.argmax(logits, axis=1)
            epoch_correct += int(tf.reduce_sum(tf.cast(preds == by, tf.int32)))
            epoch_n += by.shape[0]
        train_loss = np.mean(epoch_loss)
        train_acc = epoch_correct / epoch_n
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        val_logits = model(tf.constant(val_x), training=False)
        val_loss = float(loss_fn(tf.constant(val_y), val_logits))
        val_acc = float(tf.reduce_mean(tf.cast(tf.argmax(val_logits, axis=1) == val_y, tf.float32)))
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        print(f"Epoch {epoch + 1}/{n_epochs} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

    return model, vectorize_layer, history


def predict_priority(bow_model, bow_vec, emb_model, descriptions):
    """
    Predict urgency/priority (1-5) for one or more text descriptions using both models.
    Priority: 1 = most urgent (Red/Immediate), 5 = least urgent (Blue/Non-Urgent).
    descriptions: str or list of str (pet descriptions).
    Returns: (bow_priorities, emb_priorities) as lists of ints 1-5.
    """
    if isinstance(descriptions, str):
        descriptions = [descriptions]
    descriptions = np.array(descriptions)
    # BoW: vectorize then predict
    x_bow = bow_vec(descriptions).numpy()
    bow_logits = bow_model(tf.constant(x_bow.astype("float32")), training=False)
    bow_pred = tf.argmax(bow_logits, axis=1).numpy()
    bow_priorities = (bow_pred + 1).tolist()  # 0-4 -> 1-5
    # Embedding: raw text
    emb_logits = emb_model(tf.constant(descriptions), training=False)
    emb_pred = tf.argmax(emb_logits, axis=1).numpy()
    emb_priorities = (emb_pred + 1).tolist()
    return bow_priorities, emb_priorities


# ==========================================
# 4. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # 1. Load data
    file_path = 'triage_dataset_2500_clean.jsonl'
    texts, labels = load_data(file_path)

    # 2. Split data into Train (80%), Validation (10%), Test (10%)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )

    print(f"Training examples: {len(train_texts)}")
    print(f"Validation examples: {len(val_texts)}")
    print(f"Test examples: {len(test_texts)}\n")

    # 3. Train the Fixed Bag-of-Words Model
    bow_model, bow_vec, bow_hist = build_and_train_bow_model(
        train_texts, train_labels, val_texts, val_labels
    )
    # BoW test accuracy
    x_test_bow = bow_vec(test_texts).numpy()
    bow_test_logits = bow_model(tf.constant(x_test_bow.astype("float32")), training=False)
    bow_test_acc = float(tf.reduce_mean(tf.cast(tf.argmax(bow_test_logits, axis=1) == test_labels, tf.float32)))
    print(f"\nBoW model — Test accuracy: {bow_test_acc:.4f}")

    # 4. Train the Better Embedding Model
    emb_model, emb_vec, emb_hist = build_and_train_embedding_model(
        train_texts, train_labels, val_texts, val_labels
    )
    # Embedding test accuracy
    emb_test_logits = emb_model(tf.constant(test_texts), training=False)
    emb_test_acc = float(tf.reduce_mean(tf.cast(tf.argmax(emb_test_logits, axis=1) == test_labels, tf.float32)))
    print(f"\nEmbedding model — Test accuracy: {emb_test_acc:.4f}")

    # Save embedding model for multimodal fusion (MultimodalFusionLayer.py)
    emb_model.save("urgency_embedding_model.keras")
    print("\nSaved urgency_embedding_model.keras for fusion.")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"BoW model       — Test accuracy: {bow_test_acc:.4f}")
    print(f"Embedding model — Test accuracy: {emb_test_acc:.4f}")

    # 5. Predict priority for new descriptions
    print("\n" + "=" * 60)
    print("NEW INPUT — Predict priority for pet descriptions")
    print("=" * 60)
    print("Priority scale: 1 = most urgent (worst), 5 = least urgent (routine)")
    for p, label in PRIORITY_LABELS.items():
        print(f"  {p}: {label}")
    print()
    new_descriptions = [
        "My dog is eating and playing normally, just here for a checkup.",
        "Cat has been vomiting for two days and not eating.",
        "Dog was hit by a car, bleeding from the leg, having trouble standing.",
    ]
    bow_priors, emb_priors = predict_priority(bow_model, bow_vec, emb_model, new_descriptions)
    for i, desc in enumerate(new_descriptions):
        print(f"\nDescription: \"{desc}\"")
        print(f"  BoW model:       priority {bow_priors[i]} — {PRIORITY_LABELS.get(bow_priors[i], '?')}")
        print(f"  Embedding model: priority {emb_priors[i]} — {PRIORITY_LABELS.get(emb_priors[i], '?')}")

    print("\nAll models trained successfully!")