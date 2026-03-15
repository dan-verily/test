"""
Multimodal fusion: image (dog breed CNN) + text (urgency embedding model) -> urgency (1-5).

Uses:
  - Image: dog_breed_model.keras (from image_processing.ipynb) as feature extractor
  - Text:  urgency_embedding_model.keras (from embedding.py) as feature extractor

Setup (run once):
  1. python embedding.py          -> saves urgency_embedding_model.keras
  2. Run image_processing.ipynb   -> saves dog_breed_model.keras
  3. python MultimodalFusionLayer.py  -> builds fusion, optionally trains, saves multimodal_fusion_model.keras
"""
import os
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Paths to saved models (relative to this script's directory)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_PATH = os.path.join(_SCRIPT_DIR, "dog_breed_model.keras")
TEXT_PATH = os.path.join(_SCRIPT_DIR, "urgency_embedding_model.keras")
IMG_SIZE = (160, 160)  # must match image_processing.ipynb

# Fusion output: 5 urgency classes (0-4 -> 1-5 in labels)
NUM_URGENCY_CLASSES = 5


def load_image_branch():
    """Load CNN and return a feature extractor (output before last Dense(120))."""
    if not os.path.isfile(CNN_PATH):
        raise FileNotFoundError(
            f"Image model not found: {CNN_PATH}\n"
            "Run image_processing.ipynb and train/save the model first (saves dog_breed_model.keras)."
        )
    cnn = keras.models.load_model(CNN_PATH)
    # Last layer is Dense(120); we want the penultimate output (e.g. Dropout)
    feat = keras.Model(
        inputs=cnn.input,
        outputs=cnn.layers[-2].output,
        name="image_feature_extractor",
    )
    feat.trainable = False
    return feat


def load_text_branch():
    """Load urgency embedding model and return feature extractor (output of Dense(32))."""
    if not os.path.isfile(TEXT_PATH):
        raise FileNotFoundError(
            f"Text model not found: {TEXT_PATH}\n"
            "Run 'python embedding.py' first (from the project root). "
            "It saves urgency_embedding_model.keras after training the embedding model."
        )
    text_model = keras.models.load_model(TEXT_PATH)
    # Keras 3: build input signature by running one forward pass so .inputs is defined
    dummy = tf.constant([["placeholder text"]])
    _ = text_model(dummy)
    inp = text_model.inputs[0] if len(text_model.inputs) == 1 else text_model.inputs
    # Structure: ... -> Dense(32) -> Dropout -> Dense(5). We want Dense(32) output.
    feat = keras.Model(
        inputs=inp,
        outputs=text_model.layers[-3].output,
        name="text_feature_extractor",
    )
    feat.trainable = False
    return feat


def build_fusion_model(image_feat, text_feat, fusion_dim=256):
    """
    Build fusion model: (image, text) -> urgency logits (5 classes).
    image_feat and text_feat are frozen feature extractors.
    """
    image_in = keras.Input(shape=IMG_SIZE + (3,), name="image", dtype=tf.float32)
    text_in = keras.Input(shape=(1,), dtype=tf.string, name="text")

    # Image branch: features -> project to fusion_dim
    img_vec = image_feat(image_in)
    img_vec = layers.Dense(fusion_dim, activation="relu", name="img_proj")(img_vec)

    # Text branch: features -> project to fusion_dim
    txt_vec = text_feat(text_in)
    txt_vec = layers.Dense(fusion_dim, activation="relu", name="txt_proj")(txt_vec)

    # Fusion
    x = layers.Concatenate(name="concat")([img_vec, txt_vec])
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(fusion_dim, activation="relu", name="fusion_dense")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(NUM_URGENCY_CLASSES, activation="softmax", name="urgency")(x)

    model = keras.Model(
        inputs={"image": image_in, "text": text_in},
        outputs=out,
        name="multimodal_fusion",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_paired_dataset_from_triage(
    triage_jsonl_path,
    image_dir,
    batch_size=32,
    image_size=IMG_SIZE,
    seed=42,
):
    """
    Build a tf.data.Dataset for fusion training when you have only triage (text) labels.
    Pairs each text with a random image from image_dir (same image repeated for whole batch
    if you have no real paired data). Use only for demo; for real training use (image, text, urgency) triples.
    image_dir: path to a folder of images (e.g. dog_breed/train/Beagle) or a flat folder of .jpg.
    """
    import json
    from pathlib import Path

    texts, labels = [], []
    with open(triage_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            texts.append(d["text"])
            labels.append(d["urgency_level"] - 1)  # 1-5 -> 0-4

    texts = np.array(texts)
    labels = np.array(labels, dtype=np.int32)
    image_dir = Path(image_dir)
    image_paths = list(image_dir.rglob("*.jpg")) + list(image_dir.rglob("*.jpeg")) + list(image_dir.rglob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {image_dir}")

    def gen():
        rng = np.random.default_rng(seed)
        n = len(texts)
        for i in range(n):
            img_path = rng.choice(image_paths)
            yield str(img_path), texts[i], labels[i]

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    def load_and_preprocess(img_path, text, label):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, image_size)
        # Keep [0, 255] float; saved CNN has EfficientNet preprocess_input inside
        img = tf.cast(img, tf.float32)
        text = tf.reshape(text, (1,))
        return {"image": img, "text": text}, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1024, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def load_paired_batches_into_memory(
    triage_jsonl_path,
    image_dir,
    num_samples=640,
    batch_size=32,
    image_size=IMG_SIZE,
    seed=42,
):
    """
    Load a fixed number of (image, text, label) samples into memory as numpy/tensors.
    Avoids tf.data during training so model.fit() / training loop doesn't hang on Mac.
    """
    import json
    from pathlib import Path

    texts, labels = [], []
    with open(triage_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            texts.append(d["text"])
            labels.append(d["urgency_level"] - 1)
    texts = np.array(texts)
    labels = np.array(labels, dtype=np.int32)
    image_dir = Path(image_dir)
    image_paths = list(image_dir.rglob("*.jpg")) + list(image_dir.rglob("*.jpeg")) + list(image_dir.rglob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {image_dir}")

    rng = np.random.default_rng(seed)
    n = min(num_samples, len(texts))
    idx = rng.permutation(len(texts))[:n]
    texts = texts[idx]
    labels = labels[idx]

    images_list = []
    for _ in range(n):
        img_path = rng.choice(image_paths)
        img = tf.io.read_file(str(img_path))
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32)
        images_list.append(img)
    images = tf.stack(images_list)
    texts_t = tf.constant(np.reshape(texts, (-1, 1)))
    labels_t = tf.constant(labels)
    return images, texts_t, labels_t, n


def predict_urgency(fusion_model, images, texts):
    """
    Predict urgency (0-4) for batches of (image, text).
    images: float32 tensor (batch, 160, 160, 3) in [0,1] or preprocessed for EfficientNet
    texts:  tensor of shape (batch,) or (batch, 1) of tf.string
    Returns: (probs (batch, 5), predicted_class (batch,) as 0-4)
    """
    if texts.shape.ndims == 1:
        texts = tf.reshape(texts, (-1, 1))
    probs = fusion_model({"image": images, "text": texts}, training=False)
    pred = tf.argmax(probs, axis=1)
    return probs, pred


# ---------------------------------------------------------------------------
# Main: build fusion model and optionally train / run inference
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)

    print("Loading image branch (dog breed CNN)...")
    image_feat = load_image_branch()
    print("Loading text branch (urgency embedding model)...")
    text_feat = load_text_branch()

    print("Building fusion model...")
    fusion_model = build_fusion_model(image_feat, text_feat)
    fusion_model.summary()

    # Optional: train if you have paired data (in-memory loop to avoid tf.data hang on Mac)
    triage_path = os.path.join(_SCRIPT_DIR, "triage_dataset_2500_clean.jsonl")
    image_dir = os.path.join(_SCRIPT_DIR, "dog_breed", "train")
    if os.path.exists(triage_path) and os.path.exists(image_dir):
        print("\nLoading paired data into memory (avoids tf.data hang)...")
        batch_size = 32
        num_samples = 640  # 20 batches
        images, texts_t, labels_t, n = load_paired_batches_into_memory(
            triage_path, image_dir, num_samples=num_samples, batch_size=batch_size, seed=42
        )
        print(f"Loaded {n} samples. Training fusion layers (2 epochs)...")
        optimizer = keras.optimizers.Adam(1e-3)
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        n_epochs = 2
        for epoch in range(n_epochs):
            perm = tf.random.shuffle(tf.range(n))
            images_e = tf.gather(images, perm)
            texts_e = tf.gather(texts_t, perm)
            labels_e = tf.gather(labels_t, perm)
            epoch_loss, epoch_correct, epoch_n = [], 0, 0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                bx = images_e[start:end]
                bt = texts_e[start:end]
                by = labels_e[start:end]
                with tf.GradientTape() as tape:
                    logits = fusion_model({"image": bx, "text": bt}, training=True)
                    loss = loss_fn(by, logits)
                grads = tape.gradient(loss, fusion_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, fusion_model.trainable_variables))
                epoch_loss.append(float(loss))
                preds = tf.cast(tf.argmax(logits, axis=1), by.dtype)
                epoch_correct += int(tf.reduce_sum(tf.cast(preds == by, tf.int32)))
                epoch_n += by.shape[0]
            train_acc = epoch_correct / epoch_n
            print(f"  Epoch {epoch + 1}/{n_epochs} - loss: {np.mean(epoch_loss):.4f} - accuracy: {train_acc:.4f}")
        fusion_model.save(os.path.join(_SCRIPT_DIR, "multimodal_fusion_model.keras"))
        print("Saved multimodal_fusion_model.keras")
    else:
        print("\nSkipping training (need triage_dataset_2500_clean.jsonl and dog_breed/train).")
        print("You can still use build_fusion_model() and predict_urgency() with the unfused branches.")

    # Example: predict on one (image, text) pair
    print("\nExample: single prediction (requires one image and one description)...")
    sample_text = tf.constant(["My dog is lethargic and not eating for two days."])
    # Dummy image (replace with real image tensor for real inference)
    sample_img = tf.zeros((1,) + IMG_SIZE + (3,), dtype=tf.float32)
    probs, pred = predict_urgency(fusion_model, sample_img, sample_text)
    print(f"Urgency class (0-4): {pred.numpy()[0]}, Priority 1-5: {pred.numpy()[0] + 1}")
