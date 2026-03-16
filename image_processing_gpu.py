import pathlib
import numpy as np
import keras
from keras import layers
from keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report

FINE_TUNE = True

# ---- Paths ----
new_base_dir = pathlib.Path("dog_breed")

# ---- Load datasets ----
batch_size = 128
image_size = (224, 224)

train_dataset = image_dataset_from_directory(
    new_base_dir / "train", image_size=image_size, batch_size=batch_size
)
validation_dataset = image_dataset_from_directory(
    new_base_dir / "validation", image_size=image_size, batch_size=batch_size
)
test_dataset = image_dataset_from_directory(
    new_base_dir / "test", image_size=image_size, batch_size=batch_size, shuffle=False
)

# ---- Build model ----
num_classes = 120
img_size = (224, 224)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
])

base = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=img_size + (3,)
)
base.trainable = False

inputs = keras.Input(shape=img_size + (3,))
x = data_augmentation(inputs)
x = keras.applications.efficientnet.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary(line_length=80)

# ---- Train (full train set + early stopping) ----
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="dog_breed_model_gpu_fine_tune_224.keras",
        save_best_only=True,
        monitor="val_loss",
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    ),
]
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=callbacks,
)

# ---- Fine-tune: unfreeze top layers of base, train at lower LR ----
if FINE_TUNE:
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nFine-tuning top 20 layers of EfficientNetB0...")
    history_ft = model.fit(
        train_dataset,
        epochs=20,
        validation_data=validation_dataset,
        callbacks=callbacks,
    )

# ---- Evaluate (in-memory model — reload is broken for preprocess_input) ----
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")

y_true = np.concatenate([y.numpy() for _, y in test_dataset], axis=0)
y_prob = model.predict(test_dataset, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

print(classification_report(y_true, y_pred, target_names=test_dataset.class_names, digits=3))
