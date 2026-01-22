import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

# =========================
# Pfade
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "images"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# =========================
# Parameter
# =========================
IMG_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 25
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42

# =========================
# Datasets laden
# =========================
def load_datasets(): 
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print("Klassen:", class_names)

    normalization = layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds   = val_ds.map(lambda x, y: (normalization(x), y))

    train_ds = train_ds.shuffle(1000).cache().prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names

# =========================
# Modell (Sketch-stabil)
# =========================
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),

        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),

        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# =========================
# Main
# =========================
def main():
    train_ds, val_ds, class_names = load_datasets()

    model = build_model(
        input_shape=(IMG_SIZE, IMG_SIZE, 1),
        num_classes=len(class_names)
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODELS_DIR / "quickdraw_cnn.keras",
            save_best_only=True
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Klassen speichern
    with open(MODELS_DIR / "class_indices.json", "w", encoding="utf-8") as f:
        json.dump(
            {i: name for i, name in enumerate(class_names)},
            f,
            indent=2,
            ensure_ascii=False
        )

    print("✅ Modell & Klassen gespeichert")

# =========================
if __name__ == "__main__":
    main()
