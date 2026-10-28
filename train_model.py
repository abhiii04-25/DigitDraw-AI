"""
Train a CNN on MNIST and save it as digit_cnn.keras.

Conventions (these match app.py):
  - input shape: (28, 28, 1)
  - pixel values scaled to 0..1  (x / 255.0)
  - white digit on black background
  - output: softmax over 10 classes
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# ---------- DATA ----------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = (x_train.astype("float32") / 255.0).reshape(-1, 28, 28, 1)
x_test = (x_test.astype("float32") / 255.0).reshape(-1, 28, 28, 1)

# ---------- MODEL ----------
# The augmentation layers only run during training, so they do not
# affect predictions in the app.
model = keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),

        layers.RandomTranslation(0.1, 0.1),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ---------- TRAIN ----------
callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=2, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    ),
]

model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=25,
    batch_size=128,
    callbacks=callbacks,
)

# ---------- EVALUATE + SAVE ----------
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc * 100:.2f}%")

pred = model.predict(x_test[:5], verbose=0)
print("Probabilities sum to 1:", np.allclose(pred.sum(axis=1), 1.0, atol=1e-3))

model.save("digit_cnn.keras")
print("Saved digit_cnn.keras")