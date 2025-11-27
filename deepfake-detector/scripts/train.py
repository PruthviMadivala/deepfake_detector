import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

REAL_FRAMES = "../data/frames/real"
FAKE_FRAMES = "../data/frames/fake"

DATASET_DIR = "../data/frames/"

# Image size
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

# -----------------------------
# Data Augmentation
# -----------------------------

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# -----------------------------
# Build model — MobileNetV2
# -----------------------------

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # freeze layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
preds = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=preds)

model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)

# -----------------------------
# Train model
# -----------------------------

print("\n🚀 Training started...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

print("\n🎉 Training complete!")

# -----------------------------
# Save model
# -----------------------------

model.save("../models/deepfake_detector.h5")

print("\n💾 Saved model to: ../models/deepfake_detector.h5")
