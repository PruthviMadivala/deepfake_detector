import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

DATASET_DIR = "../data/frames_faces/"   # << we will create this
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25

# ------------------------------
# Step 1: Build Xception model
# ------------------------------

base_model = Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # freeze for initial training

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\n🔵 Starting training (Phase 1: frozen layers)...\n")

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

history = model.fit(train, validation_data=val, epochs=EPOCHS)

# ------------------------------
# Step 2: Fine-tune Xception
# ------------------------------

print("\n🟢 Fine-tuning model...\n")

base_model.trainable = True

for layer in base_model.layers[:80]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(train, validation_data=val, epochs=10)

os.makedirs("../models", exist_ok=True)
model.save("../models/deepfake_detector_xception.h5")

print("\n✅ Saved strong model to deepfake_detector_xception.h5")
