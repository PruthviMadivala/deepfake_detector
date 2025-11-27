import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

REAL_DIR = "../data/frames_faces/real"
FAKE_DIR = "../data/frames_faces/balanced_fake"
MODEL_PATH = "../models/final_deepfake_model.h5"

print("Counting images...")
num_real = len(os.listdir(REAL_DIR))
num_fake = len(os.listdir(FAKE_DIR))
print(f"Real: {num_real}, Fake: {num_fake}")

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

# Data generator
datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    "../data/frames_faces",
    classes=["real", "balanced_fake"],
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    "../data/frames_faces",
    classes=["real", "balanced_fake"],
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# Model
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])

print("\n🚀 Training started...\n")
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

print("\n🎉 Training complete!")

os.makedirs("../models", exist_ok=True)
model.save(MODEL_PATH)
print(f"\n💾 Model saved at: {MODEL_PATH}")
