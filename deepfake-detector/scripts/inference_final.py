import tensorflow as tf
import numpy as np
import cv2
import os

MODEL_PATH = "../models/final_model.h5"

def predict_image(img_path):
    model = tf.keras.models.load_model(MODEL_PATH)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        label = "FAKE"
        confidence = prediction
    else:
        label = "REAL"
        confidence = 1 - prediction

    return label, float(confidence)


if __name__ == "__main__":
    test_image = "../test/sample.jpg"   # change path to any image

    if not os.path.exists(test_image):
        print("❌ Test image not found!")
    else:
        label, conf = predict_image(test_image)
        print(f"\nRESULT: {label} ({conf:.4f})\n")
