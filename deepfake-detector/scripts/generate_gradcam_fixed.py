import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import os

MODEL_PATH = "../models/final_deepfake_model.h5"
VIDEO_PATH = "../test/trump.mp4"

OUTPUT_FACE = "gradcam_face.jpg"
OUTPUT_HEATMAP = "gradcam_heatmap.jpg"
OUTPUT_OVERLAY = "gradcam_overlay.jpg"

print("Loading model:", MODEL_PATH)
model = load_model(MODEL_PATH)
print("Model loaded.")

# ========= Find last conv layer ==========
last_conv_name = None
for layer in reversed(model.layers):
    if "conv" in layer.name.lower():
        last_conv_name = layer.name
        break

if last_conv_name is None:
    raise Exception("No conv layer found!")

print("Using last conv layer:", last_conv_name)

# ========= Extract full-resolution face ==========
def extract_face():
    cap = cv2.VideoCapture(VIDEO_PATH)
    face = None
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            break

    cap.release()
    return face

face = extract_face()

if face is None:
    raise Exception("No face detected in video!")

cv2.imwrite(OUTPUT_FACE, face)
print("Extracted high-res face:", OUTPUT_FACE)

# Resize for model
input_face = cv2.resize(face, (224, 224))
img = input_face.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

# ========= GRAD-CAM ==========
grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(last_conv_name).output, model.output]
)

with tf.GradientTape() as tape:
    conv_out, preds = grad_model(img)
    loss = preds[:, 0]

grads = tape.gradient(loss, conv_out)[0]
weights = tf.reduce_mean(grads, axis=(0, 1))
cam = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1)

cam = np.maximum(cam, 0)
cam = cam / np.max(cam)

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.resize(heatmap, (face.shape[1], face.shape[0]))

overlay = cv2.addWeighted(face, 0.6, heatmap, 0.6, 0)

cv2.imwrite(OUTPUT_HEATMAP, heatmap)
cv2.imwrite(OUTPUT_OVERLAY, overlay)

print("Saved heatmap:", OUTPUT_HEATMAP)
print("Saved overlay:", OUTPUT_OVERLAY)
print("DONE.")
