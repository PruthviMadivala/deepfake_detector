import cv2
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import os

VIDEO_PATH = "../test/trump.mp4"
MODEL_PATH = "../models/final_deepfake_model.h5"

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded!")

detector = MTCNN()

IMG_SIZE = 224

def detect_face(frame):
    faces = detector.detect_faces(frame)
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]['box']
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)
    face = frame[y:y+h, x:x+w]

    if face.size == 0:
        return None

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face, (x, y, w, h)

def compute_gradcam(img_array):
    last_conv_layer = model.get_layer("Conv_1")

    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_score = preds[:, pred_index]

    grads = tape.gradient(class_score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]

    heatmap = np.zeros(shape=conv_out.shape[0:2], dtype=np.float32)

    for i in range(pooled_grads.shape[0]):
        heatmap += pooled_grads[i] * conv_out[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

def save_graph(scores):
    plt.figure(figsize=(10,4))
    plt.plot(scores, label="Fake Probability")
    plt.xlabel("Frame")
    plt.ylabel("Probability")
    plt.title("Frame-wise Fake Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("frame_prob_graph.png")
    plt.close()

print("Running inference on:", VIDEO_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

predictions = []
first_face_for_gradcam = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_data = detect_face(frame)
    if face_data is None:
        continue

    face, box = face_data

    if first_face_for_gradcam is None:
        first_face_for_gradcam = face

    pred = model.predict(face, verbose=0)[0][0]
    predictions.append(pred)

cap.release()

if len(predictions) == 0:
    print("No face detected in the video!")
    exit()

avg = float(np.mean(predictions))
label = "FAKE" if avg > 0.5 else "REAL"

print("\n=======================")
print("FINAL RESULT:")
print({
    "prediction": label,
    "confidence": avg
})
print("=======================\n")

# Save frame graph
save_graph(predictions)

# Save Grad-CAM
if first_face_for_gradcam is not None:
    heatmap = compute_gradcam(first_face_for_gradcam)

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    orig = first_face_for_gradcam[0] * 255
    orig = orig.astype("uint8")

    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    cv2.imwrite("gradcam_heatmap.jpg", heatmap)
    cv2.imwrite("gradcam_overlay.jpg", overlay)

print("Graph + GradCAM saved successfully!")
