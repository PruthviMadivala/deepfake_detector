import cv2
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model
import os

MODEL_PATH = "../models/final_deepfake_model.h5"

# Load model
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.")

detector = MTCNN()

# ----------------------------------------------------------
# FACE PREPROCESSING (FIXED)
# ----------------------------------------------------------
def preprocess_face(frame):
    results = detector.detect_faces(frame)
    if len(results) == 0:
        return None, None

    x, y, w, h = results[0]["box"]

    # sanitize box
    x = max(0, x)
    y = max(0, y)
    w = max(10, w)
    h = max(10, h)

    x2 = min(x + w, frame.shape[1])
    y2 = min(y + h, frame.shape[0])

    face = frame[y:y2, x:x2]

    # avoid zero-size faces
    if face is None or face.size == 0:
        return None, None

    if face.shape[0] < 20 or face.shape[1] < 20:
        return None, None

    face = cv2.resize(face, (224, 224))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)

    return face, (x, y, w, h)

# ----------------------------------------------------------
# PREDICT SINGLE FACE
# ----------------------------------------------------------
def predict_face(face):
    if face is None:
        return None
    pred = model.predict(face, verbose=0)[0][0]
    return float(pred)

# ----------------------------------------------------------
# VIDEO FILE MODE
# ----------------------------------------------------------
def run_video_inference(path):
    print("Running inference on:", path)

    cap = cv2.VideoCapture(path)
    preds = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face, _ = preprocess_face(frame)
        if face is None:
            continue

        score = predict_face(face)
        if score is not None:
            preds.append(score)

    cap.release()

    if len(preds) == 0:
        print("No valid faces detected.")
        return

    avg = float(np.mean(preds))
    label = "FAKE" if avg > 0.5 else "REAL"

    print("Final Result:")
    print({
        "video": path,
        "prediction": label,
        "confidence_score": avg
    })

# ----------------------------------------------------------
# WEBCAM REALTIME MODE
# ----------------------------------------------------------
def run_webcam():
    cap = cv2.VideoCapture(0)
    print("Webcam started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face, box = preprocess_face(frame)

        if face is not None and box is not None:
            score = predict_face(face)
            label = "FAKE" if score > 0.5 else "REAL"

            x, y, w, h = box
            color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {score:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Webcam Deepfake Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------------------------------------
# MENU
# ----------------------------------------------------------
print("===============================")
print("  Deepfake Detector Menu")
print("===============================")
print("1 -> Realtime Webcam Detection")
print("2 -> Detect Deepfake from Video File")
choice = input("Enter your choice (1/2): ")

if choice == "1":
    run_webcam()
elif choice == "2":
    path = input("Enter video path (default ../test/trump.mp4): ").strip()
    if path == "":
        path = "../test/trump.mp4"
    run_video_inference(path)
else:
    print("Invalid choice.")
