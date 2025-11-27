import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

MODEL_PATH = "../models/deepfake_face_model.h5"
VIDEO_PATH = "../data/videos/real/v1.mp4"   # change this to test fake also

IMG_SIZE = 224

print(f"📌 Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

detector = MTCNN()

def extract_face(frame):
    faces = detector.detect_faces(frame)
    if len(faces) == 0:
        return None
    
    x, y, w, h = faces[0]['box']
    x, y = max(0, x), max(0, y)
    face = frame[y:y+h, x:x+w]
    
    if face is None or face.size == 0:
        return None
    
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    return face

def predict_video(video_path):
    print(f"\n🎬 Running inference on: {video_path}")

    cap = cv2.VideoCapture(video_path)

    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        face = extract_face(frame)
        if face is None:
            continue
        
        face = np.expand_dims(face, axis=0)
        pred = model.predict(face, verbose=0)[0][0]
        predictions.append(pred)

    cap.release()

    if len(predictions) == 0:
        print("⚠️ No faces detected in the video!")
        return None
    
    avg = np.mean(predictions)
    label = "FAKE" if avg > 0.5 else "REAL"

    return {
        "video": video_path,
        "prediction": label,
        "confidence_score": float(avg)
    }

# Run inference
result = predict_video(VIDEO_PATH)

print("\n==============================")
print("FINAL RESULT:")
print(result)
print("==============================")
