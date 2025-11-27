import cv2
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model
import matplotlib.pyplot as plt
import subprocess
import os

MODEL_PATH = "../models/final_deepfake_model.h5"
VIDEO_PATH = "../test/trump.mp4"

# -----------------------------
# FFPROBE FOR AUDIO SYNC
# -----------------------------
FFPROBE_PATH = r"C:\Users\suman\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffprobe.exe"


def get_audio_timestamps(video_path):
    try:
        cmd = [
            FFPROBE_PATH,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "frame=pkt_pts_time",
            "-of", "csv=p=0",
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        timestamps = result.stdout.strip().split("\n")
        return [float(t) for t in timestamps if t.strip()]
    except Exception as e:
        print("⚠ Error reading audio:", e)
        return []


def audio_video_sync_score(timestamps):
    if len(timestamps) < 5:
        return -1
    diffs = [abs(timestamps[i] - timestamps[i - 1]) for i in range(1, len(timestamps))]
    avg_gap = sum(diffs) / len(diffs)
    if avg_gap < 0.05:
        return 1
    elif avg_gap < 0.15:
        return 0.5
    else:
        return 0


# -----------------------------
# FACE DETECTOR + PREPROCESS
# -----------------------------
detector = MTCNN()

def detect_face_safe(frame):
    faces = detector.detect_faces(frame)
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]['box']

    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)

    x2 = min(x + w, frame.shape[1])
    y2 = min(y + h, frame.shape[0])

    face = frame[y:y2, x:x2]

    if face.size == 0:
        return None

    face = cv2.resize(face, (224, 224))
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)


# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded!")
print("Model input:", model.input_shape)


# -----------------------------
# PROCESS VIDEO
# -----------------------------
print("Running inference on:", VIDEO_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

frame_scores = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face = detect_face_safe(frame)
    if face is None:
        continue

    pred = model.predict(face, verbose=0)[0][0]
    frame_scores.append(pred)

cap.release()

if len(frame_scores) == 0:
    print("❌ No faces detected!")
    exit()

avg = float(np.mean(frame_scores))
label = "FAKE" if avg > 0.5 else "REAL"

# -----------------------------
# AUDIO–VIDEO SYNC
# -----------------------------
timestamps = get_audio_timestamps(VIDEO_PATH)
sync_score = audio_video_sync_score(timestamps)

# -----------------------------
# RESULT
# -----------------------------
print("\n=============================")
print("FINAL RESULT:")
print({
    "video": VIDEO_PATH,
    "prediction": label,
    "confidence_score": avg,
    "audio_video_sync": sync_score
})
print("=============================")


# -----------------------------
# PLOT GRAPH
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(frame_scores, label="Fake probability")
plt.xlabel("Frame")
plt.ylabel("Prediction (0=Real, 1=Fake)")
plt.title("Frame-wise Prediction Graph")
plt.legend()
plt.tight_layout()
plt.savefig("../output/frame_plot.png")
print("📊 Saved graph → ../output/frame_plot.png")
