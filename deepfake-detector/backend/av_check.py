"""
av_check.py
Basic Audio/Video consistency checks for Deepfake assistance.
This does NOT replace ML model but improves detection accuracy when combined.
"""

import cv2
import numpy as np
import librosa
import json

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Detect sudden frame jumps (common in deepfakes)
    inconsistencies = 0
    ret, prev_frame = cap.read()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        diff = np.abs(frame.astype(int) - prev_frame.astype(int)).mean()
        if diff < 2:  # almost static frames repeated unnaturally
            inconsistencies += 1
        prev_frame = frame
    cap.release()

    return {
        "fps": fps,
        "frames": frames,
        "inconsistencies": inconsistencies,
        "flag": "FAKE" if inconsistencies > 10 else "REAL"
    }


def analyze_audio(video_path):
    try:
        audio, sr = librosa.load(video_path, sr=None)
        # detect unnatural digital noise → deepfake blending artifacts
        noise = np.mean(np.abs(np.diff(audio)))
        return {
            "sample_rate": sr,
            "noise_level": float(noise),
            "flag": "FAKE" if noise < 0.002 else "REAL"
        }
    except Exception:
        return {"error": "No audio track"}


def av_analyze(video_path):
    v = analyze_video(video_path)
    a = analyze_audio(video_path)

    # Combine
    if ("flag" in v and v["flag"] == "FAKE") or ("flag" in a and a["flag"] == "FAKE"):
        result = "FAKE"
    else:
        result = "REAL"

    return {
        "video_analysis": v,
        "audio_analysis": a,
        "final_flag": result
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No video path provided"}))
        sys.exit(1)

    path = sys.argv[1]
    print(json.dumps(av_analyze(path)))
