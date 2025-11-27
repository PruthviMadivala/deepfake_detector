import sys, json
from inference_video import analyze_frame    # your model function

frame_path = sys.argv[1]

label, confidence = analyze_frame(frame_path)

print(json.dumps({
    "label": label,
    "confidence": confidence
}))
