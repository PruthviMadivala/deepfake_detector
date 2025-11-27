"""
realtime_webcam_enhanced.py
Adds:
  - bounding box
  - blink detection (heuristic using MTCNN keypoints)
  - live percent (fake-prob) graph overlay
  - voice alert when strong FAKE detected

Run from scripts/:
    python realtime_webcam_enhanced.py
"""

import cv2
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model
from collections import deque
import threading
import time
import sys

# ---- CONFIG ----
MODEL_PATH = "../models/final_deepfake_model.h5"   # keep same model
CAM_INDEX = 0             # webcam index
GRAPH_HISTORY = 120       # number of recent frames to show in graph
FAKE_ALERT_THRESHOLD = 0.85  # confidence threshold to trigger voice alert
BLINK_DROP_FACTOR = 0.7   # eye intensity drop factor to register blink
BLINK_WINDOW = 3          # frames within which drop+recover counts as blink
BLINK_MONITOR_SECONDS = 20  # monitor window for blink-rate anomaly
BLINK_MIN_EXPECTED = 2     # if fewer than this in monitor window -> suspicious
VOICE_ENABLED = True
# ------------------

# voice alert (non-blocking)
def alert_beep_thread():
    try:
        import winsound
        def beep_sequence():
            for f,d in [(1000,120),(700,120),(900,120)]:
                winsound.Beep(f, d)
                time.sleep(0.05)
        threading.Thread(target=beep_sequence, daemon=True).start()
    except Exception:
        # fallback: just print (or you can expand to cross-platform libs)
        print("[ALERT] FAKE detected (no sound available)")

# Load model
print("Loading model...")
model = load_model(MODEL_PATH)
input_shape = model.input_shape  # (None, h, w, c)
if input_shape is None or len(input_shape) < 4:
    raise RuntimeError("Unexpected model input shape.")
MODEL_H, MODEL_W = input_shape[1], input_shape[2]
print(f"Model loaded. Expecting input: {(MODEL_H, MODEL_W)}")

# MTCNN face detector (gives keypoints)
detector = MTCNN()

# helper: safe face extraction (returns face array ready for model or None)
def safe_face_extract(frame):
    try:
        results = detector.detect_faces(frame)
    except Exception:
        return None, None  # no box, no keypoints

    if not results:
        return None, None

    r = results[0]
    x, y, w, h = r.get("box", (0,0,0,0))
    x, y = max(0, x), max(0, y)
    w, h = max(1, w), max(1, h)
    x2 = min(x + w, frame.shape[1])
    y2 = min(y + h, frame.shape[0])

    face = frame[y:y2, x:x2]
    if face.size == 0:
        return None, None

    # validate size
    if face.shape[0] < 30 or face.shape[1] < 30:
        return None, None

    face_resized = cv2.resize(face, (MODEL_W, MODEL_H))
    face_resized = face_resized.astype("float32") / 255.0
    face_resized = np.expand_dims(face_resized, axis=0)
    return face_resized, r  # detector result includes 'keypoints' & 'box'

# Setup camera
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    sys.exit(1)
print("Webcam opened. Press Q to quit.")

# Data containers
scores = deque(maxlen=GRAPH_HISTORY)   # recent model scores (0..1)
times = deque(maxlen=GRAPH_HISTORY)    # timestamp per score
blink_timestamps = deque()             # timestamps of detected blinks
eye_baseline = None                     # running baseline brightness for eyes
eye_state = {"left": {"down_frames":0, "last_down":None},
             "right":{"down_frames":0, "last_down":None}}

last_alert_time = 0
ALERT_COOLDOWN = 6.0  # seconds between successive alerts

# drawing helpers
def draw_graph_overlay(frame, scores_deque):
    h, w = frame.shape[:2]
    graph_h = int(h * 0.25)
    graph_w = int(w * 0.4)
    graph_x = w - graph_w - 10
    graph_y = 10

    # background
    overlay = frame.copy()
    cv2.rectangle(overlay, (graph_x, graph_y), (graph_x+graph_w, graph_y+graph_h), (30,30,30), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

    # draw axes
    cv2.rectangle(frame, (graph_x, graph_y), (graph_x+graph_w, graph_y+graph_h), (200,200,200), 1)
    cv2.putText(frame, "Fake % (recent)", (graph_x+6, graph_y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)

    if len(scores_deque) < 2:
        return frame

    arr = np.array(scores_deque)
    # normalize to graph height
    xs = np.linspace(graph_x+5, graph_x+graph_w-5, num=len(arr))
    ys = graph_y + graph_h - 5 - (arr * (graph_h-10)).astype(int)

    # draw polyline
    pts = np.vstack((xs.astype(np.int32), ys.astype(np.int32))).T.reshape(-1,1,2)
    cv2.polylines(frame, [pts], False, (0,180,255), 2)

    # latest value
    latest = arr[-1]
    cv2.putText(frame, f"{latest*100:.1f}%", (graph_x+6, graph_y+graph_h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return frame

def eye_patch_mean_gray(frame, point, radius=14):
    x,y = int(point[0]), int(point[1])
    h,w = frame.shape[:2]
    x1 = max(0, x-radius); x2 = min(w, x+radius)
    y1 = max(0, y-radius); y2 = min(h, y+radius)
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_input, det = safe_face_extract(frame)
    label = "NO FACE"
    score = 0.0

    if face_input is not None:
        # prediction
        pred = model.predict(face_input, verbose=0)[0][0]
        score = float(pred)
        scores.append(score)
        times.append(time.time())
        label = "FAKE" if score > 0.5 else "REAL"

        # bounding box & keypoints
        box = det.get("box", None)
        keypoints = det.get("keypoints", {})
        if box:
            bx, by, bw, bh = box
            # clamp and convert to int
            bx, by = max(0,int(bx)), max(0,int(by))
            bw, bh = max(1,int(bw)), max(1,int(bh))
            cv2.rectangle(frame, (bx,by), (bx+bw, by+bh), (0,0,255) if label=="FAKE" else (0,255,0), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (bx, max(15, by-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) if label=="FAKE" else (0,255,0), 2)

        # blink detection (heuristic using eye patch brightness)
        if "left_eye" in keypoints and "right_eye" in keypoints:
            left_pt = keypoints["left_eye"]
            right_pt = keypoints["right_eye"]
            left_mean = eye_patch_mean_gray(frame, left_pt)
            right_mean = eye_patch_mean_gray(frame, right_pt)

            # initialize baseline
            if eye_baseline is None and left_mean is not None and right_mean is not None:
                eye_baseline = (left_mean + right_mean) / 2.0

            if eye_baseline is not None and left_mean is not None and right_mean is not None:
                avg_eye = (left_mean + right_mean) / 2.0

                # detect "down" (possible blink) when current mean drops relative to baseline
                if avg_eye < eye_baseline * BLINK_DROP_FACTOR:
                    # mark down frame for both eyes
                    # record last_down time
                    now = time.time()
                    # avoid repeated down marks every frame: use last_down cooldown
                    if eye_state["left"]["last_down"] is None or now - eye_state["left"]["last_down"] > 0.25:
                        eye_state["left"]["last_down"] = now
                    # same for right (we keep symmetric)
                    if eye_state["right"]["last_down"] is None or now - eye_state["right"]["last_down"] > 0.25:
                        eye_state["right"]["last_down"] = now
                else:
                    # recovered - if we had a recent down, count as blink
                    now = time.time()
                    for side in ("left","right"):
                        last_down = eye_state[side]["last_down"]
                        if last_down is not None and 0 < now - last_down <= BLINK_WINDOW:
                            # count blink and clear last_down
                            blink_timestamps.append(now)
                            eye_state[side]["last_down"] = None

                    # update running baseline slowly (EMA)
                    eye_baseline = (0.97 * eye_baseline) + (0.03 * avg_eye)

            # draw small circles for eyes
            try:
                lx,ly = map(int, left_pt); rx,ry = map(int, right_pt)
                cv2.circle(frame, (lx,ly), 3, (255,255,0), -1)
                cv2.circle(frame, (rx,ry), 3, (255,255,0), -1)
            except Exception:
                pass

    else:
        # no face: still maintain very small smoothing baseline decay
        if eye_baseline is not None:
            eye_baseline *= 0.999

    # analyze blink rate over monitor window
    nowt = time.time()
    # prune old blinks
    while blink_timestamps and nowt - blink_timestamps[0] > BLINK_MONITOR_SECONDS:
        blink_timestamps.popleft()
    blink_count = len(blink_timestamps)
    blink_warning = blink_count < BLINK_MIN_EXPECTED

    # draw blink info
    cv2.putText(frame, f"Blinks({BLINK_MONITOR_SECONDS}s): {blink_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2)
    if blink_warning:
        cv2.putText(frame, "Blink anomaly: SUSPICIOUS", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # draw live graph overlay
    draw_graph_overlay(frame, scores)

    # if strong fake -> alert (cooldown)
    if scores:
        latest = scores[-1]
        if latest >= FAKE_ALERT_THRESHOLD and time.time() - last_alert_time > ALERT_COOLDOWN:
            last_alert_time = time.time()
            if VOICE_ENABLED:
                alert_beep_thread()

    # show frame
    cv2.imshow("Deepfake Real-time (enhanced)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
