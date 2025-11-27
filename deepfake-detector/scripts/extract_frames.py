import cv2
import os

VIDEOS_DIR = "../data/videos"
OUTPUT_DIR = "../data/frames"

def extract(video_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_path, f"{i}.jpg")
        cv2.imwrite(frame_path, frame)
        i += 1

    cap.release()
    print(f"Saved {i} frames → {output_path}")


for label in ["real", "fake"]:
    in_dir = os.path.join(VIDEOS_DIR, label)
    out_dir = os.path.join(OUTPUT_DIR, label)

    videos = os.listdir(in_dir)
    print(f"\nProcessing {label} videos: {videos}")

    for v in videos:
        video_path = os.path.join(in_dir, v)
        folder_name = v.split(".")[0]
        output_path = os.path.join(out_dir, folder_name)

        extract(video_path, output_path)
