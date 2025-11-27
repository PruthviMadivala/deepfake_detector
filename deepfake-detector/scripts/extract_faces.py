import os
import cv2
from mtcnn import MTCNN

INPUT_DIR = "../data/frames"
OUTPUT_DIR = "../data/frames_faces"

os.makedirs(os.path.join(OUTPUT_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "fake"), exist_ok=True)

detector = MTCNN()

# -------------------------------
# FACE EXTRACTION FUNCTION
# -------------------------------
def extract_face(image):
    try:
        results = detector.detect_faces(image)
    except:
        return None

    if results is None or len(results) == 0:
        return None

    try:
        x, y, w, h = results[0]["box"]
    except:
        return None

    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    if w <= 0 or h <= 0:
        return None

    face = image[y:y+h, x:x+w]

    if face is None or face.size == 0:
        return None

    return face


# -------------------------------
# PROCESS FUNCTION TO HANDLE FOLDERS
# -------------------------------
def process(label):
    in_path = os.path.join(INPUT_DIR, label)
    out_path = os.path.join(OUTPUT_DIR, label)

    # all video folders
    video_folders = [
        f for f in os.listdir(in_path)
        if os.path.isdir(os.path.join(in_path, f))
    ]

    print(f"\n🔵 Processing {label} ({len(video_folders)} video folders)...")

    saved = 0

    for folder in video_folders:
        folder_path = os.path.join(in_path, folder)
        images = [
            img for img in os.listdir(folder_path)
            if img.lower().endswith((".jpg", ".png"))
        ]

        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            face = extract_face(img)
            if face is None:
                continue

            face = cv2.resize(face, (224, 224))

            save_name = f"{folder}_{img_name}"
            cv2.imwrite(os.path.join(out_path, save_name), face)
            saved += 1

    print(f"✅ Saved {saved} faces for {label}\n")


# RUN BOTH REAL & FAKE
process("real")
process("fake")

print("🎉 Face extraction completed!")
