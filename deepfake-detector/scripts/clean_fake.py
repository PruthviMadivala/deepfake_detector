import os
import shutil

FAKE_DIR = "../data/frames_faces/fake"
OUTPUT_DIR = "../data/frames_faces/fake_cleaned"

os.makedirs(OUTPUT_DIR, exist_ok=True)

count = 0

# Walk through ALL fake subfolders
for root, dirs, files in os.walk(FAKE_DIR):
    for file in files:
        if file.lower().endswith(".jpg"):
            src = os.path.join(root, file)
            dst = os.path.join(OUTPUT_DIR, f"{count}.jpg")
            shutil.copy(src, dst)
            count += 1

print(f"✔ Merged all fake images into fake_cleaned! Total images: {count}")
