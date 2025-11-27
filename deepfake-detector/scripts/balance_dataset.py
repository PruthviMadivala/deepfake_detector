import os
import shutil
import random

REAL_DIR = "../data/frames_faces/real"
FAKE_DIR = "../data/frames_faces/fake_cleaned"
OUT_DIR = "../data/frames_faces/balanced_fake"

os.makedirs(OUT_DIR, exist_ok=True)

real_count = len(os.listdir(REAL_DIR))
fake_files = [f for f in os.listdir(FAKE_DIR) if f.endswith(".jpg")]

print("Real:", real_count)
print("Fake:", len(fake_files))

# Shuffle fake images
random.shuffle(fake_files)

# Pick SAME number as real
balanced = fake_files[:real_count]

for idx, f in enumerate(balanced):
    src = os.path.join(FAKE_DIR, f)
    dst = os.path.join(OUT_DIR, f"{idx}.jpg")
    shutil.copy(src, dst)

print(f"\n🎉 Balanced FAKE created with {real_count} images!")
