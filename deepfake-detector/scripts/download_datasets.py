import os
import gdown
import zipfile
import tarfile
import requests
from tqdm import tqdm

DATA_ROOT = "../data"
os.makedirs(DATA_ROOT, exist_ok=True)



# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def download_file(url, output):
    """Streaming download with progress bar"""
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total, unit='iB', unit_scale=True, desc=os.path.basename(output))
    with open(output, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()


def extract_zip(path, dst):
    print(f"Extracting {path}...")
    with zipfile.ZipFile(path, 'r') as z:
        z.extractall(dst)


def extract_tar(path, dst):
    print(f"Extracting {path}...")
    with tarfile.open(path, 'r') as t:
        t.extractall(dst)



# -----------------------------
# 1) FACEFORENSICS++ (SAMPLE)
# -----------------------------
def download_faceforensics():
    print("\n📥 Downloading FaceForensics++ sample dataset...")
    url = "http://kaldir.vc.in.tum.de/FaceForensics/samples/FaceSwap/sample_videos.zip"
    out = os.path.join(DATA_ROOT, "ffpp_samples.zip")

    download_file(url, out)
    extract_zip(out, os.path.join(DATA_ROOT, "ffpp"))

    print("✅ FaceForensics++ downloaded successfully")



# -----------------------------
# 2) CELEB-DF V2  (GOOGLE DRIVE MIRROR)
# -----------------------------
def download_celebdf():
    print("\n📥 Downloading Celeb-DF v2 (Google Drive Mirror)...")

    # Public research mirror
    gdrive_url = "https://drive.google.com/uc?id=1iMfQxGc1dQc5sKXc5teLoI0lp4rWuIwo"
    out = os.path.join(DATA_ROOT, "celeb_df_v2.zip")

    gdown.download(gdrive_url, out, quiet=False)
    extract_zip(out, os.path.join(DATA_ROOT, "celebdf"))

    print("✅ Celeb-DF downloaded successfully")



# -----------------------------
# 3) DFDC MINI DATASET (4GB)
# -----------------------------
def download_dfdc():
    print("\n📥 Downloading DFDC mini dataset...")

    url = "https://storage.googleapis.com/kaggle-data-sets/551642/991661/bundle/archive.zip"
    out = os.path.join(DATA_ROOT, "dfdc_mini.zip")

    download_file(url, out)
    extract_zip(out, os.path.join(DATA_ROOT, "dfdc"))

    print("✅ DFDC mini dataset downloaded successfully")



# -----------------------------
# ORGANIZE INTO /data/videos/real AND /data/videos/fake
# -----------------------------
def organize_videos():
    print("\n📁 Organizing dataset videos...")

    import shutil

    final_real = os.path.join(DATA_ROOT, "videos/real")
    final_fake = os.path.join(DATA_ROOT, "videos/fake")

    os.makedirs(final_real, exist_ok=True)
    os.makedirs(final_fake, exist_ok=True)



    # ------------- FaceForensics++ -------------
    ff_real = os.path.join(DATA_ROOT, "ffpp/original_sequences/youtube/c23/videos")
    ff_fake = os.path.join(DATA_ROOT, "ffpp/manipulated_sequences/DeepFakeDetection/c23/videos")

    if os.path.exists(ff_real):
        for f in os.listdir(ff_real):
            shutil.copy(os.path.join(ff_real, f), final_real)

    if os.path.exists(ff_fake):
        for f in os.listdir(ff_fake):
            shutil.copy(os.path.join(ff_fake, f), final_fake)



    # ------------- Celeb-DF -------------
    celeb_real = os.path.join(DATA_ROOT, "celebdf/Celeb-real")
    celeb_fake = os.path.join(DATA_ROOT, "celebdf/Celeb-synthesis")

    if os.path.exists(celeb_real):
        for f in os.listdir(celeb_real):
            shutil.copy(os.path.join(celeb_real, f), final_real)

    if os.path.exists(celeb_fake):
        for f in os.listdir(celeb_fake):
            shutil.copy(os.path.join(celeb_fake, f), final_fake)



    # ------------- DFDC MINI -------------
    dfdc_root = os.path.join(DATA_ROOT, "dfdc")

    if os.path.exists(dfdc_root):
        for f in os.listdir(dfdc_root):
            if f.endswith(".mp4"):
                if "real" in f.lower():
                    shutil.copy(os.path.join(dfdc_root, f), final_real)
                else:
                    shutil.copy(os.path.join(dfdc_root, f), final_fake)



    print("✅ Videos organized into:")
    print("   ➤ data/videos/real")
    print("   ➤ data/videos/fake")



# -----------------------------
# MAIN FUNCTION
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting dataset download...")

    download_faceforensics()
    download_celebdf()
    download_dfdc()

    organize_videos()

    print("\n🎉 ALL DATASETS DOWNLOADED & ORGANIZED SUCCESSFULLY 🎉")
