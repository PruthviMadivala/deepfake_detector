"""
utils.py: preprocessing, device, frame extraction helpers
"""
import torch
from torchvision import transforms
import numpy as np
import cv2

device = torch.device("cpu")

# preprocess function returns tensor normalized for ImageNet pretrained backbone
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def preprocess_image(pil_image):
    """
    pil_image: PIL.Image RGB
    returns torch.Tensor (C,H,W)
    """
    return _preprocess(pil_image)

def extract_video_frames(video_path, max_frames=60):
    """
    Returns list of RGB numpy arrays (H,W,3) extracted from video.
    We sample frames evenly but return all extracted frames.
    """
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return frames
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # read frames
    success, frame = cap.read()
    while success and len(frames) < max_frames:
        # convert BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        success, frame = cap.read()
    cap.release()
    return frames
