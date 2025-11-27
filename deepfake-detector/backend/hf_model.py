import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

# Tiny local model (no downloads)
class TinyDeepfakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_hf_model():
    model = TinyDeepfakeNet()
    model.eval()
    print("Loaded Tiny Local Deepfake Model (offline, fast).")
    return model

def read_frame_as_pil(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    center = total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, center)
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

def predict_hf(model, path):
    if path.lower().endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(path).convert("RGB")
    else:
        img = read_frame_as_pil(path)
        if img is None:
            return {"label": "UNKNOWN", "confidence": 0}

    x = transform(img).unsqueeze(0)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().numpy()

    fake_prob = float(probs[1])
    real_prob = float(probs[0])

    label = "FAKE" if fake_prob > real_prob else "REAL"
    confidence = int(max(fake_prob, real_prob) * 100)

    return {"label": label, "confidence": confidence}
