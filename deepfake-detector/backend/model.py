# model.py
"""
Lightweight MobileNetV2 classifier wrapper.
Provides: build_model(num_classes), load_model(weights_path=None, device='cpu')
"""
from pathlib import Path
import torch
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

MODEL_WEIGHTS_DIR = Path(__file__).parent / "model_weights"
MODEL_WEIGHTS_DIR.mkdir(exist_ok=True)

def build_model(num_classes=2):
    # Use modern weights API to avoid warnings
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(in_features, num_classes)
    )
    return model

def load_model(weights_path: str = None, device: str = "cpu"):
    device = torch.device(device)
    model = build_model(num_classes=2)
    model.to(device)
    if weights_path:
        p = Path(weights_path)
        if p.exists():
            state = torch.load(str(p), map_location=device)
            # Support checkpoint with 'model_state_dict' or raw state dict
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            try:
                model.load_state_dict(state)
                print("Loaded weights:", weights_path)
            except Exception as e:
                print("Failed loading state_dict:", e)
        else:
            print("Weights path not found:", weights_path)
    model.eval()
    return model
