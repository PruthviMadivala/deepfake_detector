import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from utils import preprocess_image, device
from pathlib import Path
from PIL import Image


def generate_gradcam(model, video_path):
    """
    Generates GradCAM heatmap for a single representative frame.
    Saves an image at: uploads/gradcam_xxx.png
    Returns the saved file path.
    """
    # Extract 1 frame from the video
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        return {"error": "Could not extract frame from video"}

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert("RGB")

    # Preprocess
    input_tensor = preprocess_image(pil_img).unsqueeze(0).to(device)

    # Forward hook to capture activations
    features = []
    gradients = []

    def forward_hook(module, inp, out):
        features.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Target layer for GradCAM (MobileNetV2 last conv layer)
    target_layer = model.features[-1]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1)
    pred_class = probs.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    class_loss = logits[0, pred_class]
    class_loss.backward()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # GradCAM calculation
    fmap = features[0].cpu().detach().numpy()[0]
    grad = gradients[0].cpu().detach().numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    heatmap = (cam * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_colored, 0.4, frame, 0.6, 0)

    # Save heatmap
    out_path = Path("uploads") / "gradcam_output.png"
    cv2.imwrite(str(out_path), overlay)

    return str(out_path)
