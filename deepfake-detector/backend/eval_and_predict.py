# eval_and_predict.py
import argparse
from model import load_model
from utils import extract_video_frames, preprocess_image, device
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
import json

def predict_video_path(model, video_path, num_samples=8, device_str="cpu"):
    frames = extract_video_frames(video_path, max_frames=200)
    if len(frames) == 0:
        return {"label":"UNKNOWN","confidence":0}
    idxs = np.linspace(0, len(frames)-1, min(num_samples, len(frames))).astype(int)
    votes = []
    confs = []
    for i in idxs:
        pil = Image.fromarray(frames[i]).convert("RGB")
        x = preprocess_image(pil).unsqueeze(0).to(device_str)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        real, fake = float(probs[0]), float(probs[1])
        if fake >= real:
            votes.append("FAKE"); confs.append(fake)
        else:
            votes.append("REAL"); confs.append(real)
    final_label = Counter(votes).most_common(1)[0][0]
    avg_conf = int(round(float(np.mean([c for v,c in zip(votes,confs) if v==final_label]))*100))
    return {"label": final_label, "confidence": avg_conf, "votes": votes}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model_weights/improved_model.pth")
    parser.add_argument("--video", default="/mnt/data/show.mp4")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = load_model(weights_path=args.model_path, device=args.device)
    res = predict_video_path(model, args.video, device_str=args.device)
    print(json.dumps(res, indent=2))
