# train.py
"""
Train a MobileNetV2 classifier on frames extracted from videos.
Simple, small-batch training intended for quick local runs.
"""

import argparse
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import os
from model import build_model, MODEL_WEIGHTS_DIR

# Simple dataset: extract N frames per video on the fly
class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, label, frames_per_video=8, transform=None):
        self.root_dir = Path(root_dir)
        self.files = list(self.root_dir.glob("*.mp4")) + list(self.root_dir.glob("*.mov")) + list(self.root_dir.glob("*.avi"))
        self.frames_per_video = frames_per_video
        self.transform = transform
        self.label = label

    def __len__(self):
        return max(1, len(self.files) * self.frames_per_video)

    def __getitem__(self, idx):
        if len(self.files) == 0:
            # fallback: return a random noise image (avoid crash)
            img = Image.fromarray((np.random.rand(224,224,3)*255).astype('uint8'))
            if self.transform: img = self.transform(img)
            return img, torch.tensor(self.label, dtype=torch.long)
        vid_idx = idx // self.frames_per_video
        frame_idx = idx % self.frames_per_video
        video_path = str(self.files[vid_idx])
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        # sample a frame evenly
        target = int(np.linspace(0, max(total-1,0), self.frames_per_video).astype(int)[frame_idx])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            # fallback random image
            img = Image.fromarray((np.random.rand(224,224,3)*255).astype('uint8'))
        else:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.label, dtype=torch.long)

def make_dataloaders(data_root, batch_size=16, frames_per_video=8, num_workers=0):
    train_real = VideoFrameDataset(Path(data_root)/"train"/"real", label=0, frames_per_video=frames_per_video, transform=TRAIN_TRANSFORM)
    train_fake = VideoFrameDataset(Path(data_root)/"train"/"fake", label=1, frames_per_video=frames_per_video, transform=TRAIN_TRANSFORM)
    val_real = VideoFrameDataset(Path(data_root)/"val"/"real", label=0, frames_per_video=frames_per_video, transform=VAL_TRANSFORM)
    val_fake = VideoFrameDataset(Path(data_root)/"val"/"fake", label=1, frames_per_video=frames_per_video, transform=VAL_TRANSFORM)

    train_ds = torch.utils.data.ConcatDataset([train_real, train_fake])
    val_ds = torch.utils.data.ConcatDataset([val_real, val_fake])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

# Transforms
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def train_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss / max(1,total), correct/total if total>0 else 0.0

def eval_epoch(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return running_loss / max(1,total), correct/total if total>0 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data", help="data root with train/val subfolders")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames-per-video", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_model(num_classes=2).to(device)

    train_loader, val_loader = make_dataloaders(args.data_root, batch_size=args.batch_size, frames_per_video=args.frames_per_video)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    out_path = MODEL_WEIGHTS_DIR / "improved_model.pth"

    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(out_path))
            print("Saved improved model to", out_path)

    print("Done. Best val acc:", best_val_acc)
    print("Final model path:", out_path)

if __name__ == "__main__":
    main()
