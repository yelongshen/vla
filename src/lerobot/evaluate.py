"""Evaluation utilities for lerobot."""
import torch
import yaml
import os
import numpy as np
from .model import build_model
from .dataset import make_dataloaders


def load_checkpoint(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def evaluate(ckpt_path: str, cfg_path: str = None):
    cfg = None
    if cfg_path:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
    ckpt = load_checkpoint(ckpt_path)
    if cfg is None and "cfg" in ckpt:
        cfg = ckpt["cfg"]
    if cfg is None:
        raise ValueError("No config provided and checkpoint has no cfg")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    _, val_loader = make_dataloaders(cfg)
    total = 0
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(yb.numpy())
            total += xb.size(0)
            correct += (preds == yb.numpy()).sum().item()

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    acc = correct / total
    # simple per-class accuracy
    classes = np.unique(all_labels)
    per_class = {}
    for c in classes:
        idx = all_labels == c
        per_class[int(c)] = float((all_preds[idx] == all_labels[idx]).sum() / idx.sum())

    print(f"Eval results: accuracy={acc:.4f}")
    print("Per-class accuracies:")
    for k, v in per_class.items():
        print(f"  class {k}: {v:.4f}")
    return {"accuracy": acc, "per_class": per_class}
