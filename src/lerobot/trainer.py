"""Training loop and utilities for lerobot."""
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .model import build_model
from .dataset import make_dataloaders


def save_checkpoint(state: dict, out_dir: str, name: str = "checkpoint.pt"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    torch.save(state, path)
    print(f"Saved checkpoint: {path}")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(cfg_path: str):
    cfg = load_config(cfg_path) if cfg_path else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    train_loader, val_loader = make_dataloaders(cfg)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("train", {}).get("lr", 1e-3))

    epochs = cfg.get("train", {}).get("epochs", 5)
    out_dir = cfg.get("output", {}).get("dir", "outputs")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
            pbar.set_postfix({"loss": total_loss / total, "acc": correct / total})

        # Simple validation pass
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                l = criterion(logits, yb)
                val_loss += l.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        print(f"Epoch {epoch}: train_loss={total_loss/total:.4f} train_acc={correct/total:.4f} ")
        print(f"           val_loss={val_loss/val_total:.4f} val_acc={val_correct/val_total:.4f}")

        # Save checkpoint per epoch
        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg": cfg,
        }, out_dir, name=f"checkpoint_epoch{epoch}.pt")

    # Final checkpoint
    save_checkpoint({
        "epoch": epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg,
    }, out_dir, name="checkpoint.pt")

    print("Training complete")
