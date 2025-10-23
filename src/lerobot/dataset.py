"""Dataset utilities and a synthetic dataset for quick experiments."""
from typing import Tuple
import numpy as np
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, size: int = 1000, input_dim: int = 64, num_classes: int = 10, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.X = rng.randn(size, input_dim).astype("float32")
        self.y = rng.randint(0, num_classes, size).astype("int64")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_dataloaders(cfg: dict):
    from torch.utils.data import DataLoader, random_split
    t = cfg.get("dataset", {})
    size = t.get("size", 1000)
    input_dim = cfg.get("model", {}).get("input_dim", 64)
    num_classes = cfg.get("model", {}).get("output_dim", 10)
    val_split = t.get("val_split", 0.2)
    seed = cfg.get("train", {}).get("seed", 42)

    ds = SyntheticDataset(size=size, input_dim=input_dim, num_classes=num_classes, seed=seed)
    val_len = int(len(ds) * val_split)
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])

    batch_size = cfg.get("train", {}).get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
