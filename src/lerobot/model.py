"""Simple PyTorch model implementations for lerobot.

Provides a tiny MLP used as a placeholder VLA model. Replace with your own Vision-Language architecture.
"""
import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
    
def build_model(cfg: dict) -> nn.Module:
    t = cfg.get("model", {})
    if t.get("type", "mlp") == "mlp":
        return TinyMLP(t.get("input_dim", 64), t.get("hidden_dim", 128), t.get("output_dim", 10))
    raise ValueError(f"Unknown model type: {t.get('type')}")
