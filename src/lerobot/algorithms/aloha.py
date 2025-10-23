"""Tiny illustrative ALOHA-like diffusion policy stub.

This is NOT the real ALOHA implementation. It shows where to plug a diffusion-based
denoising policy that maps noisy actions to refined continuous action vectors.

The toy policy is a small MLP that takes state + noise level and predicts an action.
"""
import torch
import torch.nn as nn
import numpy as np


class TinyDenoiser(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: torch.Tensor, noise_level: torch.Tensor):
        # state: (B, state_dim), noise_level: (B,1)
        x = torch.cat([state, noise_level], dim=-1)
        return self.net(x)


class TinyALOHA:
    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu"):
        self.device = device
        self.model = TinyDenoiser(state_dim, action_dim).to(device)

    def sample(self, state_np: np.ndarray, steps: int = 10) -> np.ndarray:
        """Run a tiny iterative denoising process to produce an action.

        state_np: (state_dim,) numpy array
        returns: action numpy array
        """
        state = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0).to(self.device)
        # start from gaussian noise
        action = torch.randn(1, self.model.net[-1].out_features, device=self.device)
        for t in range(steps):
            noise_level = torch.tensor([[float((steps - t) / steps)]], device=self.device)
            pred = self.model(state, noise_level)
            # simple update: move noisy action towards pred
            action = action * 0.9 + 0.1 * pred
        return action.squeeze(0).cpu().numpy()

    def to(self, device: str):
        self.device = device
        self.model.to(device)
        return self

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
