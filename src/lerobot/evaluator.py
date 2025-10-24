"""Generic evaluator for VLA models in lerobot environments.

This module provides utilities to load a PyTorch checkpoint and run episodes in a chosen
environment (pybullet stairs or reachy). It expects the model to accept a flattened
observation vector and return a continuous action vector matching the env action_dim.

Model factory: pass a dotted path to a callable that builds the model: e.g. "mypkg.models.build_model".
The callable should accept a config dict (or None) and return an nn.Module.

If no factory is provided, the evaluator will attempt to load a checkpoint containing
"model_state" only and will require a user-provided model instance.
"""
from typing import Callable, Optional
import importlib
import os
import yaml
import numpy as np

import torch


def import_callable(dotted: str) -> Callable:
    module_name, fn_name = dotted.rsplit('.', 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, fn_name)


def load_checkpoint(ckpt_path: str) -> dict:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    return torch.load(ckpt_path, map_location='cpu')


def build_model_from_factory(factory_path: Optional[str], cfg: Optional[dict], device: str = 'cpu'):
    if not factory_path:
        return None
    factory = import_callable(factory_path)
    model = factory(cfg or {})
    model.to(device)
    return model


def flatten_observation(obs: dict) -> np.ndarray:
    # simple deterministic flattening: sort keys
    parts = []
    for k in sorted(obs.keys()):
        v = obs[k]
        a = np.asarray(v)
        parts.append(a.ravel())
    if parts:
        return np.concatenate(parts, axis=0).astype(np.float32)
    return np.zeros(0, dtype=np.float32)


def run_episode(env, model: torch.nn.Module, device: str = 'cpu', max_steps: int = 200):
    obs = env.reset()
    total_reward = 0.0
    states = []
    for t in range(max_steps):
        x = flatten_observation(obs)
        states.append(x)
        if model is None:
            # random action fallback
            action = np.zeros(len(env.joint_ids), dtype=np.float32)
        else:
            with torch.no_grad():
                inp = torch.from_numpy(x).unsqueeze(0).to(device)
                out = model(inp)
                if isinstance(out, tuple) or isinstance(out, list):
                    out = out[0]
                action = out.squeeze(0).cpu().numpy()
        obs, r, done, info = env.step(action)
        total_reward += float(r)
        if done:
            break
    return total_reward, {'steps': t + 1}


def evaluate_on_env(env_name: str, ckpt_path: Optional[str] = None, factory: Optional[str] = None, episodes: int = 5, device: str = 'cpu'):
    # choose environment
    if env_name == 'pybullet':
        from .envs.pybullet_env import PyBulletStairsEnv
        env = PyBulletStairsEnv(render=False)
    elif env_name == 'reachy':
        from .envs.reachy_env import ReachyEnv
        # ReachyEnv expects REACHY_URDF or path; pass None and let user env set it
        env = ReachyEnv(render=False)
    else:
        raise ValueError(f'Unknown env: {env_name}')

    # load checkpoint and model
    model = None
    cfg = None
    if ckpt_path:
        ckpt = load_checkpoint(ckpt_path)
        cfg = ckpt.get('cfg') or {}
        if factory:
            model = build_model_from_factory(factory, cfg, device=device)
            if hasattr(model, 'load_state_dict') and 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
        else:
            # user may have provided a checkpoint with full model
            if 'model_state' in ckpt and 'model_class' in ckpt:
                # try to import model class
                try:
                    model_cls = import_callable(ckpt['model_class'])
                    model = model_cls(cfg or {})
                    model.load_state_dict(ckpt['model_state'])
                    model.to(device)
                except Exception:
                    model = None
    if model is None:
        print('No model loaded; running with zero/random policy fallback')

    # run episodes and collect metrics
    rewards = []
    for ep in range(episodes):
        r, info = run_episode(env, model, device=device)
        print(f'Episode {ep}: reward={r:.3f} steps={info.get("steps")}')
        rewards.append(r)

    env.close()
    return {'mean_reward': float(np.mean(rewards)), 'rewards': rewards}
