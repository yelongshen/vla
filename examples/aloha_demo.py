"""Run the TinyALOHA denoising policy in the PyBullet stairs env.

This demo shows where to hook a diffusion/denoising policy into the env loop.
"""
import numpy as np
from lerobot.envs.pybullet_env import PyBulletStairsEnv
from lerobot.algorithms.aloha import TinyALOHA


def main():
    env = PyBulletStairsEnv(render=False)
    obs = env.reset()
    state = np.concatenate([obs["joint_positions"], obs["base_pos"]])
    action_dim = len(obs["joint_positions"])

    policy = TinyALOHA(state_dim=state.shape[0], action_dim=action_dim)

    # run a few steps
    for t in range(200):
        a = policy.sample(state, steps=8)
        obs, r, done, info = env.step(a)
        state = np.concatenate([obs["joint_positions"], obs["base_pos"]])
        if t % 20 == 0:
            print(f"t={t} base_z={obs['base_pos'][2]:.3f} x={obs['base_pos'][0]:.3f}")

    env.close()


if __name__ == "__main__":
    main()
