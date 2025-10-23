"""PyBullet demo: run the PyBulletStairsEnv with random continuous actions to observe stepping/climbing.

Run: python examples/pybullet_demo.py
"""
import numpy as np
from lerobot.envs.pybullet_env import PyBulletStairsEnv
import time


def main(render=False):
    env = PyBulletStairsEnv(render=render)
    obs = env.reset()
    print("Initial base height:", obs["base_pos"][2])

    action_dim = len(obs["joint_positions"])
    for t in range(200):
        # random smooth actions
        a = np.tanh(np.random.randn(action_dim) * 0.5)
        obs, r, done, info = env.step(a)
        if t % 20 == 0:
            print(f"t={t} base_z={obs['base_pos'][2]:.3f} x={obs['base_pos'][0]:.3f}")
        time.sleep(0.02)

    env.close()


if __name__ == "__main__":
    main(render=True)
