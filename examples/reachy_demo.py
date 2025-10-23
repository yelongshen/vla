"""Demo runner for ReachyEnv. Provide REACHY_URDF env var or pass path argument.

Run example:
  REACHY_URDF=/path/to/reachy.urdf python examples/reachy_demo.py
"""
import os
import numpy as np
from lerobot.envs.reachy_env import ReachyEnv


def main():
    urdf = os.environ.get("REACHY_URDF")
    if urdf is None:
        print("Set REACHY_URDF to your Reachy URDF path before running.")
        return

    env = ReachyEnv(urdf_path=urdf, render=False)
    obs = env.reset()
    action_dim = len(obs["joint_positions"])
    # compute observation flattened dimension
    obs_components = {k: np.asarray(v).shape for k, v in obs.items()}
    obs_flat_dim = sum(np.asarray(v).size for v in obs.values())
    print("Loaded Reachy with", action_dim, "action dim")
    print("Observation components:")
    for k, s in obs_components.items():
        print(f"  {k}: shape={s}")
    print("Flattened observation size:", obs_flat_dim)

    # simple zero-target sequence
    for t in range(200):
        a = np.zeros(action_dim)
        obs, r, done, info = env.step(a)
        if t % 20 == 0:
            print(f"t={t} base_z={obs['base_pos'][2]:.3f}")

    env.close()


if __name__ == "__main__":
    main()
