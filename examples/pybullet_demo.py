"""PyBullet demo: run the PyBulletStairsEnv with random continuous actions to observe stepping/climbing.

Run: python examples/pybullet_demo.py
"""
import numpy as np
from lerobot.envs.pybullet_env import PyBulletStairsEnv
import time


def main(render=False):
    env = PyBulletStairsEnv(render=render)
    obs = env.reset()

    # Print observation keys and contents summary
    try:
        obs_keys = list(obs.keys())
    except Exception:
        obs_keys = None
    print("Observation keys:", obs_keys)
    print("Initial base height:", obs.get("base_pos")[2] if obs.get("base_pos") is not None else None)

    # Action space: continuous vector of joint position targets
    action_dim = len(obs.get("joint_positions", []))

    # Print available joints and limits (if pybullet available)
    try:
        p = env._p
        print("Action dim:", action_dim)
        print("Joint ids:", env.joint_ids)
        for j in env.joint_ids:
            info = p.getJointInfo(env.robot, j)
            name = info[1].decode() if isinstance(info[1], (bytes, bytearray)) else str(info[1])
            lower = info[8]
            upper = info[9]
            print(f"  joint {j}: name={name} type={info[2]} limits=({lower},{upper})")
    except Exception as e:
        print("Could not print joint info:", e)
    for t in range(200):
        # random smooth actions
        a = np.tanh(np.random.randn(action_dim) * 0.5)
        obs, r, done, info = env.step(a)
        if t % 20 == 0:
            print(f"t={t} base_z={obs['base_pos'][2]:.3f} x={obs['base_pos'][0]:.3f}")
        time.sleep(0.02)

    env.close()


if __name__ == "__main__":
    main(render=False)
