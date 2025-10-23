"""PyBullet demo: run the PyBulletStairsEnv with random continuous actions to observe stepping/climbing.

Run: python examples/pybullet_demo.py
"""
import numpy as np
from lerobot.envs.pybullet_env import PyBulletStairsEnv
import time
import os
from PIL import Image


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

    # prepare output directory for saved frames
    out_dir = os.path.join("outputs", "pybullet")
    os.makedirs(out_dir, exist_ok=True)
    total_steps = 200
    save_count = 8
    save_every = max(1, total_steps // save_count)
    for t in range(total_steps):
        # random smooth actions
        a = np.tanh(np.random.randn(action_dim) * 0.5)
        obs, r, done, info = env.step(a)
        if t % 20 == 0:
            print(f"t={t} base_z={obs['base_pos'][2]:.3f} x={obs['base_pos'][0]:.3f}")

        # capture and save frames at intervals
        if t % save_every == 0:
            try:
                width = 320
                height = 240
                cam_target = obs.get("base_pos", [0.5, 0, 0.2])
                cam_distance = 1.0
                yaw, pitch, roll = 50, -30, 0
                up_axis_index = 2
                # use positional args for compatibility: (cameraTargetPosition, distance, yaw, pitch, roll, upAxisIndex)
                view = p.computeViewMatrixFromYawPitchRoll(cam_target, cam_distance, yaw, pitch, roll, up_axis_index)
                proj = p.computeProjectionMatrixFOV(fov=60, aspect=float(width)/height, nearVal=0.01, farVal=10.0)
                w, h, rgb, depth_buf, seg = p.getCameraImage(width, height, viewMatrix=view, projectionMatrix=proj, renderer=p.ER_TINY_RENDERER)
                # ensure rgb is uint8 and reshape according to returned w,h
                rgb_np = np.asarray(rgb, dtype=np.uint8)
                try:
                    rgb_arr = rgb_np.reshape((h, w, 4))[:, :, :3]
                except Exception:
                    # fallback: try (w,h,4)
                    rgb_arr = rgb_np.reshape((w, h, 4)).transpose(1, 0, 2)[:, :, :3]
                depth_arr = np.reshape(depth_buf, (h, w))
                seg_arr = np.reshape(seg, (h, w))

                idx = (t // save_every)
                rgb_path = os.path.join(out_dir, f"frame_{idx:03d}_rgb.png")
                depth_path = os.path.join(out_dir, f"frame_{idx:03d}_depth.npy")
                seg_path = os.path.join(out_dir, f"frame_{idx:03d}_seg.npy")
                Image.fromarray(rgb_arr).save(rgb_path)
                np.save(depth_path, depth_arr)
                np.save(seg_path, seg_arr)
                print(f"Saved frame {idx} -> {rgb_path}")
            except Exception as e:
                print("Frame capture failed:", e)

        time.sleep(0.02)

    env.close()


if __name__ == "__main__":
    main(render=False)
