"""MuJoCo RL example with PPO (falls back to CartPole if MuJoCo not available).

This script trains a PPO agent using Stable-Baselines3. If MuJoCo envs are not available
on the host, it falls back to CartPole-v1 so you can test the training loop.

Usage:
  python examples/mujoco_rl_example.py --timesteps 50000
"""
import os
import argparse
import numpy as np


def main(total_timesteps: int = 10000, env_id: str = None, render: bool = False, save_video: str = None):
    # lazy imports
    # Prefer gymnasium (actively maintained). Fall back to gym if gymnasium isn't present.
    gym = None
    gym_pkg = None
    try:
        import gymnasium as _gym
        gym = _gym
        gym_pkg = 'gymnasium'
    except Exception:
        try:
            import gym as _gym
            gym = _gym
            gym_pkg = 'gym'
        except Exception as e:
            raise RuntimeError(
                "Neither `gymnasium` nor `gym` is installed.\n"
                "Install with: pip install gymnasium stable-baselines3[extra]\n"
                "or see the migration guide: https://gymnasium.farama.org/introduction/migration_guide/"
            ) from e

    # try to select a MuJoCo env
    env = None
    selected_env = env_id
    if env_id is None:
        mujo_env_candidates = ["HalfCheetah-v4", "HalfCheetah-v2", "Ant-v4", "Ant-v2", "Humanoid-v4", "Humanoid-v2"]
        for cand in mujo_env_candidates:
            try:
                env = gym.make(cand)
                selected_env = cand
                print(f"Using MuJoCo environment: {cand}")
                break
            except Exception:
                env = None
    else:
        try:
            env = gym.make(env_id)
            selected_env = env_id
        except Exception:
            env = None

    if env is None:
        print("MuJoCo environments not available. Falling back to CartPole-v1 for demo.")
        selected_env = "CartPole-v1"
        env = gym.make(selected_env)

    # Stable Baselines3
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ModuleNotFoundError as e:
        # If shimmy or another dependency is missing, give a clear instruction
        missing = getattr(e, 'name', None) or str(e)
        raise RuntimeError(
            f"A dependency for stable-baselines3 is missing: {missing}.\n"
            "Install with: pip install stable-baselines3[extra] shimmy\n"
            "Or: pip install -r requirements.txt"
        ) from e
    except Exception as e:
        raise RuntimeError(
            "stable-baselines3 is required. Install with `pip install stable-baselines3[extra] shimmy`\n"
            "If you only want to run the fallback CartPole demo, install: pip install gym stable-baselines3[extra] shimmy"
        ) from e

    # vectorized env for training
    try:
        vec_env = make_vec_env(lambda: gym.make(selected_env), n_envs=1)
    except ModuleNotFoundError as e:
        missing = getattr(e, 'name', None) or str(e)
        raise RuntimeError(
            f"A dependency needed to create vectorized envs is missing: {missing}.\n"
            "Install shimmy and stable-baselines3 extras: pip install stable-baselines3[extra] shimmy"
        ) from e

    # Print vec_env spaces and shapes so users can inspect obs/action dimensions for their env
    try:
        print("vec_env.num_envs:", getattr(vec_env, 'num_envs', None))
        print("vec_env.observation_space:", getattr(vec_env, 'observation_space', None))
        print("vec_env.action_space:", getattr(vec_env, 'action_space', None))
        obs_space = getattr(vec_env, 'observation_space', None)
        act_space = getattr(vec_env, 'action_space', None)
        if hasattr(obs_space, 'shape'):
            print("vec_env.observation_space.shape:", obs_space.shape)
        elif hasattr(obs_space, 'spaces'):
            print("vec_env.observation_space is a Dict with keys:", list(obs_space.spaces.keys()))
            for k, s in obs_space.spaces.items():
                print(f" - {k}: shape={getattr(s, 'shape', None)}")
        if hasattr(act_space, 'shape'):
            print("vec_env.action_space.shape:", act_space.shape)
        elif hasattr(act_space, 'n'):
            print("vec_env.action_space.n:", act_space.n)
    except Exception as _:
        # best-effort printing; don't fail execution if introspection fails
        pass

    print("action_space sample:", vec_env.action_space.sample())
    print("action_space shape:", getattr(vec_env.action_space, "shape", None))
    # If continuous:
    action_dim = vec_env.action_space.shape[0] if vec_env.action_space.shape else None
    print("action_dim:", action_dim)

    model_dir = os.path.join("outputs", "mujoco")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"ppo_{selected_env.replace('/', '_')}.zip")

    print(f"Training PPO on {selected_env} for {total_timesteps} timesteps")
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # evaluation
    eval_episodes = 5
    rewards = []
    video_writer = None
    frames = []
    if save_video:
        try:
            import imageio
        except Exception:
            raise RuntimeError("imageio is required to save videos. Install with `pip install imageio[ffmpeg]`")
        try:
            eval_env = gym.make(selected_env, render_mode='rgb_array')
        except Exception:
            eval_env = gym.make(selected_env)
        env = eval_env

        
    for ep in range(eval_episodes):
        # gym vs gymnasium reset differences: gymnasium returns (obs, info)
        reset_ret = env.reset()
        if isinstance(reset_ret, tuple) and len(reset_ret) >= 1:
            ob = reset_ret[0]
        else:
            ob = reset_ret
        print("obs type:", type(ob), "shape:", getattr(np.asarray(ob), "shape", None), "dtype:", getattr(np.asarray(ob), "dtype", None))
        done = False
        ep_rew = 0.0
        while True:
            # model.predict may accept a batch or a single observation depending on the policy; handle both
            try:
                action, _ = model.predict(ob, deterministic=True)
            except Exception:
                # some gym wrappers expect flattened observations or different types; try wrapping in list
                action, _ = model.predict(np.array([ob]), deterministic=True)
                # unwrap action
                if isinstance(action, (list, tuple, np.ndarray)) and getattr(action, 'shape', None) and action.shape[0] == 1:
                    action = action[0]

            step_ret = env.step(action)
            # gym returns (obs, reward, done, info)
            # gymnasium returns (obs, reward, terminated, truncated, info)
            if len(step_ret) == 4:
                ob, r, done, info = step_ret
            elif len(step_ret) == 5:
                ob, r, terminated, truncated, info = step_ret
                done = bool(terminated or truncated)
            else:
                # unexpected format
                raise RuntimeError(f"Unexpected env.step() return shape: {type(step_ret)} len={len(step_ret)}")

            ep_rew += float(r)
            if render:
                # some envs from gymnasium need render(mode='human') or env.render() depending on backend
                try:
                    env.render()
                except Exception:
                    try:
                        env.render(mode='human')
                    except Exception:
                        pass

            if save_video:
                try:
                    # gymnasium with render_mode='rgb_array' returns frames from render()
                    frame = None
                    try:
                        frame = eval_env.render()
                    except Exception:
                        try:
                            frame = eval_env.render(mode='rgb_array')
                        except Exception:
                            frame = None
                    if frame is not None:
                        # ensure uint8 HWC
                        frm = np.asarray(frame)
                        if frm.dtype != np.uint8:
                            # scale/clip floats to uint8 if needed
                            frm = (255 * np.clip(frm, 0, 1)).astype(np.uint8)
                        frames.append(frm)
                except Exception:
                    # non-fatal: continue without saving this frame
                    pass

            if done:
                break

        rewards.append(ep_rew)
        print(f"Eval episode {ep+1} reward: {ep_rew:.2f}")

    print(f"Mean eval reward: {np.mean(rewards):.2f} (std {np.std(rewards):.2f})")
    env.close()
    # write video if requested
    if save_video and frames:
        try:
            import imageio
            imageio.get_writer(save_video, fps=30).append_data(frames[0])
            # append all frames (imageio writer can be used in a context manager, but for simplicity use mimsave)
            imageio.mimsave(save_video, frames, fps=30)
            print(f"Saved rollout video to {save_video}")
        except Exception as e:
            print("Failed to write video:", e)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=10000)
    p.add_argument('--env', type=str, default=None)
    p.add_argument('--render', action='store_true')
    p.add_argument('--save-video', type=str, default=None,
                   help='Path to save evaluation rollout mp4 (e.g. outputs/rollout.mp4)')
    args = p.parse_args()
    main(total_timesteps=args.timesteps, env_id=args.env, render=args.render, save_video=args.save_video)
