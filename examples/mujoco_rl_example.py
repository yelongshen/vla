"""MuJoCo RL example with PPO (falls back to CartPole if MuJoCo not available).

This script trains a PPO agent using Stable-Baselines3. If MuJoCo envs are not available
on the host, it falls back to CartPole-v1 so you can test the training loop.

Usage:
  python examples/mujoco_rl_example.py --timesteps 50000
"""
import os
import argparse
import numpy as np


def configure_mujoco_gl(requested_backend: str, force_backend: bool) -> str:
    """Configure MuJoCo's OpenGL backend before environments are created."""
    original_backend = os.environ.get("MUJOCO_GL")
    backend = requested_backend

    if requested_backend == "auto":
        if original_backend:
            backend = original_backend
        elif os.environ.get("DISPLAY") is None:
            backend = "egl"
        else:
            backend = "glfw"

    if backend:
        os.environ["MUJOCO_GL"] = backend

    if backend == "egl" and os.environ.get("MUJOCO_EGL_DEVICE_ID") is None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ.get("EGL_DEVICE_ID", "0")

    if original_backend and backend != original_backend:
        print(f"Overriding existing MUJOCO_GL={original_backend} with {backend} (force={force_backend}).")
    else:
        print(f"Configured MUJOCO_GL={backend} (force={force_backend}).")

    return backend


def patch_mujoco_glcontext_del():
    """Ensure MuJoCo's GLContext.__del__ tolerates partially initialized contexts."""
    try:
        import mujoco
    except Exception:
        return

    original_del = getattr(mujoco.GLContext, "__del__", None)
    if original_del is None or getattr(mujoco.GLContext, "__del__patched", False):
        return

    def safe_del(self, _original_del=original_del):
        if not hasattr(self, "_context"):
            return
        try:
            _original_del(self)
        except AttributeError:
            pass

    mujoco.GLContext.__del__ = safe_del
    mujoco.GLContext.__del__patched = True


def probe_mujoco_backend(backend: str):
    """Attempt to create a tiny MuJoCo GL context to verify the backend works."""
    try:
        import mujoco
    except Exception as exc:
        return False, f"Unable to import mujoco to probe backend: {exc}"

    patch_mujoco_glcontext_del()

    previous_backend = os.environ.get("MUJOCO_GL")
    if backend:
        os.environ["MUJOCO_GL"] = backend

    try:
        ctx = mujoco.GLContext(1, 1)
        ctx.make_current()
        ctx.free()
        return True, ""
    except Exception as exc:
        msg = str(exc)
        if "Permission denied" in msg:
            msg += "\nThe current user may not have access to /dev/dri render nodes. " \
                   "Grant access or use --mujoco-gl osmesa."
        return False, msg
    finally:
        if previous_backend is not None:
            os.environ["MUJOCO_GL"] = previous_backend
        else:
            os.environ.pop("MUJOCO_GL", None)


def main(
    total_timesteps: int = 10000,
    env_id: str = None,
    render: bool = False,
    save_video: str = None,
    mujoco_gl: str = "auto",
    force_mujoco_gl: bool = False,
):
    backend = configure_mujoco_gl(mujoco_gl, force_mujoco_gl)
    patch_mujoco_glcontext_del()
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
    mujo_env_candidates = [
        "HalfCheetah-v5",
        "HalfCheetah-v4",
        "Walker2d-v5",
        "Walker2d-v4",
        "Ant-v5",
        "Ant-v4",
        "Humanoid-v5",
        "Humanoid-v4",
    ]

    def select_mujoco_env(candidate_env_id: str):
        env_obj = None
        chosen_env = candidate_env_id
        if candidate_env_id is None:
            for cand in mujo_env_candidates:
                try:
                    env_obj = gym.make(cand)
                    chosen_env = cand
                    print(f"Using MuJoCo environment: {cand}")
                    break
                except Exception as exc:
                    print(f"Failed to create environment '{cand}': {exc}")
                    env_obj = None
        else:
            try:
                env_obj = gym.make(candidate_env_id)
                chosen_env = candidate_env_id
            except Exception as exc:
                print(f"Failed to create environment '{candidate_env_id}': {exc}")
                env_obj = None

        return env_obj, chosen_env

    env, selected_env = select_mujoco_env(env_id)

    if env is None and backend == "egl" and not force_mujoco_gl:
        print("MuJoCo env creation failed with MUJOCO_GL=egl; retrying with MUJOCO_GL=osmesa for software rendering.")
        os.environ["MUJOCO_GL"] = "osmesa"
        backend = "osmesa"
        env, selected_env = select_mujoco_env(env_id)
        print("Pass --force-mujoco-gl to keep MUJOCO_GL=egl even if environment creation fails.")

    if env is None:
        if force_mujoco_gl:
            raise RuntimeError(
                f"Failed to create MuJoCo environment with forced MUJOCO_GL={os.environ.get('MUJOCO_GL')}\n"
                "Check your EGL/GLFW installation (see MUJOCO_LOG.TXT) or rerun without --force-mujoco-gl."
            )
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
    if (save_video or render) and backend == "egl" and not force_mujoco_gl:
        ok, err_msg = probe_mujoco_backend("egl")
        if not ok:
            print(
                "MuJoCo EGL probe failed before evaluation:"
                f" {err_msg}\nSwitching to MUJOCO_GL=osmesa for rendering."
            )
            backend = "osmesa"
            os.environ["MUJOCO_GL"] = backend

    eval_episodes = 1

    def evaluate_policy_with_backend(eval_backend: str):
        nonlocal env

        rewards_local = []
        frames_written_local = 0
        capture_warning_shown_local = False
        last_render_error_local = None
        eval_env_local = None
        writer_local = None
        eval_render_mode_local = None
        prev_backend = os.environ.get("MUJOCO_GL")

        if eval_backend:
            os.environ["MUJOCO_GL"] = eval_backend
            print(f"Evaluating policy with MUJOCO_GL={eval_backend}")

        try:
            if save_video:
                try:
                    import imageio
                except Exception:
                    raise RuntimeError("imageio is required to save videos. Install with `pip install imageio[ffmpeg]`")
                writer_local = imageio.get_writer(save_video, fps=30)
                eval_render_mode_local = 'rgb_array'
            elif render:
                eval_render_mode_local = 'human'

            if eval_render_mode_local is not None:
                try:
                    eval_env_local = gym.make(selected_env, render_mode=eval_render_mode_local)
                except TypeError:
                    eval_env_local = gym.make(selected_env)
                except Exception as exc:
                    last_render_error_local = exc
                    print(
                        "Failed to create evaluation environment with MUJOCO_GL="
                        f"{os.environ.get('MUJOCO_GL')}: {exc}"
                    )
                    return [], 0, last_render_error_local
                rollout_env_local = eval_env_local
            else:
                rollout_env_local = env

            for ep in range(eval_episodes):
                reset_ret = rollout_env_local.reset()
                if isinstance(reset_ret, tuple) and len(reset_ret) >= 1:
                    ob = reset_ret[0]
                else:
                    ob = reset_ret
                print("obs type:", type(ob), "shape:", getattr(np.asarray(ob), "shape", None), "dtype:", getattr(np.asarray(ob), "dtype", None))
                done = False
                ep_rew = 0.0
                while True:
                    try:
                        action, _ = model.predict(ob, deterministic=True)
                    except Exception:
                        action, _ = model.predict(np.array([ob]), deterministic=True)
                        if isinstance(action, (list, tuple, np.ndarray)) and getattr(action, 'shape', None) and action.shape[0] == 1:
                            action = action[0]

                    step_ret = rollout_env_local.step(action)
                    if len(step_ret) == 4:
                        ob, r, done, info = step_ret
                    elif len(step_ret) == 5:
                        ob, r, terminated, truncated, info = step_ret
                        done = bool(terminated or truncated)
                    else:
                        raise RuntimeError(f"Unexpected env.step() return shape: {type(step_ret)} len={len(step_ret)}")

                    ep_rew += float(r)

                    if render:
                        try:
                            rollout_env_local.render()
                        except Exception:
                            pass

                    if save_video and writer_local is not None:
                        frame = None
                        render_error = None
                        try:
                            frame = rollout_env_local.render()
                        except TypeError as exc:
                            render_error = exc
                            # Some environments disallow positional args / modes when render_mode already set.
                            frame = None
                        except Exception as exc:
                            render_error = exc
                            frame = None

                        if frame is None:
                            # Try MuJoCo's low-level renderer, available on mujoco environments.
                            renderer = getattr(rollout_env_local, "mujoco_renderer", None)
                            if renderer is None and hasattr(rollout_env_local, "unwrapped"):
                                renderer = getattr(rollout_env_local.unwrapped, "mujoco_renderer", None)

                            if renderer is None and hasattr(rollout_env_local, "unwrapped"):
                                renderer = getattr(rollout_env_local.unwrapped, "renderer", None) or renderer

                            if renderer is not None:
                                try:
                                    frame = renderer.render('rgb_array')
                                except TypeError:
                                    render_error = render_error or TypeError("renderer.render('rgb_array') unsupported")
                                    try:
                                        frame = renderer.render()
                                    except Exception as inner_exc:
                                        render_error = inner_exc
                                        frame = None
                                except Exception as inner_exc:
                                    render_error = inner_exc
                                    frame = None

                        if frame is None:
                            if not capture_warning_shown_local:
                                msg = "Warning: env.render() returned None; unable to capture video frame."
                                if render_error is not None:
                                    msg += f" Last error: {render_error}"
                                    last_render_error_local = render_error
                                msg += f" Active MUJOCO_GL={os.environ.get('MUJOCO_GL')}"
                                msg += ". Ensure the backend is installed (see MUJOCO_LOG.TXT for details)."
                                print(msg)
                                capture_warning_shown_local = True
                        else:
                            frm = np.asarray(frame)
                            if frm.dtype != np.uint8:
                                frm = np.clip(frm, 0, 255)
                                if frm.max() <= 1.0:
                                    frm = (frm * 255).astype(np.uint8)
                                else:
                                    frm = frm.astype(np.uint8)
                            writer_local.append_data(frm)
                            frames_written_local += 1

                    #if done:
                    break

                rewards_local.append(ep_rew)
                print(f"Eval episode {ep+1} reward: {ep_rew:.2f}")

            if rewards_local:
                print(f"Mean eval reward: {np.mean(rewards_local):.2f} (std {np.std(rewards_local):.2f})")

            return rewards_local, frames_written_local, last_render_error_local

        finally:
            if writer_local is not None:
                writer_local.close()
                if frames_written_local > 0:
                    print(f"Saved rollout video to {save_video}")
                else:
                    print(
                        "Warning: no video frames were captured with MUJOCO_GL="
                        f"{eval_backend}. Inspect MUJOCO_LOG.TXT and ensure the backend is available."
                    )

            if eval_env_local is not None:
                try:
                    eval_env_local.close()
                except Exception:
                    pass

            if prev_backend is not None:
                os.environ["MUJOCO_GL"] = prev_backend
            else:
                os.environ.pop("MUJOCO_GL", None)

    print('evaluation backend', backend)
    rewards, frames_written, last_render_error = evaluate_policy_with_backend(backend)

    if save_video and frames_written == 0 and backend == "egl" and not force_mujoco_gl:
        print("Retrying evaluation rollout with MUJOCO_GL=osmesa after EGL failure.")
        try:
            if os.path.exists(save_video):
                os.remove(save_video)
        except OSError:
            pass
        backend = "osmesa"
        rewards, frames_written, last_render_error = evaluate_policy_with_backend("osmesa")
        if backend:
            os.environ["MUJOCO_GL"] = backend

    if save_video and frames_written == 0 and force_mujoco_gl and last_render_error is not None:
        raise RuntimeError(
            "MuJoCo rendering failed even though --force-mujoco-gl was set.\n"
            f"Last error: {last_render_error}\n"
            "Inspect MUJOCO_LOG.TXT for backend diagnostics."
        )

    # Close evaluation and training envs
    try:
        env.close()
    except Exception:
        pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=10000)
    p.add_argument('--env', type=str, default=None)
    p.add_argument('--render', action='store_true')
    p.add_argument('--save-video', type=str, default=None,
                   help='Path to save evaluation rollout mp4 (e.g. outputs/rollout.mp4)')
    p.add_argument('--mujoco-gl', type=str, default='auto', choices=['auto', 'egl', 'osmesa', 'glfw'],
                   help='OpenGL backend for MuJoCo (auto picks egl without DISPLAY, otherwise glfw).')
    p.add_argument('--force-mujoco-gl', action='store_true',
                   help='If set, do not fallback to another backend when MuJoCo creation fails.')
    args = p.parse_args()
    main(
        total_timesteps=args.timesteps,
        env_id=args.env,
        render=args.render,
        save_video=args.save_video,
        mujoco_gl=args.mujoco_gl,
        force_mujoco_gl=args.force_mujoco_gl,
    )
