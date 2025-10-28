"""MuJoCo RL example with PPO or SAC (falls back to CartPole if MuJoCo not available).

This script trains a PPO agent using Stable-Baselines3. If MuJoCo envs are not available
on the host, it falls back to CartPole-v1 so you can test the training loop.

Usage:
    python examples/mujoco_rl_example.py --timesteps 50000 --algo ppo
    python examples/mujoco_rl_example.py --timesteps 50000 --algo sac --policy transformer
"""
import os
import argparse
from pathlib import Path

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


def patch_offscreen_viewer_cleanup():
    """Guard OffScreenViewer.free/__del__ against partially initialized contexts."""
    module = None
    try:
        import gymnasium.envs.mujoco.mujoco_rendering as module  # type: ignore
    except Exception:
        try:
            import gym.envs.mujoco.mujoco_rendering as module  # type: ignore
        except Exception:
            return

    OffScreenViewer = getattr(module, "OffScreenViewer", None)
    if OffScreenViewer is None:
        return

    if getattr(OffScreenViewer, "__cleanup_patched", False):
        return

    original_free = getattr(OffScreenViewer, "free", None)
    if original_free is not None:
        def safe_free(self, _original_free=original_free):
            if not hasattr(self, "opengl_context") or self.opengl_context is None:
                return
            try:
                _original_free(self)
            except AttributeError:
                pass

        OffScreenViewer.free = safe_free

    original_del = getattr(OffScreenViewer, "__del__", None)
    if original_del is not None:
        def safe_del(self, _original_del=original_del):
            if not hasattr(self, "opengl_context") or self.opengl_context is None:
                return
            try:
                _original_del(self)
            except AttributeError:
                pass

        OffScreenViewer.__del__ = safe_del

    OffScreenViewer.__cleanup_patched = True


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


def resolve_episode_video_path(base_path: str, episode_index: int) -> Path:
    """Return a unique video path for an evaluation episode."""
    if not base_path:
        raise ValueError("Base path for saving videos must be provided.")

    expanded = os.path.expanduser(base_path)
    original = Path(expanded)

    if base_path.endswith(os.sep):
        directory = original
        stem = "episode"
        suffix = ".mp4"
    else:
        suffix = original.suffix
        if suffix:
            directory = original.parent if original.parent != Path("") else Path(".")
            stem = original.stem
        else:
            directory = original.parent if original.parent != Path("") else Path(".")
            stem = original.name or "episode"
            suffix = ".mp4"

    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{stem}_ep{episode_index + 1:04d}{suffix}"


def resolve_evaluation_video_base(base_path: str, timestep: int) -> str:
    """Return a video base path that encodes the training timestep."""
    expanded = os.path.expanduser(base_path)
    original = Path(expanded)

    step_fragment = f"step{timestep:07d}"

    if base_path.endswith(os.sep):
        directory = original
        filename = f"rollout_{step_fragment}.mp4"
        return str(directory / filename)

    suffix = original.suffix
    if suffix:
        directory = original.parent if original.parent != Path("") else Path(".")
        stem = original.stem
        return str(directory / f"{stem}_{step_fragment}{suffix}")

    directory = original.parent if original.parent != Path("") else Path(".")
    stem = original.name or "rollout"
    return str(directory / f"{stem}_{step_fragment}.mp4")


def main(
    total_timesteps: int = 10000,
    env_id: str = None,
    render: bool = False,
    save_video: str = None,
    eval_interval: int = 10000,
    load_checkpoint: str = None,
    policy_type: str = "mlp",
    transformer_seq_len: int = 8,
    transformer_d_model: int = 128,
    transformer_nhead: int = 4,
    transformer_layers: int = 2,
    transformer_dropout: float = 0.1,
    algo: str = "ppo",
    mujoco_gl: str = "auto",
    force_mujoco_gl: bool = False,
):
    backend = configure_mujoco_gl(mujoco_gl, force_mujoco_gl)
    patch_mujoco_glcontext_del()
    patch_offscreen_viewer_cleanup()
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

    algo_choice = (algo or "ppo").strip().lower()
    if algo_choice not in {"ppo", "sac"}:
        raise ValueError(f"Unsupported algo '{algo}'. Choose from ['ppo', 'sac'].")

    policy_choice = (policy_type or "mlp").strip().lower()
    if policy_choice not in {"mlp", "transformer"}:
        raise ValueError(f"Unsupported policy_type '{policy_type}'. Choose from ['mlp', 'transformer'].")

    use_transformer = policy_choice == "transformer"
    HistoryEnvWrapper = None
    TransformerFeatureExtractor = None
    policy_kwargs = {}
    policy_class = "MlpPolicy"

    if use_transformer:
        if transformer_seq_len <= 0:
            raise ValueError("transformer_seq_len must be positive when using transformer policy.")
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError(
                "Transformer policy requires PyTorch. Install it with `pip install torch`."
            ) from exc

        from stable_baselines3.common.policies import ActorCriticPolicy
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

        class HistoryEnvWrapper(gym.Wrapper):
            def __init__(self, env, seq_len: int):
                super().__init__(env)
                self.seq_len = int(seq_len)
                if self.seq_len <= 0:
                    raise ValueError("seq_len must be positive for HistoryEnvWrapper")
                if not isinstance(env.observation_space, gym.spaces.Box):
                    raise TypeError(
                        "Transformer policy currently supports environments with Box observation spaces only."
                    )
                self._obs_space = env.observation_space
                self._obs_dim = int(np.prod(self._obs_space.shape))
                self._action_space = env.action_space
                self._action_dim = self._infer_action_dim()
                self._combined_dim = self._obs_dim + self._action_dim
                low = np.full((self.seq_len * self._combined_dim,), -np.inf, dtype=np.float32)
                high = np.full_like(low, np.inf)
                self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
                self.action_space = env.action_space
                self._obs_hist = np.zeros((self.seq_len, self._obs_dim), dtype=np.float32)
                self._act_hist = np.zeros((self.seq_len, self._action_dim), dtype=np.float32)

            def _infer_action_dim(self) -> int:
                sample = self._action_space.sample()
                flat = self._flatten_action(sample)
                return int(flat.shape[0])

            def _flatten_obs(self, obs) -> np.ndarray:
                return np.asarray(obs, dtype=np.float32).reshape(-1)

            def _flatten_action(self, action) -> np.ndarray:
                space = self._action_space
                if isinstance(space, gym.spaces.Box):
                    return np.asarray(action, dtype=np.float32).reshape(-1)
                if isinstance(space, gym.spaces.Discrete):
                    vec = np.zeros(space.n, dtype=np.float32)
                    vec[int(action)] = 1.0
                    return vec
                if isinstance(space, (gym.spaces.MultiBinary, gym.spaces.MultiDiscrete)):
                    return np.asarray(action, dtype=np.float32).reshape(-1)
                raise TypeError(
                    f"Unsupported action space for HistoryEnvWrapper: {space}"
                )

            def _reset_buffers(self, obs) -> None:
                self._obs_hist.fill(0.0)
                self._act_hist.fill(0.0)
                self._obs_hist[-1] = self._flatten_obs(obs)

            def _push(self, obs, action_vec) -> None:
                self._obs_hist = np.roll(self._obs_hist, shift=-1, axis=0)
                self._act_hist = np.roll(self._act_hist, shift=-1, axis=0)
                self._obs_hist[-1] = self._flatten_obs(obs)
                self._act_hist[-1] = action_vec.astype(np.float32)

            def _current_observation(self) -> np.ndarray:
                stacked = np.concatenate((self._obs_hist, self._act_hist), axis=1)
                return stacked.reshape(-1).astype(np.float32)

            def reset(self, **kwargs):  # type: ignore[override]
                result = self.env.reset(**kwargs)
                if isinstance(result, tuple) and len(result) == 2:
                    obs, info = result
                    self._reset_buffers(obs)
                    return self._current_observation(), info
                obs = result
                self._reset_buffers(obs)
                return self._current_observation()

            def step(self, action):  # type: ignore[override]
                action_vec = self._flatten_action(action)
                result = self.env.step(action)
                if len(result) == 4:
                    obs, reward, done, info = result
                    self._push(obs, action_vec)
                    return self._current_observation(), reward, done, info
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    self._push(obs, action_vec)
                    return self._current_observation(), reward, terminated, truncated, info
                raise RuntimeError("Unexpected number of return values from env.step")

        class TransformerFeatureExtractor(BaseFeaturesExtractor):
            def __init__(
                self,
                observation_space,
                seq_len: int,
                d_model: int,
                nhead: int,
                num_layers: int,
                dropout: float,
            ):
                super().__init__(observation_space, features_dim=d_model)
                total_dim = int(np.prod(observation_space.shape))
                if total_dim % seq_len != 0:
                    raise ValueError(
                        f"Observation dimension {total_dim} is not divisible by seq_len {seq_len}."
                    )
                self.seq_len = seq_len
                self.step_dim = total_dim // seq_len
                self.input_projection = nn.Linear(self.step_dim, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=max(4 * d_model, d_model),
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.positional_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
                nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)
                self.layer_norm = nn.LayerNorm(d_model)

            def forward(self, observations):
                batch_size = observations.shape[0]
                seq = observations.view(batch_size, self.seq_len, self.step_dim)
                x = self.input_projection(seq)
                x = x + self.positional_embedding[:, : self.seq_len, :]
                x = self.transformer(x)
                x = self.layer_norm(x[:, -1, :])
                return x

        policy_kwargs = {
            "features_extractor_class": TransformerFeatureExtractor,
            "features_extractor_kwargs": {
                "seq_len": transformer_seq_len,
                "d_model": transformer_d_model,
                "nhead": transformer_nhead,
                "num_layers": transformer_layers,
                "dropout": transformer_dropout,
            },
        }

    print(f"Selected algorithm: {algo_choice}")
    print(f"Selected policy type: {policy_choice}")

    # try to select a MuJoCo env
    mujo_env_candidates = [
        "HalfCheetah-v5",
        "HalfCheetah-v4",
        "Walker2d-v5",
        "Walker2d-v4",
        "Ant-v5",
        "Ant-v4",
        "Humanoid-v4",
        "Humanoid-v5",
    ]

    def select_mujoco_env(candidate_env_id: str):
        env_obj = None
        chosen_env = candidate_env_id
        if candidate_env_id is None:
            for cand in mujo_env_candidates:
                try:
                    env_obj = gym.make(cand)
                    if use_transformer:
                        env_obj = HistoryEnvWrapper(env_obj, transformer_seq_len)
                    chosen_env = cand
                    print(f"Using MuJoCo environment: {cand}")
                    break
                except Exception as exc:
                    print(f"Failed to create environment '{cand}': {exc}")
                    env_obj = None
        else:
            try:
                env_obj = gym.make(candidate_env_id)
                if use_transformer:
                    env_obj = HistoryEnvWrapper(env_obj, transformer_seq_len)
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
        if use_transformer:
            env = HistoryEnvWrapper(env, transformer_seq_len)

    # Stable Baselines3
    try:
        from stable_baselines3 import PPO, SAC
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
    def _make_training_env():
        base_env = gym.make(selected_env)
        if use_transformer:
            base_env = HistoryEnvWrapper(base_env, transformer_seq_len)
        return base_env

    try:
        vec_env = make_vec_env(_make_training_env, n_envs=1)
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
    action_dim = vec_env.action_space.shape[0] if getattr(vec_env.action_space, "shape", None) else None
    print("action_dim:", action_dim)

    if algo_choice == "sac" and not isinstance(vec_env.action_space, gym.spaces.Box):
        raise TypeError(
            "SAC requires a continuous (Box) action space. Choose a MuJoCo task with continuous actions."
        )

    algo_cls = PPO if algo_choice == "ppo" else SAC
    algo_label = algo_choice.upper()

    model_dir = os.path.join("outputs", "mujoco")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{algo_choice}_{selected_env.replace('/', '_')}.zip")

    checkpoint_path = None
    if load_checkpoint:
        checkpoint_path = os.path.expanduser(load_checkpoint)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        print(f"Loading {algo_label} checkpoint from {checkpoint_path}")
        model = algo_cls.load(checkpoint_path, env=vec_env)
        model.set_env(vec_env)
        print(f"Loaded model with num_timesteps={getattr(model, 'num_timesteps', 'unknown')}")
    else:
        model = algo_cls(policy_class, vec_env, verbose=1, policy_kwargs=policy_kwargs)

    initial_model_timesteps = int(getattr(model, "num_timesteps", 0) or 0)
    print(
        f"Training {algo_label} on {selected_env} for {total_timesteps} additional timesteps"
        + (f" (starting from {initial_model_timesteps})" if initial_model_timesteps else "")
    )

    if eval_interval <= 0:
        raise ValueError("eval_interval must be a positive integer")

    # evaluation
    backend = "osmesa"
    os.environ["MUJOCO_GL"] = backend
    #os.environ["PYOPENGL_PLATFORM"] = backend

    eval_episodes = 1
    current_eval_backend = backend
    all_saved_video_paths = []
    last_render_error_global = None
    evaluation_history = []

    def evaluate_policy_with_backend(eval_backend: str, video_path_base: str = None):
        nonlocal env

        rewards_local = []
        frames_written_local = 0
        capture_warning_shown_local = False
        last_render_error_local = None
        eval_env_local = None
        eval_render_mode_local = None
        saved_video_paths_local = []
        imageio_module = None
        prev_backend = os.environ.get("MUJOCO_GL")

        if eval_backend:
            os.environ["MUJOCO_GL"] = eval_backend
            print(f"Evaluating policy with MUJOCO_GL={eval_backend}")

        try:
            if video_path_base:
                try:
                    import imageio
                    imageio_module = imageio
                except Exception:
                    raise RuntimeError("imageio is required to save videos. Install with `pip install imageio[ffmpeg]`")
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
                    return [], 0, last_render_error_local, []
                if use_transformer:
                    eval_env_local = HistoryEnvWrapper(eval_env_local, transformer_seq_len)
                rollout_env_local = eval_env_local
            else:
                rollout_env_local = env

            for ep in range(eval_episodes):
                writer_episode = None
                episode_video_path = None
                frames_written_episode = 0
                try:
                    reset_ret = rollout_env_local.reset()
                    if isinstance(reset_ret, tuple) and len(reset_ret) >= 1:
                        ob = reset_ret[0]
                    else:
                        ob = reset_ret
                    print(
                        "obs type:",
                        type(ob),
                        "shape:",
                        getattr(np.asarray(ob), "shape", None),
                        "dtype:",
                        getattr(np.asarray(ob), "dtype", None),
                    )
                    done = False
                    ep_rew = 0.0
                    while True:
                        try:
                            action, _ = model.predict(ob, deterministic=True)
                        except Exception:
                            action, _ = model.predict(np.array([ob]), deterministic=True)
                            if (
                                isinstance(action, (list, tuple, np.ndarray))
                                and getattr(action, "shape", None)
                                and action.shape[0] == 1
                            ):
                                action = action[0]

                        step_ret = rollout_env_local.step(action)
                        if len(step_ret) == 4:
                            ob, r, done, info = step_ret
                        elif len(step_ret) == 5:
                            ob, r, terminated, truncated, info = step_ret
                            done = bool(terminated or truncated)
                        else:
                            raise RuntimeError(
                                f"Unexpected env.step() return shape: {type(step_ret)} len={len(step_ret)}"
                            )

                        ep_rew += float(r)

                        if render:
                            try:
                                rollout_env_local.render()
                            except Exception:
                                pass

                        if video_path_base and imageio_module is not None:
                            if writer_episode is None:
                                try:
                                    episode_video_path = resolve_episode_video_path(video_path_base, ep)
                                except Exception as exc:
                                    raise RuntimeError(
                                        f"Failed to prepare video path for episode {ep + 1}: {exc}"
                                    ) from exc
                                try:
                                    writer_episode = imageio_module.get_writer(
                                        str(episode_video_path), fps=30
                                    )
                                except Exception as exc:
                                    raise RuntimeError(
                                        f"Unable to open video writer for {episode_video_path}: {exc}"
                                    ) from exc

                            frame = None
                            render_error = None
                            try:
                                frame = rollout_env_local.render()
                            except TypeError as exc:
                                render_error = exc
                                frame = None
                            except Exception as exc:
                                render_error = exc
                                frame = None

                            if frame is None:
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
                                writer_episode.append_data(frm)
                                frames_written_episode += 1

                        if done:
                            break

                    rewards_local.append(ep_rew)
                    print(f"Eval episode {ep+1} reward: {ep_rew:.2f}")
                finally:
                    if writer_episode is not None:
                        try:
                            writer_episode.close()
                        except Exception:
                            pass

                    if video_path_base and episode_video_path is not None:
                        if frames_written_episode > 0:
                            frames_written_local += frames_written_episode
                            saved_video_paths_local.append(str(episode_video_path))
                            print(
                                f"Saved rollout video for episode {ep + 1} to {episode_video_path}"
                            )
                        else:
                            if episode_video_path.exists():
                                try:
                                    os.remove(episode_video_path)
                                except OSError:
                                    pass
                            print(
                                "Warning: no frames captured for video"
                                f" {episode_video_path} with MUJOCO_GL={os.environ.get('MUJOCO_GL')}"
                            )

            if rewards_local:
                print(f"Mean eval reward: {np.mean(rewards_local):.2f} (std {np.std(rewards_local):.2f})")

            return rewards_local, frames_written_local, last_render_error_local, saved_video_paths_local

        finally:
            if video_path_base and frames_written_local == 0:
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

    def run_evaluation(video_path_base: str, trained_steps: int):
        nonlocal backend, current_eval_backend, last_render_error_global, all_saved_video_paths

        rewards_local, frames_written_local, last_render_error_local, saved_paths_local = (
            evaluate_policy_with_backend(current_eval_backend, video_path_base)
        )

        if (
            video_path_base
            and frames_written_local == 0
            and current_eval_backend == "egl"
            and not force_mujoco_gl
        ):
            print("Retrying evaluation rollout with MUJOCO_GL=osmesa after EGL failure.")
            try:
                for candidate in saved_paths_local:
                    if os.path.exists(candidate):
                        os.remove(candidate)
            except OSError:
                pass
            current_eval_backend = "osmesa"
            backend = current_eval_backend
            rewards_local, frames_written_local, last_render_error_local, saved_paths_local = (
                evaluate_policy_with_backend(current_eval_backend, video_path_base)
            )
            if current_eval_backend:
                os.environ["MUJOCO_GL"] = current_eval_backend

        if video_path_base and frames_written_local == 0 and last_render_error_local is not None:
            if force_mujoco_gl:
                raise RuntimeError(
                    "MuJoCo rendering failed even though --force-mujoco-gl was set.\n"
                    f"Last error: {last_render_error_local}\n"
                    "Inspect MUJOCO_LOG.TXT for backend diagnostics."
                )
            raise RuntimeError(
                "Unable to capture video frames because both EGL and OSMesa renderers failed.\n"
                f"MuJoCo reported: {last_render_error_local}\n"
                "Install headless rendering support (e.g. mesa-libOSMesa-dev / libosmesa6) or run with a working EGL GPU setup."
            )

        if saved_paths_local:
            all_saved_video_paths.extend(saved_paths_local)

        last_render_error_global = last_render_error_local

        evaluation_history.append(
            {
                "steps": trained_steps,
                "episodes": len(rewards_local),
                "mean_reward": float(np.mean(rewards_local)) if rewards_local else None,
                "std_reward": float(np.std(rewards_local)) if len(rewards_local) > 1 else 0.0,
            }
        )

        return rewards_local, frames_written_local


    timesteps_trained = 0
    iteration = 0
    while timesteps_trained < total_timesteps:
        iteration += 1
        chunk = min(eval_interval, total_timesteps - timesteps_trained)
        target_total = initial_model_timesteps + total_timesteps
        upcoming_total = initial_model_timesteps + timesteps_trained + chunk
        print(
            f"Training iteration {iteration}: {chunk} timesteps (target {upcoming_total}/{target_total})"
        )
        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=(timesteps_trained == 0 and checkpoint_path is None),
        )
        timesteps_trained += chunk

        global_timesteps = initial_model_timesteps + timesteps_trained

        if save_video:
            video_base_for_eval = resolve_evaluation_video_base(save_video, global_timesteps)
        else:
            video_base_for_eval = None

        run_evaluation(video_base_for_eval, global_timesteps)

    model.save(model_path)
    print(f"Saved model to {model_path}")

    if save_video and all_saved_video_paths:
        print("Saved evaluation videos:")
        for path in all_saved_video_paths:
            print(f"  {path}")

    if evaluation_history:
        print("Evaluation checkpoints:")
        for entry in evaluation_history:
            steps = entry["steps"]
            mean_reward = entry["mean_reward"]
            std_reward = entry["std_reward"]
            episodes = entry["episodes"]
            if mean_reward is not None:
                print(
                    f"  steps={steps}: mean_reward={mean_reward:.2f} (std {std_reward:.2f}) over {episodes} episodes"
                )
            else:
                print(f"  steps={steps}: no rewards recorded over {episodes} episodes")

    # Close evaluation and training envs
    try:
        env.close()
    except Exception:
        pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=10000)
    p.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'sac'],
                   help='RL algorithm to train (ppo = on-policy, sac = off-policy).')
    p.add_argument('--eval-interval', type=int, default=10000,
                   help='Number of training timesteps between evaluation rollouts and video capture.')
    p.add_argument('--env', type=str, default=None)
    p.add_argument('--render', action='store_true')
    p.add_argument('--save-video', type=str, default=None,
                   help='Path to save evaluation rollout mp4 (e.g. outputs/rollout.mp4)')
    p.add_argument('--policy', type=str, default='mlp', choices=['mlp', 'transformer'],
                   help='Policy architecture to use for PPO training.')
    p.add_argument('--transformer-seq-len', type=int, default=8,
                   help='Sequence length (number of past steps) for transformer policy inputs.')
    p.add_argument('--transformer-d-model', type=int, default=128,
                   help='Transformer model dimension for the custom policy.')
    p.add_argument('--transformer-nhead', type=int, default=4,
                   help='Number of attention heads in the transformer policy.')
    p.add_argument('--transformer-layers', type=int, default=2,
                   help='Number of transformer encoder layers in the policy.')
    p.add_argument('--transformer-dropout', type=float, default=0.1,
                   help='Dropout rate applied inside the transformer encoder.')
    p.add_argument('--load-checkpoint', type=str, default=None,
                   help='Path to an existing PPO checkpoint to resume training from.')
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
        eval_interval=args.eval_interval,
        load_checkpoint=args.load_checkpoint,
        policy_type=args.policy,
        algo=args.algo,
        transformer_seq_len=args.transformer_seq_len,
        transformer_d_model=args.transformer_d_model,
        transformer_nhead=args.transformer_nhead,
        transformer_layers=args.transformer_layers,
        transformer_dropout=args.transformer_dropout,
        mujoco_gl=args.mujoco_gl,
        force_mujoco_gl=args.force_mujoco_gl,
    )
