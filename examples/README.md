MuJoCo RL Example
=================

This folder contains `mujoco_rl_example.py`, an example script that trains a PPO agent using
Stable-Baselines3. The script will attempt to use a MuJoCo environment (e.g. HalfCheetah-v4).
If MuJoCo is not available on your system, the script will fall back to `CartPole-v1` so you
can verify the training loop without MuJoCo binaries.

Running the example
-------------------

1. Install requirements (recommended in a virtualenv or conda env):

```bash
pip install -r requirements.txt
```

2. Train (example):

```bash
python examples/mujoco_rl_example.py --timesteps 50000
```

3. Force a specific environment (e.g. HalfCheetah):

```bash
python examples/mujoco_rl_example.py --env HalfCheetah-v4 --timesteps 100000
```

Notes
-----
- MuJoCo requires native binaries and (depending on the package) a license or specific installation steps.
 - Prefer installing `gymnasium` (actively maintained) instead of the legacy `gym`. See the migration guide:
   https://gymnasium.farama.org/introduction/migration_guide/
 - If you only want to run the fallback CartPole demo, installing `gymnasium` and `stable-baselines3[extra]` is sufficient.
 - Stable-Baselines3 relies on additional helper packages (e.g. `shimmy`) for certain vectorized env utilities.
   If you see `ModuleNotFoundError: No module named 'shimmy'`, install it with:

```bash
pip install shimmy
```
