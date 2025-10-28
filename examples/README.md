Examples
========

This folder collects runnable demos for different simulators supported by the project.

MuJoCo RL Example (`mujoco_rl_example.py`)
-----------------------------------------

Trains a PPO or SAC agent using Stable-Baselines3. The script attempts to use a MuJoCo
environment (e.g. `HalfCheetah-v4`). If MuJoCo is not available on your system, the script falls
back to `CartPole-v1` so you can verify the training loop without MuJoCo binaries.

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


Omniverse Minimal Demo (`omniverse_minimal_demo.py`)
---------------------------------------------------

Bootstraps an Omniverse Isaac Sim session, drops a cube onto a ground plane, and runs the
simulation loop for a few seconds. This script must be executed with Isaac Sim's Python
interpreter because the Omniverse Kit libraries are not available in a stock Python install.

Quick start (inside an Omniverse Isaac Sim installation directory):

```bash
./python.sh examples/omniverse_minimal_demo.py           # with GUI
./python.sh examples/omniverse_minimal_demo.py --headless --steps 600
```

Troubleshooting tips:

- Ensure the `examples/` folder is accessible from the current Omniverse working directory.
- Pass `--headless` when running on a remote server without X11/Wayland.
- The script prints cube poses every 60 steps; increase `--steps` to watch longer simulations.
