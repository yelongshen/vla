LeRobot - VLA model training, evaluation, and deployment

This repository provides a minimal but complete platform named "lerobot" to train, evaluate, and deploy VLA-style models (Vision-Language-Action) using PyTorch and FastAPI.

Quickstart

1. Create a virtualenv and install dependencies:

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Train a small demo model with the example config:

   python -m lerobot.cli train --config configs/config.yaml

3. Evaluate a saved checkpoint:

   python -m lerobot.cli eval --ckpt outputs/checkpoint.pt --config configs/config.yaml

4. Serve a trained model locally:

   python -m lerobot.cli deploy --ckpt outputs/checkpoint.pt --host 0.0.0.0 --port 8000

Project layout

- src/lerobot: Python package with training, evaluation and deployment code.
- configs/: example YAML configurations.
- outputs/: default folder for checkpoints and logs.

AI2-THOR demo

This repo includes a minimal AI2-THOR integration for a simple manipulation demo.

1. Install dependencies including AI2-THOR:

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Run the demo (this will start an AI2-THOR controller and run a small scripted episode):

   python examples/ai2thor_demo.py

Notes: AI2-THOR may download scenes on first run. If running on a headless server, ensure you have a virtual framebuffer (Xvfb) or run in an environment that provides GPU/GL context.

PyBullet stairs demo

This repository includes a minimal PyBullet-based stairs stepping demo. It spawns a simple URDF robot and a staircase made of boxes and runs a short random-policy episode.

1. Install dependencies (this will include pybullet):

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Run the PyBullet demo:

   python examples/pybullet_demo.py

Notes: The demo runs in DIRECT (headless) mode by default. To visualize, change `render=True` in `examples/pybullet_demo.py` or run the example on a machine with X/GL support.

Reachy2 simulation

If you have a Reachy2 URDF, you can run a simple Reachy demo. Provide the URDF path via the `REACHY_URDF` environment variable:

   REACHY_URDF=/path/to/reachy.urdf python examples/reachy_demo.py

The wrapper `src/lerobot/envs/reachy_env.py` exposes `reset()`, `step(action)` and `get_observation()`; actions are joint position targets for detected actuated joints.

Contributing

This scaffold is intended as a starting point — extend models, datasets and metrics as needed.
