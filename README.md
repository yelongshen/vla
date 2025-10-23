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

Contributing

This scaffold is intended as a starting point — extend models, datasets and metrics as needed.
