#!/usr/bin/env bash
set -euo pipefail

# Quick demo: train for a few epochs, evaluate, then run server in background
python -m lerobot.cli train --config configs/config.yaml
python -m lerobot.cli eval --ckpt outputs/checkpoint.pt --config configs/config.yaml
# Note: server will block — run in background if you want
# python -m lerobot.cli deploy --ckpt outputs/checkpoint.pt --host 127.0.0.1 --port 8000

echo "Demo finished. Check outputs/ for checkpoints." 
