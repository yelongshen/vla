Quickstart - train, evaluate, deploy

1. Install deps

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Train

   python -m lerobot.cli train --config configs/config.yaml

3. Evaluate

   python -m lerobot.cli eval --ckpt outputs/checkpoint.pt

4. Serve

   python -m lerobot.cli deploy --ckpt outputs/checkpoint.pt --host 0.0.0.0 --port 8000

API

POST /predict
  body: { "inputs": [[...], [...]] }
  returns: { "predictions": [int, ...] }
