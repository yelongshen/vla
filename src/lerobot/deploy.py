"""FastAPI-based deployment server for lerobot models."""
import os
import json
import uvicorn
import torch
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from .model import build_model


class PredictRequest(BaseModel):
    inputs: List[List[float]]


def load_checkpoint(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def create_app(ckpt_path: str):
    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt.get("cfg") or {}

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"]) if "model_state" in ckpt else None
    model.eval()

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def predict(req: PredictRequest):
        import torch
        import numpy as np
        arr = np.array(req.inputs, dtype="float32")
        with torch.no_grad():
            tensor = torch.from_numpy(arr)
            logits = model(tensor)
            preds = logits.argmax(dim=1).numpy().tolist()
        return {"predictions": preds}

    return app


def serve(ckpt_path: str, host: str = "127.0.0.1", port: int = 8000):
    app = create_app(ckpt_path)
    uvicorn.run(app, host=host, port=port)
