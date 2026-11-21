"""Compute embeddings for a split with a trained checkpoint and save a similarity index.
    python -m src.build_index --ckpt checkpoints/attn_cnn_full_seed0.pt --split test --out checkpoints/index.npz
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import build_datasets
from src.models import build_model
from src.similarity import SimilarityIndex
from src.train import first_chunk, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", default="checkpoints/index.npz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck["cfg"]
    data = build_datasets(cfg["data"])
    model = build_model(cfg["model"]["name"], n_tags=ck["n_tags"], **cfg["model"].get("params", {})).to(device)
    ds = data[args.split]
    loader = DataLoader(ds, batch_size=8)
    with torch.no_grad():
        model.eval()(first_chunk(next(iter(loader))[0])[:2].to(device))
    model.load_state_dict(ck["model_state"])
    _, _, emb = predict(model, loader, device, use_amp=device.type == "cuda")
    ids = getattr(ds, "ids", np.arange(len(emb)))     # real dataset should expose clip ids/names
    SimilarityIndex().fit(emb, ids).save(args.out)
    print(f"saved index with {len(emb)} tracks -> {args.out}")


if __name__ == "__main__":
    main()
