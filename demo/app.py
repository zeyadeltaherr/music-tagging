"""Gradio demo: upload a song -> predicted tags + most similar tracks.
    python demo/app.py --ckpt checkpoints/attn_cnn_full_seed0.pt --index checkpoints/index.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import gradio as gr
import numpy as np
import torch

from src.datasets.audio import MelFrontend, load_wav, make_chunks
from src.models import build_model
from src.similarity import SimilarityIndex

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--index", required=True)
ap.add_argument("--k", type=int, default=5)
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(args.ckpt, map_location=device)
cfg, tag_names = ck["cfg"], ck["tag_names"]
model = build_model(cfg["model"]["name"], n_tags=ck["n_tags"], **cfg["model"].get("params", {})).to(device)
with torch.no_grad():
    model.eval()(torch.zeros(2, 128, 231, device=device))   # materialise lazy layers
model.load_state_dict(ck["model_state"])
model.eval()
frontend = MelFrontend().to(device)
index = SimilarityIndex.load(args.index)


@torch.no_grad()
def analyse(path):
    if path is None:
        return {}, []
    chunks = torch.from_numpy(make_chunks(load_wav(path))).to(device)
    mel = frontend(chunks)
    emb = model.embed(mel).mean(dim=0, keepdim=True)          # average over chunks
    probs = torch.sigmoid(model.classify(emb))[0].cpu().numpy()
    tags = {tag_names[i]: float(probs[i]) for i in np.argsort(-probs)[:10]}
    idx, sc = index.search(emb.cpu().numpy(), k=args.k)
    rows = [[r + 1, str(index.ids[i]), round(float(s), 3)] for r, (i, s) in enumerate(zip(idx[0], sc[0]))]
    return tags, rows


demo = gr.Interface(
    fn=analyse,
    inputs=gr.Audio(type="filepath", label="Upload a song (mp3/wav)"),
    outputs=[gr.Label(num_top_classes=10, label="Predicted tags"),
             gr.Dataframe(headers=["rank", "similar track", "cosine"], label="Most similar tracks")],
    title="Music auto-tagging and similarity search",
    description="Deep learning model that predicts music tags and finds similar tracks.",
)

if __name__ == "__main__":
    demo.launch()
