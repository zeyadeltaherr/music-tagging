"""Synthetic spectrograms with a learnable signal: used ONLY to smoke-test the code
before the real dataset is downloaded. Never report results from it."""
import numpy as np
import torch
from torch.utils.data import Dataset


class DummyMelDataset(Dataset):
    def __init__(self, n, n_tags, n_mels, n_frames, seed):
        rng = np.random.RandomState(seed)
        self.y = (rng.rand(n, n_tags) < 0.25).astype(np.float32)
        self.x = (rng.randn(n, n_mels, n_frames) * 0.5).astype(np.float32)
        band = n_mels // n_tags
        for i in range(n):
            for j in np.where(self.y[i] > 0)[0]:
                self.x[i, j * band:(j + 1) * band, :] += 1.0   # tag j = energy in band j

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i])


def build_dummy(cfg):
    n_tags = cfg.get("n_tags", 10)
    n_mels, n_frames = cfg.get("n_mels", 128), cfg.get("n_frames", 96)
    return {
        "train": DummyMelDataset(cfg.get("n_train", 256), n_tags, n_mels, n_frames, 1),
        "val": DummyMelDataset(cfg.get("n_val", 96), n_tags, n_mels, n_frames, 2),
        "test": DummyMelDataset(cfg.get("n_test", 96), n_tags, n_mels, n_frames, 3),
        "n_tags": n_tags,
        "tag_names": [f"tag_{j}" for j in range(n_tags)],
    }
