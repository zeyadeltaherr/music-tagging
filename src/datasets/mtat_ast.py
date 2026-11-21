"""MagnaTagATune dataset using official AST Fbank cache."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .mtat import TAGS_50


class MTATASTDataset(Dataset):

    def __init__(self, root, split):

        self.dir = Path(root) / "ast_fbank"

        meta = json.loads(
            (self.dir / "meta.json").read_text(encoding="utf-8")
        )

        self.labels = np.load(
            self.dir / "labels.npy"
        ).astype(np.float32)

        self.rows = np.array([
            i for i, s in enumerate(meta["split"])
            if s == split
        ])

        self.ids = np.asarray(meta["clip_ids"])[self.rows]
        self.names = np.asarray(meta["names"], dtype=object)[self.rows]
        self.songs = np.asarray(meta["songs"], dtype=object)[self.rows]

        self._mm = None

    def __len__(self):
        return len(self.rows)

    def _features(self):

        if self._mm is None:
            self._mm = np.load(
                self.dir / "fbank.npy",
                mmap_mode="r"
            )

        return self._mm

    def __getitem__(self, i):

        r = self.rows[i]

        x = np.asarray(
            self._features()[r],
            dtype=np.float32
        )

        y = self.labels[r]

        return (
            torch.from_numpy(x.copy()),
            torch.from_numpy(y.copy())
        )


def build_mtat_ast(cfg):

    root = cfg["root"]

    return {
        "train": MTATASTDataset(root, "train"),
        "val": MTATASTDataset(root, "val"),
        "test": MTATASTDataset(root, "test"),
        "n_tags": len(TAGS_50),
        "tag_names": TAGS_50,
    }