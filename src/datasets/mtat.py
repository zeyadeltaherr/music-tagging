"""MagnaTagATune loader (top-50 tags, standard train/val/test split files).

Run a sanity check (split sizes, leakage across splits, tag names):
    python -m src.datasets.mtat --root C:\\mtat_data
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Standard MagnaTagATune top-50 tag order (as used by Choi et al. 2016 / Won et al. 2020).
# The sanity check below prints tags for a few clips so you can confirm the order.
TAGS_50 = ['guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 'rock', 'fast',
           'piano', 'ambient', 'beat', 'violin', 'vocal', 'synth', 'female', 'indian', 'opera', 'male',
           'singing', 'vocals', 'no vocals', 'harpsichord', 'loud', 'quiet', 'flute', 'woman',
           'male vocal', 'no vocal', 'pop', 'soft', 'sitar', 'solo', 'man', 'classic', 'choir', 'voice',
           'new age', 'dance', 'male voice', 'female vocal', 'beats', 'harp', 'cello', 'no voice',
           'weird', 'country', 'metal', 'female voice', 'choral']
N_FRAMES = 231                    # frames per training chunk (~3.69 s)


def _read_clip_info(root):
    df = pd.read_csv(Path(root) / "clip_info_final.csv", sep="\t", quoting=csv.QUOTE_NONE,
                     encoding="utf-8", encoding_errors="replace", dtype=str)
    df = df.apply(lambda c: c.astype(str).str.strip().str.strip('"'))
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df.set_index("clip_id")


def _read_gt(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cid, vec = line.split("\t", 1)
            out[cid.strip().strip('"')] = np.array(json.loads(vec), dtype=np.float32)
    return out


def _audio_index(root):
    """Map 'f/xxx.mp3' -> absolute path, whatever folder layout the zip produced."""
    idx = {}
    for p in (Path(root) / "audio").rglob("*.mp3"):
        idx[f"{p.parent.name}/{p.name}"] = str(p)
    return idx


def read_split_tables(root, drop_empty=True):
    """Return a list of rows (dicts) for all splits."""
    root = Path(root)
    info = _read_clip_info(root)
    by_path = {p: cid for cid, p in info["mp3_path"].items()}
    audio = _audio_index(root)
    print(f"[mtat] clip_info rows: {len(info)} | mp3 files found: {len(audio)}")
    rows = []
    for split, fname in [("train", "train_gt_mtt.tsv"), ("val", "val_gt_mtt.tsv"), ("test", "test_gt_mtt.tsv")]:
        for cid, lab in _read_gt(root / fname).items():
            if drop_empty and lab.sum() == 0:
                continue                      # standard protocol: keep clips with >= 1 top-50 tag
            if cid not in info.index:
                cid = by_path.get(cid, cid)
                if cid not in info.index:
                    continue
            r = info.loc[cid]
            path = audio.get(r["mp3_path"])
            if path is None:
                continue
            rows.append({"clip_id": cid, "split": split, "labels": lab, "path": path,
                         "name": f"{r['artist']} - {r['title']} [{cid}]",
                         "song": f"{r['artist']}|{r['album']}|{r['track_number']}"})
    return rows


class MTATDataset(Dataset):
    """Reads pre-computed mel spectrograms (see preprocess.py).
    train: random chunk of 231 frames; val/test: n_eval_chunks chunks covering the clip
    (item shape (n_chunks, 128, 231)), averaged by train.predict()."""

    def __init__(self, root, split, train, n_frames=N_FRAMES, n_eval_chunks=7, crops_per_clip=1):
        self.dir = Path(root) / "mel"
        meta = json.loads((self.dir / "meta.json").read_text())
        labels = np.load(self.dir / "labels.npy")
        self.rows = np.array([i for i, s in enumerate(meta["split"]) if s == split])
        self.labels = labels[self.rows].astype(np.float32)
        self.ids = [meta["names"][i] for i in self.rows]
        self.train, self.n_frames, self.n_chunks = train, n_frames, n_eval_chunks
        self.crops = crops_per_clip if train else 1
        self._mm = None

    def __len__(self):
        return len(self.rows) * self.crops

    def _mel(self):
        if self._mm is None:
            self._mm = np.load(self.dir / "mel.npy", mmap_mode="r")
        return self._mm

    def __getitem__(self, i):
        i = i % len(self.rows)
        mm, r, nf = self._mel(), self.rows[i], self.n_frames
        total = mm.shape[2]
        if self.train:
            s = random.randint(0, total - nf)
            x = mm[r, :, s:s + nf]
        else:
            off = (total - self.n_chunks * nf) // 2
            x = np.stack([mm[r, :, off + k * nf: off + (k + 1) * nf] for k in range(self.n_chunks)])
        return torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.from_numpy(self.labels[i])


def build_mtat(cfg):
    root = cfg["root"]
    kw = dict(n_eval_chunks=cfg.get("n_eval_chunks", 7))
    return {
        "train": MTATDataset(root, "train", True, crops_per_clip=cfg.get("crops_per_clip", 1), **kw),
        "val": MTATDataset(root, "val", False, **kw),
        "test": MTATDataset(root, "test", False, **kw),
        "n_tags": len(TAGS_50),
        "tag_names": TAGS_50,
    }


def sanity_check(root):
    root = Path(root)
    rows = read_split_tables(root, drop_empty=True)
    print("clips per split (clips with >=1 top-50 tag):",
          {s: sum(r["split"] == s for r in rows) for s in ["train", "val", "test"]})
    songs = defaultdict(set)
    for r in rows:
        songs[r["song"]].add(r["split"])
    leaked = [s for s, sp in songs.items() if len(sp) > 1]
    print(f"songs: {len(songs)}  | songs appearing in more than one split: {len(leaked)}")
    if leaked:
        print("  examples:", leaked[:3])
    ann = pd.read_csv(root / "annotations_final.csv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str)
    ann.columns = [c.strip('"') for c in ann.columns]
    ann = ann.apply(lambda c: c.astype(str).str.strip().str.strip('"')).set_index("clip_id")
    print("\nTag-order check (our tag names vs raw annotation columns that are 1):")
    for r in [x for x in rows if x["split"] == "test"][:4]:
        ours = [TAGS_50[j] for j in np.where(r["labels"] > 0)[0]]
        raw = [c for c in ann.columns if c != "mp3_path" and ann.loc[r["clip_id"], c] == "1"]
        print(f"  clip {r['clip_id']}: ours={ours}\n            raw ={raw}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\mtat_data")
    sanity_check(ap.parse_args().root)
