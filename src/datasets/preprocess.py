"""One-off preprocessing: decode every clip once and store log-mel spectrograms in a
float16 memory-mapped array, so training never has to decode mp3 files again.

    python -m src.datasets.preprocess --root C:\\mtat_data

Creates  <root>/mel/mel.npy (N, 128, 1800) float16, labels.npy (N, 50), meta.json.
Takes roughly 10-30 minutes depending on the CPU. Needs ~10 GB of free disk space.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from src.datasets.audio import MelFrontend, load_wav
from src.datasets.mtat import read_split_tables

SECONDS = 28.8
WAV_SAMPLES = int(SECONDS * 16000)            # 460800
T_STORE = 1800                                # stored frames per clip


def decode(path):
    wav = load_wav(path)
    if len(wav) < WAV_SAMPLES:
        wav = np.pad(wav, (0, WAV_SAMPLES - len(wav)))
    return wav[:WAV_SAMPLES].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\mtat_data")
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0, help="debug: only process this many clips")
    args = ap.parse_args()

    root = Path(args.root)
    table = read_split_tables(root, drop_empty=True)         # list of dicts
    if args.limit:
        table = table[:args.limit]
    n = len(table)
    out_dir = root / "mel"
    out_dir.mkdir(exist_ok=True)
    mel_mm = np.lib.format.open_memmap(out_dir / "mel.npy", mode="w+",
                                       dtype=np.float16, shape=(n, 128, T_STORE))
    np.save(out_dir / "labels.npy", np.stack([r["labels"] for r in table]).astype(np.float32))
    meta = {"clip_ids": [r["clip_id"] for r in table], "split": [r["split"] for r in table],
            "names": [r["name"] for r in table], "songs": [r["song"] for r in table]}
    (out_dir / "meta.json").write_text(json.dumps(meta))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frontend = MelFrontend().to(device)
    print(f"{n} clips, {args.n_jobs} decoding workers, mel on {device}")
    t0 = time.time()
    block = 256
    with Parallel(n_jobs=args.n_jobs) as par:
        for start in tqdm(range(0, n, block), desc="blocks"):
            rows = table[start:start + block]
            wavs = par(delayed(decode)(r["path"]) for r in rows)
            for s in range(0, len(wavs), 32):
                x = torch.from_numpy(np.stack(wavs[s:s + 32])).to(device)
                m = frontend(x)[:, :, :T_STORE]
                mel_mm[start + s:start + s + len(m)] = m.cpu().numpy().astype(np.float16)
    mel_mm.flush()
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
