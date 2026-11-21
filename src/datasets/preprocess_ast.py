
"""Official AST Fbank preprocessing for MagnaTagATune.

Creates:
    <root>/ast_fbank/fbank.npy   (N, 1024, 128), float16
    <root>/ast_fbank/labels.npy
    <root>/ast_fbank/meta.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
from joblib import Parallel, delayed
from tqdm import tqdm

from transformers import ASTFeatureExtractor

from src.datasets.mtat import read_split_tables

SR = 16000
WAV_SAMPLES = int(28.8 * SR)
MAX_LENGTH = 1024


def decode(path):
    """Decode audio and preserve waveform amplitude."""
    import librosa

    wav, _ = librosa.load(path, sr=SR, mono=True)

    if len(wav) < WAV_SAMPLES:
        wav = np.pad(wav, (0, WAV_SAMPLES - len(wav)))

    return wav[:WAV_SAMPLES].astype(np.float32)


def extract_fbank(wav):
    """Extract AST-compatible Kaldi filterbank features."""
    waveform = torch.from_numpy(wav).unsqueeze(0)

   fbank = kaldi.fbank(
    waveform,
    sample_frequency=SR,
    window_type="hanning",
    num_mel_bins=128,
)

    n_frames = fbank.shape[0]

    if n_frames < MAX_LENGTH:
        padding = torch.zeros(MAX_LENGTH - n_frames, 128)
        fbank = torch.cat([fbank, padding], dim=0)
    else:
        fbank = fbank[:MAX_LENGTH]

    return fbank.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"C:\mtat_data")
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)

    table = read_split_tables(root, drop_empty=True)

    if args.limit:
        table = table[:args.limit]

    n = len(table)

    out_dir = root / "ast_fbank"
    out_dir.mkdir(exist_ok=True)

    feature_extractor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )

    fbank_mm = np.lib.format.open_memmap(
        out_dir / "fbank.npy",
        mode="w+",
        dtype=np.float16,
        shape=(n, MAX_LENGTH, 128),
    )

    np.save(
        out_dir / "labels.npy",
        np.stack([r["labels"] for r in table]).astype(np.float32),
    )

    meta = {
        "clip_ids": [r["clip_id"] for r in table],
        "split": [r["split"] for r in table],
        "names": [r["name"] for r in table],
        "songs": [r["song"] for r in table],
    }

    (out_dir / "meta.json").write_text(json.dumps(meta))

    print(f"{n} clips | {args.n_jobs} decoding workers")
    print("Extracting official AST Kaldi Fbank features...")

    t0 = time.time()
    block = 128

    with Parallel(n_jobs=args.n_jobs) as parallel:
        for start in tqdm(range(0, n, block), desc="blocks"):
            rows = table[start:start + block]

            waveforms = parallel(
                delayed(decode)(r["path"]) for r in rows
            )

            features = [
                extract_fbank(wav)
                for wav in waveforms
            ]

            fbank_mm[start:start + len(features)] = np.stack(
                features
            ).astype(np.float16)

    fbank_mm.flush()

    print(
        f"Done in {(time.time() - t0) / 60:.1f} minutes"
    )
    print(f"Saved to: {out_dir}")
    print(f"Feature shape: {fbank_mm.shape}")
    print(f"Extractor normalization: mean={feature_extractor.mean}, std={feature_extractor.std}")


if __name__ == "__main__":
    main()
