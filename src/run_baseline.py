"""Run the MagnaTagATune MFCC + one-vs-rest Logistic Regression baseline.

From the repository root:
    python -m src.run_baseline --root C:/mtat_data --workers 8

Features are cached under <root>/baseline_features.npz so later runs are faster.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm

from src.baseline import mfcc_features, run_baseline
from src.datasets.mtat import TAGS_50, read_split_tables


def extract_one(row):
    import librosa
    wav, sr = librosa.load(row["path"], sr=16000, mono=True)
    feat = mfcc_features(wav, sr=sr, n_mfcc=20)
    return feat.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"C:\mtat_data")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--force", action="store_true", help="re-extract MFCC features")
    parser.add_argument("--output", default="experiments/baseline")
    args = parser.parse_args()

    root = Path(args.root)
    cache = root / "baseline_features.npz"
    rows = read_split_tables(root, drop_empty=True)
    if not rows:
        raise RuntimeError(f"No clips found under {root}. Check the MTAT dataset files.")

    # The cache is tied to the ordered clip IDs, preventing accidental reuse with a
    # different split/order.
    ids = np.asarray([r["clip_id"] for r in rows])
    if cache.exists() and not args.force:
        saved = np.load(cache, allow_pickle=False)
        if np.array_equal(saved["ids"], ids):
            X = saved["X"].astype(np.float32)
            print(f"Loaded cached MFCC features: {X.shape} from {cache}")
        else:
            print("Feature cache IDs differ; extracting features again.")
            X = None
    else:
        X = None

    if X is None:
        X = np.empty((len(rows), 40), dtype=np.float32)
        errors = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(extract_one, row): i for i, row in enumerate(rows)}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="MFCC"):
                i = futures[fut]
                try:
                    X[i] = fut.result()
                except Exception as exc:
                    errors.append((rows[i]["clip_id"], rows[i]["path"], repr(exc)))
        if errors:
            preview = "\n".join(f"{cid}: {err}" for cid, _, err in errors[:10])
            raise RuntimeError(f"MFCC extraction failed for {len(errors)} clips. First errors:\n{preview}")
        np.savez_compressed(cache, X=X, ids=ids)
        print(f"Saved MFCC feature cache: {cache}")

    Y = np.stack([r["labels"] for r in rows]).astype(np.float32)
    split = np.asarray([r["split"] for r in rows])
    tr, va, te = split == "train", split == "val", split == "test"
    print(f"clips: train={tr.sum()} val={va.sum()} test={te.sum()} | features={X.shape[1]}")

    result, model = run_baseline(X[tr], Y[tr], X[va], Y[va], X[te], Y[te], tag_names=TAGS_50)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "mfcc_logreg.joblib")
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== MFCC + Logistic Regression (test set) ===")
    for key in ("roc_auc_macro", "pr_auc_macro", "f1_macro", "best_C"):
        print(f"{key:16s} {result[key]:.4f}")
    print(f"\nSaved model and metrics to: {out.resolve()}")


if __name__ == "__main__":
    main()
