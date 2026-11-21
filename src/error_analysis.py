import json
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from src.datasets import build_datasets
from src.models import build_model

CONFIG = "configs/pretrained_fbank.yaml"
CHECKPOINT = "checkpoints/ast_official_fbank_seed0.pt"
ROOT = Path("C:/mtat_data")
OUT = Path("experiments")
THRESHOLD = 0.5

def save_csv(path, rows):
    if not rows:
        print(f"No rows to save: {path}")
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True)

    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data = build_datasets(cfg["data"])
    test_ds = data["test"]

    loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model = build_model(
        cfg["model"]["name"],
        n_tags=data["n_tags"],
        **cfg["model"].get("params", {})
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    use_amp = device.type == "cuda" and cfg["train"].get("amp", True)

    scores_list = []
    labels_list = []

    print("Extracting test predictions...")

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):
                logits = model(x)

            scores_list.append(torch.sigmoid(logits.float()).cpu().numpy())
            labels_list.append(y.numpy())

            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(test_ds)}")

    scores = np.concatenate(scores_list, axis=0)
    labels = np.concatenate(labels_list, axis=0).astype(int)
    predictions = (scores >= THRESHOLD).astype(int)

    tag_names = checkpoint["tag_names"]

    meta = json.loads(
        (ROOT / "ast_fbank" / "meta.json").read_text(encoding="utf-8")
    )

    test_rows = test_ds.rows

    assert len(test_rows) == len(labels) == len(scores)
    assert len(tag_names) == scores.shape[1]

    per_tag = []
    false_positives = []
    false_negatives = []

    for j, tag in enumerate(tag_names):
        yt = labels[:, j]
        yp = predictions[:, j]
        ys = scores[:, j]

        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        roc = None
        ap = None

        if len(np.unique(yt)) == 2:
            roc = float(roc_auc_score(yt, ys))
            ap = float(average_precision_score(yt, ys))

        per_tag.append({
            "tag": tag,
            "support": int(yt.sum()),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc,
            "pr_auc": ap,
        })

        for i in np.where((yt == 0) & (yp == 1))[0]:
            r = int(test_rows[i])
            false_positives.append({
                "tag": tag,
                "score": float(ys[i]),
                "clip_id": meta["clip_ids"][r],
                "name": meta["names"][r],
                "song": meta["songs"][r],
            })

        for i in np.where((yt == 1) & (yp == 0))[0]:
            r = int(test_rows[i])
            false_negatives.append({
                "tag": tag,
                "score": float(ys[i]),
                "clip_id": meta["clip_ids"][r],
                "name": meta["names"][r],
                "song": meta["songs"][r],
            })

    per_tag.sort(key=lambda x: x["f1"])

    false_positives.sort(key=lambda x: x["score"], reverse=True)
    false_negatives.sort(key=lambda x: x["score"])

    save_csv(OUT / "error_analysis_per_tag.csv", per_tag)
    save_csv(OUT / "false_positives.csv", false_positives)
    save_csv(OUT / "false_negatives.csv", false_negatives)

    summary = {
        "model": "AST Official Fbank",
        "seed": 0,
        "threshold": THRESHOLD,
        "test_clips": len(labels),
        "total_false_positives": len(false_positives),
        "total_false_negatives": len(false_negatives),
        "worst_10_tags_by_f1": per_tag[:10],
        "best_10_tags_by_f1": per_tag[-10:][::-1],
    }

    (OUT / "error_analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n=== ERROR ANALYSIS COMPLETE ===")
    print("Test clips:", len(labels))
    print("False Positives:", len(false_positives))
    print("False Negatives:", len(false_negatives))

    print("\nWorst 10 tags by F1:")
    for row in per_tag[:10]:
        print(
            f"{row['tag']:20s} "
            f"F1={row['f1']:.4f} "
            f"FP={row['FP']} FN={row['FN']}"
        )

    print("\nSaved:")
    print(OUT / "error_analysis_per_tag.csv")
    print(OUT / "false_positives.csv")
    print(OUT / "false_negatives.csv")
    print(OUT / "error_analysis_summary.json")

if __name__ == "__main__":
    main()
