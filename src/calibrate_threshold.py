import json
import numpy as np
import torch
import yaml

from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.datasets import build_datasets
from src.models import build_model
from src.train import predict
from src.evaluate import tag_metrics


def main():
    config_path = "configs/pretrained_fbank.yaml"
    checkpoint_path = "checkpoints/ast_official_fbank_seed0.pt"

    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"

    print("Device:", device)
    print("Loading dataset...")

    data = build_datasets(cfg["data"])

    eval_bs = cfg["train"].get("eval_batch_size", cfg["train"]["batch_size"])

    val_loader = DataLoader(
        data["val"],
        batch_size=eval_bs,
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
        pin_memory=device.type == "cuda"
    )

    test_loader = DataLoader(
        data["test"],
        batch_size=eval_bs,
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
        pin_memory=device.type == "cuda"
    )

    print("Loading model...")

    model = build_model(
        cfg["model"]["name"],
        n_tags=data["n_tags"],
        **cfg["model"].get("params", {})
    ).to(device)

    # Materialize model layers before loading checkpoint.
    sample = next(iter(val_loader))[0]
    from src.train import first_chunk

    with torch.no_grad():
        model.eval()(first_chunk(sample)[:2].to(device))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("Predicting validation set...")
    val_scores, val_labels, _ = predict(
        model, val_loader, device, use_amp
    )

    thresholds = np.arange(0.01, 1.00, 0.01)

    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:
        predictions = (val_scores >= threshold).astype(int)

        score = f1_score(
            val_labels,
            predictions,
            average="macro",
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    val_default = f1_score(
        val_labels,
        (val_scores >= 0.5).astype(int),
        average="macro",
        zero_division=0
    )

    print("\n=== VALIDATION CALIBRATION ===")
    print(f"Default threshold: 0.50 | Macro-F1: {val_default:.4f}")
    print(f"Best threshold:    {best_threshold:.2f} | Macro-F1: {best_f1:.4f}")

    # Threshold is selected using validation only.
    # Test is evaluated after calibration is complete.
    print("\nPredicting test set...")
    test_scores, test_labels, _ = predict(
        model, test_loader, device, use_amp
    )

    test_default = tag_metrics(
        test_labels, test_scores, data["tag_names"], threshold=0.5
    )

    test_calibrated = tag_metrics(
        test_labels,
        test_scores,
        data["tag_names"],
        threshold=best_threshold
    )

    print("\n=== FINAL TEST COMPARISON ===")
    print(f"{'Metric':<18}{'Threshold 0.50':>18}{'Calibrated':>18}")

    for metric in ["roc_auc_macro", "pr_auc_macro", "f1_macro"]:
        print(
            f"{metric:<18}"
            f"{test_default[metric]:>18.4f}"
            f"{test_calibrated[metric]:>18.4f}"
        )

    print("\nSelected threshold:", best_threshold)

    output = {
        "checkpoint": checkpoint_path,
        "threshold_selection": "validation_macro_f1",
        "best_threshold": best_threshold,
        "validation": {
            "default_f1": float(val_default),
            "calibrated_f1": float(best_f1)
        },
        "test_default": {
            k: v for k, v in test_default.items() if k != "per_tag"
        },
        "test_calibrated": {
            k: v for k, v in test_calibrated.items() if k != "per_tag"
        },
        "test_per_tag_calibrated": test_calibrated["per_tag"]
    }

    Path("experiments").mkdir(exist_ok=True)

    with open(
        "experiments/threshold_calibration.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(output, f, indent=2)

    print("\nSaved: experiments/threshold_calibration.json")


if __name__ == "__main__":
    main()
