"""Training / evaluation driver shared by every deep model.

Usage (from the project root):
    python -m src.train --config configs/cnn.yaml
    python -m src.train --config configs/dummy_smoke.yaml --epochs 3 --seeds 0
The test set is touched ONCE per run, after the best validation checkpoint is restored.
"""
import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.datasets import build_datasets
from src.evaluate import aggregate_seeds, bootstrap_ci, fmt, retrieval_metrics, tag_metrics
from src.models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def spec_augment(mel, freq_mask=16, time_mask=24, n_masks=2):
    """SpecAugment-style frequency/time masking on a batch (B, F, T)."""
    mel = mel.clone()
    _, n_f, n_t = mel.shape
    for b in range(mel.size(0)):
        fill = mel[b].mean()
        for _ in range(n_masks):
            f = random.randint(0, freq_mask)
            f0 = random.randint(0, max(0, n_f - f))
            mel[b, f0:f0 + f, :] = fill
            t = random.randint(0, time_mask)
            t0 = random.randint(0, max(0, n_t - t))
            mel[b, :, t0:t0 + t] = fill
    return mel


@torch.no_grad()
def predict(model, loader, device, use_amp):
    model.eval()
    scores, labels, embs = [], [], []
    for mel, y in loader:
        mel = mel.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            if mel.dim() == 4:                       # (B, n_chunks, F, T): average chunk embeddings
                b, n = mel.shape[:2]
                emb = model.embed(mel.flatten(0, 1)).view(b, n, -1).mean(dim=1)
            else:
                emb = model.embed(mel)
            logits = model.classify(emb)
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        embs.append(emb.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(embs)


def first_chunk(mel):
    """(B, n_chunks, F, T) -> (B, F, T) using the first chunk; 3-D input is returned unchanged."""
    return mel[:, 0] if mel.dim() == 4 else mel


def measure_latency(model, sample, device, use_amp, n=30):
    """Inference latency (ms) for ONE ~3.7 s chunk (median over n runs)."""
    model.eval()
    x = first_chunk(sample)[:1].to(device)
    times = []
    with torch.no_grad():
        for i in range(n + 5):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= 5:
                times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def hardware_string(device):
    if device.type == "cuda":
        return f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)"
    return f"CPU ({platform.processor() or platform.machine()})"


def run_one_seed(cfg, seed, device):
    set_seed(seed)
    tcfg = cfg["train"]
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    data = build_datasets(cfg["data"])
    eval_bs = tcfg.get("eval_batch_size", tcfg["batch_size"])
    mk = lambda ds, shuffle: DataLoader(ds, batch_size=tcfg["batch_size"] if shuffle else eval_bs, shuffle=shuffle,
                                        num_workers=tcfg.get("num_workers", 0),
                                        pin_memory=device.type == "cuda", drop_last=shuffle)
    tr, va, te = mk(data["train"], True), mk(data["val"], False), mk(data["test"], False)

    model = build_model(cfg["model"]["name"], n_tags=data["n_tags"],
                        **cfg["model"].get("params", {})).to(device)
    with torch.no_grad():                       # materialise lazy layers
        model.eval()(first_chunk(next(iter(va))[0])[:2].to(device))
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.BCEWithLogitsLoss()

    name = f"{cfg['name']}_seed{seed}"
    ckpt_path = Path(cfg["output"]["ckpt_dir"]) / f"{name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best, bad, history = -1.0, 0, []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    for epoch in range(tcfg["epochs"]):
        model.train()
        running, nb = 0.0, 0
        for mel, y in tr:
            mel, y = mel.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if tcfg.get("augment", False):
                mel = spec_augment(mel)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(mel)
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            nb += 1
        sched.step()
        s, l, _ = predict(model, va, device, use_amp)
        val = tag_metrics(l, s)
        history.append({"epoch": epoch + 1, "train_loss": running / max(nb, 1),
                        "val_roc_auc": val["roc_auc_macro"], "val_pr_auc": val["pr_auc_macro"]})
        print(f"[{name}] epoch {epoch + 1:02d}  loss {running / max(nb, 1):.4f}  "
              f"val ROC {val['roc_auc_macro']:.4f}  PR {val['pr_auc_macro']:.4f}")
        if val["pr_auc_macro"] > best:
            best, bad = val["pr_auc_macro"], 0
            torch.save({"model_state": model.state_dict(), "cfg": cfg,
                        "tag_names": data["tag_names"], "n_tags": data["n_tags"]}, ckpt_path)
        else:
            bad += 1
            if bad >= tcfg.get("patience", 5):
                print(f"[{name}] early stopping")
                break
    train_time = time.time() - t_start
    peak_mem = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None

    # ---- single, final test evaluation with the best checkpoint ----
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    t0 = time.time()
    s, l, emb = predict(model, te, device, use_amp)
    test_time = time.time() - t0
    test = tag_metrics(l, s, data["tag_names"])
    ci = bootstrap_ci(l, s, "roc_auc_macro", n_boot=cfg.get("bootstrap", 50))
    retrieval = retrieval_metrics(emb, l, k=10)
    latency = measure_latency(model, next(iter(te))[0], device, use_amp)

    result = {
        "name": name, "seed": seed, "model": cfg["model"]["name"],
        "test": {k: v for k, v in test.items() if k != "per_tag"},
        "test_roc_auc_ci95": ci, "per_tag": test["per_tag"], "retrieval": retrieval,
        "history": history,
        "cost": {"params": n_params, "model_size_mb": ckpt_path.stat().st_size / 2**20,
                 "train_time_s": train_time, "test_pass_s": test_time,
                 "latency_ms_per_clip": latency, "peak_gpu_mem_mb": peak_mem,
                 "hardware": hardware_string(device)},
    }
    out = Path(cfg["output"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--data-kind")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.data_kind:
        cfg["data"]["kind"] = args.data_kind
    seeds = args.seeds or cfg.get("seeds", [0, 1, 2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {hardware_string(device)}   seeds: {seeds}")

    results = [run_one_seed(cfg, s, device) for s in seeds]

    keys = ["roc_auc_macro", "pr_auc_macro", "f1_macro"]
    agg = aggregate_seeds([{k: r["test"][k] for k in keys} for r in results])
    agg_ret = aggregate_seeds([r["retrieval"] for r in results])
    summary = {"name": cfg["name"], "seeds": seeds,
               "tagging_mean_sd": agg, "retrieval_mean_sd": agg_ret,
               "cost_first_seed": results[0]["cost"]}
    out = Path(cfg["output"]["results_dir"]) / f"{cfg['name']}_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print("\n=== " + cfg["name"] + " (mean ± SD over seeds) ===")
    for k in keys:
        print(f"{k:15s} {fmt(agg[k])}")
    for k, v in agg_ret.items():
        print(f"{k:15s} {fmt(v)}")


if __name__ == "__main__":
    main()
