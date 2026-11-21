"""Shared evaluation for ALL models (baseline, CNN, pretrained, advanced).

Tagging    : ROC-AUC and PR-AUC (macro + per-tag), macro-F1 at a fixed threshold.
Retrieval  : precision@k and Jaccard@k of tags between a query and its neighbours.
Statistics : bootstrap confidence intervals and mean +- SD over random seeds.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from src.similarity import SimilarityIndex


def tag_metrics(y_true, y_score, tag_names=None, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    n_tags = y_true.shape[1]
    roc = np.full(n_tags, np.nan)
    pr = np.full(n_tags, np.nan)
    for j in range(n_tags):
        pos = y_true[:, j].sum()
        if 0 < pos < len(y_true):          # tag needs both classes in this split
            roc[j] = roc_auc_score(y_true[:, j], y_score[:, j])
            pr[j] = average_precision_score(y_true[:, j], y_score[:, j])
    f1 = f1_score(y_true, (y_score >= threshold).astype(int), average="macro", zero_division=0)
    names = tag_names if tag_names is not None else [f"tag_{j}" for j in range(n_tags)]
    per_tag = [
        {"tag": names[j], "support": int(y_true[:, j].sum()),
         "roc_auc": None if np.isnan(roc[j]) else float(roc[j]),
         "pr_auc": None if np.isnan(pr[j]) else float(pr[j])}
        for j in range(n_tags)
    ]
    return {
        "roc_auc_macro": float(np.nanmean(roc)),
        "pr_auc_macro": float(np.nanmean(pr)),
        "f1_macro": float(f1),
        "per_tag": per_tag,
    }


def bootstrap_ci(y_true, y_score, metric="roc_auc_macro", n_boot=100, alpha=0.05, seed=0):
    """Percentile bootstrap CI over test clips."""
    rng = np.random.RandomState(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        vals.append(tag_metrics(y_true[idx], y_score[idx])[metric])
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def aggregate_seeds(runs):
    """runs: list of dicts of scalar metrics -> {metric: (mean, sd)}."""
    out = {}
    for key in runs[0]:
        v = np.array([r[key] for r in runs], dtype=float)
        out[key] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)
    return out


def fmt(mean_sd, digits=4):
    return f"{mean_sd[0]:.{digits}f} ± {mean_sd[1]:.{digits}f}"


def retrieval_metrics(embeddings, labels, k=10):
    """For every clip, retrieve k nearest neighbours (cosine, self excluded) and measure
    how many share tags with the query. Clips with no tags are skipped."""
    labels = np.asarray(labels).astype(bool)
    index = SimilarityIndex().fit(embeddings)
    nbrs, _ = index.search(embeddings, k=k, exclude_self=True)
    keep = labels.any(axis=1)
    prec, jac = [], []
    for q in np.where(keep)[0]:
        nb = labels[nbrs[q]]                                   # (k, n_tags)
        inter = (nb & labels[q]).sum(axis=1)
        union = (nb | labels[q]).sum(axis=1)
        prec.append(float((inter > 0).mean()))
        jac.append(float((inter / np.maximum(union, 1)).mean()))
    return {f"precision@{k}": float(np.mean(prec)), f"jaccard@{k}": float(np.mean(jac))}
