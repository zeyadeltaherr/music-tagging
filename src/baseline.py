"""Baseline: MFCC statistics + one-vs-rest logistic regression (classical ML)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import tag_metrics


def mfcc_features(wav, sr=16000, n_mfcc=20):
    import librosa
    m = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])      # (2*n_mfcc,)


def run_baseline(Xtr, Ytr, Xva, Yva, Xte, Yte, tag_names=None, Cs=(0.01, 0.1, 1.0)):
    """Pick C on the validation set, report once on the test set."""
    best, best_C = -1, None
    for C in Cs:
        clf = make_pipeline(StandardScaler(),
                            OneVsRestClassifier(LogisticRegression(C=C, max_iter=1000)))
        clf.fit(Xtr, Ytr)
        score = tag_metrics(Yva, clf.predict_proba(Xva))["pr_auc_macro"]
        if score > best:
            best, best_C, best_clf = score, C, clf
    res = tag_metrics(Yte, best_clf.predict_proba(Xte), tag_names)
    res["best_C"] = best_C
    return res, best_clf
