"""Evaluation metrics: AUC (binary / macro one-vs-rest), sensitivity, specificity,
bootstrap confidence intervals, expected calibration error and threshold selection."""
import numpy as np
from sklearn.metrics import (roc_auc_score, accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix, roc_curve, recall_score)


def auc_score(y, probs):
    if probs.shape[1] == 2:
        return roc_auc_score(y, probs[:, 1])
    return roc_auc_score(y, probs, multi_class="ovr", average="macro")


def bootstrap_ci(y, probs, fn, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < probs.shape[1]:
            continue
        vals.append(fn(y[idx], probs[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def expected_calibration_error(y, probs, n_bins=15):
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def youden_threshold(y, p_pos):
    """Threshold maximizing sensitivity + specificity - 1 (chosen on the validation set)."""
    fpr, tpr, thr = roc_curve(y, p_pos)
    return float(thr[np.argmax(tpr - fpr)])


def compute_metrics(y, probs, threshold=None):
    """threshold applies to binary tasks only; None means argmax (= 0.5)."""
    if probs.shape[1] == 2 and threshold is not None:
        pred = (probs[:, 1] >= threshold).astype(int)
    else:
        pred = probs.argmax(1)
    lo, hi = bootstrap_ci(y, probs, auc_score)
    m = {
        "auc": float(auc_score(y, probs)), "auc_ci_low": lo, "auc_ci_high": hi,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "ece": expected_calibration_error(y, probs),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(probs.shape[1])).tolist(),
        "per_class_recall": recall_score(y, pred, average=None, labels=range(probs.shape[1]),
                                         zero_division=0).tolist(),
    }
    if probs.shape[1] == 2:
        tn, fp, fn, tp = np.array(m["confusion_matrix"]).ravel()
        m["sensitivity"] = float(tp / (tp + fn))
        m["specificity"] = float(tn / (tn + fp))
        m["ppv"] = float(tp / (tp + fp)) if tp + fp else 0.0
        m["npv"] = float(tn / (tn + fn)) if tn + fn else 0.0
    return m
