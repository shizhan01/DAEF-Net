"""Evaluation metrics (Section 3.2)."""

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score


def classification_metrics(y, prob):
    k = prob.shape[1]
    pred = prob.argmax(axis=1)
    f1 = f1_score(y, pred, labels=np.arange(k), average=None, zero_division=0)
    present = np.isin(np.arange(k), np.unique(y))
    return {
        "accuracy": float(np.mean(pred == y)),
        "macro_f1": float(np.mean(f1[present])),
        "kappa": float(cohen_kappa_score(y, pred)),
        "worst_class_f1": float(np.min(f1[present])),
        "per_class_f1": f1.tolist(),
    }


def expected_calibration_error(prob, y, n_bins=15):
    conf = prob.max(axis=1)
    correct = (prob.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def negative_log_likelihood(prob, y, eps=1e-12):
    return float(-np.mean(np.log(prob[np.arange(len(y)), y] + eps)))


def brier_score(prob, y):
    onehot = np.eye(prob.shape[1])[y]
    return float(np.mean(np.sum((prob - onehot) ** 2, axis=1)))


def aurc(uncertainty, correct):
    """Area under the risk-coverage curve, ranking by increasing uncertainty."""
    order = np.argsort(uncertainty, kind="stable")
    errors = 1.0 - correct[order].astype(float)
    risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(np.mean(risk))


def ood_auroc(u_in, u_out):
    """AUROC with in-distribution events as negatives and unknown events as positives."""
    scores = np.concatenate([u_in, u_out])
    labels = np.concatenate([np.zeros(len(u_in)), np.ones(len(u_out))])
    return float(roc_auc_score(labels, scores))


def calibration_metrics(y, prob, u):
    correct = prob.argmax(axis=1) == y
    return {
        "ece": expected_calibration_error(prob, y),
        "nll": negative_log_likelihood(prob, y),
        "brier": brier_score(prob, y),
        "aurc": aurc(u, correct),
    }


def tier_report(u, correct, low, high):
    """Three-tier policy of Section 4.3: automatic, flagged and review tiers."""
    auto = u < low
    flagged = (u >= low) & (u < high)
    review = u >= high
    report = {
        "auto_fraction": float(auto.mean()),
        "flagged_fraction": float(flagged.mean()),
        "review_fraction": float(review.mean()),
    }
    if correct is not None:
        report["auto_accuracy"] = float(correct[auto].mean()) if auto.any() else float("nan")
        report["errors_in_flagged_or_review"] = int(np.sum(~correct & ~auto))
        report["errors_total"] = int(np.sum(~correct))
    return report
