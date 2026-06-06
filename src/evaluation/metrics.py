"""Evaluation metrics for ALN pCR prediction.

Includes classification metrics, calibration, subgroup analysis,
clinical decision support metrics, and decision curve analysis.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    return_curves: bool = False,
) -> dict[str, float | np.ndarray]:
    """Compute standard classification metrics.

    Parameters
    ----------
    return_curves : bool
        If True, also return raw FPR/TPR/precision/recall arrays for plotting.
    """
    y_pred = (y_prob >= threshold).astype(int)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)

    metrics: dict[str, float | np.ndarray] = {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": auc(recall, precision),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "sensitivity": tpr[np.argmax(tpr - fpr)],
        "specificity": 1 - fpr[np.argmax(tpr - fpr)],
        "brier_score": brier_score_loss(y_true, y_prob),
    }

    # Sensitivity at 90% specificity
    idx_90spec = np.argmin(np.abs(1 - fpr - 0.9))
    metrics["sensitivity_at_90spec"] = tpr[idx_90spec]

    if return_curves:
        metrics["fpr"] = fpr
        metrics["tpr"] = tpr
        metrics["roc_thresholds"] = roc_thresholds
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["pr_thresholds"] = pr_thresholds

    return metrics


def compute_net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute net benefit for Decision Curve Analysis.

    Parameters
    ----------
    y_true : binary labels
    y_prob : predicted probabilities
    thresholds : array of threshold probabilities (default: 0.01 to 0.99)

    Returns
    -------
    dict with keys: thresholds, net_benefit_model, net_benefit_all
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)

    n = len(y_true)
    prevalence = y_true.mean()
    net_benefit_model = np.zeros_like(thresholds, dtype=float)
    net_benefit_all = np.zeros_like(thresholds, dtype=float)

    for i, pt in enumerate(thresholds):
        y_pred = (y_prob >= pt).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        net_benefit_model[i] = tp / n - fp / n * (pt / (1 - pt))
        net_benefit_all[i] = prevalence - (1 - prevalence) * (pt / (1 - pt))

    return {
        "thresholds": thresholds,
        "net_benefit_model": net_benefit_model,
        "net_benefit_all": net_benefit_all,
    }


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Compute calibration curve data."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return {"prob_true": prob_true, "prob_pred": prob_pred}


def compute_false_negative_rate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute false negative rate (missed residual disease).

    FNR = FN / (FN + TP)
    For clinical use: proportion of actual non-pCR patients
    incorrectly predicted as pCR.
    """
    y_pred = (y_prob >= threshold).astype(int)
    fn = ((y_pred == 1) & (y_true == 0)).sum()
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    return fn / max(fn + tp, 1)


def surgery_decision_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Analyze potential surgical decision support.

    Compute how many patients could avoid unnecessary ALND
    and the associated false negative rate.
    """
    y_pred = (y_prob >= threshold).astype(int)

    predicted_pcr = y_pred.sum()
    actual_pcr_in_predicted = y_true[y_pred == 1].sum()
    false_negatives = predicted_pcr - actual_pcr_in_predicted

    return {
        "predicted_pcr_count": int(predicted_pcr),
        "correctly_identified_pcr": int(actual_pcr_in_predicted),
        "false_positives_avoided_alnd": int(false_negatives),
        "fnr": compute_false_negative_rate(y_true, y_prob, threshold),
        "total_actual_pcr": int(y_true.sum()),
        "pcr_detection_rate": actual_pcr_in_predicted / max(y_true.sum(), 1),
    }


def subgroup_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    group_names: dict[int, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute metrics per subgroup."""
    results = {}
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() < 5 or len(set(y_true[mask])) < 2:
            continue
        name = group_names.get(int(g), str(g)) if group_names else str(g)
        results[name] = compute_classification_metrics(y_true[mask], y_prob[mask])
        results[name]["n"] = int(mask.sum())
    return results
