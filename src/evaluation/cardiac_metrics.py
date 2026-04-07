"""
Cardiac-Specific Evaluation Metrics.

Clinical AI systems require more than standard ML metrics.
This module implements evaluation metrics aligned with:
- FDA 510(k) performance characterization standards
- ESC/AHA arrhythmia classification guidelines
- Alert fatigue reduction measurement frameworks

Key metrics for cardiac AI:
    AUROC per class: Standard discrimination metric
    AUPRC: Better than AUROC for severe class imbalance
    Sensitivity @ fixed specificity: Clinical operating points
      (e.g., 95% sensitivity required for life-threatening arrhythmias)
    PPV (Precision): Directly measures false alarm rate
    False alarm rate (FAR): Alarms per hour at operating point
    Alert fatigue index: % unnecessary alarms

References:
    Hannun et al. (2019). Cardiologist-level arrhythmia detection. Nature Med.
    Rajpurkar et al. (2017). Cardiologist-level performance using CNN. arXiv.
    FDA (2021). Artificial Intelligence/Machine Learning-Based Software as a
        Medical Device Action Plan.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ─────────────────────────────────────────────
#  Per-Class ROC / PRC Analysis
# ─────────────────────────────────────────────

def compute_auroc_per_class(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute per-class AUROC for multi-label classification.

    Args:
        y_true: (N, C) binary label matrix.
        y_score: (N, C) predicted probabilities.
        class_names: Optional class name list.

    Returns:
        Dict mapping class name → AUROC value.
    """
    n_classes = y_true.shape[1]
    names = class_names or [f"class_{i}" for i in range(n_classes)]

    result = {}
    for i, name in enumerate(names):
        # Skip classes with no positive samples
        if y_true[:, i].sum() == 0:
            result[name] = float("nan")
            continue
        try:
            auroc = roc_auc_score(y_true[:, i], y_score[:, i])
            result[name] = float(auroc)
        except ValueError:
            result[name] = float("nan")

    # Macro average (excluding NaN)
    valid = [v for v in result.values() if not np.isnan(v)]
    result["macro"] = float(np.mean(valid)) if valid else float("nan")
    return result


def compute_auprc_per_class(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Per-class Area Under Precision-Recall Curve (AUPRC).

    AUPRC is more informative than AUROC under class imbalance.
    For a rare arrhythmia with 2% prevalence, AUROC can be 0.95
    while AUPRC may be only 0.40 — AUPRC tells the real story.
    """
    n_classes = y_true.shape[1]
    names = class_names or [f"class_{i}" for i in range(n_classes)]

    result = {}
    for i, name in enumerate(names):
        if y_true[:, i].sum() == 0:
            result[name] = float("nan")
            continue
        try:
            auprc = average_precision_score(y_true[:, i], y_score[:, i])
            result[name] = float(auprc)
        except ValueError:
            result[name] = float("nan")

    valid = [v for v in result.values() if not np.isnan(v)]
    result["macro"] = float(np.mean(valid)) if valid else float("nan")
    return result


# ─────────────────────────────────────────────
#  Sensitivity / Specificity Analysis
# ─────────────────────────────────────────────

def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float = 0.90,
) -> Dict[str, float]:
    """Find sensitivity at a fixed specificity operating point.

    Clinical systems often require a fixed specificity (e.g., 90%)
    to control false alarm rate, and we report sensitivity at that point.

    For life-threatening arrhythmias (VT, VF), regulatory bodies often
    require ≥95% sensitivity — this function finds the achieved sensitivity.

    Args:
        y_true: (N,) binary labels for ONE class.
        y_score: (N,) predicted probabilities for that class.
        target_specificity: Target specificity value (e.g., 0.90, 0.95).

    Returns:
        dict with sensitivity, threshold, ppv, and actual specificity.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Specificity = 1 - FPR
    specificity = 1 - fpr

    # Find threshold where specificity ≥ target (closest)
    # Use the point that maximizes sensitivity subject to specificity constraint
    valid_mask = specificity >= target_specificity
    if not valid_mask.any():
        # Relax: use highest achievable specificity
        idx = np.argmax(specificity)
    else:
        # Among valid points, find highest sensitivity
        valid_tpr = np.where(valid_mask, tpr, -np.inf)
        idx = np.argmax(valid_tpr)

    opt_threshold = thresholds[idx]
    opt_sensitivity = tpr[idx]
    opt_specificity = specificity[idx]

    # Compute PPV at this threshold
    y_pred = (y_score >= opt_threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    ppv = tp / (tp + fp + 1e-10)

    return {
        "sensitivity": float(opt_sensitivity),
        "specificity": float(opt_specificity),
        "ppv": float(ppv),
        "threshold": float(opt_threshold),
        "target_specificity": target_specificity,
    }


def specificity_at_sensitivity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_sensitivity: float = 0.95,
) -> Dict[str, float]:
    """Find specificity at a fixed sensitivity (recall) operating point.

    Used to answer: "If we require 95% sensitivity for AF detection,
    what is the false alarm rate?"
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    valid_mask = tpr >= target_sensitivity
    if not valid_mask.any():
        idx = np.argmax(tpr)
    else:
        # Among valid (sensitivity ≥ target), maximize specificity
        valid_fpr = np.where(valid_mask, fpr, np.inf)
        idx = np.argmin(valid_fpr)

    return {
        "sensitivity": float(tpr[idx]),
        "specificity": float(1 - fpr[idx]),
        "ppv": float(
            np.sum((y_score >= thresholds[idx]) & (y_true == 1)) /
            (np.sum(y_score >= thresholds[idx]) + 1e-10)
        ),
        "threshold": float(thresholds[idx]),
        "target_sensitivity": target_sensitivity,
        "false_alarm_rate": float(fpr[idx]),
    }


def optimal_threshold_youden(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """Find optimal classification threshold by Youden's J statistic.

    Youden's J = Sensitivity + Specificity - 1 (maximized at optimal threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr + (1 - fpr) - 1
    idx = np.argmax(j_scores)
    return {
        "threshold": float(thresholds[idx]),
        "sensitivity": float(tpr[idx]),
        "specificity": float(1 - fpr[idx]),
        "youden_j": float(j_scores[idx]),
    }


# ─────────────────────────────────────────────
#  Alert Fatigue Metrics
# ─────────────────────────────────────────────

def compute_alert_fatigue_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    recording_duration_hours: float = 1.0,
) -> Dict[str, float]:
    """Compute alert fatigue metrics at a specific threshold.

    Alert fatigue is THE primary challenge in cardiac monitoring.
    These metrics quantify the clinical burden of false alarms.

    Metrics:
        false_alarm_rate: False alarms per hour of monitoring
        alarm_burden: Total alarms per hour
        unnecessary_alarm_rate: Fraction of alarms that are false positives
        alarm_savings: Reduction in alarms vs. worst-case threshold-only system

    Args:
        y_true: (N,) binary labels.
        y_score: (N,) predicted probabilities.
        threshold: Classification threshold.
        recording_duration_hours: Total recording time in hours.

    Returns:
        Dict of alert fatigue metrics.
    """
    y_pred = (y_score >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    total_alarms = tp + fp
    total_true_events = tp + fn

    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    ppv = tp / (tp + fp + 1e-10)
    npv = tn / (tn + fn + 1e-10)

    false_alarm_rate = fp / recording_duration_hours
    alarm_burden = total_alarms / recording_duration_hours
    unnecessary_rate = fp / (fp + tp + 1e-10)

    # Alarm savings vs. naive alerting on all positives
    naive_alarms = total_true_events / recording_duration_hours
    alarm_savings = max(0, 1 - alarm_burden / (naive_alarms + 1e-10))

    return {
        "threshold": float(threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1_score": float(2 * ppv * sensitivity / (ppv + sensitivity + 1e-10)),
        "false_alarm_rate_per_hour": float(false_alarm_rate),
        "alarm_burden_per_hour": float(alarm_burden),
        "unnecessary_alarm_rate": float(unnecessary_rate),
        "alarm_savings_fraction": float(alarm_savings),
    }


# ─────────────────────────────────────────────
#  Comprehensive Metrics Function
# ─────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Optional[List[str]] = None,
    operating_specificities: List[float] = [0.85, 0.90, 0.95],
) -> Dict[str, float]:
    """Compute all classification metrics for model evaluation.

    Args:
        y_true: (N, C) binary labels.
        y_score: (N, C) predicted probabilities.
        class_names: Label names.
        operating_specificities: Specificity levels for sensitivity reporting.

    Returns:
        Flat dict of all metrics suitable for logging.
    """
    n_classes = y_true.shape[1]
    names = class_names or [f"class_{i}" for i in range(n_classes)]

    metrics: Dict[str, float] = {}

    # --- AUROC ---
    auroc_dict = compute_auroc_per_class(y_true, y_score, names)
    metrics["macro_auroc"] = auroc_dict.pop("macro", float("nan"))
    for name, val in auroc_dict.items():
        metrics[f"auroc_{name}"] = val

    # --- AUPRC ---
    auprc_dict = compute_auprc_per_class(y_true, y_score, names)
    metrics["macro_auprc"] = auprc_dict.pop("macro", float("nan"))
    for name, val in auprc_dict.items():
        metrics[f"auprc_{name}"] = val

    # --- Threshold-based metrics at 0.5 ---
    y_pred = (y_score > 0.5).astype(int)

    # Per-class F1
    for i, name in enumerate(names):
        if y_true[:, i].sum() > 0:
            f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            metrics[f"f1_{name}"] = float(f1)

    metrics["macro_f1"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["micro_f1"] = float(
        f1_score(y_true, y_pred, average="micro", zero_division=0)
    )

    # Exact match (all labels correct)
    metrics["exact_match"] = float(np.all(y_pred == y_true, axis=1).mean())

    # --- Sensitivity at fixed specificities (for first class as example) ---
    # Full per-class analysis done in clinical_analysis.py
    for spec in operating_specificities:
        sens_list = []
        for i in range(n_classes):
            if y_true[:, i].sum() >= 10:  # Need enough positives
                result = sensitivity_at_specificity(
                    y_true[:, i], y_score[:, i], spec
                )
                sens_list.append(result["sensitivity"])
        if sens_list:
            metrics[f"macro_sensitivity_at_spec{int(spec*100)}"] = float(np.mean(sens_list))

    return metrics


# ─────────────────────────────────────────────
#  Confusion Matrix Analysis
# ─────────────────────────────────────────────

def per_class_confusion(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> List[Dict]:
    """Per-class binary confusion matrix analysis."""
    n_classes = y_true.shape[1]
    names = class_names or [f"class_{i}" for i in range(n_classes)]
    y_pred = (y_score >= threshold).astype(int)

    results = []
    for i, name in enumerate(names):
        if y_true[:, i].sum() == 0:
            continue
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prevalence = y_true[:, i].mean()
        results.append({
            "class": name,
            "prevalence": float(prevalence),
            "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn),
            "sensitivity": float(tp / (tp + fn + 1e-10)),
            "specificity": float(tn / (tn + fp + 1e-10)),
            "ppv": float(tp / (tp + fp + 1e-10)),
            "npv": float(tn / (tn + fn + 1e-10)),
            "f1": float(2 * tp / (2 * tp + fp + fn + 1e-10)),
        })

    return results


# ─────────────────────────────────────────────
#  Bootstrap Confidence Intervals
# ─────────────────────────────────────────────

def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap confidence interval for macro-AUROC.

    Required for clinical validation publications and FDA submissions.
    Bootstrap resampling is the standard approach for ECG classifier CIs.

    Args:
        y_true: (N, C) binary labels.
        y_score: (N, C) predicted probabilities.
        n_bootstrap: Number of bootstrap iterations (1000 is standard).
        ci: Confidence interval level (e.g., 0.95 → 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'mean', 'lower', 'upper', 'std' AUROC values.
    """
    rng = np.random.default_rng(seed)
    N = y_true.shape[0]
    bootstrap_aurocs = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.integers(0, N, size=N)
        bt_true = y_true[indices]
        bt_score = y_score[indices]

        # Compute macro AUROC
        try:
            macro = roc_auc_score(bt_true, bt_score, average="macro", multi_class="ovr")
            bootstrap_aurocs.append(macro)
        except ValueError:
            pass  # Skip if only one class in bootstrap sample

    bootstrap_aurocs = np.array(bootstrap_aurocs)
    alpha = 1 - ci
    lower = float(np.percentile(bootstrap_aurocs, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_aurocs, 100 * (1 - alpha / 2)))

    return {
        "mean": float(np.mean(bootstrap_aurocs)),
        "std": float(np.std(bootstrap_aurocs)),
        "lower": lower,
        "upper": upper,
        "ci": ci,
        "n_bootstrap": n_bootstrap,
    }


# ─────────────────────────────────────────────
#  DeLong Test (AUC Comparison)
# ─────────────────────────────────────────────

def delong_test(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> Dict[str, float]:
    """DeLong test for comparing two AUROC values (single class).

    The DeLong method is the standard statistical test for comparing
    AUC values from correlated ROC curves (same patient population).
    Used in clinical papers to demonstrate significant improvement over
    published benchmarks.

    Reference:
        DeLong, E. R., et al. (1988). Comparing the areas under two or
        more correlated receiver operating characteristic curves.
        Biometrics, 837–845.

    Args:
        y_true: (N,) binary ground truth.
        y_score_a: (N,) predicted scores for model A.
        y_score_b: (N,) predicted scores for model B.

    Returns:
        Dict with z-statistic, p-value, and individual AUROCs.
    """
    auc_a = roc_auc_score(y_true, y_score_a)
    auc_b = roc_auc_score(y_true, y_score_b)

    def structural_components(y, yhat):
        """Compute structural components (V10, V01) for DeLong variance."""
        n = len(y)
        pos = yhat[y == 1]
        neg = yhat[y == 0]
        n_pos, n_neg = len(pos), len(neg)
        if n_pos == 0 or n_neg == 0:
            return 0.0, 0.0, 0.5
        v10 = np.mean([np.sum(pos > p) + 0.5 * np.sum(pos == p) for p in neg]) / n_pos
        v01 = np.mean([np.sum(neg < p) + 0.5 * np.sum(neg == p) for p in pos]) / n_neg
        auc = (np.sum([np.sum(pos > n) + 0.5 * np.sum(pos == n) for n in neg]) /
               (n_pos * n_neg))
        return v10, v01, auc

    v10_a, v01_a, _ = structural_components(y_true, y_score_a)
    v10_b, v01_b, _ = structural_components(y_true, y_score_b)

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)

    # Variance of AUC difference (simplified DeLong)
    var_a = (v10_a * (1 - v10_a) / n_neg + v01_a * (1 - v01_a) / n_pos)
    var_b = (v10_b * (1 - v10_b) / n_neg + v01_b * (1 - v01_b) / n_pos)

    se = np.sqrt(var_a + var_b + 1e-12)
    z = (auc_a - auc_b) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "auroc_a": float(auc_a),
        "auroc_b": float(auc_b),
        "auroc_diff": float(auc_a - auc_b),
        "z_statistic": float(z),
        "p_value": float(p_value),
        "significant_p05": bool(p_value < 0.05),
        "significant_p01": bool(p_value < 0.01),
    }


if __name__ == "__main__":
    # Synthetic test
    np.random.seed(42)
    N, C = 500, 5
    y_true = (np.random.rand(N, C) > 0.8).astype(float)
    y_score = np.clip(y_true + np.random.randn(N, C) * 0.3, 0, 1)

    class_names = ["NORM", "MI", "STTC", "CD", "HYP"]
    metrics = compute_all_metrics(y_true, y_score, class_names)
    print(f"Macro AUROC: {metrics['macro_auroc']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")

    # Bootstrap CI
    ci = bootstrap_auroc_ci(y_true, y_score, n_bootstrap=200)
    print(f"AUROC 95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
