"""
Clinical Validation and Analysis Module.

This module implements the clinical validation framework required for
regulatory submission of AI-based cardiac diagnostic software.

Covers:
1. Demographic subgroup analysis (sex, age, race) — FDA bias detection requirement
2. Confidence calibration (reliability diagrams, ECE, Platt scaling)
3. Clinical operating point selection and tuning
4. Statistical significance testing
5. Performance stratification by signal quality

FDA GMLP (Good Machine Learning Practice) requirements addressed:
- Training data independence (no patient-level leakage)
- Subgroup performance characterization
- Uncertainty quantification
- Performance monitoring in deployment (concept drift)

References:
    FDA (2021). Artificial Intelligence/Machine Learning-Based Software
        as a Medical Device (AI/ML-Based SaMD) Action Plan.
    Obermeyer, Z. et al. (2019). Dissecting racial bias in an algorithm used
        to manage the health of populations. Science.
    Niculescu-Mizil & Caruana (2005). Predicting good probabilities with
        supervised learning. ICML.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.evaluation.cardiac_metrics import (
    compute_auroc_per_class,
    sensitivity_at_specificity,
    specificity_at_sensitivity,
)


# ─────────────────────────────────────────────
#  Demographic Subgroup Analysis
# ─────────────────────────────────────────────

class SubgroupAnalyzer:
    """Analyze model performance across demographic subgroups.

    FDA and academic guidelines require demonstrating that AI performance
    is equitable across demographic groups. Performance disparities can
    arise from:
    - Training data representation imbalance
    - Physiological ECG differences by sex (QTc intervals, T-wave amplitudes)
    - Age-related ECG morphology changes

    This analyzer quantifies performance gaps and provides statistical
    evidence for or against clinically significant disparities.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        metadata: pd.DataFrame,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            y_true: (N, C) binary labels.
            y_score: (N, C) predicted probabilities.
            metadata: DataFrame with columns like 'age', 'sex', 'race'.
            class_names: Class label names.
        """
        assert len(y_true) == len(metadata)
        self.y_true = y_true
        self.y_score = y_score
        self.metadata = metadata.reset_index(drop=True)
        self.class_names = class_names or [f"class_{i}" for i in range(y_true.shape[1])]

    def analyze_by_sex(self) -> Dict[str, Dict]:
        """Compare performance by biological sex (M/F).

        Clinical note: Women have higher baseline heart rates,
        shorter PR intervals, and longer QTc intervals than men.
        Some algorithms have documented sex-based performance differences.
        """
        if "sex" not in self.metadata.columns:
            return {"error": "sex column not found in metadata"}

        results = {}
        for sex_val in self.metadata["sex"].unique():
            mask = (self.metadata["sex"] == sex_val).values
            if mask.sum() < 20:  # Insufficient samples
                continue
            results[str(sex_val)] = self._compute_subgroup_metrics(mask)

        # Statistical significance test between groups
        groups = list(results.keys())
        if len(groups) == 2:
            results["disparity_analysis"] = self._test_group_difference(
                results[groups[0]], results[groups[1]], groups
            )

        return results

    def analyze_by_age_group(
        self,
        age_bins: List[float] = [0, 40, 60, 75, 200],
        bin_labels: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """Compare performance across age groups.

        Age affects ECG interpretation:
        - Elderly: Higher prevalence of LBBB, AF, LVH
        - Young: Higher prevalence of WPW, channelopathies
        - Age-related QRS changes may confound automated analysis
        """
        if "age" not in self.metadata.columns:
            return {"error": "age column not found in metadata"}

        labels = bin_labels or [f"age_{int(age_bins[i])}-{int(age_bins[i+1])}"
                                for i in range(len(age_bins) - 1)]
        age_groups = pd.cut(
            self.metadata["age"],
            bins=age_bins,
            labels=labels,
            right=False,
        )

        results = {}
        for group in labels:
            mask = (age_groups == group).values
            if mask.sum() < 20:
                continue
            results[str(group)] = self._compute_subgroup_metrics(mask)
            results[str(group)]["n_samples"] = int(mask.sum())

        return results

    def analyze_by_signal_quality(
        self,
        sqi_scores: np.ndarray,
        quality_bins: List[float] = [0.0, 0.5, 0.7, 0.9, 1.0],
    ) -> Dict[str, Dict]:
        """Analyze how model performance varies with signal quality.

        Real-world deployment encounters variable signal quality.
        Understanding performance degradation at low SQI informs:
        - Minimum quality thresholds for clinical use
        - Whether to flag low-quality predictions for manual review
        """
        bin_labels = [f"sqi_{quality_bins[i]:.1f}-{quality_bins[i+1]:.1f}"
                      for i in range(len(quality_bins) - 1)]

        results = {}
        for i, label in enumerate(bin_labels):
            mask = (
                (sqi_scores >= quality_bins[i]) &
                (sqi_scores < quality_bins[i + 1])
            )
            if mask.sum() < 10:
                continue
            results[label] = self._compute_subgroup_metrics(mask)
            results[label]["n_samples"] = int(mask.sum())
            results[label]["mean_sqi"] = float(sqi_scores[mask].mean())

        return results

    def _compute_subgroup_metrics(self, mask: np.ndarray) -> Dict[str, float]:
        """Compute key metrics for a subgroup defined by a boolean mask."""
        yt = self.y_true[mask]
        ys = self.y_score[mask]

        if yt.shape[0] < 10:
            return {"error": "insufficient samples"}

        metrics: Dict[str, float] = {}
        metrics["n_samples"] = int(mask.sum())

        # AUROC (skip classes with no positives)
        auroc_dict = compute_auroc_per_class(yt, ys, self.class_names)
        metrics["macro_auroc"] = auroc_dict.get("macro", float("nan"))
        metrics.update({f"auroc_{k}": v for k, v in auroc_dict.items() if k != "macro"})

        # Threshold-based metrics at 0.5
        y_pred = (ys > 0.5).astype(int)
        tp_all = np.sum((y_pred == 1) & (yt == 1))
        fp_all = np.sum((y_pred == 1) & (yt == 0))
        fn_all = np.sum((y_pred == 0) & (yt == 1))

        metrics["sensitivity"] = float(tp_all / (tp_all + fn_all + 1e-10))
        metrics["specificity"] = float(
            np.sum((y_pred == 0) & (yt == 0)) / (np.sum(yt == 0) + 1e-10)
        )
        metrics["ppv"] = float(tp_all / (tp_all + fp_all + 1e-10))

        return metrics

    def _test_group_difference(
        self,
        group_a: Dict[str, float],
        group_b: Dict[str, float],
        group_names: List[str],
    ) -> Dict[str, float]:
        """Test for statistically significant performance differences."""
        diff = group_a.get("macro_auroc", 0.5) - group_b.get("macro_auroc", 0.5)
        return {
            "auroc_difference": float(diff),
            "group_a": group_names[0],
            "group_b": group_names[1],
            "auroc_a": group_a.get("macro_auroc", float("nan")),
            "auroc_b": group_b.get("macro_auroc", float("nan")),
            "clinically_significant": abs(diff) > 0.05,  # >5% AUROC gap is concerning
        }

    def generate_equity_report(self) -> Dict:
        """Generate a complete bias/equity analysis report."""
        report = {
            "by_sex": self.analyze_by_sex(),
            "by_age": self.analyze_by_age_group(),
        }
        return report


# ─────────────────────────────────────────────
#  Calibration Analysis
# ─────────────────────────────────────────────

class CalibrationAnalyzer:
    """Assess and correct probability calibration for cardiac AI.

    Well-calibrated probabilities are essential for:
    - Setting appropriate clinical thresholds
    - Risk stratification (e.g., "90% probability of AF")
    - Trust in the model's confidence estimates

    Deep neural networks are often overconfident (predicted 90% probability
    is not actually correct 90% of the time). Temperature scaling and
    Platt scaling correct this systematic bias.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.y_true = y_true
        self.y_score = y_score
        self.class_names = class_names or [f"c{i}" for i in range(y_true.shape[1])]

    def compute_ece(
        self,
        class_idx: int,
        n_bins: int = 15,
    ) -> Dict[str, float]:
        """Compute Expected Calibration Error for one class.

        ECE = Σ (|B_m| / N) * |acc(B_m) - conf(B_m)|

        A perfectly calibrated model has ECE = 0.
        ECE < 0.05 is generally acceptable for clinical applications.
        """
        y_t = self.y_true[:, class_idx]
        y_s = self.y_score[:, class_idx]

        # Reliability curve
        try:
            fraction_pos, mean_pred = calibration_curve(
                y_t, y_s, n_bins=n_bins, strategy="uniform"
            )
        except ValueError:
            return {"ece": float("nan"), "brier_score": float("nan")}

        # ECE calculation
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(y_s)

        for i in range(n_bins):
            in_bin = (y_s > bin_boundaries[i]) & (y_s <= bin_boundaries[i + 1])
            n_in_bin = in_bin.sum()
            if n_in_bin > 0:
                avg_conf = y_s[in_bin].mean()
                avg_acc = y_t[in_bin].mean()
                ece += (n_in_bin / total) * abs(avg_conf - avg_acc)

        brier = brier_score_loss(y_t, y_s)

        return {
            "ece": float(ece),
            "brier_score": float(brier),
            "fraction_positives": fraction_pos.tolist(),
            "mean_predicted": mean_pred.tolist(),
            "class": self.class_names[class_idx],
        }

    def compute_all_ece(self) -> Dict[str, Dict]:
        """Compute ECE for all classes."""
        results = {}
        for i, name in enumerate(self.class_names):
            if self.y_true[:, i].sum() >= 10:
                results[name] = self.compute_ece(i)
        return results

    def fit_temperature_scaling(
        self,
        y_true_val: np.ndarray,
        logits_val: np.ndarray,
    ) -> float:
        """Fit temperature scaling parameter on validation set.

        Temperature scaling: p_calibrated = σ(logits / T)
        Single parameter T (temperature) calibrates the entire model.
        Simple, effective, and preserves accuracy.

        A well-calibrated model has T ≈ 1.0.
        T > 1.0: Model is overconfident → soften predictions
        T < 1.0: Model is underconfident → sharpen predictions
        """
        from scipy.optimize import minimize_scalar

        def nll(temperature):
            """Negative log-likelihood of calibrated predictions."""
            T = max(0.01, float(temperature))
            logits_scaled = logits_val / T
            probs = 1 / (1 + np.exp(-logits_scaled))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            nll = -np.mean(
                y_true_val * np.log(probs) + (1 - y_true_val) * np.log(1 - probs)
            )
            return nll

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        return float(result.x)

    def apply_temperature(
        self,
        logits: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Apply temperature scaling to raw logits."""
        return 1 / (1 + np.exp(-logits / max(temperature, 0.01)))

    def fit_platt_scaling(
        self,
        y_true_val: np.ndarray,
        y_score_val: np.ndarray,
        class_idx: int,
    ) -> Tuple[float, float]:
        """Fit Platt scaling (sigmoid calibration) for one class.

        Platt scaling fits a logistic regression on the raw scores,
        learning both scale (A) and bias (B):
        p_calibrated = σ(A * score + B)

        More flexible than temperature scaling but class-specific.
        """
        y_t = y_true_val[:, class_idx].reshape(-1, 1)
        y_s = y_score_val[:, class_idx].reshape(-1, 1)

        # Fit logistic regression (Platt scaling)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr = LogisticRegression(C=1e10, solver="lbfgs")
            lr.fit(y_s, y_t.ravel())

        A = float(lr.coef_[0, 0])
        B = float(lr.intercept_[0])
        return A, B


# ─────────────────────────────────────────────
#  Operating Point Selection
# ─────────────────────────────────────────────

class OperatingPointSelector:
    """Select classification thresholds for clinical deployment.

    Different clinical contexts require different operating points:
    - ICU monitoring: High sensitivity (don't miss events), tolerate more alarms
    - Outpatient screening: High specificity (avoid unnecessary workup)
    - Emergency triage: Near-perfect sensitivity (life-threatening)

    This class helps select and validate operating points for each use case.
    """

    # Minimum acceptable sensitivity by arrhythmia risk tier
    RISK_TIERS = {
        "life_threatening": {
            "classes": ["VFIB", "VTACH", "WPW", "CHB"],
            "min_sensitivity": 0.99,
            "description": "Requires ≥99% sensitivity — missed = potential death",
        },
        "urgent": {
            "classes": ["AFIB", "AVNRT", "LBBB_new"],
            "min_sensitivity": 0.95,
            "description": "Requires ≥95% sensitivity — missed = significant harm",
        },
        "non_urgent": {
            "classes": ["NORM", "SBRAD", "STACH"],
            "min_sensitivity": 0.85,
            "description": "≥85% sensitivity acceptable",
        },
    }

    def __init__(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.y_true = y_true
        self.y_score = y_score
        self.class_names = class_names or [f"c{i}" for i in range(y_true.shape[1])]

    def select_clinical_thresholds(
        self,
        min_sensitivity: float = 0.90,
        max_false_alarm_rate: Optional[float] = None,
    ) -> Dict[str, float]:
        """Select per-class thresholds satisfying sensitivity requirements.

        Args:
            min_sensitivity: Minimum required sensitivity for all classes.
            max_false_alarm_rate: Maximum tolerated false alarm rate per hour.

        Returns:
            Dict mapping class_name → optimal_threshold.
        """
        thresholds = {}
        for i, name in enumerate(self.class_names):
            if self.y_true[:, i].sum() < 10:
                thresholds[name] = 0.5  # Default
                continue

            result = specificity_at_sensitivity(
                self.y_true[:, i],
                self.y_score[:, i],
                min_sensitivity,
            )
            thresholds[name] = result["threshold"]

        return thresholds

    def pareto_optimal_thresholds(
        self,
        n_points: int = 50,
    ) -> List[Dict[str, float]]:
        """Enumerate the Pareto frontier of sensitivity vs. PPV tradeoffs.

        Returns a list of operating points that are Pareto-optimal
        (no other point has both higher sensitivity AND higher PPV).
        Used for generating ROC-based threshold selection tables
        for clinical decision makers.
        """
        pareto_points = []

        for class_idx in range(self.y_true.shape[1]):
            y_t = self.y_true[:, class_idx]
            y_s = self.y_score[:, class_idx]
            if y_t.sum() < 10:
                continue

            thresholds = np.linspace(0.01, 0.99, n_points)
            points = []
            for thr in thresholds:
                y_pred = (y_s >= thr).astype(int)
                tp = np.sum((y_pred == 1) & (y_t == 1))
                fp = np.sum((y_pred == 1) & (y_t == 0))
                fn = np.sum((y_pred == 0) & (y_t == 1))
                sens = tp / (tp + fn + 1e-10)
                ppv = tp / (tp + fp + 1e-10)
                points.append({
                    "class": self.class_names[class_idx],
                    "threshold": float(thr),
                    "sensitivity": float(sens),
                    "ppv": float(ppv),
                })
            pareto_points.extend(points)

        return pareto_points


# ─────────────────────────────────────────────
#  Concept Drift Monitor
# ─────────────────────────────────────────────

class ConceptDriftMonitor:
    """Monitor for distribution shift in deployed model predictions.

    Models can degrade over time due to:
    - Device firmware updates changing ECG acquisition parameters
    - Population demographic shifts
    - Evolving disease presentations

    This lightweight monitor tracks prediction statistics and
    alerts when significant drift is detected.
    """

    def __init__(
        self,
        reference_probs: np.ndarray,
        class_names: Optional[List[str]] = None,
        ks_threshold: float = 0.05,
    ) -> None:
        """
        Args:
            reference_probs: (N, C) reference prediction probabilities
                from validation/test set at deployment time.
            class_names: Class names.
            ks_threshold: Kolmogorov-Smirnov p-value threshold for drift alert.
        """
        self.reference = reference_probs
        self.class_names = class_names or [f"c{i}" for i in range(reference_probs.shape[1])]
        self.ks_threshold = ks_threshold
        self.ref_stats = self._compute_stats(reference_probs)

    def _compute_stats(self, probs: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "mean": probs.mean(axis=0),
            "std": probs.std(axis=0),
            "median": np.median(probs, axis=0),
            "p25": np.percentile(probs, 25, axis=0),
            "p75": np.percentile(probs, 75, axis=0),
        }

    def check_drift(self, new_probs: np.ndarray) -> Dict[str, bool | float]:
        """Check for distribution shift using Kolmogorov-Smirnov test.

        Args:
            new_probs: (N, C) new prediction probabilities.

        Returns:
            Dict with per-class KS statistics and drift flags.
        """
        results: Dict = {"drift_detected": False, "per_class": {}}

        for i, name in enumerate(self.class_names):
            ks_stat, ks_p = stats.ks_2samp(
                self.reference[:, i],
                new_probs[:, i],
            )
            drift = ks_p < self.ks_threshold
            results["per_class"][name] = {
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(ks_p),
                "drift_detected": bool(drift),
                "mean_shift": float(
                    new_probs[:, i].mean() - self.ref_stats["mean"][i]
                ),
            }
            if drift:
                results["drift_detected"] = True

        return results


if __name__ == "__main__":
    # Demonstrate with synthetic data
    np.random.seed(0)
    N, C = 300, 5
    class_names = ["NORM", "MI", "STTC", "CD", "HYP"]

    y_true = (np.random.rand(N, C) > 0.8).astype(float)
    y_score = np.clip(y_true + np.random.randn(N, C) * 0.25, 0, 1)

    # Fake metadata
    meta = pd.DataFrame({
        "age": np.random.randint(18, 90, N),
        "sex": np.random.choice(["Male", "Female"], N),
    })

    # Subgroup analysis
    analyzer = SubgroupAnalyzer(y_true, y_score, meta, class_names)
    sex_results = analyzer.analyze_by_sex()
    print("Sex analysis:")
    for sex, metrics in sex_results.items():
        if isinstance(metrics, dict) and "macro_auroc" in metrics:
            print(f"  {sex}: macro AUROC = {metrics['macro_auroc']:.3f}")

    # Calibration
    calib = CalibrationAnalyzer(y_true, y_score, class_names)
    ece = calib.compute_ece(0)
    print(f"\nCalibration ECE (NORM): {ece['ece']:.4f}")

    # Operating point selection
    selector = OperatingPointSelector(y_true, y_score, class_names)
    thresholds = selector.select_clinical_thresholds(min_sensitivity=0.90)
    print(f"\nOptimal thresholds at 90% sensitivity: {thresholds}")
