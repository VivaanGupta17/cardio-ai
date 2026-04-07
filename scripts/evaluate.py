#!/usr/bin/env python3
"""
Evaluation script: comprehensive clinical performance analysis.

Generates:
- Per-class AUROC, AUPRC, F1, sensitivity/specificity tables
- Bootstrap confidence intervals
- Demographic subgroup analysis (sex, age)
- Calibration metrics
- Alert fatigue simulation
- Clinical operating point analysis
- Summary JSON and formatted report

Usage:
    python scripts/evaluate.py \
        --checkpoint runs/resnet_ptbxl/best_model.pt \
        --data_dir /data/ptb-xl \
        --output_dir results/resnet_eval/

    # With subgroup analysis
    python scripts/evaluate.py \
        --checkpoint runs/resnet_ptbxl/best_model.pt \
        --data_dir /data/ptb-xl \
        --subgroup_analysis \
        --bootstrap_n 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast

from src.evaluation.cardiac_metrics import (
    bootstrap_auroc_ci,
    compute_all_metrics,
    compute_alert_fatigue_metrics,
    per_class_confusion,
    sensitivity_at_specificity,
)
from src.evaluation.clinical_analysis import (
    CalibrationAnalyzer,
    OperatingPointSelector,
    SubgroupAnalyzer,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_checkpoint(checkpoint_path: str) -> dict:
    return torch.load(checkpoint_path, map_location="cpu")


def run_inference(model, loader, device) -> tuple:
    """Run model inference on a DataLoader, returning probs, labels, metadata."""
    model.eval()
    all_probs, all_labels = [], []
    all_ages, all_sexes = [], []
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch in loader:
            signals = batch["signal"].to(device)
            with autocast(enabled=use_amp):
                out = model(signals)
            all_probs.append(out["probs"].cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
            all_ages.extend(batch["age"].numpy().tolist())
            all_sexes.extend(batch["sex"].numpy().tolist())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return probs, labels, np.array(all_ages), np.array(all_sexes)


def print_metrics_table(metrics: dict, class_names: list) -> None:
    """Pretty-print per-class metrics table."""
    header = f"{'Class':<15} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'Sensitivity':>12}"
    print(header)
    print("-" * len(header))

    for name in class_names:
        auroc = metrics.get(f"auroc_{name}", float("nan"))
        auprc = metrics.get(f"auprc_{name}", float("nan"))
        f1 = metrics.get(f"f1_{name}", float("nan"))
        sens = metrics.get(f"macro_sensitivity_at_spec90", float("nan"))
        print(f"{name:<15} {auroc:>8.4f} {auprc:>8.4f} {f1:>8.4f} {sens:>12.4f}")

    print("-" * len(header))
    print(f"{'Macro Avg':<15} {metrics.get('macro_auroc', 0):>8.4f} "
          f"{metrics.get('macro_auprc', 0):>8.4f} "
          f"{metrics.get('macro_f1', 0):>8.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ECG arrhythmia model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/ptbxl_config.yaml")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    parser.add_argument("--split", default="test", choices=["test", "val", "all"])
    parser.add_argument("--subgroup_analysis", action="store_true")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--alert_simulation", action="store_true")
    parser.add_argument("--bootstrap_n", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)
    class_names = checkpoint.get("class_names", config["model"].get("class_names", []))

    # Build model
    from src.models.ecg_resnet import build_ecg_resnet
    model_config = checkpoint.get("config", config["model"])
    model_config["num_classes"] = len(class_names)
    model = build_ecg_resnet(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Loaded {type(model).__name__} | Device: {device}")

    # Load data
    from src.data.ptbxl_dataset import PTBXLDataset
    from torch.utils.data import DataLoader

    dataset = PTBXLDataset(
        config["data"]["data_dir"],
        split=args.split,
        sampling_rate=config["data"]["sampling_rate"],
        label_type=config["data"]["label_type"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logger.info(f"Evaluating on {args.split} set: {len(dataset)} samples")

    # Run inference
    logger.info("Running inference...")
    probs, labels, ages, sexes = run_inference(model, loader, device)

    # Core metrics
    logger.info("Computing metrics...")
    metrics = compute_all_metrics(labels, probs, class_names)

    print("\n" + "=" * 60)
    print("CARDIOAI EVALUATION RESULTS")
    print("=" * 60)
    print_metrics_table(metrics, class_names)
    print(f"\nMacro AUROC: {metrics['macro_auroc']:.4f}")
    print(f"Macro AUPRC: {metrics['macro_auprc']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")

    # Bootstrap CI
    logger.info(f"Computing bootstrap CIs (n={args.bootstrap_n})...")
    ci = bootstrap_auroc_ci(labels, probs, n_bootstrap=args.bootstrap_n)
    print(f"\nAUROC 95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    metrics["bootstrap_ci"] = ci

    # Per-class confusion matrices
    confusion_results = per_class_confusion(labels, probs, class_names=class_names)
    metrics["confusion_per_class"] = confusion_results

    # Clinical operating points
    print("\nSensitivity at Clinically Relevant Specificities:")
    print(f"{'Class':<15} {'Sens@Spec85':>12} {'Sens@Spec90':>12} {'Sens@Spec95':>12}")
    print("-" * 53)
    for i, name in enumerate(class_names):
        if labels[:, i].sum() >= 10:
            r85 = sensitivity_at_specificity(labels[:, i], probs[:, i], 0.85)
            r90 = sensitivity_at_specificity(labels[:, i], probs[:, i], 0.90)
            r95 = sensitivity_at_specificity(labels[:, i], probs[:, i], 0.95)
            print(f"{name:<15} {r85['sensitivity']:>12.4f} {r90['sensitivity']:>12.4f} "
                  f"{r95['sensitivity']:>12.4f}")

    # Alert fatigue simulation
    if args.alert_simulation:
        print("\nAlert Fatigue Analysis (simulated 1-hour monitoring):")
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            for i, name in enumerate(class_names[:2]):  # Show first 2 classes
                if labels[:, i].sum() >= 10:
                    af = compute_alert_fatigue_metrics(
                        labels[:, i], probs[:, i], thresh, recording_duration_hours=1.0
                    )
                    print(f"  {name} @ thr={thresh}: "
                          f"FAR={af['false_alarm_rate_per_hour']:.2f}/hr "
                          f"Sens={af['sensitivity']:.3f} "
                          f"PPV={af['ppv']:.3f}")

    # Subgroup analysis
    if args.subgroup_analysis:
        import pandas as pd
        meta = pd.DataFrame({
            "age": ages,
            "sex": ["Male" if s == 1 else "Female" for s in sexes],
        })
        analyzer = SubgroupAnalyzer(labels, probs, meta, class_names)
        sex_results = analyzer.analyze_by_sex()
        age_results = analyzer.analyze_by_age_group()

        print("\nSubgroup Analysis — By Sex:")
        for sex, m in sex_results.items():
            if isinstance(m, dict) and "macro_auroc" in m:
                print(f"  {sex}: macro AUROC = {m['macro_auroc']:.4f} (n={m.get('n_samples', '?')})")

        metrics["subgroup_sex"] = sex_results
        metrics["subgroup_age"] = age_results

    # Calibration
    if args.calibration:
        calib = CalibrationAnalyzer(labels, probs, class_names)
        ece_results = calib.compute_all_ece()
        print("\nCalibration (Expected Calibration Error):")
        for name, r in ece_results.items():
            print(f"  {name}: ECE={r['ece']:.4f}, Brier={r['brier_score']:.4f}")
        metrics["calibration"] = ece_results

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(metrics), f, indent=2)

    logger.info(f"Results saved to {results_path}")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
