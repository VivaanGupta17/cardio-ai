#!/usr/bin/env python3
"""
Inference / prediction script for CardioAI.

Supports two modes:
    1. batch: Classify a set of ECG recordings, output predictions CSV
    2. streaming: Simulate real-time monitoring with alert generation

Usage:
    # Batch classification on a WFDB record
    python scripts/predict.py \
        --checkpoint runs/resnet/best_model.pt \
        --input_ecg data/patient_001.npy \
        --mode batch

    # Streaming monitoring simulation
    python scripts/predict.py \
        --checkpoint runs/resnet/best_model.pt \
        --input_ecg data/patient_001.npy \
        --mode streaming \
        --output_dir predictions/

    # Batch on directory of .npy files
    python scripts/predict.py \
        --checkpoint runs/resnet/best_model.pt \
        --input_dir data/ecg_records/ \
        --mode batch \
        --output_dir predictions/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

from src.inference.monitor import ECGMonitor, ECGChunk


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_ecg(path: str) -> np.ndarray:
    """Load ECG from various formats.

    Supports:
    - .npy: (12, T) numpy array
    - .npy: (T, 12) transposed (auto-detected)
    - WFDB format (requires wfdb package)
    """
    p = Path(path)
    if p.suffix == ".npy":
        x = np.load(p).astype(np.float32)
        if x.ndim == 2 and x.shape[0] != 12 and x.shape[1] == 12:
            x = x.T  # (T, 12) → (12, T)
        return x
    elif p.suffix in (".dat", ".hea"):
        import wfdb
        record = wfdb.rdsamp(str(p.with_suffix("")))
        return record[0].T.astype(np.float32)
    else:
        raise ValueError(f"Unsupported ECG format: {p.suffix}")


def batch_predict(
    model,
    ecg_files: list,
    class_names: list,
    device: torch.device,
    output_path: Path,
    batch_size: int = 32,
    fs: float = 500.0,
) -> pd.DataFrame:
    """Run batch prediction on a list of ECG files."""
    from src.data.ecg_preprocessing import ECGPreprocessor

    preprocessor = ECGPreprocessor(fs=fs)
    model.eval()
    results = []

    for ecg_path in ecg_files:
        try:
            ecg = load_ecg(ecg_path)
            # Preprocess
            proc = preprocessor.process(ecg)
            clean = proc["signal"]
            sqi = proc["sqi"]["overall_sqi"]

            # Normalize
            mean = clean.mean(axis=-1, keepdims=True)
            std = clean.std(axis=-1, keepdims=True) + 1e-8
            clean = (clean - mean) / std

            # Inference
            x = torch.tensor(clean[np.newaxis], dtype=torch.float32).to(device)
            with torch.no_grad():
                out = model(x)
            probs = out["probs"].cpu().numpy()[0]

            row = {
                "file": Path(ecg_path).name,
                "sqi": sqi,
                "predicted_class": class_names[probs.argmax()],
                "max_probability": float(probs.max()),
            }
            for i, name in enumerate(class_names):
                row[f"p_{name}"] = float(probs[i])
            results.append(row)

        except Exception as e:
            logging.warning(f"Failed to process {ecg_path}: {e}")
            results.append({"file": Path(ecg_path).name, "error": str(e)})

    df = pd.DataFrame(results)
    csv_path = output_path / "predictions.csv"
    df.to_csv(csv_path, index=False)
    logging.getLogger().info(f"Predictions saved to {csv_path}")
    return df


def streaming_predict(
    model,
    ecg: np.ndarray,
    class_names: list,
    device: torch.device,
    output_path: Path,
    fs: float = 500.0,
) -> dict:
    """Simulate real-time streaming monitoring on a recording."""
    monitor = ECGMonitor(
        model=model,
        class_names=class_names,
        fs=fs,
        window_size_s=10.0,
        step_size_s=2.0,
        device=device,
    )

    def on_alert(alert):
        if not alert.suppressed:
            print(f"  ALERT: {alert}")

    monitor.alert_callback = on_alert

    print(f"Streaming {ecg.shape[-1]/fs:.1f}s ECG...")
    summary = monitor.process_recording(ecg, chunk_size_s=2.0)

    # Save summary
    summary_path = output_path / "monitoring_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nMonitoring Summary:")
    print(f"  Total windows processed: {summary['total_windows_processed']}")
    print(f"  Alerts fired: {summary['total_alerts_fired']}")
    print(f"  Alerts suppressed: {summary['total_alerts_suppressed']}")
    print(f"  Suppression rate: {summary['suppression_rate']:.1%}")
    print(f"  Alarm burden: {summary['alarm_burden_per_hour']:.1f}/hour")
    print(f"\nSummary saved to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="CardioAI ECG inference")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--input_ecg", type=str, help="Single ECG file (.npy or WFDB)")
    parser.add_argument("--input_dir", type=str, help="Directory of ECG files")
    parser.add_argument("--mode", choices=["batch", "streaming"], default="batch")
    parser.add_argument("--output_dir", type=str, default="predictions")
    parser.add_argument("--fs", type=float, default=500.0, help="Sampling rate")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = checkpoint.get("class_names", [f"class_{i}" for i in range(5)])

    from src.models.ecg_resnet import build_ecg_resnet
    model_config = checkpoint.get("config", {})
    model_config["num_classes"] = len(class_names)
    model = build_ecg_resnet(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded | Classes: {class_names} | Device: {device}")

    # Collect ECG files
    if args.input_ecg:
        ecg_files = [args.input_ecg]
    elif args.input_dir:
        ecg_files = list(Path(args.input_dir).glob("*.npy"))
        logger.info(f"Found {len(ecg_files)} ECG files in {args.input_dir}")
    else:
        logger.error("Provide --input_ecg or --input_dir")
        sys.exit(1)

    if args.mode == "batch":
        df = batch_predict(model, ecg_files, class_names, device, output_dir, fs=args.fs)
        print(f"\nPredictions for {len(df)} recordings:")
        print(df.to_string())

    elif args.mode == "streaming":
        if not args.input_ecg:
            logger.error("Streaming mode requires --input_ecg (single recording)")
            sys.exit(1)
        ecg = load_ecg(args.input_ecg)
        streaming_predict(model, ecg, class_names, device, output_dir, fs=args.fs)


if __name__ == "__main__":
    main()
