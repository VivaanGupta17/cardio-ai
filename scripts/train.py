#!/usr/bin/env python3
"""
Training script for CardioAI ECG arrhythmia detection models.

Usage:
    python scripts/train.py --config configs/ptbxl_config.yaml --model ecg_resnet
    python scripts/train.py --config configs/ptbxl_config.yaml --model ecg_transformer
    python scripts/train.py --config configs/ptbxl_config.yaml --model ecg_lstm
    python scripts/train.py --config configs/ptbxl_config.yaml --model ecg_resnet \
        --data_dir /data/ptb-xl --output_dir runs/experiment_1 --epochs 100

Multi-GPU training (DataParallel):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py \
        --config configs/ptbxl_config.yaml --model ecg_resnet
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from src.data.ptbxl_dataset import get_ptbxl_dataloaders
from src.models.ecg_lstm import ECGBiLSTM
from src.models.ecg_resnet import build_ecg_resnet
from src.models.ecg_transformer import build_ecg_transformer
from src.training.trainer import ECGTrainer


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(model_name: str, config: dict) -> torch.nn.Module:
    """Build model from name and config."""
    model_config = config.get("model", {})
    model_config["model_name"] = model_name

    if model_name == "ecg_resnet":
        model = build_ecg_resnet(model_config)
    elif model_name == "ecg_transformer":
        model = build_ecg_transformer(model_config)
    elif model_name == "ecg_lstm":
        model = ECGBiLSTM(
            num_classes=model_config.get("num_classes", 27),
            in_channels=model_config.get("in_channels", 12),
            dropout=model_config.get("dropout", 0.4),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                        "Choose from: ecg_resnet, ecg_transformer, ecg_lstm")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.getLogger().info(
        f"Model: {model_name} | Params: {total_params:,} | Trainable: {trainable:,}"
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train ECG arrhythmia detection models on PTB-XL"
    )
    parser.add_argument(
        "--config", type=str, default="configs/ptbxl_config.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--model", type=str, default="ecg_resnet",
        choices=["ecg_resnet", "ecg_transformer", "ecg_lstm"],
        help="Model architecture"
    )
    parser.add_argument("--data_dir", type=str, help="Override config data_dir")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint path")
    parser.add_argument("--label_type", type=str,
                       choices=["diagnostic_superclass", "rhythm", "form", "all"],
                       help="Label type")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["lr"] = args.lr
    if args.label_type:
        config["data"]["label_type"] = args.label_type

    exp_name = f"{args.model}_{config['data']['label_type']}"
    output_dir = args.output_dir or str(
        Path(config["output"]["base_dir"]) / exp_name
    )

    setup_logging(config["output"].get("log_level", "INFO"))
    logger = logging.getLogger(__name__)
    logger.info(f"CardioAI Training | Model: {args.model} | Output: {output_dir}")
    logger.info(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Load data
    logger.info("Loading PTB-XL dataset...")
    train_loader, val_loader, test_loader, class_names = get_ptbxl_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        sampling_rate=config["data"]["sampling_rate"],
        label_type=config["data"]["label_type"],
        num_workers=config["data"]["num_workers"],
        augment_train=True,
        cache_data=config["data"].get("cache_data", False),
    )
    logger.info(f"Classes ({len(class_names)}): {class_names}")
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Update num_classes from actual dataset
    config["model"]["num_classes"] = len(class_names)

    # Build model
    model = build_model(args.model, config)

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = torch.nn.DataParallel(model)

    # Resume from checkpoint
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model, _ = ECGTrainer.load_from_checkpoint(args.resume, model)

    # Get class weights from training dataset
    class_weights = train_loader.dataset.class_weights

    # Build trainer
    trainer = ECGTrainer(
        model=model,
        config=config["training"],
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
        class_weights=class_weights if config["training"].get("use_class_weights") else None,
        output_dir=output_dir,
    )

    # Train
    trainer.train()

    # Final evaluation on test set
    logger.info("=" * 60)
    logger.info("Final evaluation on held-out test set...")
    test_metrics = trainer.validate(test_loader)
    logger.info(f"Test macro AUROC: {test_metrics['macro_auroc']:.4f}")
    logger.info(f"Test macro F1:    {test_metrics['macro_f1']:.4f}")
    logger.info("=" * 60)

    import json
    test_results_path = Path(output_dir) / "test_results.json"
    with open(test_results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Test results saved to {test_results_path}")


if __name__ == "__main__":
    main()
