"""
Training Pipeline for Multi-Label ECG Arrhythmia Classification.

Features:
- Multi-label classification with BCE + Focal Loss
- Class-weighted loss for severe class imbalance (rare arrhythmias)
- Mixed precision training (AMP) for 2× speedup on A100/V100
- Gradient clipping + cosine annealing with warm restarts
- TensorBoard and JSON logging
- Automatic checkpoint saving (best AUROC)
- Early stopping with patience

Loss function considerations:
    Standard BCE: treats all classes equally → model ignores rare classes
    Weighted BCE: upweights rare classes → better but unstable at high weights
    Focal Loss: downweights easy (well-classified) samples → focuses on hard ones
    AsymmetricLoss: stronger penalty for missed positives (false negatives)
      → preferred for medical AI where false negatives are more costly

References:
    Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV.
    Ridnik et al. (2021). Asymmetric Loss For Multi-Label Classification. ICCV.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Loss Functions
# ─────────────────────────────────────────────

class WeightedBCELoss(nn.Module):
    """Binary cross-entropy with per-class positive weights.

    Handles the PTB-XL class imbalance where NORM (~30%) vastly
    outnumbers rare conditions like RBBB (~3%) or WPW (<1%).
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(1),
        )
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.class_weights.to(logits.device),
            reduction=self.reduction,
        )
        return loss


class FocalLoss(nn.Module):
    """Multi-label focal loss for ECG classification.

    Focal loss reduces the relative loss for well-classified examples
    (easy negatives dominating the gradient) and focuses learning on
    hard misclassified samples — which in ECG are often the rare but
    clinically dangerous arrhythmias (VT, VF, WPW).

    L_focal = -α * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: Per-class positive weight (default 0.25).
        gamma: Focusing parameter (default 2.0).
            gamma=0 → standard BCE
            gamma=2 → standard focal loss (Lin et al. 2017)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Focal weight: (1 - p_t)^gamma
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha balancing
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * bce

        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            focal_loss = focal_loss * (targets * w + (1 - targets))

        return focal_loss.mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification (Ridnik et al., 2021).

    In medical AI, false negatives (missed arrhythmias) are typically
    more costly than false positives. ASL handles this asymmetry by:
    - Using a stronger gamma_neg for negative (easy) samples
    - Clipping near-zero probabilities for negatives (probability shifting)

    gamma_neg > gamma_pos means negatives are downweighted more aggressively.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Probability shifting for negatives
        if self.clip > 0:
            probs_neg = (probs + self.clip).clamp(max=1)
        else:
            probs_neg = probs

        # Compute log-probabilities
        loss_pos = targets * torch.log(probs.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=self.eps))

        loss = loss_pos + loss_neg

        # Asymmetric focusing
        p_m = torch.exp(-loss)  # probability of correct classification
        factor_pos = (1 - p_m) ** self.gamma_pos
        factor_neg = p_m ** self.gamma_neg
        factor = targets * factor_pos + (1 - targets) * factor_neg

        loss = loss * factor
        return -loss.mean()


class CombinedLoss(nn.Module):
    """Combine BCE and Focal loss with configurable weights."""

    def __init__(
        self,
        focal_weight: float = 0.5,
        bce_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.focal = FocalLoss(class_weights=class_weights)
        self.bce = WeightedBCELoss(class_weights=class_weights)
        self.fw = focal_weight
        self.bw = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.fw * self.focal(logits, targets) + self.bw * self.bce(logits, targets)


# ─────────────────────────────────────────────
#  Metric Tracking
# ─────────────────────────────────────────────

class MetricTracker:
    """Running metric accumulator for training loop."""

    def __init__(self) -> None:
        self._data: Dict[str, List[float]] = {}

    def update(self, key: str, value: float) -> None:
        if key not in self._data:
            self._data[key] = []
        self._data[key].append(float(value))

    def mean(self, key: str) -> float:
        values = self._data.get(key, [0.0])
        return float(np.mean(values))

    def reset(self) -> None:
        self._data.clear()

    def summary(self) -> Dict[str, float]:
        return {k: self.mean(k) for k in self._data}


# ─────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────

class ECGTrainer:
    """Complete training pipeline for ECG multi-label classification.

    Usage:
        trainer = ECGTrainer(model, config, train_loader, val_loader)
        trainer.train(epochs=100)
        best_checkpoint = trainer.best_checkpoint_path

    Args:
        model: PyTorch model (ECGResNet, ECGTransformer, etc.)
        config: Training configuration dict.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        class_names: List of class label names.
        class_weights: Optional per-class loss weights.
        output_dir: Directory for checkpoints and logs.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_names: Optional[List[str]] = None,
        class_weights: Optional[torch.Tensor] = None,
        output_dir: str = "runs/default",
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        logger.info(f"Training on: {self.device}")
        self.model = self.model.to(self.device)

        # Loss function
        loss_type = config.get("loss", "focal")
        if loss_type == "focal":
            self.criterion = FocalLoss(
                alpha=config.get("focal_alpha", 0.25),
                gamma=config.get("focal_gamma", 2.0),
                class_weights=class_weights,
            )
        elif loss_type == "asl":
            self.criterion = AsymmetricLoss(
                gamma_neg=config.get("asl_gamma_neg", 4.0),
                gamma_pos=config.get("asl_gamma_pos", 1.0),
            )
        elif loss_type == "combined":
            self.criterion = CombinedLoss(class_weights=class_weights)
        else:
            self.criterion = WeightedBCELoss(class_weights=class_weights)

        # Optimizer
        lr = config.get("lr", 1e-3)
        weight_decay = config.get("weight_decay", 1e-4)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler
        steps_per_epoch = len(train_loader)
        max_epochs = config.get("epochs", 100)
        scheduler_type = config.get("scheduler", "cosine")

        if scheduler_type == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=lr,
                steps_per_epoch=steps_per_epoch,
                epochs=max_epochs,
                pct_start=0.1,
            )
            self.scheduler_step_per_batch = True
        else:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config.get("T_0", 20),
                T_mult=config.get("T_mult", 2),
                eta_min=config.get("min_lr", 1e-6),
            )
            self.scheduler_step_per_batch = False

        # Mixed precision
        self.use_amp = config.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Gradient clipping
        self.grad_clip = config.get("grad_clip", 1.0)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))

        # Training state
        self.best_auroc = 0.0
        self.best_checkpoint_path = self.output_dir / "best_model.pt"
        self.history: List[Dict] = []
        self.early_stop_patience = config.get("early_stop_patience", 15)
        self._patience_counter = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        tracker = MetricTracker()
        t0 = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            signals = batch["signal"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                out = self.model(signals)
                loss = self.criterion(out["logits"], labels)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_step_per_batch:
                self.scheduler.step()

            tracker.update("loss", loss.item())

            # Compute batch accuracy (exact match) for monitoring
            with torch.no_grad():
                preds = (out["probs"] > 0.5).float()
                exact_match = (preds == labels).all(dim=1).float().mean()
                tracker.update("exact_match", exact_match.item())

            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        if not self.scheduler_step_per_batch:
            self.scheduler.step()

        metrics = tracker.summary()
        metrics["epoch_time_s"] = time.time() - t0
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        return metrics

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Run validation and compute full metrics."""
        from src.evaluation.cardiac_metrics import compute_all_metrics

        self.model.eval()
        tracker = MetricTracker()

        all_probs = []
        all_labels = []

        for batch in loader:
            signals = batch["signal"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                out = self.model(signals)
                loss = self.criterion(out["logits"], labels)

            tracker.update("loss", loss.item())
            all_probs.append(out["probs"].cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        metrics = tracker.summary()
        clf_metrics = compute_all_metrics(labels, probs, self.class_names)
        metrics.update(clf_metrics)
        return metrics

    def train(self, epochs: Optional[int] = None) -> None:
        """Full training loop."""
        n_epochs = epochs or self.config.get("epochs", 100)
        logger.info(f"Starting training for {n_epochs} epochs")

        for epoch in range(1, n_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(self.val_loader)

            # Log to TensorBoard
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)

            # Checkpoint
            macro_auroc = val_metrics.get("macro_auroc", 0.0)
            is_best = macro_auroc > self.best_auroc

            if is_best:
                self.best_auroc = macro_auroc
                self._patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(f"  *** New best AUROC: {macro_auroc:.4f} ***")
            else:
                self._patience_counter += 1
                self.save_checkpoint(epoch, val_metrics, is_best=False)

            # Log epoch summary
            epoch_summary = {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss", 0.0),
                "val_loss": val_metrics.get("loss", 0.0),
                "val_macro_auroc": macro_auroc,
                "val_macro_f1": val_metrics.get("macro_f1", 0.0),
                "lr": train_metrics.get("lr", 0.0),
            }
            self.history.append(epoch_summary)
            self._save_history()

            logger.info(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val AUROC: {macro_auroc:.4f} | "
                f"LR: {train_metrics['lr']:.6f}"
            )

            # Early stopping
            if self._patience_counter >= self.early_stop_patience:
                logger.info(
                    f"Early stopping: no improvement for "
                    f"{self.early_stop_patience} epochs."
                )
                break

        logger.info(f"Training complete. Best val AUROC: {self.best_auroc:.4f}")
        self.writer.close()

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "class_names": self.class_names,
        }

        # Always save latest
        latest_path = self.output_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            torch.save(checkpoint, self.best_checkpoint_path)

        # Periodic saves
        if epoch % 10 == 0:
            torch.save(checkpoint, self.output_dir / f"checkpoint_epoch{epoch:04d}.pt")

    def _save_history(self) -> None:
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        model: nn.Module,
    ) -> Tuple[nn.Module, Dict]:
        """Load model weights from a training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint["metrics"]
