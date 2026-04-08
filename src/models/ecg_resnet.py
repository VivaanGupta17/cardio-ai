"""
ECG ResNet: 1D Convolutional Residual Network for Multi-Label Arrhythmia Classification.

Architecture inspired by:
- He et al. (2016) Deep Residual Learning for Image Recognition
- Ribeiro et al. (2020) Automatic diagnosis of the 12-lead ECG using a deep neural network
  (Nature Communications) — iRhythm/Telehealth adaptation

Designed for PTB-XL 12-lead ECG classification (27 SCP diagnostic labels).
Input: (batch, 12 leads, 5000 samples) = 10 seconds @ 500 Hz
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
# ─────────────────────────────────────────────
#  Building Blocks
# ─────────────────────────────────────────────

class SEBlock1d(nn.Module):
    """Squeeze-and-Excitation block for channel-wise recalibration.

    Helps the network learn which ECG leads (or feature channels)
    are most informative for a given rhythm/morphology.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        s = self.pool(x).squeeze(-1)           # (B, C)
        s = self.fc(s).unsqueeze(-1)           # (B, C, 1)
        return x * s
class ResBlock1d(nn.Module):
    """Basic 1D residual block with optional SE attention.

    Uses pre-activation design (BN → ReLU → Conv) for better gradient flow,
    matching the Ribeiro et al. architecture that achieves cardiologist-level
    performance on 12-lead ECG.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.1,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False
        )
        self.dropout = nn.Dropout(dropout)
        self.se = SEBlock1d(out_channels) if use_se else nn.Identity()

        # Shortcut projection if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.se(out)

        return out + residual
class MultiScaleBlock1d(nn.Module):
    """Multi-scale feature extraction block using parallel convolutions.

    ECG signals contain clinically relevant features at different temporal scales:
    - P-wave: ~80–120 ms (40–60 samples @ 500 Hz)
    - QRS complex: ~60–100 ms (30–50 samples)
    - T-wave: ~160–300 ms (80–150 samples)
    - RR interval: ~600–1000 ms (300–500 samples)

    Using kernels of different sizes captures these multi-scale patterns simultaneously.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        assert out_channels % 4 == 0
        branch_ch = out_channels // 4

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(branch_ch), nn.ReLU(inplace=True),
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_channels, branch_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(branch_ch), nn.ReLU(inplace=True),
        )
        self.branch15 = nn.Sequential(
            nn.Conv1d(in_channels, branch_ch, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(branch_ch), nn.ReLU(inplace=True),
        )
        self.branch31 = nn.Sequential(
            nn.Conv1d(in_channels, branch_ch, kernel_size=31, padding=15, bias=False),
            nn.BatchNorm1d(branch_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.branch3(x),
            self.branch7(x),
            self.branch15(x),
            self.branch31(x),
        ], dim=1)
# ─────────────────────────────────────────────
#  Main Model
# ─────────────────────────────────────────────

class ECGResNet(nn.Module):
    """1D ResNet for multi-label 12-lead ECG classification.

    Architecture stages:
        Stem → Stage1 → Stage2 → Stage3 → Stage4 → GAP → Classifier

    Temporal downsampling via strided convolutions (not pooling) preserves
    fine-grained morphological features important for diagnosing subtle
    ST changes and P-wave abnormalities.

    Args:
        num_classes: Number of output labels (default 27 for PTB-XL).
        in_channels: Number of ECG leads (default 12).
        base_filters: Base filter count; doubled at each stage.
        dropout: Dropout rate before classifier.
        use_se: Whether to use SE attention in residual blocks.
    """

    STAGE_CONFIG: List[Tuple[int, int, int]] = [
        # (num_blocks, out_channels, stride)
        (3, 64,  2),
        (4, 128, 2),
        (6, 256, 2),
        (3, 512, 2),
    ]

    def __init__(
        self,
        num_classes: int = 27,
        in_channels: int = 12,
        base_filters: int = 64,
        dropout: float = 0.5,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Stem: initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_filters, base_filters, kernel_size=15, padding=7, bias=False),
        )

        # Multi-scale input mixing
        self.ms_block = MultiScaleBlock1d(base_filters, base_filters)

        # Residual stages
        stages = []
        prev_channels = base_filters
        for n_blocks, out_ch, stride in self.STAGE_CONFIG:
            blocks = []
            for i in range(n_blocks):
                s = stride if i == 0 else 1
                ic = prev_channels if i == 0 else out_ch
                blocks.append(ResBlock1d(ic, out_ch, stride=s, use_se=use_se))
            stages.append(nn.Sequential(*blocks))
            prev_channels = out_ch
        self.stages = nn.ModuleList(stages)

        # Global average pooling + classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(prev_channels, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings (for transfer learning, visualization)."""
        x = self.stem(x)
        x = self.ms_block(x)
        for stage in self.stages:
            x = stage(x)
        x = self.gap(x).squeeze(-1)
        return x

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: ECG tensor of shape (B, 12, T).
            return_features: If True, also return penultimate embeddings.

        Returns:
            dict with keys:
                'logits': raw logits (B, num_classes)
                'probs': sigmoid probabilities (B, num_classes)
                'features': embeddings if return_features=True
        """
        features = self.extract_features(x)
        features_dropped = self.dropout(features)
        logits = self.classifier(features_dropped)

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probs": torch.sigmoid(logits),
        }
        if return_features:
            out["features"] = features
        return out
# ─────────────────────────────────────────────
#  Gradient-CAM Wrapper (interpretability)
# ─────────────────────────────────────────────

class GradCAMWrapper(nn.Module):
    """Wraps ECGResNet to compute Grad-CAM saliency maps over time.

    Produces temporal saliency maps showing which time segments
    drive each arrhythmia prediction — useful for clinical validation
    and for demonstrating interpretability to FDA reviewers.
    """

    def __init__(self, model: ECGResNet, target_layer_idx: int = -1) -> None:
        super().__init__()
        self.model = model
        self.target_layer = model.stages[target_layer_idx][-1]
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def save_gradient(grad: torch.Tensor) -> None:
            self._gradients = grad

        def save_activation(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
            self._activations = out
            out.register_hook(save_gradient)

        self.target_layer.register_forward_hook(save_activation)

    def forward(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            probs: (B, num_classes)
            cam: (B, T_original) normalized saliency map
        """
        out = self.model(x)
        probs = out["probs"]

        if class_idx is None:
            class_idx = probs.argmax(dim=1)[0].item()

        self.model.zero_grad()
        score = out["logits"][:, class_idx].sum()
        score.backward()

        # CAM: average over channels, ReLU
        grads = self._gradients            # (B, C, T')
        acts = self._activations           # (B, C, T')
        weights = grads.mean(dim=-1, keepdim=True)
        cam = (weights * acts).sum(dim=1)  # (B, T')
        cam = F.relu(cam)

        # Upsample to original signal length
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-1], mode="linear", align_corners=False)
        cam = cam.squeeze(1)
        # Normalize per sample
        cam_min = cam.amin(dim=-1, keepdim=True)
        cam_max = cam.amax(dim=-1, keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return probs, cam

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Memory-efficient inference (no gradient tracking)."""
        self.eval()
        return self.forward(x)

# ─────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────

def build_ecg_resnet(config: dict) -> ECGResNet:
    """Build ECGResNet from a config dict (e.g., loaded from YAML)."""
    return ECGResNet(
        num_classes=config.get("num_classes", 27),
        in_channels=config.get("in_channels", 12),
        base_filters=config.get("base_filters", 64),
        dropout=config.get("dropout", 0.5),
        use_se=config.get("use_se", True),
    )
# ─────────────────────────────────────────────
#  Sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model = ECGResNet(num_classes=27)
    x = torch.randn(4, 12, 5000)  # batch=4, 12 leads, 10s @ 500Hz
    out = model(x, return_features=True)
    print(f"Logits:   {out['logits'].shape}")   # (4, 27)
    print(f"Probs:    {out['probs'].shape}")    # (4, 27)
    print(f"Features: {out['features'].shape}") # (4, 512)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
