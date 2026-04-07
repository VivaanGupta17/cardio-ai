"""
Multi-Sensor Fusion for Cardiac Monitoring.

Fuses ECG, accelerometer (motion/activity), and thoracic impedance signals
to predict cardiac events — analogous to Boston Scientific HeartLogic HF
prediction system, which monitors:

    HeartLogic index = f(ECG, S1/S2 heart sounds, thoracic impedance,
                         respiration, heart rate, activity level)

Clinical motivation:
    Single-sensor systems are limited:
    - ECG alone: misses fluid overload (needs impedance)
    - Impedance alone: affected by respiration, body position
    - Activity alone: doesn't capture ischemia
    Multi-sensor fusion reduces false positives by 56% in HF monitoring
    (Boston Scientific HeartLogic pivotal trial, HeartLogic score ≥ 16).

References:
    Abraham et al. (2018) Sustained efficacy of pulmonary artery pressure
        to guide adjustment of chronic heart failure therapy. JACC HF.
    Boehmer et al. (2017) A Multisensor Algorithm Predicts Heart Failure
        Events in Patients With Implanted Devices. JACC HF.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  Per-Modality Encoders
# ─────────────────────────────────────────────

class ECGEncoder(nn.Module):
    """Lightweight ECG encoder for use in fusion model.

    Extracts 256-d embedding from 12-lead ECG.
    Shares architectural pattern with ECGResNet but smaller
    for efficient multi-modal training.
    """

    def __init__(self, in_channels: int = 12, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),

            nn.Conv1d(256, out_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B, out_dim)


class AccelerometerEncoder(nn.Module):
    """3-axis accelerometer encoder (activity/motion).

    Captures:
    - Physical activity level (key HF decompensation indicator)
    - Body position (upright vs. supine — orthopnea detection)
    - Cardiac motion (seismocardiogram component at high sampling rates)

    Input: (B, 3, T) — 3-axis accel at ~32–64 Hz
    """

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),

            nn.Conv1d(64, out_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )
        # Activity-level feature (scalar summary statistics)
        self.activity_head = nn.Linear(4, 32)  # [mean, std, max, entropy]

    def forward(
        self,
        x: torch.Tensor,
        activity_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, 3, T)
            activity_stats: (B, 4) optional scalar features
        """
        feat = self.net(x).squeeze(-1)  # (B, out_dim)
        if activity_stats is not None:
            act_feat = F.relu(self.activity_head(activity_stats))  # (B, 32)
            feat = torch.cat([feat, act_feat], dim=1)
        return feat


class ImpedanceEncoder(nn.Module):
    """Thoracic impedance encoder (fluid status).

    Thoracic impedance decreases as pulmonary fluid increases.
    Monitoring trends (not absolute values) is key for HF management.

    Captures:
    - Respiratory rate (impedance oscillates with breathing)
    - Tidal volume proxy
    - Pulmonary fluid trend (slower 24–48h timescale)

    Input: (B, 1, T) — impedance waveform at ~64 Hz
    """

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(16), nn.ReLU(inplace=True),

            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),

            nn.Conv1d(64, out_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────
#  Cross-Modal Attention
# ─────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """Attention mechanism for cross-modal feature interaction.

    Allows each modality to attend to relevant features in other modalities.
    Example: The impedance encoder can "look at" the ECG features to
    contextualize whether high impedance correlates with low HR (good prognosis)
    or high HR (decompensation).

    Uses query-key-value attention with each modality as both Q and K/V targets.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(self.d_head)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, d_model) — modality seeking context
            context: (B, N, d_model) — N other modality tokens

        Returns:
            (B, d_model) — enriched query representation
        """
        B, N, D = context.shape
        q = self.q_proj(query).unsqueeze(1)   # (B, 1, D)
        k = self.k_proj(context)               # (B, N, D)
        v = self.v_proj(context)               # (B, N, D)

        # Reshape for multi-head attention
        q = q.view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, 1, N)
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, 1, D).squeeze(1)
        out = self.out_proj(out)

        return self.norm(query + out)


import math  # noqa: E402 — needed for CrossModalAttention


# ─────────────────────────────────────────────
#  Fusion Strategies
# ─────────────────────────────────────────────

class EarlyFusion(nn.Module):
    """Early fusion: concatenate raw signals, then process jointly.

    Pros: Can learn cross-modal correlations at the signal level.
    Cons: Requires all modalities to be time-aligned and same resolution.
    Suitable for: Synchronized acquisition (ICM, Holter with sensors).
    """

    def __init__(self, n_channels_total: int = 16, out_dim: int = 256) -> None:
        """
        Args:
            n_channels_total: Sum of channels across all modalities.
                ECG (12) + Accel (3) + Impedance (1) = 16
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels_total, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),

            nn.Conv1d(256, out_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

    def forward(
        self,
        ecg: torch.Tensor,
        accel: torch.Tensor,
        impedance: torch.Tensor,
    ) -> torch.Tensor:
        """All inputs must be same temporal length T."""
        x = torch.cat([ecg, accel, impedance], dim=1)
        return self.encoder(x).squeeze(-1)


class LateFusion(nn.Module):
    """Late fusion: encode each modality separately, fuse embeddings.

    Pros: Handles missing modalities gracefully (zero-out missing).
    Cons: Loses low-level cross-modal correlations.
    Suitable for: Asynchronous data, missing modality scenarios (ICM vs. CRT-D).
    """

    def __init__(
        self,
        ecg_dim: int = 256,
        accel_dim: int = 128,
        impedance_dim: int = 128,
        out_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        in_dim = ecg_dim + accel_dim + impedance_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        ecg_feat: torch.Tensor,
        accel_feat: torch.Tensor,
        impedance_feat: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            *_feat: Modality embeddings.
            modality_mask: (B, 3) binary mask, 0 = modality missing.
                           Missing modalities are zeroed out.
        """
        if modality_mask is not None:
            ecg_feat = ecg_feat * modality_mask[:, 0:1]
            accel_feat = accel_feat * modality_mask[:, 1:2]
            impedance_feat = impedance_feat * modality_mask[:, 2:3]

        fused = torch.cat([ecg_feat, accel_feat, impedance_feat], dim=1)
        return self.fusion_mlp(fused)


# ─────────────────────────────────────────────
#  Main Fusion Model
# ─────────────────────────────────────────────

class MultiSensorFusionModel(nn.Module):
    """Complete multi-sensor fusion model for cardiac event prediction.

    Implements both early and late fusion with cross-modal attention,
    and combines them with a learned gating mechanism.

    HeartLogic-inspired: predicts 14-day heart failure event risk
    (analagous to HeartLogic index threshold crossing).

    Args:
        num_classes: Output labels (HF event, AF onset, etc.).
        fusion_mode: 'late', 'early', or 'hybrid'.
        ecg_channels: ECG lead count (12).
    """

    def __init__(
        self,
        num_classes: int = 5,
        fusion_mode: str = "hybrid",
        ecg_channels: int = 12,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        assert fusion_mode in ("late", "early", "hybrid")
        self.fusion_mode = fusion_mode
        self.num_classes = num_classes

        # Modality-specific encoders
        self.ecg_encoder = ECGEncoder(ecg_channels, out_dim=256)
        self.accel_encoder = AccelerometerEncoder(out_dim=128)
        self.impedance_encoder = ImpedanceEncoder(out_dim=128)

        # Project all modalities to same dimension for attention
        self.ecg_proj = nn.Linear(256, 256)
        self.accel_proj = nn.Linear(128, 256)
        self.impedance_proj = nn.Linear(128, 256)

        # Cross-modal attention: each modality attends to the others
        self.ecg_cross_attn = CrossModalAttention(256)
        self.accel_cross_attn = CrossModalAttention(256)
        self.impedance_cross_attn = CrossModalAttention(256)

        if fusion_mode in ("early", "hybrid"):
            # Early fusion path (requires same-resolution signals)
            self.early_fusion = EarlyFusion(n_channels_total=16, out_dim=256)

        if fusion_mode == "hybrid":
            # Gating network: learn to blend early and late fusion
            self.gate = nn.Sequential(
                nn.Linear(256 + 256, 2),
                nn.Softmax(dim=-1),
            )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # HeartLogic-style composite index head (scalar risk score)
        self.risk_score_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def _encode_modalities(
        self,
        ecg: torch.Tensor,
        accel: torch.Tensor,
        impedance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ecg_feat = self.ecg_encoder(ecg)
        accel_feat = self.accel_encoder(accel)
        impedance_feat = self.impedance_encoder(impedance)

        # Project to common dimension
        e = self.ecg_proj(ecg_feat)
        a = self.accel_proj(accel_feat)
        z = self.impedance_proj(impedance_feat)

        # Cross-modal attention (each attends to the other two)
        context = torch.stack([e, a, z], dim=1)  # (B, 3, 256)
        e_refined = self.ecg_cross_attn(e, context)
        a_refined = self.accel_cross_attn(a, context)
        z_refined = self.impedance_cross_attn(z, context)

        return e_refined, a_refined, z_refined

    def forward(
        self,
        ecg: torch.Tensor,
        accel: torch.Tensor,
        impedance: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        return_risk_score: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            ecg: (B, 12, T_ecg)
            accel: (B, 3, T_accel)
            impedance: (B, 1, T_imp)
            modality_mask: (B, 3) — optional mask for missing modalities
            return_risk_score: Include scalar HeartLogic-style index.

        Returns:
            dict: logits, probs, [risk_score], [fusion_gate]
        """
        e, a, z = self._encode_modalities(ecg, accel, impedance)

        if self.fusion_mode == "late":
            fused = (e + a + z) / 3.0

        elif self.fusion_mode == "early":
            # Resample all to same length (use ECG length as reference)
            T = ecg.shape[-1]
            accel_rs = F.interpolate(accel, size=T, mode="linear", align_corners=False)
            impedance_rs = F.interpolate(impedance, size=T, mode="linear", align_corners=False)
            fused = self.early_fusion(ecg, accel_rs, impedance_rs)

        else:  # hybrid
            # Late path: cross-modal attention average
            late = (e + a + z) / 3.0

            # Early path
            T = ecg.shape[-1]
            accel_rs = F.interpolate(accel, size=T, mode="linear", align_corners=False)
            impedance_rs = F.interpolate(impedance, size=T, mode="linear", align_corners=False)
            early = self.early_fusion(ecg, accel_rs, impedance_rs)

            # Gating
            gate = self.gate(torch.cat([early, late], dim=1))  # (B, 2)
            fused = gate[:, 0:1] * early + gate[:, 1:2] * late

        logits = self.classifier(fused)
        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probs": torch.sigmoid(logits),
        }

        if return_risk_score:
            out["risk_score"] = self.risk_score_head(fused)  # (B, 1) — 0 to 1

        if self.fusion_mode == "hybrid":
            out["fusion_gate"] = gate  # (B, 2): [early_weight, late_weight]

        return out


if __name__ == "__main__":
    model = MultiSensorFusionModel(num_classes=5, fusion_mode="hybrid")

    # Simulate: 10s ECG @ 500Hz, 10s accel @ 64Hz, 10s impedance @ 64Hz
    ecg = torch.randn(2, 12, 5000)
    accel = torch.randn(2, 3, 640)
    impedance = torch.randn(2, 1, 640)

    out = model(ecg, accel, impedance)
    print(f"Logits:     {out['logits'].shape}")
    print(f"Risk score: {out['risk_score'].shape}")
    print(f"Fusion gate: {out['fusion_gate']}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
