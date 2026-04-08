"""
ECG Transformer: Lead-Wise Self-Attention for 12-Lead Arrhythmia Classification.

Architecture design rationale:
- Treats each ECG lead as a "token" (analogous to words in NLP)
- Self-attention enables the model to learn inter-lead relationships
  (e.g., reciprocal ST changes across inferior/lateral leads in MI)
- Chunk-based temporal encoding handles 5000-sample sequences efficiently

Key clinical insight: Many arrhythmias and conduction defects manifest
differently across leads. Inferior MI elevates ST in II, III, aVF but
depresses in I, aVL — the attention mechanism captures these correlations
without hand-crafted feature engineering.

References:
    Vaswani et al. (2017) Attention Is All You Need
    Li et al. (2022) TransECG: A novel deep transformer architecture
    Chen et al. (2021) Self-supervised ECG representation learning
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  Positional Encodings
# ─────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for temporal position.

    Using fixed (non-learned) encoding ensures the model generalizes
    to different ECG lengths at inference time — important for
    clinical deployment where recording durations vary.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LeadPositionalEncoding(nn.Module):
    """Learned positional encoding for the 12 ECG leads.

    Encodes anatomical position of leads:
    - Limb leads (I, II, III, aVR, aVL, aVF): frontal plane
    - Precordial leads (V1–V6): horizontal plane

    The learned encoding allows the model to use lead identity
    as an inductive bias without hard-coding anatomical assumptions.
    """

    def __init__(self, n_leads: int = 12, d_model: int = 256) -> None:
        super().__init__()
        # Lead ordering: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        self.encoding = nn.Embedding(n_leads, d_model)
        self.register_buffer("positions", torch.arange(n_leads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_leads, d_model)
        return x + self.encoding(self.positions).unsqueeze(0)


# ─────────────────────────────────────────────
#  Temporal Feature Extractor
# ─────────────────────────────────────────────

class TemporalPatchEmbedding(nn.Module):
    """Convert raw ECG time series to patch embeddings.

    Splits each lead into non-overlapping patches and projects
    to d_model dimensions. Analogous to ViT's patch projection.

    For ECG at 500 Hz / 10s = 5000 samples:
    - patch_size=50 → 100 patches per lead (50ms resolution)
    - patch_size=100 → 50 patches per lead (100ms resolution — 1 cardiac cycle subunit)
    """

    def __init__(
        self,
        patch_size: int = 50,
        d_model: int = 256,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, d_model // 2, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)  — single lead
        return self.proj(x).transpose(1, 2)  # (B, n_patches, d_model)


# ─────────────────────────────────────────────
#  Transformer Components
# ─────────────────────────────────────────────

class ECGTransformerEncoderLayer(nn.Module):
    """Custom Transformer encoder layer with pre-layer normalization.

    Pre-LN (norm first) is more stable for training deep transformers
    without learning rate warmup (Xiong et al., 2020).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(
            normed, normed, normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out

        # Feed-forward with pre-norm
        x = x + self.ff(self.norm2(x))
        return x, attn_weights


# ─────────────────────────────────────────────
#  Main Model: Two-Level Transformer
# ─────────────────────────────────────────────

class ECGTransformer(nn.Module):
    """Two-level Transformer for 12-lead ECG classification.

    Level 1 (Temporal Transformer): Attends over time patches within each lead.
    Level 2 (Lead Transformer): Attends over 12 leads, capturing inter-lead
        correlations (e.g., reciprocal changes in MI, axis deviation).

    This hierarchical design:
    1. Reduces computational cost vs full 5000×5000 attention
    2. Reflects ECG's natural structure (temporal + spatial lead layout)
    3. Allows visualizing which leads drive the prediction

    Args:
        num_classes: Output labels (27 for PTB-XL).
        n_leads: Number of ECG leads (12).
        d_model: Transformer embedding dimension.
        n_heads: Number of attention heads.
        n_temporal_layers: Transformer layers for temporal encoding.
        n_lead_layers: Transformer layers for cross-lead encoding.
        patch_size: Samples per temporal patch.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_classes: int = 27,
        n_leads: int = 12,
        d_model: int = 256,
        n_heads: int = 8,
        n_temporal_layers: int = 4,
        n_lead_layers: int = 4,
        patch_size: int = 50,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_leads = n_leads
        self.d_model = d_model

        # Per-lead temporal patch embedding (shared weights across leads)
        self.patch_embed = TemporalPatchEmbedding(patch_size=patch_size, d_model=d_model)

        # Temporal positional encoding
        self.temporal_pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        # CLS token for temporal aggregation
        self.temporal_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.temporal_cls, std=0.02)

        # Temporal Transformer (per-lead)
        self.temporal_encoder = nn.ModuleList([
            ECGTransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_temporal_layers)
        ])

        # Lead-level positional encoding (learned, anatomical)
        self.lead_pos_enc = LeadPositionalEncoding(n_leads, d_model)

        # CLS token for final classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Lead Transformer (cross-lead interactions)
        self.lead_encoder = nn.ModuleList([
            ECGTransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_lead_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode_lead(self, lead_signal: torch.Tensor) -> torch.Tensor:
        """Encode a single lead's temporal signal.

        Args:
            lead_signal: (B, 1, T)

        Returns:
            (B, d_model) — CLS token representation of this lead
        """
        x = self.patch_embed(lead_signal)             # (B, n_patches, d_model)
        x = self.temporal_pos_enc(x)

        # Prepend CLS token
        cls = self.temporal_cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)               # (B, 1+n_patches, d_model)

        attn_weights_list = []
        for layer in self.temporal_encoder:
            x, w = layer(x)
            attn_weights_list.append(w)

        return x[:, 0, :]  # CLS token: (B, d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the two-level ECG transformer.

        Args:
            x: Input tensor of shape (B, 12, T) where B is batch size,
               12 is the number of leads, and T is the sequence length
               (typically 5000 for 10 s at 500 Hz).
            return_attention: If True, include per-lead attention weights
               in the output dict. Useful for interpretability.

        Returns:
            dict with keys 'logits' (B, num_classes), 'probs' (B, num_classes),
            and optionally 'lead_attention' (B, 12) if return_attention=True.
        """
        B, L, T = x.shape
        assert L == self.n_leads

        # Encode each lead independently (shared weights → efficient)
        lead_tokens = []
        for i in range(L):
            lead_repr = self.encode_lead(x[:, i : i + 1, :])  # (B, d_model)
            lead_tokens.append(lead_repr)
        lead_tokens = torch.stack(lead_tokens, dim=1)  # (B, 12, d_model)

        # Add lead-wise positional encoding
        lead_tokens = self.lead_pos_enc(lead_tokens)   # (B, 12, d_model)

        # Prepend global CLS token
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, lead_tokens], dim=1)     # (B, 13, d_model)

        lead_attn_weights = []
        for layer in self.lead_encoder:
            seq, w = layer(seq)
            lead_attn_weights.append(w)

        # CLS token → classification
        cls_repr = self.norm(seq[:, 0, :])
        cls_repr = self.dropout(cls_repr)
        logits = self.classifier(cls_repr)

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probs": torch.sigmoid(logits),
        }
        if return_attention:
            # Lead attention weights from last layer: (B, n_heads, 13, 13)
            # Slice CLS → lead tokens: (B, n_heads, 12)
            out["lead_attention"] = lead_attn_weights[-1][:, :, 0, 1:]
        return out

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Memory-efficient inference wrapper (no gradient tracking).

        Use this instead of forward() during deployment to avoid
        storing intermediate activations.
        """
        self.eval()
        return self.forward(x, return_attention=False)


# ─────────────────────────────────────────────
#  Lead Attention Visualization
# ─────────────────────────────────────────────

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def get_lead_importance(
    model: ECGTransformer,
    x: torch.Tensor,
) -> Dict[str, float]:
    """Compute per-lead importance scores from attention weights.

    Aggregates attention from CLS token to each lead across all heads.
    Returns a dict mapping lead name to importance score (sums to 1).

    Clinically interpretable: should show high attention to inferior
    leads (II, III, aVF) for inferior MI, precordial leads (V1–V4)
    for LBBB, etc.
    """
    model.eval()
    with torch.no_grad():
        out = model(x, return_attention=True)

    # (B, n_heads, 12) — mean over batch and heads
    attn = out["lead_attention"].mean(dim=[0, 1])  # (12,)
    attn = attn / attn.sum()
    return {name: score.item() for name, score in zip(LEAD_NAMES, attn)}


# ─────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────

def build_ecg_transformer(config: dict) -> ECGTransformer:
    return ECGTransformer(
        num_classes=config.get("num_classes", 27),
        n_leads=config.get("in_channels", 12),
        d_model=config.get("d_model", 256),
        n_heads=config.get("n_heads", 8),
        n_temporal_layers=config.get("n_temporal_layers", 4),
        n_lead_layers=config.get("n_lead_layers", 4),
        patch_size=config.get("patch_size", 50),
        d_ff=config.get("d_ff", 1024),
        dropout=config.get("dropout", 0.1),
    )


if __name__ == "__main__":
    model = ECGTransformer(num_classes=27)
    x = torch.randn(2, 12, 5000)
    out = model(x, return_attention=True)
    print(f"Logits:         {out['logits'].shape}")        # (2, 27)
    print(f"Probs:          {out['probs'].shape}")         # (2, 27)
    print(f"Lead attention: {out['lead_attention'].shape}") # (2, 8, 12)

    importances = get_lead_importance(model, x[:1])
    print("Lead importances:", importances)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
