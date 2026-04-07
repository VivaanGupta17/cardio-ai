"""
ECG Bidirectional LSTM: Recurrent Baseline for Arrhythmia Classification.

Bidirectional LSTMs capture both past and future context in ECG signals,
which is important for:
- P-wave detection (precedes QRS)
- ST segment analysis (follows QRS)
- Long-range dependencies (RR interval patterns in AF, Mobitz II)

This serves as a strong baseline and is useful for:
1. Ablation studies against ResNet and Transformer
2. Deployed on lower-power MCUs (simpler architecture)
3. Online/streaming mode with forward-only pass (unidirectional variant)

References:
    Hochreiter & Schmidhuber (1997) Long Short-Term Memory
    Faust et al. (2018) Deep learning for healthcare using wearable sensors
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStem(nn.Module):
    """CNN stem for local feature extraction before LSTM.

    Processing raw 500Hz ECG directly with LSTM is expensive (5000 steps).
    The CNN stem:
    1. Reduces temporal resolution by 8× (625 LSTM steps)
    2. Extracts local waveform features (P, QRS, T morphology)
    3. Projects across-lead information into feature channels
    """

    def __init__(self, in_channels: int = 12, out_channels: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            # Block 1: fine features (local morphology)
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),

            # Block 2: stride 2 → 2500 steps
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),

            # Block 3: stride 2 → 1250 steps
            nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(96), nn.ReLU(inplace=True),

            # Block 4: stride 2 → 625 steps
            nn.Conv1d(96, out_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)  # (B, out_channels, T//8)


class AttentionPool(nn.Module):
    """Temporal attention pooling for LSTM output.

    Learns which timesteps are most important for classification.
    More informative than simple last-step or mean pooling because
    arrhythmia-relevant segments may be sparsely distributed.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_out: (B, T, H)

        Returns:
            context: (B, H) — weighted sum of timesteps
            weights: (B, T) — attention weights (interpretable)
        """
        scores = self.attention(lstm_out).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=-1)            # (B, T)
        context = (weights.unsqueeze(-1) * lstm_out).sum(dim=1)  # (B, H)
        return context, weights


class ECGBiLSTM(nn.Module):
    """Bidirectional LSTM with CNN stem and attention pooling.

    Architecture:
        CNN Stem → BiLSTM Stack → Attention Pool → Classifier

    The bidirectional design processes ECG in both forward and backward
    directions, enabling the model to use future context when classifying
    any given timestep (offline mode / full-recording classification).

    For real-time/streaming use, set bidirectional=False.

    Args:
        num_classes: Output labels.
        in_channels: ECG leads (12).
        cnn_out_channels: CNN stem output channels.
        lstm_hidden_size: LSTM hidden units per direction.
        num_lstm_layers: Stacked LSTM layers.
        bidirectional: Use bidirectional LSTM.
        dropout: Dropout between LSTM layers and before classifier.
    """

    def __init__(
        self,
        num_classes: int = 27,
        in_channels: int = 12,
        cnn_out_channels: int = 128,
        lstm_hidden_size: int = 256,
        num_lstm_layers: int = 3,
        bidirectional: bool = True,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        directions = 2 if bidirectional else 1

        # CNN feature extractor
        self.cnn_stem = ConvStem(in_channels, cnn_out_channels)

        # Stacked BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden_size * directions
        self.attn_pool = AttentionPool(lstm_out_dim)

        # Layer norm stabilizes training
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(lstm_out_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 12, T) — 12-lead ECG
            return_attention: Return temporal attention weights.

        Returns:
            dict with 'logits', 'probs', optionally 'attention_weights'
        """
        # CNN feature extraction
        features = self.cnn_stem(x)               # (B, C, T//8)
        features = features.transpose(1, 2)       # (B, T//8, C) for LSTM

        # BiLSTM
        lstm_out, _ = self.lstm(features)         # (B, T//8, H*dirs)
        lstm_out = self.layer_norm(lstm_out)

        # Attention pooling
        context, attn_weights = self.attn_pool(lstm_out)  # (B, H*dirs), (B, T//8)
        context = self.dropout(context)

        # Classification
        logits = self.classifier(context)

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probs": torch.sigmoid(logits),
        }
        if return_attention:
            out["attention_weights"] = attn_weights
        return out

    @torch.jit.export
    def streaming_step(
        self,
        x_chunk: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Single-step streaming inference (unidirectional mode only).

        Args:
            x_chunk: (B, 12, chunk_size) — new ECG chunk
            hidden: Previous LSTM hidden state (h_n, c_n)

        Returns:
            probs: (B, num_classes)
            new_hidden: Updated hidden state tuple
        """
        assert not self.bidirectional, "Streaming only supported in unidirectional mode"
        features = self.cnn_stem(x_chunk).transpose(1, 2)
        lstm_out, new_hidden = self.lstm(features, hidden)
        # Use last timestep
        last = self.layer_norm(lstm_out[:, -1, :])
        logits = self.classifier(self.dropout(last))
        return torch.sigmoid(logits), new_hidden


# ─────────────────────────────────────────────
#  TCN (Temporal Convolutional Network) Alternative
# ─────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """Causal 1D convolution — no future information leakage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remove future padding
        return self.conv(x)[:, :, : x.size(-1)]


class TCNBlock(nn.Module):
    """Temporal Convolutional Network residual block."""

    def __init__(
        self,
        n_channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(n_channels, n_channels, kernel_size, dilation),
            nn.BatchNorm1d(n_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(n_channels, n_channels, kernel_size, dilation),
            nn.BatchNorm1d(n_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(x) + x)


class ECGTCN(nn.Module):
    """TCN variant for ECG — useful as a causal (streaming-friendly) baseline.

    TCNs have receptive fields that grow exponentially with dilation,
    allowing efficient long-range temporal modeling without recurrence.
    Receptive field = 2 × (kernel_size - 1) × sum(dilations)
    """

    def __init__(
        self,
        num_classes: int = 27,
        in_channels: int = 12,
        n_channels: int = 128,
        n_levels: int = 8,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, n_channels, 1)

        # Exponentially increasing dilation
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(n_channels, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_levels)
        ])

        receptive_field = 1 + 2 * (kernel_size - 1) * (2**n_levels - 1)
        self._receptive_field = receptive_field

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(n_channels, num_classes)

    @property
    def receptive_field(self) -> int:
        """Number of samples this model can look back in time."""
        return self._receptive_field

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.input_proj(x)
        for block in self.tcn_blocks:
            x = block(x)
        x = self.gap(x).squeeze(-1)
        logits = self.classifier(x)
        return {"logits": logits, "probs": torch.sigmoid(logits)}


if __name__ == "__main__":
    # BiLSTM
    model = ECGBiLSTM(num_classes=27)
    x = torch.randn(2, 12, 5000)
    out = model(x, return_attention=True)
    print(f"BiLSTM Logits: {out['logits'].shape}")
    print(f"BiLSTM Attn:   {out['attention_weights'].shape}")
    print(f"BiLSTM params: {sum(p.numel() for p in model.parameters()):,}")

    # TCN
    tcn = ECGTCN(num_classes=27)
    out_tcn = tcn(x)
    print(f"TCN Logits:    {out_tcn['logits'].shape}")
    print(f"TCN receptive field: {tcn.receptive_field} samples "
          f"({tcn.receptive_field / 500:.1f}s @ 500Hz)")
    print(f"TCN params:    {sum(p.numel() for p in tcn.parameters()):,}")
