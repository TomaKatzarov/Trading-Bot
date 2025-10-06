"""Shared transformer-based feature encoder for multi-agent reinforcement learning.

This module implements the shared feature encoder described in the Phase 2 execution
plan. The encoder processes dictionary observations emitted by the trading
environment and produces a unified embedding that will be consumed by all symbol
agents. The architecture is based on a transformer encoder with sinusoidal
positional encodings and supports returning either a pooled embedding or the full
sequence of token representations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor
import torch.nn as nn


@dataclass
class EncoderConfig:
    """Configuration container for :class:`FeatureEncoder` parameters."""

    # Technical feature sequence dimensions
    technical_seq_len: int = 24
    technical_feature_dim: int = 23

    # Auxiliary feature dimensions
    sl_prob_dim: int = 3
    position_dim: int = 5
    portfolio_dim: int = 8
    regime_dim: int = 10

    # Transformer architecture parameters
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Output configuration
    output_dim: int = 256
    max_seq_len: int = 50

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""

        min_required_tokens = self.technical_seq_len + 4  # technical + 4 auxiliary tokens
        if self.max_seq_len < min_required_tokens:
            raise ValueError(
                "max_seq_len must be at least technical_seq_len + 4 tokens; got "
                f"{self.max_seq_len} < {min_required_tokens}."
            )
        if self.d_model % self.nhead != 0:
            raise ValueError(
                "d_model must be divisible by nhead for multi-head attention; got "
                f"d_model={self.d_model}, nhead={self.nhead}."
            )


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        """Initialize the positional encoding module.

        Args:
            d_model: Dimensionality of the model (embedding size).
            dropout: Dropout probability applied after adding positional encodings.
            max_len: Maximum supported sequence length.
        """

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encodings to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor with positional encodings added, same shape as ``x``.
        """

        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class FeatureEncoder(nn.Module):
    """Shared transformer encoder that processes dict observations into embeddings.

    The encoder projects each component of the observation dictionary into a common
    embedding space, concatenates them along the token dimension, applies sinusoidal
    positional encoding, and processes the resulting sequence with a stack of
    transformer encoder layers. The final representation is obtained via global
    average pooling and a linear projection to ``config.output_dim``.
    """

    def __init__(self, config: EncoderConfig | None = None) -> None:
        """Construct a new :class:`FeatureEncoder`.

        Args:
            config: Optional :class:`EncoderConfig` instance. If ``None`` a default
                configuration is used.
        """

        super().__init__()
        self.config = config or EncoderConfig()

        # Input projections for each observation component
        self.technical_proj = nn.Linear(
            self.config.technical_feature_dim, self.config.d_model
        )
        self.sl_prob_proj = nn.Linear(self.config.sl_prob_dim, self.config.d_model)
        self.position_proj = nn.Linear(self.config.position_dim, self.config.d_model)
        self.portfolio_proj = nn.Linear(self.config.portfolio_dim, self.config.d_model)
        self.regime_proj = nn.Linear(self.config.regime_dim, self.config.d_model)

        # Positional encoding and transformer stack
        self.positional_encoding = PositionalEncoding(
            d_model=self.config.d_model,
            dropout=self.config.dropout,
            max_len=self.config.max_seq_len,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.config.num_layers,
            enable_nested_tensor=False,
        )

        self.sequence_norm = nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        self.output_proj = nn.Linear(self.config.d_model, self.config.output_dim)
        self.output_dropout = nn.Dropout(p=self.config.dropout)

        self._init_weights()

    def forward(
        self, observations: Dict[str, Tensor], return_sequence: bool = False
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """Encode a batch of observation dictionaries.

        Args:
            observations: Mapping containing the observation components emitted by
                the trading environment. Expected keys: ``technical``, ``sl_probs``,
                ``position``, ``portfolio``, and ``regime``.
            return_sequence: If ``True``, the method returns a tuple containing the
                pooled embedding and the full transformer output sequence. When
                ``False`` (default), only the pooled embedding is returned.

        Returns:
            Either a single tensor containing the pooled embeddings with shape
            ``(batch_size, output_dim)``, or a tuple ``(pooled, sequence)`` where the
            sequence tensor has shape ``(batch_size, total_tokens, d_model)``.
        """

        technical = observations["technical"].float()
        sl_probs = observations["sl_probs"].float()
        position = observations["position"].float()
        portfolio = observations["portfolio"].float()
        regime = observations["regime"].float()

        technical_tokens = self.technical_proj(technical)
        sl_tokens = self.sl_prob_proj(sl_probs).unsqueeze(1)
        position_tokens = self.position_proj(position).unsqueeze(1)
        portfolio_tokens = self.portfolio_proj(portfolio).unsqueeze(1)
        regime_tokens = self.regime_proj(regime).unsqueeze(1)

        sequence = torch.cat(
            [technical_tokens, sl_tokens, position_tokens, portfolio_tokens, regime_tokens],
            dim=1,
        )

        encoded = self.positional_encoding(sequence)
        encoded = self.transformer(encoded)
        encoded = self.sequence_norm(encoded)

        pooled = encoded.mean(dim=1)
        pooled = self.output_dropout(pooled)
        pooled = self.output_proj(pooled)

        if return_sequence:
            return pooled, encoded
        return pooled

    def _init_weights(self) -> None:
        """Initialize encoder weights via centralized strategy helpers."""

        from .initialization import init_encoder

        init_encoder(self, strategy="xavier_uniform", gain=1.0)