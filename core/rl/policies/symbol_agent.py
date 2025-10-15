"""Symbol-level actor-critic policy networks for the trading system.

This module implements the symbol agent policy described in Phase 2 of the RL
roadmap. Each agent consumes embeddings produced by the shared
:class:`core.rl.policies.feature_encoder.FeatureEncoder` and emits discrete
trading actions alongside value estimates required for PPO-style algorithms.

Key capabilities:
- Optional weight sharing for the feature encoder across 143 symbol agents.
- Action masking that enforces portfolio and position constraints.
- PPO-compatible interface providing ``forward``, ``get_value`` and
  ``evaluate_actions`` methods.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions import Categorical

from .feature_encoder import EncoderConfig, FeatureEncoder

logger = logging.getLogger(__name__)


@dataclass
class SymbolAgentConfig:
    """Configuration container for :class:`SymbolAgent` instances."""

    encoder_config: EncoderConfig
    action_dim: int = 7
    hidden_dim: int = 128
    dropout: float = 0.1
    use_shared_encoder: bool = True


class ActionMasker(nn.Module):
    """Generate action masks enforcing trading constraints."""

    def __init__(self, action_dim: int = 7, exposure_threshold: float = 0.9) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.exposure_threshold = exposure_threshold

        self.register_buffer("_buy_indices", torch.tensor([1, 2, 3], dtype=torch.long))
        self.register_buffer("_sell_indices", torch.tensor([4, 5, 6], dtype=torch.long))
        self.register_buffer(
            "_increase_indices", torch.tensor([1, 2, 3, 6], dtype=torch.long)
        )

    @torch.no_grad()
    def get_mask(self, observations: Dict[str, Tensor]) -> Tensor:
        """Compute a boolean mask of valid actions for each observation in the batch.

        Args:
            observations: Observation dictionary containing ``position`` and
                ``portfolio`` tensors with shapes ``(batch_size, 5)`` and
                ``(batch_size, 8)`` respectively.

        Returns:
            Boolean tensor of shape ``(batch_size, action_dim)`` where ``True``
            indicates the action is permitted.
        """

        if "position" not in observations:
            raise KeyError("'position' key missing from observations")
        if "portfolio" not in observations:
            raise KeyError("'portfolio' key missing from observations")

        position = observations["position"].float()
        portfolio = observations["portfolio"].float()

        # Defensive sanitisation to avoid NaNs propagating through comparisons
        position = torch.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
        portfolio = torch.nan_to_num(portfolio, nan=0.0, posinf=0.0, neginf=0.0)

        if position.dim() != 2 or portfolio.dim() != 2:
            raise ValueError(
                "Position and portfolio tensors must be two-dimensional (batch, features)."
            )
        if position.size(0) != portfolio.size(0):
            raise ValueError("Position and portfolio batches must have matching sizes.")

        batch_size = position.size(0)
        device = position.device

        mask = torch.ones(batch_size, self.action_dim, dtype=torch.bool, device=device)

        # Position vector schema (see TradingEnvironment._get_position_state):
        # [0] -> indicator (1.0 when a position is open)
        # [1] -> entry price
        # [2] -> unrealised PnL pct
        # [3] -> holding period
        # [4] -> position size as % of equity
        indicator = position[:, 0]
        size_pct = position[:, 4] if position.size(1) > 4 else torch.zeros_like(indicator)
        has_position = indicator > 0.5
        no_position = ~has_position

        # Portfolio vector schema (TradingEnvironment._get_portfolio_state):
        # [0] -> equity, [1] -> cash, [2] -> exposure_pct, ...
        exposure = portfolio[:, 2] if portfolio.size(1) > 2 else torch.zeros_like(indicator)
        high_exposure = exposure >= self.exposure_threshold

        if no_position.any():
            for idx in self._sell_indices.tolist():
                mask[no_position, idx] = False
        if has_position.any():
            # STRICTER MASKING (2025-10-08 Anti-Collapse Improvement #4):
            # Block BUY_* when a position exists to avoid double entries, but
            # keep SELL_* pathways open so the policy can always exit.
            for idx in self._buy_indices.tolist():  # [1, 2, 3]
                mask[has_position, idx] = False
            logger.debug(
                "Action masking: blocked BUY actions for %d envs with existing positions",
                has_position.sum().item()
            )
            # Ensure SELL actions remain valid for all active positions
            for idx in self._sell_indices.tolist():
                mask[has_position, idx] = True

        if high_exposure.any():
            for idx in self._increase_indices.tolist():
                mask[high_exposure, idx] = False

        add_index = int(self._increase_indices[-1].item()) if self._increase_indices.numel() > 0 else 6
        if no_position.any():
            mask[no_position, add_index] = False
        else:
            # Prevent pyramiding beyond allowed sizing even when a position exists
            at_size_limit = size_pct >= self.exposure_threshold
            if at_size_limit.any():
                mask[at_size_limit, add_index] = False

        mask[:, 0] = True  # HOLD action always permitted

        # Diagnostics: warn when masking leaves <=2 actions (typically HOLD+ADD)
        valid_counts = mask.sum(dim=1)
        limited_mask = valid_counts <= 2
        if limited_mask.any():
            logger.debug(
                "Action mask limited options to %s actions for batch indices %s",
                valid_counts[limited_mask].tolist(),
                torch.nonzero(limited_mask, as_tuple=False).view(-1).tolist(),
            )

        if has_position.any():
            sell_matrix = mask[:, self._sell_indices]
            sells_available = sell_matrix.any(dim=1)
            missing_rows = torch.nonzero(has_position & (~sells_available), as_tuple=False).view(-1)
            if missing_rows.numel() > 0:
                logger.warning(
                    "Action mask removed all SELL options for %d samples; reinstating SELL_PARTIAL",
                    missing_rows.numel(),
                )
                sell_partial_idx = int(self._sell_indices[0].item())
                mask[missing_rows, sell_partial_idx] = True

        return mask


class SymbolAgent(nn.Module):
    """Actor-critic policy head operating on shared encoder embeddings."""

    def __init__(
        self,
        config: SymbolAgentConfig,
        shared_encoder: Optional[FeatureEncoder] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.action_dim = config.action_dim

        if shared_encoder is not None:
            self.encoder = shared_encoder
            self.owns_encoder = False
        else:
            self.encoder = FeatureEncoder(config.encoder_config)
            self.owns_encoder = True

        latent_dim = config.encoder_config.output_dim

        self.actor = nn.Sequential(
            nn.Linear(latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

        self.action_masker = ActionMasker(action_dim=config.action_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize actor and critic using centralized helpers."""

        from .initialization import init_actor, init_critic

        # CRITICAL FIX (2025-10-08 v2): Increased gain from 0.01 to 0.3 to prevent
        # premature policy collapse. Previous gain=0.1 still collapsed to single action
        # within 10k steps. gain=0.3 provides much stronger initial exploration while
        # remaining stable (3× SB3 discrete defaults, validated empirically).
        # This creates initial logits in [-0.09, +0.09] range vs [-0.03, +0.03].
        init_actor(self.actor, strategy="orthogonal", gain=0.3)
        init_critic(self.critic, strategy="orthogonal", gain=1.0)

    def forward(
        self,
        observations: Dict[str, Tensor],
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Select actions for a batch of observations.

        Args:
            observations: Observation dictionary produced by the trading
                environment.
            deterministic: When ``True`` select the greedily optimal action,
                otherwise sample from the masked categorical distribution.

        Returns:
            Tuple ``(actions, log_probs, values)`` with shapes ``(batch,)``.
        """

        features = self.encoder(observations)

        logits = self.actor(features)
        mask = self.action_masker.get_mask(observations)
        masked_logits = logits.masked_fill(~mask, float("-inf"))

        distribution = Categorical(logits=masked_logits)
        if deterministic:
            actions = masked_logits.argmax(dim=-1)
        else:
            actions = distribution.sample()

        log_probs = distribution.log_prob(actions)
        values = self.critic(features).squeeze(-1)
        return actions, log_probs, values

    @torch.no_grad()
    def get_value(self, observations: Dict[str, Tensor]) -> Tensor:
        """Return value estimates for PPO advantages."""

        features = self.encoder(observations)
        return self.critic(features).squeeze(-1)

    def evaluate_actions(
        self,
        observations: Dict[str, Tensor],
        actions: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluate provided actions for PPO training.

        Returns ``(log_probs, values, entropy)`` each shaped ``(batch,)``.
        """

        features = self.encoder(observations)

        logits = self.actor(features)
        mask = self.action_masker.get_mask(observations)
        masked_logits = logits.masked_fill(~mask, float("-inf"))

        distribution = Categorical(logits=masked_logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        values = self.critic(features).squeeze(-1)
        return log_probs, values, entropy
