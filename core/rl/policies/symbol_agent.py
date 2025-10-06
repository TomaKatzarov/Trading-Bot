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

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions import Categorical

from .feature_encoder import EncoderConfig, FeatureEncoder


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

        if position.dim() != 2 or portfolio.dim() != 2:
            raise ValueError(
                "Position and portfolio tensors must be two-dimensional (batch, features)."
            )
        if position.size(0) != portfolio.size(0):
            raise ValueError("Position and portfolio batches must have matching sizes.")

        batch_size = position.size(0)
        device = position.device

        mask = torch.ones(batch_size, self.action_dim, dtype=torch.bool, device=device)

        position_size = position[:, 1]
        exposure = portfolio[:, 1].to(position_size.device)

        has_position = position_size > 0.0
        no_position = ~has_position
        high_exposure = exposure >= self.exposure_threshold

        if no_position.any():
            for idx in self._sell_indices.tolist():
                mask[no_position, idx] = False
        if has_position.any():
            for idx in self._buy_indices.tolist():
                mask[has_position, idx] = False
        if high_exposure.any():
            for idx in self._increase_indices.tolist():
                mask[high_exposure, idx] = False

        mask[:, 0] = True  # HOLD action always permitted
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

        init_actor(self.actor, strategy="orthogonal", gain=0.01)
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
