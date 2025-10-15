"""Intrinsic Curiosity Module tailored for the trading RL environments."""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ICMConfig:
    """Configuration for the intrinsic curiosity module."""

    state_dim: int = 512
    action_dim: int = 1
    hidden_dim: int = 256
    feature_dim: int = 128
    beta: float = 0.2
    eta: float = 0.01
    extrinsic_weight: float = 0.9
    intrinsic_weight: float = 0.1


class TradingICM(nn.Module):
    """Intrinsic Curiosity Module used to stabilize exploration in sparse rewards."""

    def __init__(self, config: ICMConfig):
        super().__init__()
        self.config = config

        self.feature_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.feature_dim),
            nn.LayerNorm(config.feature_dim),
        )

        self.forward_model = nn.Sequential(
            nn.Linear(config.feature_dim + config.action_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.feature_dim),
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(config.feature_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

        self.register_buffer("intrinsic_reward_mean", torch.tensor(0.0))
        self.register_buffer("intrinsic_reward_std", torch.tensor(1.0))
        self.register_buffer("update_count", torch.tensor(0))

    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        state_features = self.feature_encoder(state)
        next_state_features = self.feature_encoder(next_state)

        if action.dim() == 1:
            action = action.unsqueeze(-1)

        predicted_next = self.forward_model(torch.cat([state_features, action], dim=-1))
        predicted_action = self.inverse_model(
            torch.cat([state_features, next_state_features], dim=-1)
        )

        forward_loss = F.mse_loss(
            predicted_next,
            next_state_features.detach(),
            reduction="none",
        ).mean(dim=-1)

        inverse_loss = F.mse_loss(
            predicted_action,
            action.detach(),
            reduction="none",
        ).mean(dim=-1)

        intrinsic_reward = self.config.eta * forward_loss.detach()
        intrinsic_reward = self._normalize_reward(intrinsic_reward)

        forward_loss_mean = forward_loss.mean()
        inverse_loss_mean = inverse_loss.mean()
        total_loss = (1 - self.config.beta) * inverse_loss_mean + self.config.beta * forward_loss_mean

        losses = {
            "forward_loss": forward_loss_mean,
            "inverse_loss": inverse_loss_mean,
            "total_loss": total_loss,
            "intrinsic_reward_mean": intrinsic_reward.mean(),
        }

        return intrinsic_reward, losses

    def _normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        if self.training and self.update_count > 10:
            batch_mean = reward.mean()
            batch_std = reward.std() + 1e-8

            alpha = 0.01
            self.intrinsic_reward_mean = (1 - alpha) * self.intrinsic_reward_mean + alpha * batch_mean
            self.intrinsic_reward_std = (1 - alpha) * self.intrinsic_reward_std + alpha * batch_std

            normalized = (reward - self.intrinsic_reward_mean) / self.intrinsic_reward_std
            return torch.clamp(normalized, -5, 5)

        self.update_count += 1
        return reward
