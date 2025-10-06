"""Reward shaping utilities for reinforcement learning trading agents.

This module provides a configurable reward shaping engine that aggregates
multiple objectives to guide agents toward profitable, risk-aware behavior.

Components
---------
1. Profit and loss (PnL) reward: primary trading signal.
2. Transaction cost penalty: penalizes friction from trading activity.
3. Time efficiency reward: encourages quick wins and discourages slow losses.
4. Sharpe ratio contribution: adds risk-adjusted perspective.
5. Drawdown penalty: protects against severe equity declines.
6. Position sizing reward: incentivizes optimal capital deployment.
7. Hold penalty: optional component discouraging excessive inaction.

The reward shaper exposes detailed component tracking and aggregate statistics
for transparency and downstream analysis. All components are normalized to
similar scales for stable learning dynamics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward function components.

    Component Weights (sum to ~1.0 for interpretability):
    - pnl_weight: Direct profit/loss signal
    - transaction_cost_weight: Penalty for trading friction
    - time_efficiency_weight: Reward for quick wins
    - sharpe_weight: Risk-adjusted return component
    - drawdown_weight: Penalty for severe losses
    - sizing_weight: Reward for optimal position sizing
    - hold_penalty_weight: Discourage excessive holding
    """

    # Component weights
    pnl_weight: float = 0.45  # Primary signal
    transaction_cost_weight: float = 0.15  # Cost awareness without over-penalizing
    time_efficiency_weight: float = 0.15  # Reward speed
    sharpe_weight: float = 0.05  # Risk-adjusted focus
    drawdown_weight: float = 0.10  # Severe loss penalty
    sizing_weight: float = 0.05  # Capital utilization
    hold_penalty_weight: float = 0.0  # Optional: discourage holding too long

    # Normalization parameters
    pnl_scale: float = 0.01  # Expected PnL scale (1%)
    time_horizon: int = 8  # Max holding period (hours)
    target_sharpe: float = 1.0  # Target Sharpe ratio
    max_drawdown_threshold: float = 0.05  # 5% loss triggers penalty

    # Shaping parameters
    win_bonus_multiplier: float = 1.2  # Extra reward for wins
    loss_penalty_multiplier: float = 1.5  # Extra penalty for losses
    quick_win_bonus: float = 0.5  # Bonus for wins in <4 hours
    early_stop_bonus: float = 0.3  # Bonus for hitting stops cleanly

    # Risk management
    max_single_loss: float = 0.02  # 2% max loss per trade
    severe_loss_penalty: float = -5.0  # Large penalty for >2% loss

    # Transaction cost assumptions
    base_transaction_cost_pct: float = 0.001  # 0.10% default cost
    failed_action_penalty: float = -0.05  # Penalty for invalid attempts

    # Numerical safety
    min_equity: float = 1e-6  # Avoid divide-by-zero in PnL

    # Optional smoothing
    reward_clip: float = 10.0  # Absolute cap for component outputs

    component_keys: Tuple[str, ...] = field(
        init=False,
        default=(
            "pnl",
            "transaction_cost",
            "time_efficiency",
            "sharpe",
            "drawdown",
            "sizing",
            "hold_penalty",
        ),
    )

    def __post_init__(self) -> None:
        """Validate configuration values and surface potential issues."""
        if self.pnl_scale <= 0:
            raise ValueError("pnl_scale must be positive to normalize rewards")

        if self.time_horizon <= 0:
            raise ValueError("time_horizon must be positive")

        if self.target_sharpe <= 0:
            logger.warning(
                "target_sharpe is non-positive; Sharpe rewards will be zeroed"
            )

        total_weight = (
            self.pnl_weight
            + self.transaction_cost_weight
            + self.time_efficiency_weight
            + self.sharpe_weight
            + self.drawdown_weight
            + self.sizing_weight
            + self.hold_penalty_weight
        )

        if not 0.9 <= total_weight <= 1.1:
            logger.warning(
                "Component weights sum to %.3f; consider normalizing near 1.0",
                total_weight,
            )

        if self.base_transaction_cost_pct < 0:
            raise ValueError("base_transaction_cost_pct must be non-negative")

        if self.reward_clip <= 0:
            raise ValueError("reward_clip must be positive")


class RewardShaper:
    """Multi-objective reward calculation with component tracking."""

    def __init__(self, config: RewardConfig) -> None:
        """Initialize reward shaper.

        Args:
            config: Reward configuration specifying component weights and scaling.
        """
        self.config = config

        # Episode tracking
        self.episode_rewards: List[float] = []
        self.component_history: List[Dict[str, float]] = []

        # Historical statistics across episodes
        self.reward_stats: Dict[str, List[float]] = {
            "total_rewards": [],
            "pnl_rewards": [],
            "cost_penalties": [],
            "time_rewards": [],
            "sharpe_rewards": [],
            "drawdown_penalties": [],
            "sizing_rewards": [],
            "hold_penalties": [],
        }

        logger.info(
            (
                "RewardShaper initialized with weights: PnL=%.2f, Cost=%.2f, Time=%.2f, "
                "Sharpe=%.2f, Drawdown=%.2f, Sizing=%.2f, Hold=%.2f"
            ),
            config.pnl_weight,
            config.transaction_cost_weight,
            config.time_efficiency_weight,
            config.sharpe_weight,
            config.drawdown_weight,
            config.sizing_weight,
            config.hold_penalty_weight,
        )

    def compute_reward(
        self,
        action: int,
        action_executed: bool,
        prev_equity: float,
        current_equity: float,
        position_info: Optional[Dict] = None,
        trade_info: Optional[Dict] = None,
        portfolio_state: Optional[Dict] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute total reward from multiple components.

        Args:
            action: Index of action taken (0-6 mapping to environment actions).
            action_executed: Whether the environment executed the action.
            prev_equity: Equity before the step.
            current_equity: Equity after the step.
            position_info: Current position details, if any.
            trade_info: Trade details when a position is closed.
            portfolio_state: Aggregate portfolio metrics available for shaping.

        Returns:
            Tuple consisting of the weighted total reward and a dictionary of
            component contributions before weighting.
        """
        components: Dict[str, float] = {}

        components["pnl"] = self._compute_pnl_reward(
            prev_equity=prev_equity,
            current_equity=current_equity,
            trade_info=trade_info,
        )

        components["transaction_cost"] = self._compute_cost_penalty(
            action=action,
            action_executed=action_executed,
            trade_info=trade_info,
        )

        components["time_efficiency"] = self._compute_time_reward(
            trade_info=trade_info,
            position_info=position_info,
        )

        components["sharpe"] = self._compute_sharpe_reward(
            portfolio_state=portfolio_state,
        )

        components["drawdown"] = self._compute_drawdown_penalty(
            current_equity=current_equity,
            portfolio_state=portfolio_state,
        )

        components["sizing"] = self._compute_sizing_reward(
            position_info=position_info,
            portfolio_state=portfolio_state,
            action=action,
        )

        if self.config.hold_penalty_weight > 0:
            components["hold_penalty"] = self._compute_hold_penalty(
                action=action,
                position_info=position_info,
            )
        else:
            components["hold_penalty"] = 0.0

        total_reward = self._aggregate_components(components)

        self._record_step(total_reward=total_reward, components=components)

        return total_reward, components

    # ------------------------------------------------------------------
    # Component calculations
    # ------------------------------------------------------------------
    def _compute_pnl_reward(
        self,
        prev_equity: float,
        current_equity: float,
        trade_info: Optional[Dict],
    ) -> float:
        """P&L reward: direct profit/loss signal normalized to [-clip, clip]."""
        safe_prev_equity = max(prev_equity, self.config.min_equity)
        equity_change = current_equity - safe_prev_equity
        pnl_pct = equity_change / safe_prev_equity

        normalized_pnl = pnl_pct / self.config.pnl_scale

        if normalized_pnl > 0:
            reward = normalized_pnl * self.config.win_bonus_multiplier
        else:
            reward = normalized_pnl * self.config.loss_penalty_multiplier

        if trade_info:
            trade_pnl_pct = trade_info.get("pnl_pct", 0.0)
            if trade_pnl_pct < -self.config.max_single_loss:
                reward += self.config.severe_loss_penalty

        return float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))

    def _compute_cost_penalty(
        self,
        action: int,
        action_executed: bool,
        trade_info: Optional[Dict],
    ) -> float:
        """Transaction cost penalty: explicit cost awareness."""
        if not action_executed:
            return self.config.failed_action_penalty

        if action == 0:
            return 0.0

        cost_pct = self.config.base_transaction_cost_pct
        normalized_cost = -cost_pct / self.config.pnl_scale

        if action in {1, 2, 3, 4, 5}:  # Opening or closing actions
            reward = normalized_cost
        elif action == 6:  # Add position, slightly more expensive
            reward = -1.5 * cost_pct / self.config.pnl_scale
        else:
            reward = normalized_cost

        if action in {4, 5} and trade_info:
            pnl_pct = trade_info.get("pnl_pct", 0.0)
            if np.isfinite(pnl_pct):
                if abs(pnl_pct - 0.025) < 0.005:
                    reward += self.config.early_stop_bonus
                elif abs(pnl_pct + 0.02) < 0.005:
                    reward += 0.5 * self.config.early_stop_bonus

        return float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))

    def _compute_time_reward(
        self,
        trade_info: Optional[Dict],
        position_info: Optional[Dict],
    ) -> float:
        """Time efficiency reward: encourage quick profitable exits."""
        if not trade_info:
            return 0.0

        holding_hours = float(trade_info.get("holding_hours", 0.0))
        pnl_pct = float(trade_info.get("pnl_pct", 0.0))

        if pnl_pct > 0 and holding_hours < 4:
            multiplier = max(0.0, 1.0 - holding_hours / 4.0)
            reward = self.config.quick_win_bonus * multiplier
            return float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))

        if pnl_pct < 0 and holding_hours > 6:
            excess_hours = holding_hours - 6.0
            penalty = -0.3 * (excess_hours / self.config.time_horizon)
            return float(np.clip(penalty, -self.config.reward_clip, self.config.reward_clip))

        if (
            position_info
            and position_info.get("is_open", False)
            and holding_hours >= self.config.time_horizon
        ):
            reward = -0.1 * (holding_hours - self.config.time_horizon) / self.config.time_horizon
            return float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))

        return 0.0

    def _compute_sharpe_reward(self, portfolio_state: Optional[Dict]) -> float:
        """Sharpe ratio reward: encourage risk-adjusted returns."""
        if not portfolio_state:
            return 0.0

        sharpe = float(portfolio_state.get("sharpe_ratio", 0.0))
        if not np.isfinite(sharpe) or self.config.target_sharpe <= 0:
            return 0.0

        sharpe_reward = (sharpe - self.config.target_sharpe) / self.config.target_sharpe
        return float(np.clip(sharpe_reward, -2.0, 2.0))

    def _compute_drawdown_penalty(
        self,
        current_equity: float,
        portfolio_state: Optional[Dict],
    ) -> float:
        """Drawdown penalty: severe penalty for large losses."""
        if not portfolio_state:
            return 0.0

        peak_equity = float(portfolio_state.get("peak_equity", current_equity))
        peak_equity = max(peak_equity, self.config.min_equity)

        drawdown = (peak_equity - current_equity) / peak_equity
        drawdown = max(drawdown, 0.0)

        if drawdown < self.config.max_drawdown_threshold:
            return 0.0

        excess_dd = drawdown - self.config.max_drawdown_threshold
        penalty = -10.0 * (excess_dd / 0.05) ** 2
        penalty = max(penalty, -20.0)

        return float(np.clip(penalty, -self.config.reward_clip, 0.0))

    def _compute_sizing_reward(
        self,
        position_info: Optional[Dict],
        portfolio_state: Optional[Dict],
        action: int,
    ) -> float:
        """Position sizing reward: encourage optimal capital utilization."""
        if action == 0 or not portfolio_state:
            return 0.0

        deployed_pct = float(portfolio_state.get("deployed_pct", 0.0))

        if 0.5 <= deployed_pct <= 0.8:
            reward = 0.3
        elif 0.3 <= deployed_pct < 0.5:
            reward = 0.1
        elif 0.8 < deployed_pct <= 1.0:
            reward = -0.2
        else:
            reward = -0.5

        return float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))

    def _compute_hold_penalty(
        self,
        action: int,
        position_info: Optional[Dict],
    ) -> float:
        """Hold penalty: discourage excessive holding without action."""
        if action != 0 or not position_info:
            return 0.0

        if not position_info.get("is_open", False):
            return 0.0

        holding_hours = float(position_info.get("duration", 0.0))
        if holding_hours <= 6.0:
            return 0.0

        penalty = -0.1 * (holding_hours - 6.0) / self.config.time_horizon
        return float(np.clip(penalty, -self.config.reward_clip, 0.0))

    # ------------------------------------------------------------------
    # Aggregation & tracking helpers
    # ------------------------------------------------------------------
    def _aggregate_components(self, components: Dict[str, float]) -> float:
        """Aggregate individual component scores into a weighted reward."""
        total = (
            self.config.pnl_weight * components.get("pnl", 0.0)
            + self.config.transaction_cost_weight * components.get("transaction_cost", 0.0)
            + self.config.time_efficiency_weight * components.get("time_efficiency", 0.0)
            + self.config.sharpe_weight * components.get("sharpe", 0.0)
            + self.config.drawdown_weight * components.get("drawdown", 0.0)
            + self.config.sizing_weight * components.get("sizing", 0.0)
            + self.config.hold_penalty_weight * components.get("hold_penalty", 0.0)
        )
        return float(np.clip(total, -self.config.reward_clip, self.config.reward_clip))

    def _record_step(self, total_reward: float, components: Dict[str, float]) -> None:
        """Record step-level rewards for later analysis."""
        self.episode_rewards.append(total_reward)
        self.component_history.append(components.copy())

        self.reward_stats["total_rewards"].append(total_reward)
        self.reward_stats["pnl_rewards"].append(components.get("pnl", 0.0))
        self.reward_stats["cost_penalties"].append(components.get("transaction_cost", 0.0))
        self.reward_stats["time_rewards"].append(components.get("time_efficiency", 0.0))
        self.reward_stats["sharpe_rewards"].append(components.get("sharpe", 0.0))
        self.reward_stats["drawdown_penalties"].append(components.get("drawdown", 0.0))
        self.reward_stats["sizing_rewards"].append(components.get("sizing", 0.0))
        self.reward_stats["hold_penalties"].append(components.get("hold_penalty", 0.0))

    # ------------------------------------------------------------------
    # Episode & analysis helpers
    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """Reset per-episode tracking structures."""
        self.episode_rewards.clear()
        self.component_history.clear()

    def get_episode_stats(self) -> Dict[str, float]:
        """Return summary statistics for the current episode."""
        if not self.episode_rewards:
            return {}

        stats: Dict[str, float] = {
            "total_reward_mean": float(np.mean(self.episode_rewards)),
            "total_reward_std": float(np.std(self.episode_rewards)),
            "total_reward_sum": float(np.sum(self.episode_rewards)),
            "steps": float(len(self.episode_rewards)),
        }

        if self.component_history:
            for key in self.config.component_keys:
                values = [c.get(key, 0.0) for c in self.component_history]
                stats[f"{key}_mean"] = float(np.mean(values))
                stats[f"{key}_sum"] = float(np.sum(values))

        return stats

    def get_component_contributions(self) -> Dict[str, float]:
        """Return relative contribution of components within an episode."""
        if not self.component_history:
            return {}

        totals: Dict[str, float] = {}
        for key in self.config.component_keys:
            values = [c.get(key, 0.0) for c in self.component_history]
            totals[key] = float(np.sum(values))

        total_abs = sum(abs(v) for v in totals.values())
        if total_abs == 0:
            return {key: 0.0 for key in totals}

        contributions = {key: (value / total_abs) * 100.0 for key, value in totals.items()}
        return contributions

    def get_recent_components(self, n_steps: int = 20) -> List[Dict[str, float]]:
        """Return the most recent component values for inspection."""
        if n_steps <= 0:
            return []
        return self.component_history[-n_steps:]

    def get_running_means(self, window: int = 50) -> Dict[str, float]:
        """Compute running means for each component over the latest window."""
        if window <= 0:
            raise ValueError("window must be positive")

        if not self.component_history:
            return {key: 0.0 for key in self.config.component_keys}

        recent = self.component_history[-window:]
        means = {}
        for key in self.config.component_keys:
            values = [c.get(key, 0.0) for c in recent]
            means[key] = float(np.mean(values)) if values else 0.0
        return means

    def update_config(self, **kwargs: float) -> None:
        """Update configuration weights or parameters at runtime."""
        for attr, value in kwargs.items():
            if not hasattr(self.config, attr):
                raise AttributeError(f"RewardConfig has no attribute '{attr}'")
            setattr(self.config, attr, value)

        logger.info("Reward configuration updated: %s", kwargs)

    def summarize_reward_stats(self) -> Dict[str, float]:
        """Summarize global reward statistics across episodes."""
        summary: Dict[str, float] = {}
        for key, values in self.reward_stats.items():
            if not values:
                summary[f"{key}_mean"] = 0.0
                summary[f"{key}_std"] = 0.0
                continue
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
        return summary

    def clear_global_stats(self) -> None:
        """Clear global reward statistics (does not touch current episode)."""
        for values in self.reward_stats.values():
            values.clear()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def merge_component_histories(histories: Iterable[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate component histories from multiple episodes."""
        totals: Dict[str, float] = {}
        for entry in histories:
            for key, value in entry.items():
                totals[key] = totals.get(key, 0.0) + float(value)
        return totals

    def describe(self) -> str:
        """Return human-readable summary of the reward configuration."""
        config_items = {
            "pnl_weight": self.config.pnl_weight,
            "transaction_cost_weight": self.config.transaction_cost_weight,
            "time_efficiency_weight": self.config.time_efficiency_weight,
            "sharpe_weight": self.config.sharpe_weight,
            "drawdown_weight": self.config.drawdown_weight,
            "sizing_weight": self.config.sizing_weight,
            "hold_penalty_weight": self.config.hold_penalty_weight,
            "pnl_scale": self.config.pnl_scale,
            "time_horizon": self.config.time_horizon,
            "target_sharpe": self.config.target_sharpe,
            "max_drawdown_threshold": self.config.max_drawdown_threshold,
        }
        return ", ".join(f"{key}={value}" for key, value in config_items.items())


__all__ = ["RewardConfig", "RewardShaper"]
