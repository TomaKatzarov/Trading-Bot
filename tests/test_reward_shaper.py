"""Comprehensive test suite for RewardShaper.

Validates component correctness, configuration handling, edge cases,
statistics tracking, and aggregate behavior of the multi-objective reward
shaper used in the trading environment.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pytest

from core.rl.environments.reward_shaper import RewardConfig, RewardShaper


class TestRewardConfig:
    """Test RewardConfig dataclass."""

    def test_default_weights_sum(self) -> None:
        """Test default weights approximately sum to 1.0."""
        config = RewardConfig()

        total = (
            config.pnl_weight
            + config.transaction_cost_weight
            + config.time_efficiency_weight
            + config.sharpe_weight
            + config.drawdown_weight
            + config.sizing_weight
            + config.hold_penalty_weight
        )

        assert 0.9 <= total <= 1.1

    def test_custom_weights(self) -> None:
        """Test custom weight configuration."""
        config = RewardConfig(
            pnl_weight=0.5,
            transaction_cost_weight=0.3,
            time_efficiency_weight=0.2,
        )

        assert config.pnl_weight == pytest.approx(0.5)
        assert config.transaction_cost_weight == pytest.approx(0.3)
        assert config.time_efficiency_weight == pytest.approx(0.2)

    def test_invalid_pnl_scale_raises(self) -> None:
        """Negative pnl_scale should raise ValueError."""
        with pytest.raises(ValueError):
            RewardConfig(pnl_scale=-0.01)

    def test_negative_transaction_cost_raises(self) -> None:
        """Negative transaction cost configuration should error."""
        with pytest.raises(ValueError):
            RewardConfig(base_transaction_cost_pct=-0.0001)

    def test_reward_clip_positive(self) -> None:
        """Reward clip must be positive."""
        with pytest.raises(ValueError):
            RewardConfig(reward_clip=0.0)


class TestPnLReward:
    """Test P&L reward component."""

    def test_positive_pnl(self) -> None:
        """Test reward for profitable trade."""
        config = RewardConfig(pnl_scale=0.01)
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=5,
            action_executed=True,
            prev_equity=100_000,
            current_equity=101_000,
            trade_info={"pnl_pct": 0.01, "holding_hours": 4},
        )

        assert components["pnl"] > 0
        assert components["pnl"] == pytest.approx(1.2, rel=0.05)

    def test_negative_pnl(self) -> None:
        """Test penalty for losing trade."""
        config = RewardConfig(pnl_scale=0.01)
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=5,
            action_executed=True,
            prev_equity=100_000,
            current_equity=99_000,
            trade_info={"pnl_pct": -0.01, "holding_hours": 4},
        )

        assert components["pnl"] < 0
        assert components["pnl"] == pytest.approx(-1.5, rel=0.05)

    def test_severe_loss_penalty(self) -> None:
        """Test severe penalty for large losses."""
        config = RewardConfig(pnl_scale=0.01, max_single_loss=0.02)
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=5,
            action_executed=True,
            prev_equity=100_000,
            current_equity=97_000,
            trade_info={"pnl_pct": -0.03, "holding_hours": 4},
        )

        assert components["pnl"] < -3.0

    def test_zero_prev_equity_safe(self) -> None:
        """Ensure zero previous equity handled safely and clipped."""
        config = RewardConfig(pnl_scale=0.01)
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=5,
            action_executed=True,
            prev_equity=0.0,
            current_equity=0.0,
            trade_info={"pnl_pct": 0.0, "holding_hours": 1},
        )

        assert math.isfinite(components["pnl"])
        assert components["pnl"] == pytest.approx(-config.reward_clip)


class TestTransactionCostPenalty:
    """Test transaction cost component."""

    def test_hold_no_cost(self) -> None:
        """HOLD action should not incur cost."""
        config = RewardConfig()
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=0,
            action_executed=True,
            prev_equity=100_000,
            current_equity=100_000,
        )

        assert components["transaction_cost"] == 0.0

    def test_buy_cost_penalty(self) -> None:
        """BUY actions incur cost penalty."""
        config = RewardConfig(pnl_scale=0.01)
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=2,
            action_executed=True,
            prev_equity=100_000,
            current_equity=100_000,
        )

        assert components["transaction_cost"] < 0
        assert -0.2 < components["transaction_cost"] < -0.1

    def test_sell_cost_penalty(self) -> None:
        """SELL actions incur cost penalty."""
        config = RewardConfig(pnl_scale=0.01)
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=5,
            action_executed=True,
            prev_equity=100_000,
            current_equity=100_000,
            trade_info={"pnl_pct": 0.01, "holding_hours": 4},
        )

        assert components["transaction_cost"] < 0

    def test_early_stop_bonus(self) -> None:
        """Bonus for hitting take-profit target reduces penalty."""
        config = RewardConfig(pnl_scale=0.01, early_stop_bonus=0.3)
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=5,
            action_executed=True,
            prev_equity=100_000,
            current_equity=102_500,
            trade_info={"pnl_pct": 0.025, "holding_hours": 3},
        )

        assert components["transaction_cost"] > -0.15

    def test_failed_action_penalty(self) -> None:
        """Failed actions incur small penalty."""
        config = RewardConfig()
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=1,
            action_executed=False,
            prev_equity=100_000,
            current_equity=100_000,
        )

        assert components["transaction_cost"] == pytest.approx(-0.1)

    def test_add_position_extra_cost(self) -> None:
        """Add-position actions carry higher cost."""
        config = RewardConfig(pnl_scale=0.01)
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=6,
            action_executed=True,
            prev_equity=100_000,
            current_equity=100_000,
        )

        assert components["transaction_cost"] < -0.15


class TestTimeEfficiencyReward:
    """Test time efficiency component."""

    def test_quick_win_bonus(self) -> None:
        """Quick profitable exits earn bonus."""
        config = RewardConfig(quick_win_bonus=0.5)
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=5,
            action_executed=True,
            prev_equity=100_000,
            current_equity=101_000,
            trade_info={"pnl_pct": 0.01, "holding_hours": 2},
        )

        assert components["time_efficiency"] > 0.2

    def test_slow_loss_penalty(self) -> None:
        """Holding losers too long incurs penalty."""
        config = RewardConfig()
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=5,
            action_executed=True,
            prev_equity=100_000,
            current_equity=99_000,
            trade_info={"pnl_pct": -0.01, "holding_hours": 7},
        )

        assert components["time_efficiency"] < 0

    def test_no_trade_neutral(self) -> None:
        """No completed trade yields neutral time reward."""
        config = RewardConfig()
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=0,
            action_executed=True,
            prev_equity=100_000,
            current_equity=100_000,
        )

        assert components["time_efficiency"] == 0.0


class TestSharpeReward:
    """Test Sharpe ratio component."""

    def test_above_target_sharpe_positive(self) -> None:
        """Sharpe above target yields positive reward."""
        config = RewardConfig(target_sharpe=0.5)
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=0,
            action_executed=True,
            prev_equity=100_000,
            current_equity=100_000,
            portfolio_state={"sharpe_ratio": 0.75},
        )

        assert components["sharpe"] == pytest.approx(0.5, rel=0.1)

    def test_below_target_sharpe_negative(self) -> None:
        """Sharpe below target yields negative reward."""
        config = RewardConfig(target_sharpe=0.5)
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=0,
            action_executed=True,
            prev_equity=100_000,
            current_equity=100_000,
            portfolio_state={"sharpe_ratio": 0.25},
        )

        assert components["sharpe"] < 0

    def test_nan_sharpe_returns_zero(self) -> None:
        """Non-finite Sharpe ratio should produce zero reward."""
        config = RewardConfig(target_sharpe=0.5)
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=0,
            action_executed=True,
            prev_equity=100_000,
            current_equity=100_000,
            portfolio_state={"sharpe_ratio": float("nan")},
        )

        assert components["sharpe"] == 0.0


class TestDrawdownPenalty:
    """Test drawdown penalty component."""

    def test_no_penalty_small_drawdown(self) -> None:
        """Drawdown within threshold should not penalize."""
        config = RewardConfig(max_drawdown_threshold=0.05)
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=0,
            action_executed=True,
            prev_equity=100_000,
            current_equity=97_000,
            portfolio_state={"peak_equity": 100_000},
        )

        assert components["drawdown"] == 0.0

    def test_severe_drawdown_penalty(self) -> None:
        """Severe drawdowns should be penalized heavily."""
        config = RewardConfig(max_drawdown_threshold=0.05)
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=0,
            action_executed=True,
            prev_equity=100_000,
            current_equity=90_000,
            portfolio_state={"peak_equity": 100_000},
        )

        assert components["drawdown"] < -5.0


class TestSizingReward:
    """Test position sizing component."""

    def test_optimal_sizing_reward(self) -> None:
        """Optimal deployment yields positive reward."""
        config = RewardConfig()
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=1,
            action_executed=True,
            prev_equity=100_000,
            current_equity=100_000,
            portfolio_state={"deployed_pct": 0.6},
        )

        assert components["sizing"] > 0

    def test_over_deployment_penalty(self) -> None:
        """Full deployment triggers penalty."""
        config = RewardConfig()
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=1,
            action_executed=True,
            prev_equity=100_000,
            current_equity=100_000,
            portfolio_state={"deployed_pct": 0.95},
        )

        assert components["sizing"] < 0

    def test_under_deployment_penalty(self) -> None:
        """Under-deployment penalized."""
        config = RewardConfig()
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=1,
            action_executed=True,
            prev_equity=100_000,
            current_equity=100_000,
            portfolio_state={"deployed_pct": 0.1},
        )

        assert components["sizing"] < 0


class TestHoldPenalty:
    """Test hold penalty component."""

    def test_hold_penalty_applies_when_enabled(self) -> None:
        """Hold penalty weight enables negative reward for long holds."""
        config = RewardConfig(hold_penalty_weight=0.1)
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=0,
            action_executed=True,
            prev_equity=100_000,
            current_equity=100_000,
            position_info={"is_open": True, "duration": 8},
        )

        assert components["hold_penalty"] < 0

    def test_no_hold_penalty_when_position_closed(self) -> None:
        """Hold penalty should be zero without open position."""
        config = RewardConfig(hold_penalty_weight=0.1)
        shaper = RewardShaper(config)

        _, components = shaper.compute_reward(
            action=0,
            action_executed=True,
            prev_equity=100_000,
            current_equity=100_000,
            position_info={"is_open": False, "duration": 8},
        )

        assert components["hold_penalty"] == 0.0


class TestWeightedCombination:
    """Test weighted combination of components."""

    def test_total_reward_calculation(self) -> None:
        """Total reward should match weighted sum of components."""
        config = RewardConfig(
            pnl_weight=0.5,
            transaction_cost_weight=0.3,
            time_efficiency_weight=0.2,
            sharpe_weight=0.0,
            drawdown_weight=0.0,
            sizing_weight=0.0,
            hold_penalty_weight=0.0,
        )
        shaper = RewardShaper(config)

        total, components = shaper.compute_reward(
            action=5,
            action_executed=True,
            prev_equity=100_000,
            current_equity=101_000,
            trade_info={"pnl_pct": 0.01, "holding_hours": 2},
        )

        expected = (
            0.5 * components["pnl"]
            + 0.3 * components["transaction_cost"]
            + 0.2 * components["time_efficiency"]
        )

        assert total == pytest.approx(expected, abs=0.01)

    def test_total_reward_clipped(self) -> None:
        """Total reward should respect clip bound."""
        config = RewardConfig(
            reward_clip=1.0,
            pnl_weight=1.0,
            transaction_cost_weight=0.0,
            time_efficiency_weight=0.0,
            sharpe_weight=0.0,
            drawdown_weight=0.0,
            sizing_weight=0.0,
            hold_penalty_weight=0.0,
            pnl_scale=0.0001,
        )
        shaper = RewardShaper(config)

        total, _ = shaper.compute_reward(
            action=5,
            action_executed=True,
            prev_equity=100_000,
            current_equity=120_000,
            trade_info={"pnl_pct": 0.2, "holding_hours": 1},
        )

        assert total == pytest.approx(1.0)


class TestEpisodeTracking:
    """Test episode tracking and statistics."""

    def test_episode_stats(self) -> None:
        """Episode statistics should include core metrics."""
        config = RewardConfig()
        shaper = RewardShaper(config)

        for i in range(10):
            shaper.compute_reward(
                action=0,
                action_executed=True,
                prev_equity=100_000 + i * 100,
                current_equity=100_000 + (i + 1) * 100,
            )

        stats = shaper.get_episode_stats()

        assert "total_reward_mean" in stats
        assert "total_reward_sum" in stats
        assert stats["steps"] == 10

    def test_component_contributions(self) -> None:
        """Component contribution analysis should sum to ~100%."""
        config = RewardConfig()
        shaper = RewardShaper(config)

        shaper.compute_reward(
            action=5,
            action_executed=True,
            prev_equity=100_000,
            current_equity=102_000,
            trade_info={"pnl_pct": 0.02, "holding_hours": 3},
            portfolio_state={"sharpe_ratio": 0.8, "deployed_pct": 0.6},
        )

        contributions = shaper.get_component_contributions()

        assert "pnl" in contributions
        assert "transaction_cost" in contributions
        assert sum(abs(v) for v in contributions.values()) == pytest.approx(100.0, rel=0.1)

    def test_reset_episode(self) -> None:
        """Resetting episode clears histories."""
        config = RewardConfig()
        shaper = RewardShaper(config)

        for _ in range(5):
            shaper.compute_reward(
                action=0,
                action_executed=True,
                prev_equity=100_000,
                current_equity=100_000,
            )

        assert len(shaper.episode_rewards) == 5

        shaper.reset_episode()

        assert len(shaper.episode_rewards) == 0
        assert len(shaper.component_history) == 0

    def test_running_means_window(self) -> None:
        """Running means should average over specified window."""
        config = RewardConfig()
        shaper = RewardShaper(config)

        for _ in range(5):
            shaper.compute_reward(
                action=5,
                action_executed=True,
                prev_equity=100_000,
                current_equity=101_000,
                trade_info={"pnl_pct": 0.01, "holding_hours": 3},
            )

        means = shaper.get_running_means(window=3)

        assert set(config.component_keys) <= set(means.keys())

    def test_recent_components(self) -> None:
        """Recent component retrieval returns requested steps."""
        config = RewardConfig()
        shaper = RewardShaper(config)

        for _ in range(4):
            shaper.compute_reward(
                action=0,
                action_executed=True,
                prev_equity=100_000,
                current_equity=100_000,
            )

        recent = shaper.get_recent_components(n_steps=2)

        assert len(recent) == 2

    def test_get_recent_components_zero(self) -> None:
        """Zero steps returns empty list."""
        config = RewardConfig()
        shaper = RewardShaper(config)

        assert shaper.get_recent_components(0) == []

    def test_get_running_means_invalid_window(self) -> None:
        """Invalid window should raise ValueError."""
        config = RewardConfig()
        shaper = RewardShaper(config)

        with pytest.raises(ValueError):
            shaper.get_running_means(window=0)


class TestConfigurationUpdates:
    """Test configuration mutation helpers."""

    def test_update_config_changes_weight(self) -> None:
        """Update config should change internal weights."""
        config = RewardConfig()
        shaper = RewardShaper(config)

        shaper.update_config(pnl_weight=0.6, sizing_weight=0.1)

        assert config.pnl_weight == pytest.approx(0.6)
        assert config.sizing_weight == pytest.approx(0.1)

    def test_update_invalid_attribute_raises(self) -> None:
        """Updating unknown attribute should raise error."""
        config = RewardConfig()
        shaper = RewardShaper(config)

        with pytest.raises(AttributeError):
            shaper.update_config(nonexistent_weight=0.1)

    def test_summarize_and_clear_stats(self) -> None:
        """Summaries reflect accumulated stats and can be cleared."""
        config = RewardConfig()
        shaper = RewardShaper(config)

        for _ in range(3):
            shaper.compute_reward(
                action=5,
                action_executed=True,
                prev_equity=100_000,
                current_equity=101_000,
                trade_info={"pnl_pct": 0.01, "holding_hours": 2},
            )

        summary = shaper.summarize_reward_stats()
        assert summary["total_rewards_mean"] > 0

        shaper.clear_global_stats()
        cleared = shaper.summarize_reward_stats()
        assert cleared["total_rewards_mean"] == 0

    def test_describe_returns_config_string(self) -> None:
        """Describe should list key configuration values."""
        config = RewardConfig()
        shaper = RewardShaper(config)

        description = shaper.describe()
        assert "pnl_weight" in description
        assert "target_sharpe" in description


class TestUtilityHelpers:
    """Test free functions and static helpers."""

    def test_merge_component_histories(self) -> None:
        """Merging component histories should sum values."""
        history: Dict[str, float] = RewardShaper.merge_component_histories(
            [
                {"pnl": 1.0, "transaction_cost": -0.1},
                {"pnl": 0.5, "transaction_cost": -0.05},
            ]
        )

        assert history["pnl"] == pytest.approx(1.5)
        assert history["transaction_cost"] == pytest.approx(-0.15)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
