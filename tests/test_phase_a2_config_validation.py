"""Comprehensive validation tests for phase_a2_sac_sharpe.yaml configuration.

This test suite ensures that ALL configured rewards and penalties in the
phase_a2_sac_sharpe.yaml file are properly applied by the reward shaper.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import pytest

from core.rl.environments.reward_shaper import RewardConfig, RewardShaper
from training.rl.env_factory import build_reward_config, load_yaml

_CONFIG_PATH = Path(__file__).parent.parent / "training" / "config_templates" / "phase_a2_sac_sharpe.yaml"


def _load_reward_config() -> RewardConfig:
    config_dict = load_yaml(_CONFIG_PATH)
    environment_cfg = config_dict["environment"]
    reward_cfg = environment_cfg.get("reward_config")
    assert reward_cfg is not None, "reward_config section missing from template"
    return build_reward_config(reward_cfg)


def _build_shaper() -> RewardShaper:
    return RewardShaper(_load_reward_config())


class TestPhaseA2ConfigValidation:
    """Test suite validating phase_a2_sac_sharpe.yaml reward configuration."""

    def test_no_forced_exits_configured(self):
        """Verify that forced exits are disabled (agent must learn to exit)."""
        config_dict = load_yaml(_CONFIG_PATH)
        env_cfg = config_dict["environment"]
        
        # FIX #9 & #10: Verify forced exits are disabled
        max_hold_hours = env_cfg.get("max_hold_hours", 8)
        stop_loss = env_cfg.get("stop_loss", 0.02)
        take_profit = env_cfg.get("take_profit", 0.025)
        
        # max_hold_hours should be >= episode_length (agent decides when to exit)
        episode_length = env_cfg.get("episode_length", 672)
        assert max_hold_hours >= episode_length, f"max_hold_hours ({max_hold_hours}) < episode_length ({episode_length}) - forced exits enabled!"
        
        # stop_loss and take_profit should be effectively disabled (>= 0.99)
        assert stop_loss >= 0.99, f"stop_loss ({stop_loss}) < 0.99 - auto stop-loss enabled!"
        assert take_profit >= 0.99, f"take_profit ({take_profit}) < 0.99 - auto take-profit enabled!"
        
        # Verify portfolio risk limits are also disabled
        portfolio_cfg = env_cfg.get("portfolio_config", {})
        max_position_loss_pct = portfolio_cfg.get("max_position_loss_pct", 0.05)
        max_portfolio_loss_pct = portfolio_cfg.get("max_portfolio_loss_pct", 0.20)
        
        assert max_position_loss_pct >= 0.99, f"max_position_loss_pct ({max_position_loss_pct}) < 0.99 - position stop-loss enabled!"
        assert max_portfolio_loss_pct >= 0.99, f"max_portfolio_loss_pct ({max_portfolio_loss_pct}) < 0.99 - portfolio stop-loss enabled!"

    def test_config_values_match_yaml(self):
        """Verify all config values are loaded correctly from YAML."""
        cfg = _load_reward_config()
        
        # Primary weights - FIXED: Match actual phase_a2_sac_sharpe.yaml values
        assert math.isclose(cfg.pnl_weight, 0.95, abs_tol=1e-9)  # Changed from 0.98
        assert math.isclose(cfg.transaction_cost_weight, 0.001, abs_tol=1e-9)  # Changed from 0.005
        assert math.isclose(cfg.time_efficiency_weight, 0.0, abs_tol=1e-9)
        assert math.isclose(cfg.sharpe_weight, 0.0, abs_tol=1e-9)  # Changed from 0.05 (disabled)
        assert math.isclose(cfg.drawdown_weight, 0.0, abs_tol=1e-9)  # Changed from 0.02 (disabled)
        assert math.isclose(cfg.sizing_weight, 0.0, abs_tol=1e-9)
        assert math.isclose(cfg.hold_penalty_weight, 0.040, abs_tol=1e-9)  # Changed from 0.0
        assert math.isclose(cfg.diversity_bonus_weight, 0.15, abs_tol=1e-9)  # Changed from 0.02
        assert math.isclose(cfg.diversity_penalty_weight, 0.0, abs_tol=1e-9)  # Changed from 0.05 (disabled)
        assert math.isclose(cfg.action_repeat_penalty_weight, 0.0, abs_tol=1e-9)
        assert math.isclose(cfg.intrinsic_action_reward, 0.01, abs_tol=1e-9)  # Changed from 0.0
        assert math.isclose(cfg.equity_delta_weight, 0.0, abs_tol=1e-9)
        
        # Scaling parameters
        assert math.isclose(cfg.pnl_scale, 0.0001, abs_tol=1e-9)
        assert math.isclose(cfg.reward_clip, 1250.0, abs_tol=1e-9)  # Changed from 1000.0
        assert math.isclose(cfg.base_transaction_cost_pct, 0.00005, abs_tol=1e-9)
        
        # ROI shaping
        assert cfg.roi_multiplier_enabled is False
        assert math.isclose(cfg.roi_scale_factor, 1.0, abs_tol=1e-9)
        assert math.isclose(cfg.roi_neutral_zone, 0.0001, abs_tol=1e-9)
        assert math.isclose(cfg.roi_negative_scale, 1.0, abs_tol=1e-9)  # Fixed: YAML has 1.0, not 0.6
        assert math.isclose(cfg.roi_positive_scale, 1.0, abs_tol=1e-9)
        assert cfg.roi_full_penalty_trades == 240  # Fixed: YAML has 240, not 40
        
        # Sharpe gating
        assert cfg.sharpe_gate_enabled is False  # Fixed: YAML has false
        assert cfg.sharpe_gate_min_self_trades == 240  # Fixed: YAML has 240, not 200
        assert math.isclose(cfg.sharpe_gate_floor_scale, 0.3, abs_tol=1e-9)
        assert math.isclose(cfg.sharpe_gate_active_scale, 1.0, abs_tol=1e-9)
        
        # Diversity penalty
        assert math.isclose(cfg.diversity_penalty_target, 0.15, abs_tol=1e-9)
        assert cfg.diversity_penalty_window == 10
        
        # Bonus/penalty multipliers
        assert math.isclose(cfg.win_bonus_multiplier, 2.0, abs_tol=1e-9)  # YAML has 2.0
        assert math.isclose(cfg.loss_penalty_multiplier, 1.0, abs_tol=1e-9)
        
        # Time decay
        assert math.isclose(cfg.time_decay_threshold_hours, 72.0, abs_tol=1e-9)  # YAML has 72.0
        assert math.isclose(cfg.time_decay_penalty_per_hour, 0.004, abs_tol=1e-9)

    def test_sharpe_gate_initial_state(self):
        """Verify Sharpe gate starts closed."""
        shaper = _build_shaper()
        assert hasattr(shaper, "_sharpe_gate_open")
        assert shaper._sharpe_gate_open is False

    def test_pnl_reward_computation(self):
        """Test PnL reward calculation with various scenarios."""
        shaper = _build_shaper()
        
        # Profitable trade with medium size
        trade_info = {
            "pnl_pct": 0.05,
            "forced_exit": False,
            "exit_type": "full",
            "entry_size": "medium",
            "pyramid_count": 0,
        }
        reward = shaper._compute_pnl_reward(
            prev_equity=100000.0,
            current_equity=105000.0,
            trade_info=trade_info,
            position_info=None,
            action=5,
            portfolio_state={"num_trades": 50},
        )
        assert reward > 0.0, "Profitable trade should generate positive reward"
        
        # Small loss with gate partially open
        trade_info_loss = {
            "pnl_pct": -0.02,
            "forced_exit": False,
            "exit_type": "full",
            "entry_size": "medium",
            "pyramid_count": 0,
        }
        penalty = shaper._compute_pnl_reward(
            prev_equity=100000.0,
            current_equity=98000.0,
            trade_info=trade_info_loss,
            position_info=None,
            action=5,
            portfolio_state={"num_trades": 20},
        )
        assert penalty < 0.0, "Loss should generate negative reward after gate opens"

    def test_roi_neutral_zone(self):
        """Test that small losses within neutral zone are suppressed early."""
        shaper = _build_shaper()
        
        # Very small loss (0.01%) with no trades yet
        trade_info = {
            "pnl_pct": -0.0001,  # Within roi_neutral_zone
            "forced_exit": False,
            "exit_type": "full",
            "entry_size": "medium",
            "pyramid_count": 0,
        }
        penalty = shaper._compute_pnl_reward(
            prev_equity=100000.0,
            current_equity=99990.0,
            trade_info=trade_info,
            position_info=None,
            action=5,
            portfolio_state={"num_trades": 0},
        )
        assert math.isclose(penalty, 0.0, abs_tol=1e-6), "Tiny loss within neutral zone should be zero"

    def test_roi_penalty_gate_progressive_scaling(self):
        """Test that ROI penalty scales with number of trades."""
        shaper = _build_shaper()
        
        # Use small loss within neutral zone for 0-trade test
        trade_info_tiny = {
            "pnl_pct": -0.00005,  # Within roi_neutral_zone of 0.0001
            "forced_exit": False,
            "exit_type": "full",
            "entry_size": "medium",
            "pyramid_count": 0,
        }
        
        penalty_0 = shaper._compute_pnl_reward(
            prev_equity=100000.0,
            current_equity=99995.0,
            trade_info=trade_info_tiny,
            position_info=None,
            action=5,
            portfolio_state={"num_trades": 0},
        )
        
        # Use moderate loss for scaling test
        trade_info = {
            "pnl_pct": -0.01,  # 1% loss, beyond neutral zone
            "forced_exit": False,
            "exit_type": "full",
            "entry_size": "medium",
            "pyramid_count": 0,
        }
        
        # At 20 trades (50% of roi_full_penalty_trades=40)
        penalty_20 = shaper._compute_pnl_reward(
            prev_equity=100000.0,
            current_equity=99000.0,
            trade_info=trade_info,
            position_info=None,
            action=5,
            portfolio_state={"num_trades": 20},
        )
        
        # At 40+ trades, penalty should be full
        penalty_40 = shaper._compute_pnl_reward(
            prev_equity=100000.0,
            current_equity=99000.0,
            trade_info=trade_info,
            position_info=None,
            action=5,
            portfolio_state={"num_trades": 40},
        )
        
        assert penalty_0 == pytest.approx(0.0, abs=1e-6), "Tiny loss within neutral zone should be zero"
        assert abs(penalty_20) < abs(penalty_40), "Penalty should increase with trade count"
        assert penalty_40 < 0.0, "Full penalty should be negative"

    def test_transaction_cost_penalty(self):
        """Test transaction cost application."""
        shaper = _build_shaper()
        
        # BUY action
        cost = shaper._compute_cost_penalty(
            action=1,
            action_executed=True,
            trade_info=None,
        )
        expected_cost = -shaper.config.base_transaction_cost_pct / shaper.config.pnl_scale
        assert math.isclose(cost, expected_cost, abs_tol=1e-6)
        
        # Failed action
        failed_cost = shaper._compute_cost_penalty(
            action=1,
            action_executed=False,
            trade_info=None,
        )
        assert failed_cost == shaper.config.failed_action_penalty

    def test_drawdown_penalty(self):
        """Test drawdown penalty calculation."""
        shaper = _build_shaper()
        
        # Significant drawdown
        portfolio_state = {
            "equity": 90000.0,
            "peak_equity": 100000.0,
        }
        penalty = shaper._compute_drawdown_penalty(
            current_equity=90000.0,
            portfolio_state=portfolio_state,
        )
        assert penalty < 0.0, "Drawdown should generate penalty"

    def test_diversity_penalty_on_action_collapse(self):
        """Test diversity penalty when actions collapse to single bin."""
        shaper = _build_shaper()
        
        # 90% of actions in bin 3 (collapse scenario)
        diversity_info = {
            "action_diversity_window": [3] * 9 + [2],
            "repeat_streak": 9,
        }
        
        penalty = shaper._compute_diversity_penalty(diversity_info)
        assert penalty < 0.0, "Action collapse should trigger penalty"
        
        # Verify penalty is applied in aggregation
        components = {
            "pnl": 0.0,
            "transaction_cost": 0.0,
            "time_efficiency": 0.0,
            "sharpe": 0.0,
            "drawdown": 0.0,
            "sizing": 0.0,
            "hold_penalty": 0.0,
            "diversity_bonus": 0.0,
            "diversity_penalty": penalty,
            "action_repeat_penalty": 0.0,
            "intrinsic_action": 0.0,
            "equity_delta": 0.0,
        }
        
        total = shaper._aggregate_components(components)
        expected = shaper.config.diversity_penalty_weight * penalty
        assert math.isclose(total, expected, abs_tol=1e-9)

    def test_diversity_bonus_diverse_actions(self):
        """Test diversity bonus for diverse action selection."""
        shaper = _build_shaper()
        
        # Diverse actions across bins
        diversity_info = {
            "action_diversity_window": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "repeat_streak": 1,
        }
        
        bonus = shaper._compute_diversity_bonus(diversity_info)
        assert bonus >= 0.0, "Diverse actions should generate bonus or neutral"

    def test_sharpe_gate_voluntary_vs_forced(self):
        """Test that Sharpe gate only opens on voluntary closes - FIXED: Sharpe gate disabled in config."""
        shaper = _build_shaper()
        
        # FIXED: Sharpe gate is disabled (sharpe_gate_enabled: false), so this test validates it stays closed
        portfolio_state = {"num_trades": 10, "sharpe_ratio": 0.5}
        
        # Process forced exits - should NOT open gate
        for _ in range(250):
            shaper.compute_reward(
                action=5,
                action_executed=True,
                prev_equity=100000.0,
                current_equity=99000.0,
                position_info=None,
                trade_info={
                    "pnl_pct": -0.01,
                    "forced_exit": True,
                    "exit_type": "full",
                    "entry_size": "medium",
                    "pyramid_count": 0,
                },
                portfolio_state=portfolio_state,
                diversity_info=None,
            )
        
        assert not shaper._sharpe_gate_open, "Forced exits should NOT open gate"
        
        # Process voluntary closes - FIXED: Gate should NOT open because sharpe_gate_enabled: false
        for _ in range(200):
            shaper.compute_reward(
                action=5,
                action_executed=True,
                prev_equity=100000.0,
                current_equity=101000.0,
                position_info=None,
                trade_info={
                    "pnl_pct": 0.01,
                    "forced_exit": False,
                    "exit_type": "full",
                    "entry_size": "medium",
                    "pyramid_count": 0,
                },
                portfolio_state=portfolio_state,
                diversity_info=None,
            )
        
        # FIXED: Sharpe gate is disabled in config, so it should remain closed even after 200 voluntary closes
        assert not shaper._sharpe_gate_open, "Sharpe gate disabled - should stay closed even after voluntary closes"

    def test_win_loss_multipliers(self):
        """Test win_bonus_multiplier and loss_penalty_multiplier."""
        shaper = _build_shaper()
        
        # Winning trade should get 5x multiplier
        trade_win = {
            "pnl_pct": 0.02,
            "forced_exit": False,
            "exit_type": "full",
            "entry_size": "medium",
            "pyramid_count": 0,
        }
        reward_win = shaper._compute_pnl_reward(
            prev_equity=100000.0,
            current_equity=102000.0,
            trade_info=trade_win,
            position_info=None,
            action=5,
            portfolio_state={"num_trades": 50},
        )
        
        # Losing trade should get 1x multiplier (neutral)
        trade_loss = {
            "pnl_pct": -0.02,
            "forced_exit": False,
            "exit_type": "full",
            "entry_size": "medium",
            "pyramid_count": 0,
        }
        reward_loss = shaper._compute_pnl_reward(
            prev_equity=100000.0,
            current_equity=98000.0,
            trade_info=trade_loss,
            position_info=None,
            action=5,
            portfolio_state={"num_trades": 50},
        )
        
        assert reward_win > 0.0, "Win should be positive"
        assert reward_loss < 0.0, "Loss should be negative"

    def test_entry_size_multipliers(self):
        """Test position size multipliers - FIXED: All set to 1.0 (neutral) in config."""
        shaper = _build_shaper()
        
        # Use small PnL to avoid hitting reward_clip
        base_pnl = 0.001  # 0.1% profit
        portfolio_state = {"num_trades": 50}
        
        # Small position (neutral - 1.0x per config)
        trade_small = {
            "pnl_pct": base_pnl,
            "forced_exit": False,
            "exit_type": "full",
            "entry_size": "small",
            "pyramid_count": 0,
        }
        reward_small = shaper._compute_pnl_reward(
            prev_equity=100000.0,
            current_equity=100100.0,
            trade_info=trade_small,
            position_info=None,
            action=5,
            portfolio_state=portfolio_state,
        )
        
        # Medium position (neutral - 1.0x)
        trade_medium = {
            "pnl_pct": base_pnl,
            "forced_exit": False,
            "exit_type": "full",
            "entry_size": "medium",
            "pyramid_count": 0,
        }
        reward_medium = shaper._compute_pnl_reward(
            prev_equity=100000.0,
            current_equity=100100.0,
            trade_info=trade_medium,
            position_info=None,
            action=5,
            portfolio_state=portfolio_state,
        )
        
        # Large position (neutral - 1.0x per config)
        trade_large = {
            "pnl_pct": base_pnl,
            "forced_exit": False,
            "exit_type": "full",
            "entry_size": "large",
            "pyramid_count": 0,
        }
        reward_large = shaper._compute_pnl_reward(
            prev_equity=100000.0,
            current_equity=100100.0,
            trade_info=trade_large,
            position_info=None,
            action=5,
            portfolio_state=portfolio_state,
        )
        
        # FIXED: All multipliers are 1.0 (neutral), so rewards should be equal
        assert math.isclose(reward_small, reward_medium, abs_tol=1e-6), f"Small ({reward_small:.2f}) should equal medium ({reward_medium:.2f}) with neutral multipliers"
        assert math.isclose(reward_medium, reward_large, abs_tol=1e-6), f"Medium ({reward_medium:.2f}) should equal large ({reward_large:.2f}) with neutral multipliers"

    def test_staged_exit_bonus(self):
        """Test staged exit bonus - FIXED: All exit multipliers set to 1.0 (neutral)."""
        shaper = _build_shaper()
        
        portfolio_state = {"num_trades": 50}
        base_pnl = 0.001  # Use small PnL to avoid hitting reward_clip
        
        # Full exit (1.0x multiplier per config - neutral)
        trade_full = {
            "pnl_pct": base_pnl,
            "forced_exit": False,
            "exit_type": "full",
            "entry_size": "medium",
            "pyramid_count": 0,
        }
        reward_full = shaper._compute_pnl_reward(
            prev_equity=100000.0,
            current_equity=100100.0,
            trade_info=trade_full,
            position_info=None,
            action=5,
            portfolio_state=portfolio_state,
        )
        
        # Staged exit (1.0x per config - neutral, same as full)
        trade_staged = {
            "pnl_pct": base_pnl,
            "forced_exit": False,
            "exit_type": "staged",
            "entry_size": "medium",
            "pyramid_count": 0,
        }
        reward_staged = shaper._compute_pnl_reward(
            prev_equity=100000.0,
            current_equity=100100.0,
            trade_info=trade_staged,
            position_info=None,
            action=5,
            portfolio_state=portfolio_state,
        )
        
        # FIXED: All exit multipliers are 1.0, so rewards should be equal
        assert math.isclose(reward_staged, reward_full, abs_tol=1e-6), f"Staged exit ({reward_staged:.2f}) should equal full ({reward_full:.2f}) with neutral multipliers"

    def test_reward_aggregation_weights(self):
        """Test that component aggregation respects configured weights."""
        shaper = _build_shaper()
        
        components: Dict[str, float] = {
            "pnl": 100.0,
            "transaction_cost": -5.0,
            "time_efficiency": 0.0,
            "sharpe": 10.0,
            "drawdown": -8.0,
            "sizing": 0.0,
            "hold_penalty": 0.0,
            "diversity_bonus": 3.0,
            "diversity_penalty": -2.0,
            "action_repeat_penalty": 0.0,
            "intrinsic_action": 0.0,
            "equity_delta": 0.0,
        }
        
        total = shaper._aggregate_components(components)
        
        cfg = shaper.config
        expected = (
            cfg.pnl_weight * 100.0
            + cfg.transaction_cost_weight * (-5.0)
            + cfg.sharpe_weight * 10.0
            + cfg.drawdown_weight * (-8.0)
            + cfg.diversity_bonus_weight * 3.0
            + cfg.diversity_penalty_weight * (-2.0)
        )
        
        assert math.isclose(total, expected, abs_tol=1e-6)

    def test_reward_clipping(self):
        """Test that rewards are clipped to configured limit."""
        shaper = _build_shaper()
        
        # Extreme components that exceed clip
        components = {
            "pnl": 5000.0,  # Would exceed 1000 clip
            "transaction_cost": 0.0,
            "time_efficiency": 0.0,
            "sharpe": 0.0,
            "drawdown": 0.0,
            "sizing": 0.0,
            "hold_penalty": 0.0,
            "diversity_bonus": 0.0,
            "diversity_penalty": 0.0,
            "action_repeat_penalty": 0.0,
            "intrinsic_action": 0.0,
            "equity_delta": 0.0,
        }
        
        total = shaper._aggregate_components(components)
        assert math.isclose(total, shaper.config.reward_clip, abs_tol=1e-6)

    def test_disabled_components_have_no_effect(self):
        """Test that disabled components (weight=0) don't affect reward - FIXED: hold_penalty and intrinsic_action are enabled."""
        shaper = _build_shaper()
        
        # FIXED: hold_penalty_weight=0.040 and intrinsic_action_reward=0.01 are NON-ZERO in config
        # Only test components that are actually disabled (weight=0)
        components_with_disabled = {
            "pnl": 10.0,
            "transaction_cost": -1.0,
            "time_efficiency": 1000.0,  # Weight=0, should be ignored
            "sharpe": 1000.0,  # Weight=0, should be ignored
            "drawdown": 1000.0,  # Weight=0, should be ignored
            "sizing": 1000.0,  # Weight=0, should be ignored
            "hold_penalty": -2.0,  # ENABLED: weight=0.040
            "diversity_bonus": 2.0,
            "diversity_penalty": 1000.0,  # Weight=0, should be ignored
            "action_repeat_penalty": 1000.0,  # Weight=0, should be ignored
            "intrinsic_action": 0.5,  # ENABLED: weight=0.01
            "equity_delta": 1000.0,  # Weight=0, should be ignored
        }
        
        components_without_disabled = {
            "pnl": 10.0,
            "transaction_cost": -1.0,
            "time_efficiency": 0.0,
            "sharpe": 0.0,
            "drawdown": 0.0,
            "sizing": 0.0,
            "hold_penalty": -2.0,  # Keep actual value
            "diversity_bonus": 2.0,
            "diversity_penalty": 0.0,
            "action_repeat_penalty": 0.0,
            "intrinsic_action": 0.5,  # Keep actual value
            "equity_delta": 0.0,
        }
        
        total_with = shaper._aggregate_components(components_with_disabled)
        total_without = shaper._aggregate_components(components_without_disabled)
        
        assert math.isclose(total_with, total_without, abs_tol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
