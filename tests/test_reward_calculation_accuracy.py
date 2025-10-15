"""Surgical precision tests for reward calculation accuracy.

This test suite validates that rewards are calculated with mathematical precision
and that all reward components behave as documented.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pytest

from core.rl.environments.reward_shaper import RewardConfig, RewardShaper


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def base_shaper() -> RewardShaper:
    """Create a reward shaper with default configuration."""
    config = RewardConfig()
    return RewardShaper(config)


@pytest.fixture
def isolated_pnl_shaper() -> RewardShaper:
    """Create shaper with only PnL component enabled."""
    config = RewardConfig(
        pnl_weight=1.0,
        transaction_cost_weight=0.0,
        time_efficiency_weight=0.0,
        sharpe_weight=0.0,
        drawdown_weight=0.0,
        sizing_weight=0.0,
        hold_penalty_weight=0.0,
        diversity_bonus_weight=0.0,
        diversity_penalty_weight=0.0,
        pnl_scale=0.0001,
        reward_clip=1000.0,
    )
    return RewardShaper(config)


# ============================================================================
# TEST 1: PNL REWARD CALCULATION
# ============================================================================

def test_pnl_reward_scales_with_profit_percentage():
    """Test that PnL reward scales linearly with profit percentage."""
    config = RewardConfig(
        pnl_weight=1.0,
        transaction_cost_weight=0.0,
        time_efficiency_weight=0.0,
        sharpe_weight=0.0,
        drawdown_weight=0.0,
        sizing_weight=0.0,
        pnl_scale=0.0001,
        reward_clip=1000.0,
        roi_multiplier_enabled=False,
    )
    shaper = RewardShaper(config)
    
    trade_info = {
        "pnl_pct": 0.01,  # 1% profit
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }
    
    reward_1pct = shaper._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=101000.0,
        trade_info=trade_info,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 50},
    )
    
    # Double the profit
    trade_info["pnl_pct"] = 0.02
    reward_2pct = shaper._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=102000.0,
        trade_info=trade_info,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 50},
    )
    
    # Reward should approximately double (within rounding)
    ratio = reward_2pct / reward_1pct if reward_1pct != 0 else 0
    assert 1.8 < ratio < 2.2, f"Expected ~2x reward, got {ratio:.2f}x"


def test_pnl_reward_positive_for_profit(isolated_pnl_shaper: RewardShaper):
    """Test that profitable trades generate positive rewards."""
    trade_info = {
        "pnl_pct": 0.05,  # 5% profit
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }
    
    reward = isolated_pnl_shaper._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=105000.0,
        trade_info=trade_info,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 50},
    )
    
    assert reward > 0.0, "Profitable trade should generate positive reward"


def test_pnl_reward_negative_for_loss(isolated_pnl_shaper: RewardShaper):
    """Test that losing trades generate negative rewards."""
    trade_info = {
        "pnl_pct": -0.03,  # 3% loss
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }
    
    reward = isolated_pnl_shaper._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=97000.0,
        trade_info=trade_info,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 50},
    )
    
    assert reward < 0.0, "Losing trade should generate negative reward"


def test_pnl_reward_loss_penalty_multiplier_increases_magnitude():
    """Test that loss_penalty_multiplier increases loss penalties."""
    config_base = RewardConfig(
        pnl_weight=1.0,
        transaction_cost_weight=0.0,
        pnl_scale=0.01,  # Larger scale to avoid clipping
        loss_penalty_multiplier=1.0,
        roi_multiplier_enabled=False,
        reward_clip=100.0,  # Higher clip
    )
    
    config_amplified = RewardConfig(
        pnl_weight=1.0,
        transaction_cost_weight=0.0,
        pnl_scale=0.01,  # Larger scale to avoid clipping
        loss_penalty_multiplier=2.0,  # Double the penalty
        roi_multiplier_enabled=False,
        reward_clip=100.0,  # Higher clip
    )
    
    shaper_base = RewardShaper(config_base)
    shaper_amplified = RewardShaper(config_amplified)
    
    trade_info = {
        "pnl_pct": -0.005,  # Smaller loss to avoid clip
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }
    
    reward_base = shaper_base._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=99500.0,
        trade_info=trade_info,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 50},
    )
    
    reward_amplified = shaper_amplified._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=99500.0,
        trade_info=trade_info,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 50},
    )
    
    # Amplified penalty should be roughly 2x (allow some tolerance for rounding)
    assert abs(reward_amplified) >= abs(reward_base) * 1.9, f"Amplified loss penalty should be ~2x: base={reward_base}, amplified={reward_amplified}"


def test_pnl_reward_respects_reward_clip(isolated_pnl_shaper: RewardShaper):
    """Test that PnL rewards are clipped to configured bounds."""
    trade_info = {
        "pnl_pct": 0.50,  # 50% profit (extreme)
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }
    
    reward = isolated_pnl_shaper._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=150000.0,
        trade_info=trade_info,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 50},
    )
    
    clip = isolated_pnl_shaper.config.reward_clip
    assert -clip <= reward <= clip, f"Reward {reward} exceeds clip bound {clip}"


# ============================================================================
# TEST 2: POSITION SIZING MULTIPLIERS
# ============================================================================

def test_position_size_multipliers_can_differentiate():
    """Test that position size multipliers work when configured differently.
    
    Note: phase_a2_sac_sharpe.yaml has ALL multipliers set to 1.0 (neutral)
    to simplify the reward function. This test validates that the mechanism
    WORKS when different values are configured.
    """
    config = RewardConfig(
        pnl_weight=1.0,
        transaction_cost_weight=0.0,
        pnl_scale=0.01,
        position_size_small_multiplier=1.5,  # Different from production config
        position_size_medium_multiplier=1.0,
        roi_multiplier_enabled=False,
        reward_clip=100.0,
    )
    shaper = RewardShaper(config)

    trade_info_small = {
        "pnl_pct": 0.005,
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "small",
        "pyramid_count": 0,
    }

    trade_info_medium = {
        "pnl_pct": 0.005,
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }

    reward_small = shaper._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=100500.0,
        trade_info=trade_info_small,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 50},
    )
    
    reward_medium = shaper._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=100500.0,
        trade_info=trade_info_medium,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 50},
    )
    
    # When configured with different multipliers, rewards should differ
    assert reward_small >= reward_medium * 1.4, f"Small position multiplier (1.5x) should increase reward: small={reward_small}, medium={reward_medium}"
def test_neutral_position_multipliers_produce_equal_rewards():
    """Test that neutral position size multipliers (1.0) produce equal rewards.
    
    This validates the phase_a2_sac_sharpe.yaml config where ALL position size
    multipliers are set to 1.0 to simplify the reward function.
    """
    config = RewardConfig(
        pnl_weight=0.95,
        transaction_cost_weight=0.001,
        pnl_scale=0.0001,  # Match production config
        position_size_small_multiplier=1.0,  # NEUTRAL (production config)
        position_size_medium_multiplier=1.0,  # NEUTRAL (production config)
        position_size_large_multiplier=1.0,  # NEUTRAL (production config)
        roi_multiplier_enabled=False,
        reward_clip=1250.0,  # Match production config
    )
    shaper = RewardShaper(config)

    trade_info_small = {
        "pnl_pct": 0.01,  # 1% profit
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "small",
        "pyramid_count": 0,
    }

    trade_info_medium = {
        "pnl_pct": 0.01,
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }
    
    trade_info_large = {
        "pnl_pct": 0.01,
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "large",
        "pyramid_count": 0,
    }

    reward_small = shaper._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=101000.0,
        trade_info=trade_info_small,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 50},
    )
    
    reward_medium = shaper._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=101000.0,
        trade_info=trade_info_medium,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 50},
    )
    
    reward_large = shaper._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=101000.0,
        trade_info=trade_info_large,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 50},
    )
    
    # With all multipliers = 1.0, all rewards should be equal
    assert abs(reward_small - reward_medium) < 0.01, f"Neutral multipliers should produce equal rewards: small={reward_small}, medium={reward_medium}"
    assert abs(reward_medium - reward_large) < 0.01, f"Neutral multipliers should produce equal rewards: medium={reward_medium}, large={reward_large}"
# ============================================================================
# TEST 3: EXIT TYPE MULTIPLIERS
# ============================================================================

def test_neutral_exit_multipliers_produce_equal_rewards():
    """Test that neutral exit multipliers (1.0) produce equal rewards.
    
    This validates the phase_a2_sac_sharpe.yaml config where partial and full
    exit multipliers are both set to 1.0 to simplify the reward function.
    """
    config = RewardConfig(
        pnl_weight=0.95,
        transaction_cost_weight=0.001,
        pnl_scale=0.0001,  # Match production config
        partial_exit_multiplier=1.0,  # NEUTRAL (production config)
        full_exit_multiplier=1.0,  # NEUTRAL (production config)
        staged_exit_bonus=1.0,  # NEUTRAL (production config)
        roi_multiplier_enabled=False,
        reward_clip=1250.0,
    )
    shaper = RewardShaper(config)

    trade_info_partial = {
        "pnl_pct": 0.01,  # 1% profit
        "forced_exit": False,
        "exit_type": "partial",
        "entry_size": "medium",
        "pyramid_count": 0,
    }

    trade_info_full = {
        "pnl_pct": 0.01,
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }

    reward_partial = shaper._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=101000.0,
        trade_info=trade_info_partial,
        position_info=None,
        action=4,
        portfolio_state={"num_trades": 50},
    )
    
    reward_full = shaper._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=101000.0,
        trade_info=trade_info_full,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 50},
    )
    
    # With neutral multipliers (1.0), partial and full exits should give equal rewards
    assert abs(reward_partial - reward_full) < 0.01, f"Neutral exit multipliers should produce equal rewards: partial={reward_partial}, full={reward_full}"
# ============================================================================
# TEST 4: TRANSACTION COST CALCULATION
# ============================================================================
# NOTE: Transaction costs are integrated into the overall reward computation
# and not exposed as a separate method. These tests verify that transaction
# costs are properly accounted for in the final reward.

def test_transaction_cost_reduces_reward():
    """Test that transaction costs reduce overall reward."""
    config_with_cost = RewardConfig(
        pnl_weight=1.0,
        transaction_cost_weight=1.0,
        base_transaction_cost_pct=0.001,
        pnl_scale=0.0001,
        roi_multiplier_enabled=False,
    )
    
    config_no_cost = RewardConfig(
        pnl_weight=1.0,
        transaction_cost_weight=0.0,
        base_transaction_cost_pct=0.001,
        pnl_scale=0.0001,
        roi_multiplier_enabled=False,
    )
    
    shaper_with_cost = RewardShaper(config_with_cost)
    shaper_no_cost = RewardShaper(config_no_cost)
    
    trade_info = {
        "pnl_pct": 0.01,
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }
    
    reward_with_cost = shaper_with_cost._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=101000.0,
        trade_info=trade_info,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 50},
    )
    
    reward_no_cost = shaper_no_cost._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=101000.0,
        trade_info=trade_info,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 50},
    )
    
    # Transaction costs should reduce net reward
    assert reward_with_cost <= reward_no_cost, "Transaction costs should reduce or equal reward"


# ============================================================================
# TEST 5: DIVERSITY BONUS/PENALTY
# ============================================================================

def test_diversity_bonus_for_varied_actions():
    """Test that diverse action distribution generates bonus."""
    config = RewardConfig(
        diversity_bonus_weight=1.0,
    )
    shaper = RewardShaper(config)
    
    # Diverse action distribution
    diversity_info = {
        "action_diversity_window": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3],  # 6 unique actions
        "repeat_streak": 1,
    }
    
    bonus = shaper._compute_diversity_bonus(diversity_info)
    
    assert bonus > 0.0, "Diverse actions should generate positive bonus"


def test_diversity_penalty_for_action_collapse():
    """Test that action collapse (e.g., 90% HOLD) generates penalty."""
    config = RewardConfig(
        diversity_penalty_weight=1.0,
        diversity_penalty_target=0.2,  # Max 20% for any single action
    )
    shaper = RewardShaper(config)
    
    # Heavily biased toward one action
    diversity_info = {
        "action_diversity_window": [0] * 9 + [1],  # 90% action 0
        "repeat_streak": 9,
    }
    
    penalty = shaper._compute_diversity_penalty(diversity_info)
    
    assert penalty < 0.0, "Action collapse should generate negative penalty"


def test_diversity_penalty_zero_for_balanced_distribution():
    """Test that balanced action distribution has no penalty."""
    config = RewardConfig(
        diversity_penalty_weight=1.0,
        diversity_penalty_target=0.2,
    )
    shaper = RewardShaper(config)
    
    # Balanced distribution (5 actions, 20% each)
    diversity_info = {
        "action_diversity_window": [0, 1, 2, 3, 4] * 2,
        "repeat_streak": 1,
    }
    
    penalty = shaper._compute_diversity_penalty(diversity_info)
    
    # Should be close to zero
    assert abs(penalty) < 0.1, "Balanced distribution should have minimal penalty"


# ============================================================================
# TEST 6: REWARD COMPONENT AGGREGATION
# ============================================================================

def test_aggregation_applies_component_weights_correctly():
    """Test that component weights are applied during aggregation."""
    config = RewardConfig(
        pnl_weight=0.7,
        transaction_cost_weight=0.2,
        diversity_bonus_weight=0.1,
    )
    shaper = RewardShaper(config)
    
    components = {
        "pnl": 10.0,
        "transaction_cost": -2.0,
        "time_efficiency": 0.0,
        "sharpe": 0.0,
        "drawdown": 0.0,
        "sizing": 0.0,
        "hold_penalty": 0.0,
        "diversity_bonus": 5.0,
        "diversity_penalty": 0.0,
        "action_repeat_penalty": 0.0,
        "intrinsic_action": 0.0,
        "equity_delta": 0.0,
    }
    
    total = shaper._aggregate_components(components)
    
    expected = (
        0.7 * 10.0 +  # pnl
        0.2 * -2.0 +  # transaction_cost
        0.1 * 5.0     # diversity_bonus
    )
    
    assert math.isclose(total, expected, abs_tol=1e-6), \
        f"Expected {expected}, got {total}"


def test_aggregation_respects_reward_clip():
    """Test that aggregated reward is clipped."""
    config = RewardConfig(
        pnl_weight=1.0,
        reward_clip=5.0,
    )
    shaper = RewardShaper(config)
    
    components = {
        "pnl": 100.0,  # Very large value
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
    
    assert -5.0 <= total <= 5.0, f"Reward {total} not clipped to [-5, 5]"


# ============================================================================
# TEST 7: ROI-BASED SCALING
# ============================================================================

def test_roi_negative_scale_reduces_small_losses():
    """Test that ROI negative scaling mechanism works when enabled.
    
    Note: The phase_a2_sac_sharpe.yaml config has roi_multiplier_enabled=False,
    so this test documents the behavior when ROI shaping IS enabled. The
    roi_negative_scale parameter only affects rewards when roi_multiplier_enabled=True.
    """
    # Test _apply_roi_shaping directly with different negative scales
    config_scaled = RewardConfig(
        progressive_roi_enabled=False,  # Disable progressive ROI to test negative scaling
        roi_negative_scale=0.5,  # Reduce loss penalties by 50%
        roi_neutral_zone=0.001,  # Small neutral zone
        roi_full_penalty_trades=60,
    )
    
    config_unscaled = RewardConfig(
        progressive_roi_enabled=False,  # Disable progressive ROI to test negative scaling
        roi_negative_scale=1.0,  # No scaling
        roi_neutral_zone=0.001,  # Small neutral zone
        roi_full_penalty_trades=60,
    )
    
    shaper_scaled = RewardShaper(config_scaled)
    shaper_unscaled = RewardShaper(config_unscaled)
    
    # Test the _apply_roi_shaping method directly with a loss beyond neutral zone
    pnl_pct = -0.01  # 1% loss
    trades_completed = 80.0  # Beyond roi_full_penalty_trades
    
    adjusted_scaled = shaper_scaled._apply_roi_shaping(pnl_pct, trades_completed)
    adjusted_unscaled = shaper_unscaled._apply_roi_shaping(pnl_pct, trades_completed)
    
    # The scaled version should have smaller loss magnitude
    assert abs(adjusted_scaled) < abs(adjusted_unscaled), \
        f"ROI negative scaling should reduce adjusted ROI magnitude: scaled={adjusted_scaled}, unscaled={adjusted_unscaled}"


# ============================================================================
# TEST 8: NUMERICAL STABILITY
# ============================================================================

def test_reward_calculation_handles_zero_equity():
    """Test that reward calculation handles edge case of zero equity."""
    config = RewardConfig(pnl_weight=1.0)
    shaper = RewardShaper(config)
    
    trade_info = {
        "pnl_pct": 0.01,
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }
    
    # Should not crash with zero equity
    try:
        reward = shaper._compute_pnl_reward(
            prev_equity=0.0,
            current_equity=0.0,
            trade_info=trade_info,
            position_info=None,
            action=5,
            portfolio_state={"num_trades": 50},
        )
        success = True
    except (ZeroDivisionError, ValueError):
        success = False
    
    assert success, "Reward calculation should handle zero equity gracefully"


def test_reward_calculation_handles_nan_inputs():
    """Test that NaN inputs don't cause application crashes.
    
    Note: It's acceptable for the reward shaper to either:
    1. Return NaN/inf (which will be caught by clipping)
    2. Raise ValueError/TypeError for invalid input
    3. Return 0.0 for safety
    
    What we're testing is that it doesn't cause unexpected crashes.
    """
    config = RewardConfig(pnl_weight=1.0, roi_multiplier_enabled=False)
    shaper = RewardShaper(config)
    
    trade_info = {
        "pnl_pct": float('nan'),
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }
    
    # Should not crash with unhandled exception
    # NaN in calculations is acceptable (will be clipped)
    try:
        reward = shaper._compute_pnl_reward(
            prev_equity=100000.0,
            current_equity=101000.0,
            trade_info=trade_info,
            position_info=None,
            action=5,
            portfolio_state={"num_trades": 50},
        )
        # Result can be NaN, inf, or finite - all are handled by clipping
        success = True
    except (ValueError, TypeError):
        # It's acceptable to raise these for invalid input
        success = True
    except Exception:
        # Any other exception is unexpected
        success = False
    
    assert success, "Should handle NaN inputs without unexpected crashes"


# ============================================================================
# TEST 9: FORCED EXIT PENALTIES
# ============================================================================

def test_forced_exit_base_penalty():
    """Test that forced exits incur base penalty."""
    config = RewardConfig(
        time_efficiency_weight=1.0,
        forced_exit_base_penalty=0.5,
        forced_exit_loss_scale=0.0,  # Disable loss scaling
    )
    shaper = RewardShaper(config)
    
    trade_info = {
        "pnl_pct": 0.0,  # Neutral PnL
        "holding_hours": 5,
        "action": "position_loss_limit",
        "forced_exit": True,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }
    
    penalty = shaper._compute_time_reward(
        trade_info=trade_info,
        position_info=None,
    )
    
    assert penalty <= -0.5, "Forced exit should incur base penalty"


def test_forced_exit_scales_with_loss():
    """Test that forced exit penalty scales with loss magnitude."""
    config = RewardConfig(
        time_efficiency_weight=1.0,
        forced_exit_base_penalty=0.2,
        forced_exit_loss_scale=3.0,
        forced_exit_penalty_cap=2.0,
    )
    shaper = RewardShaper(config)
    
    trade_info_small_loss = {
        "pnl_pct": -0.01,
        "holding_hours": 5,
        "action": "position_loss_limit",
        "forced_exit": True,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }
    
    trade_info_large_loss = {
        "pnl_pct": -0.05,
        "holding_hours": 5,
        "action": "position_loss_limit",
        "forced_exit": True,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }
    
    penalty_small = shaper._compute_time_reward(
        trade_info=trade_info_small_loss,
        position_info=None,
    )
    
    penalty_large = shaper._compute_time_reward(
        trade_info=trade_info_large_loss,
        position_info=None,
    )
    
    assert penalty_large < penalty_small, \
        "Larger loss should incur larger forced exit penalty"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
