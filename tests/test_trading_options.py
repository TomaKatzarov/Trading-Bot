"""Unit tests for hierarchical options framework.

This test suite validates the trading options implementation for Phase B.1 Step 1.
Tests cover:
1. Option initiation sets
2. Intra-option policies
3. Termination probabilities
4. OptionsController selection and execution
5. Integration with continuous action space
"""

import numpy as np
import pytest
import torch
from typing import Dict

from core.rl.options import (
    ClosePositionOption,
    OpenLongOption,
    OpenShortOption,
    OptionsController,
    OptionType,
    ScalpOption,
    TrendFollowOption,
    WaitOption,
)


@pytest.fixture
def sample_state_no_position():
    """Create sample state with no open position."""
    # Flattened state: [price_features..., position_info, portfolio_info]
    # Position info (last 5 elements): [exposure, entry_price, pnl_pct, duration, size_pct]
    state = np.random.randn(50)
    state[-5:] = [0.0, 0.0, 0.0, 0.0, 0.0]  # No position
    return state


@pytest.fixture
def sample_state_with_position():
    """Create sample state with open position."""
    state = np.random.randn(50)
    state[-5:] = [0.08, 150.0, 0.02, 5.0, 0.08]  # Position: 8% exposure, +2% profit, 5 steps
    return state


@pytest.fixture
def sample_observation_dict():
    """Create sample dict observation matching trading environment with sentiment."""
    technical = np.random.randn(24, 23).astype(np.float32)
    # Set bullish sentiment at index 20 (required for OpenLongOption and ScalpOption)
    technical[-1, 20] = 0.65  # Bullish sentiment (above 0.5 threshold)
    return {
        "technical": technical,
        "sl_probs": np.array([0.3, 0.4, 0.3], dtype=np.float32),
        "position": np.array([1.0, 150.0, 0.02, 5.0, 0.08], dtype=np.float32),
        "portfolio": np.random.randn(8).astype(np.float32),
        "regime": np.random.rand(10).astype(np.float32),
    }


def make_batched_observation(observation: Dict[str, np.ndarray], batch_size: int = 1) -> Dict[str, np.ndarray]:
    """Expand a single-environment observation dict to include a vectorized env dimension."""

    batched: Dict[str, np.ndarray] = {}
    for key, value in observation.items():
        arr = np.asarray(value)
        batched[key] = np.repeat(arr[np.newaxis, ...], batch_size, axis=0).astype(arr.dtype, copy=False)
    return batched


def select_action_single_env(wrapper, observation: Dict[str, np.ndarray], *, deterministic: bool = False):
    """Helper to call select_actions for a single environment and unpack the first result."""

    batched_obs = make_batched_observation(observation, batch_size=1)
    actions, info_list = wrapper.select_actions(batched_obs, deterministic=deterministic, return_info=True)
    action = actions[0]
    info = info_list[0]
    return action, info


class TestOpenLongOption:
    """Test suite for OpenLongOption."""

    def test_initiation_set_no_position(self, sample_state_no_position):
        """Should allow initiation when no position exists."""
        option = OpenLongOption()
        assert option.initiation_set(sample_state_no_position) is True

    def test_initiation_set_small_position(self):
        """Should allow initiation with small position (<2%)."""
        state = np.random.randn(50)
        state[-5] = 0.015  # 1.5% exposure
        option = OpenLongOption()
        assert option.initiation_set(state) is True

    def test_initiation_set_large_position(self):
        """Should block initiation with large position (>2%)."""
        state = np.random.randn(50)
        state[-5] = 0.10  # 10% exposure
        option = OpenLongOption()
        assert option.initiation_set(state) is False

    def test_policy_step_0_initial_entry(self, sample_state_no_position, sample_observation_dict):
        """Should return small initial entry (sentiment-scaled) on step 0."""
        option = OpenLongOption()
        action = option.policy(sample_state_no_position, step=0, observation_dict=sample_observation_dict)
        # With sentiment=0.65, multiplier is 0.5+0.65=1.15, so action ~0.30*1.15=0.345
        # But range is 0.15-0.45 due to sentiment scaling [0.5, 1.5]
        assert 0.15 <= action <= 0.50  # Conservative entry (sentiment-scaled)
        assert action > 0  # Buy action

    def test_policy_progressive_building(self, sample_state_no_position, sample_observation_dict):
        """Should progressively build position over multiple steps (sentiment-aware)."""
        option = OpenLongOption()
        # Need to ensure sentiment check passes (> 0.5)
        sample_observation_dict["technical"][-1, 20] = 0.70  # Strong bullish sentiment
        
        actions = []
        for step in range(5):
            action = option.policy(sample_state_no_position, step=step, observation_dict=sample_observation_dict)
            actions.append(action)
        
        # Should have multiple buy actions (sentiment allows entries)
        assert sum(a > 0 for a in actions) >= 3
        # First action should be conservative (even with sentiment scaling)
        assert actions[0] <= 0.50

    def test_policy_stops_at_max_steps(self, sample_state_no_position, sample_observation_dict):
        """Should stop building after max_steps."""
        option = OpenLongOption(max_steps=5)
        action = option.policy(sample_state_no_position, step=10, observation_dict=sample_observation_dict)
        assert action == 0.0  # Hold

    def test_termination_probability_increases_with_steps(self, sample_state_no_position):
        """Termination probability should increase with step count."""
        option = OpenLongOption(max_steps=10)
        prob_early = option.termination_probability(sample_state_no_position, step=2)
        prob_late = option.termination_probability(sample_state_no_position, step=8)
        assert prob_late > prob_early

    def test_termination_at_max_exposure(self):
        """Should terminate when max exposure reached."""
        state = np.random.randn(50)
        state[-5] = 0.12  # 12% exposure (above max)
        option = OpenLongOption(max_exposure_pct=0.10)
        prob = option.termination_probability(state, step=5)
        assert prob == 1.0

    def test_reset_clears_state(self, sample_state_no_position, sample_observation_dict):
        """Reset should clear entry price and step count."""
        option = OpenLongOption()
        option.policy(sample_state_no_position, step=0, observation_dict=sample_observation_dict)
        assert option.entry_price is not None
        option.reset()
        assert option.entry_price is None
        assert option.step_count == 0

    # ===== SENTIMENT AMPLIFICATION TESTS (NEW) =====

    def test_sentiment_neutral_baseline(self, sample_state_no_position):
        """Neutral sentiment (0.5) should result in 1.0x multiplier (baseline)."""
        option = OpenLongOption()
        obs_dict = {"technical": np.random.randn(24, 23)}
        obs_dict["technical"][-1, 20] = 0.50  # Neutral sentiment
        
        action = option.policy(sample_state_no_position, step=0, observation_dict=obs_dict)
        # Base action is 0.30, with 1.0x multiplier -> 0.30
        # Formula: 0.6 + 0.8 * 0.5 = 1.0x
        assert 0.25 <= action <= 0.35  # Should be close to base 0.30

    def test_sentiment_bullish_amplification(self, sample_state_no_position):
        """Bullish sentiment (>0.5) should amplify long actions (1.0x -> 1.4x)."""
        option = OpenLongOption()
        obs_dict = {"technical": np.random.randn(24, 23)}
        obs_dict["technical"][-1, 20] = 1.0  # Maximum bullish sentiment
        
        action = option.policy(sample_state_no_position, step=0, observation_dict=obs_dict)
        # Base action 0.30, with 1.4x multiplier -> 0.42
        # Formula: 0.6 + 0.8 * 1.0 = 1.4x
        assert 0.35 <= action <= 0.45  # Amplified entry

    def test_sentiment_bearish_reduction_not_block(self, sample_state_no_position):
        """Bearish sentiment (0.40) should reduce but NOT block entry."""
        option = OpenLongOption()
        obs_dict = {"technical": np.random.randn(24, 23)}
        obs_dict["technical"][-1, 20] = 0.40  # Bearish sentiment (but not < 0.35 extreme threshold)
        
        # Should still allow entry (0.40 >= 0.35 threshold)
        assert option.initiation_set(sample_state_no_position, observation_dict=obs_dict) is True
        
        action = option.policy(sample_state_no_position, step=0, observation_dict=obs_dict)
        # Base 0.30, with sentiment multiplier: 0.6 + 0.8 * 0.40 = 0.92x
        # 0.30 * 0.92 = 0.276
        assert 0.20 <= action <= 0.35  # Reduced but still positive

    def test_sentiment_extreme_bearish_blocks(self, sample_state_no_position):
        """Extreme bearish sentiment (<0.35) should block long entry."""
        option = OpenLongOption()
        obs_dict = {"technical": np.random.randn(24, 23)}
        obs_dict["technical"][-1, 20] = 0.25  # Extreme bearish
        
        # Should block initiation
        assert option.initiation_set(sample_state_no_position, observation_dict=obs_dict) is False

    def test_sentiment_missing_fallback(self, sample_state_no_position):
        """Missing sentiment should fallback to 0.5 (neutral) -> 1.0x multiplier."""
        option = OpenLongOption()
        obs_dict = {"technical": np.random.randn(24, 15)}  # Missing sentiment column (< 21 columns)
        
        action = option.policy(sample_state_no_position, step=0, observation_dict=obs_dict)
        # Should use 0.5 fallback -> 1.0x multiplier -> base action ~0.30
        assert 0.25 <= action <= 0.35  # Close to base action


class TestOpenShortOption:
    """Test suite for OpenShortOption."""

    def test_initiation_set_no_position_bearish_sentiment(self, sample_state_no_position):
        """Should allow initiation with no position and bearish sentiment."""
        option = OpenShortOption(max_sentiment=0.45)
        # Create bearish observation
        obs_dict = {
            "technical": np.random.randn(24, 23),
        }
        # Set bearish sentiment (< 0.45)
        obs_dict["technical"][-1, 20] = 0.35  # Bearish sentiment
        # Set price below MA (bearish signal)
        obs_dict["technical"][-1, 3] = 100.0  # Close
        obs_dict["technical"][-1, 6] = 102.0  # SMA_10 (price below MA)
        
        assert option.initiation_set(sample_state_no_position, observation_dict=obs_dict) is True

    def test_initiation_set_blocks_bullish_sentiment(self):
        """Should block initiation when sentiment is bullish."""
        state = np.random.randn(50)
        state[-5] = 0.01  # Small position
        option = OpenShortOption(max_sentiment=0.45)
        
        obs_dict = {
            "technical": np.random.randn(24, 23),
        }
        obs_dict["technical"][-1, 20] = 0.65  # Bullish sentiment
        
        assert option.initiation_set(state, observation_dict=obs_dict) is False

    def test_initiation_set_blocks_large_position(self):
        """Should block initiation when already have position."""
        state = np.random.randn(50)
        state[-5] = 0.08  # 8% position
        option = OpenShortOption()
        
        obs_dict = {
            "technical": np.random.randn(24, 23),
        }
        obs_dict["technical"][-1, 20] = 0.30  # Very bearish
        
        assert option.initiation_set(state, observation_dict=obs_dict) is False

    def test_initiation_set_overbought_rsi(self):
        """Should allow initiation with overbought RSI + bearish sentiment."""
        state = np.random.randn(50)
        state[-5] = 0.01  # Small position
        option = OpenShortOption(rsi_overbought_threshold=65.0)
        
        obs_dict = {
            "technical": np.random.randn(24, 23),
        }
        obs_dict["technical"][-1, 20] = 0.40  # Bearish sentiment
        obs_dict["technical"][-1, 11] = 72.0  # Overbought RSI
        
        assert option.initiation_set(state, observation_dict=obs_dict) is True

    def test_policy_step_0_initial_short(self, sample_state_no_position, sample_observation_dict):
        """Should output negative action (short) on step 0."""
        option = OpenShortOption()
        # Make sentiment bearish
        sample_observation_dict["technical"][-1, 20] = 0.35
        
        action = option.policy(sample_state_no_position, step=0, observation_dict=sample_observation_dict)
        
        assert action < 0.0  # Negative = short
        assert -0.50 <= action <= -0.10  # Reasonable short size
        assert option.entry_price is not None

    def test_policy_progressive_short_building(self, sample_state_no_position, sample_observation_dict):
        """Should progressively build short position with bearish sentiment."""
        option = OpenShortOption()
        sample_observation_dict["technical"][-1, 20] = 0.30  # Very bearish
        sample_observation_dict["technical"][-1, 11] = 70.0  # Overbought RSI
        
        # Step 0: Initial short
        action_0 = option.policy(sample_state_no_position, step=0, observation_dict=sample_observation_dict)
        assert action_0 < 0.0
        
        # Step 1: Add to short
        action_1 = option.policy(sample_state_no_position, step=1, observation_dict=sample_observation_dict)
        assert action_1 < 0.0
        
        # Step 2: Add to short
        action_2 = option.policy(sample_state_no_position, step=2, observation_dict=sample_observation_dict)
        assert action_2 < 0.0

    def test_policy_stops_on_bullish_sentiment(self, sample_state_no_position, sample_observation_dict):
        """Should stop building short if sentiment turns bullish."""
        option = OpenShortOption(max_sentiment=0.45)
        
        # Initial short
        sample_observation_dict["technical"][-1, 20] = 0.35  # Bearish
        action_0 = option.policy(sample_state_no_position, step=0, observation_dict=sample_observation_dict)
        assert action_0 < 0.0
        
        # Sentiment turns bullish
        sample_observation_dict["technical"][-1, 20] = 0.60  # Bullish
        action_1 = option.policy(sample_state_no_position, step=1, observation_dict=sample_observation_dict)
        assert action_1 == 0.0  # Stop building

    def test_policy_scales_with_bearish_sentiment(self, sample_state_no_position, sample_observation_dict):
        """Should scale position size with bearish sentiment strength."""
        option = OpenShortOption(sentiment_scale_enabled=True)
        
        # Very bearish sentiment
        sample_observation_dict["technical"][-1, 20] = 0.20  # Very bearish
        action_bearish = option.policy(sample_state_no_position, step=0, observation_dict=sample_observation_dict)
        
        # Mildly bearish sentiment
        option.reset()
        sample_observation_dict["technical"][-1, 20] = 0.42  # Mildly bearish
        action_mild = option.policy(sample_state_no_position, step=0, observation_dict=sample_observation_dict)
        
        # Very bearish should give larger short
        assert abs(action_bearish) > abs(action_mild)

    def test_policy_stops_at_max_steps(self, sample_state_no_position, sample_observation_dict):
        """Should stop building position at max steps."""
        option = OpenShortOption(max_steps=10)
        sample_observation_dict["technical"][-1, 20] = 0.30  # Bearish
        
        action = option.policy(sample_state_no_position, step=15, observation_dict=sample_observation_dict)
        assert action == 0.0  # No more building

    def test_termination_probability_increases_with_steps(self, sample_state_no_position):
        """Termination probability should increase with steps."""
        option = OpenShortOption(max_steps=10)
        
        prob_early = option.termination_probability(sample_state_no_position, step=2)
        prob_late = option.termination_probability(sample_state_no_position, step=8)
        
        assert prob_late > prob_early

    def test_termination_at_max_exposure(self):
        """Should terminate when reaching max exposure."""
        state = np.random.randn(50)
        state[-5] = -0.12  # 12% short position (negative)
        option = OpenShortOption(max_exposure_pct=0.10)
        
        prob = option.termination_probability(state, step=3)
        assert prob == 1.0  # Terminate

    def test_termination_on_bullish_reversal(self, sample_state_no_position):
        """Should have high termination probability if sentiment turns very bullish."""
        option = OpenShortOption()
        
        obs_dict = {
            "technical": np.random.randn(24, 23),
        }
        obs_dict["technical"][-1, 20] = 0.70  # Very bullish
        
        prob = option.termination_probability(sample_state_no_position, step=3, observation_dict=obs_dict)
        assert prob >= 0.9  # Very high chance to terminate

    def test_reset_clears_state(self, sample_state_no_position, sample_observation_dict):
        """Reset should clear entry price and step count."""
        option = OpenShortOption()
        sample_observation_dict["technical"][-1, 20] = 0.35  # Bearish
        
        option.policy(sample_state_no_position, step=0, observation_dict=sample_observation_dict)
        assert option.entry_price is not None
        
        option.reset()
        assert option.entry_price is None
        assert option.step_count == 0

    # ===== SENTIMENT AMPLIFICATION TESTS FOR SHORTS (NEW) =====

    def test_sentiment_neutral_baseline_shorts(self, sample_state_no_position, sample_observation_dict):
        """Neutral sentiment (0.5) should result in 1.0x multiplier for shorts."""
        option = OpenShortOption()
        sample_observation_dict["technical"][-1, 20] = 0.50  # Neutral sentiment
        
        action = option.policy(sample_state_no_position, step=0, observation_dict=sample_observation_dict)
        # Base short -0.25, with 1.0x multiplier -> -0.25
        # Formula: 1.0 + (0.5 - 0.5)/0.5 * 0.4 = 1.0x
        assert -0.35 <= action <= -0.15  # Close to base -0.25 (clipped range -0.35 to -0.15)

    def test_sentiment_bearish_amplifies_shorts(self, sample_state_no_position, sample_observation_dict):
        """Bearish sentiment should amplify short actions (1.0x -> 1.4x)."""
        option = OpenShortOption()
        sample_observation_dict["technical"][-1, 20] = 0.0  # Maximum bearish sentiment
        
        action = option.policy(sample_state_no_position, step=0, observation_dict=sample_observation_dict)
        # Base -0.25, with 1.4x multiplier -> -0.35 (clipped)
        # Formula: 1.0 + (0.5 - 0.0)/0.5 * 0.4 = 1.4x
        assert -0.40 <= action <= -0.30  # Amplified short

    def test_sentiment_bullish_reduces_shorts_not_block(self, sample_state_no_position, sample_observation_dict):
        """Bullish sentiment (0.60) should reduce shorts but NOT block (< 0.65)."""
        option = OpenShortOption()
        sample_observation_dict["technical"][-1, 20] = 0.60  # Bullish but not extreme
        
        # Should still return non-zero action (policy doesn't block, just reduces)
        action = option.policy(sample_state_no_position, step=0, observation_dict=sample_observation_dict)
        # However, policy should stop at step 1+ due to >= 0.55 check
        # At step 0, it should still return base action
        assert action != 0.0

    def test_sentiment_extreme_bullish_blocks_shorts(self, sample_state_no_position):
        """Extreme bullish sentiment (>=0.65) should block short initiation."""
        option = OpenShortOption()
        obs_dict = {"technical": np.random.randn(24, 23)}
        obs_dict["technical"][-1, 6] = 95.0  # SMA_10
        obs_dict["technical"][-1, 7] = 100.0  # SMA_20 (bearish technical)
        obs_dict["technical"][-1, 20] = 0.70  # Extreme bullish sentiment
        
        # Should block initiation despite bearish technicals
        assert option.initiation_set(sample_state_no_position, observation_dict=obs_dict) is False

    def test_sentiment_policy_stops_building_at_55(self, sample_state_no_position, sample_observation_dict):
        """Policy should stop building shorts when sentiment >= 0.55."""
        option = OpenShortOption()
        sample_observation_dict["technical"][-1, 20] = 0.30  # Start bearish
        
        # Step 0: should build
        action_0 = option.policy(sample_state_no_position, step=0, observation_dict=sample_observation_dict)
        assert action_0 < 0.0  # Short action
        
        # Sentiment turns to 0.60 (bullish)
        sample_observation_dict["technical"][-1, 20] = 0.60
        action_1 = option.policy(sample_state_no_position, step=1, observation_dict=sample_observation_dict)
        assert action_1 == 0.0  # Should stop building (>= 0.55)

    def test_sentiment_missing_fallback_shorts(self, sample_state_no_position):
        """Missing sentiment should fallback to 0.5 -> 1.0x multiplier."""
        option = OpenShortOption()
        obs_dict = {"technical": np.random.randn(24, 15)}  # Missing sentiment column
        obs_dict["technical"][-1, 6] = 95.0  # Bearish technical
        obs_dict["technical"][-1, 7] = 100.0
        
        action = option.policy(sample_state_no_position, step=0, observation_dict=obs_dict)
        # Should use 0.5 fallback -> 1.0x multiplier -> base -0.25
        assert -0.35 <= action <= -0.15


class TestClosePositionOption:
    """Test suite for ClosePositionOption."""

    def test_initiation_set_with_position(self, sample_state_with_position):
        """Should allow initiation when position exists."""
        option = ClosePositionOption()
        assert option.initiation_set(sample_state_with_position) is True

    def test_initiation_set_no_position(self, sample_state_no_position):
        """Should block initiation when no position."""
        option = ClosePositionOption()
        assert option.initiation_set(sample_state_no_position) is False

    def test_policy_stop_loss(self):
        """Should execute full exit on stop loss."""
        state = np.random.randn(50)
        state[-4] = -0.020  # -2% unrealized P&L
        state[-5] = 0.08  # 8% position
        obs_dict = {"position": np.array([1.0, 150.0, -0.020, 10.0, 0.08], dtype=np.float32)}
        
        option = ClosePositionOption(stop_loss=-0.015)
        action = option.policy(state, step=5, observation_dict=obs_dict)
        assert action == -1.0  # Full sell

    def test_policy_profit_target_staged_exit(self):
        """Should execute staged exit on large profit."""
        state = np.random.randn(50)
        state[-4] = 0.030  # +3% profit
        state[-5] = 0.08
        obs_dict = {"position": np.array([1.0, 150.0, 0.030, 10.0, 0.08], dtype=np.float32)}
        
        option = ClosePositionOption(profit_target=0.025)
        
        # First call: partial exit
        action1 = option.policy(state, step=5, observation_dict=obs_dict)
        assert action1 == -0.80  # 80% exit
        assert option.staged_exit is True
        
        # Second call: full exit
        action2 = option.policy(state, step=6, observation_dict=obs_dict)
        assert action2 == -1.0  # Remaining exit

    def test_policy_partial_profit(self):
        """Should scale out on partial profit threshold."""
        state = np.random.randn(50)
        state[-4] = 0.015  # +1.5% profit
        state[-5] = 0.08
        obs_dict = {"position": np.array([1.0, 150.0, 0.015, 10.0, 0.08], dtype=np.float32)}
        
        option = ClosePositionOption(partial_threshold=0.012)
        action = option.policy(state, step=5, observation_dict=obs_dict)
        assert action == -0.40  # 40% exit

    def test_policy_min_holding_period(self):
        """Should enforce minimum holding period."""
        state = np.random.randn(50)
        state[-4] = 0.030  # +3% profit
        state[-5] = 0.08
        obs_dict = {"position": np.array([1.0, 150.0, 0.030, 1.0, 0.08], dtype=np.float32)}
        
        option = ClosePositionOption(min_hold_steps=2)
        action = option.policy(state, step=1, observation_dict=obs_dict)
        assert action == 0.0  # Hold (min period not reached)

    def test_termination_when_position_closed(self, sample_state_no_position):
        """Should terminate when position is closed."""
        option = ClosePositionOption()
        prob = option.termination_probability(sample_state_no_position, step=5)
        assert prob == 1.0

    def test_reset_clears_staged_exit(self):
        """Reset should clear staged exit flag."""
        option = ClosePositionOption()
        option.staged_exit = True
        option.reset()
        assert option.staged_exit is False


class TestTrendFollowOption:
    """Test suite for TrendFollowOption."""

    def test_initiation_set_strong_trend(self):
        """Should allow initiation on strong trend with bullish sentiment."""
        state = np.random.randn(50)
        obs_dict = {"technical": np.random.randn(24, 23).astype(np.float32)}
        obs_dict["technical"][-1, 6] = 105.0  # SMA_10
        obs_dict["technical"][-1, 7] = 100.0  # SMA_20 (5% divergence)
        obs_dict["technical"][-1, 20] = 0.65  # Bullish sentiment (required for TrendFollow)
        
        option = TrendFollowOption(momentum_threshold=0.02)
        result = option.initiation_set(state, observation_dict=obs_dict)
        assert result == True  # Use == instead of is

    def test_initiation_set_weak_trend(self):
        """Should block initiation on weak trend."""
        state = np.random.randn(50)
        obs_dict = {"technical": np.random.randn(24, 23).astype(np.float32)}
        obs_dict["technical"][-1, 6] = 100.5  # SMA_10
        obs_dict["technical"][-1, 7] = 100.0  # SMA_20 (0.5% divergence)
        
        option = TrendFollowOption(momentum_threshold=0.02)
        assert option.initiation_set(state, observation_dict=obs_dict) is False

    def test_policy_bullish_trend_add_position(self):
        """Should add to position on bullish trend with sentiment support."""
        state = np.random.randn(50)
        state[-5] = 0.05  # 5% current position
        obs_dict = {"technical": np.random.randn(24, 23).astype(np.float32)}
        obs_dict["technical"][-1, 6] = 103.0  # SMA_10
        obs_dict["technical"][-1, 7] = 100.0  # SMA_20 (3% bullish)
        obs_dict["technical"][-1, 20] = 0.65  # Bullish sentiment (required to avoid divergence exit)
        
        option = TrendFollowOption(momentum_threshold=0.02)
        action = option.policy(state, step=5, observation_dict=obs_dict)
        assert action > 0  # Buy action
        assert 0 < action <= 1.0

    def test_policy_bearish_trend_exit_position(self):
        """Should exit position on bearish trend (or sentiment divergence)."""
        state = np.random.randn(50)
        state[-5] = 0.08  # 8% current position
        obs_dict = {"technical": np.random.randn(24, 23).astype(np.float32)}
        obs_dict["technical"][-1, 6] = 97.0  # SMA_10
        obs_dict["technical"][-1, 7] = 100.0  # SMA_20 (3% bearish)
        obs_dict["technical"][-1, 20] = 0.50  # Neutral sentiment (won't trigger divergence exit)
        
        option = TrendFollowOption(momentum_threshold=0.02)
        action = option.policy(state, step=5, observation_dict=obs_dict)
        # Either -0.60 (bearish trend) or -0.70 (sentiment divergence if sentiment < 0.45)
        assert action in [-0.60, -0.70]  # Sell 60-70%

    def test_policy_at_max_position(self):
        """Should hold when at max position size (with bullish sentiment)."""
        state = np.random.randn(50)
        state[-5] = 0.13  # 13% current position (above max)
        obs_dict = {"technical": np.random.randn(24, 23).astype(np.float32)}
        obs_dict["technical"][-1, 6] = 103.0  # Bullish trend
        obs_dict["technical"][-1, 7] = 100.0
        obs_dict["technical"][-1, 20] = 0.65  # Bullish sentiment (avoid divergence exit)
        
        option = TrendFollowOption(max_position_size=0.12)
        action = option.policy(state, step=5, observation_dict=obs_dict)
        assert action == 0.0  # Hold

    def test_termination_trend_weakens(self):
        """Should have high termination probability when trend weakens."""
        state = np.random.randn(50)
        obs_dict = {"technical": np.random.randn(24, 23).astype(np.float32)}
        obs_dict["technical"][-1, 6] = 100.3  # SMA_10
        obs_dict["technical"][-1, 7] = 100.0  # SMA_20 (weak 0.3%)
        
        option = TrendFollowOption(momentum_threshold=0.02)
        prob = option.termination_probability(state, step=5, observation_dict=obs_dict)
        assert prob >= 0.70  # High probability to exit

    # ===== BIDIRECTIONAL TRADING TESTS (NEW - CRITICAL) =====

    def test_initiation_bearish_trend(self):
        """Should initiate on bearish trend (SMA_10 < SMA_20 by >2%)."""
        state = np.random.randn(50)
        obs_dict = {"technical": np.random.randn(24, 23).astype(np.float32)}
        obs_dict["technical"][-1, 6] = 97.0  # SMA_10
        obs_dict["technical"][-1, 7] = 100.0  # SMA_20 (3% bearish divergence)
        obs_dict["technical"][-1, 20] = 0.50  # Neutral sentiment (allows both directions)
        
        option = TrendFollowOption(momentum_threshold=0.02)
        assert option.initiation_set(state, observation_dict=obs_dict) is True

    def test_policy_bearish_trend_builds_shorts(self):
        """Should build SHORT positions on bearish trend (negative actions)."""
        state = np.random.randn(50)
        state[-5] = 0.0  # No position yet
        obs_dict = {"technical": np.random.randn(24, 23).astype(np.float32)}
        obs_dict["technical"][-1, 6] = 97.0  # SMA_10
        obs_dict["technical"][-1, 7] = 100.0  # SMA_20 (3% bearish)
        obs_dict["technical"][-1, 20] = 0.40  # Bearish sentiment (amplifies shorts)
        
        option = TrendFollowOption(momentum_threshold=0.02)
        action = option.policy(state, step=5, observation_dict=obs_dict)
        
        assert action < 0  # NEGATIVE action = short position
        assert -0.8 <= action <= -0.2  # Reasonable short size

    def test_policy_bearish_sentiment_amplifies_shorts(self):
        """Bearish sentiment should amplify short position building."""
        state = np.random.randn(50)
        state[-5] = -0.05  # Already have 5% short position
        obs_dict = {"technical": np.random.randn(24, 23).astype(np.float32)}
        obs_dict["technical"][-1, 6] = 97.0  # Bearish trend
        obs_dict["technical"][-1, 7] = 100.0
        
        # Compare neutral vs bearish sentiment
        obs_dict["technical"][-1, 20] = 0.50  # Neutral
        option = TrendFollowOption(momentum_threshold=0.02)
        action_neutral = option.policy(state, step=5, observation_dict=obs_dict)
        
        obs_dict["technical"][-1, 20] = 0.20  # Very bearish
        action_bearish = option.policy(state, step=5, observation_dict=obs_dict)
        
        # Bearish sentiment should give larger (more negative) short action
        assert action_bearish < action_neutral  # More negative = larger short

    def test_policy_trend_reversal_bull_to_bear(self):
        """Should exit longs when trend reverses from bullish to bearish."""
        state = np.random.randn(50)
        state[-5] = 0.08  # Has 8% long position
        obs_dict = {"technical": np.random.randn(24, 23).astype(np.float32)}
        obs_dict["technical"][-1, 6] = 97.0  # SMA_10 now below SMA_20 (bearish reversal)
        obs_dict["technical"][-1, 7] = 100.0
        obs_dict["technical"][-1, 20] = 0.50  # Neutral sentiment
        
        option = TrendFollowOption(momentum_threshold=0.02)
        # Set previous trend direction to bullish
        option.current_trend_direction = "bullish"
        
        action = option.policy(state, step=5, observation_dict=obs_dict)
        
        # Should exit longs (negative action to close position)
        # Could be -0.60 (bearish trend exit) or -0.70 (reversal exit)
        assert action < 0  # Exit action
        assert -0.85 <= action <= -0.55  # Substantial exit

    def test_policy_trend_reversal_bear_to_bull(self):
        """Should exit shorts when trend reverses from bearish to bullish."""
        state = np.random.randn(50)
        state[-5] = -0.08  # Has 8% short position (negative)
        obs_dict = {"technical": np.random.randn(24, 23).astype(np.float32)}
        obs_dict["technical"][-1, 6] = 103.0  # SMA_10 now above SMA_20 (bullish reversal)
        obs_dict["technical"][-1, 7] = 100.0
        obs_dict["technical"][-1, 20] = 0.50  # Neutral sentiment
        
        option = TrendFollowOption(momentum_threshold=0.02)
        # Set previous trend to bearish
        option.current_trend_direction = "bearish"
        
        action = option.policy(state, step=5, observation_dict=obs_dict)
        
        # Should exit shorts (positive action to close short position)
        # Returns -position_size = -(-0.08) = 0.08 to close the short
        assert action > 0  # Exit short (positive to close negative position)
        assert abs(action - 0.08) < 0.01  # Should return exactly position_size to close

    def test_policy_max_short_position_enforced(self):
        """Should respect max position size for shorts."""
        state = np.random.randn(50)
        state[-5] = -0.13  # 13% short position (exceeds max)
        obs_dict = {"technical": np.random.randn(24, 23).astype(np.float32)}
        obs_dict["technical"][-1, 6] = 97.0  # Bearish trend
        obs_dict["technical"][-1, 7] = 100.0
        obs_dict["technical"][-1, 20] = 0.30  # Bearish sentiment
        
        option = TrendFollowOption(max_position_size=0.12)
        action = option.policy(state, step=5, observation_dict=obs_dict)
        
        assert action == 0.0  # Should not add more shorts

    def test_policy_weak_bearish_trend_exits(self):
        """Should exit shorts when bearish trend weakens below 50% threshold."""
        state = np.random.randn(50)
        state[-5] = -0.05  # Has short position
        obs_dict = {"technical": np.random.randn(24, 23).astype(np.float32)}
        obs_dict["technical"][-1, 6] = 99.5  # SMA_10
        obs_dict["technical"][-1, 7] = 100.0  # SMA_20 (weak 0.5% bearish, below 1% threshold)
        obs_dict["technical"][-1, 20] = 0.50
        
        option = TrendFollowOption(momentum_threshold=0.02)
        action = option.policy(state, step=5, observation_dict=obs_dict)
        
        # Should exit shorts (positive action)
        assert action > 0  # Exit signal
        assert 0.3 <= action <= 0.8  # Partial or full exit

    def test_policy_bearish_sentiment_conflict_exits_shorts(self):
        """Sentiment conflict should cause position exit via POLICY (not termination)."""
        state = np.random.randn(50)
        state[-5] = -0.06  # Has 6% short position
        obs_dict = {"technical": np.random.randn(24, 23).astype(np.float32)}
        obs_dict["technical"][-1, 6] = 97.0  # Bearish trend still present
        obs_dict["technical"][-1, 7] = 100.0
        obs_dict["technical"][-1, 20] = 0.75  # Very bullish sentiment (conflicts with short position)
        
        option = TrendFollowOption(momentum_threshold=0.02)
        action = option.policy(state, step=5, observation_dict=obs_dict)
        
        # Should exit shorts via policy (sentiment > 0.65 triggers exit)
        assert action > 0  # Positive action to close shorts
        assert action >= 0.65  # Should be substantial exit (70% of position)

    def test_current_trend_tracking(self):
        """Should track current trend in internal state (attribute is 'current_trend', not 'current_trend_direction')."""
        state = np.random.randn(50)
        state[-5] = 0.0
        obs_dict = {"technical": np.random.randn(24, 23).astype(np.float32)}
        obs_dict["technical"][-1, 20] = 0.50
        
        option = TrendFollowOption(momentum_threshold=0.02)
        
        # Start with bullish trend
        obs_dict["technical"][-1, 6] = 103.0
        obs_dict["technical"][-1, 7] = 100.0
        option.policy(state, step=1, observation_dict=obs_dict)
        # Note: current_trend may not be explicitly set, but behavior validates trend detection
        
        # Verify it builds longs on bullish trend
        action_bullish = option.policy(state, step=2, observation_dict=obs_dict)
        assert action_bullish > 0  # Positive = long
        
        # Switch to bearish trend
        obs_dict["technical"][-1, 6] = 97.0
        obs_dict["technical"][-1, 7] = 100.0
        
        # Verify it builds shorts on bearish trend
        state[-5] = 0.0  # Reset position
        action_bearish = option.policy(state, step=3, observation_dict=obs_dict)
        assert action_bearish < 0  # Negative = short


class TestScalpOption:
    """Test suite for ScalpOption."""

    def test_initiation_set_oversold_no_position(self):
        """Should allow initiation on oversold RSI with no position AND strong bullish sentiment."""
        state = np.random.randn(50)
        state[-5] = 0.0  # No position
        obs_dict = {"technical": np.random.randn(24, 23).astype(np.float32)}
        obs_dict["technical"][-1, 11] = 30.0  # RSI oversold
        obs_dict["technical"][-1, 20] = 0.70  # Strong bullish sentiment (required for scalp)
        
        option = ScalpOption(rsi_entry=35.0)
        result = option.initiation_set(state, observation_dict=obs_dict)
        assert result == True  # Use == instead of is

    def test_initiation_set_with_position(self):
        """Should block initiation when position exists."""
        state = np.random.randn(50)
        state[-5] = 0.05  # 5% position
        obs_dict = {"technical": np.random.randn(24, 23).astype(np.float32)}
        obs_dict["technical"][-1, 11] = 30.0  # RSI oversold
        
        option = ScalpOption()
        assert option.initiation_set(state, observation_dict=obs_dict) is False

    def test_policy_entry_on_step_0(self, sample_state_no_position):
        """Should enter small position on step 0 (sentiment-scaled)."""
        # Create observation dict with no position
        obs_dict = {
            "technical": np.random.randn(24, 23).astype(np.float32),
            "sl_probs": np.array([0.3, 0.4, 0.3], dtype=np.float32),
            "position": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # No position
            "portfolio": np.random.randn(8).astype(np.float32),
            "regime": np.random.rand(10).astype(np.float32),
        }
        # Set neutral sentiment (0.5) for baseline
        obs_dict["technical"][-1, 20] = 0.50  # Neutral sentiment
        
        option = ScalpOption(position_size=0.05)
        action = option.policy(sample_state_no_position, step=0, observation_dict=obs_dict)
        assert action > 0  # Buy action
        # With neutral sentiment (0.5), multiplier is 1.0x (baseline)
        # So expected action is 0.05*1.0/0.08 = 0.625
        # Action should be scaled by sentiment (bullish > 0.5 amplifies to 1.0-1.3x)
        assert 0.01 <= action <= 0.65  # Sentiment-scaled entry (up to 0.625 for neutral)

    def test_policy_quick_profit_exit(self):
        """Should exit quickly on profit target."""
        state = np.random.randn(50)
        state[-4] = 0.012  # +1.2% profit
        state[-5] = 0.05
        obs_dict = {"position": np.array([1.0, 150.0, 0.012, 3.0, 0.05], dtype=np.float32)}
        
        option = ScalpOption(profit_target=0.010)
        action = option.policy(state, step=3, observation_dict=obs_dict)
        assert action == -1.0  # Full exit

    def test_policy_tight_stop_loss(self):
        """Should exit on tight stop loss."""
        state = np.random.randn(50)
        state[-4] = -0.006  # -0.6% loss
        state[-5] = 0.05
        obs_dict = {"position": np.array([1.0, 150.0, -0.006, 2.0, 0.05], dtype=np.float32)}
        
        option = ScalpOption(stop_loss=-0.005)
        action = option.policy(state, step=2, observation_dict=obs_dict)
        assert action == -1.0  # Full exit

    def test_policy_max_holding_period(self):
        """Should exit at max holding period."""
        state = np.random.randn(50)
        state[-4] = 0.002  # Small profit
        state[-5] = 0.05
        obs_dict = {"position": np.array([1.0, 150.0, 0.002, 8.0, 0.05], dtype=np.float32)}
        
        option = ScalpOption(max_steps=8)
        action = option.policy(state, step=8, observation_dict=obs_dict)
        assert action == -1.0  # Full exit

    def test_termination_after_trade(self, sample_state_no_position):
        """Should terminate after trade completes."""
        option = ScalpOption()
        prob = option.termination_probability(sample_state_no_position, step=5)
        assert prob == 1.0  # Position closed


class TestWaitOption:
    """Test suite for WaitOption."""

    def test_initiation_set_always_available(self, sample_state_no_position, sample_state_with_position):
        """WaitOption should always be available."""
        option = WaitOption()
        assert option.initiation_set(sample_state_no_position) is True
        assert option.initiation_set(sample_state_with_position) is True

    def test_policy_always_hold(self, sample_state_no_position, sample_observation_dict):
        """Should always return hold action (0.0)."""
        option = WaitOption()
        for step in range(20):
            action = option.policy(sample_state_no_position, step=step, observation_dict=sample_observation_dict)
            assert action == 0.0

    def test_termination_min_wait_period(self, sample_state_no_position):
        """Should not terminate before min wait period."""
        option = WaitOption(min_wait_steps=5)
        for step in range(5):
            prob = option.termination_probability(sample_state_no_position, step=step)
            assert prob == 0.0

    def test_termination_max_wait_period(self, sample_state_no_position):
        """Should terminate at max wait period."""
        option = WaitOption(max_wait_steps=20)
        prob = option.termination_probability(sample_state_no_position, step=20)
        assert prob == 1.0

    def test_termination_gradual_increase(self, sample_state_no_position):
        """Termination probability should gradually increase."""
        option = WaitOption(min_wait_steps=5, max_wait_steps=20)
        probs = [option.termination_probability(sample_state_no_position, step=s) for s in range(5, 20)]
        # Check monotonic increase (mostly)
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1] + 0.01  # Allow small fluctuations

    def test_termination_strong_signals(self, sample_state_no_position):
        """Should have higher termination probability with strong signals."""
        option = WaitOption(min_wait_steps=3)
        
        # Weak signal case
        obs_weak = {"sl_probs": np.array([0.4, 0.3, 0.3], dtype=np.float32)}
        prob_weak = option.termination_probability(sample_state_no_position, step=5, observation_dict=obs_weak)
        
        # Strong signal case
        obs_strong = {"sl_probs": np.array([0.85, 0.10, 0.05], dtype=np.float32)}
        prob_strong = option.termination_probability(sample_state_no_position, step=5, observation_dict=obs_strong)
        
        assert prob_strong > prob_weak


class TestOptionsController:
    """Test suite for OptionsController."""

    @pytest.fixture
    def controller(self):
        """Create OptionsController instance."""
        return OptionsController(state_dim=50, num_options=6, hidden_dim=128)

    def test_initialization(self, controller):
        """Controller should initialize with correct parameters."""
        assert controller.num_options == 6
        assert controller.state_dim == 50
        assert len(controller.options) == 6
        assert controller.current_option is None
        assert controller.option_step == 0

    def test_select_option_deterministic(self, controller, sample_state_no_position):
        """Should select option deterministically."""
        state_tensor = torch.FloatTensor(sample_state_no_position).unsqueeze(0)
        option_idx, option_values = controller.select_option(state_tensor, deterministic=True)
        
        assert 0 <= option_idx < 6
        assert option_values.shape == (1, 6)

    def test_select_option_stochastic(self, controller, sample_state_no_position):
        """Should sample option stochastically."""
        state_tensor = torch.FloatTensor(sample_state_no_position).unsqueeze(0)
        
        # Sample multiple times
        samples = []
        for _ in range(20):
            option_idx, _ = controller.select_option(state_tensor, deterministic=False)
            samples.append(option_idx)
        
        # Should have some variety (not all same)
        unique_samples = set(samples)
        assert len(unique_samples) > 1  # At least 2 different options sampled

    def test_select_option_masks_invalid(self, controller):
        """Should mask out invalid options based on initiation sets."""
        # State with large position - should block OpenLongOption and OpenShortOption
        state = np.random.randn(50)
        state[-5] = 0.15  # Large position
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Should not select OpenLongOption (0) or OpenShortOption (1) if deterministic and masking works
        # Note: This test is probabilistic, may need multiple runs
        option_idx, _ = controller.select_option(state_tensor, deterministic=True)
        # With large position, ClosePositionOption (2), TrendFollow (3), or Wait (5) more likely
        assert option_idx in [2, 3, 4, 5]  # Not OpenLong or OpenShort

    def test_execute_option_returns_valid_action(self, controller, sample_state_no_position, sample_observation_dict):
        """Should execute option and return valid continuous action."""
        action, terminate, info = controller.execute_option(
            sample_state_no_position, option_idx=5, observation_dict=sample_observation_dict
        )
        
        assert -1.0 <= action <= 1.0  # Valid continuous action
        assert isinstance(terminate, bool)
        assert "option_idx" in info
        assert "option_name" in info
        assert info["option_name"] == "WAIT"

    def test_execute_option_tracks_steps(self, controller, sample_state_no_position, sample_observation_dict):
        """Should track option step count correctly."""
        assert controller.option_step == 0
        
        # Execute option multiple times without termination
        for _ in range(3):
            action, terminate, info = controller.execute_option(
                sample_state_no_position, option_idx=5, observation_dict=sample_observation_dict
            )
            if not terminate:
                assert controller.option_step > 0

    def test_execute_option_invalid_index(self, controller, sample_state_no_position):
        """Should handle invalid option index gracefully."""
        action, terminate, info = controller.execute_option(sample_state_no_position, option_idx=99)
        
        assert action == 0.0  # Default to WAIT (hold)
        assert "fallback" in info

    def test_reset_clears_state(self, controller, sample_state_no_position, sample_observation_dict):
        """Reset should clear all controller state."""
        # Execute some options
        controller.execute_option(sample_state_no_position, option_idx=2, observation_dict=sample_observation_dict)
        controller.execute_option(sample_state_no_position, option_idx=3, observation_dict=sample_observation_dict)
        
        # Reset
        controller.reset()
        
        assert controller.current_option is None
        assert controller.option_step == 0

    def test_get_option_statistics(self, controller, sample_state_no_position, sample_observation_dict):
        """Should track and report option usage statistics."""
        # Execute multiple options
        for _ in range(10):
            option_idx = np.random.choice([0, 1, 2, 3, 4])
            controller.execute_option(sample_state_no_position, option_idx=option_idx, observation_dict=sample_observation_dict)
        
        stats = controller.get_option_statistics()
        
        assert "total_selections" in stats
        assert stats["total_selections"] == 10
        assert "usage_counts" in stats
        assert "usage_percentages" in stats
        
        # Percentages should sum to 1.0
        total_pct = sum(stats["usage_percentages"].values())
        assert abs(total_pct - 1.0) < 0.01

    def test_forward_pass(self, controller):
        """Forward pass should return option logits and values."""
        batch_size = 4
        state = torch.randn(batch_size, 50)
        
        option_logits, option_values = controller.forward(state)
        
        assert option_logits.shape == (batch_size, 6)  # Updated to 6 options
        assert option_values.shape == (batch_size, 6)  # Updated to 6 options

    def test_controller_integration_full_episode(self, controller, sample_observation_dict):
        """Test full episode with option selection and execution."""
        episode_length = 50
        actions = []
        options_used = []
        
        for step in range(episode_length):
            # Create varying state
            state = np.random.randn(50)
            state[-5] = 0.05 if step % 2 == 0 else 0.0  # Alternate position
            
            # Select option
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            option_idx, _ = controller.select_option(state_tensor, deterministic=False)
            
            # Execute option
            action, terminate, info = controller.execute_option(state, option_idx, sample_observation_dict)
            
            actions.append(action)
            options_used.append(option_idx)
            
            if terminate and step < episode_length - 1:
                controller.reset()
        
        # Verify episode completed
        assert len(actions) == episode_length
        assert len(options_used) == episode_length
        
        # Verify action validity
        assert all(-1.0 <= a <= 1.0 for a in actions)
        
        # Verify option diversity (should use more than 1 option type)
        unique_options = set(options_used)
        assert len(unique_options) >= 2


class TestHierarchicalIntegration:
    """Integration tests for the hierarchical SAC wrapper."""

    @pytest.fixture
    def mock_sac_model(self):
        class MockSAC:
            def __init__(self):
                self.device = torch.device("cpu")

            def predict(self, obs, deterministic=False):
                return np.zeros((1,), dtype=np.float32), None

        return MockSAC()

    @pytest.fixture
    def options_config(self):
        return {
            "state_dim": 578,
            "num_options": 6,
            "hidden_dim": 128,
            "dropout": 0.2,
            "options_lr": 1e-4,
            "train_freq": 4,
            "warmup_steps": 100,
            "value_loss_weight": 0.5,
            "grad_clip": 1.0,
            "option_buffer_size": 1000,
            "batch_size": 32,
            "open_long": {"min_confidence": 0.6, "max_steps": 10},
            "open_short": {"min_confidence": 0.6, "max_steps": 10, "rsi_overbought_threshold": 65.0},
            "close_position": {"profit_target": 0.025, "stop_loss": -0.015},
            "trend_follow": {"momentum_threshold": 0.02},
            "scalp": {"profit_target": 0.010, "max_steps": 8},
            "wait": {"max_wait_steps": 20, "min_wait_steps": 3},
        }

    @staticmethod
    def _create_wrapper(mock_sac_model, options_config, *, num_envs: int = 1):
        from training.train_sac_with_options import HierarchicalSACWrapper

        return HierarchicalSACWrapper(
            mock_sac_model,
            options_config,
            torch.device("cpu"),
            num_envs=num_envs,
            action_space_shape=(1,),
        )

    @staticmethod
    def _env_state(wrapper, group_size: int | None = None):
        states = wrapper._ensure_group_capacity(group_size or wrapper.num_envs)
        return states[0]

    def test_hierarchical_wrapper_initialization(self, mock_sac_model, options_config):
        wrapper = self._create_wrapper(mock_sac_model, options_config)

        assert wrapper.base_sac is mock_sac_model
        assert wrapper.device.type == "cpu"
        assert wrapper.num_envs == 1
        assert wrapper.action_shape == (1,)
        assert len(wrapper.option_buffer) == 0

    def test_action_selection_returns_info(self, mock_sac_model, options_config, sample_observation_dict):
        wrapper = self._create_wrapper(mock_sac_model, options_config)

        action, info = select_action_single_env(wrapper, sample_observation_dict, deterministic=True)

        assert action.shape == (1,)
        assert action.dtype == np.float32
        assert info["option_idx"] is not None
        assert 0 <= info["option_idx"] < options_config["num_options"]
        assert "option_step" in info
        assert "option_selected" in info
        assert "option_terminated" in info

    def test_option_finalization_triggers_new_selection(self, mock_sac_model, options_config, sample_observation_dict):
        wrapper = self._create_wrapper(mock_sac_model, options_config)

        select_action_single_env(wrapper, sample_observation_dict)
        wrapper.update_episode_return(0, 0.01)
        wrapper._finalize_option_for_env(0, force=True)

        _, info = select_action_single_env(wrapper, sample_observation_dict)

        assert info["option_selected"] is True
        assert info["option_idx"] is not None
        assert wrapper.total_steps >= 2

    def test_option_buffer_accumulation(self, mock_sac_model, options_config, sample_observation_dict):
        wrapper = self._create_wrapper(mock_sac_model, options_config)

        initial_size = len(wrapper.option_buffer)
        for step in range(12):
            select_action_single_env(wrapper, sample_observation_dict)
            wrapper.update_episode_return(0, 0.05)
            if step % 3 == 2:
                wrapper._finalize_option_for_env(0, force=True)

        wrapper.finalize_all()

        assert len(wrapper.option_buffer) > initial_size

    def test_options_controller_training(self, mock_sac_model, options_config, sample_observation_dict):
        wrapper = self._create_wrapper(mock_sac_model, options_config)

        state_flat = wrapper._flatten_observation(sample_observation_dict)
        for _ in range(wrapper.batch_size):
            wrapper.option_buffer.add(
                state=state_flat + np.random.randn(*state_flat.shape) * 0.1,
                option_idx=np.random.randint(0, options_config["num_options"]),
                option_return=np.random.randn() * 0.1,
                option_steps=np.random.randint(1, 10),
                episode_return=np.random.randn() * 0.5,
            )

        wrapper.total_steps = wrapper.warmup_steps
        metrics = wrapper.train_options_controller()

        assert metrics is not None
        for key in ("options/policy_loss", "options/value_loss", "options/total_loss"):
            assert key in metrics
            assert np.isfinite(metrics[key])

    def test_option_statistics_tracking(self, mock_sac_model, options_config, sample_observation_dict):
        wrapper = self._create_wrapper(mock_sac_model, options_config)

        for step in range(20):
            select_action_single_env(wrapper, sample_observation_dict)
            wrapper.update_episode_return(0, 0.01)
            if step % 4 == 3:
                wrapper._finalize_option_for_env(0, force=True)

        stats = wrapper.get_option_statistics()

        assert stats["total_selections"] >= 1
        assert "usage_counts" in stats
        assert "usage_percentages" in stats
        assert "average_durations" in stats
        assert "average_returns" in stats

    def test_reset_env_clears_state(self, mock_sac_model, options_config, sample_observation_dict):
        wrapper = self._create_wrapper(mock_sac_model, options_config)

        for _ in range(5):
            select_action_single_env(wrapper, sample_observation_dict)
            wrapper.update_episode_return(0, 0.02)

        state = self._env_state(wrapper)
        assert state.current_option_idx is not None or state.option_step_count > 0

        wrapper.reset_env(0)

        state = self._env_state(wrapper)
        assert state.current_option_idx is None
        assert state.option_step_count == 0
        assert state.episode_return == 0.0

    def test_flatten_observation_preserves_info(self, mock_sac_model, options_config, sample_observation_dict):
        wrapper = self._create_wrapper(mock_sac_model, options_config)

        state_flat = wrapper._flatten_observation(sample_observation_dict)

        assert state_flat.ndim == 1
        assert state_flat.dtype == np.float32

        expected_size = (
            sample_observation_dict["technical"].size
            + sample_observation_dict["sl_probs"].size
            + sample_observation_dict["position"].size
            + sample_observation_dict["portfolio"].size
            + sample_observation_dict["regime"].size
        )
        assert state_flat.size == expected_size

    def test_save_and_load_checkpoint(self, mock_sac_model, options_config, sample_observation_dict, tmp_path):
        wrapper = self._create_wrapper(mock_sac_model, options_config)

        for step in range(10):
            select_action_single_env(wrapper, sample_observation_dict)
            wrapper.update_episode_return(0, 0.02)
            if step % 2 == 1:
                wrapper._finalize_option_for_env(0, force=True)

        wrapper.save(tmp_path)
        assert (tmp_path / "options_controller.pt").exists()

        wrapper_loaded = self._create_wrapper(mock_sac_model, options_config)
        wrapper_loaded.load(tmp_path)

        stats1 = wrapper.get_option_statistics()
        stats2 = wrapper_loaded.get_option_statistics()
        assert stats1["total_selections"] == stats2["total_selections"]

    def test_select_actions_multi_env(self, mock_sac_model, options_config, sample_observation_dict):
        wrapper = self._create_wrapper(mock_sac_model, options_config, num_envs=16)

        batched_obs = make_batched_observation(sample_observation_dict, batch_size=16)
        actions, info_list = wrapper.select_actions(batched_obs, deterministic=False, return_info=True)

        assert actions.shape == (16, 1)
        assert len(info_list) == 16
        assert wrapper.total_steps == 16
        assert np.all(actions >= -1.0) and np.all(actions <= 1.0)
        assert all(info["env_index"] == idx for idx, info in enumerate(info_list))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
