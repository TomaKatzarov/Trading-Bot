"""
Golden Shot Implementation Tests (Phase A2 SAC_5)

Tests the three-layer defense system against action exploitation:
1. Adaptive action masking (environment level)
2. State-dependent reward scaling (reward shaper level)
3. Dynamic win multiplier adjustment (reward component level)

Test Coverage:
- Action masking: excessive buying, min hold period, progressive entry
- Context scaling: over-trading penalty, patient trading bonus, low win rate penalty
- Adaptive win multiplier: concentration penalty, entropy bonus, over-trading penalty
- Configuration loading: all Golden Shot parameters
- End-to-end reward computation: integration of all systems
"""

import pytest
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

from core.rl.environments.continuous_trading_env import ContinuousTradingEnvironment
from core.rl.environments.reward_shaper import RewardConfig, RewardShaper
from core.rl.environments.trading_env import TradeAction, TradingConfig
from core.rl.environments.portfolio_manager import PortfolioConfig


@pytest.fixture
def sample_config():
    """Create a sample trading config for testing."""
    portfolio_config = PortfolioConfig(
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_bps=5.0,
        min_position_value_pct=0.05,
        max_position_size_pct=0.65,
        max_position_loss_pct=0.05,
        max_portfolio_loss_pct=0.20,
    )
    
    reward_config = RewardConfig(
        # Golden Shot parameters
        adaptive_win_multiplier_enabled=True,
        adaptive_win_base_multiplier=1.8,
        adaptive_win_max_multiplier=2.5,
        adaptive_win_min_multiplier=1.2,
        action_concentration_threshold=0.50,
        high_entropy_threshold=2.5,
        overtrading_action_threshold=0.30,
        
        context_scaling_enabled=True,
        overtrading_threshold=0.30,
        overtrading_penalty_scale=0.5,
        patient_trading_threshold=0.15,
        patient_trading_bonus=1.2,
        low_winrate_threshold=0.40,
        low_winrate_penalty_scale=0.7,
        
        # Standard parameters
        pnl_weight=0.85,
        transaction_cost_weight=0.10,
        win_bonus_multiplier=1.8,
        loss_penalty_multiplier=1.5,
        entropy_bonus_weight=0.20,
        entropy_bonus_target=2.5,
        diversity_penalty_weight=0.15,
        momentum_weight=0.10,
    )
    
    config = TradingConfig(
        symbol="SPY",
        data_path=Path("data/historical/SPY/1Hour/data.parquet"),
        sl_checkpoints={},
        portfolio_config=portfolio_config,
        reward_config=reward_config,
        episode_length=1000,
        lookback_window=24,
        stop_loss=0.02,
        take_profit=0.025,
        max_hold_hours=8,
    )
    
    return config


# ============================================================================
# TEST SUITE 1: ADAPTIVE ACTION MASKING
# ============================================================================

class TestAdaptiveActionMasking:
    """Test suite for action masking in ContinuousTradingEnvironment."""
    
    def test_excessive_buying_blocked(self, sample_config):
        """Test Rule 1: Block excessive buying when >70% buy concentration."""
        # Create environment with mock data
        with patch.object(ContinuousTradingEnvironment, '_load_data'):
            env = ContinuousTradingEnvironment(sample_config)
            env.data = MagicMock()
            env.current_step = 50
            
            # Simulate 20 recent buy actions (>70% concentration)
            env.executed_action_history = [
                TradeAction.BUY_MEDIUM.value
            ] * 15 + [TradeAction.HOLD.value] * 5
            
            # Create mock position
            env.portfolio.positions = {}
            
            # Test masking with positive action (buy)
            smoothed = 0.8  # Strong buy signal
            context: Dict = {}
            
            masked, context = env._mask_invalid_action(smoothed, context)
            
            # Should be forced to HOLD
            assert masked == 0.0, "Excessive buying should be blocked"
            assert context["mask_reason"] == "excessive_buying"
            assert context["buy_concentration"] > 0.70
    
    def test_min_hold_period_enforced(self, sample_config):
        """Test Rule 2: Enforce minimum 2-hour holding period."""
        with patch.object(ContinuousTradingEnvironment, '_load_data'):
            env = ContinuousTradingEnvironment(sample_config)
            env.data = MagicMock()
            env.current_step = 5
            
            # Create mock position opened at step 4 (1 hour ago)
            mock_position = MagicMock()
            mock_position.entry_step = 4
            mock_position.symbol = "SPY"
            # Use position_id as key (new multi-position format)
            env.portfolio.positions = {"SPY_4_1234567890": mock_position}
            
            # Test masking with negative action (sell)
            smoothed = -0.8  # Strong sell signal
            context: Dict = {}
            
            masked, context = env._mask_invalid_action(smoothed, context)
            
            # Should be forced to HOLD (need 2 hours)
            assert masked == 0.0, "Sell should be blocked (min hold not met)"
            assert context["mask_reason"] == "min_hold_enforced"
            assert context["holding_steps"] == 1
            assert context["min_hold_required"] == 2
    
    def test_progressive_position_building(self, sample_config):
        """Test Rule 3: Cap entry size to 35% when opening new position."""
        with patch.object(ContinuousTradingEnvironment, '_load_data'):
            env = ContinuousTradingEnvironment(sample_config)
            env.data = MagicMock()
            env.current_step = 10
            
            # No existing position
            env.portfolio.positions = {}
            
            # Test masking with large buy action
            smoothed = 0.9  # Aggressive buy (>35% threshold)
            context: Dict = {}
            
            masked, context = env._mask_invalid_action(smoothed, context)
            
            # Should be capped to 0.35
            assert masked == 0.35, "Large entry should be capped to 35%"
            assert context["mask_reason"] == "progressive_entry"
            assert context["original_action"] == 0.9
            assert context["masked_action"] == 0.35


# ============================================================================
# TEST SUITE 2: STATE-DEPENDENT REWARD SCALING
# ============================================================================

class TestContextScaling:
    """Test suite for state-dependent reward scaling."""
    
    def test_overtrading_penalty(self):
        """Test that over-trading (>30% trade rate) reduces rewards by 50%."""
        config = RewardConfig(
            context_scaling_enabled=True,
            overtrading_threshold=0.30,
            overtrading_penalty_scale=0.5,
        )
        shaper = RewardShaper(config)
        
        base_reward = 10.0
        portfolio_state = {
            "num_trades": 500.0,  # Very high: 500 trades / 1500 steps = 33% > 30% threshold
            "sharpe_ratio": 0.5,
        }
        
        scaled = shaper._apply_context_scaling(base_reward, portfolio_state, None)
        
        # Should be reduced by 50%
        assert scaled == pytest.approx(5.0, rel=0.01), "Over-trading should reduce reward by 50%"
    
    def test_patient_trading_bonus(self):
        """Test that patient trading (<15% rate + positive Sharpe) gets 20% bonus."""
        config = RewardConfig(
            context_scaling_enabled=True,
            patient_trading_threshold=0.15,
            patient_trading_bonus=1.2,
        )
        shaper = RewardShaper(config)
        
        base_reward = 10.0
        portfolio_state = {
            "num_trades": 10.0,  # Low trade count
            "sharpe_ratio": 0.8,  # Positive Sharpe
        }
        
        # Patient profitable trading (10 trades / 100 steps = 10% rate)
        scaled = shaper._apply_context_scaling(base_reward, portfolio_state, None)
        
        # Should get 20% bonus
        assert scaled == pytest.approx(12.0, rel=0.01), "Patient trading should get 20% bonus"
    
    def test_low_winrate_penalty(self):
        """Test that low win rate (Sharpe < -0.5) reduces rewards by 30%."""
        config = RewardConfig(
            context_scaling_enabled=True,
            low_winrate_penalty_scale=0.7,
        )
        shaper = RewardShaper(config)
        
        base_reward = 10.0
        portfolio_state = {
            "num_trades": 20.0,
            "sharpe_ratio": -0.6,  # Negative Sharpe (proxy for low win rate)
        }
        
        scaled = shaper._apply_context_scaling(base_reward, portfolio_state, None)
        
        # Should be reduced by 30%
        assert scaled == pytest.approx(7.0, rel=0.01), "Low win rate should reduce reward by 30%"


# ============================================================================
# TEST SUITE 3: ADAPTIVE WIN MULTIPLIER
# ============================================================================

class TestAdaptiveWinMultiplier:
    """Test suite for dynamic win bonus adjustment."""
    
    def test_concentration_penalty(self):
        """Test that >50% action concentration reduces win bonus."""
        config = RewardConfig(
            adaptive_win_multiplier_enabled=True,
            adaptive_win_base_multiplier=1.8,
            adaptive_win_min_multiplier=1.2,
            action_concentration_threshold=0.50,
        )
        shaper = RewardShaper(config)
        
        # High concentration (bin-19 spam): 12 out of 15 actions are the same
        diversity_info = {
            "action_diversity_window": [19] * 12 + [0, 1, 2],
        }
        portfolio_state = {"num_trades": 15.0}
        
        multiplier = shaper._compute_adaptive_win_multiplier(diversity_info, portfolio_state)
        
        # Should be reduced (80% concentration, 30% over threshold)
        assert multiplier < 1.8, "High concentration should reduce win bonus"
        assert multiplier >= 1.2, "Should respect minimum bound"
    
    def test_high_entropy_bonus(self):
        """Test that high action entropy increases win bonus."""
        config = RewardConfig(
            adaptive_win_multiplier_enabled=True,
            adaptive_win_base_multiplier=1.8,
            adaptive_win_max_multiplier=2.5,
            high_entropy_threshold=1.5,  # Achievable threshold
            action_concentration_threshold=0.50,  # Default
        )
        shaper = RewardShaper(config)
        
        # Perfect uniform distribution across many actions for high entropy
        # Using 8 actions uniformly: entropy = log(8) = 2.08
        diversity_info = {
            "action_diversity_window": [0, 1, 2, 3, 4, 5, 6, 7] * 4,  # 32 actions, perfectly uniform
        }
        portfolio_state = {"num_trades": 20.0}  # Low enough to avoid over-trading penalty
        
        multiplier = shaper._compute_adaptive_win_multiplier(diversity_info, portfolio_state)
        
        # Should be increased due to high entropy (>1.5)
        assert multiplier > 1.8, f"High entropy should increase win bonus, got {multiplier}"
        assert multiplier <= 2.5, "Should respect maximum bound"
    
    def test_overtrading_action_penalty(self):
        """Test that over-trading reduces win bonus."""
        config = RewardConfig(
            adaptive_win_multiplier_enabled=True,
            adaptive_win_base_multiplier=1.8,
            overtrading_action_threshold=0.30,
            high_entropy_threshold=2.5,  # High threshold so entropy bonus doesn't interfere
        )
        shaper = RewardShaper(config)
        
        # Moderate diversity (not high enough for entropy bonus) + over-trading
        diversity_info = {
            "action_diversity_window": [0, 1, 2, 3] * 5,  # 20 actions, 4 unique (entropy ~1.39)
        }
        portfolio_state = {
            "num_trades": 500.0,  # 500 trades / 1500 steps = 33% > 30% threshold
        }
        
        multiplier = shaper._compute_adaptive_win_multiplier(diversity_info, portfolio_state)
        
        # Should be reduced due to over-trading (no entropy bonus to offset it)
        assert multiplier < 1.8, f"Over-trading should reduce win bonus, got {multiplier}"


# ============================================================================
# TEST SUITE 4: CONFIGURATION LOADING
# ============================================================================

class TestConfigurationLoading:
    """Test that Golden Shot parameters load correctly from YAML."""
    
    def test_golden_shot_params_in_config(self):
        """Test that all Golden Shot parameters are present in config file."""
        config_path = Path("training/config_templates/phase_a2_sac_sharpe.yaml")
        
        if not config_path.exists():
            pytest.skip("Config file not found")
        
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        reward_cfg = config["environment"]["reward_config"]
        
        # Check adaptive win multiplier parameters
        assert "adaptive_win_multiplier_enabled" in reward_cfg
        assert "adaptive_win_base_multiplier" in reward_cfg
        assert reward_cfg["adaptive_win_base_multiplier"] == 1.8
        assert "adaptive_win_max_multiplier" in reward_cfg
        assert "adaptive_win_min_multiplier" in reward_cfg
        assert "action_concentration_threshold" in reward_cfg
        assert "high_entropy_threshold" in reward_cfg
        assert "overtrading_action_threshold" in reward_cfg
        
        # Check context scaling parameters
        assert "context_scaling_enabled" in reward_cfg
        assert "overtrading_threshold" in reward_cfg
        assert "overtrading_penalty_scale" in reward_cfg
        assert "patient_trading_threshold" in reward_cfg
        assert "patient_trading_bonus" in reward_cfg
        assert "low_winrate_threshold" in reward_cfg
        assert "low_winrate_penalty_scale" in reward_cfg
        
        # Check updated weights - FIXED: Match actual phase_a2_sac_sharpe.yaml values
        assert reward_cfg["pnl_weight"] == 0.95  # Changed from 0.85
        assert reward_cfg["transaction_cost_weight"] == 0.001  # Changed from 0.10
        assert reward_cfg["win_bonus_multiplier"] == 2.0  # Changed from 1.8
        assert reward_cfg["loss_penalty_multiplier"] == 1.0  # Changed from 1.5
        assert reward_cfg["entropy_bonus_weight"] == 0.20  # Kept at 0.20 per actual config
        assert reward_cfg["diversity_penalty_weight"] == 0.0  # Changed from 0.15 (disabled)
        assert reward_cfg["momentum_weight"] == 0.0  # Changed from 0.10 (disabled)
        
        # Check new penalties - DISABLED in phase_a2 config for simplification
        assert "trade_frequency_penalty_weight" in reward_cfg
        assert reward_cfg["trade_frequency_penalty_weight"] == 0.0  # YAML has 0.0 (disabled)
        assert "hold_bonus_weight" in reward_cfg
        assert reward_cfg["hold_bonus_weight"] == 0.0  # YAML has 0.0 (disabled)


# ============================================================================
# TEST SUITE 5: END-TO-END INTEGRATION
# ============================================================================

class TestEndToEndIntegration:
    """Test complete reward computation with all Golden Shot systems."""
    
    def test_profitable_trade_with_diversity(self):
        """Test that profitable trade with high diversity gets maximum reward."""
        config = RewardConfig(
            # Enable all Golden Shot features
            adaptive_win_multiplier_enabled=True,
            adaptive_win_base_multiplier=1.8,
            adaptive_win_max_multiplier=2.5,
            high_entropy_threshold=2.5,
            
            context_scaling_enabled=True,
            patient_trading_threshold=0.15,
            patient_trading_bonus=1.2,
            
            pnl_weight=0.85,
            pnl_scale=0.01,
            realized_pnl_weight=1.0,
            unrealized_pnl_weight=0.0,
        )
        shaper = RewardShaper(config)
        
        # Profitable trade info
        trade_info = {
            "pnl_pct": 0.05,  # 5% profit
            "holding_hours": 4.0,
            "action": "agent_close",
            "forced_exit": False,
            "exit_type": "full",
            "entry_size": "small",
            "pyramid_count": 0,
        }
        
        # High diversity portfolio state
        portfolio_state = {
            "num_trades": 10.0,  # Low trade count (patient)
            "sharpe_ratio": 1.0,  # Positive Sharpe
            "peak_equity": 110000.0,
            "deployed_pct": 0.5,
            "equity": 105000.0,
        }
        
        # High entropy diversity info
        diversity_info = {
            "action_diversity_window": [0, 1, 2, 3, 4, 5] * 3 + [0, 1],  # High diversity
            "episode_step": 100,
        }
        
        # Compute reward
        total_reward, components = shaper.compute_reward(
            action=5,  # SELL_ALL
            action_executed=True,
            prev_equity=100000.0,
            current_equity=105000.0,
            position_info=None,
            trade_info=trade_info,
            portfolio_state=portfolio_state,
            diversity_info=diversity_info,
        )
        
        # Should get strong positive reward:
        # - 5% profit normalized to 5.0
        # - Adaptive multiplier >1.8 (high diversity)
        # - Patient trading bonus 1.2×
        # - Small position size bonus 1.2×
        # Note: With pnl_weight=0.85 and other components, final reward is scaled down
        assert total_reward > 6.0, f"Profitable diverse patient trade should get strong reward, got {total_reward}"
        assert components["pnl"] > 0, "PnL component should be positive"
    
    def test_exploitative_trade_gets_reduced_reward(self):
        """Test that profitable trade with exploitation gets reduced reward."""
        config = RewardConfig(
            adaptive_win_multiplier_enabled=True,
            adaptive_win_base_multiplier=1.8,
            adaptive_win_min_multiplier=1.2,
            action_concentration_threshold=0.50,
            
            context_scaling_enabled=True,
            overtrading_threshold=0.30,
            overtrading_penalty_scale=0.5,
            
            pnl_weight=0.85,
            pnl_scale=0.01,
            realized_pnl_weight=1.0,
            unrealized_pnl_weight=0.0,
        )
        shaper = RewardShaper(config)
        
        # Same profitable trade
        trade_info = {
            "pnl_pct": 0.05,  # 5% profit
            "holding_hours": 4.0,
            "action": "agent_close",
            "forced_exit": False,
            "exit_type": "full",
            "entry_size": "small",
            "pyramid_count": 0,
        }
        
        # Over-trading portfolio
        portfolio_state = {
            "num_trades": 50.0,  # High trade count (over-trading)
            "sharpe_ratio": 0.5,
            "peak_equity": 110000.0,
            "deployed_pct": 0.5,
            "equity": 105000.0,
        }
        
        # Low diversity (bin-19 spam)
        diversity_info = {
            "action_diversity_window": [19] * 12 + [0, 1, 2],  # 80% concentration
            "episode_step": 100,
        }
        
        # Compute reward
        total_reward, components = shaper.compute_reward(
            action=5,  # SELL_ALL
            action_executed=True,
            prev_equity=100000.0,
            current_equity=105000.0,
            position_info=None,
            trade_info=trade_info,
            portfolio_state=portfolio_state,
            diversity_info=diversity_info,
        )
        
        # Should get reduced reward due to:
        # - Adaptive multiplier <1.8 (low diversity)
        # - Over-trading penalty 0.5×
        # - High concentration penalty
        assert total_reward > 0, "Still profitable, should be positive"
        assert total_reward < 10.0, "But reduced due to exploitation"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
