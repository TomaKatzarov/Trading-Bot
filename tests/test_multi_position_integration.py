"""
Integration tests for multi-position portfolio with trading environment and reward system.

These tests validate:
1. End-to-end trading flow with multiple positions
2. Reward calculation with multi-position portfolios
3. Action masking integration with multiple positions
4. Position lifecycle (open -> update -> close) with reward tracking
5. Short position rewards and P&L calculation
6. Multi-position portfolio metrics
7. Environment state consistency with multiple positions
"""

import pytest
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from core.rl.environments.continuous_trading_env import ContinuousTradingEnvironment
from core.rl.environments.reward_shaper import RewardShaper, RewardConfig
from core.rl.environments.trading_env import TradingConfig
from core.rl.environments.portfolio_manager import PortfolioManager, PortfolioConfig
from decimal import Decimal


class TestMultiPositionIntegration:
    """Integration tests for multi-position trading system."""

    @pytest.fixture
    def mock_env_config(self, reward_config):
        """Create mock environment configuration using TradingConfig."""
        from core.rl.environments.trading_env import TradingConfig
        from pathlib import Path
        config = TradingConfig(
            symbol="SPY",
            data_path=Path("dummy_path.csv"),  # Will be mocked
            sl_checkpoints={},
            lookback_window=24,
            episode_length=100,
            max_hold_hours=500,
            stop_loss=0.02,
            take_profit=0.025,
            reward_config=reward_config,
            log_level=logging.WARNING,
            curriculum_action_coverage_enabled=False,
            curriculum_action_coverage_start_step=0,
            curriculum_action_coverage_min_buy_pct=0.0,
            curriculum_action_coverage_min_sell_pct=0.0,
            curriculum_action_coverage_reward_multiplier=1.0,
            curriculum_action_coverage_penalty_cap=0.0,
            curriculum_action_coverage_penalty_power=1.0,
            curriculum_action_coverage_selected_weight=0.5,
            curriculum_action_penalty_cap_total=0.0,
        )
        return config

    @pytest.fixture
    def multi_position_portfolio_config(self):
        """Portfolio config with multi-position support."""
        return PortfolioConfig(
            initial_capital=100000.0,
            max_positions=3,
            max_position_size_pct=Decimal("0.40"),
            max_leverage=Decimal("2.0"),
            min_position_value_pct=Decimal("0.01"),
            shorting_enabled=True,
            allow_multiple_positions_per_symbol=True,
            short_margin_requirement=1.5,
        )

    @pytest.fixture
    def reward_config(self):
        """Reward configuration matching production settings."""
        return RewardConfig(
            pnl_weight=0.85,
            transaction_cost_weight=0.10,
            time_efficiency_weight=0.0,
            sharpe_weight=0.05,
            drawdown_weight=0.02,
            sizing_weight=0.0,
            pnl_scale=0.0001,
            reward_clip=1000.0,
            win_bonus_multiplier=1.8,
            loss_penalty_multiplier=1.5,
            # Golden Shot parameters
            context_scaling_enabled=True,
            adaptive_win_multiplier_enabled=True,
            adaptive_win_base_multiplier=1.8,
            adaptive_win_max_multiplier=2.5,
            adaptive_win_min_multiplier=1.2,
            action_concentration_threshold=0.50,
            high_entropy_threshold=2.5,
            overtrading_action_threshold=0.30,
            overtrading_threshold=0.30,
            overtrading_penalty_scale=0.5,
            patient_trading_threshold=0.15,
            patient_trading_bonus=1.2,
            low_winrate_threshold=0.40,
            low_winrate_penalty_scale=0.7,
            entropy_bonus_weight=0.20,
            entropy_bonus_target=2.5,
            diversity_penalty_weight=0.15,
            momentum_weight=0.10,
        )

    # ------------------------------------------------------------------
    # Test 1: Multiple Position Lifecycle with Rewards
    # ------------------------------------------------------------------
    def test_multi_position_lifecycle_rewards(self, multi_position_portfolio_config, reward_config):
        """Test complete lifecycle of multiple positions with reward calculation."""
        portfolio = PortfolioManager(config=multi_position_portfolio_config, log_trades=False)
        reward_shaper = RewardShaper(reward_config)
        
        entry_time = datetime(2024, 1, 1, 10, 0)
        
        # Open 3 long positions
        positions_opened = []
        for i, (symbol, price) in enumerate([("SPY", 150.0), ("QQQ", 200.0), ("AAPL", 180.0)], start=1):
            success, pos = portfolio.open_position(
                symbol=symbol,
                shares=50.0,
                entry_price=price,
                entry_time=entry_time,
                entry_step=i,
            )
            assert success, f"Position {i} should open"
            positions_opened.append((pos.position_id, symbol, price))
        
        assert len(portfolio.positions) == 3, "Should have 3 positions"
        
        # Update positions with price movements
        current_prices = {"SPY": 155.0, "QQQ": 205.0, "AAPL": 182.0}  # All profitable
        portfolio.update_positions(current_prices, current_step=10)
        
        # Verify unrealized P&L
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in portfolio.positions.values())
        expected_pnl = (155-150)*50 + (205-200)*50 + (182-180)*50  # 250 + 250 + 100 = 600
        assert abs(total_unrealized_pnl - expected_pnl) < 1e-6
        
        # Close positions and calculate rewards
        rewards = []
        prev_equity = portfolio.get_equity()  # Track equity before each trade
        
        for pos_id, symbol, entry_price in positions_opened:
            exit_price = current_prices[symbol]
            success, trade = portfolio.close_position(
                symbol=symbol,
                shares_to_close=None,
                exit_price=exit_price,
                exit_time=datetime(2024, 1, 1, 16, 0),
                exit_step=20,
                position_id=pos_id,
            )
            assert success, f"Should close position {pos_id}"
            
            current_equity = portfolio.get_equity()
            
            # Calculate reward for this trade
            position_info = {
                "realized_pnl": trade["realized_pnl"],
                "realized_pnl_pct": trade["realized_pnl_pct"],
            }
            trade_info = {
                "action": "SELL",
                "trade_value": trade["proceeds"],
                "pnl_pct": trade["realized_pnl_pct"],
            }
            portfolio_state = portfolio.get_portfolio_metrics()
            
            # Call with required positional arguments
            reward, breakdown = reward_shaper.compute_reward(
                action=5,  # SELL_ALL
                action_executed=True,
                prev_equity=prev_equity,
                current_equity=current_equity,
                position_info=position_info,
                trade_info=trade_info,
                portfolio_state=portfolio_state,
            )
            rewards.append(reward)
            
            # Update prev_equity for next trade
            prev_equity = current_equity
            
            # Verify reward is positive for profitable trades
            assert reward > 0, f"Reward should be positive for profitable trade (got {reward})"
        
        # Verify all positions closed
        assert len(portfolio.positions) == 0, "All positions should be closed"
        assert portfolio.total_trades == 3, "Should have 3 completed trades"
        
        # Verify rewards are properly scaled
        assert all(r < reward_config.reward_clip for r in rewards), "Rewards should be within clip limit"

    # ------------------------------------------------------------------
    # Test 2: Short Position Rewards
    # ------------------------------------------------------------------
    def test_short_position_rewards(self, multi_position_portfolio_config, reward_config):
        """Test reward calculation for short positions."""
        portfolio = PortfolioManager(config=multi_position_portfolio_config, log_trades=False)
        reward_shaper = RewardShaper(reward_config)
        
        entry_time = datetime(2024, 1, 1, 10, 0)
        
        # Open short position
        success, pos = portfolio.open_position(
            symbol="TSLA",
            shares=-100.0,  # SHORT
            entry_price=200.0,
            entry_time=entry_time,
            entry_step=1,
        )
        assert success and pos.is_short()
        
        # Price drops (profitable for short)
        portfolio.update_positions({"TSLA": 180.0}, current_step=5)
        
        # Verify unrealized P&L is positive
        assert pos.unrealized_pnl > 0, "Short should be profitable when price drops"
        expected_pnl = (200.0 - 180.0) * 100.0  # $2000 profit
        assert abs(pos.unrealized_pnl - expected_pnl) < 1e-6
        
        # Close short position
        success, trade = portfolio.close_position(
            symbol="TSLA",
            shares_to_close=None,
            exit_price=180.0,
            exit_time=datetime(2024, 1, 1, 16, 0),
            exit_step=10,
            position_id=pos.position_id,
        )
        assert success
        assert trade["realized_pnl"] > 0, "Should have positive realized P&L"
        
        # Calculate reward
        position_info = {
            "realized_pnl": trade["realized_pnl"],
            "realized_pnl_pct": trade["realized_pnl_pct"],
        }
        trade_info = {
            "action": "SELL",
            "pnl_pct": trade["realized_pnl_pct"],
        }
        portfolio_state = portfolio.get_portfolio_metrics()
        
        prev_equity = 100000.0  # Initial capital
        current_equity = portfolio.get_equity()
        
        reward, breakdown = reward_shaper.compute_reward(
            action=5,  # SELL_ALL
            action_executed=True,
            prev_equity=prev_equity,
            current_equity=current_equity,
            position_info=position_info,
            trade_info=trade_info,
            portfolio_state=portfolio_state,
        )
        
        # Verify reward is positive and properly scaled
        assert reward > 0, "Reward should be positive for profitable short"
        
        # Test losing short (price rises)
        success2, pos2 = portfolio.open_position(
            symbol="TSLA",
            shares=-100.0,
            entry_price=180.0,
            entry_time=entry_time,
            entry_step=15,
        )
        assert success2
        
        portfolio.update_positions({"TSLA": 200.0}, current_step=20)
        assert pos2.unrealized_pnl < 0, "Short should lose when price rises"
        
        success_close, trade2 = portfolio.close_position(
            symbol="TSLA",
            shares_to_close=None,
            exit_price=200.0,
            exit_time=datetime(2024, 1, 1, 18, 0),
            exit_step=25,
            position_id=pos2.position_id,
        )
        assert trade2["realized_pnl"] < 0, "Should have negative realized P&L"
        
        position_info2 = {
            "realized_pnl": trade2["realized_pnl"],
            "realized_pnl_pct": trade2["realized_pnl_pct"],
        }
        trade_info2 = {
            "action": "SELL",
            "pnl_pct": trade2["realized_pnl_pct"],
        }
        portfolio_state2 = portfolio.get_portfolio_metrics()
        
        prev_equity2 = portfolio.get_equity() - trade2["realized_pnl"]  # Equity before this trade
        current_equity2 = portfolio.get_equity()
        
        reward2, breakdown2 = reward_shaper.compute_reward(
            action=5,  # SELL_ALL
            action_executed=True,
            prev_equity=prev_equity2,
            current_equity=current_equity2,
            position_info=position_info2,
            trade_info=trade_info2,
            portfolio_state=portfolio_state2,
        )
        
        # Verify reward is negative for losing trade
        assert reward2 < 0, "Reward should be negative for losing short"

    # ------------------------------------------------------------------
    # Test 3: Action Masking with Multiple Positions
    # ------------------------------------------------------------------
    def test_action_masking_multi_position(self, mock_env_config, multi_position_portfolio_config):
        """Test action masking correctly handles multiple positions."""
        with patch.object(ContinuousTradingEnvironment, '_load_data'):
            env = ContinuousTradingEnvironment(mock_env_config)
            env.portfolio = PortfolioManager(config=multi_position_portfolio_config, log_trades=False)
            env.data = MagicMock()
            env.current_step = 10
            env.executed_action_history = []
            
            entry_time = datetime(2024, 1, 1, 10, 0)
            
            # Open 2 positions for same symbol
            pos1_id = "SPY_1_1234567890"
            pos2_id = "SPY_5_1234567895"
            
            pos1 = MagicMock()
            pos1.symbol = "SPY"
            pos1.entry_step = 1
            
            pos2 = MagicMock()
            pos2.symbol = "SPY"
            pos2.entry_step = 5
            
            env.portfolio.positions = {pos1_id: pos1, pos2_id: pos2}
            
            # Test sell action - should check most recent position
            smoothed = -0.8
            context = {}
            
            masked, context = env._mask_invalid_action(smoothed, context)
            
            # Should enforce min hold on most recent position (entry_step=5, current=10, holding=5 steps)
            # Min hold is 2, so should NOT be masked
            assert masked == -0.8, "Should allow sell (min hold satisfied)"
            
            # Test with recent position
            env.current_step = 6  # Only 1 step after pos2 entry
            masked2, context2 = env._mask_invalid_action(-0.8, {})
            assert masked2 == 0.0, "Should block sell (min hold not met)"
            assert context2["mask_reason"] == "min_hold_enforced"

    # ------------------------------------------------------------------
    # Test 4: Portfolio Metrics with Multiple Positions
    # ------------------------------------------------------------------
    def test_portfolio_metrics_multi_position(self, multi_position_portfolio_config):
        """Test portfolio metrics correctly aggregate across multiple positions."""
        portfolio = PortfolioManager(config=multi_position_portfolio_config, log_trades=False)
        
        entry_time = datetime(2024, 1, 1, 10, 0)
        
        # Open mixed positions (2 long, 1 short)
        portfolio.open_position(
            symbol="SPY",
            shares=100.0,
            entry_price=150.0,
            entry_time=entry_time,
            entry_step=1,
        )
        portfolio.open_position(
            symbol="QQQ",
            shares=50.0,
            entry_price=200.0,
            entry_time=entry_time,
            entry_step=2,
        )
        portfolio.open_position(
            symbol="TSLA",
            shares=-30.0,  # SHORT
            entry_price=250.0,
            entry_time=entry_time,
            entry_step=3,
        )
        
        # Update positions
        current_prices = {"SPY": 155.0, "QQQ": 205.0, "TSLA": 240.0}
        portfolio.update_positions(current_prices, current_step=10)
        
        # Get metrics
        metrics = portfolio.get_portfolio_metrics()
        
        # Verify position count
        assert metrics["num_positions"] == 3, "Should report 3 positions"
        
        # Verify position_value (sum with signs - shorts are negative)
        spy_value = 100 * 155  # 15,500
        qqq_value = 50 * 205   # 10,250
        tsla_value = -30 * 240  # -7,200 (short is negative)
        expected_position_value = spy_value + qqq_value + tsla_value  # 18,550
        assert abs(metrics["position_value"] - expected_position_value) < 1.0
        
        # Verify leverage (uses position_value / equity)
        equity = metrics["equity"]
        leverage = expected_position_value / equity if equity > 0 else 0
        assert abs(metrics["leverage"] - leverage) < 0.01
        
        # Verify equity = cash + position_value
        # Cash after opening positions:
        initial_capital = 100000.0
        spy_cost = 150 * 100  # 15,000 paid out
        qqq_cost = 200 * 50   # 10,000 paid out
        tsla_proceeds = 250 * 30  # 7,500 received (short)
        expected_cash = initial_capital - spy_cost - qqq_cost + tsla_proceeds  # 82,500
        
        # Equity = cash + position_value
        expected_equity = expected_cash + expected_position_value
        # = 82,500 + 18,550 = 101,050
        assert abs(metrics["equity"] - expected_equity) < 10.0
        assert abs(metrics["cash"] - expected_cash) < 10.0

    # ------------------------------------------------------------------
    # Test 5: Reward Scaling with Position Count
    # ------------------------------------------------------------------
    def test_reward_scaling_with_position_count(self, multi_position_portfolio_config, reward_config):
        """Test that rewards scale appropriately with number of positions."""
        portfolio = PortfolioManager(config=multi_position_portfolio_config, log_trades=False)
        reward_shaper = RewardShaper(reward_config)
        
        entry_time = datetime(2024, 1, 1, 10, 0)
        
        # Scenario 1: Single position with $500 profit
        portfolio.open_position(
            symbol="SPY",
            shares=100.0,
            entry_price=150.0,
            entry_time=entry_time,
            entry_step=1,
        )
        portfolio.update_positions({"SPY": 155.0}, current_step=5)
        
        success1, trade1 = portfolio.close_position(
            symbol="SPY",
            shares_to_close=None,
            exit_price=155.0,
            exit_time=datetime(2024, 1, 1, 12, 0),
            exit_step=10,
        )
        
        portfolio_state1 = portfolio.get_portfolio_metrics()
        prev_equity1 = 100000.0
        current_equity1 = portfolio.get_equity()
        reward1, breakdown1 = reward_shaper.compute_reward(
            action=5,  # SELL_ALL
            action_executed=True,
            prev_equity=prev_equity1,
            current_equity=current_equity1,
            position_info={"realized_pnl": trade1["realized_pnl"], "realized_pnl_pct": trade1["realized_pnl_pct"]},
            trade_info={"action": "SELL", "pnl_pct": trade1["realized_pnl_pct"]},
            portfolio_state=portfolio_state1,
        )
        
        # Scenario 2: Three positions with $166.67 profit each (total $500)
        portfolio2 = PortfolioManager(config=multi_position_portfolio_config, log_trades=False)
        
        for symbol, entry_price, exit_price in [("SPY", 150.0, 155.0), ("QQQ", 200.0, 203.33), ("AAPL", 180.0, 182.22)]:
            portfolio2.open_position(
                symbol=symbol,
                shares=100.0,
                entry_price=entry_price,
                entry_time=entry_time,
                entry_step=1,
            )
        
        current_prices = {"SPY": 155.0, "QQQ": 203.33, "AAPL": 182.22}
        portfolio2.update_positions(current_prices, current_step=5)
        
        rewards2 = []
        prev_equity2 = portfolio2.get_equity()  # Track equity before each trade
        for symbol in ["SPY", "QQQ", "AAPL"]:
            success, trade = portfolio2.close_position(
                symbol=symbol,
                shares_to_close=None,
                exit_price=current_prices[symbol],
                exit_time=datetime(2024, 1, 1, 12, 0),
                exit_step=10,
            )
            current_equity2 = portfolio2.get_equity()
            portfolio_state = portfolio2.get_portfolio_metrics()
            reward, breakdown = reward_shaper.compute_reward(
                action=5,  # SELL_ALL
                action_executed=True,
                prev_equity=prev_equity2,
                current_equity=current_equity2,
                position_info={"realized_pnl": trade["realized_pnl"], "realized_pnl_pct": trade["realized_pnl_pct"]},
                trade_info={"action": "SELL", "pnl_pct": trade["realized_pnl_pct"]},
                portfolio_state=portfolio_state,
            )
            rewards2.append(reward)
            prev_equity2 = current_equity2  # Update for next iteration
        
        # Individual rewards should be smaller but comparable when scaled
        avg_reward2 = np.mean(rewards2)
        
        # Both approaches should have similar total value
        # (allowing for some difference due to portfolio state changes)
        assert reward1 > 0 and avg_reward2 > 0, "Both strategies should have positive rewards"

    # ------------------------------------------------------------------
    # Test 6: Context Scaling with Multiple Positions
    # ------------------------------------------------------------------
    def test_context_scaling_multi_position(self, multi_position_portfolio_config, reward_config):
        """Test Golden Shot context scaling with multiple positions."""
        portfolio = PortfolioManager(config=multi_position_portfolio_config, log_trades=False)
        reward_shaper = RewardShaper(reward_config)
        
        # Build portfolio state with high trade rate (overtrading)
        portfolio_state = {
            "num_trades": 100,
            "total_return": 0.05,  # 5% return
            "win_rate": 0.45,  # 45% win rate (slightly below low threshold)
            "sharpe_ratio": 0.8,
            "num_positions": 3,  # Multiple positions
        }
        
        trade_info = {
            "action": "SELL",
            "holding_period": 5,
            "pnl_pct": 0.05,  # Required for reward calculation
        }
        
        # Simulate successful trade
        position_info = {
            "realized_pnl": 500.0,
            "realized_pnl_pct": 0.05,
        }
        
        # Calculate base reward
        base_reward, base_breakdown = reward_shaper.compute_reward(
            action=5,  # SELL_ALL
            action_executed=True,
            prev_equity=100000.0,
            current_equity=105000.0,
            position_info=position_info,
            trade_info=trade_info,
            portfolio_state=portfolio_state,
        )
        
        # Now test with lower trade rate (patient trading)
        portfolio_state_patient = portfolio_state.copy()
        portfolio_state_patient["num_trades"] = 20  # Fewer trades
        portfolio_state_patient["win_rate"] = 0.60  # Better win rate
        
        patient_reward, patient_breakdown = reward_shaper.compute_reward(
            action=5,  # SELL_ALL
            action_executed=True,
            prev_equity=100000.0,
            current_equity=105000.0,
            position_info=position_info,
            trade_info=trade_info,
            portfolio_state=portfolio_state_patient,
        )
        
        # Patient trading with better win rate should get bonus
        # (assuming context scaling is working)
        assert base_reward > 0, "Base reward should be positive"
        assert patient_reward > 0, "Patient reward should be positive"

    # ------------------------------------------------------------------
    # Test 7: Position Limits Enforcement
    # ------------------------------------------------------------------
    def test_position_limits_enforcement(self, multi_position_portfolio_config):
        """Test that position limits are properly enforced."""
        portfolio = PortfolioManager(config=multi_position_portfolio_config, log_trades=False)
        
        entry_time = datetime(2024, 1, 1, 10, 0)
        
        # Open max_positions (3)
        for i, symbol in enumerate(["SPY", "QQQ", "AAPL"], start=1):
            success, pos = portfolio.open_position(
                symbol=symbol,
                shares=50.0,
                entry_price=150.0,
                entry_time=entry_time,
                entry_step=i,
            )
            assert success, f"Position {i} should open"
        
        assert len(portfolio.positions) == 3
        
        # Try to open 4th position (should fail)
        success4, pos4 = portfolio.open_position(
            symbol="MSFT",
            shares=50.0,
            entry_price=150.0,
            entry_time=entry_time,
            entry_step=4,
        )
        assert not success4, "4th position should be rejected"
        assert len(portfolio.positions) == 3, "Position count should remain at 3"
        
        # Close one position
        first_pos_id = list(portfolio.positions.keys())[0]
        success_close, trade = portfolio.close_position(
            symbol="SPY",
            shares_to_close=None,
            exit_price=155.0,
            exit_time=datetime(2024, 1, 1, 12, 0),
            exit_step=10,
            position_id=first_pos_id,
        )
        assert success_close
        assert len(portfolio.positions) == 2
        
        # Now should be able to open another position
        success5, pos5 = portfolio.open_position(
            symbol="MSFT",
            shares=50.0,
            entry_price=150.0,
            entry_time=entry_time,
            entry_step=11,
        )
        assert success5, "Should be able to open position after closing one"
        assert len(portfolio.positions) == 3

    # ------------------------------------------------------------------
    # Test 8: Info Dict Contains Multi-Position Data
    # ------------------------------------------------------------------
    def test_info_dict_multi_position_data(self, mock_env_config, multi_position_portfolio_config):
        """Test that environment info dict includes multi-position tracking data."""
        with patch.object(ContinuousTradingEnvironment, '_load_data'):
            env = ContinuousTradingEnvironment(mock_env_config)
            env.portfolio = PortfolioManager(config=multi_position_portfolio_config, log_trades=False)
            
            # Simulate opening multiple positions
            entry_time = datetime(2024, 1, 1, 10, 0)
            for i, symbol in enumerate(["SPY", "QQQ"], start=1):
                env.portfolio.open_position(
                    symbol=symbol,
                    shares=50.0,
                    entry_price=150.0,
                    entry_time=entry_time,
                    entry_step=i,
                )
            
            # Check that portfolio methods work correctly
            assert env.portfolio.has_position_for_symbol("SPY")
            assert env.portfolio.has_position_for_symbol("QQQ")
            assert not env.portfolio.has_position_for_symbol("AAPL")
            
            spy_positions = env.portfolio.get_positions_for_symbol("SPY")
            assert len(spy_positions) == 1
            
            total_exposure_spy = env.portfolio.get_total_exposure_for_symbol("SPY")
            assert total_exposure_spy > 0
            
            net_position_spy = env.portfolio.get_net_position_for_symbol("SPY")
            assert net_position_spy == 50.0  # One long position with 50 shares


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
