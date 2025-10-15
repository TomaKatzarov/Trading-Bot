"""
Comprehensive test suite for multi-position portfolio with shorting support.

Tests:
1. Multiple long positions for same symbol
2. Short position opening and P&L calculation
3. Mixed long/short positions
4. Position limit enforcement (max_positions)
5. Capital allocation across multiple positions
6. Position-specific closing (by position_id)
7. Margin requirements for shorts
8. Helper methods (get_positions_for_symbol, etc.)
9. Edge cases (partial closes, zero positions, etc.)
"""

import pytest
from datetime import datetime
from decimal import Decimal
from core.rl.environments.portfolio_manager import (
    PortfolioConfig,
    PortfolioManager,
    Position,
)


class TestMultiPositionPortfolio:
    """Test suite for multi-position portfolio functionality."""

    @pytest.fixture
    def config_multi_position(self):
        """Portfolio config allowing multiple positions with shorting."""
        return PortfolioConfig(
            initial_capital=100000.0,
            max_positions=3,
            max_position_size_pct=Decimal("0.30"),
            max_leverage=Decimal("2.0"),
            min_position_value_pct=Decimal("0.01"),
            shorting_enabled=True,
            allow_multiple_positions_per_symbol=True,
            short_margin_requirement=1.5,
        )

    @pytest.fixture
    def config_single_position(self):
        """Portfolio config for single position (backward compatibility)."""
        return PortfolioConfig(
            initial_capital=100000.0,
            max_positions=1,
            max_position_size_pct=Decimal("0.50"),
            shorting_enabled=False,
            allow_multiple_positions_per_symbol=False,
        )

    @pytest.fixture
    def manager_multi(self, config_multi_position):
        """Portfolio manager with multi-position support."""
        return PortfolioManager(config=config_multi_position, log_trades=False)

    @pytest.fixture
    def manager_single(self, config_single_position):
        """Portfolio manager with single-position constraint."""
        return PortfolioManager(config=config_single_position, log_trades=False)

    # ------------------------------------------------------------------
    # Test 1: Multiple long positions for same symbol
    # ------------------------------------------------------------------
    def test_multiple_long_positions_same_symbol(self, manager_multi):
        """Test opening multiple long positions for the same symbol."""
        symbol = "AAPL"
        entry_time = datetime(2024, 1, 1, 10, 0)
        
        # Open first long position
        success1, pos1 = manager_multi.open_position(
            symbol=symbol,
            shares=100.0,
            entry_price=150.0,
            entry_time=entry_time,
            entry_step=1,
        )
        assert success1, "First long position should open"
        assert pos1 is not None
        assert pos1.is_long()
        
        # Open second long position for same symbol
        success2, pos2 = manager_multi.open_position(
            symbol=symbol,
            shares=50.0,
            entry_price=152.0,
            entry_time=entry_time,
            entry_step=2,
        )
        assert success2, "Second long position should open"
        assert pos2 is not None
        assert pos2.is_long()
        
        # Verify we have 2 positions
        assert len(manager_multi.positions) == 2
        positions_for_symbol = manager_multi.get_positions_for_symbol(symbol)
        assert len(positions_for_symbol) == 2
        
        # Verify each position has unique ID
        pos_ids = [pos.position_id for pos in positions_for_symbol]
        assert len(set(pos_ids)) == 2, "Position IDs should be unique"
        
        # Verify net position
        net_shares = manager_multi.get_net_position_for_symbol(symbol)
        assert net_shares == 150.0, "Net position should be 150 shares long"

    # ------------------------------------------------------------------
    # Test 2: Short position opening and P&L calculation
    # ------------------------------------------------------------------
    def test_short_position_pnl(self, manager_multi):
        """Test short position opening and correct P&L calculation."""
        symbol = "TSLA"
        entry_time = datetime(2024, 1, 1, 10, 0)
        
        # Open short position (negative shares)
        success, pos = manager_multi.open_position(
            symbol=symbol,
            shares=-100.0,  # SHORT
            entry_price=200.0,
            entry_time=entry_time,
            entry_step=1,
        )
        assert success, "Short position should open"
        assert pos is not None
        assert pos.is_short()
        assert pos.shares == -100.0
        
        # Initial cash should increase (we received proceeds from short sale)
        # Cash = 100000 + 20000 - 0 = 120000 (no commission/slippage in this test)
        initial_cash = 100000.0
        short_proceeds = 100.0 * 200.0
        expected_cash = initial_cash + short_proceeds
        assert abs(manager_multi.cash - expected_cash) < 1e-6
        
        # Update position: price falls (profit for short)
        pos.update(180.0)
        assert pos.unrealized_pnl > 0, "Short should profit when price falls"
        expected_pnl = (200.0 - 180.0) * 100.0  # $2000 profit
        assert abs(pos.unrealized_pnl - expected_pnl) < 1e-6
        
        # Update position: price rises (loss for short)
        pos.update(220.0)
        assert pos.unrealized_pnl < 0, "Short should lose when price rises"
        expected_pnl = (200.0 - 220.0) * 100.0  # $-2000 loss
        assert abs(pos.unrealized_pnl - expected_pnl) < 1e-6

    # ------------------------------------------------------------------
    # Test 3: Mixed long/short positions
    # ------------------------------------------------------------------
    def test_mixed_long_short_positions(self, manager_multi):
        """Test portfolio with both long and short positions."""
        entry_time = datetime(2024, 1, 1, 10, 0)
        
        # Open long position
        success1, pos1 = manager_multi.open_position(
            symbol="AAPL",
            shares=100.0,
            entry_price=150.0,
            entry_time=entry_time,
            entry_step=1,
        )
        assert success1 and pos1.is_long()
        
        # Open short position
        success2, pos2 = manager_multi.open_position(
            symbol="TSLA",
            shares=-50.0,
            entry_price=200.0,
            entry_time=entry_time,
            entry_step=2,
        )
        assert success2 and pos2.is_short()
        
        # Open another long position
        success3, pos3 = manager_multi.open_position(
            symbol="MSFT",
            shares=75.0,
            entry_price=300.0,
            entry_time=entry_time,
            entry_step=3,
        )
        assert success3 and pos3.is_long()
        
        # Verify we have 3 positions (max limit)
        assert len(manager_multi.positions) == 3
        
        # Verify position types
        position_types = [pos.position_type() for pos in manager_multi.positions.values()]
        assert position_types.count("LONG") == 2
        assert position_types.count("SHORT") == 1
        
        # Try to open 4th position (should fail due to max_positions=3)
        success4, pos4 = manager_multi.open_position(
            symbol="NVDA",
            shares=20.0,
            entry_price=400.0,
            entry_time=entry_time,
            entry_step=4,
        )
        assert not success4, "Should fail when max_positions reached"

    # ------------------------------------------------------------------
    # Test 4: Position limit enforcement
    # ------------------------------------------------------------------
    def test_max_positions_enforcement(self, manager_multi):
        """Test that max_positions limit is properly enforced."""
        entry_time = datetime(2024, 1, 1, 10, 0)
        
        # Open 3 positions (at limit)
        for i, symbol in enumerate(["AAPL", "TSLA", "MSFT"], start=1):
            success, pos = manager_multi.open_position(
                symbol=symbol,
                shares=10.0,
                entry_price=100.0,
                entry_time=entry_time,
                entry_step=i,
            )
            assert success, f"Position {i} should open"
        
        # Try to open 4th position
        success, pos = manager_multi.open_position(
            symbol="NVDA",
            shares=10.0,
            entry_price=100.0,
            entry_time=entry_time,
            entry_step=4,
        )
        assert not success, "Should fail when max_positions reached"
        assert len(manager_multi.positions) == 3

    # ------------------------------------------------------------------
    # Test 5: Capital allocation across multiple positions
    # ------------------------------------------------------------------
    def test_capital_allocation_multiple_positions(self, manager_multi):
        """Test proper capital allocation with multiple positions."""
        entry_time = datetime(2024, 1, 1, 10, 0)
        initial_cash = manager_multi.cash
        
        # Open 2 long positions
        manager_multi.open_position(
            symbol="AAPL",
            shares=100.0,
            entry_price=150.0,
            entry_time=entry_time,
            entry_step=1,
        )
        cash_after_1 = manager_multi.cash
        assert cash_after_1 == initial_cash - 15000.0
        
        manager_multi.open_position(
            symbol="TSLA",
            shares=50.0,
            entry_price=200.0,
            entry_time=entry_time,
            entry_step=2,
        )
        cash_after_2 = manager_multi.cash
        assert cash_after_2 == initial_cash - 15000.0 - 10000.0
        
        # Open 1 short position (should increase cash)
        manager_multi.open_position(
            symbol="MSFT",
            shares=-30.0,
            entry_price=300.0,
            entry_time=entry_time,
            entry_step=3,
        )
        cash_after_3 = manager_multi.cash
        # Cash should increase by short proceeds
        assert cash_after_3 == cash_after_2 + 9000.0
        
        # Verify available capital accounts for margin requirements
        available = manager_multi.get_available_capital()
        assert available > 0, "Should have available capital"

    # ------------------------------------------------------------------
    # Test 6: Position-specific closing by position_id
    # ------------------------------------------------------------------
    def test_close_specific_position_by_id(self, manager_multi):
        """Test closing a specific position by position_id."""
        entry_time = datetime(2024, 1, 1, 10, 0)
        symbol = "AAPL"
        
        # Open 2 positions for same symbol
        success1, pos1 = manager_multi.open_position(
            symbol=symbol,
            shares=100.0,
            entry_price=150.0,
            entry_time=entry_time,
            entry_step=1,
        )
        success2, pos2 = manager_multi.open_position(
            symbol=symbol,
            shares=50.0,
            entry_price=152.0,
            entry_time=entry_time,
            entry_step=2,
        )
        assert len(manager_multi.positions) == 2
        
        # Close first position by position_id
        success, trade = manager_multi.close_position(
            symbol=symbol,
            shares_to_close=None,  # Close all
            exit_price=155.0,
            exit_time=datetime(2024, 1, 1, 16, 0),
            exit_step=10,
            position_id=pos1.position_id,
        )
        assert success, "Should close first position"
        assert trade is not None
        assert trade["position_id"] == pos1.position_id
        
        # Verify only 1 position remains
        assert len(manager_multi.positions) == 1
        remaining_pos = list(manager_multi.positions.values())[0]
        assert remaining_pos.position_id == pos2.position_id

    # ------------------------------------------------------------------
    # Test 7: Margin requirements for shorts
    # ------------------------------------------------------------------
    def test_short_margin_requirements(self, manager_multi):
        """Test that short positions require proper margin."""
        entry_time = datetime(2024, 1, 1, 10, 0)
        
        # Try to open very large short position
        # Short value = 500 shares * $200 = $100,000
        # Margin required = $100,000 * 1.5 = $150,000
        # Available capital = $100,000 (insufficient)
        success, pos = manager_multi.open_position(
            symbol="TSLA",
            shares=-500.0,  # SHORT
            entry_price=200.0,
            entry_time=entry_time,
            entry_step=1,
        )
        assert not success, "Should fail due to insufficient margin"
        
        # Try smaller short (should succeed)
        # Short value = 100 shares * $200 = $20,000
        # Margin required = $20,000 * 1.5 = $30,000 (within capital)
        success2, pos2 = manager_multi.open_position(
            symbol="TSLA",
            shares=-100.0,
            entry_price=200.0,
            entry_time=entry_time,
            entry_step=2,
        )
        assert success2, "Should succeed with reasonable margin"

    # ------------------------------------------------------------------
    # Test 8: Helper methods
    # ------------------------------------------------------------------
    def test_position_query_helpers(self, manager_multi):
        """Test position query helper methods."""
        entry_time = datetime(2024, 1, 1, 10, 0)
        
        # Open multiple positions
        success1, pos1 = manager_multi.open_position(
            symbol="AAPL",
            shares=100.0,
            entry_price=150.0,
            entry_time=entry_time,
            entry_step=1,
        )
        success2, pos2 = manager_multi.open_position(
            symbol="AAPL",
            shares=50.0,
            entry_price=152.0,
            entry_time=entry_time,
            entry_step=2,
        )
        success3, pos3 = manager_multi.open_position(
            symbol="TSLA",
            shares=-30.0,
            entry_price=200.0,
            entry_time=entry_time,
            entry_step=3,
        )
        
        # Test get_positions_for_symbol
        aapl_positions = manager_multi.get_positions_for_symbol("AAPL")
        assert len(aapl_positions) == 2
        
        tsla_positions = manager_multi.get_positions_for_symbol("TSLA")
        assert len(tsla_positions) == 1
        
        # Test get_position_by_id
        retrieved_pos = manager_multi.get_position_by_id(pos1.position_id)
        assert retrieved_pos is not None
        assert retrieved_pos.position_id == pos1.position_id
        
        # Test has_position_for_symbol
        assert manager_multi.has_position_for_symbol("AAPL")
        assert manager_multi.has_position_for_symbol("TSLA")
        assert not manager_multi.has_position_for_symbol("MSFT")
        
        # Test get_net_position_for_symbol
        aapl_net = manager_multi.get_net_position_for_symbol("AAPL")
        assert aapl_net == 150.0  # 100 + 50
        
        tsla_net = manager_multi.get_net_position_for_symbol("TSLA")
        assert tsla_net == -30.0  # Short
        
        # Test get_total_exposure_for_symbol
        aapl_exposure = manager_multi.get_total_exposure_for_symbol("AAPL")
        assert aapl_exposure > 0

    # ------------------------------------------------------------------
    # Test 9: Backward compatibility (single position mode)
    # ------------------------------------------------------------------
    def test_backward_compatibility_single_position(self, manager_single):
        """Test that single-position mode still works (backward compatibility)."""
        entry_time = datetime(2024, 1, 1, 10, 0)
        
        # Open first position
        success1, pos1 = manager_single.open_position(
            symbol="AAPL",
            shares=100.0,
            entry_price=150.0,
            entry_time=entry_time,
            entry_step=1,
        )
        assert success1, "First position should open"
        
        # Try to open second position (should fail due to max_positions=1)
        success2, pos2 = manager_single.open_position(
            symbol="TSLA",
            shares=50.0,
            entry_price=200.0,
            entry_time=entry_time,
            entry_step=2,
        )
        assert not success2, "Second position should fail in single-position mode"
        assert len(manager_single.positions) == 1
        
        # Try to open position for same symbol (should also fail)
        success3, pos3 = manager_single.open_position(
            symbol="AAPL",
            shares=50.0,
            entry_price=152.0,
            entry_time=entry_time,
            entry_step=3,
        )
        assert not success3, "Should not allow multiple positions for same symbol"

    # ------------------------------------------------------------------
    # Test 10: Shorting disabled mode
    # ------------------------------------------------------------------
    def test_shorting_disabled(self, manager_single):
        """Test that shorting can be disabled."""
        entry_time = datetime(2024, 1, 1, 10, 0)
        
        # Try to open short position (should fail)
        success, pos = manager_single.open_position(
            symbol="TSLA",
            shares=-100.0,  # SHORT
            entry_price=200.0,
            entry_time=entry_time,
            entry_step=1,
        )
        assert not success, "Short should fail when shorting_enabled=False"
        assert len(manager_single.positions) == 0

    # ------------------------------------------------------------------
    # Test 11: Partial position close
    # ------------------------------------------------------------------
    def test_partial_position_close(self, manager_multi):
        """Test closing part of a position."""
        entry_time = datetime(2024, 1, 1, 10, 0)
        
        # Open position with 100 shares
        success, pos = manager_multi.open_position(
            symbol="AAPL",
            shares=100.0,
            entry_price=150.0,
            entry_time=entry_time,
            entry_step=1,
        )
        initial_position_id = pos.position_id
        
        # Close 40 shares (partial)
        success_close, trade = manager_multi.close_position(
            symbol="AAPL",
            shares_to_close=40.0,
            exit_price=155.0,
            exit_time=datetime(2024, 1, 1, 16, 0),
            exit_step=10,
            position_id=initial_position_id,
        )
        assert success_close, "Partial close should succeed"
        
        # Verify position still exists with reduced shares
        assert len(manager_multi.positions) == 1
        remaining_pos = manager_multi.get_position_by_id(initial_position_id)
        assert remaining_pos is not None
        assert abs(remaining_pos.shares - 60.0) < 1e-6, "Should have 60 shares remaining"

    # ------------------------------------------------------------------
    # Test 12: Close short position with profit
    # ------------------------------------------------------------------
    def test_close_short_with_profit(self, manager_multi):
        """Test closing a short position that has made profit."""
        entry_time = datetime(2024, 1, 1, 10, 0)
        
        # Open short position at $200
        success, pos = manager_multi.open_position(
            symbol="TSLA",
            shares=-100.0,
            entry_price=200.0,
            entry_time=entry_time,
            entry_step=1,
        )
        assert pos.is_short()
        initial_cash = manager_multi.cash
        
        # Close short at $180 (profit = $20 per share)
        success_close, trade = manager_multi.close_position(
            symbol="TSLA",
            shares_to_close=None,  # Close all
            exit_price=180.0,
            exit_time=datetime(2024, 1, 1, 16, 0),
            exit_step=10,
            position_id=pos.position_id,
        )
        assert success_close
        assert trade["realized_pnl"] > 0, "Should have profit when short closes lower"
        expected_pnl = (200.0 - 180.0) * 100.0  # $2000
        assert abs(trade["realized_pnl"] - expected_pnl) < 1e-6

    # ------------------------------------------------------------------
    # Test 13: Position diversity metrics
    # ------------------------------------------------------------------
    def test_position_diversity(self, manager_multi):
        """Test that multiple positions enable portfolio diversity."""
        entry_time = datetime(2024, 1, 1, 10, 0)
        
        # Open 3 different symbol positions
        symbols = ["AAPL", "TSLA", "MSFT"]
        for i, symbol in enumerate(symbols, start=1):
            success, pos = manager_multi.open_position(
                symbol=symbol,
                shares=10.0,
                entry_price=100.0,
                entry_time=entry_time,
                entry_step=i,
            )
            assert success
        
        # Verify unique symbols
        unique_symbols = set(pos.symbol for pos in manager_multi.positions.values())
        assert len(unique_symbols) == 3, "Should have 3 unique symbols"

    # ------------------------------------------------------------------
    # Test 14: Leverage calculation with multiple positions
    # ------------------------------------------------------------------
    def test_leverage_with_multiple_positions(self, manager_multi):
        """Test leverage calculation with multiple positions."""
        entry_time = datetime(2024, 1, 1, 10, 0)
        
        # Open multiple positions to increase leverage
        manager_multi.open_position(
            symbol="AAPL",
            shares=100.0,
            entry_price=150.0,
            entry_time=entry_time,
            entry_step=1,
        )
        manager_multi.open_position(
            symbol="TSLA",
            shares=50.0,
            entry_price=200.0,
            entry_time=entry_time,
            entry_step=2,
        )
        
        # Calculate total exposure
        total_exposure = sum(abs(pos.current_value) for pos in manager_multi.positions.values())
        equity = manager_multi.get_equity()
        leverage = total_exposure / equity if equity > 0 else 0
        
        # Verify leverage is within limits
        assert leverage <= float(manager_multi.config.max_leverage) + 1e-6, "Leverage should be within limits"
        assert leverage > 0, "Leverage should be positive with open positions"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
