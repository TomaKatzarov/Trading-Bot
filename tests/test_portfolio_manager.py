"""Comprehensive tests for the RL portfolio manager module."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pytest

from core.rl.environments.portfolio_manager import (
    PortfolioConfig,
    PortfolioManager,
    Position,
)


def _ts(offset: int = 0) -> datetime:
    """Return a deterministic timestamp offset by hours."""
    base = datetime(2025, 1, 1, 9, 30)
    return base + timedelta(hours=offset)


class TestPortfolioConfig:
    """Validate the PortfolioConfig dataclass."""

    def test_default_config(self) -> None:
        config = PortfolioConfig()

        assert config.max_position_size_pct == pytest.approx(0.10)
        assert config.max_positions == 1
        assert config.initial_capital == 100_000.0
        assert config.reserve_capital_pct == pytest.approx(0.10)

    def test_validation_success(self) -> None:
        config = PortfolioConfig(
            max_position_size_pct=0.25,
            max_total_exposure_pct=1.5,
            max_leverage=1.25,
            max_positions=3,
            reserve_capital_pct=0.05,
            margin_requirement=0.25,
            max_position_loss_pct=0.10,
            max_portfolio_loss_pct=0.35,
        )

        # Should not raise
        config.validate()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_position_size_pct": 0},
            {"max_total_exposure_pct": 0},
            {"max_leverage": 0.0},
            {"max_positions": 0},
            {"reserve_capital_pct": 1.5},
            {"margin_requirement": -0.1},
            {"max_portfolio_loss_pct": 0.0},
            {"commission_rate": 1.5},
        ],
    )
    def test_validation_failures(self, kwargs: Dict) -> None:
        config = PortfolioConfig(**kwargs)
        with pytest.raises(AssertionError):
            config.validate()


class TestPositionTracking:
    """Exercise core position lifecycle flows."""

    def test_open_position_reduces_cash(self) -> None:
        config = PortfolioConfig(initial_capital=100_000, max_position_size_pct=1.0)
        manager = PortfolioManager(config)

        success, position = manager.open_position(
            symbol="AAPL",
            shares=100,
            entry_price=150.0,
            entry_time=_ts(),
            entry_step=5,
            commission=15.0,
            slippage=5.0,
        )

        assert success
        assert position is not None
        assert position.symbol == "AAPL"
        assert position.shares == pytest.approx(100.0)
        total_cost = 100 * 150.0 + 15.0 + 5.0
        assert manager.cash == pytest.approx(config.initial_capital - total_cost)

    def test_cannot_open_duplicate_symbol(self) -> None:
        manager = PortfolioManager(PortfolioConfig(max_position_size_pct=1.0))
        manager.open_position("AAPL", 10, 100.0, _ts(), 1)

        success, _ = manager.open_position("AAPL", 5, 120.0, _ts(1), 2)

        assert not success

    def test_position_update_marks_to_market(self) -> None:
        manager = PortfolioManager(PortfolioConfig(max_position_size_pct=1.0))
        manager.open_position("AAPL", 50, 100.0, _ts(), 1)

        manager.update_positions({"AAPL": 112.5}, current_step=2)
        # With multi-position support, get position by symbol
        aapl_positions = manager.get_positions_for_symbol("AAPL")
        assert len(aapl_positions) == 1
        pos = aapl_positions[0]

        assert pos.current_price == pytest.approx(112.5)
        assert pos.current_value == pytest.approx(50 * 112.5)
        assert pos.unrealized_pnl == pytest.approx(50 * (112.5 - 100.0))
        assert pos.unrealized_pnl_pct == pytest.approx((112.5 - 100.0) / 100.0)

    def test_close_full_position_records_trade(self) -> None:
        manager = PortfolioManager(PortfolioConfig(initial_capital=100_000, max_position_size_pct=1.0))
        manager.open_position("AAPL", 100, 100.0, _ts(), 1, commission=10.0)

        success, trade = manager.close_position(
            symbol="AAPL",
            shares_to_close=None,
            exit_price=110.0,
            exit_time=_ts(5),
            exit_step=10,
            exit_reason="target_hit",
            commission=11.0,
        )

        assert success
        assert trade is not None
        assert trade["realized_pnl"] == pytest.approx((110.0 * 100 - 11.0) - (100.0 * 100 + 10.0))
        assert "AAPL" not in manager.positions
        assert manager.total_trades == 1
        assert manager.winning_trades == 1
        assert manager.total_pnl > 0

    def test_close_partial_position_updates_remaining(self) -> None:
        manager = PortfolioManager(PortfolioConfig(max_position_size_pct=1.0))
        manager.open_position("AAPL", 100, 100.0, _ts(), 1)

        success, trade = manager.close_position(
            symbol="AAPL",
            shares_to_close=40,
            exit_price=105.0,
            exit_time=_ts(3),
            exit_step=7,
        )

        assert success
        assert trade is not None
        # With multi-position support, get position by symbol
        aapl_positions = manager.get_positions_for_symbol("AAPL")
        assert len(aapl_positions) == 1
        remaining = aapl_positions[0]
        assert remaining.shares == pytest.approx(60.0)
        assert remaining.cost_basis == pytest.approx(100.0 * 100.0 - trade["cost_basis"])
        assert trade["realized_pnl"] == pytest.approx(40 * 105.0 - trade["commission"] - trade["slippage"] - 40 * 100.0)

    def test_get_holding_period(self) -> None:
        position = Position(
            symbol="AAPL",
            shares=10,
            entry_price=100.0,
            entry_time=_ts(),
            entry_step=15,
            cost_basis=1_000.0,
        )

        assert position.get_holding_period(20) == 5
        assert position.get_holding_period(15) == 0

    def test_position_to_dict_includes_metadata(self) -> None:
        position = Position(
            symbol="AAPL",
            shares=10,
            entry_price=100.0,
            entry_time=_ts(),
            entry_step=15,
            cost_basis=1_000.0,
            position_id="AAPL_15",
            entry_reason="test",
        )
        position.update(110.0)
        as_dict = position.to_dict()

        assert as_dict["symbol"] == "AAPL"
        assert as_dict["current_price"] == pytest.approx(110.0)
        assert "position_id" in as_dict
        assert "entry_reason" in as_dict


class TestRiskLimits:
    """Cover risk related checks and auto liquidation scenarios."""

    def test_position_size_limit(self) -> None:
        manager = PortfolioManager(PortfolioConfig(initial_capital=100_000, max_position_size_pct=0.10))
        can_open, reason = manager.can_open_position("AAPL", target_value=15_000)

        assert not can_open
        assert reason is not None and "exceeds_position_limit" in reason

    def test_max_positions_limit(self) -> None:
        manager = PortfolioManager(PortfolioConfig(max_positions=1, max_position_size_pct=1.0))
        manager.open_position("AAPL", 10, 100.0, _ts(), 1)

        can_open, reason = manager.can_open_position("GOOGL", target_value=5_000)

        assert not can_open
        assert reason == "max_positions_reached_1"

    def test_exposure_or_leverage_guards_prevent_overextension(self) -> None:
        manager = PortfolioManager(
            PortfolioConfig(
                initial_capital=100_000,
                max_position_size_pct=0.9,
                max_total_exposure_pct=0.9,
                max_leverage=1.0,
            )
        )

        can_open, reason = manager.can_open_position("AAPL", target_value=95_000)

        assert not can_open
        assert reason is not None
        assert any(
            marker in reason
            for marker in ("exceeds_position_limit", "exceeds_leverage_limit", "insufficient_capital")
        )

    def test_position_loss_triggers_auto_close(self) -> None:
        manager = PortfolioManager(
            PortfolioConfig(
                initial_capital=100_000,
                max_position_size_pct=1.0,
                max_position_loss_pct=0.05,
            )
        )
        manager.open_position("AAPL", 100, 100.0, _ts(), 1)

        # Price drops 10%
        manager.update_positions({"AAPL": 90.0}, current_step=2)
        forced_trades = manager.enforce_risk_limits({"AAPL": 90.0}, _ts(2), 2)

        assert forced_trades
        assert forced_trades[0]["exit_reason"] == "position_loss_limit"
        assert "AAPL" not in manager.positions

    def test_portfolio_drawdown_triggers_emergency_liquidation(self) -> None:
        manager = PortfolioManager(
            PortfolioConfig(
                initial_capital=100_000,
                max_position_size_pct=1.0,
                max_portfolio_loss_pct=0.20,
            )
        )
        manager.open_position("AAPL", 400, 125.0, _ts(), 1)

        # Update price to induce >20% drawdown
        manager.update_positions({"AAPL": 70.0}, current_step=5)
        forced_trades = manager.enforce_risk_limits({"AAPL": 70.0}, _ts(5), 5)

        assert forced_trades
        assert all(
            trade["exit_reason"] in {"portfolio_drawdown_limit", "position_loss_limit"}
            for trade in forced_trades
        )
        assert not manager.positions

    def test_enforce_risk_limits_skips_unknown_prices(self) -> None:
        manager = PortfolioManager(PortfolioConfig(max_position_size_pct=1.0, max_portfolio_loss_pct=0.01))
        manager.open_position("AAPL", 10, 100.0, _ts(), 1)

        # Force drawdown but don't provide price mapping -> no action
        manager.update_positions({"AAPL": 0.0}, current_step=3)
        forced_trades = manager.enforce_risk_limits({}, _ts(3), 3)

        assert forced_trades == []
        # With multi-position support, check by symbol
        aapl_positions = manager.get_positions_for_symbol("AAPL")
        assert len(aapl_positions) == 1


class TestCapitalManagement:
    """Validate capital availability and exposure computations."""

    def test_available_capital_respects_reserve(self) -> None:
        manager = PortfolioManager(PortfolioConfig(initial_capital=120_000, reserve_capital_pct=0.25))

        assert manager.get_available_capital() == pytest.approx(120_000 * 0.75)

    def test_available_capital_after_trade_and_margin(self) -> None:
        manager = PortfolioManager(
            PortfolioConfig(
                initial_capital=100_000,
                reserve_capital_pct=0.10,
                margin_requirement=0.5,
                max_position_size_pct=1.0,
            )
        )
        manager.open_position("AAPL", 200, 100.0, _ts(), 1)

        # Cost basis = 20,000; margin locked = 10,000; reserve = equity * 0.10
        available = manager.get_available_capital()
        equity = manager.get_equity()
        reserve = equity * 0.10
        margin_locked = 20_000 * 0.5
        expected = manager.cash - reserve - margin_locked

        assert available == pytest.approx(max(0.0, expected))

    def test_max_position_value_limits(self) -> None:
        manager = PortfolioManager(
            PortfolioConfig(
                initial_capital=100_000,
                max_position_size_pct=0.10,
                reserve_capital_pct=0.10,
            )
        )

        max_value = manager.get_max_position_value("AAPL")

        assert max_value == pytest.approx(10_000.0)

    def test_equity_calculation(self) -> None:
        manager = PortfolioManager(PortfolioConfig(initial_capital=100_000, max_position_size_pct=1.0))
        manager.open_position("AAPL", 100, 100.0, _ts(), 1)
        manager.update_positions({"AAPL": 110.0}, current_step=2)

        expected_equity = manager.cash + 110.0 * 100
        assert manager.get_equity() == pytest.approx(expected_equity)


class TestPortfolioAnalytics:
    """Ensure analytics cover trade statistics and ratios."""

    def test_metrics_after_profitable_trade(self) -> None:
        manager = PortfolioManager(PortfolioConfig(initial_capital=100_000, max_position_size_pct=1.0))
        manager.open_position("AAPL", 100, 100.0, _ts(), 1)
        manager.update_positions({"AAPL": 110.0}, current_step=5)
        manager.close_position("AAPL", None, 110.0, _ts(6), 6)

        metrics = manager.get_portfolio_metrics()

        assert metrics["equity"] == pytest.approx(manager.get_equity())
        assert metrics["total_trades"] == 1
        assert metrics["winning_trades"] == 1
        assert metrics["total_pnl"] > 0
        assert metrics["available_capital"] == pytest.approx(manager.get_available_capital())

    def test_sharpe_ratio_positive_trend(self) -> None:
        manager = PortfolioManager(PortfolioConfig())
        manager.equity_curve = [100_000 + i * 250 for i in range(40)]

        metrics = manager.get_portfolio_metrics()

        assert metrics["sharpe_ratio"] > 0
        assert metrics["sortino_ratio"] >= 0

    def test_drawdown_tracking(self) -> None:
        manager = PortfolioManager(PortfolioConfig(initial_capital=100_000))
        manager.equity_curve = [100_000, 120_000, 90_000, 110_000, 95_000]
        manager.peak_equity = 120_000
        manager.max_drawdown = max((120_000 - v) / 120_000 for v in manager.equity_curve)

        metrics = manager.get_portfolio_metrics()

        assert metrics["max_drawdown"] == pytest.approx(manager.max_drawdown)
        assert metrics["max_drawdown_pct"] == pytest.approx(manager.max_drawdown * 100)

    def test_position_summary_and_closed_positions(self) -> None:
        manager = PortfolioManager(PortfolioConfig(max_position_size_pct=1.0))
        manager.open_position("AAPL", 10, 100.0, _ts(), 1)
        manager.close_position("AAPL", None, 105.0, _ts(2), 2)

        summary = manager.get_position_summary()
        closed = manager.get_closed_positions()

        assert summary == []
        assert len(closed) == 1
        assert closed[0]["symbol"] == "AAPL"


class TestEdgeCases:
    """Probe unusual scenarios for robustness."""

    def test_close_nonexistent_position(self) -> None:
        manager = PortfolioManager(PortfolioConfig())
        success, trade = manager.close_position("AAPL", None, 100.0, _ts(), 1)

        assert not success
        assert trade is None

    def test_bankruptcy_drawdown_reaches_one(self) -> None:
        manager = PortfolioManager(
            PortfolioConfig(
                initial_capital=100_000,
                max_position_size_pct=1.0,
                max_portfolio_loss_pct=1.0,
                reserve_capital_pct=0.0,
            )
        )
        opened, _ = manager.open_position("AAPL", 1_000, 100.0, _ts(), 1)
        assert opened
        manager.update_positions({"AAPL": 0.0}, current_step=3)

        assert manager.max_drawdown == pytest.approx(1.0, rel=1e-6)
        forced_trades = manager.enforce_risk_limits({"AAPL": 0.0}, _ts(3), 3)
        assert forced_trades
        assert manager.get_equity() == pytest.approx(manager.cash)

    def test_risk_violations_cached(self) -> None:
        manager = PortfolioManager(
            PortfolioConfig(
                initial_capital=100_000,
                max_position_size_pct=1.0,
                max_portfolio_loss_pct=0.05,
            )
        )
        manager.open_position("AAPL", 800, 100.0, _ts(), 1)
        manager.update_positions({"AAPL": 80.0}, current_step=3)
        manager.check_risk_violations(3)

        violations = manager.get_risk_violations()

        assert violations
        assert any(v["type"] == "portfolio_drawdown" for v in violations)

    def test_reset_clears_state(self) -> None:
        manager = PortfolioManager(PortfolioConfig(initial_capital=100_000, max_position_size_pct=1.0))
        manager.open_position("AAPL", 10, 100.0, _ts(), 1)
        manager.total_trades = 5
        manager.risk_violations.append({"type": "test"})

        manager.reset()

        assert manager.cash == pytest.approx(100_000.0)
        assert manager.positions == {}
        assert manager.total_trades == 0
        assert manager.risk_violations == []
        assert manager.equity_curve == [100_000.0]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])  # pragma: no cover
