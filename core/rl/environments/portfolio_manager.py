"""Portfolio state management and risk controls for RL trading agents."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """Configuration for portfolio management.

    Risk Limits:
        max_position_size_pct: Maximum position size as % of equity.
        max_total_exposure_pct: Maximum total exposure (sum of all positions).
        max_leverage: Maximum leverage ratio allowed.
        max_positions: Maximum number of concurrent positions.

    Capital Management:
        initial_capital: Starting capital.
        reserve_capital_pct: Percentage of capital held in reserve.
        margin_requirement: Margin requirement for positions (0 = no margin).

    Risk Controls:
        max_position_loss_pct: Auto-close if single position loses this % of cost basis.
        max_portfolio_loss_pct: Emergency stop if portfolio drawdown exceeds this %.
        max_correlation: Maximum correlation between positions (multi-symbol support).

    Transaction Costs:
        commission_rate: Commission rate applied to trades.
        slippage_bps: Slippage in basis points per trade.
    """

    # Risk limits
    max_position_size_pct: float = 0.10
    max_total_exposure_pct: float = 1.0
    max_leverage: float = 1.0
    max_positions: int = 1

    # Capital management
    initial_capital: float = 100_000.0
    reserve_capital_pct: float = 0.10
    margin_requirement: float = 0.0
    min_position_value_pct: float = 0.0

    # Risk controls
    max_position_loss_pct: float = 0.05
    max_portfolio_loss_pct: float = 0.20
    max_correlation: float = 0.8

    # Transaction costs
    commission_rate: float = 0.001
    slippage_bps: float = 5.0
    
    # Multi-position & Shorting (Golden Shot Enhancement)
    allow_multiple_positions_per_symbol: bool = True  # Allow multiple entries for same symbol
    shorting_enabled: bool = True  # Enable short selling
    short_margin_requirement: float = 1.5  # 150% margin for shorts (regulatory standard)

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert 0 < self.max_position_size_pct <= 1.0
        assert 0 < self.max_total_exposure_pct <= 2.0
        assert self.max_leverage >= 1.0
        assert self.max_positions >= 1
        assert 0 <= self.reserve_capital_pct < 1.0
        assert 0 <= self.margin_requirement <= 1.0
        assert 0 <= self.min_position_value_pct <= 1.0
        assert 0 <= self.max_position_loss_pct < 1.0
        assert 0 < self.max_portfolio_loss_pct <= 1.0
        assert 0 <= self.commission_rate < 1.0
        assert self.slippage_bps >= 0


@dataclass
class Position:
    """Represents a single trading position."""

    symbol: str
    shares: float
    entry_price: float
    entry_time: datetime
    entry_step: int
    cost_basis: float
    position_id: str = ""
    entry_reason: str = "agent_decision"

    # fields updated dynamically
    current_price: float = 0.0
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    
    # Cost tracking for pyramiding (ADD_POSITION)
    commission: float = 0.0
    slippage: float = 0.0
    
    # Metadata for tracking entry/exit strategies (used by continuous_trading_env)
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize mutable default values."""
        if self.metadata is None:
            self.metadata = {}

    def update(self, current_price: float) -> None:
        """Update position valuations using the current market price.
        
        For LONG positions (shares > 0):
            - current_value = shares * price (positive)
            - unrealized_pnl = current_value - cost_basis
            - Profit when price rises
            
        For SHORT positions (shares < 0):
            - current_value = shares * price (negative)
            - unrealized_pnl = (entry_price - current_price) * abs(shares)
            - Profit when price falls
        """
        self.current_price = current_price
        self.current_value = self.shares * current_price
        
        # P&L calculation handles both longs and shorts correctly
        if self.shares > 0:  # LONG position
            # Long: profit when price rises
            self.unrealized_pnl = self.current_value - self.cost_basis
        else:  # SHORT position
            # Short: profit = (entry_price - current_price) * |shares|
            # If we sold at 200 and price is now 180, we profit $20/share
            self.unrealized_pnl = (self.entry_price - current_price) * abs(self.shares)
            
        cost_basis = abs(self.cost_basis) if self.cost_basis != 0 else 1e-12
        self.unrealized_pnl_pct = self.unrealized_pnl / cost_basis

    def get_holding_period(self, current_step: int) -> int:
        """Return the holding period in environment steps."""
        return max(0, current_step - self.entry_step)
    
    def is_long(self) -> bool:
        """Return True if this is a long position."""
        return self.shares > 0
    
    def is_short(self) -> bool:
        """Return True if this is a short position."""
        return self.shares < 0
    
    def position_type(self) -> str:
        """Return 'LONG' or 'SHORT'."""
        return "LONG" if self.shares > 0 else "SHORT"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize position attributes to a dictionary."""
        return {
            "symbol": self.symbol,
            "shares": self.shares,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "entry_step": self.entry_step,
            "cost_basis": self.cost_basis,
            "current_price": self.current_price,
            "current_value": self.current_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "position_id": self.position_id,
            "entry_reason": self.entry_reason,
            "position_type": self.position_type(),
            "metadata": self.metadata,
        }


class PortfolioManager:
    """Portfolio state manager with risk controls and analytics."""

    def __init__(
        self,
        config: PortfolioConfig,
        *,
        log_trades: bool = True,
        log_level: Optional[int] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.log_trades = bool(log_trades)
        if log_level is not None:
            logger.setLevel(log_level)

        # State
        self.cash: float = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Dict] = []

        # Performance trackers
        self.equity_curve: List[float] = [config.initial_capital]
        self.peak_equity: float = config.initial_capital
        self.max_drawdown: float = 0.0

        # Trade statistics
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.total_pnl: float = 0.0

        # Risk state
        self.risk_violations: List[Dict] = []

        if self.log_trades and logger.isEnabledFor(logging.INFO):
            logger.info(
                "PortfolioManager initialized: capital=$%s, max_position=%s%%, max_positions=%s",
                f"{config.initial_capital:,.0f}",
                f"{config.max_position_size_pct * 100:.1f}",
                config.max_positions,
            )

    # ------------------------------------------------------------------
    # Capital & exposure utilities
    # ------------------------------------------------------------------
    def get_equity(self) -> float:
        """Return total equity (cash + market value of open positions)."""
        position_value = sum(position.current_value for position in self.positions.values())
        return self.cash + position_value

    def get_available_capital(self) -> float:
        """Capital available for new positions after reserves and margin."""
        equity = self.get_equity()
        reserve = equity * self.config.reserve_capital_pct
        deployed_cost_basis = sum(position.cost_basis for position in self.positions.values())
        margin_locked = deployed_cost_basis * self.config.margin_requirement
        available = self.cash - reserve - margin_locked
        return float(max(0.0, available))

    def get_max_position_value(self, symbol: str) -> float:
        """Compute the maximum allowed position value for a symbol."""
        equity = self.get_equity()
        max_from_position_limit = equity * float(self.config.max_position_size_pct)

        current_exposure = sum(position.current_value for position in self.positions.values())
        max_total_exposure = equity * float(self.config.max_total_exposure_pct)
        max_from_exposure_limit = max(0.0, max_total_exposure - current_exposure)

        available_capital = self.get_available_capital()

        max_value = min(max_from_position_limit, max_from_exposure_limit, available_capital)
        return float(max(0.0, max_value))

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------
    def can_open_position(self, symbol: str, target_value: float, is_short: bool = False) -> Tuple[bool, Optional[str]]:
        """Determine if a new position can be opened.
        
        Args:
            symbol: Trading symbol
            target_value: Position value (absolute, always positive)
            is_short: True if this is a short position
            
        Returns:
            Tuple of (can_open, reason_if_not)
        """
        # Check if shorting is allowed
        if is_short and not self.config.shorting_enabled:
            return False, "shorting_disabled"
        
        # Check position count limit
        if len(self.positions) >= self.config.max_positions:
            return False, f"max_positions_reached_{self.config.max_positions}"

        # If multiple positions per symbol not allowed, check for existing symbol position
        if not self.config.allow_multiple_positions_per_symbol:
            # Check if symbol already has a position
            if any(pos.symbol == symbol for pos in self.positions.values()):
                return False, "position_already_exists_for_symbol"

        # Check position size limit
        max_allowed = self.get_max_position_value(symbol)
        if target_value > max_allowed + 1e-8:
            return False, f"exceeds_position_limit_{max_allowed:.2f}"

        # Check available capital (shorts require margin)
        available_capital = self.get_available_capital()
        required_capital = target_value
        if is_short:
            # Shorts require additional margin
            required_capital = target_value * self.config.short_margin_requirement
            
        if required_capital > available_capital + 1e-8:
            return False, f"insufficient_capital_{available_capital:.2f}"

        # Check equity
        equity = self.get_equity()
        if equity <= 0:
            return False, "non_positive_equity"

        # Check minimum position value
        min_value = equity * float(self.config.min_position_value_pct)
        if min_value > 0.0 and target_value + 1e-8 < min_value:
            return False, f"below_min_position_value_{min_value:.2f}"

        # Check total exposure and leverage
        total_exposure_after = sum(abs(position.current_value) for position in self.positions.values()) + target_value
        leverage_after = total_exposure_after / equity if equity > 0 else float("inf")
        if leverage_after > float(self.config.max_leverage) + 1e-8:
            return False, f"exceeds_leverage_limit_{float(self.config.max_leverage):.2f}"

        return True, None

    def open_position(
        self,
        symbol: str,
        shares: float,
        entry_price: float,
        entry_time: datetime,
        entry_step: int,
        commission: float = 0.0,
        slippage: float = 0.0,
        entry_reason: str = "agent_decision",
    ) -> Tuple[bool, Optional[Position]]:
        """Open a new position and update portfolio state.
        
        Args:
            shares: Number of shares (positive for LONG, negative for SHORT)
        """
        if shares == 0:
            return False, None

        is_short = shares < 0
        position_value = abs(shares) * entry_price
        total_cost = position_value + commission + slippage

        can_open, reason = self.can_open_position(symbol, total_cost, is_short=is_short)
        if not can_open:
            logger.warning("Cannot open position for %s: %s", symbol, reason)
            return False, None

        # Generate unique position_id (now that we allow multiple positions per symbol)
        position_id = f"{symbol}_{entry_step}_{int(entry_time.timestamp() * 1000)}"

        position = Position(
            symbol=symbol,
            shares=shares,  # Can be negative for shorts
            entry_price=entry_price,
            entry_time=entry_time,
            entry_step=entry_step,
            cost_basis=total_cost,
            position_id=position_id,
            entry_reason=entry_reason,
        )
        position.update(entry_price)

        # Update cash
        if is_short:
            # Short: receive proceeds but lock margin
            self.cash += position_value - (commission + slippage)
            # Margin requirement is handled by can_open_position check
        else:
            # Long: pay for shares
            self.cash -= total_cost
            
        # Store with position_id as key (not symbol)
        self.positions[position_id] = position

        if self.log_trades and logger.isEnabledFor(logging.INFO):
            logger.info(
                "Opened position %s: shares=%.4f price=%.4f value=%.2f commission=%.2f slippage=%.2f",
                symbol,
                shares,
                entry_price,
                position_value,
                commission,
                slippage,
            )
        return True, position

    def close_position(
        self,
        symbol: str,
        shares_to_close: Optional[float],
        exit_price: float,
        exit_time: datetime,
        exit_step: int,
        exit_reason: str = "agent_decision",
        commission: float = 0.0,
        slippage: float = 0.0,
        position_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[Dict]]:
        """Close a position (fully or partially).
        
        Args:
            position_id: If provided, close specific position by ID. Otherwise, find position by symbol.
            shares_to_close: Number of shares to close (positive value, regardless of long/short).
                            If None, closes entire position.
        """
        # Find position by position_id or symbol
        if position_id is not None:
            if position_id not in self.positions:
                logger.warning("Attempted to close non-existent position_id %s", position_id)
                return False, None
            position = self.positions[position_id]
        else:
            # Legacy support: find position by symbol (for backward compatibility)
            matching_positions = [pos_id for pos_id, pos in self.positions.items() if pos.symbol == symbol]
            if not matching_positions:
                logger.warning("Attempted to close non-existent position for %s", symbol)
                return False, None
            position_id = matching_positions[0]  # Close first matching position
            position = self.positions[position_id]

        is_short = position.is_short()
        shares_to_close = abs(position.shares) if shares_to_close is None else abs(shares_to_close)
        if shares_to_close <= 0:
            return False, None

        shares_to_close = min(shares_to_close, abs(position.shares))
        
        # Calculate proceeds and P&L
        gross_proceeds = shares_to_close * exit_price
        net_proceeds = gross_proceeds - commission - slippage

        cost_fraction = shares_to_close / abs(position.shares)
        allocated_cost = position.cost_basis * cost_fraction
        
        # P&L calculation differs for long vs short
        if is_short:
            # Short: profit when price falls, loss when price rises
            # We received money when opening, now we pay to buy back shares
            realized_pnl = allocated_cost - net_proceeds
        else:
            # Long: profit when price rises
            realized_pnl = net_proceeds - allocated_cost
            
        allocated_cost = allocated_cost if allocated_cost != 0 else 1e-12
        realized_pnl_pct = realized_pnl / allocated_cost

        holding_period = exit_step - position.entry_step

        trade_result = {
            "symbol": symbol,
            "position_id": position_id,
            "position_type": position.position_type(),
            "shares": shares_to_close if not is_short else -shares_to_close,  # Report with sign
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "entry_time": position.entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
            "entry_step": position.entry_step,
            "exit_step": exit_step,
            "holding_period": holding_period,
            "cost_basis": allocated_cost,
            "proceeds": net_proceeds,
            "realized_pnl": realized_pnl,
            "realized_pnl_pct": realized_pnl_pct,
            "exit_reason": exit_reason,
            "commission": commission,
            "slippage": slippage,
        }

        # Update cash
        if is_short:
            # Short: pay to buy back shares
            self.cash -= net_proceeds
        else:
            # Long: receive proceeds from sale
            self.cash += net_proceeds

        # Update or remove position
        if shares_to_close >= abs(position.shares) - 1e-8:
            del self.positions[position_id]
        else:
            # Partial close: update shares (maintain sign for short positions)
            if is_short:
                position.shares += shares_to_close  # Add positive to negative
            else:
                position.shares -= shares_to_close
            position.cost_basis -= allocated_cost
            position.update(exit_price)

        self.closed_positions.append(trade_result)
        self.total_trades += 1
        if realized_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        self.total_pnl += realized_pnl

        if self.log_trades and logger.isEnabledFor(logging.INFO):
            logger.info(
                "Closed position %s: shares=%.4f price=%.4f pnl=%.2f reason=%s",
                symbol,
                shares_to_close,
                exit_price,
                realized_pnl,
                exit_reason,
            )
        return True, trade_result

    # ------------------------------------------------------------------
    # Risk monitoring & enforcement
    # ------------------------------------------------------------------
    def update_positions(self, current_prices: Dict[str, float], current_step: int) -> None:
        """Refresh mark-to-market valuations for all positions."""
        for position_id, position in self.positions.items():
            if position.symbol in current_prices:
                position.update(current_prices[position.symbol])

        equity = self.get_equity()
        self.equity_curve.append(equity)
        if equity > self.peak_equity:
            self.peak_equity = equity

        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, drawdown)

    def check_risk_violations(self, current_step: int) -> List[Dict]:
        """Check for any risk limit violations."""
        violations: List[Dict] = []

        if self.max_drawdown > self.config.max_portfolio_loss_pct:
            violations.append(
                {
                    "type": "portfolio_drawdown",
                    "severity": "critical",
                    "value": self.max_drawdown,
                    "limit": self.config.max_portfolio_loss_pct,
                    "step": current_step,
                    "action_required": "close_all_positions",
                }
            )

        for symbol, position in self.positions.items():
            if position.unrealized_pnl_pct < -self.config.max_position_loss_pct:
                violations.append(
                    {
                        "type": "position_loss",
                        "severity": "high",
                        "symbol": symbol,
                        "loss_pct": position.unrealized_pnl_pct,
                        "limit": -self.config.max_position_loss_pct,
                        "step": current_step,
                        "action_required": f"close_position_{symbol}",
                    }
                )

        for violation in violations:
            logger.warning("Risk violation detected: %s", violation)
            self.risk_violations.append(violation)

        return violations

    def enforce_risk_limits(
        self,
        current_prices: Dict[str, float],
        current_time: datetime,
        current_step: int,
    ) -> List[Dict]:
        """Automatically enforce risk limits by closing positions."""
        violations = self.check_risk_violations(current_step)
        forced_trades: List[Dict] = []

        for violation in violations:
            if violation["type"] == "portfolio_drawdown":
                logger.critical("Portfolio drawdown limit exceeded. Closing all positions.")
                # With multi-position support, positions.keys() returns position_ids, not symbols
                for position_id in list(self.positions.keys()):
                    position = self.positions[position_id]
                    symbol = position.symbol
                    if symbol not in current_prices:
                        continue
                    _, trade = self.close_position(
                        symbol=symbol,
                        shares_to_close=None,
                        exit_price=current_prices[symbol],
                        exit_time=current_time,
                        exit_step=current_step,
                        exit_reason="portfolio_drawdown_limit",
                    )
                    if trade:
                        trade["forced_exit"] = True
                        trade["forced_exit_reason"] = "portfolio_drawdown_limit"
                        trade["risk_limit"] = {
                            "type": "portfolio_drawdown",
                            "threshold_pct": float(self.config.max_portfolio_loss_pct),
                            "observed_pct": float(violation.get("value", 0.0)),
                        }
                        forced_trades.append(trade)

            elif violation["type"] == "position_loss":
                position_id = violation["symbol"]  # This is actually position_id in multi-position
                if position_id not in self.positions:
                    continue
                position = self.positions[position_id]
                symbol = position.symbol
                if symbol not in current_prices:
                    continue
                logger.warning("Position loss limit exceeded for %s. Auto-closing position.", position_id)
                _, trade = self.close_position(
                    symbol=symbol,
                    shares_to_close=None,
                    exit_price=current_prices[symbol],
                    exit_time=current_time,
                    exit_step=current_step,
                    exit_reason="position_loss_limit",
                )
                if trade:
                    trade["forced_exit"] = True
                    trade["forced_exit_reason"] = "position_loss_limit"
                    trade["risk_limit"] = {
                        "type": "position_loss",
                        "threshold_pct": float(self.config.max_position_loss_pct),
                        "observed_pct": float(violation.get("loss_pct", 0.0)),
                    }
                    forced_trades.append(trade)

        return forced_trades

    # ------------------------------------------------------------------
    # Position query helpers
    # ------------------------------------------------------------------
    def get_positions_for_symbol(self, symbol: str) -> List[Position]:
        """Get all positions for a specific symbol.
        
        Returns:
            List of Position objects for the given symbol
        """
        return [pos for pos in self.positions.values() if pos.symbol == symbol]
    
    def get_position_by_id(self, position_id: str) -> Optional[Position]:
        """Get a position by its ID.
        
        Returns:
            Position object if found, None otherwise
        """
        return self.positions.get(position_id)
    
    def has_position_for_symbol(self, symbol: str) -> bool:
        """Check if any position exists for a symbol.
        
        Returns:
            True if at least one position exists for the symbol
        """
        return any(pos.symbol == symbol for pos in self.positions.values())
    
    def get_total_exposure_for_symbol(self, symbol: str) -> float:
        """Get total exposure (sum of absolute values) for a symbol.
        
        Returns:
            Total exposure value across all positions for the symbol
        """
        return sum(abs(pos.current_value) for pos in self.positions.values() if pos.symbol == symbol)
    
    def get_net_position_for_symbol(self, symbol: str) -> float:
        """Get net position (long - short) for a symbol in shares.
        
        Returns:
            Net shares (positive = net long, negative = net short, zero = flat)
        """
        return sum(pos.shares for pos in self.positions.values() if pos.symbol == symbol)

    # ------------------------------------------------------------------
    # Analytics & reporting
    # ------------------------------------------------------------------
    def get_portfolio_metrics(self) -> Dict:
        """Return a dictionary of portfolio-level metrics."""
        equity = self.get_equity()
        initial_capital = self.config.initial_capital
        total_return = (equity - initial_capital) / initial_capital if initial_capital else 0.0

        total_trades = max(1, self.total_trades)
        win_rate = self.winning_trades / total_trades

        sharpe = 0.0
        sortino = 0.0
        if len(self.equity_curve) > 2:
            curve = np.asarray(self.equity_curve, dtype=np.float64)
            returns = np.diff(curve) / curve[:-1]
            if returns.size > 0 and np.std(returns) > 0:
                sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
            downside_returns = returns[returns < 0]
            if downside_returns.size > 0 and np.std(downside_returns) > 0:
                sortino = float(np.mean(returns) / np.std(downside_returns) * np.sqrt(252))

        position_value = sum(position.current_value for position in self.positions.values())
        exposure_pct = position_value / equity if equity > 0 else 0.0
        leverage = position_value / equity if equity > 0 else 0.0

        return {
            "equity": equity,
            "cash": self.cash,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "peak_equity": self.peak_equity,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "num_positions": len(self.positions),
            "position_value": position_value,
            "exposure_pct": exposure_pct,
            "leverage": leverage,
            "available_capital": self.get_available_capital(),
        }

    def get_position_summary(self) -> List[Dict]:
        """Return serialized summaries of all open positions."""
        return [position.to_dict() for position in self.positions.values()]

    def get_closed_positions(self) -> List[Dict]:
        """Return the list of closed position records."""
        return list(self.closed_positions)

    def get_risk_violations(self) -> List[Dict]:
        """Return recorded risk violations."""
        return list(self.risk_violations)

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset portfolio state to initial configuration."""
        self.cash = self.config.initial_capital
        self.positions.clear()
        self.closed_positions.clear()
        self.equity_curve = [self.config.initial_capital]
        self.peak_equity = self.config.initial_capital
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.risk_violations.clear()
        if self.log_trades and logger.isEnabledFor(logging.INFO):
            logger.info("PortfolioManager state reset to initial conditions.")
