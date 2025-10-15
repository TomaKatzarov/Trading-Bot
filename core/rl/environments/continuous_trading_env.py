"""Continuous-action variant of the trading environment."""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces

from .trading_env import TradeAction, TradingConfig, TradingEnvironment


class ContinuousTradingEnvironment(TradingEnvironment):
    """Gymnasium environment with a one-dimensional continuous action space."""

    def __init__(self, config: TradingConfig, seed: Optional[int] = None) -> None:
        super().__init__(config, seed)
        settings = getattr(config, "continuous_settings", {}) or {}

        self.hold_threshold: float = float(settings.get("hold_threshold", 0.1))
        self.max_position_pct: float = float(settings.get("max_position_pct", 0.15))
        self.transaction_cost: float = float(settings.get("transaction_cost", 0.003))
        self.smoothing_window: int = max(1, int(settings.get("smoothing_window", 3)))
        self.min_trade_value: float = float(settings.get("min_trade_value", 25.0))
        self.min_hold_steps: int = max(0, int(settings.get("min_hold_steps", 1)))

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._continuous_history: deque[float] = deque(maxlen=self.smoothing_window)
        self._continuous_trade_context: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _smooth_action(self, raw_action: float) -> float:
        self._continuous_history.append(raw_action)
        if not self._continuous_history:
            return raw_action
        return float(np.mean(self._continuous_history))

    def _compute_entry_label(self, equity: float, target_value: float) -> str:
        pct = 0.0 if equity <= 0 else target_value / equity
        if pct <= 0.035:
            return "small"
        if pct <= 0.08:
            return "medium"
        return "large"

    def _mask_invalid_action(self, smoothed: float, context: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Apply adaptive action masking to prevent exploitative behavior.
        
        Rules:
        1. Block excessive buying (>70% buy concentration in last 20 steps)
        2. Enforce minimum holding period (2 hours = 2 steps for hourly data)
        3. Progressive position building (only small entries when no position)
        
        Args:
            smoothed: Smoothed continuous action [-1, 1]
            context: Action context dictionary
            
        Returns:
            Tuple of (masked_action, updated_context)
        """
        symbol = self.config.symbol
        # Get positions for this symbol (now supports multiple positions)
        symbol_positions = self.portfolio.get_positions_for_symbol(symbol)
        has_position = len(symbol_positions) > 0
        position_count = len(self.portfolio.positions)
        max_positions = self.portfolio.config.max_positions
        
        # Rule 1: Block excessive buying (anti-bin-19 spam)
        if smoothed > 0.0:  # Buy action
            lookback = 20
            recent_actions = self.executed_action_history[-lookback:] if len(self.executed_action_history) >= lookback else self.executed_action_history
            if len(recent_actions) >= 10:  # Need minimum sample
                buy_count = sum(1 for a in recent_actions if a in [TradeAction.BUY_SMALL.value, TradeAction.BUY_MEDIUM.value, TradeAction.BUY_LARGE.value])
                buy_pct = buy_count / len(recent_actions)
                if buy_pct > 0.70:  # Block if >70% buy concentration
                    context["mask_reason"] = "excessive_buying"
                    context["buy_concentration"] = buy_pct
                    return 0.0, context  # Force HOLD
        
        # Rule 2: Enforce minimum holding period (2 hours)
        # Check all positions for this symbol
        if smoothed < 0.0 and has_position:  # Sell action with position
            # Find the most recent position for min hold check
            most_recent_pos = max(symbol_positions, key=lambda p: p.entry_step)
            holding_steps = self.current_step - int(most_recent_pos.entry_step)
            min_hold = 2  # 2 steps = 2 hours for hourly data
            if holding_steps < min_hold:
                context["mask_reason"] = "min_hold_enforced"
                context["holding_steps"] = holding_steps
                context["min_hold_required"] = min_hold
                return 0.0, context  # Force HOLD
        
        # Rule 3: Progressive position building (start small when no position)
        if smoothed > 0.0 and not has_position:  # New position entry
            # Cap magnitude to force small entry
            max_magnitude = 0.35  # Max 35% of max_position_pct
            if abs(smoothed) > max_magnitude:
                original = smoothed
                smoothed = max_magnitude if smoothed > 0 else -max_magnitude
                context["mask_reason"] = "progressive_entry"
                context["original_action"] = original
                context["masked_action"] = smoothed
        
        return smoothed, context

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self._continuous_history.clear()
        self._continuous_trade_context = None
        return super().reset(seed=seed, options=options)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if action is None:
            raise ValueError("Action cannot be None for continuous environment")

        raw = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        smoothed = self._smooth_action(raw)
        
        # Apply adaptive action masking (Golden Shot anti-exploit defense)
        mask_context: Dict[str, Any] = {}
        smoothed, mask_context = self._mask_invalid_action(smoothed, mask_context)
        
        symbol = self.config.symbol
        # Get positions for this symbol (supports multiple positions)
        symbol_positions = self.portfolio.get_positions_for_symbol(symbol)
        position = symbol_positions[0] if symbol_positions else None  # Use first position for legacy compatibility
        current_price = float(self.data.loc[self.current_step, "close"])
        available_capital = float(self.portfolio.get_available_capital())
        max_position_value = float(self.portfolio.get_max_position_value(symbol))
        equity = float(self.portfolio.get_equity())
        min_position_pct = float(getattr(self.portfolio.config, "min_position_value_pct", 0.0))
        portfolio_min_value = equity * min_position_pct if equity > 0 else 0.0
        # Enforce the portfolio's minimum trade size so evaluation actions do not collapse to HOLD.
        trade_floor = max(self.min_trade_value, portfolio_min_value)
        context: Dict[str, Any] = {
            "raw_action": raw,
            "smoothed_action": smoothed,
            "trade_type": "hold",
            "trade_value": 0.0,
            "shares": 0.0,
            "action": TradeAction.HOLD.value,
            "min_trade_value": trade_floor,
        }

        discrete_action = TradeAction.HOLD

        if abs(smoothed) < self.hold_threshold:
            context["reject_reason"] = "below_hold_threshold"
        elif smoothed > 0.0:
            desired_value = abs(smoothed) * self.max_position_pct * available_capital
            desired_value = min(desired_value, max_position_value)
            context["desired_trade_value"] = desired_value
            capital_limit = min(max_position_value, available_capital)
            if capital_limit <= 0:
                context["reject_reason"] = "no_available_capital"
            elif trade_floor > capital_limit + 1e-8:
                context["reject_reason"] = "floor_exceeds_limit"
                context["trade_limit"] = capital_limit
            else:
                adjusted_value = float(np.clip(desired_value, trade_floor, capital_limit))
                if adjusted_value < trade_floor - 1e-8:
                    context["reject_reason"] = "adjustment_below_floor"
                else:
                    context.update({
                        "trade_type": "buy",
                        "trade_value": adjusted_value,
                        "shares": adjusted_value / max(current_price, 1e-8),
                        "action": TradeAction.BUY_MEDIUM.value,
                        "trade_limit": capital_limit,
                        "adjusted": adjusted_value != desired_value,
                    })
                    discrete_action = TradeAction.BUY_MEDIUM
        else:
            if position is not None and position.shares > 0:
                holding_steps = self.current_step - int(position.entry_step)
                if self.min_hold_steps > 0 and holding_steps < self.min_hold_steps:
                    context["reject_reason"] = "min_hold_enforced"
                else:
                    position_value = float(position.current_value)
                    desired_value = abs(smoothed) * position_value
                    desired_value = min(desired_value, position_value)
                    context["desired_trade_value"] = desired_value
                    sell_floor = min(max(trade_floor, self.min_trade_value), position_value)
                    adjusted_value = max(desired_value, sell_floor)
                    adjusted_value = min(adjusted_value, position_value)
                    shares_target = adjusted_value / max(current_price, 1e-8)
                    if shares_target > 0:
                        close_all = adjusted_value >= position_value * 0.98 or shares_target >= position.shares * 0.98
                        shares_target = position.shares if close_all else min(shares_target, position.shares)
                        context.update({
                            "trade_type": "sell",
                            "trade_value": adjusted_value,
                            "shares": shares_target,
                            "close_all": close_all,
                            "action": (
                                TradeAction.SELL_ALL.value if close_all else TradeAction.SELL_PARTIAL.value
                            ),
                            "adjusted": adjusted_value != desired_value,
                        })
                        discrete_action = TradeAction.SELL_ALL if close_all else TradeAction.SELL_PARTIAL
                    else:
                        context["reject_reason"] = "non_positive_shares"
            else:
                context["reject_reason"] = "no_position"

        # Carry context only when required
        if context["trade_type"] in {"buy", "sell"}:
            self._continuous_trade_context = context
        else:
            self._continuous_trade_context = None

        # Temporarily restore the discrete action space so TradingEnvironment validation accepts the mapped action.
        original_space = self.action_space
        self.action_space = getattr(self, "discrete_action_space", spaces.Discrete(len(TradeAction)))
        try:
            observation, reward, terminated, truncated, info = super().step(discrete_action.value)
        finally:
            self.action_space = original_space
            # ensure context does not leak into subsequent steps
            self._continuous_trade_context = None

        info.setdefault("continuous_action", {}).update({
            "raw": context["raw_action"],
            "smoothed": context["smoothed_action"],
            "trade_type": context.get("trade_type", "hold"),
            "trade_value": context.get("trade_value", 0.0),
            "shares": context.get("shares", 0.0),
            "discrete_action": discrete_action.name,
            "min_trade_value": context.get("min_trade_value", 0.0),
            "reject_reason": context.get("reject_reason"),
            "trade_limit": context.get("trade_limit"),
            "adjusted": context.get("adjusted", False),
            "desired_trade_value": context.get("desired_trade_value"),
            "mask_reason": mask_context.get("mask_reason"),
            "buy_concentration": mask_context.get("buy_concentration"),
            "masked_action": mask_context.get("masked_action"),
            "original_action": mask_context.get("original_action"),
            # Multi-position tracking
            "position_count": len(self.portfolio.positions),
            "max_positions": self.portfolio.config.max_positions,
            "positions_for_symbol": len(self.portfolio.get_positions_for_symbol(self.config.symbol)),
        })
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Trading mechanics overrides
    # ------------------------------------------------------------------
    def _execute_action(self, action: int) -> Tuple[bool, Dict[str, Any]]:
        ctx = self._continuous_trade_context
        if not ctx or ctx.get("action") != int(action) or ctx.get("trade_type") == "hold":
            return super()._execute_action(action)

        trade_type = ctx.get("trade_type")
        info: Dict[str, Any]
        current_price = float(self.data.loc[self.current_step, "close"])
        current_time = self.data.loc[self.current_step, "timestamp"].to_pydatetime()
        info = {
            "action_name": self._action_name(action),
            "price": current_price,
        }
        portfolio_cfg = self.portfolio.config
        symbol = self.config.symbol

        if trade_type == "buy":
            target_value = float(ctx.get("trade_value", 0.0))
            if target_value <= 0:
                info["reject_reason"] = "non_positive_target"
                return False, info

            max_allowed = float(self.portfolio.get_max_position_value(symbol))
            available_capital = float(self.portfolio.get_available_capital())
            limit = min(max_allowed, available_capital)
            if target_value > limit:
                target_value = limit
            if target_value < self.min_trade_value:
                info["reject_reason"] = "insufficient_target"
                return False, info

            equity_before = float(self.portfolio.get_equity())
            entry_price = current_price * (1 + portfolio_cfg.slippage_bps / 10_000)
            shares = target_value / max(entry_price, 1e-8)
            if shares <= 0:
                info["reject_reason"] = "zero_shares"
                return False, info

            commission = target_value * portfolio_cfg.commission_rate
            slippage_cost = target_value * (portfolio_cfg.slippage_bps / 10_000)
            extra_cost = target_value * self.transaction_cost
            success, position = self.portfolio.open_position(
                symbol=symbol,
                shares=shares,
                entry_price=entry_price,
                entry_time=current_time,
                entry_step=self.current_step,
                commission=commission + extra_cost,
                slippage=slippage_cost,
            )

            if not success or position is None:
                info["reject_reason"] = "portfolio_rejected"
                return False, info

            entry_label = self._compute_entry_label(max(equity_before, 1e-8), target_value)
            if hasattr(position, "metadata"):
                position.metadata = getattr(position, "metadata", {}) or {}
                position.metadata["entry_size"] = entry_label
                position.metadata["pyramid_count"] = int(position.metadata.get("pyramid_count", 0))
                position.metadata["partial_exit_taken"] = False

            info.update({
                "entry_price": entry_price,
                "shares": shares,
                "trade_value": target_value,
                "entry_size": entry_label,
            })
            return True, info

        if trade_type == "sell":
            # Get positions for this symbol
            symbol_positions = self.portfolio.get_positions_for_symbol(symbol)
            if not symbol_positions:
                info["reject_reason"] = "no_position"
                return False, info
            position = symbol_positions[0]  # Close first/oldest position

            shares_to_close = float(ctx.get("shares", 0.0))
            if shares_to_close <= 0:
                info["reject_reason"] = "zero_shares"
                return False, info

            shares_to_close = min(shares_to_close, position.shares)
            commission = (shares_to_close * current_price) * portfolio_cfg.commission_rate
            slippage_cost = (shares_to_close * current_price) * (portfolio_cfg.slippage_bps / 10_000)
            extra_cost = (shares_to_close * current_price) * self.transaction_cost

            entry_size = "medium"
            pyramid_count = 0
            partial_exit_taken = False
            if hasattr(position, "metadata") and position.metadata:
                entry_size = position.metadata.get("entry_size", "medium")
                pyramid_count = int(position.metadata.get("pyramid_count", 0))
                partial_exit_taken = bool(position.metadata.get("partial_exit_taken", False))

            success, trade = self.portfolio.close_position(
                symbol=symbol,
                shares_to_close=shares_to_close,
                exit_price=current_price,
                exit_time=current_time,
                exit_step=self.current_step,
                exit_reason="agent_continuous_close",
                commission=commission + extra_cost,
                slippage=slippage_cost,
            )

            if not success or trade is None:
                info["reject_reason"] = "portfolio_rejected"
                return False, info

            close_all = bool(ctx.get("close_all", False) or shares_to_close >= position.shares)
            trade.setdefault("trigger", "agent_continuous_close")
            trade["entry_size"] = entry_size
            trade["pyramid_count"] = pyramid_count
            trade["forced_exit"] = False
            if close_all:
                trade["closed"] = True
                trade["exit_type"] = "staged" if partial_exit_taken else "full"
            else:
                trade["exit_type"] = "partial"

            info["trade"] = trade
            info["shares_sold"] = shares_to_close

            # Check if position still exists after close
            remaining_positions = self.portfolio.get_positions_for_symbol(symbol)
            if remaining_positions:
                remaining = remaining_positions[0]  # Get the remaining/updated position
                if hasattr(remaining, "metadata"):
                    remaining.metadata = getattr(remaining, "metadata", {}) or {}
                    remaining.metadata["entry_size"] = entry_size
                    remaining.metadata["pyramid_count"] = pyramid_count
                    if not close_all:
                        remaining.metadata["partial_exit_taken"] = True
                else:
                    remaining.metadata["partial_exit_taken"] = False

            return True, info

        return super()._execute_action(action)


class HybridActionEnvironment(ContinuousTradingEnvironment):
    """Environment adapter that accepts either discrete or continuous actions."""

    def __init__(self, config: TradingConfig, seed: Optional[int] = None) -> None:
        super().__init__(config, seed)
        self.discrete_action_space = spaces.Discrete(len(TradeAction))

    def step(self, action: Any) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if isinstance(action, (int, np.integer)):
            original_space = self.action_space
            self.action_space = self.discrete_action_space
            try:
                return TradingEnvironment.step(self, int(action))
            finally:
                self.action_space = original_space
        if isinstance(action, np.ndarray) and action.ndim == 0:
            original_space = self.action_space
            self.action_space = self.discrete_action_space
            try:
                return TradingEnvironment.step(self, int(action.item()))
            finally:
                self.action_space = original_space
        return super().step(action)
