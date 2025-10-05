"""Gymnasium-compatible trading environment for RL agent training."""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding

from .feature_extractor import FeatureConfig, FeatureExtractor
from .regime_indicators import RegimeIndicators
from .reward_shaper import RewardConfig, RewardShaper

try:  # pragma: no cover - import guard
    from scripts.sl_checkpoint_utils import load_sl_checkpoint  # type: ignore
except ImportError:  # pragma: no cover - graceful degradation when utilities unavailable
    load_sl_checkpoint = None
logger = logging.getLogger(__name__)

# Configuration & Data Structures

@dataclass
class TradingConfig:
    """Configuration for trading environment"""

    symbol: str
    data_path: Path
    sl_checkpoints: Dict[str, Path]

    initial_capital: float = 100_000.0
    max_position_size: float = 0.10  # 10% of capital
    max_positions: int = 1  # Single symbol = 1 max

    commission_rate: float = 0.001  # 0.1%
    slippage_bps: float = 5.0  # 5 basis points

    stop_loss: float = 0.03  # 3%
    take_profit: float = 0.015  # 1.5%
    max_hold_hours: int = 24

    lookback_window: int = 24
    episode_length: int = 1000  # Max steps per episode

    train_start: Optional[str] = None  # '2023-10-02'
    train_end: Optional[str] = None  # '2025-05-01'
    val_start: Optional[str] = None  # '2025-05-01'
    val_end: Optional[str] = None  # '2025-08-01'

    reward_scaling: float = 1.0
    equity_reward_weight: float = 0.6
    drawdown_penalty_weight: float = 0.2
    action_regularization_weight: float = 0.1
    risk_overshoot_penalty: float = 0.1
    bankruptcy_penalty: float = 10.0

    min_cash_reserve: float = 100.0
    epsilon: float = 1e-8

    log_trades: bool = True
    log_level: int = logging.INFO

    reward_config: Optional[RewardConfig] = None

@dataclass
class PositionState:
    """Represents an open trading position."""

    shares: float
    entry_price: float
    entry_step: int
    cost_basis: float
    entry_equity: float

@dataclass
class TradeEvent:
    """Record describing a trade executed within the environment."""

    step: int
    action: str
    price: float
    shares: float
    timestamp: pd.Timestamp
    proceeds: Optional[float] = None
    cost: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    holding_hours: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RewardBreakdown:
    """Container for reward components (tracked for diagnostics)."""

    equity: float = 0.0
    drawdown: float = 0.0
    action: float = 0.0
    risk: float = 0.0
    total: float = 0.0
class TradeAction(IntEnum):
    """Enumeration matching discrete action space semantics."""

    HOLD = 0
    BUY_SMALL = 1
    BUY_MEDIUM = 2
    BUY_LARGE = 3
    SELL_PARTIAL = 4
    SELL_ALL = 5
    ADD_POSITION = 6

# Utility helpers

def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Avoid division by zero returning ``default`` when denominator is ~0."""

    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator

def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))

def _compute_drawdown(equity_curve: Iterable[float]) -> Tuple[float, float]:
    """Compute max drawdown and current drawdown from equity sequence."""

    peak = -math.inf
    max_dd = 0.0
    current_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
            current_dd = 0.0
            continue
        dd = _safe_divide(peak - value, peak, 0.0)
        current_dd = dd
        if dd > max_dd:
            max_dd = dd
    if peak == -math.inf:
        peak = 0.0
    return max_dd, current_dd

def _timestamp_str(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "NaT"
    return ts.isoformat()

# Trading Environment

class TradingEnvironment(gym.Env):
    """Production-grade single-symbol trading environment for Gymnasium."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 1,
    }

    DEFAULT_FEATURE_COLUMNS: Tuple[str, ...] = (
        "open", "high", "low", "close", "volume", "vwap",
        "SMA_10", "SMA_20", "MACD", "MACD_signal", "MACD_hist",
        "RSI_14", "Stochastic_K", "Stochastic_D", "ADX_14", "ATR_14",
        "BB_bandwidth", "OBV", "Volume_SMA_20", "Return_1h",
        "sentiment_score_hourly_ffill", "DayOfWeek_sin", "DayOfWeek_cos",
    )

    REGIME_FEATURE_CANDIDATES: Tuple[str, ...] = (
        "Return_4h", "Return_12h", "ATR_normalized", "volatility_4h",
        "volume_zscore", "trend_strength", "momentum_score", "liquidity_score",
        "macro_beta", "sentiment_trend", "drawdown_market", "breadth_ratio",
    )

    SL_MODEL_ORDER: Tuple[str, ...] = ("mlp", "lstm", "gru")

    def __init__(self, config: TradingConfig, seed: Optional[int] = None):
        """Initialize environment with configuration"""

        super().__init__()
        self.config = config
        self.render_mode = None
        self._np_random = None

        # Configure logging early
        if not logger.handlers:
            logging.basicConfig(level=config.log_level)
        else:
            logger.setLevel(config.log_level)

        self.sl_models: Dict[str, Any] = {}
        self.feature_extractor: FeatureExtractor | None = None
        self.regime_indicators: RegimeIndicators | None = None
        self.feature_cols: List[str] = []
        self.regime_feature_names: List[str] = []

        reward_config = config.reward_config or RewardConfig()
        self.reward_shaper = RewardShaper(reward_config)
        self.peak_equity = config.initial_capital
        logger.info(
            "TradingEnvironment initialized for %s with reward weights: PnL=%.2f, Cost=%.2f",
            config.symbol,
            reward_config.pnl_weight,
            reward_config.transaction_cost_weight,
        )

        self._load_data()
        self._load_sl_models()
        self._define_spaces()

        # State variables
        self.current_step: int = 0
        self.episode_step: int = 0
        self.cash: float = config.initial_capital
        self.position: Optional[PositionState] = None
        self.trades: List[TradeEvent] = []
        self.equity_curve: List[float] = []
        self.max_equity: float = config.initial_capital
        self.min_equity: float = config.initial_capital
        self.max_drawdown: float = 0.0
        self.current_drawdown: float = 0.0
        self.last_action: TradeAction = TradeAction.HOLD
        self._last_reward: RewardBreakdown = RewardBreakdown()
        self._portfolio_cache: Optional[np.ndarray] = None

        if seed is not None:
            self.seed(seed)
        else:
            self.seed()

    # Gymnasium API

    def seed(self, seed: Optional[int] = None) -> List[int]:
        self._np_random, seed_used = seeding.np_random(seed)
        return [seed_used]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment to start new episode

        Args:
            seed: Random seed
            options: Additional options (e.g., start_idx for testing)

        Returns:
            observation: Initial observation
            info: Additional information
        """

        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.cash = float(self.config.initial_capital)
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.max_equity = self.config.initial_capital
        self.min_equity = self.config.initial_capital
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.episode_step = 0
        self.last_action = TradeAction.HOLD
        self._last_reward = RewardBreakdown()
        self._portfolio_cache = None
        self.reward_shaper.reset_episode()
        self.peak_equity = self.config.initial_capital

        if options and "start_idx" in options:
            start_idx = int(options["start_idx"])
        else:
            valid_start = self.config.lookback_window
            valid_end = len(self.data) - self.config.episode_length - 1
            if valid_end <= valid_start:
                raise ValueError("Not enough data for full episode")
            start_idx = int(self._np_random.integers(valid_start, valid_end))

        self.current_step = start_idx
        observation = self._get_observation()
        info = self._get_info()
        logger.debug("Environment reset at step %s", self.current_step)
        return observation, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step

        Args:
            action: Action index (0-6)

        Returns:
            observation: Next state
            reward: Reward for this transition
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut off
            info: Additional information
        """

        if not self.action_space.contains(int(action)):
            raise gym.error.InvalidAction(f"Action {action} outside action space")

        prev_equity = self._calculate_equity()
        self.last_action = TradeAction(int(action))
        action_executed, action_info = self._execute_action(self.last_action)
        position_close_event = self._update_position()
        position_closed = bool(action_info.get("closed")) or bool(position_close_event)

        self.current_step += 1
        self.episode_step += 1

        current_equity = self._calculate_equity()
        self.equity_curve.append(current_equity)
        self.max_equity = max(self.max_equity, current_equity)
        self.min_equity = min(self.min_equity, current_equity)
        self.max_drawdown, self.current_drawdown = _compute_drawdown(self.equity_curve)

        reward = self._compute_reward(
            action=self.last_action,
            action_executed=action_executed,
            prev_equity=prev_equity,
            current_equity=current_equity,
            position_closed=position_closed,
        )

        terminated = False
        truncated = False

        if self.episode_step >= self.config.episode_length:
            truncated = True
        if self.current_step >= len(self.data) - 1:
            terminated = True
        if current_equity <= self.config.initial_capital * 0.5:
            terminated = True
            reward -= self.config.bankruptcy_penalty
            self._last_reward.risk -= self.config.bankruptcy_penalty
            logger.warning(
                "Bankruptcy condition triggered: equity %.2f <= %.2f",
                current_equity,
                self.config.initial_capital * 0.5,
            )

        observation = self._get_observation()
        info = self._get_info()
        info.update(
            {
                "action_executed": bool(action_executed),
                "action_info": action_info,
                "position_closed": position_close_event or (action_info if action_info.get("closed") else None),
                "equity": current_equity,
                "reward_breakdown": self._last_reward.__dict__.copy(),
            }
        )

        if terminated or truncated:
            info["episode"] = {
                "r": float(sum(self.equity_curve) - len(self.equity_curve) * self.config.initial_capital),
                "l": int(self.episode_step),
                "equity_final": float(current_equity),
                "max_drawdown": float(self.max_drawdown),
            }

        if self.render_mode == "human":
            self._render_human(info)

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:  # pragma: no cover - interactive IO
        if self.render_mode == "human":
            info = self._get_info()
            self._render_human(info)
            return None
        elif self.render_mode == "rgb_array":
            # Placeholder for potential visualization hook
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return None

    def close(self) -> None:
        self.position = None
        self.trades.clear()

    # Data Loading and Initialization

    def _load_data(self) -> None:
        """Load historical data and initialize feature extraction"""

        logger.info(f"Loading data for {self.config.symbol}")

        if not self.config.data_path.exists():
            raise FileNotFoundError(f"Data path {self.config.data_path} does not exist")

        data = pd.read_parquet(self.config.data_path)
        if "timestamp" not in data.columns:
            raise ValueError("Expected 'timestamp' column in data")

        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp").reset_index(drop=True)

        if self.config.train_start and self.config.train_end:
            start = pd.Timestamp(self.config.train_start)
            end = pd.Timestamp(self.config.train_end)
            mask = (
                (data["timestamp"] >= start)
                & (data["timestamp"] <= end)
            )
            data = data[mask].reset_index(drop=True)

        if data.empty:
            raise ValueError("Loaded dataset is empty after filtering; check configuration")

        feature_config = FeatureConfig(
            normalize_method="zscore",
            normalize_window=252,
        )
        self.feature_extractor = FeatureExtractor(
            data,
            feature_config,
            lookback_window=self.config.lookback_window,
            sl_models=self.sl_models,
        )

        self.regime_indicators = RegimeIndicators(data)

        self.data = data
        self.feature_cols = self.feature_extractor.get_feature_names()
        self.regime_feature_names = self.regime_indicators.get_regime_names()

        logger.info(
            "Loaded %d timesteps for %s, %d features",
            len(data),
            self.config.symbol,
            self.feature_extractor.get_feature_count(),
        )

    def _load_sl_models(self) -> None:
        """Load pre-trained SL models"""

        if self.sl_models is None:
            self.sl_models = {}
        else:
            self.sl_models.clear()
        if load_sl_checkpoint is None:
            logger.warning("SL checkpoint utilities unavailable; using random predictions")
            if self.feature_extractor is not None:
                self.feature_extractor.sl_models = {}
            return

        for name, checkpoint_path in self.config.sl_checkpoints.items():
            try:
                model_bundle = load_sl_checkpoint(checkpoint_path)
            except Exception as exc:  # pragma: no cover - logging side effect
                logger.warning("Failed to load checkpoint %s: %s", name, exc)
                continue

            if model_bundle is None:
                logger.warning("Checkpoint loader returned None for %s", name)
                continue

            self.sl_models[name.lower()] = model_bundle
            logger.info("Loaded SL model: %s", name)

        if len(self.sl_models) == 0:
            logger.warning("No SL models loaded - using random predictions")
        if self.feature_extractor is not None:
            self.feature_extractor.sl_models = self.sl_models

    def _define_spaces(self) -> None:
        """Define observation and action spaces"""

        self.action_space = spaces.Discrete(len(TradeAction))

        lookback = self.config.lookback_window
        num_features = self.feature_extractor.get_feature_count() if self.feature_extractor else 0
        regime_dim = len(self.regime_feature_names) if self.regime_feature_names else 10

        self.observation_space = spaces.Dict(
            {
                "technical": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(lookback, num_features),
                    dtype=np.float32,
                ),
                "sl_probs": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(len(self.SL_MODEL_ORDER),),
                    dtype=np.float32,
                ),
                "position": spaces.Box(
                    low=np.array([0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32),
                    high=np.array([
                        1.0, np.finfo(np.float32).max, 1.0,
                        float(self.config.max_hold_hours) + 1.0,
                        float(self.config.max_position_size) + 0.01,
                    ], dtype=np.float32),
                    dtype=np.float32,
                ),
                "portfolio": spaces.Box(
                    low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0], dtype=np.float32),
                    high=np.array([
                        np.finfo(np.float32).max, np.finfo(np.float32).max, 1.0,
                        float(self.config.max_positions), 1.0, 1.0, 1.0,
                        np.finfo(np.float32).max,
                    ], dtype=np.float32),
                    dtype=np.float32,
                ),
                "regime": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(regime_dim,),
                    dtype=np.float32,
                ),
            }
        )

    # Observation helpers

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current observation using feature extractor

        Returns:
            Dict observation matching observation_space
        """

        if self.feature_extractor is None or self.regime_indicators is None:
            raise RuntimeError("Feature pipeline not initialized")

        self._portfolio_cache = None

        technical_features = self.feature_extractor.extract_window(
            self.current_step,
            normalize=True,
        )

        sl_probs = self.feature_extractor.get_sl_predictions(
            self.current_step,
            window=technical_features,
        )

        total = float(sl_probs.sum())
        if total <= 0:
            sl_probs = np.full_like(sl_probs, 1.0 / max(len(sl_probs), 1))
        else:
            sl_probs = (sl_probs / total).astype(np.float32, copy=False)

        position_state = self._get_position_state()
        portfolio_state = self._get_portfolio_state()
        regime_state = self.regime_indicators.get_regime_vector(self.current_step)

        return {
            "technical": technical_features,
            "sl_probs": sl_probs,
            "position": position_state,
            "portfolio": portfolio_state,
            "regime": regime_state,
        }

    def _get_position_state(self) -> np.ndarray:
        if self.position is None:
            state = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return state

        current_price = float(self.data.loc[self.current_step, "close"])
        position_value = self.position.shares * current_price
        unrealized_pnl = position_value - self.position.cost_basis
        duration = float(self.current_step - self.position.entry_step)
        size_pct = _safe_divide(position_value, self._calculate_equity(), 0.0)

        state = np.array(
            [
                1.0,
                float(self.position.entry_price),
                float(_safe_divide(unrealized_pnl, max(self.position.cost_basis, self.config.epsilon), 0.0)),
                duration,
                size_pct,
            ],
            dtype=np.float32,
        )
        return state

    def _get_portfolio_state(self) -> np.ndarray:
        if self._portfolio_cache is not None:
            return self._portfolio_cache

        equity = self._calculate_equity()
        position_count = 1 if self.position is not None else 0
        deployed_pct = _safe_divide(equity - self.cash, equity, 0.0)
        realized_pnl = self._calculate_realized_pnl()
        pct_change = _safe_divide(equity - self.config.initial_capital, self.config.initial_capital, 0.0)
        max_dd = float(self.max_drawdown)
        current_dd = float(self.current_drawdown)

        state = np.array(
            [
                float(equity),
                float(self.cash),
                float(deployed_pct),
                float(position_count),
                float(_clip01(pct_change + 0.5)),
                float(max_dd),
                float(current_dd),
                float(realized_pnl),
            ],
            dtype=np.float32,
        )
        self._portfolio_cache = state
        return state

    # Trading Mechanics

    def _execute_action(self, action: TradeAction) -> Tuple[bool, Dict[str, Any]]:
        current_price = float(self.data.loc[self.current_step, "close"])
        info: Dict[str, Any] = {"action_name": action.name, "price": current_price}

        # Action 0: HOLD
        if action == TradeAction.HOLD:
            return True, info

        # Buy actions
        if action in (TradeAction.BUY_SMALL, TradeAction.BUY_MEDIUM, TradeAction.BUY_LARGE):
            if self.position is not None:
                info["reject_reason"] = "position_exists"
                return False, info

            sizing_map = {
                TradeAction.BUY_SMALL: 0.025,
                TradeAction.BUY_MEDIUM: 0.06,
                TradeAction.BUY_LARGE: 0.09,
            }
            target_pct = sizing_map[action]
            target_pct = min(target_pct, self.config.max_position_size)
            target_value = self._calculate_equity() * target_pct

            if target_value > self.cash - self.config.min_cash_reserve:
                info["reject_reason"] = "insufficient_cash"
                return False, info

            shares = target_value / current_price
            if shares <= 0:
                info["reject_reason"] = "zero_shares"
                return False, info

            entry_price = current_price * (1 + self.config.slippage_bps / 10_000)
            commission = target_value * self.config.commission_rate
            slippage_cost = target_value * (self.config.slippage_bps / 10_000)
            total_cost = target_value + commission + slippage_cost

            if total_cost > self.cash:
                info["reject_reason"] = "insufficient_cash"
                return False, info

            self.position = PositionState(
                shares=shares,
                entry_price=entry_price,
                entry_step=self.current_step,
                cost_basis=total_cost,
                entry_equity=self._calculate_equity(),
            )
            self.cash -= total_cost

            trade_event = TradeEvent(
                step=self.current_step,
                action="BUY",
                price=entry_price,
                shares=shares,
                cost=total_cost,
                timestamp=self.data.loc[self.current_step, "timestamp"],
                metadata={"target_pct": float(target_pct)},
            )
            self._record_trade(trade_event)
            info.update({"entry_price": entry_price, "shares": shares, "cost": total_cost})
            return True, info

        # SELL_PARTIAL
        if action == TradeAction.SELL_PARTIAL:
            if self.position is None:
                info["reject_reason"] = "no_position"
                return False, info

            shares_to_sell = self.position.shares * 0.5
            if shares_to_sell <= 0:
                info["reject_reason"] = "zero_shares"
                return False, info

            current_value = shares_to_sell * current_price
            commission, slippage_cost, proceeds = self._calculate_transaction_costs(current_value)

            pnl = proceeds - self.position.cost_basis * 0.5
            new_cost_basis = self.position.cost_basis * 0.5

            self.position.shares -= shares_to_sell
            self.position.cost_basis = new_cost_basis
            self.cash += proceeds

            trade_event = TradeEvent(
                step=self.current_step,
                action="SELL_PARTIAL",
                price=current_price,
                shares=shares_to_sell,
                proceeds=proceeds,
                pnl=pnl,
                timestamp=self.data.loc[self.current_step, "timestamp"],
                metadata={"commission": commission, "slippage_cost": slippage_cost},
            )
            self._record_trade(trade_event)
            info.update({"shares_sold": shares_to_sell, "pnl": pnl})
            return True, info

        # SELL_ALL
        if action == TradeAction.SELL_ALL:
            if self.position is None:
                info["reject_reason"] = "no_position"
                return False, info

            closed_info = self._close_position("SELL_ALL", current_price)
            info.update(closed_info)
            return True, info

        # ADD_POSITION
        if action == TradeAction.ADD_POSITION:
            if self.position is None:
                info["reject_reason"] = "no_position"
                return False, info

            current_pnl_pct = _safe_divide(
                current_price - self.position.entry_price,
                self.position.entry_price,
                0.0,
            )
            if current_pnl_pct <= 0:
                info["reject_reason"] = "position_not_profitable"
                return False, info

            add_value = self._calculate_equity() * 0.02
            if add_value > self.cash - self.config.min_cash_reserve:
                info["reject_reason"] = "insufficient_cash"
                return False, info

            shares_to_add = add_value / current_price
            commission, slippage_cost, total_cost = self._calculate_transaction_costs(add_value)

            total_shares = self.position.shares + shares_to_add
            weighted_entry = (
                (self.position.shares * self.position.entry_price + shares_to_add * current_price)
                / total_shares
            )

            self.position.shares = total_shares
            self.position.entry_price = weighted_entry
            self.position.cost_basis += total_cost
            self.cash -= total_cost

            trade_event = TradeEvent(
                step=self.current_step,
                action="ADD_POSITION",
                price=current_price,
                shares=shares_to_add,
                cost=total_cost,
                timestamp=self.data.loc[self.current_step, "timestamp"],
                metadata={"commission": commission, "slippage_cost": slippage_cost},
            )
            self._record_trade(trade_event)

            info.update({"shares_added": shares_to_add, "new_avg_entry": weighted_entry})
            return True, info

        info["reject_reason"] = "unknown_action"
        return False, info

    def _calculate_transaction_costs(self, notionals: float) -> Tuple[float, float, float]:
        commission = notionals * self.config.commission_rate
        slippage_cost = notionals * (self.config.slippage_bps / 10_000)
        proceeds = notionals - commission - slippage_cost
        return commission, slippage_cost, proceeds

    def _close_position(self, trigger: str, exit_price: float) -> Dict[str, Any]:
        if self.position is None:
            return {"closed": False}

        shares = self.position.shares
        current_value = shares * exit_price
        commission, slippage_cost, proceeds = self._calculate_transaction_costs(current_value)
        pnl = proceeds - self.position.cost_basis
        pnl_pct = _safe_divide(pnl, self.position.cost_basis, 0.0)
        holding_hours = self.current_step - self.position.entry_step

        self.cash += proceeds
        trade_event = TradeEvent(
            step=self.current_step,
            action=trigger,
            price=exit_price,
            shares=shares,
            proceeds=proceeds,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_hours=holding_hours,
            timestamp=self.data.loc[self.current_step, "timestamp"],
            metadata={"commission": commission, "slippage_cost": slippage_cost},
        )
        self._record_trade(trade_event)
        self.position = None

        info = {
            "closed": True,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "holding_hours": holding_hours,
            "trigger": trigger,
        }
        return info

    def _update_position(self) -> Optional[Dict[str, Any]]:
        if self.position is None:
            return None

        current_price = float(self.data.loc[self.current_step, "close"])
        entry_price = self.position.entry_price
        price_change_pct = _safe_divide(current_price - entry_price, entry_price, 0.0)

        if price_change_pct <= -self.config.stop_loss:
            logger.debug("Stop loss triggered at step %s", self.current_step)
            return self._close_position("STOP_LOSS", current_price)

        if price_change_pct >= self.config.take_profit:
            logger.debug("Take profit triggered at step %s", self.current_step)
            return self._close_position("TAKE_PROFIT", current_price)

        duration = self.current_step - self.position.entry_step
        if duration >= self.config.max_hold_hours:
            logger.debug("Max hold duration reached at step %s", self.current_step)
            return self._close_position("MAX_HOLD", current_price)

        equity = self._calculate_equity()
        position_value = self.position.shares * current_price
        size_pct = _safe_divide(position_value, equity, 0.0)
        if size_pct > self.config.max_position_size + 1e-4:
            logger.debug("Position size exceeded limit at step %s", self.current_step)
            return self._close_position("SIZE_LIMIT", current_price)

        return None

    def _compute_reward(
        self,
        action: TradeAction,
        action_executed: bool,
        prev_equity: float,
        current_equity: float,
        position_closed: bool,
    ) -> float:
        position_info = None
        if self.position is not None:
            position_info = {
                "is_open": True,
                "entry_price": float(self.position.entry_price),
                "entry_step": int(self.position.entry_step),
                "duration": self.current_step - self.position.entry_step,
                "shares": float(self.position.shares),
            }

        trade_info = None
        if position_closed and self.trades:
            last_trade = self.trades[-1]
            trade_info = {
                "pnl_pct": float(last_trade.pnl_pct or 0.0),
                "holding_hours": float(last_trade.holding_hours or 0.0),
                "action": last_trade.action,
            }

        portfolio_state = self._get_portfolio_state_dict()

        total_reward, components = self.reward_shaper.compute_reward(
            action=int(action),
            action_executed=action_executed,
            prev_equity=prev_equity,
            current_equity=current_equity,
            position_info=position_info,
            trade_info=trade_info,
            portfolio_state=portfolio_state,
        )

        self._last_reward.total = total_reward
        self._last_reward.equity = components.get("pnl", 0.0)
        self._last_reward.drawdown = components.get("drawdown", 0.0)
        self._last_reward.action = (
            components.get("transaction_cost", 0.0) + components.get("time_efficiency", 0.0)
        )
        self._last_reward.risk = components.get("sharpe", 0.0)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Step %s: Total reward=%.3f, Components: PnL=%.3f, Cost=%.3f, Time=%.3f",
                self.current_step,
                total_reward,
                components.get("pnl", 0.0),
                components.get("transaction_cost", 0.0),
                components.get("time_efficiency", 0.0),
            )

        return total_reward

    def _get_portfolio_state_dict(self) -> Dict[str, float]:
        current_equity = self._calculate_equity()
        self.peak_equity = max(getattr(self, "peak_equity", self.config.initial_capital), current_equity)

        deployed = 0.0
        if self.position is not None:
            deployed = float(self.position.shares) * float(self.data.loc[self.current_step, "close"])

        deployed_pct = deployed / current_equity if current_equity > 0 else 0.0

        sharpe = 0.0
        if len(self.equity_curve) > 20:
            returns = np.diff(self.equity_curve[-20:]) / self.equity_curve[-21:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

        return {
            "equity": current_equity,
            "peak_equity": self.peak_equity,
            "deployed_pct": deployed_pct,
            "sharpe_ratio": sharpe,
            "num_trades": len(self.trades),
        }

    # Portfolio Accounting & Diagnostics

    def _calculate_equity(self) -> float:
        equity = self.cash
        if self.position is not None:
            current_price = float(self.data.loc[self.current_step, "close"])
            equity += self.position.shares * current_price
        return float(equity)

    def _calculate_realized_pnl(self) -> float:
        realized = 0.0
        for trade in self.trades:
            if trade.pnl is not None:
                realized += trade.pnl
        return realized

    def _get_info(self) -> Dict[str, float]:
        timestamp = self.data.loc[self.current_step, "timestamp"]
        info = {
            "step": self.current_step,
            "episode_step": self.episode_step,
            "timestamp": _timestamp_str(timestamp),
            "cash": float(self.cash),
            "equity": float(self._calculate_equity()),
            "max_equity": float(self.max_equity),
            "min_equity": float(self.min_equity),
            "max_drawdown": float(self.max_drawdown),
            "current_drawdown": float(self.current_drawdown),
            "position": None,
            "last_action": self.last_action.name,
        }
        if self.position is not None:
            info["position"] = {
                "shares": float(self.position.shares),
                "entry_price": float(self.position.entry_price),
                "entry_step": int(self.position.entry_step),
                "cost_basis": float(self.position.cost_basis),
            }
        if self.config.log_trades:
            info["trades"] = [trade.__dict__ for trade in self.trades[-5:]]

        if self.episode_step > 0:
            reward_stats = self.reward_shaper.get_episode_stats()
            info["reward_stats"] = reward_stats

            if self.episode_step % 100 == 0:
                info["reward_contributions"] = self.reward_shaper.get_component_contributions()

        return info

    def _record_trade(self, trade_event: TradeEvent) -> None:
        if not self.config.log_trades:
            return
        self.trades.append(trade_event)
        logger.debug(
            "Trade executed: %s shares=%.4f price=%.4f pnl=%s",
            trade_event.action,
            trade_event.shares,
            trade_event.price,
            trade_event.pnl,
        )

    # Rendering (human)

    def _render_human(self, info: Dict[str, Any]) -> None:  # pragma: no cover - console I/O
        position_info = info.get("position") or {}
        print(
            f"Step {info['episode_step']:04d} (idx {info['step']:06d}) | "
            f"Time {info['timestamp']} | Action {info['last_action']} | "
            f"Equity {info['equity']:.2f} | Cash {info['cash']:.2f} | "
            f"Drawdown {info['current_drawdown']:.2%} | Position {position_info}",
        )
        if info.get("action_info"):
            print(f"  Action info: {info['action_info']}")
        if info.get("reward_breakdown"):
            rb = info["reward_breakdown"]
            print(
                "  Reward -> total: {total:.6f}, equity: {equity:.6f}, drawdown: {drawdown:.6f}, "
                "action: {action:.6f}, risk: {risk:.6f}".format(**rb)
            )

__all__ = [
    "TradingConfig",
    "TradingEnvironment",
    "TradeAction",
    "PositionState",
    "TradeEvent",
    "RewardBreakdown",
]
