"""Gymnasium-compatible trading environment for RL agent training."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding

from .feature_extractor import FeatureConfig, FeatureExtractor
from .portfolio_manager import PortfolioConfig, PortfolioManager
from .regime_indicators import RegimeIndicators
from .reward_shaper import RewardConfig, RewardShaper

COLUMN_ALIASES: Dict[str, str] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "VWAP": "vwap",
    "MACD_line": "MACD",
    "Stoch_K": "Stochastic_K",
    "Stoch_D": "Stochastic_D",
}

try:  # pragma: no cover - import guard
    from scripts.sl_checkpoint_utils import load_sl_checkpoint  # type: ignore
except ImportError:  # pragma: no cover - graceful degradation when utilities unavailable
    load_sl_checkpoint = None

logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """Configuration for trading environment."""

    symbol: str
    data_path: Path
    sl_checkpoints: Dict[str, Path]

    # Preferred configuration path
    portfolio_config: Optional[PortfolioConfig] = None

    # Legacy parameters (used when portfolio_config is not supplied)
    initial_capital: float = 100_000.0
    commission_rate: float = 0.001
    slippage_bps: float = 5.0
    stop_loss: float = 0.02
    take_profit: float = 0.025
    max_hold_hours: int = 8

    # Environment settings
    lookback_window: int = 24
    episode_length: int = 1000
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    val_start: Optional[str] = None
    val_end: Optional[str] = None

    # Reward configuration
    reward_config: Optional[RewardConfig] = None

    # Logging / diagnostics
    log_trades: bool = True
    log_level: int = logging.INFO

    def get_portfolio_config(self) -> PortfolioConfig:
        """Materialize a :class:`PortfolioConfig` from legacy settings."""

        if self.portfolio_config is not None:
            return self.portfolio_config

        config = PortfolioConfig(
            initial_capital=self.initial_capital,
            commission_rate=self.commission_rate,
            slippage_bps=self.slippage_bps,
            max_position_size_pct=0.10,
            max_total_exposure_pct=1.0,
            max_positions=1,
        )
        return config


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


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Avoid division by zero returning ``default`` when denominator is ~0."""

    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _timestamp_str(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "NaT"
    return ts.isoformat()

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
        super().__init__()
        self.config = config
        self.render_mode = None
        self._np_random = None

        if not logger.handlers:
            logging.basicConfig(level=config.log_level)
        else:
            logger.setLevel(config.log_level)

        self.sl_models: Dict[str, Any] = {}
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.regime_indicators: Optional[RegimeIndicators] = None
        self.feature_cols: List[str] = []
        self.regime_feature_names: List[str] = []

        reward_config = config.reward_config or RewardConfig()
        self.reward_shaper = RewardShaper(reward_config)
        self.portfolio = PortfolioManager(config.get_portfolio_config())

        self._load_data()
        self._load_sl_models()
        self._define_spaces()

        self.current_step: int = 0
        self.episode_step: int = 0
        self.last_action: TradeAction = TradeAction.HOLD
        self.equity_curve: List[float] = []
        self._last_reward: RewardBreakdown = RewardBreakdown()
        self._last_closed_trade: Optional[Dict[str, Any]] = None

        if seed is not None:
            self.seed(seed)
        else:
            self.seed()

        logger.info(
            "TradingEnvironment initialized for %s with PortfolioManager (capital=$%s)",
            config.symbol,
            f"{self.portfolio.config.initial_capital:,.0f}",
        )

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

        self.portfolio.reset()
        self.reward_shaper.reset_episode()
        self.episode_step = 0
        self.last_action = TradeAction.HOLD
        self._last_reward = RewardBreakdown()
        self._last_closed_trade = None
        self.equity_curve = []

        if options and "start_idx" in options:
            start_idx = int(options["start_idx"])
        else:
            valid_start = self.config.lookback_window
            valid_end = len(self.data) - self.config.episode_length - 1
            if valid_end <= valid_start:
                raise ValueError("Not enough data for full episode")
            start_idx = int(self._np_random.integers(valid_start, valid_end))

        self.current_step = start_idx
        self._mark_to_market_current()
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

        prev_equity = self.portfolio.get_equity()
        self.last_action = TradeAction(int(action))
        action_executed, action_info = self._execute_action(int(action))
        forced_trades = self._update_position()
        if forced_trades:
            action_info.setdefault("forced_trades", forced_trades)

        position_closed_info: Optional[Dict[str, Any]] = None
        if action_info.get("trade") is not None:
            position_closed_info = action_info["trade"]
        if forced_trades:
            position_closed_info = forced_trades[0]

        self.current_step += 1
        self.episode_step += 1
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1

        self._mark_to_market_current()
        current_equity = self.portfolio.get_equity()

        reward = self._compute_reward(
            action=self.last_action,
            action_executed=action_executed,
            prev_equity=prev_equity,
            current_equity=current_equity,
            position_closed=position_closed_info is not None,
        )

        terminated = False
        truncated = False

        if self.episode_step >= self.config.episode_length:
            truncated = True
        if self.current_step >= len(self.data) - 1:
            terminated = True
        if current_equity <= self.portfolio.config.initial_capital * 0.5:
            terminated = True
            logger.warning(
                "Bankruptcy condition triggered: equity %.2f <= %.2f",
                current_equity,
                self.portfolio.config.initial_capital * 0.5,
            )

        observation = self._get_observation()
        info = self._get_info()
        info.update(
            {
                "action_executed": bool(action_executed),
                "action_info": action_info,
                "position_closed": position_closed_info,
                "equity": current_equity,
                "reward_breakdown": self._last_reward.__dict__.copy(),
            }
        )
        self._last_closed_trade = position_closed_info

        if terminated or truncated:
            metrics = self.portfolio.get_portfolio_metrics()
            info["episode"] = {
                "r": float(metrics["total_pnl"]),
                "l": int(self.episode_step),
                "equity_final": float(current_equity),
                "max_drawdown": float(metrics["max_drawdown"]),
            }
            # Persist end-of-episode diagnostics so callers can consume them even if
            # the vectorised environment resets the underlying environment instance
            # immediately after this step returns.
            info["terminal_metrics"] = dict(metrics)
            info["terminal_trades"] = list(self.portfolio.get_closed_positions())
            info["terminal_reward_stats"] = self.reward_shaper.get_episode_stats()
            info["terminal_equity_curve"] = list(self.portfolio.equity_curve)

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
        self.portfolio.reset()

    # Data Loading and Initialization

    def _load_data(self) -> None:
        """Load historical data and initialize feature extraction"""

        logger.info("Loading data for %s", self.config.symbol)

        if not self.config.data_path.exists():
            raise FileNotFoundError(f"Data path {self.config.data_path} does not exist")

        data = pd.read_parquet(self.config.data_path)

        if "timestamp" not in data.columns:
            if isinstance(data.index, pd.DatetimeIndex):
                data = data.reset_index()
                index_name = data.columns[0]
                data = data.rename(columns={index_name: "timestamp"})
            elif "datetime" in data.columns:
                data = data.rename(columns={"datetime": "timestamp"})
            else:
                raise ValueError("Expected 'timestamp' column in data")

        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp").reset_index(drop=True)

        # Normalize base OHLCV columns while preserving raw price levels
        base_aliases = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "VWAP": "vwap",
        }

        for source, dest in base_aliases.items():
            if source not in data.columns:
                continue

            if dest in data.columns:
                original = data[source].astype(float)
                candidate = data[dest].astype(float)
                if not np.allclose(original.fillna(0.0), candidate.fillna(0.0), equal_nan=True):
                    alt_name = f"{dest}_normalized"
                    suffix = 1
                    while alt_name in data.columns:
                        alt_name = f"{dest}_normalized_{suffix}"
                        suffix += 1
                    data = data.rename(columns={dest: alt_name})
            data = data.rename(columns={source: dest})

        # Apply remaining alias normalization without overwriting existing columns
        for source, dest in COLUMN_ALIASES.items():
            if source in base_aliases:
                continue
            if source in data.columns and dest not in data.columns:
                data = data.rename(columns={source: dest})

        # Drop duplicated columns created by alias normalization (keep first/raw values)
        if data.columns.duplicated().any():
            data = data.loc[:, ~data.columns.duplicated(keep="first")]

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
        portfolio_cfg = self.portfolio.config

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
                        1.0,
                        np.finfo(np.float32).max,
                        1.0,
                        float(self.config.max_hold_hours) + 1.0,
                        float(portfolio_cfg.max_position_size_pct) + 0.05,
                    ], dtype=np.float32),
                    dtype=np.float32,
                ),
                "portfolio": spaces.Box(
                    low=np.array([
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -1.0,
                        -10.0,
                        -10.0,
                        -np.finfo(np.float32).max,
                    ], dtype=np.float32),
                    high=np.array([
                        np.finfo(np.float32).max,
                        np.finfo(np.float32).max,
                        1.0,
                        float(portfolio_cfg.max_positions),
                        1.0,
                        np.finfo(np.float32).max,
                        np.finfo(np.float32).max,
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
        position = self.portfolio.positions.get(self.config.symbol)
        if position is None:
            return np.zeros(5, dtype=np.float32)

        equity = self.portfolio.get_equity()
        size_pct = _safe_divide(position.current_value, equity, 0.0)
        return np.array(
            [
                1.0,
                float(position.entry_price),
                float(position.unrealized_pnl_pct),
                float(position.get_holding_period(self.current_step)),
                float(size_pct),
            ],
            dtype=np.float32,
        )

    def _get_portfolio_state(self) -> np.ndarray:
        metrics = self.portfolio.get_portfolio_metrics()
        return np.array(
            [
                float(metrics["equity"]),
                float(metrics["cash"]),
                float(metrics["exposure_pct"]),
                float(metrics["num_positions"]),
                float(metrics["total_return"]),
                float(metrics["sharpe_ratio"]),
                float(metrics["sortino_ratio"]),
                float(metrics["total_pnl"]),
            ],
            dtype=np.float32,
        )

    # Trading Mechanics

    def _execute_action(self, action: int) -> Tuple[bool, Dict[str, Any]]:
        current_price = float(self.data.loc[self.current_step, "close"])
        current_time = self.data.loc[self.current_step, "timestamp"].to_pydatetime()
        info: Dict[str, Any] = {"action_name": self._action_name(action), "price": current_price}
        symbol = self.config.symbol
        portfolio_cfg = self.portfolio.config

        trade_action = TradeAction(int(action))

        if trade_action == TradeAction.HOLD:
            return True, info

        if trade_action in (TradeAction.BUY_SMALL, TradeAction.BUY_MEDIUM, TradeAction.BUY_LARGE):
            if symbol in self.portfolio.positions:
                info["reject_reason"] = "position_exists"
                return False, info

            sizing_map = {TradeAction.BUY_SMALL: 0.025, TradeAction.BUY_MEDIUM: 0.06, TradeAction.BUY_LARGE: 0.09}
            target_pct = sizing_map[trade_action]
            equity = self.portfolio.get_equity()
            target_value = equity * target_pct

            if target_value <= 0:
                info["reject_reason"] = "invalid_target"
                return False, info

            shares = target_value / current_price
            if shares <= 0:
                info["reject_reason"] = "zero_shares"
                return False, info

            entry_price = current_price * (1 + portfolio_cfg.slippage_bps / 10_000)
            commission = target_value * portfolio_cfg.commission_rate
            slippage_cost = target_value * (portfolio_cfg.slippage_bps / 10_000)

            success, position = self.portfolio.open_position(
                symbol=symbol,
                shares=shares,
                entry_price=entry_price,
                entry_time=current_time,
                entry_step=self.current_step,
                commission=commission,
                slippage=slippage_cost,
            )

            if not success or position is None:
                info["reject_reason"] = "portfolio_rejected"
                return False, info

            info.update({"entry_price": entry_price, "shares": shares, "target_pct": target_pct})
            return True, info

        if trade_action == TradeAction.SELL_PARTIAL:
            if symbol not in self.portfolio.positions:
                info["reject_reason"] = "no_position"
                return False, info

            position = self.portfolio.positions[symbol]
            shares_to_sell = position.shares * 0.5
            if shares_to_sell <= 0:
                info["reject_reason"] = "zero_shares"
                return False, info

            commission = (shares_to_sell * current_price) * portfolio_cfg.commission_rate
            slippage_cost = (shares_to_sell * current_price) * (portfolio_cfg.slippage_bps / 10_000)

            success, trade = self.portfolio.close_position(
                symbol=symbol,
                shares_to_close=shares_to_sell,
                exit_price=current_price,
                exit_time=current_time,
                exit_step=self.current_step,
                exit_reason="agent_partial_close",
                commission=commission,
                slippage=slippage_cost,
            )

            if success and trade:
                trade.setdefault("trigger", "agent_partial_close")
                info["trade"] = trade
                info["shares_sold"] = shares_to_sell
            return success, info

        if trade_action == TradeAction.SELL_ALL:
            if symbol not in self.portfolio.positions:
                info["reject_reason"] = "no_position"
                return False, info

            position = self.portfolio.positions[symbol]
            commission = (position.shares * current_price) * portfolio_cfg.commission_rate
            slippage_cost = (position.shares * current_price) * (portfolio_cfg.slippage_bps / 10_000)

            success, trade = self.portfolio.close_position(
                symbol=symbol,
                shares_to_close=None,
                exit_price=current_price,
                exit_time=current_time,
                exit_step=self.current_step,
                exit_reason="agent_full_close",
                commission=commission,
                slippage=slippage_cost,
            )

            if success and trade:
                trade.setdefault("trigger", "agent_full_close")
                trade["closed"] = True
                info["trade"] = trade
            return success, info

        if trade_action == TradeAction.ADD_POSITION:
            info["reject_reason"] = "add_not_implemented"
            return False, info

        info["reject_reason"] = "unknown_action"
        return False, info

    def _update_position(self) -> List[Dict[str, Any]]:
        symbol = self.config.symbol
        current_price = float(self.data.loc[self.current_step, "close"])
        current_time = self.data.loc[self.current_step, "timestamp"].to_pydatetime()

        self._mark_to_market_price(current_price, self.current_step)

        trades: List[Dict[str, Any]] = []
        position = self.portfolio.positions.get(symbol)
        if position is None:
            return trades

        def _close(exit_reason: str) -> Optional[Dict[str, Any]]:
            success, trade = self.portfolio.close_position(
                symbol=symbol,
                shares_to_close=None,
                exit_price=current_price,
                exit_time=current_time,
                exit_step=self.current_step,
                exit_reason=exit_reason,
            )
            if success and trade:
                trade["trigger"] = exit_reason
                trade["closed"] = True
                trades.append(trade)
            return trade if success else None

        if position.unrealized_pnl_pct <= -self.config.stop_loss:
            logger.debug("Stop-loss triggered at step %s", self.current_step)
            _close("stop_loss")
            position = self.portfolio.positions.get(symbol)
            if position is None:
                return trades

        if position.unrealized_pnl_pct >= self.config.take_profit:
            logger.debug("Take-profit triggered at step %s", self.current_step)
            _close("take_profit")
            position = self.portfolio.positions.get(symbol)
            if position is None:
                return trades

        holding_period = position.get_holding_period(self.current_step)
        if holding_period >= self.config.max_hold_hours:
            logger.debug("Max hold time reached at step %s", self.current_step)
            _close("max_hold_time")
            position = self.portfolio.positions.get(symbol)
            if position is None:
                return trades

        risk_trades = self.portfolio.enforce_risk_limits(
            {symbol: current_price},
            current_time,
            self.current_step,
        )
        for trade in risk_trades:
            trade.setdefault("trigger", trade.get("exit_reason", "risk_limit"))
            trade["closed"] = True
        trades.extend(risk_trades)
        return trades

    def _compute_reward(
        self,
        action: TradeAction,
        action_executed: bool,
        prev_equity: float,
        current_equity: float,
        position_closed: bool,
    ) -> float:
        position_info = None
        position = self.portfolio.positions.get(self.config.symbol)
        if position is not None:
            position_info = {
                "is_open": True,
                "entry_price": float(position.entry_price),
                "entry_step": int(position.entry_step),
                "duration": position.get_holding_period(self.current_step),
                "shares": float(position.shares),
            }

        trade_info = None
        if position_closed and self._last_closed_trade:
            trade_info = {
                "pnl_pct": float(self._last_closed_trade.get("realized_pnl_pct", 0.0)),
                "holding_hours": float(self._last_closed_trade.get("holding_period", 0)),
                "action": self._last_closed_trade.get("exit_reason", self._last_closed_trade.get("trigger", "")),
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
        metrics = self.portfolio.get_portfolio_metrics()
        return {
            "equity": metrics["equity"],
            "peak_equity": metrics["peak_equity"],
            "deployed_pct": metrics["exposure_pct"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "num_trades": metrics["total_trades"],
        }

    # Portfolio Accounting & Diagnostics

    def _mark_to_market_price(self, price: float, step: int) -> None:
        self.portfolio.update_positions({self.config.symbol: price}, step)
        self.equity_curve = list(self.portfolio.equity_curve)

    def _mark_to_market_current(self) -> None:
        price = float(self.data.loc[self.current_step, "close"])
        self._mark_to_market_price(price, self.current_step)

    def _get_info(self) -> Dict[str, Any]:
        timestamp = self.data.loc[self.current_step, "timestamp"]
        metrics = self.portfolio.get_portfolio_metrics()
        position = self.portfolio.positions.get(self.config.symbol)

        info: Dict[str, Any] = {
            "step": self.current_step,
            "episode_step": self.episode_step,
            "timestamp": _timestamp_str(timestamp),
            "cash": float(metrics["cash"]),
            "equity": float(metrics["equity"]),
            "total_return_pct": float(metrics["total_return_pct"]),
            "sharpe_ratio": float(metrics["sharpe_ratio"]),
            "sortino_ratio": float(metrics["sortino_ratio"]),
            "max_drawdown_pct": float(metrics["max_drawdown_pct"]),
            "win_rate": float(metrics["win_rate"]),
            "exposure_pct": float(metrics["exposure_pct"]),
            "num_trades": int(metrics["total_trades"]),
            "position_open": bool(position is not None),
            "last_action": self.last_action.name,
        }

        # Attach a snapshot of the portfolio metrics so downstream consumers (e.g.
        # evaluation loops running on vectorised envs that auto-reset) can rely on
        # the info payload without re-querying the environment state.
        info["metrics"] = dict(metrics)

        if position is not None:
            info["position"] = {
                "shares": float(position.shares),
                "entry_price": float(position.entry_price),
                "entry_step": int(position.entry_step),
                "unrealized_pnl": float(position.unrealized_pnl),
                "unrealized_pnl_pct": float(position.unrealized_pnl_pct),
            }
        else:
            info["position"] = None

        if self.config.log_trades:
            info["recent_trades"] = self.portfolio.get_closed_positions()[-5:]

        if self.episode_step > 0:
            reward_stats = self.reward_shaper.get_episode_stats()
            info["reward_stats"] = reward_stats

            if self.episode_step % 100 == 0:
                info["reward_contributions"] = self.reward_shaper.get_component_contributions()

        return info

    def _action_name(self, action: int) -> str:
        try:
            return TradeAction(int(action)).name
        except ValueError:
            return str(action)

    # Rendering (human)

    def _render_human(self, info: Dict[str, Any]) -> None:  # pragma: no cover - console I/O
        position_info = info.get("position") or {}
        print(
            f"Step {info['episode_step']:04d} (idx {info['step']:06d}) | "
            f"Time {info['timestamp']} | Action {info['last_action']} | "
            f"Equity {info['equity']:.2f} | Cash {info['cash']:.2f} | "
            f"Exposure {info['exposure_pct']:.2%} | Position {position_info}",
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
    "RewardBreakdown",
]
