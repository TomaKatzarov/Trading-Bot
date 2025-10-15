"""Gymnasium-compatible trading environment for RL agent training."""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Suppress PyTorch TF32 deprecation warnings before torch import
warnings.filterwarnings("ignore", message=".*TF32.*", category=UserWarning)

import gymnasium as gym
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import torch

# Configure TF32 immediately after torch import to suppress warnings in worker processes
if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
    try:
        torch.set_float32_matmul_precision("high")
    except RuntimeError:
        pass

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
    # Removed MACD_line, Stoch_K, Stoch_D aliases - Phase 3 format already uses these names
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
    sl_inference_device: Optional[str] = None

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
    evaluation_mode: bool = False

    # Reward configuration
    reward_config: Optional[RewardConfig] = None

    # Exploration curriculum (2025-10-08 Anti-Collapse v4 → v9 → DEPRECATED v10)
    exploration_curriculum_enabled: bool = False  # DEPRECATED: Using epsilon-greedy instead
    exploration_phase1_end_step: int = 20000
    exploration_phase1_min_action_pct: float = 0.10
    exploration_phase1_penalty: float = -5.0
    exploration_phase2_end_step: int = 50000
    exploration_phase2_min_action_pct: float = 0.05
    exploration_phase2_penalty: float = -2.0
    exploration_evaluation_window: int = 100
    exploration_excluded_actions: Optional[List[str]] = None  # Will default to ["HOLD"] in __post_init__
    exploration_require_sell_actions: bool = True  # V9: Specifically require SELL actions (SELL_PARTIAL or SELL_ALL)
    exploration_min_sell_pct: float = 0.05  # V9: Minimum 5% of actions must be SELL to learn exits
    exploration_sell_penalty_multiplier: float = 5.0  # Multiplier applied to base penalty when SELL usage falls short
    exploration_require_buy_actions: bool = False  # Optional aggregated BUY enforcement (spanning BUY_* actions)
    exploration_min_buy_pct: float = 0.05  # Minimum % of BUY_* actions required when enforcement enabled
    exploration_buy_penalty_multiplier: float = 3.0  # Multiplier applied to base penalty when BUY usage falls short
    curriculum_action_coverage_enabled: bool = False
    curriculum_action_coverage_start_step: int = 10000
    curriculum_action_coverage_min_buy_pct: float = 0.04
    curriculum_action_coverage_min_sell_pct: float = 0.04
    curriculum_action_coverage_reward_multiplier: float = 0.8
    curriculum_action_coverage_penalty_cap: float = 1.5
    curriculum_action_coverage_penalty_power: float = 2.0
    curriculum_action_coverage_selected_weight: float = 0.5
    curriculum_action_penalty_cap_total: float = 1.5
    
    # Epsilon-Greedy Exploration (2025-10-08 v10 - PROPER APPROACH!)
    epsilon_greedy_enabled: bool = False
    epsilon_start: float = 0.5  # Start with 50% random exploration
    epsilon_end: float = 0.01   # End with 1% exploration
    epsilon_decay_steps: int = 50000  # Linear decay duration
    epsilon_current: float = 0.5  # Will be updated dynamically
    
    # Action Restrictions (2025-10-08 v10 RADICAL FIX)
    disabled_actions: Optional[List[str]] = None  # Actions to completely disable (e.g., ["ADD_POSITION"])

    # ADD_POSITION gating (Stage 3 risk controls)
    add_position_gate_enabled: bool = False
    add_position_gate_max_exposure_pct: float = 0.12
    add_position_gate_min_unrealized_pct: float = 0.0
    add_position_gate_base_penalty: float = 0.25
    add_position_gate_severity_multiplier: float = 0.5
    add_position_gate_penalty_cap: float = 1.2
    add_position_gate_violation_decay: int = 2000

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
        "SMA_10", "SMA_20", "MACD_line", "MACD_signal", "MACD_hist",
        "RSI_14", "Stoch_K", "Stoch_D", "ADX_14", "ATR_14",
        "BB_bandwidth", "OBV", "Volume_SMA_20", "1h_return",
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
        self.portfolio = PortfolioManager(
            config.get_portfolio_config(),
            log_trades=config.log_trades,
            log_level=config.log_level,
        )

        self._load_data()
        self._load_sl_models()
        self._define_spaces()

        self.current_step: int = 0
        self.episode_step: int = 0
        self.last_action: TradeAction = TradeAction.HOLD
        self.equity_curve: List[float] = []
        self._last_reward: RewardBreakdown = RewardBreakdown()
        self._last_reward_components: Dict[str, float] = {}
        self._last_closed_trade: Optional[Dict[str, Any]] = None

        # Action diversity tracking (2025-10-08 Anti-Collapse)
        self.action_history: List[int] = []  # Full episode action history (attempted actions)
        self.selected_action_history: List[int] = []  # Policy outputs before environment overrides
        self.executed_action_history: List[int] = []  # Actual actions that affected the environment
        self.consecutive_action_count: int = 0  # Count of current action streak
        self.action_diversity_window: List[int] = []  # Rolling 50-step window
        self.max_consecutive_actions: int = 3  # Hard limit on repetitions
        self.telemetry_selected_counter: int = 0  # Count policy-selected voluntary trades
        self.telemetry_executed_counter: int = 0  # Count executed voluntary trades
        
        # Exploration curriculum tracking (2025-10-08 Anti-Collapse v4)
        self.total_env_steps: int = 0  # Cumulative steps across all episodes
        self.curriculum_evaluation_window: List[int] = []  # Rolling window for curriculum checks (executed)
        self.curriculum_selected_window: List[int] = []  # Rolling window for policy-selected actions
        self.curriculum_episode_violations: int = 0  # Count violations this episode
        self.curriculum_episode_penalty: float = 0.0  # Total penalty this episode
        self._last_add_evaluation: Optional[Dict[str, Any]] = None
        self._add_eval_history: List[Dict[str, Any]] = []
        if config.exploration_excluded_actions is None:
            self.curriculum_excluded_actions = {TradeAction.HOLD.value}
        else:
            self.curriculum_excluded_actions = {
                TradeAction[action].value for action in config.exploration_excluded_actions
            }

        if config.exploration_curriculum_enabled:
            logger.info(
                "Exploration curriculum ENABLED: Phase1=%d steps (min=%.1f%%, penalty=%.1f), Phase2=%d steps (min=%.1f%%, penalty=%.1f)",
                config.exploration_phase1_end_step,
                config.exploration_phase1_min_action_pct * 100,
                config.exploration_phase1_penalty,
                config.exploration_phase2_end_step,
                config.exploration_phase2_min_action_pct * 100,
                config.exploration_phase2_penalty,
            )
        else:
            logger.info("Exploration curriculum DISABLED")

        # Curriculum Stage 4 coverage controls
        self.curriculum_action_coverage_enabled = bool(config.curriculum_action_coverage_enabled)
        self.curriculum_action_coverage_start_step = max(0, int(config.curriculum_action_coverage_start_step))
        self.curriculum_action_coverage_min_buy_pct = float(max(0.0, config.curriculum_action_coverage_min_buy_pct))
        self.curriculum_action_coverage_min_sell_pct = float(max(0.0, config.curriculum_action_coverage_min_sell_pct))
        self.curriculum_action_coverage_reward_multiplier = float(
            max(0.0, min(1.0, config.curriculum_action_coverage_reward_multiplier))
        )
        self.curriculum_action_coverage_penalty_cap = float(max(0.0, config.curriculum_action_coverage_penalty_cap))
        self.curriculum_action_coverage_penalty_power = float(max(0.0, config.curriculum_action_coverage_penalty_power))
        selected_weight = float(np.clip(config.curriculum_action_coverage_selected_weight, 0.0, 1.0))
        self.curriculum_action_coverage_selected_weight = selected_weight
        self.curriculum_action_coverage_executed_weight = 1.0 - selected_weight
        self.curriculum_penalty_cap_total = float(max(0.0, config.curriculum_action_penalty_cap_total))
        self._curriculum_last_multiplier: float = 1.0
        self._curriculum_last_detail: Dict[str, Any] = {}

        if seed is not None:
            self.seed(seed)
        else:
            self.seed()

        # Stage 3 add-position gate configuration
        self.add_gate_enabled = bool(config.add_position_gate_enabled)
        self.add_gate_max_exposure_pct = float(config.add_position_gate_max_exposure_pct)
        self.add_gate_min_unrealized_pct = float(config.add_position_gate_min_unrealized_pct)
        self.add_gate_base_penalty = max(0.0, float(config.add_position_gate_base_penalty))
        self.add_gate_severity_multiplier = max(0.0, float(config.add_position_gate_severity_multiplier))
        self.add_gate_penalty_cap = max(0.0, float(config.add_position_gate_penalty_cap))
        self.add_gate_violation_decay = max(1, int(config.add_position_gate_violation_decay))
        self._pending_add_gate_penalty: float = 0.0
        self._last_add_gate_penalty_info: Optional[Dict[str, Any]] = None
        self._add_gate_violation_streak: int = 0
        self._last_add_gate_violation_step: Optional[int] = None
        if self.add_gate_enabled:
            logger.info(
                "ADD_POSITION gate ENABLED: max_exposure=%.2f%%, min_unrealized=%.2f%%, base_penalty=%.3f, severity=%.2f, cap=%.3f, decay=%d steps",
                self.add_gate_max_exposure_pct * 100,
                self.add_gate_min_unrealized_pct * 100,
                self.add_gate_base_penalty,
                self.add_gate_severity_multiplier,
                self.add_gate_penalty_cap,
                self.add_gate_violation_decay,
            )

        # Epsilon scheduling (Stage 4 annealing support)
        self._epsilon_schedule_start_step = 0
        self._epsilon_initial = float(self.config.epsilon_start)
        self._epsilon_target = float(self.config.epsilon_end)
        self._epsilon_decay_steps = max(1, int(self.config.epsilon_decay_steps or 1))

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

        # Reset action diversity tracking (2025-10-08 Anti-Collapse)
        self.action_history = []
        self.selected_action_history = []
        self.executed_action_history = []
        self.consecutive_action_count = 0
        self.action_diversity_window = []
        self.telemetry_selected_counter = 0
        self.telemetry_executed_counter = 0
        
        # Reset curriculum episode counters
        self.curriculum_episode_violations = 0
        self.curriculum_episode_penalty = 0.0
        self._add_eval_history.clear()
        # Note: curriculum_evaluation_window persists across episodes
        # Note: total_env_steps persists across episodes
        self._curriculum_last_multiplier = 1.0
        self._curriculum_last_detail = {}

        self._pending_add_gate_penalty = 0.0
        self._last_add_gate_penalty_info = None
        self._add_gate_violation_streak = 0
        self._last_add_gate_violation_step = None

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

        # Reset per-step telemetry caches
        self._pending_add_gate_penalty = 0.0
        self._last_add_gate_penalty_info = None
        self._last_add_evaluation = None
        position_for_eval = self.portfolio.positions.get(self.config.symbol)
        if position_for_eval is not None:
            try:
                self._evaluate_add_position(position_for_eval)
            except Exception:  # pragma: no cover - diagnostic safety net
                logger.debug("ADD_POSITION pre-check failed", exc_info=True)

        # EPSILON-GREEDY EXPLORATION (2025-10-08 v10)
        # Randomly explore with probability epsilon to discover reward landscape
        action_int = int(action)
        policy_action_idx = action_int
        if self.config.epsilon_greedy_enabled:
            # Update epsilon (linear decay)
            if self.total_env_steps <= self.config.epsilon_decay_steps:
                progress = self.total_env_steps / self.config.epsilon_decay_steps
                self.config.epsilon_current = self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)
            else:
                self.config.epsilon_current = self.config.epsilon_end
            
            # Epsilon-greedy: random action with probability epsilon
            if np.random.random() < self.config.epsilon_current:
                available_actions = self._get_valid_actions()
                if available_actions:
                    action_int = int(np.random.choice(available_actions))
                    if self.total_env_steps % 1000 == 0:  # Log occasionally
                        logger.debug(
                            "ε-greedy: Random valid action %s (ε=%.3f)",
                            TradeAction(action_int).name,
                            self.config.epsilon_current,
                        )
        
        # V10 RADICAL FIX: Block disabled actions even if policy selects them
        if self.config.disabled_actions:
            for action_name in self.config.disabled_actions:
                try:
                    disabled_idx = TradeAction[action_name].value
                    if action_int == disabled_idx:
                        action_int = TradeAction.HOLD.value  # Override to HOLD
                        logger.debug(f"Blocked disabled action {action_name}, forcing HOLD")
                except KeyError:
                    pass
        
        # Track action repetitions and enforce diversity (2025-10-08 Anti-Collapse)
        # (Keep existing code below...)
        
        # Update consecutive action counter
        if len(self.action_history) > 0 and self.action_history[-1] == action_int:
            self.consecutive_action_count += 1
        else:
            self.consecutive_action_count = 1
        
        # Enforce max consecutive actions (override to HOLD if exceeded)
        if self.consecutive_action_count > self.max_consecutive_actions:
            logger.debug(
                "Action repetition limit hit: %s repeated %d times, forcing HOLD",
                TradeAction(action_int).name,
                self.consecutive_action_count,
            )
            action_int = TradeAction.HOLD
            self.consecutive_action_count = 1  # Reset counter
        
        # Update tracking history
        self.action_history.append(action_int)
        self.action_diversity_window.append(action_int)
        if len(self.action_diversity_window) > 50:
            self.action_diversity_window.pop(0)

        prev_equity = self.portfolio.get_equity()
        self.last_action = TradeAction(action_int)
        action_executed, action_info = self._execute_action(action_int)
        executed_action_idx = action_int if action_executed else TradeAction.HOLD.value
        action_info["executed_action_idx"] = executed_action_idx
        action_info["executed_action_name"] = TradeAction(executed_action_idx).name
        action_info["action_selected_idx"] = policy_action_idx
        action_info["action_selected_name"] = TradeAction(policy_action_idx).name
        action_info["action_executed"] = bool(action_executed)
        if self._last_add_evaluation is not None:
            action_info.setdefault("add_evaluation", dict(self._last_add_evaluation))

        # Track policy-selected vs executed actions for diagnostics
        self.selected_action_history.append(int(policy_action_idx))
        if int(policy_action_idx) != TradeAction.HOLD.value:
            self.telemetry_selected_counter += 1
        self.executed_action_history.append(int(executed_action_idx))
        if int(executed_action_idx) != TradeAction.HOLD.value:
            self.telemetry_executed_counter += 1

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
            position_closed_info=position_closed_info,  # FIX #13: Pass current trade info!
        )

        # Apply exploration curriculum penalty (2025-10-08 Anti-Collapse v4)
        # CRITICAL: Update curriculum window BEFORE evaluating to include current action
        # Curriculum diversity should reflect the actions that actually affected the
        # environment. Counting attempted (but rejected) actions was letting the
        # policy satisfy requirements without producing real trades.
        self.curriculum_evaluation_window.append(int(executed_action_idx))
        window_size = self.config.exploration_evaluation_window
        if len(self.curriculum_evaluation_window) > window_size:
            self.curriculum_evaluation_window.pop(0)
        self.curriculum_selected_window.append(int(policy_action_idx))
        if len(self.curriculum_selected_window) > window_size:
            self.curriculum_selected_window.pop(0)
        
        self.total_env_steps += 1
        
        curriculum_penalty = self._apply_exploration_curriculum()
        if curriculum_penalty != 0.0:
            reward += curriculum_penalty
        if self._curriculum_last_multiplier != 1.0:
            reward *= self._curriculum_last_multiplier

        gate_penalty = self._pending_add_gate_penalty
        gate_penalty_info = (
            dict(self._last_add_gate_penalty_info)
            if isinstance(self._last_add_gate_penalty_info, dict)
            else None
        )
        if gate_penalty != 0.0:
            reward += gate_penalty
        self._pending_add_gate_penalty = 0.0

        # Decay violation streak if the agent goes long enough without infractions
        self._decay_add_gate_streak()

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
                "reward_components": dict(self._last_reward_components),
                "selected_action_idx": int(policy_action_idx),
                "selected_action_name": TradeAction(policy_action_idx).name,
                "executed_action_idx": int(executed_action_idx),
                "executed_action_name": TradeAction(executed_action_idx).name,
            }
        )

        if gate_penalty_info is not None or gate_penalty != 0.0:
            payload = gate_penalty_info or {}
            payload = dict(payload)
            if gate_penalty != 0.0:
                payload["penalty"] = gate_penalty
            payload.setdefault("step", int(self.total_env_steps))
            info["add_position_gate"] = payload
            if self.config.exploration_curriculum_enabled:
                curriculum_block = info.setdefault("curriculum", {})
                curriculum_block["add_position_gate_penalty"] = float(payload.get("penalty", gate_penalty))

        if self.config.exploration_curriculum_enabled:
            curriculum_block = info.setdefault("curriculum", {})
            curriculum_block["penalty"] = float(curriculum_penalty)
            curriculum_block["episode_penalty"] = float(self.curriculum_episode_penalty)
            curriculum_block["episode_violations"] = int(self.curriculum_episode_violations)
            if self._curriculum_last_multiplier != 1.0:
                curriculum_block["reward_multiplier"] = float(self._curriculum_last_multiplier)
            if self._curriculum_last_detail:
                # Merge without clobbering mandatory keys
                for key, value in self._curriculum_last_detail.items():
                    if key in {"penalty", "episode_penalty", "episode_violations"}:
                        continue
                    curriculum_block.setdefault(key, value)

        # Avoid carrying gate payloads into the next step once logged
        self._last_add_gate_penalty_info = None

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

            # Provide per-episode action diagnostics for downstream metrics
            from collections import Counter

            executed_counts = Counter(self.executed_action_history)
            selected_counts = Counter(self.selected_action_history)

            def _normalise_action_counts(counter: Counter) -> Dict[str, int]:
                return {
                    TradeAction(idx).name: int(counter.get(idx, 0))
                    for idx in range(len(TradeAction))
                }

            if executed_counts:
                total_executed = sum(executed_counts.values())
                if total_executed > 0:
                    executed_distribution = {
                        TradeAction(idx).name: float(executed_counts.get(idx, 0) / total_executed)
                        for idx in range(len(TradeAction))
                    }
                    info["action_counts"] = _normalise_action_counts(executed_counts)
                    info["action_distribution"] = executed_distribution
            if selected_counts:
                info["policy_action_counts"] = _normalise_action_counts(selected_counts)

            info.setdefault("episode", {}).setdefault("telemetry", {})
            info["episode"]["telemetry"].update(
                {
                    "selected_voluntary_trades": int(self.telemetry_selected_counter),
                    "executed_voluntary_trades": int(self.telemetry_executed_counter),
                }
            )

            if self._add_eval_history:
                total_samples = len(self._add_eval_history)
                allowed_samples = sum(1 for entry in self._add_eval_history if entry.get("can_add"))
                average_conf = float(
                    sum(entry.get("confidence", 0.0) for entry in self._add_eval_history) / total_samples
                )
                max_conf = float(max(entry.get("confidence", 0.0) for entry in self._add_eval_history))
                min_conf = float(min(entry.get("confidence", 0.0) for entry in self._add_eval_history))
                reason_counter = Counter(
                    "allowed" if entry.get("can_add") else (entry.get("reason") or "blocked")
                    for entry in self._add_eval_history
                )
                add_summary = {
                    "samples": total_samples,
                    "allow_rate": float(allowed_samples / total_samples),
                    "avg_confidence": average_conf,
                    "max_confidence": max_conf,
                    "min_confidence": min_conf,
                    "reason_counts": dict(reason_counter),
                }
                info.setdefault("episode", {}).setdefault("diagnostics", {})["add_position"] = add_summary
                info["add_evaluation_summary"] = add_summary
                if logger.isEnabledFor(logging.INFO):
                    reasons_str = ", ".join(f"{key}:{value}" for key, value in reason_counter.items()) or "none"
                    logger.info(
                        "📈 [ADD_POSITION] Episode diagnostics (env_step=%d): allow_rate=%.2f, avg_conf=%.2f, samples=%d, reasons=%s",
                        self.total_env_steps,
                        add_summary["allow_rate"],
                        add_summary["avg_confidence"],
                        total_samples,
                        reasons_str,
                    )

            if self.config.exploration_curriculum_enabled:
                curriculum_summary = {
                    "episode_penalty": float(self.curriculum_episode_penalty),
                    "violations": int(self.curriculum_episode_violations),
                }
                info.setdefault("episode", {}).setdefault("diagnostics", {})["curriculum"] = curriculum_summary
                info["curriculum"]["episode_summary"] = curriculum_summary

            
            # Log curriculum summary if enabled (only every 10 episodes to avoid spam)
            if (
                self.config.exploration_curriculum_enabled
                and self.curriculum_episode_violations > 0
                and self.total_env_steps % 5000 == 0  # Print every ~10 episodes (500 steps each)
            ):
                avg_penalty_per_step = self.curriculum_episode_penalty / max(1, self.episode_step)
                logger.info(
                    "📊 [Curriculum Summary] Episode ended at step %d: %d total violations, %.1f total penalty (avg %.2f/step)",
                    self.total_env_steps,
                    self.curriculum_episode_violations,
                    self.curriculum_episode_penalty,
                    avg_penalty_per_step,
                )

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

        if getattr(self.config, "evaluation_mode", False):
            start_raw = self.config.val_start or self.config.train_start
            end_raw = self.config.val_end or self.config.train_end
        else:
            start_raw = self.config.train_start
            end_raw = self.config.train_end

        mask = None
        if start_raw:
            start = pd.Timestamp(start_raw)
            mask = (data["timestamp"] >= start) if mask is None else (mask & (data["timestamp"] >= start))
        if end_raw:
            end = pd.Timestamp(end_raw)
            mask = (data["timestamp"] <= end) if mask is None else (mask & (data["timestamp"] <= end))

        if mask is not None:
            data = data[mask].reset_index(drop=True)
            if getattr(self.config, "evaluation_mode", False):
                logger.info(
                    "Validation window applied for evaluation: %s to %s (rows=%d)",
                    start_raw or "-inf",
                    end_raw or "+inf",
                    len(data),
                )

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
        self._column_positions = {col: idx for idx, col in enumerate(self.data.columns)}
        self._numeric_column_cache: Dict[str, np.ndarray] = {}
        self._timestamp_series = self.data["timestamp"] if "timestamp" in self.data.columns else None
        for col, series in self.data.items():
            if col == "timestamp":
                continue
            if is_numeric_dtype(series.dtype):
                self._numeric_column_cache[col] = series.to_numpy(copy=False)

        self.feature_cols = self.feature_extractor.get_feature_names()
        self.regime_feature_names = self.regime_indicators.get_regime_names()

        logger.info(
            "Loaded %d timesteps for %s, %d features",
            len(data),
            self.config.symbol,
            self.feature_extractor.get_feature_count(),
        )

    def _value_at(self, column: str, idx: int) -> Any:
        cache = self._numeric_column_cache.get(column) if hasattr(self, "_numeric_column_cache") else None
        if cache is not None:
            return cache[idx]
        pos = self._column_positions.get(column) if hasattr(self, "_column_positions") else None
        if pos is None:
            raise KeyError(f"Column '{column}' is not available in cached data")
        return self.data.iat[idx, pos]

    def _float_at(self, column: str, idx: int) -> float:
        return float(self._value_at(column, idx))

    def _timestamp_at(self, idx: int) -> pd.Timestamp:
        if self._timestamp_series is not None:
            return self._timestamp_series.iat[idx]
        raise KeyError("'timestamp' column is not available in dataset")

    def _load_sl_models(self) -> None:
        """Load pre-trained SL models"""

        if self.sl_models is None:
            self.sl_models = {}
        else:
            self.sl_models.clear()
        raw_device_pref = self.config.sl_inference_device or "cpu"
        device_pref = "cpu"
        if isinstance(raw_device_pref, str):
            normalized = raw_device_pref.strip().lower()
            if normalized in {"auto", "default", "gpu"}:
                if torch.cuda.is_available():
                    device_pref = "cuda"
                else:
                    logger.warning("SL inference device requested as '%s' but CUDA is unavailable; using CPU", raw_device_pref)
            elif normalized.startswith("cuda"):
                if torch.cuda.is_available():
                    device_pref = raw_device_pref
                else:
                    logger.warning("CUDA device '%s' requested for SL inference but CUDA is unavailable; falling back to CPU", raw_device_pref)
            else:
                device_pref = raw_device_pref
        else:
            device_pref = "cpu"

        if load_sl_checkpoint is None:
            logger.warning("SL checkpoint utilities unavailable; using random predictions")
            if self.feature_extractor is not None:
                self.feature_extractor.update_sl_models({})
            return

        for name, checkpoint_path in self.config.sl_checkpoints.items():
            try:
                try:
                    model_bundle = load_sl_checkpoint(checkpoint_path, device=device_pref)
                except TypeError:
                    model_bundle = load_sl_checkpoint(checkpoint_path)
            except Exception as exc:  # pragma: no cover - logging side effect
                logger.warning("Failed to load checkpoint %s: %s", name, exc)
                continue

            if model_bundle is None:
                logger.warning("Checkpoint loader returned None for %s", name)
                continue

            asset_id = None
            if getattr(model_bundle, "asset_mapping", None):
                symbol_key = self.config.symbol
                asset_id = model_bundle.asset_mapping.get(symbol_key)
                if asset_id is None:
                    asset_id = model_bundle.asset_mapping.get(symbol_key.upper())
                if asset_id is None:
                    asset_id = model_bundle.asset_mapping.get(symbol_key.lower())
                if asset_id is None and symbol_key.endswith(".X"):
                    base_symbol = symbol_key.split(".")[0]
                    asset_id = model_bundle.asset_mapping.get(base_symbol)
                if asset_id is not None:
                    try:
                        model_bundle.default_asset_id = int(asset_id)
                    except (TypeError, ValueError):
                        logger.warning(
                            "SL checkpoint %s returned non-integer asset id %s for %s",
                            name,
                            asset_id,
                            symbol_key,
                        )
                        asset_id = None
                if asset_id is None:
                    logger.warning(
                        "Symbol %s not present in SL checkpoint %s asset mapping; defaulting to asset id 0",
                        self.config.symbol,
                        name,
                    )

            self.sl_models[name.lower()] = model_bundle
            device_info = getattr(model_bundle, "device", "unknown")
            logger.info("Loaded SL model: %s (device=%s)", name, device_info)

        if len(self.sl_models) == 0:
            logger.warning("No SL models loaded - using random predictions")
        if self.feature_extractor is not None:
            self.feature_extractor.update_sl_models(self.sl_models)

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
                        -np.finfo(np.float32).max,  # Fixed: unrealized_pnl can go way below -10.0
                        -np.finfo(np.float32).max,  # Fixed: unrealized_pnl_pct can go way below -10.0
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

    def _evaluate_add_position(self, position: Any) -> Tuple[bool, float, str]:
        """Return whether ADD_POSITION is permitted along with confidence score and reason."""

        reward_cfg = getattr(self.reward_shaper, "config", None)
        unrealized_pct = float(getattr(position, "unrealized_pnl_pct", 0.0))
        if self._timestamp_series is not None:
            timestamp = self._timestamp_series.iat[self.current_step]
        else:
            timestamp = pd.NaT

        try:
            pyramid_count = 0
            if hasattr(position, "metadata") and position.metadata:
                pyramid_count = int(position.metadata.get("pyramid_count", 0))
        except Exception:  # pragma: no cover
            pyramid_count = 0

        threshold = 0.75
        min_profit = 0.02
        max_adds = 2
        enabled = False

        if reward_cfg is not None:
            enabled = bool(getattr(reward_cfg, "add_position_enabled", False))
            threshold = float(getattr(reward_cfg, "add_position_confidence_threshold", threshold))
            min_profit = float(getattr(reward_cfg, "add_position_min_profit_pct", min_profit))
            max_adds = int(getattr(reward_cfg, "add_position_max_adds", max_adds))

        evaluation: Dict[str, Any] = {
            "can_add": False,
            "confidence": 0.0,
            "reason": "add_disabled" if not enabled else "",
            "threshold": threshold,
            "min_profit_pct": min_profit,
            "unrealized_pnl_pct": unrealized_pct,
            "pyramid_count": pyramid_count,
            "max_adds": max_adds,
            "timestamp": _timestamp_str(timestamp),
            "step_index": int(self.current_step),
            "env_step": int(self.total_env_steps),
        }

        entry_size_label = "medium"
        if hasattr(position, "metadata") and getattr(position, "metadata", None):
            entry_size_label = position.metadata.get("entry_size", "medium")

        sizing_map = {"small": 0.025, "medium": 0.06, "large": 0.09}
        planned_pct = sizing_map.get(entry_size_label, 0.06)
        evaluation["entry_size"] = entry_size_label
        evaluation["planned_pct"] = planned_pct

        try:
            metrics = self.portfolio.get_portfolio_metrics()
            current_exposure_pct = float(metrics.get("exposure_pct", 0.0))
        except Exception:  # pragma: no cover - defensive logging safety
            current_exposure_pct = 0.0

        gate_preview = self._check_add_position_gate(
            position,
            planned_pct=planned_pct,
            current_exposure_pct=current_exposure_pct,
            future_exposure_pct=current_exposure_pct + planned_pct,
            preview=True,
        )
        evaluation["gate_preview"] = gate_preview

        if not enabled:
            confidence = 0.0
        elif unrealized_pct < min_profit:
            evaluation["reason"] = "insufficient_profit"
            confidence = 0.0
        elif pyramid_count >= max_adds:
            evaluation["reason"] = "max_pyramids_reached"
            confidence = 0.0
        else:
            confidence = self._estimate_add_position_confidence(position)
            evaluation["confidence"] = confidence
            if confidence < threshold:
                evaluation["reason"] = "insufficient_confidence"
            else:
                evaluation["can_add"] = True
                evaluation["reason"] = ""

        evaluation["confidence"] = float(confidence)

        self._last_add_evaluation = evaluation
        self._add_eval_history.append(dict(evaluation))
        if len(self._add_eval_history) > 1000:
            self._add_eval_history.pop(0)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "ADD_POSITION evaluation: can_add=%s, confidence=%.3f (threshold=%.2f), unrealized=%.3f, pyramids=%d/%d, reason=%s",
                evaluation["can_add"],
                evaluation["confidence"],
                threshold,
                unrealized_pct,
                pyramid_count,
                max_adds,
                evaluation["reason"] or "ok",
            )

        return evaluation["can_add"], evaluation["confidence"], evaluation["reason"]

    def _check_add_position_gate(
        self,
        position: Any,
        *,
        planned_pct: float,
        current_exposure_pct: float,
        future_exposure_pct: float,
        preview: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate Stage 3 add-position gate constraints.

        Returns a dictionary describing whether the action is allowed, the
        primary blocking reason (if any), and diagnostic metrics for telemetry.
        """

        metrics = {
            "planned_pct": float(planned_pct),
            "current_exposure_pct": float(current_exposure_pct),
            "future_exposure_pct": float(future_exposure_pct),
            "max_exposure_pct": float(self.add_gate_max_exposure_pct),
            "min_unrealized_pct": float(self.add_gate_min_unrealized_pct),
            "unrealized_pnl_pct": float(getattr(position, "unrealized_pnl_pct", 0.0)),
        }

        result: Dict[str, Any] = {
            "allowed": True,
            "reason": "",
            "metrics": metrics,
            "violations": [],
            "severity": 0.0,
            "preview": bool(preview),
        }

        if not self.add_gate_enabled:
            return result

        violations: List[Dict[str, Any]] = []

        exposure_limit = self.add_gate_max_exposure_pct
        if exposure_limit > 0 and future_exposure_pct > exposure_limit + 1e-6:
            over_pct = future_exposure_pct - exposure_limit
            severity = _clip01(over_pct / max(exposure_limit, 1e-6))
            violations.append({
                "type": "exposure",
                "severity": float(severity),
                "future_exposure_pct": float(future_exposure_pct),
            })

        unrealized_pct = metrics["unrealized_pnl_pct"]
        min_unrealized = self.add_gate_min_unrealized_pct
        if unrealized_pct < min_unrealized - 1e-6:
            deficit = min_unrealized - unrealized_pct
            severity = _clip01(deficit / max(abs(min_unrealized) + 1e-6, 1.0))
            violations.append({
                "type": "unrealized",
                "severity": float(severity),
                "unrealized_pnl_pct": float(unrealized_pct),
            })

        if not violations:
            return result

        # Determine primary reason (exposure takes precedence)
        primary = next((v for v in violations if v["type"] == "exposure"), violations[0])
        result["allowed"] = False
        result["reason"] = "add_gate_exposure" if primary["type"] == "exposure" else "add_gate_unrealized"
        result["violations"] = violations
        result["severity"] = float(max(v.get("severity", 0.0) for v in violations))
        return result

    def _register_add_gate_violation(self, gate_result: Dict[str, Any]) -> float:
        """Record an ADD_POSITION gate violation and compute its penalty."""

        self._last_add_gate_penalty_info = {
            "reason": gate_result.get("reason", "add_gate_violation"),
            "metrics": dict(gate_result.get("metrics", {})),
            "violations": list(gate_result.get("violations", [])),
            "severity": float(gate_result.get("severity", 0.0)),
            "streak": int(self._add_gate_violation_streak),
            "penalty": 0.0,
            "step": int(self.total_env_steps),
        }

        if not self.add_gate_enabled or self.add_gate_base_penalty <= 0.0:
            return 0.0

        current_step = self.total_env_steps
        if (
            self._last_add_gate_violation_step is not None
            and (current_step - self._last_add_gate_violation_step) >= self.add_gate_violation_decay
        ):
            self._add_gate_violation_streak = 0

        self._add_gate_violation_streak += 1
        self._last_add_gate_violation_step = current_step

        magnitude = self.add_gate_base_penalty * (
            1.0 + (self._add_gate_violation_streak - 1) * self.add_gate_severity_multiplier
        )
        if self.add_gate_penalty_cap > 0.0:
            magnitude = min(magnitude, self.add_gate_penalty_cap)

        penalty = -float(magnitude)
        self._pending_add_gate_penalty += penalty
        self.curriculum_episode_penalty += penalty
        self.curriculum_episode_violations += 1

        self._last_add_gate_penalty_info.update(
            {
                "penalty": penalty,
                "streak": int(self._add_gate_violation_streak),
                "step": int(current_step),
            }
        )

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "🚫 ADD_POSITION gate violation: step=%d reason=%s penalty=%.3f streak=%d severity=%.2f",
                current_step,
                self._last_add_gate_penalty_info["reason"],
                penalty,
                self._add_gate_violation_streak,
                self._last_add_gate_penalty_info.get("severity", 0.0),
            )

        return penalty

    def _reset_add_gate_streak(self) -> None:
        """Reset violation streak after a successful ADD_POSITION execution."""

        self._add_gate_violation_streak = 0
        self._last_add_gate_violation_step = None
        if self._last_add_gate_penalty_info:
            self._last_add_gate_penalty_info = {
                **self._last_add_gate_penalty_info,
                "penalty": 0.0,
                "streak": 0,
            }

    def _decay_add_gate_streak(self) -> None:
        """Decay violation streak after a cooldown period without violations."""

        if not self.add_gate_enabled:
            return
        if self._add_gate_violation_streak == 0:
            return
        if self._last_add_gate_violation_step is None:
            return

        if (self.total_env_steps - self._last_add_gate_violation_step) >= self.add_gate_violation_decay:
            self._add_gate_violation_streak = 0
            self._last_add_gate_violation_step = None

    def _estimate_add_position_confidence(self, position: Any) -> float:
        """Heuristic confidence score (0-1) for pyramiding an existing position."""

        reward_cfg = getattr(self.reward_shaper, "config", None)
        min_profit_cfg = 0.02
        if reward_cfg is not None:
            try:
                min_profit_cfg = float(getattr(reward_cfg, "add_position_min_profit_pct", min_profit_cfg))
            except Exception:  # pragma: no cover - fall back to default
                min_profit_cfg = 0.02

        unrealized_pct = float(getattr(position, "unrealized_pnl_pct", 0.0))

        sl_conf = 0.5
        if self.feature_extractor is not None:
            try:
                sl_probs = self.feature_extractor.get_sl_predictions(self.current_step)
                if sl_probs.size > 0:
                    sl_conf = float(np.clip(np.mean(sl_probs), 0.0, 1.0))
            except Exception:  # pragma: no cover - fallback to neutral confidence
                sl_conf = 0.5

        return_conf = 0.5
        if "1h_return" in self._column_positions:
            try:
                short_return = self._float_at("1h_return", self.current_step)
                return_conf = float(0.5 + 0.5 * np.tanh(short_return * 50.0))
            except Exception:  # pragma: no cover
                return_conf = 0.5

        breakout_conf = 0.0
        try:
            lookback = 12
            start_idx = max(0, self.current_step - lookback)
            high_cache = self._numeric_column_cache.get("high")
            if high_cache is not None:
                window = high_cache[start_idx : self.current_step + 1]
                if window.size > 0:
                    recent_high = float(np.max(window))
                    current_high = float(high_cache[self.current_step])
                    if recent_high > 0:
                        ratio = current_high / recent_high
                        breakout_conf = float(np.clip((ratio - 0.995) / 0.01, 0.0, 1.0))
            else:
                window_series = self.data.loc[start_idx:self.current_step, "high"]
                if not window_series.empty:
                    recent_high = float(window_series.max())
                    current_high = float(self._float_at("high", self.current_step))
                    if recent_high > 0:
                        ratio = current_high / recent_high
                        breakout_conf = float(np.clip((ratio - 0.995) / 0.01, 0.0, 1.0))
        except Exception:  # pragma: no cover
            breakout_conf = 0.0

        pnl_conf = float(np.clip(unrealized_pct / 0.05, 0.0, 1.0))
        profit_margin_boost = float(
            np.clip(
                (unrealized_pct - min_profit_cfg) / max(min_profit_cfg, 1e-6),
                0.0,
                1.0,
            )
        )

        confidence = (
            0.35 * sl_conf
            + 0.25 * return_conf
            + 0.2 * breakout_conf
            + 0.2 * pnl_conf
            + 0.1 * profit_margin_boost
        )
        return float(np.clip(confidence, 0.0, 1.0))

    def _get_valid_actions(self) -> List[int]:
        """Return a list of actions that are valid given the current state."""

        valid_actions: List[int] = [TradeAction.HOLD.value]
        has_position = self.config.symbol in self.portfolio.positions

        disabled_indices = set()
        if self.config.disabled_actions:
            for action_name in self.config.disabled_actions:
                try:
                    disabled_indices.add(TradeAction[action_name].value)
                except KeyError:
                    logger.warning("Unknown action name in disabled_actions: %s", action_name)

        if has_position:
            valid_actions.extend([TradeAction.SELL_PARTIAL.value, TradeAction.SELL_ALL.value])
            position = self.portfolio.positions.get(self.config.symbol)
            if (
                TradeAction.ADD_POSITION.value not in disabled_indices
                and position is not None
            ):
                can_add, _, _ = self._evaluate_add_position(position)
                if can_add:
                    valid_actions.append(TradeAction.ADD_POSITION.value)
        else:
            valid_actions.extend(
                [
                    TradeAction.BUY_SMALL.value,
                    TradeAction.BUY_MEDIUM.value,
                    TradeAction.BUY_LARGE.value,
                ]
            )

        # Remove any disabled actions and ensure uniqueness
        filtered = [action for action in valid_actions if action not in disabled_indices]
        # Guarantee HOLD inclusion
        if TradeAction.HOLD.value not in filtered:
            filtered.append(TradeAction.HOLD.value)

        return filtered

    # Trading Mechanics

    def _execute_action(self, action: int) -> Tuple[bool, Dict[str, Any]]:
        current_price = self._float_at("close", self.current_step)
        current_ts = self._timestamp_at(self.current_step)
        current_time = (
            current_ts.to_pydatetime()
            if hasattr(current_ts, "to_pydatetime")
            else pd.Timestamp(current_ts).to_pydatetime()
        )
        info: Dict[str, Any] = {"action_name": self._action_name(action), "price": current_price}
        symbol = self.config.symbol
        
        # V10 CRITICAL: Block disabled actions EVERYWHERE (train AND eval)
        if self.config.disabled_actions:
            action_name = TradeAction(action).name
            if action_name in self.config.disabled_actions:
                logger.debug(f"_execute_action: Blocked disabled action {action_name}, treating as HOLD")
                info["reject_reason"] = f"action_disabled_{action_name}"
                return False, info
        portfolio_cfg = self.portfolio.config

        trade_action = TradeAction(int(action))

        if trade_action == TradeAction.HOLD:
            return True, info

        if trade_action in (TradeAction.BUY_SMALL, TradeAction.BUY_MEDIUM, TradeAction.BUY_LARGE):
            if symbol in self.portfolio.positions:
                info["reject_reason"] = "position_exists"
                return False, info

            # Position sizing map (V3.1: Track for reward multipliers)
            sizing_map = {
                TradeAction.BUY_SMALL: (0.025, "small"),    # 2.5% of equity
                TradeAction.BUY_MEDIUM: (0.06, "medium"),   # 6% of equity
                TradeAction.BUY_LARGE: (0.09, "large")      # 9% of equity
            }
            target_pct, entry_size = sizing_map[trade_action]
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

            # Track entry size for reward calculation (V3.1)
            if hasattr(position, 'metadata'):
                position.metadata = position.metadata or {}
                position.metadata['entry_size'] = entry_size
                position.metadata['pyramid_count'] = 0
                position.metadata['partial_exit_taken'] = False
            
            info.update({
                "entry_price": entry_price,
                "shares": shares,
                "target_pct": target_pct,
                "entry_size": entry_size  # V3.1: Track for rewards
            })
            return True, info

        if trade_action == TradeAction.SELL_PARTIAL:
            # With multi-position support, check by symbol not by key
            symbol_positions = self.portfolio.get_positions_for_symbol(symbol)
            if not symbol_positions:
                info["reject_reason"] = "no_position"
                return False, info

            # Use the first/oldest position for the symbol
            position = symbol_positions[0]
            shares_to_sell = position.shares * 0.5
            if shares_to_sell <= 0:
                info["reject_reason"] = "zero_shares"
                return False, info

            commission = (shares_to_sell * current_price) * portfolio_cfg.commission_rate
            slippage_cost = (shares_to_sell * current_price) * (portfolio_cfg.slippage_bps / 10_000)

            # Track metadata before closing (V3.1)
            entry_size = "medium"  # Default
            pyramid_count = 0
            partial_exit_taken = False
            if hasattr(position, 'metadata') and position.metadata:
                entry_size = position.metadata.get('entry_size', 'medium')
                pyramid_count = position.metadata.get('pyramid_count', 0)
                partial_exit_taken = position.metadata.get('partial_exit_taken', False)
            
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
                # V3.1: Track exit type and position metadata for rewards
                trade["exit_type"] = "partial"
                trade["entry_size"] = entry_size
                trade["pyramid_count"] = pyramid_count
                trade["forced_exit"] = False
                info["trade"] = trade
                info["shares_sold"] = shares_to_sell
                
                # Mark that partial exit was taken
                remaining_positions = self.portfolio.get_positions_for_symbol(symbol)
                if remaining_positions:
                    remaining_position = remaining_positions[0]
                    if hasattr(remaining_position, 'metadata'):
                        remaining_position.metadata = remaining_position.metadata or {}
                        remaining_position.metadata['entry_size'] = entry_size
                        remaining_position.metadata['pyramid_count'] = pyramid_count
                        remaining_position.metadata['partial_exit_taken'] = True
            
            return success, info

        if trade_action == TradeAction.SELL_ALL:
            # With multi-position support, check by symbol not by key
            symbol_positions = self.portfolio.get_positions_for_symbol(symbol)
            if not symbol_positions:
                info["reject_reason"] = "no_position"
                return False, info

            # Use the first/oldest position for the symbol
            position = symbol_positions[0]
            commission = (position.shares * current_price) * portfolio_cfg.commission_rate
            slippage_cost = (position.shares * current_price) * (portfolio_cfg.slippage_bps / 10_000)

            # Track metadata before closing (V3.1)
            entry_size = "medium"  # Default
            pyramid_count = 0
            partial_exit_taken = False
            if hasattr(position, 'metadata') and position.metadata:
                entry_size = position.metadata.get('entry_size', 'medium')
                pyramid_count = position.metadata.get('pyramid_count', 0)
                partial_exit_taken = position.metadata.get('partial_exit_taken', False)
            
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
                # V3.1: Track exit type for rewards
                # If partial exit was already taken, this is "staged" exit (bonus!)
                if partial_exit_taken:
                    trade["exit_type"] = "staged"  # PARTIAL → ALL (1.1× bonus)
                else:
                    trade["exit_type"] = "full"    # Direct SELL_ALL (1.0×)
                trade["entry_size"] = entry_size
                trade["pyramid_count"] = pyramid_count
                trade["forced_exit"] = False
                info["trade"] = trade
            return success, info

        if trade_action == TradeAction.ADD_POSITION:
            # With multi-position support, check by symbol not by key
            symbol_positions = self.portfolio.get_positions_for_symbol(symbol)
            if not symbol_positions:
                info["reject_reason"] = "no_position_to_add"
                return False, info

            # Use the first/oldest position for the symbol
            position = symbol_positions[0]
            entry_size = "medium"
            if hasattr(position, "metadata") and position.metadata:
                entry_size = position.metadata.get("entry_size", "medium")
            sizing_map = {"small": 0.025, "medium": 0.06, "large": 0.09}
            target_pct = sizing_map.get(entry_size, 0.06)

            try:
                metrics = self.portfolio.get_portfolio_metrics()
                current_exposure_pct = float(metrics.get("exposure_pct", 0.0))
            except Exception:  # pragma: no cover - metrics fallback
                current_exposure_pct = 0.0
            gate_result = self._check_add_position_gate(
                position,
                planned_pct=target_pct,
                current_exposure_pct=current_exposure_pct,
                future_exposure_pct=current_exposure_pct + target_pct,
            )
            info["add_gate_detail"] = dict(gate_result)
            if not gate_result.get("allowed", True):
                penalty = self._register_add_gate_violation(gate_result)
                if penalty != 0.0:
                    info["add_gate_penalty"] = penalty
                info["reject_reason"] = gate_result.get("reason", "add_gate_blocked")
                info["add_confidence"] = gate_result.get("confidence", 0.0)
                info["add_gate_blocked"] = True
                return False, info

            can_add, confidence, reason = self._evaluate_add_position(position)
            info["add_gate_detail"] = dict(info.get("add_gate_detail", {}))
            evaluation_snapshot = self._last_add_evaluation if isinstance(self._last_add_evaluation, dict) else None
            if evaluation_snapshot:
                preview = evaluation_snapshot.get("gate_preview")
                if isinstance(preview, dict) and preview:
                    info["add_gate_detail"]["preview"] = dict(preview)
                info["add_gate_detail"].setdefault("evaluation_reason", evaluation_snapshot.get("reason"))
                info["add_gate_detail"].setdefault("evaluation_confidence", evaluation_snapshot.get("confidence"))

            if not can_add:
                info["reject_reason"] = reason
                info["add_confidence"] = confidence
                return False, info

            equity = self.portfolio.get_equity()
            target_value = equity * target_pct
            if target_value <= 0:
                info["reject_reason"] = "invalid_target"
                info["add_confidence"] = confidence
                return False, info

            additional_shares = target_value / current_price
            if additional_shares <= 0:
                info["reject_reason"] = "zero_shares"
                info["add_confidence"] = confidence
                return False, info

            entry_price = current_price * (1 + portfolio_cfg.slippage_bps / 10_000)
            commission = target_value * portfolio_cfg.commission_rate
            slippage_cost = target_value * (portfolio_cfg.slippage_bps / 10_000)

            total_cost_before = position.shares * position.entry_price
            additional_cost = additional_shares * entry_price
            new_total_shares = position.shares + additional_shares
            new_avg_entry = (total_cost_before + additional_cost) / new_total_shares

            position.shares = new_total_shares
            position.entry_price = new_avg_entry
            position.commission += commission
            position.slippage += slippage_cost

            pyramid_count = 0
            if hasattr(position, "metadata"):
                position.metadata = position.metadata or {}
                pyramid_count = int(position.metadata.get("pyramid_count", 0)) + 1
                position.metadata["pyramid_count"] = pyramid_count
                position.metadata.setdefault("entry_size", entry_size)

            logger.info(
                "ADD_POSITION: Added %.2f shares at $%.2f (confidence=%.2f, pyramid #%d), "
                "new total: %.2f shares @ avg $%.2f",
                additional_shares,
                entry_price,
                confidence,
                pyramid_count,
                new_total_shares,
                new_avg_entry,
            )

            info.update(
                {
                    "shares_added": additional_shares,
                    "new_entry_price": new_avg_entry,
                    "new_total_shares": new_total_shares,
                    "pyramid_count": pyramid_count,
                    "add_confidence": confidence,
                }
            )
            self._reset_add_gate_streak()
            return True, info

        info["reject_reason"] = "unknown_action"
        return False, info

    def _update_position(self) -> List[Dict[str, Any]]:
        symbol = self.config.symbol
        current_price = self._float_at("close", self.current_step)
        current_ts = self._timestamp_at(self.current_step)
        current_time = (
            current_ts.to_pydatetime()
            if hasattr(current_ts, "to_pydatetime")
            else pd.Timestamp(current_ts).to_pydatetime()
        )

        self._mark_to_market_price(current_price, self.current_step)

        trades: List[Dict[str, Any]] = []
        # With multi-position support, check by symbol
        symbol_positions = self.portfolio.get_positions_for_symbol(symbol)
        if not symbol_positions:
            return trades
        
        # Use the first/oldest position for stop-loss/take-profit checks
        position = symbol_positions[0]

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
                trade["forced_exit"] = True
                trade["forced_exit_reason"] = exit_reason
                trades.append(trade)
            return trade if success else None

        if position.unrealized_pnl_pct <= -self.config.stop_loss:
            logger.debug("Stop-loss triggered at step %s", self.current_step)
            _close("stop_loss")
            symbol_positions = self.portfolio.get_positions_for_symbol(symbol)
            if not symbol_positions:
                return trades
            position = symbol_positions[0]

        if position.unrealized_pnl_pct >= self.config.take_profit:
            logger.debug("Take-profit triggered at step %s", self.current_step)
            _close("take_profit")
            symbol_positions = self.portfolio.get_positions_for_symbol(symbol)
            if not symbol_positions:
                return trades
            position = symbol_positions[0]

        holding_period = position.get_holding_period(self.current_step)
        if holding_period >= self.config.max_hold_hours:
            logger.debug("Max hold time reached at step %s", self.current_step)
            _close("max_hold_time")
            symbol_positions = self.portfolio.get_positions_for_symbol(symbol)
            if not symbol_positions:
                return trades
            position = symbol_positions[0]

        risk_trades = self.portfolio.enforce_risk_limits(
            {symbol: current_price},
            current_time,
            self.current_step,
        )
        for trade in risk_trades:
            trade.setdefault("trigger", trade.get("exit_reason", "risk_limit"))
            trade["closed"] = True
            trade["forced_exit"] = True
            trade["forced_exit_reason"] = trade.get("trigger")
        trades.extend(risk_trades)
        return trades

    def _compute_reward(
        self,
        action: TradeAction,
        action_executed: bool,
        prev_equity: float,
        current_equity: float,
        position_closed: bool,
        position_closed_info: Optional[Dict[str, Any]] = None,  # FIX #13: Use current trade!
    ) -> float:
        position_info = None
        open_positions = self.portfolio.get_positions_for_symbol(self.config.symbol)
        if open_positions:
            # Use the position with largest absolute exposure as primary reference.
            primary = max(open_positions, key=lambda pos: abs(pos.current_value) or abs(pos.shares))
            aggregate_shares = float(sum(pos.shares for pos in open_positions))
            position_info = {
                "is_open": True,
                "entry_price": float(primary.entry_price),
                "entry_step": int(primary.entry_step),
                "duration": primary.get_holding_period(self.current_step),
                "shares": float(primary.shares),
                "is_long": bool(primary.shares > 0),
                "is_short": bool(primary.shares < 0),
                "position_count": len(open_positions),
                "aggregate_shares": aggregate_shares,
            }

        trade_info = None
        # FIX #13: Use position_closed_info directly (current trade), not _last_closed_trade (previous trade!)
        if position_closed and position_closed_info:
            trade_info = {
                "pnl_pct": float(position_closed_info.get("realized_pnl_pct", 0.0)),
                "holding_hours": float(position_closed_info.get("holding_period", 0)),
                "action": position_closed_info.get("exit_reason", position_closed_info.get("trigger", "")),
                "forced_exit": bool(position_closed_info.get("forced_exit", False)),
                "forced_exit_reason": position_closed_info.get("forced_exit_reason"),
                "exit_type": position_closed_info.get("exit_type"),
                "entry_size": position_closed_info.get("entry_size"),
                "pyramid_count": position_closed_info.get("pyramid_count", 0),
            }

        portfolio_state = self._get_portfolio_state_dict()

        # Pass action diversity info for reward shaping (2025-10-08 Anti-Collapse)
        diversity_info = {
            "action_diversity_window": self.action_diversity_window.copy(),
            "episode_step": self.episode_step,
            "repeat_streak": self.consecutive_action_count,
        }

        total_reward, components = self.reward_shaper.compute_reward(
            action=int(action),
            action_executed=action_executed,
            prev_equity=prev_equity,
            current_equity=current_equity,
            position_info=position_info,
            trade_info=trade_info,
            portfolio_state=portfolio_state,
            diversity_info=diversity_info,
        )

        self._last_reward.total = total_reward
        self._last_reward.equity = components.get("pnl", 0.0)
        self._last_reward.drawdown = components.get("drawdown", 0.0)
        self._last_reward.action = (
            components.get("transaction_cost", 0.0) + components.get("time_efficiency", 0.0)
        )
        self._last_reward.risk = components.get("sharpe", 0.0)
        self._last_reward_components = components.copy()

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

    def _apply_exploration_curriculum(self) -> float:
        """
        Apply exploration curriculum penalties and Stage 4 coverage controls.

        The curriculum enforces balanced action usage (phases 1 & 2) and, when
        enabled, applies Stage 4 coverage alignment that blends executed and
        policy-selected actions. Stage 4 can scale rewards via a multiplier and
        applies capped, power-scaled penalties when aggregate BUY/SELL coverage
        falls below configured thresholds.

        Returns:
            Negative penalty applied to the reward (0.0 if no adjustment).
        """
        if not self.config.exploration_curriculum_enabled:
            self._curriculum_last_multiplier = 1.0
            self._curriculum_last_detail = {}
            return 0.0

        self._curriculum_last_multiplier = 1.0
        self._curriculum_last_detail = {}

        # Determine current curriculum phase and per-action thresholds
        if self.total_env_steps <= self.config.exploration_phase1_end_step:
            phase = 1
            min_pct = float(self.config.exploration_phase1_min_action_pct)
            base_penalty = float(self.config.exploration_phase1_penalty)
            ramp_duration = min(5000, self.config.exploration_phase1_end_step // 4)
            if ramp_duration > 0:
                ramp_progress = min(1.0, self.total_env_steps / ramp_duration)
                penalty_per_violation = base_penalty * ramp_progress
            else:
                penalty_per_violation = base_penalty
            display_penalty = base_penalty
        elif self.total_env_steps <= self.config.exploration_phase2_end_step:
            phase = 2
            min_pct = float(self.config.exploration_phase2_min_action_pct)
            penalty_per_violation = float(self.config.exploration_phase2_penalty)
            display_penalty = penalty_per_violation
        else:
            phase = 3
            min_pct = 0.0
            penalty_per_violation = 0.0
            display_penalty = float(self.config.exploration_phase2_penalty)
            if self.total_env_steps == self.config.exploration_phase2_end_step + 1:
                logger.info(
                    "🎓 [Curriculum] Entering Phase 3 at step %d - Natural convergence (no constraints)",
                    self.total_env_steps,
                )

        # Log phase transitions once for operator visibility
        if self.total_env_steps == 1:
            sell_requirement = None
            if self.config.exploration_require_sell_actions:
                sell_requirement = self.config.exploration_min_sell_pct * 100
            buy_requirement = None
            if self.config.exploration_require_buy_actions:
                buy_requirement = self.config.exploration_min_buy_pct * 100

            requirement_bits = []
            if sell_requirement is not None:
                requirement_bits.append(f"SELL≥{sell_requirement:.1f}%")
            if buy_requirement is not None:
                requirement_bits.append(f"BUY≥{buy_requirement:.1f}%")
            requirement_str = f" | {'; '.join(requirement_bits)}" if requirement_bits else ""

            phase1_msg = (
                f"🎓 [Curriculum] Phase 1 START - Strict diversity "
                f"(per-action ≥{min_pct*100:.1f}%, penalty up to {display_penalty:.2f}"
                f" | current ramp {penalty_per_violation:.3f}{requirement_str})"
            )
            print(phase1_msg)
            logger.info(phase1_msg.replace("🎓 ", ""))
        elif self.total_env_steps == self.config.exploration_phase1_end_step + 1:
            phase2_msg = (
                f"🎓 [Curriculum] Phase 2 START at step {self.total_env_steps} - Relaxed"
                f" (per-action ≥{min_pct*100:.1f}%, penalty {display_penalty:.2f})"
            )
            print(phase2_msg)
            logger.info(phase2_msg.replace("🎓 ", ""))

        window_size = self.config.exploration_evaluation_window
        min_window_size = min(10, window_size)
        executed_window_size = len(self.curriculum_evaluation_window)
        selected_window_size = len(self.curriculum_selected_window)

        detail: Dict[str, Any] = {
            "phase": int(phase),
            "window_executed": executed_window_size,
            "window_selected": selected_window_size,
            "min_pct": float(min_pct),
            "penalty_base": float(penalty_per_violation),
        }

        if executed_window_size < min_window_size and selected_window_size < min_window_size:
            detail["reason"] = "insufficient_history"
            detail["required_window"] = int(min_window_size)
            self._curriculum_last_detail = detail
            return 0.0

        from collections import Counter

        actual_window_size = executed_window_size
        action_counts = Counter(self.curriculum_evaluation_window)
        selected_counts = Counter(self.curriculum_selected_window)
        detail["window_effective"] = int(actual_window_size)

        penalty_total = 0.0
        aggregated_logs: List[Tuple[str, float, float, float, float]] = []
        violations: List[Dict[str, Any]] = []

        # Aggregated SELL coverage during early phases
        if self.config.exploration_require_sell_actions and phase in (1, 2) and actual_window_size > 0:
            sell_actions = (TradeAction.SELL_PARTIAL.value, TradeAction.SELL_ALL.value)
            sell_count = sum(action_counts.get(idx, 0) for idx in sell_actions)
            sell_pct = sell_count / actual_window_size
            min_sell_pct = max(0.0, float(self.config.exploration_min_sell_pct))
            if sell_pct < min_sell_pct:
                multiplier = float(getattr(self.config, "exploration_sell_penalty_multiplier", 5.0))
                shortfall = max(0.0, min_sell_pct - sell_pct)
                severity = _clip01(shortfall / max(min_sell_pct, 1e-6))
                severity_scale = 0.75 + 1.25 * severity
                sell_penalty = penalty_per_violation * multiplier * severity_scale
                penalty_total += sell_penalty
                self.curriculum_episode_violations += 1
                self.curriculum_episode_penalty += sell_penalty
                aggregated_logs.append(("SELL", sell_pct, min_sell_pct, sell_penalty, severity))

        # Optional aggregated BUY coverage (BUY_* grouped)
        if self.config.exploration_require_buy_actions and phase in (1, 2) and actual_window_size > 0:
            buy_actions = (
                TradeAction.BUY_SMALL.value,
                TradeAction.BUY_MEDIUM.value,
                TradeAction.BUY_LARGE.value,
            )
            buy_count = sum(action_counts.get(idx, 0) for idx in buy_actions)
            buy_pct = buy_count / actual_window_size
            min_buy_pct = max(0.0, float(getattr(self.config, "exploration_min_buy_pct", 0.0)))
            if buy_pct < min_buy_pct:
                multiplier = float(getattr(self.config, "exploration_buy_penalty_multiplier", 3.0))
                shortfall = max(0.0, min_buy_pct - buy_pct)
                severity = _clip01(shortfall / max(min_buy_pct, 1e-6))
                severity_scale = 0.75 + 1.25 * severity
                buy_penalty = penalty_per_violation * multiplier * severity_scale
                penalty_total += buy_penalty
                self.curriculum_episode_violations += 1
                self.curriculum_episode_penalty += buy_penalty
                aggregated_logs.append(("BUY", buy_pct, min_buy_pct, buy_penalty, severity))

        # Per-action minimum usage (excluding HOLD or other excluded actions)
        all_actions = set(range(len(TradeAction)))
        required_actions = all_actions - self.curriculum_excluded_actions
        if actual_window_size > 0 and min_pct > 0.0:
            for action_idx in required_actions:
                action_count = action_counts.get(action_idx, 0)
                action_pct = action_count / actual_window_size
                if action_pct < min_pct:
                    shortfall = max(0.0, min_pct - action_pct)
                    severity = _clip01(shortfall / max(min_pct, 1e-6))
                    severity_scale = 0.6 + 1.4 * severity
                    action_penalty = penalty_per_violation * severity_scale
                    penalty_total += action_penalty
                    self.curriculum_episode_violations += 1
                    self.curriculum_episode_penalty += action_penalty
                    violations.append(
                        {
                            "action": TradeAction(action_idx).name,
                            "count": int(action_count),
                            "pct": float(action_pct),
                            "min_required": float(min_pct),
                            "penalty": float(action_penalty),
                            "severity": float(severity),
                        }
                    )

        if violations and self.episode_step % 500 == 0 and self.episode_step > 0:
            violation_str = ", ".join(
                [
                    f"{v['action']}={v['count']}/{actual_window_size} ({v['pct']:.1%}, need ≥{v['min_required']:.1%}, sev={v['severity']:.2f}, pen={v['penalty']:.3f})"
                    for v in violations[:3]
                ]
            )
            if len(violations) > 3:
                violation_str += f" +{len(violations) - 3} more"
            total_penalty = sum(v["penalty"] for v in violations)
            logger.info(
                "⚠️  [Curriculum] Phase %d (env_step=%d, ep_step=%d): %d category violations → penalty=%.3f | %s",
                phase,
                self.total_env_steps,
                self.episode_step,
                len(violations),
                total_penalty,
                violation_str,
            )

        if aggregated_logs:
            for bucket, pct, minimum_required, bucket_penalty, severity in aggregated_logs:
                logger.debug(
                    "⚠️  [Curriculum] Phase %d (env_step=%d, ep_step=%d): %s usage %.1f%% < %.1f%% → penalty=%.3f (severity=%.2f)",
                    phase,
                    self.total_env_steps,
                    self.episode_step,
                    bucket,
                    pct * 100,
                    minimum_required * 100,
                    bucket_penalty,
                    severity,
                )
            if self.episode_step % 250 == 0 and self.episode_step > 0:
                summary = "; ".join(
                    [
                        f"{bucket} {pct*100:.1f}%< {minimum_required*100:.1f}% (penalty={bucket_penalty:.3f}, sev={severity:.2f})"
                        for bucket, pct, minimum_required, bucket_penalty, severity in aggregated_logs
                    ]
                )
                logger.info(
                    "🚦 [Curriculum] Phase %d (env_step=%d, ep_step=%d): Aggregated coverage penalties → %s",
                    phase,
                    self.total_env_steps,
                    self.episode_step,
                    summary,
                )

        # Stage 4 coverage enforcement (executed + selected weighting)
        coverage_detail: Dict[str, Any] = {
            "active": False,
            "weights": {
                "executed": float(self.curriculum_action_coverage_executed_weight),
                "selected": float(self.curriculum_action_coverage_selected_weight),
            },
            "penalty": 0.0,
        }
        coverage_penalty = 0.0
        coverage_components: List[Dict[str, Any]] = []
        coverage_multiplier = 1.0
        coverage_active = (
            self.curriculum_action_coverage_enabled
            and self.total_env_steps >= self.curriculum_action_coverage_start_step
        )
        if coverage_active and (executed_window_size >= min_window_size or selected_window_size >= min_window_size):
            coverage_detail["active"] = True
            coverage_detail["window_executed"] = int(executed_window_size)
            coverage_detail["window_selected"] = int(selected_window_size)
            coverage_detail["start_step"] = int(self.curriculum_action_coverage_start_step)

            executed_weight = float(self.curriculum_action_coverage_executed_weight)
            selected_weight = float(self.curriculum_action_coverage_selected_weight)

            buy_actions = (
                TradeAction.BUY_SMALL.value,
                TradeAction.BUY_MEDIUM.value,
                TradeAction.BUY_LARGE.value,
            )
            sell_actions = (TradeAction.SELL_PARTIAL.value, TradeAction.SELL_ALL.value)

            executed_buy_pct = _safe_divide(float(sum(action_counts.get(idx, 0) for idx in buy_actions)), float(executed_window_size))
            selected_buy_pct = _safe_divide(float(sum(selected_counts.get(idx, 0) for idx in buy_actions)), float(selected_window_size))
            executed_sell_pct = _safe_divide(float(sum(action_counts.get(idx, 0) for idx in sell_actions)), float(executed_window_size))
            selected_sell_pct = _safe_divide(float(sum(selected_counts.get(idx, 0) for idx in sell_actions)), float(selected_window_size))

            weighted_buy_pct = executed_buy_pct * executed_weight + selected_buy_pct * selected_weight
            weighted_sell_pct = executed_sell_pct * executed_weight + selected_sell_pct * selected_weight

            min_buy = float(self.curriculum_action_coverage_min_buy_pct)
            min_sell = float(self.curriculum_action_coverage_min_sell_pct)
            buy_shortfall = max(0.0, min_buy - weighted_buy_pct)
            sell_shortfall = max(0.0, min_sell - weighted_sell_pct)

            coverage_detail.update(
                {
                    "buy_pct": {
                        "executed": float(executed_buy_pct),
                        "selected": float(selected_buy_pct),
                        "weighted": float(weighted_buy_pct),
                        "min": min_buy,
                    },
                    "sell_pct": {
                        "executed": float(executed_sell_pct),
                        "selected": float(selected_sell_pct),
                        "weighted": float(weighted_sell_pct),
                        "min": min_sell,
                    },
                    "buy_shortfall": float(buy_shortfall),
                    "sell_shortfall": float(sell_shortfall),
                }
            )

            penalty_reference = max(
                abs(float(penalty_per_violation)),
                abs(float(display_penalty)),
                abs(float(self.config.exploration_phase1_penalty)),
                abs(float(self.config.exploration_phase2_penalty)),
                1.0,
            )
            penalty_power = max(1.0, float(self.curriculum_action_coverage_penalty_power))
            penalty_cap = float(self.curriculum_action_coverage_penalty_cap)

            for bucket, shortfall, target in (
                ("buy", buy_shortfall, min_buy),
                ("sell", sell_shortfall, min_sell),
            ):
                if shortfall <= 0.0 or target <= 0.0:
                    continue
                ratio = _clip01(shortfall / max(target, 1e-6))
                penalty_mag = penalty_reference * (ratio ** penalty_power)
                penalty_mag = min(penalty_mag, penalty_cap)
                penalty_component = -float(penalty_mag)
                coverage_penalty += penalty_component
                self.curriculum_episode_violations += 1
                self.curriculum_episode_penalty += penalty_component
                coverage_components.append(
                    {
                        "bucket": bucket,
                        "shortfall": float(shortfall),
                        "target": float(target),
                        "ratio": float(ratio),
                        "penalty": float(penalty_component),
                    }
                )

            if coverage_components:
                coverage_penalty = float(max(-penalty_cap, coverage_penalty))
                coverage_multiplier = float(np.clip(self.curriculum_action_coverage_reward_multiplier, 0.0, 1.0))
                self._curriculum_last_multiplier = coverage_multiplier
                coverage_detail["multiplier"] = float(coverage_multiplier)
                if self.episode_step % 250 == 0 or self.total_env_steps % 2000 == 0:
                    component_summary = "; ".join(
                        f"{c['bucket'].upper()} shortfall {c['shortfall']*100:.1f}% → pen {c['penalty']:.3f}"
                        for c in coverage_components
                    )
                    logger.info(
                        "📉 [Curriculum] Stage 4 coverage penalty (env_step=%d, ep_step=%d): penalty=%.3f, multiplier=%.2f | %s",
                        self.total_env_steps,
                        self.episode_step,
                        coverage_penalty,
                        coverage_multiplier,
                        component_summary,
                    )

            coverage_detail["components"] = coverage_components
            coverage_detail["penalty"] = float(coverage_penalty)
        else:
            if not self.curriculum_action_coverage_enabled:
                coverage_detail["reason"] = "disabled"
            elif self.total_env_steps < self.curriculum_action_coverage_start_step:
                coverage_detail["reason"] = "pre_start_step"
            else:
                coverage_detail["reason"] = "insufficient_window"

        coverage_detail["multiplier"] = float(coverage_multiplier)

        penalty_total += coverage_penalty

        if aggregated_logs:
            detail["aggregated"] = [
                {
                    "bucket": bucket,
                    "pct": float(pct),
                    "min": float(minimum_required),
                    "penalty": float(bucket_penalty),
                    "severity": float(severity),
                }
                for bucket, pct, minimum_required, bucket_penalty, severity in aggregated_logs
            ]
        if violations:
            detail["violations"] = violations

        detail["coverage"] = coverage_detail
        detail["reward_multiplier"] = float(self._curriculum_last_multiplier)

        raw_penalty_total = penalty_total
        clipped_penalty = float(
            np.clip(raw_penalty_total, -self.curriculum_penalty_cap_total, self.curriculum_penalty_cap_total)
        )
        if clipped_penalty != raw_penalty_total:
            adjustment = clipped_penalty - raw_penalty_total
            if adjustment != 0.0:
                self.curriculum_episode_penalty += adjustment
            detail["penalty_clipped"] = True
            detail["penalty_clip_delta"] = float(adjustment)

        detail["penalty_total_raw"] = float(raw_penalty_total)
        detail["penalty_total"] = float(clipped_penalty)
        detail["episode_penalty_running"] = float(self.curriculum_episode_penalty)
        detail["episode_violations_running"] = int(self.curriculum_episode_violations)

        self._curriculum_last_detail = detail

        if clipped_penalty != 0.0:
            return clipped_penalty

        if (
            self.episode_step % 500 == 0
            and self.episode_step > 0
            and self._curriculum_last_multiplier == 1.0
        ):
            logger.info(
                "✅ [Curriculum] Phase %d (env_step=%d, ep_step=%d): Coverage requirements satisfied",
                phase,
                self.total_env_steps,
                self.episode_step,
            )

        return 0.0

    def _get_portfolio_state_dict(self) -> Dict[str, Any]:
        """
        Get portfolio state dictionary for reward calculation.
        
        GOLDEN SHOT FIX: Now includes data and current_step references
        so momentum reward can calculate from actual price history.
        """
        metrics = self.portfolio.get_portfolio_metrics()
        return {
            "equity": metrics["equity"],
            "peak_equity": metrics["peak_equity"],
            "deployed_pct": metrics["exposure_pct"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "num_trades": metrics["total_trades"],
            # GOLDEN SHOT: Add data and current_step for momentum calculation
            "data": self.data,
            "current_step": self.current_step,
        }

    # Portfolio Accounting & Diagnostics

    def _mark_to_market_price(self, price: float, step: int) -> None:
        self.portfolio.update_positions({self.config.symbol: price}, step)
        self.equity_curve = list(self.portfolio.equity_curve)

    def _mark_to_market_current(self) -> None:
        price = self._float_at("close", self.current_step)
        self._mark_to_market_price(price, self.current_step)

    def _get_info(self) -> Dict[str, Any]:
        timestamp = self._timestamp_at(self.current_step)
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

            info["telemetry_selected_voluntary_trades"] = int(self.telemetry_selected_counter)
            info["telemetry_executed_voluntary_trades"] = int(self.telemetry_executed_counter)

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
