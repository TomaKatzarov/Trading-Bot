"""Reward shaping utilities for reinforcement learning trading agents.

This module provides a configurable reward shaping engine that aggregates
multiple objectives to guide agents toward profitable, risk-aware behavior.

Components
---------
1. Profit and loss (PnL) reward: primary trading signal.
2. Transaction cost penalty: penalizes friction from trading activity.
3. Time efficiency reward: encourages quick wins and discourages slow losses.
4. Sharpe ratio contribution: adds risk-adjusted perspective.
5. Drawdown penalty: protects against severe equity declines.
6. Position sizing reward: incentivizes optimal capital deployment.
7. Hold penalty: optional component discouraging excessive inaction.

The reward shaper exposes detailed component tracking and aggregate statistics
for transparency and downstream analysis. All components are normalized to
similar scales for stable learning dynamics.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward function components.

    Component Weights (sum to ~1.0 for interpretability):
    - pnl_weight: Direct profit/loss signal
    - transaction_cost_weight: Penalty for trading friction
    - time_efficiency_weight: Reward for quick wins
    - sharpe_weight: Risk-adjusted return component
    - drawdown_weight: Penalty for severe losses
    - sizing_weight: Reward for optimal position sizing
    - hold_penalty_weight: Discourage excessive holding
    - intrinsic_action_reward: Small reward for valid action execution (exploration boost)

    Stability guards:
    - min_trades_for_sharpe: Suppress Sharpe penalties until enough trades occur
    - neutral_exposure_pct: Exposure treated as neutral for sizing rewards
    - sizing_optimal_low/high: Preferred exposure band rewarded positively
    - sizing_positive_bonus / sizing_moderate_bonus: Reward magnitudes within band
    - sizing_penalty_high: Penalty magnitude when exposure exceeds preferred band
    """

    # Component weights
    pnl_weight: float = 0.45  # Primary signal
    transaction_cost_weight: float = 0.15  # Cost awareness without over-penalizing
    time_efficiency_weight: float = 0.15  # Reward speed
    sharpe_weight: float = 0.05  # Risk-adjusted focus
    drawdown_weight: float = 0.10  # Severe loss penalty
    sizing_weight: float = 0.05  # Capital utilization
    hold_penalty_weight: float = 0.0  # Optional: discourage holding too long
    diversity_bonus_weight: float = 0.0  # Action diversity reward (2025-10-08 Anti-Collapse)
    diversity_penalty_weight: float = 0.0  # Penalty when single action dominates window (2025-10-13 Anti-Collapse v2)
    diversity_penalty_target: float = 0.2  # Target max fraction for any single action bin
    diversity_penalty_window: int = 10  # Window length used when computing diversity penalty
    action_repeat_penalty_weight: float = 0.0  # Penalty for excessive action repetition (2025-10-09 Anti-Collapse v12)
    intrinsic_action_reward: float = 0.0  # Small reward for valid actions (override via config)
    equity_delta_weight: float = 0.0  # Direct equity delta reinforcement (normalized by pnl_scale)

    # Normalization parameters
    pnl_scale: float = 0.01  # Expected PnL scale (1%)
    time_horizon: int = 8  # Max holding period (hours)
    target_sharpe: float = 1.0  # Target Sharpe ratio
    max_drawdown_threshold: float = 0.05  # 5% loss triggers penalty
    roi_multiplier_enabled: bool = True  # Enable ROI-based PnL scaling (2025-10-08 Anti-Collapse)
    roi_scale_factor: float = 2.0  # Multiplier for ROI bonus (higher ROI = more reward)
    roi_gate_floor_scale: float = 0.4  # Fractional scale applied when Sharpe gate closed
    roi_neutral_zone: float = 0.005  # ROI below ~0.5% treated as noise early in training
    roi_negative_scale: float = 0.35  # Downscale small negative ROI penalties to aid exploration
    roi_positive_scale: float = 1.0  # Retain positive ROI magnitude (optionally >1.0 to boost)
    roi_full_penalty_trades: int = 60  # Trades required before full ROI penalty strength applies
    
    # Realized vs Unrealized PnL (2025-10-08 Fix: Only reward closed positions)
    realized_pnl_weight: float = 1.0  # Weight for realized PnL (closed positions) - SET TO 1.0!
    unrealized_pnl_weight: float = 0.0  # Weight for unrealized PnL (open positions) - SET TO 0.0!
    closing_bonus_multiplier: float = 1.0  # Multiplier for closing rewards (default 1.0)
    
    # Position Sizing Rewards (2025-10-08 Professional Trading v3.1)
    # Encourage conservative sizing (risk management over profit maximization)
    position_size_small_multiplier: float = 1.2   # 20% bonus for conservative entry
    position_size_medium_multiplier: float = 1.0  # Neutral
    position_size_large_multiplier: float = 0.8   # 20% penalty for aggressive entry
    
    # Exit Strategy Rewards (2025-10-08 Professional Trading v3.1)
    # Encourage scaling out of winners (partial exits + full exits)
    partial_exit_multiplier: float = 0.8   # 80% reward for partial exit (keeps winners running)
    full_exit_multiplier: float = 1.0      # 100% reward for full exit
    staged_exit_bonus: float = 1.1         # 10% bonus if SELL_PARTIAL → SELL_ALL on same position
    
    # ADD_POSITION Rewards (2025-10-08 Confidence-Based Pyramiding)
    # Allow adding to winning positions when model is highly confident
    add_position_enabled: bool = True                      # Enable ADD_POSITION action
    add_position_min_profit_pct: float = 0.02             # Require 2%+ unrealized profit
    add_position_confidence_threshold: float = 0.75        # Require 75%+ model certainty
    add_position_immediate_reward: float = 0.0            # No immediate reward (like BUY)
    add_position_pyramid_bonus: float = 1.3               # 30% bonus when final SELL closes pyramided position
    add_position_max_adds: int = 2                        # Max 2 additions per position
    add_position_invalid_penalty: float = -1.0            # Penalty for invalid ADD attempts

    # Shaping parameters
    win_bonus_multiplier: float = 1.2  # Extra reward for wins
    loss_penalty_multiplier: float = 1.5  # Extra penalty for losses
    quick_win_bonus: float = 0.5  # Bonus for wins in <4 hours
    early_stop_bonus: float = 0.3  # Bonus for hitting stops cleanly
    manual_exit_bonus: float = 0.05  # Bonus when agent proactively closes positions
    forced_exit_penalty: float = 0.0  # Legacy constant penalty (replaced by Stage 2 scaling)
    forced_exit_base_penalty: float = 0.2  # Base penalty applied when forced exit occurs (Stage 2)
    forced_exit_loss_scale: float = 2.5  # Additional penalty per unit loss percentage (Stage 2)
    forced_exit_penalty_cap: float = 1.5  # Maximum forced exit penalty magnitude

    # Risk management
    max_single_loss: float = 0.02  # 2% max loss per trade
    severe_loss_penalty: float = -5.0  # Large penalty for >2% loss

    # Transaction cost assumptions
    base_transaction_cost_pct: float = 0.001  # 0.10% default cost
    failed_action_penalty: float = -0.05  # Penalty for invalid attempts (override in configs for stricter regimes)

    # Numerical safety
    min_equity: float = 1e-6  # Avoid divide-by-zero in PnL

    # Optional smoothing
    reward_clip: float = 10.0  # Absolute cap for component outputs

    # Stability knobs
    min_trades_for_sharpe: int = 5
    neutral_exposure_pct: float = 0.1
    sizing_optimal_low: float = 0.35
    sizing_optimal_high: float = 0.75
    sizing_positive_bonus: float = 0.25
    sizing_moderate_bonus: float = -0.05
    sizing_penalty_high: float = 0.3

    # Sharpe/ROI gating (Stage 2 reward hygiene)
    sharpe_gate_enabled: bool = False
    sharpe_gate_window: int = 40
    sharpe_gate_min_self_trades: int = 20
    sharpe_gate_floor_scale: float = 0.33
    sharpe_gate_active_scale: float = 1.0
    sharpe_neutral_zone: float = 0.4  # |Sharpe| below this treated as neutral (no penalty)
    sharpe_negative_scale: float = 0.4  # Downscale negative Sharpe penalties
    sharpe_positive_scale: float = 1.0  # Retain positive Sharpe rewards
    sharpe_full_penalty_trades: int = 40  # Trades required before full Sharpe penalty applies

    # Time decay penalty parameters (Stage 2)
    time_decay_threshold_hours: float = 18.0
    time_decay_penalty_per_hour: float = 0.003
    time_decay_max_penalty: float = 0.05

    # ===== PHASE 2: ADVANCED REWARD AUGMENTATIONS (2025-10-13) =====
    # Progressive ROI Scaling (smooth reward curve instead of binary win/loss)
    progressive_roi_enabled: bool = True
    progressive_roi_thresholds: Tuple[float, ...] = (0.05, 0.02, 0.0, -0.01, -0.02)
    progressive_roi_multipliers: Tuple[float, ...] = (3.0, 2.0, 1.5, 0.5, 1.0, 2.0)
    
    # Action Entropy Bonus (anti-collapse mechanism)
    entropy_bonus_weight: float = 0.1
    entropy_bonus_target: float = 2.0
    entropy_bonus_scale: float = 0.5
    
    # Momentum Reward (trend following)
    momentum_weight: float = 0.05
    momentum_window: int = 10
    momentum_max_reward: float = 0.3
    
    # ===== GOLDEN SHOT: STATE-DEPENDENT REWARD SCALING =====
    # Context-aware reward modulation based on trading behavior
    context_scaling_enabled: bool = True
    overtrading_threshold: float = 0.30  # >30% trade rate triggers penalty
    overtrading_penalty_scale: float = 0.5  # 50% reward reduction
    patient_trading_threshold: float = 0.15  # <15% trade rate + positive Sharpe = bonus
    patient_trading_bonus: float = 1.2  # 20% reward bonus
    low_winrate_threshold: float = 0.40  # <40% win rate triggers penalty
    low_winrate_penalty_scale: float = 0.7  # 30% reward reduction
    trade_frequency_penalty_weight: float = 0.0  # Separate penalty for excessive trading
    hold_bonus_weight: float = 0.0  # Bonus for patient holding
    
    # ===== GOLDEN SHOT: ADAPTIVE WIN MULTIPLIER =====
    # Dynamic win bonus based on action diversity and entropy
    adaptive_win_multiplier_enabled: bool = True
    adaptive_win_base_multiplier: float = 1.8  # Base win bonus (reduced from 2.5)
    adaptive_win_max_multiplier: float = 2.5  # Maximum bonus when diverse
    adaptive_win_min_multiplier: float = 1.2  # Minimum bonus when exploiting
    action_concentration_threshold: float = 0.50  # >50% single action = reduce bonus
    high_entropy_threshold: float = 2.5  # Entropy >2.5 = increase bonus
    overtrading_action_threshold: float = 0.30  # >30% actions = reduce bonus

    component_keys: Tuple[str, ...] = field(
        init=False,
        default=(
            "pnl",
            "transaction_cost",
            "time_efficiency",
            "sharpe",
            "drawdown",
            "sizing",
            "hold_penalty",
            "diversity_bonus",  # 2025-10-08 Anti-Collapse
            "diversity_penalty",  # 2025-10-13 Anti-Collapse v2
            "action_repeat_penalty",  # 2025-10-09 Repeat penalty
            "intrinsic_action",  # 2025-10-08 Exploration Fix
            "equity_delta",
            "forced_exit_flag",  # 2025-10-09 Telemetry instrumentation
            "entropy_bonus",  # 2025-10-13 Action diversity bonus
            "momentum_reward",  # 2025-10-13 Trend following reward
        ),
    )

    def __post_init__(self) -> None:
        """Validate configuration values and surface potential issues."""
        if self.pnl_scale <= 0:
            raise ValueError("pnl_scale must be positive to normalize rewards")

        if self.time_horizon <= 0:
            raise ValueError("time_horizon must be positive")

        if self.target_sharpe <= 0:
            logger.warning(
                "target_sharpe is non-positive; Sharpe rewards will be zeroed"
            )

        total_weight = (
            self.pnl_weight
            + self.transaction_cost_weight
            + self.time_efficiency_weight
            + self.sharpe_weight
            + self.drawdown_weight
            + self.sizing_weight
            + self.hold_penalty_weight
            + self.diversity_bonus_weight
            + self.diversity_penalty_weight
            + self.equity_delta_weight
        )

        if not 0.9 <= total_weight <= 1.1:
            logger.warning(
                "Component weights sum to %.3f; consider normalizing near 1.0",
                total_weight,
            )

        if self.base_transaction_cost_pct < 0:
            raise ValueError("base_transaction_cost_pct must be non-negative")

        if self.reward_clip <= 0:
            raise ValueError("reward_clip must be positive")

        if self.sizing_optimal_high < self.sizing_optimal_low:
            logger.warning(
                "sizing_optimal_high (%.3f) is less than sizing_optimal_low (%.3f); swapping",
                self.sizing_optimal_high,
                self.sizing_optimal_low,
            )
            self.sizing_optimal_low, self.sizing_optimal_high = (
                self.sizing_optimal_high,
                self.sizing_optimal_low,
            )

        self.min_trades_for_sharpe = max(0, int(self.min_trades_for_sharpe))

        self.sharpe_gate_window = max(1, int(self.sharpe_gate_window))
        self.sharpe_gate_min_self_trades = max(0, int(self.sharpe_gate_min_self_trades))
        if not 0.0 <= self.sharpe_gate_floor_scale <= self.sharpe_gate_active_scale:
            raise ValueError(
                "sharpe_gate_floor_scale must be within [0, sharpe_gate_active_scale]"
            )
        if self.sharpe_gate_active_scale <= 0:
            raise ValueError("sharpe_gate_active_scale must be positive")

        if not 0.0 <= self.roi_gate_floor_scale <= 1.0:
            raise ValueError("roi_gate_floor_scale must be within [0, 1]")

        self.roi_neutral_zone = max(0.0, float(self.roi_neutral_zone))
        self.roi_negative_scale = max(0.0, float(self.roi_negative_scale))
        self.roi_positive_scale = max(0.0, float(self.roi_positive_scale))
        self.roi_full_penalty_trades = max(1, int(self.roi_full_penalty_trades))
        self.diversity_penalty_target = float(np.clip(self.diversity_penalty_target, 0.0, 1.0))
        self.diversity_penalty_window = max(1, int(self.diversity_penalty_window))
        self.sharpe_neutral_zone = max(0.0, float(self.sharpe_neutral_zone))
        self.sharpe_negative_scale = float(max(0.0, self.sharpe_negative_scale))
        self.sharpe_positive_scale = float(max(0.0, self.sharpe_positive_scale))
        self.sharpe_full_penalty_trades = max(1, int(self.sharpe_full_penalty_trades))

        if self.forced_exit_base_penalty < 0:
            raise ValueError("forced_exit_base_penalty must be non-negative")
        if self.forced_exit_loss_scale < 0:
            raise ValueError("forced_exit_loss_scale must be non-negative")
        if self.forced_exit_penalty_cap < self.forced_exit_base_penalty:
            logger.warning(
                "forced_exit_penalty_cap (%.3f) is less than base penalty (%.3f); adjusting",
                self.forced_exit_penalty_cap,
                self.forced_exit_base_penalty,
            )
            self.forced_exit_penalty_cap = self.forced_exit_base_penalty

        if self.time_decay_threshold_hours < 0:
            raise ValueError("time_decay_threshold_hours must be non-negative")
        if self.time_decay_penalty_per_hour < 0:
            raise ValueError("time_decay_penalty_per_hour must be non-negative")
        if self.time_decay_max_penalty < 0:
            raise ValueError("time_decay_max_penalty must be non-negative")


class RewardShaper:
    """Multi-objective reward calculation with component tracking."""

    def __init__(self, config: RewardConfig) -> None:
        """Initialize reward shaper.

        Args:
            config: Reward configuration specifying component weights and scaling.
        """
        self.config = config

        # Episode tracking
        self.episode_rewards: List[float] = []
        self.component_history: List[Dict[str, float]] = []

        # Historical statistics across episodes
        self.reward_stats: Dict[str, List[float]] = {
            "total_rewards": [],
            "pnl_rewards": [],
            "cost_penalties": [],
            "time_rewards": [],
            "sharpe_rewards": [],
            "drawdown_penalties": [],
            "sizing_rewards": [],
            "hold_penalties": [],
            "diversity_bonuses": [],  # 2025-10-08 Anti-Collapse
            "diversity_penalties": [],  # 2025-10-13 Anti-Collapse v2
            "repeat_penalties": [],   # 2025-10-09 Repeat penalty tracking
            "intrinsic_rewards": [],
            "equity_delta_rewards": [],
            "forced_exit_flags": [],
            "entropy_bonuses": [],  # 2025-10-13 Phase 2
            "momentum_rewards": [],  # 2025-10-13 Phase 2
        }

        gate_window = max(self.config.sharpe_gate_window, self.config.sharpe_gate_min_self_trades)
        self._voluntary_close_window: Deque[int] = deque(maxlen=gate_window)
        self._sharpe_gate_open: bool = False

        logger.info(
            (
                "RewardShaper initialized with weights: PnL=%.2f, Cost=%.2f, Time=%.2f, "
                "Sharpe=%.2f, Drawdown=%.2f, Sizing=%.2f, Hold=%.2f, EquityDelta=%.2f"
            ),
            config.pnl_weight,
            config.transaction_cost_weight,
            config.time_efficiency_weight,
            config.sharpe_weight,
            config.drawdown_weight,
            config.sizing_weight,
            config.hold_penalty_weight,
            config.equity_delta_weight,
        )

    def compute_reward(
        self,
        action: int,
        action_executed: bool,
        prev_equity: float,
        current_equity: float,
        position_info: Optional[Dict] = None,
        trade_info: Optional[Dict] = None,
        portfolio_state: Optional[Dict] = None,
        diversity_info: Optional[Dict] = None,  # 2025-10-08 Anti-Collapse
    ) -> Tuple[float, Dict[str, float]]:
        """Compute total reward from multiple components.

        Args:
            action: Index of action taken (0-6 mapping to environment actions).
            action_executed: Whether the environment executed the action.
            prev_equity: Equity before the step.
            current_equity: Equity after the step.
            position_info: Current position details, if any.
            trade_info: Trade details when a position is closed.
            portfolio_state: Aggregate portfolio metrics available for shaping.

        Returns:
            Tuple consisting of the weighted total reward and a dictionary of
            component contributions before weighting.
        """
        self._update_trade_gate(trade_info)

        components: Dict[str, float] = {}



        components["pnl"] = self._compute_pnl_reward(
            prev_equity=prev_equity,
            current_equity=current_equity,
            trade_info=trade_info,
            position_info=position_info,
            action=action,  # NEW (V3.1): Pass action for size/exit logic
            portfolio_state=portfolio_state,
            diversity_info=diversity_info,  # GOLDEN SHOT: For adaptive win multiplier
        )

        components["transaction_cost"] = self._compute_cost_penalty(
            action=action,
            action_executed=action_executed,
            trade_info=trade_info,
        )

        components["time_efficiency"] = self._compute_time_reward(
            trade_info=trade_info,
            position_info=position_info,
        )

        components["sharpe"] = self._compute_sharpe_reward(
            portfolio_state=portfolio_state,
        )

        components["drawdown"] = self._compute_drawdown_penalty(
            current_equity=current_equity,
            portfolio_state=portfolio_state,
        )

        components["sizing"] = self._compute_sizing_reward(
            position_info=position_info,
            portfolio_state=portfolio_state,
            action=action,
        )

        components["equity_delta"] = self._compute_equity_delta(
            prev_equity=prev_equity,
            current_equity=current_equity,
        )

        if self.config.hold_penalty_weight > 0:
            components["hold_penalty"] = self._compute_hold_penalty(
                action=action,
                position_info=position_info,
            )
        else:
            components["hold_penalty"] = 0.0

        # Action diversity bonus (2025-10-08 Anti-Collapse)
        if self.config.diversity_bonus_weight > 0 and diversity_info is not None:
            components["diversity_bonus"] = self._compute_diversity_bonus(diversity_info)
        else:
            components["diversity_bonus"] = 0.0

        if self.config.diversity_penalty_weight > 0 and diversity_info is not None:
            components["diversity_penalty"] = self._compute_diversity_penalty(diversity_info)
        else:
            components["diversity_penalty"] = 0.0

        if self.config.action_repeat_penalty_weight > 0 and diversity_info is not None:
            components["action_repeat_penalty"] = self._compute_action_repeat_penalty(diversity_info)
        else:
            components["action_repeat_penalty"] = 0.0
        
        # Intrinsic action reward (2025-10-08 Exploration Fix)
        # Give small positive reward for valid action execution (not HOLD)
        # This provides immediate feedback that "doing something" is good
        if self.config.intrinsic_action_reward > 0 and action_executed and action != 0:
            components["intrinsic_action"] = 1.0  # Base bonus for any valid action
        else:
            components["intrinsic_action"] = 0.0

        # Forced exit attribution flag for telemetry (1.0 if manager closed the position)
        if trade_info is not None and trade_info.get("forced_exit"):
            components["forced_exit_flag"] = 1.0
        else:
            components["forced_exit_flag"] = 0.0
        
        # ===== PHASE 2 AUGMENTATIONS (2025-10-13) =====
        # Entropy bonus (anti-collapse)
        if self.config.entropy_bonus_weight > 0 and diversity_info is not None:
            components["entropy_bonus"] = self._compute_entropy_bonus(diversity_info)
        else:
            components["entropy_bonus"] = 0.0
        
        # Momentum reward (trend following)
        if self.config.momentum_weight > 0 and position_info is not None:
            components["momentum_reward"] = self._compute_momentum_reward(position_info, portfolio_state)
        else:
            components["momentum_reward"] = 0.0

        total_reward = self._aggregate_components(components)

        self._record_step(total_reward=total_reward, components=components)

        return total_reward, components

    # ------------------------------------------------------------------
    # Component calculations
    # ------------------------------------------------------------------
    def _compute_pnl_reward(
        self,
        prev_equity: float,
        current_equity: float,
        trade_info: Optional[Dict],
        position_info: Optional[Dict] = None,
        action: Optional[int] = None,
        portfolio_state: Optional[Dict] = None,
        diversity_info: Optional[Dict] = None,
    ) -> float:
        """
        P&L reward: PROFESSIONAL TRADING STRATEGY (2025-10-08 V3.1)
        
        CORE PRINCIPLE: All rewards come from REALIZED profits via SELL actions.
        
        ENHANCED FEATURES (V3.1):
        1. BUY actions: NO immediate reward, but SIZE tracked for later multiplier
           - BUY_SMALL (1): 1.2× final reward (conservative bonus)
           - BUY_MEDIUM (2): 1.0× final reward (neutral)
           - BUY_LARGE (3): 0.8× final reward (aggressive penalty)
        
        2. SELL actions: Profit-scaled rewards with exit strategy bonuses
           - SELL_PARTIAL (4): 0.8× immediate reward, keeps 50% running
           - SELL_ALL (5): 1.0× immediate reward, closes everything
           - Staged exit (PARTIAL→ALL): 1.1× bonus on final exit
        
        3. HOLD action: Context-dependent penalties
           - Winning position: -0.01 per step (encourages selling)
           - Losing position: -0.005 per step (encourages cutting)
           - No position: 0.0 (neutral)
        
        4. ADD_POSITION (6): Confidence-based pyramiding
           - Requires 2%+ profit AND 75%+ model certainty
           - No immediate reward (like BUY)
           - 1.3× bonus on final SELL (pyramiding bonus)
           - Invalid attempts: -1.0 penalty
        
        Reward flow examples:
        - BUY_SMALL → HOLD → SELL_ALL at +5%: +5% × 1.2 (size) × 1.0 (full) = +6.0
        - BUY_SMALL → SELL_PARTIAL at +5% → SELL_ALL at +8%: 
            +5% × 1.2 × 0.8 + 8% × 1.2 × 1.1 = +4.8 + 10.56 = +15.36!
        - BUY_LARGE → SELL_ALL at +5%: +5% × 0.8 (size) × 1.0 (full) = +4.0
        """
        
        if prev_equity <= self.config.min_equity and current_equity <= self.config.min_equity:
            return -self.config.reward_clip

        # ===== CASE 1: SELL Action - Position Closed =====
        if trade_info is not None:
            # Position closed - ONLY source of rewards!
            realized_pnl_pct = trade_info.get("pnl_pct", 0.0)
            trades_completed = float(portfolio_state.get("num_trades", 0.0)) if portfolio_state else 0.0
            adjusted_roi_pct = self._apply_roi_shaping(realized_pnl_pct, trades_completed)
            normalized_pnl = adjusted_roi_pct / self.config.pnl_scale
            
            # ===== GOLDEN SHOT: ADAPTIVE WIN MULTIPLIER =====
            # Compute dynamic win bonus based on action diversity and behavior
            adaptive_win_multiplier = self._compute_adaptive_win_multiplier(diversity_info, portfolio_state)
            
            # Scale reward with profit size
            if normalized_pnl > 0:
                # Profitable trade - BIG reward! (use adaptive multiplier)
                base_reward = normalized_pnl * adaptive_win_multiplier
            else:
                # Losing trade - penalty (should have held longer or cut earlier)
                base_reward = normalized_pnl * self.config.loss_penalty_multiplier
            
            # Severe loss penalty
            if realized_pnl_pct < -self.config.max_single_loss:
                base_reward += self.config.severe_loss_penalty
            
            # ===== POSITION SIZE MULTIPLIER (V3.1) =====
            # Reward conservative sizing (BUY_SMALL > BUY_MEDIUM > BUY_LARGE)
            entry_size = trade_info.get("entry_size", "medium")
            size_multiplier = {
                "small": self.config.position_size_small_multiplier,   # 1.2× (20% bonus)
                "medium": self.config.position_size_medium_multiplier, # 1.0× (neutral)
                "large": self.config.position_size_large_multiplier,   # 0.8× (20% penalty)
            }.get(entry_size, 1.0)
            
            base_reward = base_reward * size_multiplier
            
            # ===== EXIT STRATEGY MULTIPLIER (V3.1) =====
            # Differentiate SELL_PARTIAL vs SELL_ALL
            exit_type = trade_info.get("exit_type", "full")
            
            if exit_type == "partial":
                # SELL_PARTIAL: 80% reward but keeps position running
                exit_multiplier = self.config.partial_exit_multiplier  # 0.8×
            elif exit_type == "staged":
                # Staged exit (PARTIAL → ALL): Bonus for professional trading!
                exit_multiplier = self.config.staged_exit_bonus  # 1.1× (10% bonus)
            else:
                # SELL_ALL: Full reward
                exit_multiplier = self.config.full_exit_multiplier  # 1.0×
            
            base_reward = base_reward * exit_multiplier
            
            # ===== PYRAMIDING BONUS (V3.1) =====
            # Bonus if position was added to via ADD_POSITION
            pyramid_count = trade_info.get("pyramid_count", 0)
            if pyramid_count > 0:
                # 30% bonus for successfully pyramiding winners
                base_reward = base_reward * self.config.add_position_pyramid_bonus  # 1.3×

            # ===== ROI MULTIPLIER (2025-10-08 Anti-Collapse) =====
            if self.config.roi_multiplier_enabled:
                roi_multiplier = self._roi_multiplier(adjusted_roi_pct, trades_completed)
                base_reward = base_reward * roi_multiplier
            
            # Apply closing bonus multiplier
            final_reward = base_reward * self.config.closing_bonus_multiplier
            
            # Weight by realized_pnl_weight (should be 1.0)
            final_reward = final_reward * self.config.realized_pnl_weight
            
            # ===== GOLDEN SHOT: CONTEXT SCALING =====
            # Apply state-dependent scaling based on trading behavior
            final_reward = self._apply_context_scaling(final_reward, portfolio_state, trade_info)
            
            logger.debug(
                "REALIZED PnL (SELL): pnl_pct=%.4f, size=%s (×%.2f), exit=%s (×%.2f), "
                "pyramid=%d (×%.2f), final=%.4f",
                realized_pnl_pct, entry_size, size_multiplier, exit_type, exit_multiplier,
                pyramid_count, self.config.add_position_pyramid_bonus if pyramid_count > 0 else 1.0,
                final_reward
            )
            
            return float(np.clip(final_reward, -self.config.reward_clip, self.config.reward_clip))
        
        # ===== CASE 2: HOLD/BUY/ADD - No Position Closed =====
        else:
            # Check if we have unrealized PnL (position open)
            if position_info is not None:
                # Position is open - apply HOLD logic
                # Get unrealized PnL from equity change
                equity_change = current_equity - max(prev_equity, self.config.min_equity)
                unrealized_pnl_pct = equity_change / max(prev_equity, self.config.min_equity)
                
                # HOLD reward logic (INDEPENDENT of unrealized_pnl_weight):
                # - If position profitable: small penalty (encourages selling)
                # - If position losing: small penalty (encourages cutting losses)
                if unrealized_pnl_pct > 0.01:  # Position up >1%
                    # Penalize holding winners (should sell to realize)
                    # Use small fixed penalty to create urgency without overwhelming signal
                    return -0.01
                elif unrealized_pnl_pct < -0.01:  # Position down >1%
                    # Small penalty for holding losers (encourage cutting)
                    return -0.005
                else:
                    # Small movement, neutral
                    return 0.0
            else:
                # No position open (BUY action or HOLD with no position)
                # NO reward - opening positions doesn't earn anything
                return 0.0

    def _compute_cost_penalty(
        self,
        action: int,
        action_executed: bool,
        trade_info: Optional[Dict],
    ) -> float:
        """Transaction cost penalty: explicit cost awareness."""
        if not action_executed:
            return self.config.failed_action_penalty

        if action == 0:
            return 0.0

        cost_pct = self.config.base_transaction_cost_pct
        normalized_cost = -cost_pct / self.config.pnl_scale

        if action in {1, 2, 3, 4, 5}:  # Opening or closing actions
            reward = normalized_cost
        elif action == 6:  # Add position, slightly more expensive
            reward = -1.5 * cost_pct / self.config.pnl_scale
        else:
            reward = normalized_cost

        if action in {4, 5} and trade_info:
            pnl_pct = trade_info.get("pnl_pct", 0.0)
            if np.isfinite(pnl_pct):
                if abs(pnl_pct - 0.025) < 0.005:
                    reward += self.config.early_stop_bonus
                elif abs(pnl_pct + 0.02) < 0.005:
                    reward += 0.5 * self.config.early_stop_bonus

        return float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))

    def _compute_time_reward(
        self,
        trade_info: Optional[Dict],
        position_info: Optional[Dict],
    ) -> float:
        """Time efficiency reward: encourage quick profitable exits."""
        if not trade_info:
            return self._compute_time_decay_penalty(position_info)

        holding_hours = float(trade_info.get("holding_hours", 0.0))
        pnl_pct = float(trade_info.get("pnl_pct", 0.0))

        if trade_info.get("forced_exit"):
            penalty = self._forced_exit_penalty_value(pnl_pct)
            return float(np.clip(penalty, -self.config.reward_clip, 0.0))

        exit_reason = str(trade_info.get("action", "")) if trade_info else ""

        if pnl_pct > 0 and holding_hours < 4:
            multiplier = max(0.0, 1.0 - holding_hours / 4.0)
            reward = self.config.quick_win_bonus * multiplier
            return float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))

        if pnl_pct < 0 and holding_hours > 6:
            excess_hours = holding_hours - 6.0
            penalty = -0.3 * (excess_hours / self.config.time_horizon)
            return float(np.clip(penalty, -self.config.reward_clip, self.config.reward_clip))

        if (
            position_info
            and position_info.get("is_open", False)
            and holding_hours >= self.config.time_horizon
        ):
            reward = -0.1 * (holding_hours - self.config.time_horizon) / self.config.time_horizon
            return float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))

        if exit_reason:
            if exit_reason.startswith("agent_"):
                return float(np.clip(self.config.manual_exit_bonus, -self.config.reward_clip, self.config.reward_clip))

            penalty = -abs(float(self.config.forced_exit_penalty))
            return float(np.clip(penalty, -self.config.reward_clip, 0.0))

        return 0.0

    def _compute_sharpe_reward(self, portfolio_state: Optional[Dict]) -> float:
        """Sharpe ratio reward: encourage risk-adjusted returns."""
        if not portfolio_state:
            return 0.0
        trades = float(portfolio_state.get("num_trades", self.config.min_trades_for_sharpe))

        if trades < self.config.min_trades_for_sharpe:
            return 0.0

        sharpe = float(portfolio_state.get("sharpe_ratio", 0.0))
        if not np.isfinite(sharpe) or self.config.target_sharpe <= 0:
            return 0.0

        # Neutral band: treat near-zero Sharpe as noise to avoid hammering the policy early
        if abs(sharpe) <= self.config.sharpe_neutral_zone:
            scaled = 0.0
        else:
            sharpe_delta = (sharpe - self.config.target_sharpe) / max(1e-6, self.config.target_sharpe)
            if sharpe_delta < 0:
                scaled = sharpe_delta * self.config.sharpe_negative_scale
            else:
                scaled = sharpe_delta * self.config.sharpe_positive_scale

        if scaled == 0.0:
            return 0.0

        # Gradually ramp penalty as trade count increases to avoid early-trade cliffs
        penalty_scale = min(1.0, trades / float(self.config.sharpe_full_penalty_trades))
        scaled *= penalty_scale

        if self.config.sharpe_gate_enabled and not self._sharpe_gate_open:
            scaled *= self.config.sharpe_gate_floor_scale
        else:
            scaled *= self.config.sharpe_gate_active_scale

        return float(np.clip(scaled, -2.0, 2.0))

    def _compute_drawdown_penalty(
        self,
        current_equity: float,
        portfolio_state: Optional[Dict],
    ) -> float:
        """Drawdown penalty: severe penalty for large losses."""
        if not portfolio_state:
            return 0.0

        peak_equity = float(portfolio_state.get("peak_equity", current_equity))
        peak_equity = max(peak_equity, self.config.min_equity)

        drawdown = (peak_equity - current_equity) / peak_equity
        drawdown = max(drawdown, 0.0)

        if drawdown < self.config.max_drawdown_threshold:
            return 0.0

        excess_dd = drawdown - self.config.max_drawdown_threshold
        penalty = -10.0 * (excess_dd / 0.05) ** 2
        penalty = max(penalty, -20.0)

        return float(np.clip(penalty, -self.config.reward_clip, 0.0))

    def _compute_sizing_reward(
        self,
        position_info: Optional[Dict],
        portfolio_state: Optional[Dict],
        action: int,
    ) -> float:
        """Position sizing reward: encourage optimal capital utilization."""
        if action == 0 or not portfolio_state:
            return 0.0

        deployed_pct = float(portfolio_state.get("deployed_pct", 0.0))

        neutral = max(0.0, float(self.config.neutral_exposure_pct))
        optimal_low = max(neutral, float(self.config.sizing_optimal_low))
        optimal_high = max(optimal_low, float(self.config.sizing_optimal_high))
        bonus_moderate = float(self.config.sizing_moderate_bonus)
        bonus_optimal = float(self.config.sizing_positive_bonus)
        penalty_high = max(0.0, float(self.config.sizing_penalty_high))

        if deployed_pct < neutral:
            reward = 0.0
        elif neutral <= deployed_pct < optimal_low:
            reward = bonus_moderate
        elif optimal_low <= deployed_pct <= optimal_high:
            reward = bonus_optimal
        elif optimal_high < deployed_pct <= 1.0:
            reward = -penalty_high
        else:
            reward = -2.0 * penalty_high if penalty_high > 0 else -0.3

        return float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))

    def _compute_equity_delta(
        self,
        prev_equity: float,
        current_equity: float,
    ) -> float:
        """Normalize raw equity changes to penalize churn-induced drawdowns."""

        baseline = max(prev_equity, self.config.min_equity)
        if baseline <= 0:
            return 0.0

        delta_pct = (current_equity - prev_equity) / baseline
        if not np.isfinite(delta_pct):
            return 0.0

        normalized = delta_pct / max(self.config.pnl_scale, 1e-9)
        return float(np.clip(normalized, -self.config.reward_clip, self.config.reward_clip))

    def _compute_hold_penalty(
        self,
        action: int,
        position_info: Optional[Dict],
    ) -> float:
        """Hold penalty: discourage excessive holding without action."""
        if action != 0 or not position_info:
            return 0.0

        if not position_info.get("is_open", False):
            return 0.0

        holding_hours = float(position_info.get("duration", 0.0))
        if holding_hours <= 6.0:
            return 0.0

        penalty = -0.1 * (holding_hours - 6.0) / self.config.time_horizon
        return float(np.clip(penalty, -self.config.reward_clip, 0.0))

    def _compute_diversity_bonus(
        self,
        diversity_info: Dict,
    ) -> float:
        """
        Reward action diversity to prevent policy collapse (2025-10-08 Anti-Collapse #2).
        
        Encourages agents to use multiple actions rather than collapsing to a single action.
        Analyzes the 50-step rolling window of actions and awards tiered bonuses:
        - 5+ unique actions: +0.3 (excellent diversity)
        - 4 unique actions: +0.2 (good diversity)
        - 3 unique actions: +0.1 (acceptable diversity)
        - <3 unique actions: 0.0 (poor diversity, potential collapse)
        
        This works synergistically with action repetition limits (max 3 consecutive)
        to prevent 99%+ single-action collapse scenarios.
        
        Args:
            diversity_info: Dict containing "action_diversity_window" (list of recent actions)
        
        Returns:
            Bonus reward in [0.0, 0.3]
        """
        window = diversity_info.get("action_diversity_window", [])
        
        # Need minimum history for meaningful diversity calculation
        if len(window) < 10:
            return 0.0
        
        # Count unique actions in window
        unique_actions = len(set(window))
        
        # Tiered bonuses for diversity
        if unique_actions >= 5:
            bonus = 0.3
            logger.debug("Diversity bonus: %d unique actions in window → +%.2f", unique_actions, bonus)
            return bonus
        elif unique_actions == 4:
            bonus = 0.2
            logger.debug("Diversity bonus: %d unique actions in window → +%.2f", unique_actions, bonus)
            return bonus
        elif unique_actions == 3:
            bonus = 0.1
            logger.debug("Diversity bonus: %d unique actions in window → +%.2f", unique_actions, bonus)
            return bonus
        else:
            # <3 unique actions indicates potential collapse
            logger.debug("Diversity bonus: %d unique actions in window → 0.0 (low diversity)", unique_actions)
            return 0.0

    def _compute_action_repeat_penalty(
        self,
        diversity_info: Dict,
    ) -> float:
        """Penalise excessive repetition of the same action."""

        repeat_streak = int(diversity_info.get("repeat_streak", 1) or 1)
        if repeat_streak <= 2:
            return 0.0

        # Scale penalty with streak length; cap to avoid overwhelming PnL
        penalty = 0.05 * (repeat_streak - 2)

        window = diversity_info.get("action_diversity_window", []) or []
        if window:
            most_common = max(window.count(action) for action in set(window))
            dominance = most_common / len(window)
            penalty *= 1.0 + dominance  # Increase penalty when one action dominates

        penalty = min(penalty, 0.4)
        return float(-np.clip(penalty, 0.0, self.config.reward_clip))

    def _compute_diversity_penalty(
        self,
        diversity_info: Dict,
    ) -> float:
        """Penalise action collapse when a single bin dominates."""

        window_raw = diversity_info.get("action_diversity_window") or []
        if not window_raw:
            return 0.0

        window = list(window_raw[-self.config.diversity_penalty_window :])
        if len(window) < 2:
            return 0.0

        actions = np.asarray(window, dtype=int)
        counts = np.bincount(actions, minlength=int(actions.max() + 1))
        max_fraction = float(np.max(counts)) / float(len(actions))

        excess = max(0.0, max_fraction - self.config.diversity_penalty_target)
        if excess <= 0.0:
            return 0.0

        normaliser = max(1e-6, 1.0 - self.config.diversity_penalty_target)
        penalty = -min(1.0, excess / normaliser)

        logger.debug(
            "Diversity penalty: window=%s max_fraction=%.3f target=%.3f → %.3f",
            window,
            max_fraction,
            self.config.diversity_penalty_target,
            penalty,
        )

        return float(np.clip(penalty, -1.0, 0.0))
    
    def _compute_entropy_bonus(self, diversity_info: Dict) -> float:
        """Reward maintaining action diversity to prevent collapse (Phase 2)."""
        window_raw = diversity_info.get("action_diversity_window") or []
        if not window_raw or len(window_raw) < 5:
            return 0.0
        
        # Calculate entropy of action distribution
        actions = np.asarray(list(window_raw), dtype=int)
        counts = np.bincount(actions, minlength=int(actions.max() + 1))
        probs = counts / float(len(actions))
        
        # Compute Shannon entropy: -sum(p * log(p))
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log(p + 1e-10)
        
        # Scale bonus based on how close to target entropy
        target = self.config.entropy_bonus_target
        if entropy >= target:
            return self.config.entropy_bonus_scale  # Max bonus
        else:
            return self.config.entropy_bonus_scale * (entropy / target)
    
    def _compute_momentum_reward(
        self, 
        position_info: Optional[Dict],
        portfolio_state: Optional[Dict]
    ) -> float:
        """
        Reward trading with the trend (Phase 2 - GOLDEN SHOT FIXED).
        
        CRITICAL FIX: Calculate momentum from actual price data passed via portfolio_state,
        not from price_history (which wasn't populated). This fixes the 0.0 momentum bug.
        """
        if not position_info or not position_info.get("is_open", False):
            return 0.0
        
        # GOLDEN SHOT FIX: Extract data and current_step from portfolio_state
        if not portfolio_state:
            return 0.0
            
        data = portfolio_state.get("data")
        current_step = portfolio_state.get("current_step")
        
        if data is None or current_step is None:
            return 0.0
        
        # Calculate momentum from actual price data
        try:
            if current_step < self.config.momentum_window:
                return 0.0  # Not enough history
            
            # Try lowercase 'close' first (normalized), then uppercase 'Close' (raw)
            close_col = 'close' if 'close' in data.columns else 'Close'
            current_price = float(data.iloc[current_step][close_col])
            past_price = float(data.iloc[current_step - self.config.momentum_window][close_col])
            
            if past_price <= 0:
                return 0.0
            
            momentum = (current_price - past_price) / past_price
            
            # Reward alignment with momentum
            is_long = position_info.get("is_long", True)
            
            if is_long and momentum > 0:
                # Long position in uptrend - reward
                reward = min(self.config.momentum_max_reward, momentum * 10.0)
                return reward
            elif is_long and momentum < 0:
                # Long position in downtrend - penalize
                penalty = max(-self.config.momentum_max_reward, momentum * 10.0)
                return penalty
            elif not is_long and momentum < 0:
                # Short position in downtrend - reward
                reward = min(self.config.momentum_max_reward, abs(momentum) * 10.0)
                return reward
            elif not is_long and momentum > 0:
                # Short position in uptrend - penalize
                penalty = max(-self.config.momentum_max_reward, -momentum * 10.0)
                return penalty
        except (KeyError, IndexError, ValueError) as e:
            # Silently handle errors (data access issues)
            return 0.0
        
        return 0.0

    def _update_trade_gate(self, trade_info: Optional[Dict]) -> None:
        """Update voluntary trade tracking for Sharpe/ROI gating."""

        if not self.config.sharpe_gate_enabled:
            return

        if trade_info is None:
            return

        forced_exit = bool(trade_info.get("forced_exit", False))
        self._voluntary_close_window.append(0 if forced_exit else 1)

        voluntary_closes = sum(self._voluntary_close_window)
        self._sharpe_gate_open = voluntary_closes >= self.config.sharpe_gate_min_self_trades

    def _trade_penalty_scale(self, trades_completed: float, full_penalty_trades: int) -> float:
        """Return scaling factor for penalties based on trade count."""

        if full_penalty_trades <= 0:
            return 1.0
        return float(min(1.0, max(0.0, trades_completed) / float(full_penalty_trades)))

    def _apply_roi_shaping(self, realized_pnl_pct: float, trades_completed: float) -> float:
        """Apply neutral-zone and trade-count scaling to realized ROI."""
        
        # Use progressive ROI scaling if enabled (Phase 2 enhancement)
        if self.config.progressive_roi_enabled:
            return self._progressive_roi_reward(realized_pnl_pct)

        roi = float(realized_pnl_pct)
        neutral = max(0.0, float(self.config.roi_neutral_zone))

        if roi >= 0.0:
            # Optionally boost positive ROI; no neutral suppression to preserve signal discovery.
            return roi * float(max(0.0, self.config.roi_positive_scale))

        magnitude = abs(roi)
        if magnitude <= neutral:
            return 0.0

        excess = magnitude - neutral
        penalty_scale = self._trade_penalty_scale(trades_completed, int(self.config.roi_full_penalty_trades))
        adjusted = -excess * float(max(0.0, self.config.roi_negative_scale))
        return adjusted * penalty_scale
    
    def _progressive_roi_reward(self, roi_pct: float) -> float:
        """Smooth reward curve instead of binary win/loss cliff (Phase 2)."""
        thresholds = self.config.progressive_roi_thresholds
        multipliers = self.config.progressive_roi_multipliers
        
        # Find which bracket roi_pct falls into
        for i, threshold in enumerate(thresholds):
            if roi_pct > threshold:
                return roi_pct * multipliers[i]
        
        # If below all thresholds, use the last multiplier
        return roi_pct * multipliers[-1]
    
    def _apply_context_scaling(
        self,
        base_reward: float,
        portfolio_state: Optional[Dict],
        trade_info: Optional[Dict],
    ) -> float:
        """
        Apply state-dependent reward scaling based on trading behavior (Golden Shot).
        
        Scaling rules:
        1. Over-trading penalty: >30% trade rate → 50% reward reduction
        2. Patient trading bonus: <15% trade rate + positive Sharpe → 20% bonus
        3. Low win rate penalty: <40% win rate → 30% reward reduction
        
        Args:
            base_reward: Unscaled reward value
            portfolio_state: Portfolio metrics dictionary
            trade_info: Trade information dictionary (if position closed)
            
        Returns:
            Scaled reward value
        """
        if not self.config.context_scaling_enabled or portfolio_state is None:
            return base_reward
        
        num_trades = float(portfolio_state.get("num_trades", 0.0))
        sharpe = float(portfolio_state.get("sharpe_ratio", 0.0))
        
        # Need minimum trades for meaningful statistics
        if num_trades < 10:
            return base_reward
        
        scaling_factor = 1.0
        
        # Rule 1: Penalize over-trading (>30% trade rate in episode)
        # Estimate trade rate from num_trades (assumes typical episode ~1000-1500 steps)
        # Conservative estimate: assume 1500 steps to avoid false positives
        assumed_episode_length = 1500.0
        trade_rate = num_trades / assumed_episode_length
        
        if trade_rate > self.config.overtrading_threshold:
            # Over-trading detected - reduce rewards to discourage spam
            scaling_factor *= self.config.overtrading_penalty_scale  # 50% reduction
        
        # Rule 2: Reward patient trading (<15% trade rate + positive Sharpe)
        elif trade_rate < self.config.patient_trading_threshold and sharpe > 0.0:
            # Patient profitable trading - boost rewards!
            scaling_factor *= self.config.patient_trading_bonus  # 20% bonus
        
        # Rule 3: Penalize low win rate (<40%)
        # Win rate tracking would require episode-level state, so we approximate
        # using Sharpe: negative Sharpe suggests poor win rate
        if sharpe < -0.5:  # Proxy for low win rate
            scaling_factor *= self.config.low_winrate_penalty_scale  # 30% reduction
        
        return base_reward * scaling_factor
    
    def _compute_adaptive_win_multiplier(
        self,
        diversity_info: Optional[Dict],
        portfolio_state: Optional[Dict],
    ) -> float:
        """
        Compute dynamic win bonus multiplier based on agent behavior (Golden Shot).
        
        Adjustments:
        1. Reduce multiplier when action concentration >50% (anti-bin-19 spam)
        2. Increase multiplier when action entropy >2.5 (reward diversity)
        3. Reduce multiplier when over-trading >30% (discourage spam)
        
        Returns:
            Multiplier value between adaptive_win_min_multiplier and adaptive_win_max_multiplier
        """
        if not self.config.adaptive_win_multiplier_enabled:
            return self.config.adaptive_win_base_multiplier
        
        # Start with base multiplier
        multiplier = self.config.adaptive_win_base_multiplier
        
        if diversity_info is None:
            return multiplier
        
        window_raw = diversity_info.get("action_diversity_window") or []
        if not window_raw or len(window_raw) < 10:
            return multiplier  # Need minimum sample
        
        actions = np.asarray(list(window_raw), dtype=int)
        
        # Rule 1: Penalize action concentration (bin-19 spam detection)
        if len(actions) > 0:
            counts = np.bincount(actions)
            max_count = int(np.max(counts))
            concentration = max_count / len(actions)
            
            if concentration > self.config.action_concentration_threshold:
                # Heavy concentration detected - reduce win bonus
                excess = concentration - self.config.action_concentration_threshold
                penalty = 1.0 - (excess * 2.0)  # 2× penalty rate
                multiplier *= max(0.5, penalty)  # At least 50% of base
        
        # Rule 2: Reward high entropy (action diversity)
        counts = np.bincount(actions, minlength=int(actions.max() + 1))
        probs = counts / float(len(actions))
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log(p + 1e-10)
        
        if entropy > self.config.high_entropy_threshold:
            # High diversity - boost win bonus!
            bonus = 1.0 + ((entropy - self.config.high_entropy_threshold) * 0.2)
            multiplier *= min(1.5, bonus)  # Max 50% boost
        
        # Rule 3: Penalize over-trading
        if portfolio_state is not None:
            num_trades = float(portfolio_state.get("num_trades", 0.0))
            if num_trades >= 10:
                # Estimate trade rate (same logic as context scaling)
                assumed_episode_length = 1500.0
                trade_rate = num_trades / assumed_episode_length
                
                if trade_rate > self.config.overtrading_action_threshold:
                    # Over-trading - reduce win bonus
                    excess = trade_rate - self.config.overtrading_action_threshold
                    penalty = 1.0 - (excess * 1.5)  # 1.5× penalty rate
                    multiplier *= max(0.7, penalty)  # At least 70% of current
        
        # Clamp to configured range
        multiplier = np.clip(
            multiplier,
            self.config.adaptive_win_min_multiplier,
            self.config.adaptive_win_max_multiplier
        )
        
        return float(multiplier)

    def _roi_multiplier(self, adjusted_roi_pct: float, trades_completed: float) -> float:
        """Compute ROI multiplier with Sharpe gate and trade-count scaling."""

        scale = self.config.roi_scale_factor
        if (
            self.config.sharpe_gate_enabled
            and not self._sharpe_gate_open
            and adjusted_roi_pct > 0.0
        ):
            scale = self.config.roi_scale_factor * self.config.roi_gate_floor_scale

        multiplier_input = adjusted_roi_pct
        if adjusted_roi_pct < 0.0:
            penalty_scale = self._trade_penalty_scale(trades_completed, int(self.config.roi_full_penalty_trades))
            multiplier_input = adjusted_roi_pct * penalty_scale

        roi_multiplier = 1.0 + (scale * multiplier_input)
        return max(0.2, roi_multiplier)

    def _forced_exit_penalty_value(self, pnl_pct: float) -> float:
        """Compute penalty for forced exits based on realized loss."""

        penalty = self.config.forced_exit_base_penalty
        penalty += max(0.0, -pnl_pct) * self.config.forced_exit_loss_scale
        penalty = min(penalty, self.config.forced_exit_penalty_cap)

        if self.config.forced_exit_penalty < 0:
            penalty += abs(self.config.forced_exit_penalty)

        penalty = min(penalty, self.config.reward_clip)
        return -penalty

    def _compute_time_decay_penalty(self, position_info: Optional[Dict]) -> float:
        """Apply a small decay penalty when positions overstay their welcome."""

        if not position_info or not position_info.get("is_open", False):
            return 0.0

        duration_hours = float(position_info.get("duration", 0.0))
        if duration_hours <= self.config.time_decay_threshold_hours:
            return 0.0

        excess_hours = duration_hours - self.config.time_decay_threshold_hours
        penalty = excess_hours * self.config.time_decay_penalty_per_hour
        penalty = min(penalty, self.config.time_decay_max_penalty)
        if penalty <= 0:
            return 0.0

        return float(np.clip(-penalty, -self.config.reward_clip, 0.0))

    # ------------------------------------------------------------------
    # Aggregation & tracking helpers
    # ------------------------------------------------------------------
    def _aggregate_components(self, components: Dict[str, float]) -> float:
        """Aggregate individual component scores into a weighted reward."""
        total = (
            self.config.pnl_weight * components.get("pnl", 0.0)
            + self.config.transaction_cost_weight * components.get("transaction_cost", 0.0)
            + self.config.time_efficiency_weight * components.get("time_efficiency", 0.0)
            + self.config.sharpe_weight * components.get("sharpe", 0.0)
            + self.config.drawdown_weight * components.get("drawdown", 0.0)
            + self.config.sizing_weight * components.get("sizing", 0.0)
            + self.config.hold_penalty_weight * components.get("hold_penalty", 0.0)
            + self.config.diversity_bonus_weight * components.get("diversity_bonus", 0.0)  # 2025-10-08 Anti-Collapse Fix
            + self.config.diversity_penalty_weight * components.get("diversity_penalty", 0.0)  # 2025-10-13 Anti-Collapse v2
            + self.config.action_repeat_penalty_weight * components.get("action_repeat_penalty", 0.0)
            + self.config.intrinsic_action_reward * components.get("intrinsic_action", 0.0)  # 2025-10-08 Exploration Fix
            + self.config.equity_delta_weight * components.get("equity_delta", 0.0)
            + self.config.entropy_bonus_weight * components.get("entropy_bonus", 0.0)  # 2025-10-13 Phase 2
            + self.config.momentum_weight * components.get("momentum_reward", 0.0)  # 2025-10-13 Phase 2
        )
        return float(np.clip(total, -self.config.reward_clip, self.config.reward_clip))

    def _record_step(self, total_reward: float, components: Dict[str, float]) -> None:
        """Record step-level rewards for later analysis."""
        self.episode_rewards.append(total_reward)
        self.component_history.append(components.copy())

        self.reward_stats["total_rewards"].append(total_reward)
        self.reward_stats["pnl_rewards"].append(components.get("pnl", 0.0))
        self.reward_stats["cost_penalties"].append(components.get("transaction_cost", 0.0))
        self.reward_stats["time_rewards"].append(components.get("time_efficiency", 0.0))
        self.reward_stats["sharpe_rewards"].append(components.get("sharpe", 0.0))
        self.reward_stats["drawdown_penalties"].append(components.get("drawdown", 0.0))
        self.reward_stats["sizing_rewards"].append(components.get("sizing", 0.0))
        self.reward_stats["hold_penalties"].append(components.get("hold_penalty", 0.0))
        self.reward_stats["diversity_bonuses"].append(components.get("diversity_bonus", 0.0))  # 2025-10-08 Anti-Collapse
        self.reward_stats["diversity_penalties"].append(components.get("diversity_penalty", 0.0))
        self.reward_stats["repeat_penalties"].append(components.get("action_repeat_penalty", 0.0))
        self.reward_stats["intrinsic_rewards"].append(components.get("intrinsic_action", 0.0))
        self.reward_stats["equity_delta_rewards"].append(components.get("equity_delta", 0.0))
        self.reward_stats["forced_exit_flags"].append(components.get("forced_exit_flag", 0.0))
        self.reward_stats["entropy_bonuses"].append(components.get("entropy_bonus", 0.0))  # 2025-10-13 Phase 2
        self.reward_stats["momentum_rewards"].append(components.get("momentum_reward", 0.0))  # 2025-10-13 Phase 2

    # ------------------------------------------------------------------
    # Episode & analysis helpers
    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """Reset per-episode tracking structures."""
        self.episode_rewards.clear()
        self.component_history.clear()
        self._voluntary_close_window.clear()
        self._sharpe_gate_open = False

    def get_episode_stats(self) -> Dict[str, float]:
        """Return summary statistics for the current episode."""
        if not self.episode_rewards:
            return {}

        stats: Dict[str, float] = {
            "total_reward_mean": float(np.mean(self.episode_rewards)),
            "total_reward_std": float(np.std(self.episode_rewards)),
            "total_reward_sum": float(np.sum(self.episode_rewards)),
            "steps": float(len(self.episode_rewards)),
        }

        if self.component_history:
            for key in self.config.component_keys:
                values = [c.get(key, 0.0) for c in self.component_history]
                stats[f"{key}_mean"] = float(np.mean(values))
                stats[f"{key}_sum"] = float(np.sum(values))

        return stats

    def get_component_contributions(self) -> Dict[str, float]:
        """Return relative contribution of components within an episode."""
        if not self.component_history:
            return {}

        totals: Dict[str, float] = {}
        for key in self.config.component_keys:
            values = [c.get(key, 0.0) for c in self.component_history]
            totals[key] = float(np.sum(values))

        total_abs = sum(abs(v) for v in totals.values())
        if total_abs == 0:
            return {key: 0.0 for key in totals}

        contributions = {key: (value / total_abs) * 100.0 for key, value in totals.items()}
        return contributions

    def get_recent_components(self, n_steps: int = 20) -> List[Dict[str, float]]:
        """Return the most recent component values for inspection."""
        if n_steps <= 0:
            return []
        return self.component_history[-n_steps:]

    def get_running_means(self, window: int = 50) -> Dict[str, float]:
        """Compute running means for each component over the latest window."""
        if window <= 0:
            raise ValueError("window must be positive")

        if not self.component_history:
            return {key: 0.0 for key in self.config.component_keys}

        recent = self.component_history[-window:]
        means = {}
        for key in self.config.component_keys:
            values = [c.get(key, 0.0) for c in recent]
            means[key] = float(np.mean(values)) if values else 0.0
        return means

    def update_config(self, **kwargs: float) -> None:
        """Update configuration weights or parameters at runtime."""
        for attr, value in kwargs.items():
            if not hasattr(self.config, attr):
                raise AttributeError(f"RewardConfig has no attribute '{attr}'")
            setattr(self.config, attr, value)

        logger.info("Reward configuration updated: %s", kwargs)

        if "sharpe_gate_window" in kwargs or "sharpe_gate_min_self_trades" in kwargs:
            gate_window = max(self.config.sharpe_gate_window, self.config.sharpe_gate_min_self_trades)
            recent = list(self._voluntary_close_window)[-gate_window:]
            self._voluntary_close_window = deque(recent, maxlen=gate_window)
            voluntary_closes = sum(self._voluntary_close_window)
            self._sharpe_gate_open = voluntary_closes >= self.config.sharpe_gate_min_self_trades

    @property
    def sharpe_gate_open(self) -> bool:
        """Return whether the Sharpe/ROI gate is currently open."""

        return self._sharpe_gate_open

    def summarize_reward_stats(self) -> Dict[str, float]:
        """Summarize global reward statistics across episodes."""
        summary: Dict[str, float] = {}
        for key, values in self.reward_stats.items():
            if not values:
                summary[f"{key}_mean"] = 0.0
                summary[f"{key}_std"] = 0.0
                continue
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
        return summary

    def clear_global_stats(self) -> None:
        """Clear global reward statistics (does not touch current episode)."""
        for values in self.reward_stats.values():
            values.clear()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def merge_component_histories(histories: Iterable[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate component histories from multiple episodes."""
        totals: Dict[str, float] = {}
        for entry in histories:
            for key, value in entry.items():
                totals[key] = totals.get(key, 0.0) + float(value)
        return totals

    def describe(self) -> str:
        """Return human-readable summary of the reward configuration."""
        config_items = {
            "pnl_weight": self.config.pnl_weight,
            "transaction_cost_weight": self.config.transaction_cost_weight,
            "time_efficiency_weight": self.config.time_efficiency_weight,
            "sharpe_weight": self.config.sharpe_weight,
            "drawdown_weight": self.config.drawdown_weight,
            "sizing_weight": self.config.sizing_weight,
            "hold_penalty_weight": self.config.hold_penalty_weight,
            "pnl_scale": self.config.pnl_scale,
            "time_horizon": self.config.time_horizon,
            "target_sharpe": self.config.target_sharpe,
            "max_drawdown_threshold": self.config.max_drawdown_threshold,
        }
        return ", ".join(f"{key}={value}" for key, value in config_items.items())


__all__ = ["RewardConfig", "RewardShaper"]
