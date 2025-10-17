Comprehensive Implementation Plan for Enhanced Options Trading Framework

Executive Summary
This implementation plan provides a methodical, low-risk approach to enhancing your existing reward shaper with advanced options trading configurations, while leveraging your sophisticated multi-position portfolio management system. The plan preserves all your well-tuned reward components while adding multi-position awareness and risk controls.

Current State Assessment
Working Components
✅ Sophisticated RewardShaper V3.1 with:
Professional trading strategy rewards
ROI-based progressive scaling
Anti-collapse mechanisms (diversity penalties/bonuses)
Context-aware scaling
Adaptive win multipliers
✅ Multi-position PortfolioManager with:
Support for multiple positions per symbol
Short selling capability
Position-specific tracking via position_id
Comprehensive risk controls
✅ SAC with ICM training pipeline
✅ Hierarchical Options Framework (Phase B.1)
✅ Continuous action space (-1 to +1)
Key Integration Points
Preserve all existing reward logic - it's well-tuned and sophisticated
Add multi-position awareness to reward calculations
Integrate ATR-based stops as additional reward signals
Enhance position-specific reward attribution
Implementation Phases:

Phase 1: Foundation Updates - Extend Current System (Week 1)
Goal: Add new capabilities without breaking existing reward shaper

1.1 Create Enhanced Configuration Structure
TradingBotAI
├── core
│   └── rl
│       ├── environments
│       │   ├── reward_shaper.py (KEEP EXISTING)
│       │   ├── reward_shaper_extensions.py (NEW - additions)
│       │   └── action_rejection.py (NEW)
│       ├── options
│       │   ├── trading_options.py (UPDATE)
│       │   ├── atr_based_stops.py (NEW)
│       │   └── delta_sizing.py (NEW)
│       └── utils
│           └── multi_position_tracker.py (NEW)
└── training
    └── config_templates
        ├── phase_b1_options.yaml (BACKUP)
        └── phase_b2_enhanced_options.yaml (NEW)

1.2 Create Reward Shaper Extensions (Non-Breaking Additions)
File: core/rl/environments/reward_shaper_extensions.py

"""Extensions to existing reward shaper for multi-position and ATR support."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from .reward_shaper import RewardShaper, RewardConfig

class MultiPositionRewardExtensions:
    """Extends RewardShaper with multi-position awareness.
    
    This class works alongside the existing RewardShaper, providing
    additional reward components for multi-position strategies without
    modifying the core reward logic.
    """
    
    def __init__(self, base_shaper: RewardShaper):
        """Initialize with reference to existing reward shaper."""
        self.base_shaper = base_shaper
        self.position_rewards = {}  # Track rewards per position_id
        self.atr_manager = None  # Will be initialized separately
        
    def compute_multi_position_bonus(
        self,
        positions: List[Dict],
        portfolio_state: Optional[Dict],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute additional rewards for smart multi-position management.
        
        Rewards:
        - Position diversification (multiple entries at different levels)
        - Smart pyramiding (adding to winners, not losers)
        - Hedging strategies (mixed long/short positions)
        """
        breakdown = {}
        
        if not positions or len(positions) < 2:
            return 0.0, breakdown
            
        # 1. Position diversification bonus
        symbols = set(pos['symbol'] for pos in positions)
        entry_prices = [pos['entry_price'] for pos in positions]
        
        # Reward spreading entries across price levels
        if len(entry_prices) > 1:
            price_spread = np.std(entry_prices) / np.mean(entry_prices)
            breakdown['entry_spread_bonus'] = min(0.02, price_spread * 0.5)
            
        # 2. Smart pyramiding bonus (adding to winners)
        winning_positions = [p for p in positions if p.get('unrealized_pnl', 0) > 0]
        if len(winning_positions) > 1:
            # Reward having multiple winning positions
            win_rate = len(winning_positions) / len(positions)
            breakdown['multi_winner_bonus'] = win_rate * 0.03
            
        # 3. Hedging bonus (long + short positions)
        long_positions = [p for p in positions if p.get('shares', 0) > 0]
        short_positions = [p for p in positions if p.get('shares', 0) < 0]
        
        if long_positions and short_positions:
            # Reward balanced hedging
            long_value = sum(abs(p.get('current_value', 0)) for p in long_positions)
            short_value = sum(abs(p.get('current_value', 0)) for p in short_positions)
            hedge_ratio = min(long_value, short_value) / max(long_value, short_value, 1e-8)
            breakdown['hedge_balance_bonus'] = hedge_ratio * 0.02
            
        return sum(breakdown.values()), breakdown
    
    def compute_atr_alignment_reward(
        self,
        position_info: Optional[Dict],
        atr_value: float,
        action: int,
    ) -> float:
        """
        Reward actions aligned with ATR-based risk management.
        
        - Reward taking profits near ATR targets
        - Reward stopping losses near ATR stops
        - Penalize premature exits
        """
        if not position_info or atr_value <= 0:
            return 0.0
            
        unrealized_pnl_pct = position_info.get('unrealized_pnl_pct', 0)
        entry_price = position_info.get('entry_price', 0)
        current_price = position_info.get('current_price', entry_price)
        
        if entry_price <= 0:
            return 0.0
            
        # Calculate ATR-based targets (from research)
        atr_take_profit_pct = (atr_value * 1.5) / entry_price  # 1.5x ATR for TP
        atr_stop_loss_pct = (atr_value * 2.0) / entry_price    # 2.0x ATR for SL
        
        # Check if we're near ATR levels
        near_take_profit = abs(unrealized_pnl_pct - atr_take_profit_pct) < 0.002
        near_stop_loss = abs(unrealized_pnl_pct + atr_stop_loss_pct) < 0.002
        
        # Reward closing near ATR levels
        if action in [4, 5]:  # SELL_PARTIAL or SELL_ALL
            if near_take_profit and unrealized_pnl_pct > 0:
                return 0.05  # Reward taking profit at target
            elif near_stop_loss and unrealized_pnl_pct < 0:
                return 0.02  # Small reward for disciplined stop
                
        # Penalize premature exits
        if action in [4, 5] and abs(unrealized_pnl_pct) < atr_stop_loss_pct * 0.5:
            return -0.01  # Penalty for exiting too early
            
        return 0.0
    
    def compute_position_specific_reward(
        self,
        position_id: str,
        position_data: Dict,
        action: int,
    ) -> Dict[str, float]:
        """
        Track rewards specific to individual positions for better attribution.
        """
        if position_id not in self.position_rewards:
            self.position_rewards[position_id] = {
                'total_reward': 0.0,
                'holding_period': 0,
                'entry_quality': 0.0,
                'exit_quality': 0.0,
            }
            
        # Update position-specific metrics
        rewards = self.position_rewards[position_id]
        rewards['holding_period'] = position_data.get('holding_period', 0)
        
        # Assess entry quality (for new positions)
        if rewards['holding_period'] <= 1:
            # Reward good entry timing based on subsequent price movement
            rewards['entry_quality'] = self._assess_entry_quality(position_data)
            
        # Assess exit quality (when closing)
        if action in [4, 5]:  # SELL_PARTIAL or SELL_ALL
            rewards['exit_quality'] = self._assess_exit_quality(position_data)
            
        return rewards
    
    def _assess_entry_quality(self, position_data: Dict) -> float:
        """Assess quality of entry based on initial price movement."""
        # Simplified assessment - can be enhanced
        unrealized_pnl_pct = position_data.get('unrealized_pnl_pct', 0)
        if unrealized_pnl_pct > 0:
            return min(0.02, unrealized_pnl_pct)  # Good entry
        return max(-0.01, unrealized_pnl_pct * 0.5)  # Poor entry
        
    def _assess_exit_quality(self, position_data: Dict) -> float:
        """Assess quality of exit based on captured profit."""
        unrealized_pnl_pct = position_data.get('unrealized_pnl_pct', 0)
        # Good exit if capturing profit or cutting loss appropriately
        if unrealized_pnl_pct > 0.01:  # Taking profit
            return 0.03
        elif unrealized_pnl_pct < -0.02:  # Cutting loss
            return 0.01  # Small reward for discipline
        return 0.0



Phase 2: Enhance Existing Reward Shaper (Week 1-2)
2.1 Update Existing Reward Shaper to Use Extensions
Modifications to reward_shaper.py:

# Add to imports section
from .reward_shaper_extensions import MultiPositionRewardExtensions

# In RewardShaper.__init__(), add:
def __init__(self, config: RewardConfig) -> None:
    # ... existing initialization ...
    
    # Add multi-position extensions
    self.multi_position_ext = MultiPositionRewardExtensions(self)
    self.use_multi_position_rewards = True  # Can be toggled via config
    
    # Add ATR tracking
    self.atr_values = {}  # symbol -> ATR value mapping

# Update compute_reward() method to include multi-position rewards:
def compute_reward(
    self,
    action: int,
    action_executed: bool,
    prev_equity: float,
    current_equity: float,
    position_info: Optional[Dict] = None,
    trade_info: Optional[Dict] = None,
    portfolio_state: Optional[Dict] = None,
    diversity_info: Optional[Dict] = None,
    # NEW PARAMETERS
    all_positions: Optional[List[Dict]] = None,  # All open positions
    atr_value: Optional[float] = None,  # Current ATR for the symbol
) -> Tuple[float, Dict[str, float]]:
    """Enhanced compute_reward with multi-position support."""
    
    # Call original implementation
    base_reward, components = self._compute_reward_original(
        action, action_executed, prev_equity, current_equity,
        position_info, trade_info, portfolio_state, diversity_info
    )
    
    # Add multi-position enhancements if enabled
    if self.use_multi_position_rewards and all_positions:
        # 1. Multi-position management bonus
        multi_bonus, multi_breakdown = self.multi_position_ext.compute_multi_position_bonus(
            all_positions, portfolio_state
        )
        for key, value in multi_breakdown.items():
            components[f"multi_{key}"] = value
            
        # 2. ATR alignment reward
        if atr_value is not None and atr_value > 0:
            atr_reward = self.multi_position_ext.compute_atr_alignment_reward(
                position_info, atr_value, action
            )
            components["atr_alignment"] = atr_reward
            
        # 3. Position-specific tracking
        if position_info and 'position_id' in position_info:
            position_rewards = self.multi_position_ext.compute_position_specific_reward(
                position_info['position_id'], position_info, action
            )
            # Store for analytics but don't add to total (already in base_reward)
            
    # Aggregate all components
    total_reward = self._aggregate_components(components)
    self._record_step(total_reward, components)
    
    return total_reward, components

# Keep original method for backward compatibility
def _compute_reward_original(self, ...):
    # Move existing compute_reward logic here
    pass

Phase 3: ATR-Based Risk Management Integration (Week 2)
3.1 Implement ATR Manager
File: core/rl/options/atr_based_stops.py

"""ATR-based adaptive stop loss and take profit calculations."""

import numpy as np
from typing import Dict, Optional, Tuple
import pandas as pd

class ATRStopManager:
    """Manages ATR-based stop losses and take profits with multi-position support."""
    
    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period
        self.atr_cache = {}  # symbol -> (timestamp, atr_value)
        self.position_stops = {}  # position_id -> (stop_loss, take_profit)
        
    def calculate_atr(self, 
                     ohlc_data: pd.DataFrame,
                     symbol: str) -> float:
        """Calculate Average True Range for a symbol."""
        if len(ohlc_data) < self.atr_period:
            return 0.0
            
        high = ohlc_data['high'].values
        low = ohlc_data['low'].values
        close = ohlc_data['close'].values
        
        # True Range calculation
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(true_range[-self.atr_period:])
        
        # Cache the result
        self.atr_cache[symbol] = (pd.Timestamp.now(), float(atr))
        return float(atr)
    
    def set_position_stops(self,
                          position_id: str,
                          entry_price: float,
                          atr: float,
                          is_long: bool = True,
                          stop_multiplier: float = 2.0,
                          tp_multiplier: float = 1.5) -> Tuple[float, float]:
        """
        Set ATR-based stops for a specific position.
        
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        if is_long:
            stop_loss = entry_price - (atr * stop_multiplier)
            take_profit = entry_price + (atr * tp_multiplier)
        else:  # Short position
            stop_loss = entry_price + (atr * stop_multiplier)
            take_profit = entry_price - (atr * tp_multiplier)
            
        self.position_stops[position_id] = (stop_loss, take_profit)
        return stop_loss, take_profit
    
    def check_stop_conditions(self,
                             position_id: str,
                             current_price: float) -> Optional[str]:
        """
        Check if position should be closed based on ATR stops.
        
        Returns:
            'stop_loss', 'take_profit', or None
        """
        if position_id not in self.position_stops:
            return None
            
        stop_loss, take_profit = self.position_stops[position_id]
        
        # Check stop loss
        if current_price <= stop_loss:
            return 'stop_loss'
            
        # Check take profit
        if current_price >= take_profit:
            return 'take_profit'
            
        return None
    
    def update_trailing_stop(self,
                            position_id: str,
                            current_price: float,
                            atr: float,
                            is_long: bool = True,
                            trailing_multiplier: float = 1.0) -> float:
        """Update trailing stop for a position in profit."""
        if position_id not in self.position_stops:
            return 0.0
            
        current_stop, take_profit = self.position_stops[position_id]
        
        if is_long:
            new_stop = current_price - (atr * trailing_multiplier)
            if new_stop > current_stop:
                self.position_stops[position_id] = (new_stop, take_profit)
                return new_stop
        else:  # Short
            new_stop = current_price + (atr * trailing_multiplier)
            if new_stop < current_stop:
                self.position_stops[position_id] = (new_stop, take_profit)
                return new_stop
                
        return current_stop
    
    def get_stop_info(self, position_id: str) -> Dict[str, float]:
        """Get current stop information for a position."""
        if position_id not in self.position_stops:
            return {}
            
        stop_loss, take_profit = self.position_stops[position_id]
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'stop_distance_pct': None,  # Can be calculated if needed
            'tp_distance_pct': None,
        }

Phase 4: Action Rejection & Delta Sizing (Week 2-3)
4.1 Implement Action Rejection Layer
File: core/rl/environments/action_rejection.py

"""Hard constraint enforcement for trading actions with multi-position support."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ActionRejectionLayer:
    """Enforces hard constraints on trading actions."""
    
    def __init__(self, 
                 max_total_exposure_pct: float = 0.10,
                 max_per_symbol_exposure_pct: float = 0.06,
                 max_drawdown_shutdown: float = 0.10,
                 max_positions_per_symbol: int = 3):
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_per_symbol_exposure_pct = max_per_symbol_exposure_pct
        self.max_drawdown_shutdown = max_drawdown_shutdown
        self.max_positions_per_symbol = max_positions_per_symbol
        self.rejection_log = []
        self.rejection_stats = {
            'total_rejections': 0,
            'exposure_rejections': 0,
            'drawdown_rejections': 0,
            'position_limit_rejections': 0,
        }
        
    def validate_action(self,
                       action: float,
                       symbol: str,
                       current_positions: List[Dict],
                       portfolio_metrics: Dict,
                       account_equity: float) -> Tuple[float, bool, str]:
        """
        Validate and potentially modify action based on hard constraints.
        
        Args:
            action: Proposed action [-1, 1]
            symbol: Trading symbol
            current_positions: List of current position dictionaries
            portfolio_metrics: Current portfolio state
            account_equity: Current account equity
            
        Returns:
            - Modified action (potentially scaled down)
            - Whether action was modified
            - Reason for modification
        """
        
        # Extract key metrics
        current_drawdown = portfolio_metrics.get('max_drawdown', 0)
        total_exposure = portfolio_metrics.get('exposure_pct', 0)
        
        # Check drawdown shutdown
        if current_drawdown >= self.max_drawdown_shutdown:
            self._log_rejection('drawdown_shutdown', action, 0.0, symbol)
            return 0.0, True, "Drawdown shutdown triggered"
        
        # Count positions for this symbol
        symbol_positions = [p for p in current_positions if p['symbol'] == symbol]
        num_symbol_positions = len(symbol_positions)
        
        # Check position limit per symbol
        if action > 0 and num_symbol_positions >= self.max_positions_per_symbol:
            self._log_rejection('position_limit', action, 0.0, symbol)
            return 0.0, True, f"Max positions ({self.max_positions_per_symbol}) for {symbol}"
        
        # Calculate symbol exposure
        symbol_exposure = sum(abs(p.get('current_value', 0)) 
                             for p in symbol_positions) / account_equity
        
        # Check if action would exceed limits
        planned_exposure = abs(action) * 0.1  # Approximate exposure from action
        
        # Check total exposure limit
        new_total_exposure = total_exposure + planned_exposure
        if new_total_exposure > self.max_total_exposure_pct:
            # Scale down action
            max_additional = max(0, self.max_total_exposure_pct - total_exposure)
            scale_factor = max_additional / planned_exposure if planned_exposure > 0 else 0
            modified_action = action * scale_factor
            
            self._log_rejection('total_exposure', action, modified_action, symbol)
            return modified_action, True, "Total exposure limit"
        
        # Check per-symbol exposure limit
        new_symbol_exposure = symbol_exposure + planned_exposure
        if new_symbol_exposure > self.max_per_symbol_exposure_pct:
            # Scale down action
            max_additional = max(0, self.max_per_symbol_exposure_pct - symbol_exposure)
            scale_factor = max_additional / planned_exposure if planned_exposure > 0 else 0
            modified_action = action * scale_factor
            
            self._log_rejection('symbol_exposure', action, modified_action, symbol)
            return modified_action, True, "Symbol exposure limit"
        
        return action, False, "Action approved"
    
    def _log_rejection(self, reason: str, original: float, 
                      modified: float, symbol: str) -> None:
        """Log rejection for monitoring."""
        self.rejection_log.append({
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'symbol': symbol,
            'original_action': original,
            'modified_action': modified,
        })
        
        self.rejection_stats['total_rejections'] += 1
        if 'exposure' in reason:
            self.rejection_stats['exposure_rejections'] += 1
        elif 'drawdown' in reason:
            self.rejection_stats['drawdown_rejections'] += 1
        elif 'position' in reason:
            self.rejection_stats['position_limit_rejections'] += 1
    
    def get_rejection_rate(self, window: int = 100) -> float:
        """Calculate recent rejection rate."""
        if not self.rejection_log:
            return 0.0
        recent = self.rejection_log[-window:]
        return len([r for r in recent if r['modified_action'] == 0]) / len(recent)
    
    def reset_stats(self) -> None:
        """Reset rejection statistics."""
        self.rejection_stats = {k: 0 for k in self.rejection_stats}
        self.rejection_log.clear()

4.2 Implement Delta-Adjusted Sizing
File: core/rl/options/delta_sizing.py

"""Delta-adjusted position sizing for options trading."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class OptionContract:
    """Represents an option contract with Greeks."""
    underlying_price: float
    strike_price: float
    delta: float
    contract_size: int = 100
    option_type: str = 'CALL'  # CALL or PUT
    
class DeltaAdjustedSizer:
    """Calculates delta-adjusted position sizes with multi-position support."""
    
    def __init__(self, 
                 account_equity: float,
                 risk_per_trade_pct: float = 0.01,
                 kelly_fraction: float = 0.25):
        self.account_equity = account_equity
        self.risk_per_trade_pct = risk_per_trade_pct
        self.kelly_fraction = kelly_fraction
        self.position_sizes = {}  # position_id -> size info
        
    def calculate_underlying_exposure(self,
                                    contract: OptionContract,
                                    num_contracts: int) -> float:
        """Calculate underlying-equivalent exposure."""
        return abs(num_contracts * contract.contract_size * contract.delta)
    
    def size_position_from_risk(self,
                               contract: OptionContract,
                               stop_distance: float,
                               atr: float,
                               existing_positions: List[Dict],
                               max_exposure_pct: float = 0.04) -> Tuple[int, Dict[str, float]]:
        """
        Size position based on risk and exposure limits.
        
        Returns:
            - Number of contracts
            - Sizing breakdown dictionary
        """
        breakdown = {}
        
        # 1. Risk-based sizing (using ATR stop)
        risk_amount = self.account_equity * self.risk_per_trade_pct
        
        # Use ATR-based stop distance if not provided
        if stop_distance <= 0 and atr > 0:
            stop_distance = atr * 2.0  # 2x ATR stop
            
        dollar_risk_per_contract = (contract.underlying_price * 
                                   contract.contract_size * 
                                   abs(contract.delta) * 
                                   stop_distance / contract.underlying_price)
        
        contracts_from_risk = int(risk_amount / max(dollar_risk_per_contract, 1))
        breakdown['risk_based_contracts'] = contracts_from_risk
        
        # 2. Exposure-based cap
        max_exposure = self.account_equity * max_exposure_pct
        
        # Account for existing positions in the same symbol
        symbol_exposure = sum(abs(p.get('current_value', 0)) 
                            for p in existing_positions 
                            if p.get('symbol') == contract.underlying_price)
        
        available_exposure = max(0, max_exposure - symbol_exposure)
        
        max_contracts_from_exposure = int(
            available_exposure / 
            (contract.underlying_price * contract.contract_size * abs(contract.delta))
        )
        breakdown['exposure_based_contracts'] = max_contracts_from_exposure
        
        # 3. Kelly criterion adjustment (optional)
        if self.kelly_fraction > 0:
            # Simplified Kelly sizing
            win_rate = 0.55  # Assumed or calculated from history
            avg_win_loss_ratio = 1.5  # Assumed or calculated
            
            kelly_f = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
            kelly_contracts = int(contracts_from_risk * kelly_f * self.kelly_fraction)
            breakdown['kelly_adjusted_contracts'] = kelly_contracts
            contracts_from_risk = min(contracts_from_risk, kelly_contracts)
        
        # Take minimum and ensure at least 1
        final_contracts = max(1, min(contracts_from_risk, max_contracts_from_exposure))
        breakdown['final_contracts'] = final_contracts
        breakdown['position_value'] = final_contracts * contract.contract_size * contract.underlying_price
        breakdown['delta_adjusted_exposure'] = self.calculate_underlying_exposure(contract, final_contracts)
        
        return final_contracts, breakdown
    
    def validate_aggregate_exposure(self,
                                   all_positions: List[Dict]) -> Dict[str, float]:
        """Validate total portfolio exposure across all positions."""
        total_exposure = sum(abs(p.get('current_value', 0)) for p in all_positions)
        
        metrics = {
            'total_exposure': total_exposure,
            'exposure_pct': total_exposure / self.account_equity,
            'num_positions': len(all_positions),
            'avg_position_size': total_exposure / len(all_positions) if all_positions else 0,
        }
        
        # Add per-symbol exposure
        symbol_exposures = {}
        for pos in all_positions:
            symbol = pos.get('symbol', 'UNKNOWN')
            if symbol not in symbol_exposures:
                symbol_exposures[symbol] = 0
            symbol_exposures[symbol] += abs(pos.get('current_value', 0))
            
        metrics['symbol_exposures'] = symbol_exposures
        metrics['max_symbol_exposure_pct'] = (
            max(symbol_exposures.values()) / self.account_equity 
            if symbol_exposures else 0
        )
        
        return metrics

Phase 5: Update Trading Options (Week 3)
5.1 Enhance Options with New Parameters
Updates to trading_options.py:

# Add imports
from ..options.atr_based_stops import ATRStopManager
from ..options.delta_sizing import DeltaAdjustedSizer

class OpenLongOption(TradingOption):
    """Enhanced with ATR stops, delta sizing, and multi-position support."""
    
    def __init__(
        self,
        # Existing parameters
        min_confidence: float = 0.60,
        max_steps: int = 12,
        
        # NEW: Enhanced parameters from research
        rsi_period: int = 14,
        rsi_oversold_threshold: float = 35.0,
        max_exposure_pct: float = 0.04,
        action_scale: float = 0.75,
        atr_stop_multiplier: float = 2.0,
        atr_take_profit_multiplier: float = 1.5,
        min_duration: int = 3,
        min_sentiment: float = 0.45,
        sentiment_scale_enabled: bool = True,
        sentiment_scale_range: Tuple[float, float] = (0.85, 1.25),
    ) -> None:
        super().__init__("OpenLong")
        
        # Store all parameters
        self.min_confidence = min_confidence
        self.max_steps = max_steps
        self.rsi_period = rsi_period
        self.rsi_oversold_threshold = rsi_oversold_threshold
        self.max_exposure_pct = max_exposure_pct
        self.action_scale = action_scale
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_take_profit_multiplier = atr_take_profit_multiplier
        self.min_duration = min_duration
        self.min_sentiment = min_sentiment
        self.sentiment_scale_enabled = sentiment_scale_enabled
        self.sentiment_scale_range = sentiment_scale_range
        
        # Initialize managers
        self.atr_manager = ATRStopManager(period=14)
        self.position_tracker = {}  # Track position-specific state
        
    def can_initiate(self, state: np.ndarray, 
                    observation_dict: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """Enhanced initiation check with multi-position awareness."""
        
        # Extract portfolio state
        portfolio = observation_dict.get('portfolio', {}) if observation_dict else {}
        all_positions = observation_dict.get('all_positions', []) if observation_dict else []
        
        # Check if we have room for more positions
        symbol_positions = [p for p in all_positions if p.get('symbol') == self.symbol]
        if len(symbol_positions) >= 3:  # Max 3 positions per symbol
            return False
            
        # Check exposure limits
        current_exposure = portfolio.get('exposure_pct', 0)
        if current_exposure >= 0.08:  # Leave room for new position
            return False
            
        # Original confidence and indicator checks
        sl_probs = self._extract_sl_probs(state, observation_dict)
        confidence = np.mean(sl_probs) if sl_probs is not None else 0.0
        
        if confidence < self.min_confidence:
            return False
            
        # Check RSI condition
        rsi = self._extract_rsi(state, observation_dict)
        if rsi is None or rsi > self.rsi_oversold_threshold:
            return False
            
        # Check sentiment if available
        if self.min_sentiment > 0:
            sentiment = self._extract_sentiment(state, observation_dict)
            if sentiment is not None and sentiment < self.min_sentiment:
                return False
                
        return True
    
    def policy(self, state: np.ndarray, step: int,
              observation_dict: Optional[Dict[str, np.ndarray]] = None) -> float:
        """Enhanced policy with ATR-based sizing and multi-position awareness."""
        
        # Get all positions for this symbol
        all_positions = observation_dict.get('all_positions', []) if observation_dict else []
        symbol_positions = [p for p in all_positions if p.get('symbol') == self.symbol]
        
        # Calculate ATR for position sizing
        atr = self.atr_manager.calculate_atr(
            observation_dict.get('ohlc_data'),
            self.symbol
        ) if observation_dict else 0.01
        
        # Base action calculation
        sl_probs = self._extract_sl_probs(state, observation_dict)
        confidence = np.mean(sl_probs) if sl_probs is not None else 0.5
        
        # Progressive position building with decay
        if len(symbol_positions) == 0:
            # First position - full size
            base_action = confidence - 0.45
        elif len(symbol_positions) == 1:
            # Second position - reduced size
            base_action = (confidence - 0.55) * 0.7
        else:
            # Third position - minimal size
            base_action = (confidence - 0.60) * 0.5
            
        # Apply sentiment scaling if enabled
        if self.sentiment_scale_enabled:
            sentiment = self._extract_sentiment(state, observation_dict)
            if sentiment is not None:
                scale_factor = self._calculate_sentiment_scale(sentiment)
                base_action *= scale_factor
                
        # Apply action scale
        action = base_action * self.action_scale
        
        # Store ATR stops for this position (if new)
        if step == 0 and len(symbol_positions) == 0:
            current_price = self._extract_current_price(state, observation_dict)
            if current_price > 0:
                position_id = f"{self.symbol}_{step}"
                stop_loss, take_profit = self.atr_manager.set_position_stops(
                    position_id=position_id,
                    entry_price=current_price,
                    atr=atr,
                    is_long=True,
                    stop_multiplier=self.atr_stop_multiplier,
                    tp_multiplier=self.atr_take_profit_multiplier
                )
                self.position_tracker[position_id] = {
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_step': step,
                }
                
        return np.clip(action, -1.0, 1.0)
    
    def should_terminate(self, state: np.ndarray, step: int,
                        observation_dict: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """Enhanced termination with multi-position awareness."""
        
        # Check max duration
        if step >= self.max_steps:
            return True
            
        # Check minimum duration
        if step < self.min_duration:
            return False
            
        # Check if any position hit ATR stops
        all_positions = observation_dict.get('all_positions', []) if observation_dict else []
        current_price = self._extract_current_price(state, observation_dict)
        
        for pos in all_positions:
            position_id = pos.get('position_id')
            if position_id in self.position_tracker:
                stop_condition = self.atr_manager.check_stop_conditions(
                    position_id, current_price
                )
                if stop_condition is not None:
                    return True
                    
        # Check if all positions are profitable (consider exiting)
        if all_positions:
            all_profitable = all(p.get('unrealized_pnl', 0) > 0 for p in all_positions)
            avg_pnl_pct = np.mean([p.get('unrealized_pnl_pct', 0) for p in all_positions])
            
            if all_profitable and avg_pnl_pct > 0.02:  # 2% average profit
                return True
                
        return False
    
    def _calculate_sentiment_scale(self, sentiment: float) -> float:
        """Calculate sentiment-based scaling factor."""
        min_scale, max_scale = self.sentiment_scale_range
        
        if sentiment < 0.5:
            # Bearish sentiment - reduce position
            return min_scale + (sentiment / 0.5) * (1.0 - min_scale)
        else:
            # Bullish sentiment - increase position
            return 1.0 + ((sentiment - 0.5) / 0.5) * (max_scale - 1.0)

# Similar enhancements for other options (OpenShort, ClosePosition, etc.)
# Following the same pattern with multi-position awareness

Phase 6: Configuration & Validation (Week 3-4)
6.1 Create Enhanced Configuration
File: training/config_templates/phase_b2_enhanced_options.yaml

# Phase B.2: Enhanced Options with existing reward shaper
# Builds upon phase_b1_options.yaml with multi-position support

# Copy all existing sections from phase_b1_options.yaml
experiment:
  name: "phase_b2_enhanced_options"
  description: "Enhanced options with multi-position and ATR-based risk management"
  # ... [keep existing]

# Enhance reward configuration (additions to existing)
reward_config:
  # Keep all existing weights - they're well tuned
  pnl_weight: 0.45
  transaction_cost_weight: 0.15
  time_efficiency_weight: 0.15
  sharpe_weight: 0.05
  drawdown_weight: 0.10
  sizing_weight: 0.05
  diversity_bonus_weight: 0.02
  diversity_penalty_weight: 0.03
  
  # NEW: Multi-position reward components
  multi_position_bonus_weight: 0.02  # Reward smart multi-position management
  atr_alignment_weight: 0.01  # Reward ATR-based exit timing
  position_quality_weight: 0.01  # Track position-specific quality

# Enhanced options configuration
options:
  enabled: true
  
  # Multi-position settings (NEW)
  max_positions_per_symbol: 3
  position_scaling: [1.0, 0.7, 0.5]  # Size decay for 1st, 2nd, 3rd position
  
  # Risk management (NEW)
  use_atr_stops: true
  use_delta_sizing: true
  use_action_rejection: true
  
  # Option-specific configurations (ENHANCED)
  open_long:
    min_confidence: 0.60
    max_steps: 12
    rsi_period: 14
    rsi_oversold_threshold: 35.0
    max_exposure_pct: 0.04
    min_sentiment: 0.45
    sentiment_scale_enabled: true
    sentiment_scale_range: [0.85, 1.25]
    action_scale: 0.75
    min_duration: 3
    atr_stop_multiplier: 2.0
    atr_take_profit_multiplier: 1.5
    
  open_short:
    min_confidence: 0.60
    max_steps: 12
    rsi_period: 14
    rsi_overbought_threshold: 65.0
    max_exposure_pct: 0.04
    max_sentiment: 0.55
    sentiment_scale_enabled: true
    sentiment_scale_range: [0.85, 1.25]
    action_scale: 0.75
    min_duration: 3
    atr_stop_multiplier: 2.0
    atr_take_profit_multiplier: 1.5
    
  # ... [continue with other options]

# Portfolio risk controls (ENHANCED)
portfolio_controls:
  max_total_exposure_pct: 0.10
  max_per_symbol_exposure_pct: 0.06
  max_positions_per_symbol: 3  # Allow pyramiding
  risk_per_trade_pct: 0.01
  kelly_fraction: 0.25
  max_drawdown_shutdown: 0.10
  slippage_estimate: 0.0015
  commission_per_contract: 0.65
  
  # Multi-position specific
  allow_multiple_positions_per_symbol: true
  shorting_enabled: true
  short_margin_requirement: 1.5

Phase 7: Integration & Testing (Week 4)
7.1 Update Training Script
Modifications to train_sac_with_options.py:

# Add imports
from core.rl.environments.reward_shaper_extensions import MultiPositionRewardExtensions
from core.rl.options.atr_based_stops import ATRStopManager
from core.rl.options.delta_sizing import DeltaAdjustedSizer
from core.rl.environments.action_rejection import ActionRejectionLayer

def create_enhanced_env(config: Dict, symbol: str):
    """Create environment with enhanced reward shaper and risk controls."""
    
    # Create base environment
    env = TradingEnvironment(config['environment'])
    
    # Get existing reward shaper
    reward_shaper = env.reward_shaper
    
    # Add multi-position extensions
    reward_shaper.multi_position_ext = MultiPositionRewardExtensions(reward_shaper)
    reward_shaper.use_multi_position_rewards = config['reward_config'].get(
        'multi_position_bonus_weight', 0) > 0
    
    # Initialize ATR manager
    atr_manager = ATRStopManager(period=14)
    
    # Initialize action rejection
    action_rejection = ActionRejectionLayer(
        max_total_exposure_pct=config['portfolio_controls']['max_total_exposure_pct'],
        max_per_symbol_exposure_pct=config['portfolio_controls']['max_per_symbol_exposure_pct'],
        max_drawdown_shutdown=config['portfolio_controls']['max_drawdown_shutdown'],
        max_positions_per_symbol=config['portfolio_controls'].get('max_positions_per_symbol', 3)
    )
    
    # Attach to environment
    env.atr_manager = atr_manager
    env.action_rejection = action_rejection
    
    return env

# In main training function
def train_with_enhanced_options(config_path: Path, symbol: str):
    """Train SAC with enhanced options framework."""
    
    # Load configuration
    config = load_yaml(config_path)
    
    # Create enhanced environment
    env = create_enhanced_env(config, symbol)
    
    # Initialize enhanced options with new parameters
    options = []
    for option_name, option_config in config['options'].items():
        if option_name == 'open_long':
            option = OpenLongOption(**option_config)
        elif option_name == 'open_short':
            option = OpenShortOption(**option_config)
        # ... etc
        
        options.append(option)
    
    # Continue with existing training setup...

Phase 8: Testing Protocol (Week 4-5)
8.1 Progressive Testing Strategy

1.Unit Tests for New Components

# tests/test_multi_position_rewards.py
def test_multi_position_bonus():
    """Test multi-position reward calculations."""
    # Test with multiple winning positions
    # Test with hedged positions
    # Test with position diversification

# tests/test_atr_stops.py  
def test_atr_calculation():
    """Test ATR calculation accuracy."""
    
def test_position_specific_stops():
    """Test stop management for multiple positions."""

# tests/test_action_rejection.py
def test_multi_position_limits():
    """Test position count limits per symbol."""

2.Integration Test with Existing Reward Shaper

def test_reward_shaper_compatibility():
    """Ensure new components don't break existing rewards."""
    
    # Create reward shaper with extensions
    config = RewardConfig()
    shaper = RewardShaper(config)
    
    # Test that original rewards still work
    base_reward, components = shaper.compute_reward(...)
    assert 'pnl' in components
    assert 'transaction_cost' in components
    
    # Test that new components integrate properly
    enhanced_reward, components = shaper.compute_reward(
        ...,
        all_positions=[...],  # New parameter
        atr_value=0.5  # New parameter
    )
    assert 'multi_entry_spread_bonus' in components
    assert 'atr_alignment' in components

    3.Gradual Rollout
Phase 1: Test with single position (disable multi-position) to ensure no regression
Phase 2: Enable 2 positions per symbol
Phase 3: Full 3 positions with all features
Phase 9: Monitoring & Optimization (Week 5+)
9.1 Enhanced Monitoring Dashboard
class EnhancedMonitoringDashboard:
    """Monitor multi-position and risk management metrics."""
    
    def __init__(self, existing_dashboard):
        self.base_dashboard = existing_dashboard
        self.multi_position_metrics = {
            'positions_per_symbol': {},
            'position_quality_scores': {},
            'atr_stop_effectiveness': [],
            'rejection_rates': [],
        }
        
    def update_metrics(self, env_state, reward_components):
        """Update monitoring with multi-position awareness."""
        
        # Update base metrics
        self.base_dashboard.update_metrics(env_state)
        
        # Track position distribution
        positions = env_state.get('all_positions', [])
        symbol_counts = {}
        for pos in positions:
            symbol = pos['symbol']
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
        self.multi_position_metrics['positions_per_symbol'] = symbol_counts
        
        # Track ATR stop effectiveness
        if 'atr_alignment' in reward_components:
            self.multi_position_metrics['atr_stop_effectiveness'].append(
                reward_components['atr_alignment']
            )
            
        # Track rejection rate
        if hasattr(env_state, 'action_rejection'):
            rejection_rate = env_state.action_rejection.get_rejection_rate()
            self.multi_position_metrics['rejection_rates'].append(rejection_rate)


Risk Mitigation
Key Differences from Original Plan
Preserve existing reward shaper - No replacement, only enhancement
Backward compatibility - All existing functionality remains
Optional features - Multi-position rewards can be toggled off
Gradual integration - Test each feature independently
Rollback Strategy
Immediate: Set use_multi_position_rewards = False
Quick: Revert to single position by setting max_positions_per_symbol = 1
Full: Use original phase_b1_options.yaml configuration
Success Metrics
Week 1-2 Goals
✅ Multi-position reward extensions working
✅ ATR manager calculating stops correctly
✅ No degradation in single-position performance
Week 3-4 Goals
✅ Action rejection preventing overexposure
✅ Multi-position strategies showing positive rewards
✅ Position quality tracking operational
Week 5+ Targets
Sharpe ratio > baseline + 10%
Maximum drawdown < 8%
Successful pyramiding in trending markets
ATR stop hit rate > 60%
Action rejection rate < 5%
Conclusion
This amended plan builds upon your excellent existing reward shaper rather than replacing it. The multi-position extensions work alongside your current implementation, adding new reward components for advanced position management while preserving all the sophisticated logic you've already developed. This approach minimizes risk while maximizing the benefits of multi-position trading capabilities.