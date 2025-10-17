"""Hierarchical Options for Trading Strategies.

This module implements the Options Framework for hierarchical reinforcement learning
in trading. Each option represents a high-level trading strategy that can execute
multi-step action sequences. The framework enables temporal abstraction by allowing
the agent to think in terms of strategic objectives rather than individual actions.

Architecture:
-----------
1. TradingOption (Abstract Base Class)
   - Defines the interface for all trading options
   - Each option has: initiation set, intra-option policy, termination condition

2. Concrete Options (6 trading strategies)
   - OpenLongOption: Progressive position building (bullish)
   - OpenShortOption: Progressive short building (bearish)
   - ClosePositionOption: Exit management with profit/loss logic
   - TrendFollowOption: Trend-aligned position management
   - ScalpOption: Quick profits from small moves
   - WaitOption: Intelligent market observation

3. OptionsController (Neural Network)
   - High-level policy that selects which option to execute
   - Learns option selection based on current market state
   - Provides option values (Q-values) for credit assignment

Integration with Continuous Actions:
----------------------------------
- Options output continuous actions in range [-1, 1]
- Actions are interpreted by ContinuousTradingEnvironment
- Positive: Buy signal (magnitude = position size)
- Negative: Sell signal (magnitude = exit size)
- Near-zero: Hold signal

State Space Compatibility:
------------------------
The options framework expects a flattened observation vector compatible with
the trading environment's Dict observation space:

Dict observation structure:
    - technical: (lookback, num_features) - Technical indicators sequence
    - sl_probs: (3,) - Supervised learning model predictions [mlp, lstm, gru]
    - position: (5,) - [is_open, entry_price, unrealized_pnl_pct, duration, size_pct]
    - portfolio: (8,) - [equity, cash, exposure_pct, num_positions, total_return, sharpe, sortino, total_pnl]
    - regime: (regime_dim,) - Market regime indicators

Flattened state indices (for reference):
    - Close price: Extract from technical[-1, 3] (4th feature = 'close')
    - SMA_10: Extract from technical[-1, 6]
    - SMA_20: Extract from technical[-1, 7]
    - RSI: Extract from technical[-1, 11]
    - Position size: position[4] or aggregate from multiple positions
    - Unrealized PnL %: position[2]
    - Exposure: portfolio[2]

Phase B.1 Implementation Notes:
------------------------------
This implementation integrates with:
1. ContinuousTradingEnvironment (Phase A complete)
2. SAC algorithm with ICM (Phase A complete)
3. Feature encoder (shared 3.24M transformer)
4. Reward shaper V3.1 (multi-objective)
5. Portfolio manager (multi-position support)

Next Steps (Phase B.2):
---------------------
- Integrate OptionsController into SAC policy architecture
- Add option-level value functions for temporal credit assignment
- Implement option chaining for complex strategies
- Add HER (Hindsight Experience Replay) support
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

logger = logging.getLogger(__name__)


class OptionType(IntEnum):
    """Available trading options with strategic intent."""

    OPEN_LONG = 0  # Build long position progressively
    OPEN_SHORT = 1  # Build short position on bearish signals
    CLOSE_POSITION = 2  # Exit management (profit/loss)
    TREND_FOLLOW = 3  # Ride strong trends
    SCALP = 4  # Quick profit taking
    WAIT = 5  # Intelligent observation


class TradingOption(ABC):
    """Base class for hierarchical trading options.

    An option is a temporally extended action that:
    1. Can only be initiated from certain states (initiation set)
    2. Executes a learned or heuristic intra-option policy
    3. Terminates probabilistically based on state and duration
    """

    def __init__(self, name: str = "BaseOption") -> None:
        """Initialize the trading option.

        Args:
            name: Human-readable name for logging and debugging
        """
        self.name = name
        self.step_count: int = 0
        self.entry_state: Optional[Dict[str, float]] = None

    @abstractmethod
    def initiation_set(self, state: np.ndarray, observation_dict: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """Check if this option can be initiated from the current state.

        Args:
            state: Flattened observation vector
            observation_dict: Original dict observation (optional, for advanced checks)

        Returns:
            True if option can be initiated, False otherwise
        """
        pass

    @abstractmethod
    def policy(self, state: np.ndarray, step: int, observation_dict: Optional[Dict[str, np.ndarray]] = None) -> float:
        """Execute the intra-option policy to produce a continuous action.

        Args:
            state: Flattened observation vector
            step: Step number since option initiation
            observation_dict: Original dict observation (optional)

        Returns:
            Continuous action in range [-1.0, 1.0]
        """
        pass

    @abstractmethod
    def termination_probability(
        self, state: np.ndarray, step: int, observation_dict: Optional[Dict[str, np.ndarray]] = None
    ) -> float:
        """Compute probability of terminating this option.

        Args:
            state: Flattened observation vector
            step: Step number since option initiation
            observation_dict: Original dict observation (optional)

        Returns:
            Probability in range [0.0, 1.0]
        """
        pass

    def reset(self) -> None:
        """Reset option state (called on termination or environment reset)."""
        self.step_count = 0
        self.entry_state = None


# ============================================================================
# Termination Helper Functions
# ============================================================================


def sigmoid_termination_prob(step: int, min_duration: int, max_steps: int, steepness: float = 0.5) -> float:
    """Compute smooth sigmoid termination probability.
    
    Args:
        step: Current step count
        min_duration: Minimum steps before allowing termination (returns 0.0 before this)
        max_steps: Maximum steps (returns 1.0 at this point)
        steepness: Controls how quickly probability rises (higher = sharper transition)
        
    Returns:
        Termination probability in [0, 1]
        
    Behavior:
        - step < min_duration: prob = 0.0 (forced continuation)
        - step in [min_duration, max_steps]: smooth sigmoid rise
        - step >= max_steps: prob = 1.0 (forced termination)
    """
    if step < min_duration:
        return 0.0
    
    if step >= max_steps:
        return 1.0
    
    # Normalize to [0, 1] range
    progress = (step - min_duration) / max(1, max_steps - min_duration)
    
    # Sigmoid: 1 / (1 + exp(-k * (x - 0.5)))
    # Centers at 0.5 progress, with adjustable steepness
    x = progress - 0.5
    sigmoid = 1.0 / (1.0 + np.exp(-steepness * 10 * x))
    
    # Scale to [0.0, 0.8] to leave room for stochastic sampling
    # (we don't want 100% termination until max_steps)
    return sigmoid * 0.8


def extract_position_features(
    state: Optional[np.ndarray], observation_dict: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[float, float, float, float, float]:
    """Return (flag, entry_price, pnl_pct, duration, size_pct) from obs/state.

    Falls back to zeros when no position information is available or the arrays are
    shorter than expected. Keeping this logic centralized ensures every option uses
    the same convention regardless of how observations are flattened.
    """

    position_vec: Optional[np.ndarray] = None

    if observation_dict is not None:
        pos_val = observation_dict.get("position")
        if pos_val is not None:
            position_vec = np.asarray(pos_val, dtype=np.float32).ravel()

    if position_vec is None and state is not None:
        if isinstance(state, np.ndarray) and state.size >= 5:
            position_vec = np.asarray(state[-5:], dtype=np.float32).ravel()

    if position_vec is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    if position_vec.size < 5:
        padded = np.zeros(5, dtype=np.float32)
        padded[: position_vec.size] = position_vec
        position_vec = padded

    flag, entry_price, pnl_pct, duration, size_pct = (float(position_vec[i]) for i in range(5))
    return flag, entry_price, pnl_pct, duration, size_pct


class OpenLongOption(TradingOption):
    """Option for opening long positions progressively.

    Strategy:
        - Start with a small position (conservative entry)
        - Add to position if price remains favorable (pyramiding)
        - Stop building when target exposure reached or conditions deteriorate
        - Use technical indicators AND sentiment to confirm entry conditions

    Risk Management:
        - Initial entry: 30% of max position size (scaled by sentiment)
        - Follow-up entries: 50% if price within 1% of entry AND sentiment positive
        - Cap total additions to prevent overexposure
        - Block entries on negative sentiment (<0.5)

    Sentiment Integration:
        - Sentiment stored in technical[-1, 20] (sentiment_score_hourly_ffill)
        - Range: [0, 1] where >0.5 = bullish, <0.5 = bearish
        - Scales position size: 0.8+ sentiment → 1.2x action, 0.2- sentiment → 0.5x action
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        max_steps: int = 10,
        rsi_oversold_threshold: float = 40.0,
        max_exposure_pct: float = 0.10,
        min_sentiment: float = 0.50,  # NEW: Minimum bullish sentiment to enter
        sentiment_scale_enabled: bool = True,  # NEW: Scale actions by sentiment strength
    ) -> None:
        """Initialize OpenLongOption.

        Args:
            min_confidence: Minimum SL probability to consider entry
            max_steps: Maximum steps to build position
            rsi_oversold_threshold: RSI level for oversold condition
            max_exposure_pct: Maximum position size as % of equity
            min_sentiment: Minimum sentiment score to initiate (0.5 = neutral)
            sentiment_scale_enabled: Whether to scale actions by sentiment strength
        """
        super().__init__(name="OpenLong")
        self.min_confidence = min_confidence
        self.max_steps = max_steps
        self.rsi_oversold = rsi_oversold_threshold
        self.max_exposure = max_exposure_pct
        self.min_sentiment = min_sentiment
        self.sentiment_scale_enabled = sentiment_scale_enabled
        self.entry_price: Optional[float] = None
        self.entry_sentiment: Optional[float] = None  # NEW: Track entry sentiment

    def initiation_set(self, state: np.ndarray, observation_dict: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """Can initiate if no position or position is small.
        
        Sentiment amplifies decision but is NOT required (defaults to 0.5 neutral if missing).
        """
        # Position size comes from helper to avoid brittle indexing assumptions
        _, _, _, _, position_size = extract_position_features(state, observation_dict)
        position_size = abs(position_size)

        # Must have very small or no position to initiate
        if position_size >= 0.02:
            return False

        # Extract sentiment from observation
        sentiment = self._extract_sentiment(state, observation_dict)

        # Block ONLY if sentiment is STRONGLY bearish (< 0.35)
        # Neutral sentiment (0.5) is fine - technicals will drive decision
        if sentiment < 0.35:
            return False  # Too risky - strong bearish sentiment

        # Can initiate with neutral or bullish sentiment
        return True
    
    def _extract_sentiment(self, state: np.ndarray, observation_dict: Optional[Dict[str, np.ndarray]] = None) -> float:
        """Extract sentiment score from observation.
        
        Sentiment is stored in technical features at index 20 (sentiment_score_hourly_ffill).
        Range: [0, 1] where 0.5 is neutral, >0.5 is bullish, <0.5 is bearish.
        """
        try:
            if observation_dict is not None and "technical" in observation_dict:
                technical = observation_dict["technical"]
                # Sentiment is at index 20: OHLCV(6) + Technical(14) + Sentiment(20)
                sentiment = float(technical[-1, 20])
                # Clamp to valid range
                return float(np.clip(sentiment, 0.0, 1.0))
        except (IndexError, ValueError, KeyError):
            pass
        
        # Fallback to neutral sentiment if unavailable
        return 0.5

    def policy(self, state: np.ndarray, step: int, observation_dict: Optional[Dict[str, np.ndarray]] = None) -> float:
        """Progressive buying strategy with technical AND sentiment confirmation."""
        try:
            # Get current price, RSI, and sentiment
            if observation_dict is not None and "technical" in observation_dict:
                technical = observation_dict["technical"]
                current_price = float(technical[-1, 3])  # Close price
                rsi = float(technical[-1, 11]) if technical.shape[1] > 11 else 50.0
            else:
                # Fallback: assume state has price encoded
                current_price = float(state[3]) if len(state) > 3 else 100.0
                rsi = 50.0  # Neutral if unavailable

            # Extract sentiment (defaults to 0.5 neutral if unavailable)
            sentiment = self._extract_sentiment(state, observation_dict)

            # Compute sentiment multiplier for position sizing (AMPLIFIER)
            # Neutral (0.5) → 1.0x baseline, More bullish → larger multiplier
            # sentiment 0.50 → 1.0x, sentiment 0.70 → 1.2x, sentiment 0.90 → 1.4x
            if self.sentiment_scale_enabled and sentiment > 0.5:
                # Only amplify if bullish; neutral = 1.0x baseline
                bullish_strength = (sentiment - 0.5) / 0.5  # 0.0 to 1.0 scale
                sentiment_mult = 1.0 + bullish_strength * 0.4  # 1.0x to 1.4x
            else:
                sentiment_mult = 1.0  # Neutral or bearish → baseline

            # Store entry conditions on first step
            if step == 0:
                self.entry_price = current_price
                self.entry_sentiment = sentiment
                # Initial entry (technicals + sentiment amplifier)
                base_action = 0.30
                return float(np.clip(base_action * sentiment_mult, 0.20, 0.42))

            # Progressive additions based on price, RSI, AND sentiment
            if self.entry_price is not None and step < 3:
                price_change = (current_price - self.entry_price) / max(self.entry_price, 1e-8)
                
                # Stop if sentiment turned STRONGLY bearish (< 0.35)
                # Neutral sentiment (0.5) is fine - technicals drive decision
                if sentiment < 0.35:
                    return 0.0  # Sentiment override - too risky
                
                # Add more if price favorable (not falling hard)
                if price_change > -0.01:
                    # RSI confirmation for oversold bounce
                    if rsi < self.rsi_oversold:
                        # Strong technical + any bullish tilt → medium entry
                        return float(np.clip(0.40 * sentiment_mult, 0.25, 0.56))
                    else:
                        # Weaker technical → smaller entry (sentiment can still amplify)
                        return float(np.clip(0.30 * sentiment_mult, 0.20, 0.42))

            # Smaller additions in later steps (if not strongly bearish)
            if step < self.max_steps and sentiment > 0.40:
                return float(np.clip(0.20 * sentiment_mult, 0.15, 0.28))

        except (IndexError, ValueError) as e:
            logger.warning("OpenLongOption policy error: %s", e)

        # Stop building position
        return 0.0

    def termination_probability(
        self, state: np.ndarray, step: int, observation_dict: Optional[Dict[str, np.ndarray]] = None
    ) -> float:
        """Terminate when position established or max steps reached.
        
        IMPROVED TERMINATION LOGIC:
        - Uses sigmoid curve for smooth termination (not instant)
        - min_duration enforced by wrapper (centralized duration management)
        - Success-based termination (achieved target exposure)
        - Time-based fallback with smooth probability increase
        
        NOTE: min_duration is enforced by HierarchicalSACWrapper before calling this method.
        Do NOT check min_duration here to avoid double-checking.
        """
        try:
            # Check current position size
            _, _, _, _, position_size = extract_position_features(state, observation_dict)
            position_size = abs(position_size)

            # TERMINATION CONDITION 1: Success - Position size exceeds target
            if position_size >= self.max_exposure * 0.9:  # 90% of target
                return 1.0  # Immediate termination on success

            # TERMINATION CONDITION 2: Max steps reached (forced termination)
            if step >= self.max_steps:
                return 1.0

            # TERMINATION CONDITION 3: Smooth time-based termination
            # Use sigmoid curve starting from step 0 (wrapper enforces min_duration)
            prob = sigmoid_termination_prob(step, min_duration=0, max_steps=self.max_steps, steepness=0.4)
            
            return prob

        except (IndexError, ValueError) as exc:
            logger.warning("OpenLongOption termination fallback due to error: %s", exc)
            return 0.5  # Neutral probability on error

    def reset(self) -> None:
        """Reset option state."""
        super().reset()
        self.entry_price = None
        self.entry_sentiment = None


class OpenShortOption(TradingOption):
    """Option for building short positions on bearish technical signals + negative sentiment.

    Strategy:
        - Initiates when technicals show trend reversal AND sentiment is bearish
        - Progressively builds short position over multiple steps
        - Monitors RSI for overbought conditions (good short entry)
        - Requires negative sentiment (< 0.45) to initiate
        - Scales position size based on sentiment strength

    Entry Conditions (Initiation Set):
        - No existing position or very small position (< 2% exposure)
        - Sentiment < 0.45 (bearish)
        - Technical confirmation:
          * Price below moving average (trend reversal)
          * OR RSI > 65 (overbought, due for pullback)
          * OR negative price momentum

    Intra-Option Policy:
        - Step 0: Small initial short (15-30% scaled by bearish sentiment strength)
        - Steps 1-3: Progressive additions if:
          * Sentiment remains bearish (< 0.45)
          * Price hasn't rallied against us (< 1% adverse move)
          * RSI confirms overbought conditions
        - Later steps: Smaller additions with strong bearish confirmation

    Termination:
        - Position size reaches max_exposure limit
        - Max steps reached (position fully established)
        - Probabilistic termination increases with steps

    Risk Management:
        - Max exposure: 10% of equity (configurable)
        - Scales down if sentiment not extremely bearish
        - Stops adding if price moves against position
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        max_steps: int = 10,
        rsi_overbought_threshold: float = 65.0,
        max_exposure_pct: float = 0.10,
        max_sentiment: float = 0.45,  # Maximum sentiment to initiate (bearish)
        sentiment_scale_enabled: bool = True,
    ) -> None:
        """Initialize OpenShortOption.

        Args:
            min_confidence: Minimum SL model confidence for entry
            max_steps: Maximum steps to build full position
            rsi_overbought_threshold: RSI level for overbought (short entry)
            max_exposure_pct: Maximum position size as % of equity
            max_sentiment: Maximum sentiment score to initiate (< 0.5 = bearish)
            sentiment_scale_enabled: Whether to scale actions by bearish sentiment strength
        """
        super().__init__(name="OpenShort")
        self.min_confidence = min_confidence
        self.max_steps = max_steps
        self.rsi_overbought = rsi_overbought_threshold
        self.max_exposure = max_exposure_pct
        self.max_sentiment = max_sentiment
        self.sentiment_scale_enabled = sentiment_scale_enabled
        self.entry_price: Optional[float] = None
        self.entry_sentiment: Optional[float] = None

    def initiation_set(self, state: np.ndarray, observation_dict: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """Can initiate if no position AND technicals show bearish signals.
        
        Sentiment amplifies decision but is NOT required (defaults to 0.5 neutral if missing).
        """
        # Check position size (must be very small or none)
        _, _, _, _, position_size = extract_position_features(state, observation_dict)
        if abs(position_size) >= 0.02:  # Already have meaningful position
            return False

        # Technical confirmation: look for reversal signals (PRIMARY DRIVER)
        if observation_dict is not None and "technical" in observation_dict:
            technical = observation_dict["technical"]
            try:
                current_price = float(technical[-1, 3])  # Close
                sma_10 = float(technical[-1, 6]) if technical.shape[1] > 6 else current_price
                sma_20 = float(technical[-1, 7]) if technical.shape[1] > 7 else current_price
                rsi = float(technical[-1, 11]) if technical.shape[1] > 11 else 50.0

                # Look for bearish signals:
                # 1. Price below short-term MA (downtrend)
                price_below_ma = current_price < sma_10
                
                # 2. Death cross: short MA below long MA
                death_cross = sma_10 < sma_20
                
                # 3. RSI overbought (due for pullback)
                rsi_overbought = rsi > self.rsi_overbought

                # If ANY strong technical signal present, allow entry
                # Sentiment will amplify in policy(), not block initiation
                if price_below_ma or death_cross or rsi_overbought:
                    # Extract sentiment for additional confirmation (optional)
                    sentiment = self._extract_sentiment(state, observation_dict)
                    
                    # Block only if sentiment is STRONGLY bullish (>= 0.65)
                    # This prevents shorting into obvious bull runs
                    if sentiment >= 0.65:
                        return False  # Too risky - strong bullish sentiment
                    
                    return True  # Technicals confirm, sentiment neutral/bearish

            except (IndexError, ValueError):
                pass

        # No technical data available - cannot initiate safely
        return False

    def _extract_sentiment(self, state: np.ndarray, observation_dict: Optional[Dict[str, np.ndarray]] = None) -> float:
        """Extract sentiment score from observation.
        
        Sentiment range: [0, 1] where 0.5 is neutral, >0.5 is bullish, <0.5 is bearish.
        """
        try:
            if observation_dict is not None and "technical" in observation_dict:
                technical = observation_dict["technical"]
                sentiment = float(technical[-1, 20])  # Sentiment at index 20
                return float(np.clip(sentiment, 0.0, 1.0))
        except (IndexError, ValueError, KeyError):
            pass
        
        return 0.5  # Neutral if unavailable

    def policy(self, state: np.ndarray, step: int, observation_dict: Optional[Dict[str, np.ndarray]] = None) -> float:
        """Progressive shorting strategy with technical AND sentiment confirmation.
        
        Returns negative actions to indicate short positions.
        """
        try:
            # Get current price, RSI, and sentiment
            if observation_dict is not None and "technical" in observation_dict:
                technical = observation_dict["technical"]
                current_price = float(technical[-1, 3])  # Close price
                rsi = float(technical[-1, 11]) if technical.shape[1] > 11 else 50.0
            else:
                current_price = float(state[3]) if len(state) > 3 else 100.0
                rsi = 50.0

            # Extract sentiment (defaults to 0.5 neutral if unavailable)
            sentiment = self._extract_sentiment(state, observation_dict)

            # Compute bearish sentiment multiplier for position sizing (AMPLIFIER)
            # Neutral (0.5) → 1.0x baseline, More bearish → larger multiplier
            # sentiment 0.50 → 1.0x, sentiment 0.30 → 1.2x, sentiment 0.15 → 1.4x
            if self.sentiment_scale_enabled and sentiment < 0.5:
                # Only amplify if bearish; neutral = 1.0x baseline
                bearish_strength = (0.5 - sentiment) / 0.5  # 0.0 to 1.0 scale
                sentiment_mult = 1.0 + bearish_strength * 0.4  # 1.0x to 1.4x
            else:
                sentiment_mult = 1.0  # Neutral or bullish → baseline

            # Store entry conditions on first step
            if step == 0:
                self.entry_price = current_price
                self.entry_sentiment = sentiment
                # Initial short (technicals drove entry, sentiment amplifies size)
                # Negative action = short position
                base_action = -0.25  # 25% short baseline
                return float(np.clip(base_action * sentiment_mult, -0.35, -0.15))

            # Progressive additions based on price, RSI, AND sentiment
            if self.entry_price is not None and step < 3:
                price_change = (current_price - self.entry_price) / max(self.entry_price, 1e-8)
                
                # Stop if sentiment turned bullish (>= 0.55)
                # Neutral sentiment (0.5) is fine - technicals drive decision
                if sentiment >= 0.55:
                    return 0.0  # Sentiment override - too risky to continue shorts
                
                # Add more if price favorable for shorts (not rallying hard)
                if price_change < 0.01:  # Price not rallying hard against us
                    # RSI confirmation for overbought short entry
                    if rsi > self.rsi_overbought:
                        # Strong technical + any bearish tilt → medium short
                        return float(np.clip(-0.35 * sentiment_mult, -0.50, -0.20))
                    else:
                        # Weaker technical → smaller short (sentiment can still amplify)
                        return float(np.clip(-0.25 * sentiment_mult, -0.35, -0.15))

            # Smaller additions in later steps (if not bullish)
            if step < self.max_steps and sentiment < 0.55:
                return float(np.clip(-0.18 * sentiment_mult, -0.25, -0.10))

        except (IndexError, ValueError) as e:
            logger.warning("OpenShortOption policy error: %s", e)

        # Stop building position
        return 0.0

    def termination_probability(
        self, state: np.ndarray, step: int, observation_dict: Optional[Dict[str, np.ndarray]] = None
    ) -> float:
        """Terminate when short position established or max steps reached.
        
        IMPROVED TERMINATION LOGIC:
        - Sigmoid curve for smooth termination
        - min_duration enforced by wrapper (centralized duration management)
        - Success-based: achieved target short exposure
        - Sentiment-based: exit if sentiment turns bullish
        
        NOTE: min_duration is enforced by HierarchicalSACWrapper before calling this method.
        Do NOT check min_duration here to avoid double-checking.
        """
        try:
            # Check current position size (absolute value for shorts)
            _, _, _, _, position_size = extract_position_features(state, observation_dict)
            position_size = abs(position_size)

            # TERMINATION CONDITION 1: Success - Short position size exceeds target
            if position_size >= self.max_exposure * 0.9:  # 90% of target
                return 1.0  # Immediate termination on success

            # TERMINATION CONDITION 2: Max steps reached
            if step >= self.max_steps:
                return 1.0

            # TERMINATION CONDITION 3: Sentiment turned bullish - exit early
            sentiment = self._extract_sentiment(state, observation_dict)
            if sentiment > 0.60:  # Turned bullish (was 0.55, increase threshold)
                return 0.8  # High probability to terminate (but not 100% to allow learning)

            # TERMINATION CONDITION 4: Smooth time-based termination
            # Use sigmoid starting from step 0 (wrapper enforces min_duration)
            prob = sigmoid_termination_prob(step, min_duration=0, max_steps=self.max_steps, steepness=0.4)
            
            return prob

        except (IndexError, ValueError) as exc:
            logger.warning("OpenShortOption termination fallback due to error: %s", exc)
            return 0.5

    def reset(self) -> None:
        """Reset option state."""
        super().reset()
        self.entry_price = None
        self.entry_sentiment = None


class ClosePositionOption(TradingOption):
    """Option for closing positions based on P&L targets AND sentiment.

    Strategy:
        - Monitor unrealized P&L against targets
        - Full exit on stop loss (protect capital)
        - Partial exits on profit targets (lock gains, let winners run)
        - Staged exits for large profits (80% -> 100%)
        - **NEW**: Tighten stops when sentiment turns negative
        - **NEW**: Take profits earlier if sentiment deteriorates

    Risk Management:
        - Stop loss: -1.5% (configurable, tightened to -1.05% on bad sentiment)
        - Partial profit: +1.2% (scale out 30-40%)
        - Full profit: +2.5% (scale out 80%)
        - Sentiment-based early exit: Take profits if sentiment drops below 0.35

    Sentiment Integration:
        - Tightens stop loss by 30% when sentiment < 0.40
        - Exits earlier (80%) if sentiment < 0.35 even with profits
        - Takes smaller partial profits (50%) when sentiment weak
    """

    def __init__(
        self,
        profit_target: float = 0.025,
        stop_loss: float = -0.015,
        partial_threshold: float = 0.012,
        min_hold_steps: int = 2,
        sentiment_exit_threshold: float = 0.35,  # NEW: Exit if sentiment very bearish
        sentiment_stop_tighten: float = 0.40,  # NEW: Tighten stops when sentiment < this
        use_sentiment_trailing: bool = True,  # NEW: Enable sentiment-based trailing stops
    ) -> None:
        """Initialize ClosePositionOption.

        Args:
            profit_target: P&L % for large profit exit
            stop_loss: P&L % for stop loss exit
            partial_threshold: P&L % for partial profit taking
            min_hold_steps: Minimum steps before considering exit
            sentiment_exit_threshold: Exit if sentiment drops below this (even if profitable)
            sentiment_stop_tighten: Tighten stop loss when sentiment below this
            use_sentiment_trailing: Whether to use sentiment-based trailing stops
        """
        super().__init__(name="ClosePosition")
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.partial_threshold = partial_threshold
        self.min_hold = min_hold_steps
        self.sentiment_exit_threshold = sentiment_exit_threshold
        self.sentiment_stop_tighten = sentiment_stop_tighten
        self.use_sentiment_trailing = use_sentiment_trailing
        self.staged_exit = False  # Track if partial exit occurred

    def _extract_sentiment(self, observation_dict: Dict[str, np.ndarray]) -> float:
        """Extract sentiment score from observation.

        Args:
            observation_dict: Environment observation

        Returns:
            sentiment_score: Sentiment in [0, 1], default 0.5 (neutral)
        """
        try:
            technical = observation_dict.get("technical", None)
            if technical is not None and technical.shape[1] > 20:
                sentiment = float(technical[-1, 20])  # Last timestep, sentiment index
                return np.clip(sentiment, 0.0, 1.0)
        except (IndexError, TypeError, ValueError):
            pass
        return 0.5  # Neutral sentiment fallback

    def initiation_set(self, state: np.ndarray, observation_dict: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """Can initiate if has an open position."""
        try:
            pos_flag, _, _, _, position_size = extract_position_features(state, observation_dict)
            return (pos_flag > 0.5) or (abs(position_size) > 0.01)

        except (IndexError, ValueError):
            return False

    def policy(self, state: np.ndarray, step: int, observation_dict: Optional[Dict[str, np.ndarray]] = None) -> float:
        """Exit strategy based on P&L, holding period, AND sentiment.
        
        **NEW**: Sentiment integration for adaptive risk management.
        """
        try:
            # Extract unrealized P&L and position info
            _, _, unrealized_pnl_pct, holding_period_raw, position_size = extract_position_features(
                state, observation_dict
            )
            holding_period = int(holding_period_raw) if holding_period_raw is not None else step

            # Extract sentiment
            sentiment = self._extract_sentiment(observation_dict) if observation_dict else 0.5

            # **NEW**: Sentiment emergency exit (very bearish sentiment)
            if self.use_sentiment_trailing and sentiment < self.sentiment_exit_threshold:
                logger.info(
                    "ClosePosition: Sentiment emergency exit (sentiment=%.2f, P&L=%.2f%%)",
                    sentiment,
                    unrealized_pnl_pct * 100,
                )
                return -0.80  # Exit 80% immediately

            # Enforce minimum holding period (prevent overtrading)
            if holding_period < self.min_hold:
                return 0.0  # Hold

            # **NEW**: Dynamic stop loss (tighten when sentiment weak)
            effective_stop = self.stop_loss
            if sentiment < self.sentiment_stop_tighten:
                effective_stop = self.stop_loss * 0.7  # Tighten by 30%
                logger.debug(
                    "ClosePosition: Tightened stop to %.2f%% (sentiment=%.2f)",
                    effective_stop * 100,
                    sentiment,
                )

            # STOP LOSS: Full immediate exit
            if unrealized_pnl_pct < effective_stop:
                logger.info("ClosePosition: Stop loss triggered (%.2f%%)", unrealized_pnl_pct * 100)
                return -1.0  # Sell everything

            # LARGE PROFIT: Staged exit
            if unrealized_pnl_pct > self.profit_target:
                if not self.staged_exit:
                    # First exit: take 80% off
                    self.staged_exit = True
                    logger.info("ClosePosition: Large profit exit 80%% (%.2f%%)", unrealized_pnl_pct * 100)
                    return -0.80
                else:
                    # Second exit: close remaining
                    logger.info("ClosePosition: Final exit remaining 20%% (%.2f%%)", unrealized_pnl_pct * 100)
                    return -1.0

            # PARTIAL PROFIT: Scale out to lock some gains
            # **NEW**: Larger exits on weak sentiment
            if unrealized_pnl_pct > self.partial_threshold:
                if not self.staged_exit:
                    self.staged_exit = True
                    exit_size = -0.50 if sentiment < 0.45 else -0.40
                    logger.info(
                        "ClosePosition: Partial profit exit %.0f%% (%.2f%%, sentiment=%.2f)",
                        abs(exit_size) * 100,
                        unrealized_pnl_pct * 100,
                        sentiment,
                    )
                    return exit_size

            # Hold for now
            return 0.0

        except (IndexError, ValueError) as e:
            logger.warning("ClosePositionOption policy error: %s", e)
            return 0.0

    def termination_probability(
        self, state: np.ndarray, step: int, observation_dict: Optional[Dict[str, np.ndarray]] = None
    ) -> float:
        """Terminate when position is closed or exit executed.
        
        IMPROVED TERMINATION LOGIC:
        - Success-based: position closed successfully
        - min_duration enforced by wrapper (centralized duration management)
        - Low probability to give up if position still open (needs time to close)
        
        NOTE: min_duration is enforced by HierarchicalSACWrapper before calling this method.
        Do NOT check min_duration here to avoid double-checking.
        """
        try:
            # TERMINATION CONDITION 1: Position closed successfully
            _, _, _, _, position_size = extract_position_features(state, observation_dict)
            if abs(position_size) < 0.01:
                return 1.0  # Position closed, terminate immediately

            # TERMINATION CONDITION 2: Low probability to give up if position still open
            # ClosePosition needs multiple steps to execute properly
            # Wrapper enforces min_duration, so we can return low probability safely
            return 0.02  # Very low probability (was 0.05, reduce further)

        except (IndexError, ValueError) as exc:
            logger.warning("ClosePositionOption termination fallback due to error: %s", exc)
            return 0.5

    def reset(self) -> None:
        """Reset option state."""
        super().reset()
        self.staged_exit = False


class TrendFollowOption(TradingOption):
    """Option for BIDIRECTIONAL trend following strategies WITH sentiment amplification.

    Strategy:
        - Detect strong trends using moving average crossovers (SMA_10 vs SMA_20)
        - **BIDIRECTIONAL**: Follow bullish trends (longs) AND bearish trends (shorts)
        - Add to positions during trend continuation
        - Hold existing positions while trend persists
        - Exit when trend weakens, reverses, or sentiment diverges

    Technical Indicators:
        - SMA_10 / SMA_20 crossover for trend direction
        - Positive divergence (> +2%) → Bullish trend → Long positions
        - Negative divergence (< -2%) → Bearish trend → Short positions
        - Weak divergence (< ±1%) → Exit signal

    Sentiment Integration:
        - **AMPLIFIER, NOT REQUIREMENT** (defaults to 0.5 neutral if missing)
        - Bullish trend + bullish sentiment → Larger longs (1.0-1.4x)
        - Bearish trend + bearish sentiment → Larger shorts (1.0-1.4x)
        - Blocks only on EXTREME divergence:
          * Bullish trend + very bearish sentiment (< 0.35) → Too risky
          * Bearish trend + very bullish sentiment (> 0.65) → Too risky
        - Exits on moderate divergence to protect capital

    Position Management:
        - Max position: 12% of equity (configurable)
        - Progressive building: 40% of remaining capacity per step
        - Exit 60-70% on trend reversal or sentiment divergence
        - Full exit on complete trend reversal
    """

    def __init__(
        self,
        momentum_threshold: float = 0.02,
        max_position_size: float = 0.12,
        use_sentiment_scaling: bool = True,
    ) -> None:
        """Initialize TrendFollowOption.

        Args:
            momentum_threshold: Minimum % divergence between SMA_10 and SMA_20 for trend
            max_position_size: Maximum position size to build (absolute value, same for long/short)
            use_sentiment_scaling: Whether to scale positions by sentiment confidence
        """
        super().__init__(name="TrendFollow")
        self.momentum_threshold = momentum_threshold
        self.max_position = max_position_size
        self.use_sentiment_scaling = use_sentiment_scaling
        self.current_trend: Optional[str] = None  # Track 'bullish', 'bearish', or None

    def _extract_sentiment(self, observation_dict: Dict[str, np.ndarray]) -> float:
        """Extract sentiment score from observation.

        Args:
            observation_dict: Environment observation

        Returns:
            sentiment_score: Sentiment in [0, 1], default 0.5 (neutral)
        """
        try:
            technical = observation_dict.get("technical", None)
            if technical is not None and technical.shape[1] > 20:
                sentiment = float(technical[-1, 20])  # Last timestep, sentiment index
                return np.clip(sentiment, 0.0, 1.0)
        except (IndexError, TypeError, ValueError):
            pass
        return 0.5  # Neutral sentiment fallback

    def initiation_set(self, state: np.ndarray, observation_dict: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """Can initiate if strong trend detected (BULLISH or BEARISH).
        
        Technicals are PRIMARY, sentiment only blocks EXTREME divergence.
        """
        try:
            # Extract SMAs from technical features
            if observation_dict is not None and "technical" in observation_dict:
                technical = observation_dict["technical"]
                sma_10 = float(technical[-1, 6])
                sma_20 = float(technical[-1, 7])
            else:
                # Fallback: assume state has SMAs encoded
                sma_10 = float(state[6]) if len(state) > 6 else 0.0
                sma_20 = float(state[7]) if len(state) > 7 else 0.0

            if sma_20 <= 0:
                return False

            # Calculate trend strength (BIDIRECTIONAL)
            sma_diff_pct = (sma_10 - sma_20) / sma_20
            
            # Check for strong trends in EITHER direction
            has_bullish_trend = sma_diff_pct > self.momentum_threshold
            has_bearish_trend = sma_diff_pct < -self.momentum_threshold
            
            if not (has_bullish_trend or has_bearish_trend):
                return False  # No strong trend

            # Extract sentiment
            sentiment = self._extract_sentiment(observation_dict) if observation_dict else 0.5

            # Block only on EXTREME divergence (technical vs sentiment conflict)
            if has_bullish_trend and sentiment < 0.35:
                # Bullish trend but very bearish sentiment - too risky
                return False
            
            if has_bearish_trend and sentiment > 0.65:
                # Bearish trend but very bullish sentiment - too risky
                return False

            # Trend confirmed, sentiment not extreme
            self.current_trend = 'bullish' if has_bullish_trend else 'bearish'
            return True

        except (IndexError, ValueError, ZeroDivisionError):
            return False

    def policy(self, state: np.ndarray, step: int, observation_dict: Optional[Dict[str, np.ndarray]] = None) -> float:
        """BIDIRECTIONAL trend-aligned actions WITH sentiment amplification.
        
        Returns:
            Positive actions for long positions (bullish trend)
            Negative actions for short positions (bearish trend)
            Exit actions when trend reverses or weakens
        """
        try:
            # Extract SMAs and position
            if observation_dict is not None and "technical" in observation_dict:
                technical = observation_dict["technical"]
                sma_10 = float(technical[-1, 6])
                sma_20 = float(technical[-1, 7])
            else:
                sma_10 = float(state[6]) if len(state) > 6 else 0.0
                sma_20 = float(state[7]) if len(state) > 7 else 0.0

            _, _, _, _, position_size = extract_position_features(state, observation_dict)

            if sma_20 <= 0:
                return 0.0

            # Extract sentiment
            sentiment = self._extract_sentiment(observation_dict) if observation_dict else 0.5

            # Calculate trend strength (BIDIRECTIONAL)
            trend_strength = (sma_10 - sma_20) / sma_20
            
            # Determine current trend
            has_bullish_trend = trend_strength > self.momentum_threshold
            has_bearish_trend = trend_strength < -self.momentum_threshold

            # === SENTIMENT DIVERGENCE EXIT LOGIC ===
            # Exit if sentiment strongly conflicts with position
            if position_size > 0.01:  # Have long position
                if sentiment < 0.35:  # Very bearish sentiment
                    logger.info(
                        "TrendFollow: Bearish sentiment divergence, exiting long (sentiment=%.2f, trend=%.2f%%)",
                        sentiment, trend_strength * 100
                    )
                    return -0.70  # Exit 70% of long
                    
            elif position_size < -0.01:  # Have short position
                if sentiment > 0.65:  # Very bullish sentiment
                    logger.info(
                        "TrendFollow: Bullish sentiment divergence, exiting short (sentiment=%.2f, trend=%.2f%%)",
                        sentiment, trend_strength * 100
                    )
                    return 0.70  # Exit 70% of short (positive action to close shorts)

            # === BULLISH TREND: Build/Maintain Long Positions ===
            if has_bullish_trend:
                # If we have a short position, reverse it first
                if position_size < -0.01:
                    logger.info("TrendFollow: Trend reversal - exiting short to go long")
                    return -position_size  # Close short (returns positive value)
                
                # Build long position if room available
                if position_size < self.max_position:
                    # Sentiment amplification for longs
                    # Neutral (0.5) → 1.0x, Bullish (> 0.5) → 1.0-1.4x
                    sentiment_mult = 1.0
                    if self.use_sentiment_scaling and sentiment > 0.5:
                        bullish_strength = (sentiment - 0.5) / 0.5
                        sentiment_mult = 1.0 + bullish_strength * 0.4
                    
                    # Add 40% of remaining capacity
                    remaining = self.max_position - position_size
                    add_size = min(0.04, remaining * 0.4 * sentiment_mult)
                    
                    # Return positive action (long)
                    action = add_size / 0.08  # Normalize
                    return float(np.clip(action, 0.0, 1.0))
                else:
                    return 0.0  # Already at max long

            # === BEARISH TREND: Build/Maintain Short Positions ===
            elif has_bearish_trend:
                # If we have a long position, reverse it first
                if position_size > 0.01:
                    logger.info("TrendFollow: Trend reversal - exiting long to go short")
                    return -0.70  # Exit 70% of long first
                
                # Build short position if room available
                # position_size is negative for shorts, so check absolute value
                if abs(position_size) < self.max_position:
                    # Sentiment amplification for shorts
                    # Neutral (0.5) → 1.0x, Bearish (< 0.5) → 1.0-1.4x
                    sentiment_mult = 1.0
                    if self.use_sentiment_scaling and sentiment < 0.5:
                        bearish_strength = (0.5 - sentiment) / 0.5
                        sentiment_mult = 1.0 + bearish_strength * 0.4
                    
                    # Add 40% of remaining capacity
                    remaining = self.max_position - abs(position_size)
                    add_size = min(0.04, remaining * 0.4 * sentiment_mult)
                    
                    # Return negative action (short)
                    action = -add_size / 0.08  # Normalize
                    return float(np.clip(action, -1.0, 0.0))
                else:
                    return 0.0  # Already at max short

            # === WEAK TREND: Exit Positions ===
            else:
                # Trend weakened, exit any positions
                if abs(position_size) > 0.01:
                    logger.info("TrendFollow: Weak trend, exiting position (%.1f%%)", position_size * 100)
                    if position_size > 0:
                        return -0.60  # Exit 60% of long
                    else:
                        return 0.60  # Exit 60% of short (positive to close)
                
                return 0.0  # No position, no trend

        except (IndexError, ValueError, ZeroDivisionError) as e:
            logger.warning("TrendFollowOption policy error: %s", e)
            return 0.0

    def termination_probability(
        self, state: np.ndarray, step: int, observation_dict: Optional[Dict[str, np.ndarray]] = None
    ) -> float:
        """Terminate when trend weakens significantly.
        
        IMPROVED TERMINATION LOGIC:
        - Success-based: trend exhausted (< 50% of threshold)
        - min_duration enforced by wrapper (centralized duration management)
        - Smooth sigmoid termination for time-based fallback
        
        NOTE: min_duration is enforced by HierarchicalSACWrapper before calling this method.
        Do NOT check min_duration here to avoid double-checking.
        """
        try:
            # TERMINATION CONDITION 1: Check trend strength
            if observation_dict is not None and "technical" in observation_dict:
                technical = observation_dict["technical"]
                sma_10 = float(technical[-1, 6])
                sma_20 = float(technical[-1, 7])
            else:
                sma_10 = float(state[6]) if len(state) > 6 else 0.0
                sma_20 = float(state[7]) if len(state) > 7 else 0.0

            if sma_20 <= 0:
                return 1.0

            sma_diff_pct = abs(sma_10 - sma_20) / sma_20

            # Success: Terminate if trend < 50% of threshold (trend exhausted)
            if sma_diff_pct < self.momentum_threshold * 0.5:
                return 0.85  # High probability (was 0.80)

            # TERMINATION CONDITION 2: Smooth time-based fallback
            # TrendFollow doesn't have explicit max_steps, use reasonable default
            # Wrapper enforces min_duration, so start from step 0
            max_steps = 50  # Assume reasonable max for trend following
            prob = sigmoid_termination_prob(step, min_duration=0, max_steps=max_steps, steepness=0.3)
            
            return min(prob, 0.15)  # Cap at 15% for time-based (trend should control)

        except (IndexError, ValueError, ZeroDivisionError) as exc:
            logger.warning("TrendFollowOption termination fallback due to error: %s", exc)
            return 0.5

    def reset(self) -> None:
        """Reset option state."""
        super().reset()
        self.current_trend = None


class ScalpOption(TradingOption):
    """Option for scalping quick profits WITH sentiment reversal detection.

    Strategy:
        - Enter on short-term oversold conditions (RSI < 35)
        - **NEW**: Require strong bullish sentiment (> 0.60) for conviction
        - Exit quickly on small profit (+0.5% - 1.0%)
        - **NEW**: Exit immediately if sentiment reverses (< 0.45)
        - Tight stop loss to minimize losses
        - High turnover, small positions

    Risk Management:
        - Maximum 5% position size
        - +1.0% profit target
        - -0.5% stop loss
        - Maximum 8 steps (8 hours for hourly data)

    Sentiment Integration:
        - Requires sentiment >= 0.45 for entry (allows neutral 0.5 fallback)
        - Exits immediately if sentiment < 0.40 (very bearish reversal)
        - Scales position size by sentiment strength (0.6-1.0x multiplier)
        - Works WITHOUT sentiment data (uses 0.5 neutral fallback)
    """

    def __init__(
        self,
        profit_target: float = 0.010,
        stop_loss: float = -0.005,
        max_steps: int = 8,
        rsi_entry: float = 35.0,
        position_size: float = 0.05,
        min_sentiment_entry: float = 0.45,  # NEW: Require non-bearish sentiment (allows neutral 0.5 fallback)
        sentiment_exit_threshold: float = 0.40,  # NEW: Exit if sentiment very bearish
        use_sentiment_sizing: bool = True,  # NEW: Scale by sentiment
    ) -> None:
        """Initialize ScalpOption.

        Args:
            profit_target: Quick profit target (%)
            stop_loss: Tight stop loss (%)
            max_steps: Maximum holding period
            rsi_entry: RSI threshold for entry signal
            position_size: Target position size
            min_sentiment_entry: Minimum sentiment for scalp (0.45 allows neutral fallback)
            sentiment_exit_threshold: Exit if sentiment very bearish
            use_sentiment_sizing: Scale position by sentiment confidence
        """
        super().__init__(name="Scalp")
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_steps = max_steps
        self.rsi_entry = rsi_entry
        self.position_size = position_size
        self.min_sentiment_entry = min_sentiment_entry
        self.sentiment_exit_threshold = sentiment_exit_threshold
        self.use_sentiment_sizing = use_sentiment_sizing
        self.entry_time: Optional[int] = None

    def _extract_sentiment(self, observation_dict: Dict[str, np.ndarray]) -> float:
        """Extract sentiment score from observation.

        Args:
            observation_dict: Environment observation

        Returns:
            sentiment_score: Sentiment in [0, 1], default 0.5 (neutral)
        """
        try:
            technical = observation_dict.get("technical", None)
            if technical is not None and technical.shape[1] > 20:
                sentiment = float(technical[-1, 20])  # Last timestep, sentiment index
                return np.clip(sentiment, 0.0, 1.0)
        except (IndexError, TypeError, ValueError):
            pass
        return 0.5  # Neutral sentiment fallback

    def initiation_set(self, state: np.ndarray, observation_dict: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """Can initiate on oversold RSI with no position (technicals primary, sentiment amplifies)."""
        try:
            # Check no position
            _, _, _, _, position_size = extract_position_features(state, observation_dict)
            if abs(position_size) > 0.01:
                return False

            # Check RSI oversold (PRIMARY DRIVER)
            if observation_dict is not None and "technical" in observation_dict:
                technical = observation_dict["technical"]
                rsi = float(technical[-1, 11])
            else:
                rsi = 50.0  # Neutral if unavailable

            if rsi >= self.rsi_entry:
                return False  # Not oversold enough

            # Block only if sentiment STRONGLY bearish (< 0.35)
            # Neutral sentiment (0.5) is fine - RSI drives decision
            sentiment = self._extract_sentiment(observation_dict) if observation_dict else 0.5
            if sentiment < 0.35:
                return False  # Too risky - very bearish sentiment

            return True  # RSI oversold + sentiment not bearish

        except (IndexError, ValueError):
            return False

    def policy(self, state: np.ndarray, step: int, observation_dict: Optional[Dict[str, np.ndarray]] = None) -> float:
        """Quick entry and exit logic WITH sentiment reversal detection."""
        try:
            # Track entry time
            if self.entry_time is None:
                self.entry_time = step

            # Extract sentiment
            sentiment = self._extract_sentiment(observation_dict) if observation_dict else 0.5

            # Get P&L if position exists
            pos_flag, _, unrealized_pnl_pct, _, position_size = extract_position_features(state, observation_dict)
            is_open = (pos_flag > 0.5) or (abs(position_size) > 0.01)

            # If no position, enter with small size (scaled by sentiment AMPLIFIER)
            if not is_open and step == 0:
                position_size = self.position_size
                
                # Sentiment-based sizing (AMPLIFIER)
                # Neutral (0.5) → 1.0x baseline, Bullish (0.5-1.0) → 1.0-1.3x
                if self.use_sentiment_sizing and sentiment > 0.5:
                    bullish_strength = (sentiment - 0.5) / 0.5  # 0.0 to 1.0
                    sentiment_mult = 1.0 + bullish_strength * 0.3  # 1.0x to 1.3x
                    position_size *= sentiment_mult
                
                # Return normalized action
                # Scale to action range: 0.05 position_size -> ~0.625 action
                return min(0.625, position_size / 0.08)  # Normalize to action space

            # If position exists, manage exit
            if is_open:
                # **NEW**: Sentiment reversal exit (immediate)
                if sentiment < self.sentiment_exit_threshold:
                    logger.info(
                        "Scalp: Sentiment reversal exit (sentiment=%.2f, P&L=%.2f%%)",
                        sentiment,
                        unrealized_pnl_pct * 100,
                    )
                    return -1.0

                # Stop loss
                if unrealized_pnl_pct < self.stop_loss:
                    return -1.0

                # Profit target
                if unrealized_pnl_pct > self.profit_target:
                    return -1.0

                # Max holding period
                if step >= self.max_steps:
                    return -1.0

            return 0.0  # Hold

        except (IndexError, ValueError) as e:
            logger.warning("ScalpOption policy error: %s", e)
            return 0.0

    def termination_probability(
        self, state: np.ndarray, step: int, observation_dict: Optional[Dict[str, np.ndarray]] = None
    ) -> float:
        """Terminate quickly after trade completes.
        
        IMPROVED TERMINATION LOGIC:
        - Success-based: position closed (scalp completed)
        - min_duration enforced by wrapper (centralized duration management)
        - Smooth sigmoid for time-based termination
        
        NOTE: min_duration is enforced by HierarchicalSACWrapper before calling this method.
        Do NOT check min_duration here to avoid double-checking.
        """
        try:
            # TERMINATION CONDITION 1: Trade closed successfully
            _, _, _, _, position_size = extract_position_features(state, observation_dict)
            if abs(position_size) < 0.01:
                return 1.0  # Immediate termination on trade completion

            # TERMINATION CONDITION 2: Max steps reached (scalp timeout)
            if step >= self.max_steps:
                return 1.0

            # TERMINATION CONDITION 3: Smooth sigmoid for time-based
            # Wrapper enforces min_duration, so start from step 0
            prob = sigmoid_termination_prob(step, min_duration=0, max_steps=self.max_steps, steepness=0.6)
            
            return prob

        except (IndexError, ValueError) as exc:
            logger.warning("ScalpOption termination fallback due to error: %s", exc)
            return 0.5

    def reset(self) -> None:
        """Reset option state."""
        super().reset()
        self.entry_time = None


class WaitOption(TradingOption):
    """Option for intelligent market observation WITH sentiment monitoring.

    Strategy:
        - Observe market when no clear signals present
        - Monitor technical indicators and regime changes
        - **NEW**: Track sentiment shifts for early opportunity detection
        - Accumulate information for future decision making
        - Exit when strong signals emerge OR sentiment extreme

    Use Cases:
        - Choppy/sideways markets
        - Low conviction periods
        - Regime transitions
        - Portfolio rebalancing cooldown

    Sentiment Integration:
        - Exits early if sentiment reaches extremes (< 0.30 or > 0.75)
        - Reduces waiting time when sentiment becomes decisive
        - Helps identify regime shifts faster
    """

    def __init__(
        self,
        max_wait_steps: int = 20,
        min_wait_steps: int = 3,
        sentiment_extreme_high: float = 0.75,  # NEW: Exit wait on strong bullish sentiment
        sentiment_extreme_low: float = 0.30,  # NEW: Exit wait on strong bearish sentiment
        use_sentiment_exit: bool = True,  # NEW: Enable sentiment-based early exit
    ) -> None:
        """Initialize WaitOption.

        Args:
            max_wait_steps: Maximum steps to wait
            min_wait_steps: Minimum steps before considering termination
            sentiment_extreme_high: Exit wait if sentiment exceeds this (opportunity)
            sentiment_extreme_low: Exit wait if sentiment below this (risk aversion)
            use_sentiment_exit: Whether to use sentiment for early exit
        """
        super().__init__(name="Wait")
        self.max_wait = max_wait_steps
        self.min_wait = min_wait_steps
        self.sentiment_extreme_high = sentiment_extreme_high
        self.sentiment_extreme_low = sentiment_extreme_low
        self.use_sentiment_exit = use_sentiment_exit

    def initiation_set(self, state: np.ndarray, observation_dict: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """Can always initiate (default fallback option)."""
        return True

    def policy(self, state: np.ndarray, step: int, observation_dict: Optional[Dict[str, np.ndarray]] = None) -> float:
        """Hold and observe - no trading action."""
        return 0.0  # Always hold

    def _extract_sentiment(self, observation_dict: Dict[str, np.ndarray]) -> float:
        """Extract sentiment score from observation.

        Args:
            observation_dict: Environment observation

        Returns:
            sentiment_score: Sentiment in [0, 1], default 0.5 (neutral)
        """
        try:
            technical = observation_dict.get("technical", None)
            if technical is not None and technical.shape[1] > 20:
                sentiment = float(technical[-1, 20])  # Last timestep, sentiment index
                return np.clip(sentiment, 0.0, 1.0)
        except (IndexError, TypeError, ValueError):
            pass
        return 0.5  # Neutral sentiment fallback

    def termination_probability(
        self, state: np.ndarray, step: int, observation_dict: Optional[Dict[str, np.ndarray]] = None
    ) -> float:
        """Terminate when strong signals emerge, sentiment extremes, or max wait reached.
        
        IMPROVED TERMINATION LOGIC:
        - min_duration enforced by wrapper (centralized duration management)
        - Success-based: strong signals or sentiment extremes
        - Smooth sigmoid for time-based progression
        
        NOTE: min_duration is enforced by HierarchicalSACWrapper before calling this method.
        Do NOT check min_duration here to avoid double-checking.
        """
        try:
            # TERMINATION CONDITION 1: Maximum wait period
            if step >= self.max_wait:
                return 1.0

            # TERMINATION CONDITION 2: Sentiment extremes (strong signal to act)
            if self.use_sentiment_exit and observation_dict is not None:
                sentiment = self._extract_sentiment(observation_dict)
                
                # Very bullish sentiment → opportunity to enter
                if sentiment > self.sentiment_extreme_high:
                    logger.info("Wait: Exiting on strong bullish sentiment (%.2f)", sentiment)
                    return 0.85  # Increased from 0.80
                
                # Very bearish sentiment → opportunity to short
                if sentiment < self.sentiment_extreme_low:
                    logger.info("Wait: Exiting on strong bearish sentiment (%.2f)", sentiment)
                    return 0.85  # Increased from 0.80

            # TERMINATION CONDITION 3: Strong technical signals
            if observation_dict is not None:
                # Check SL probabilities for strong conviction
                if "sl_probs" in observation_dict:
                    sl_probs = observation_dict["sl_probs"]
                    max_prob = float(np.max(sl_probs))
                    if max_prob > 0.75:  # Strong signal
                        return 0.65  # Increased from 0.60

                # Check for strong trend
                if "technical" in observation_dict:
                    technical = observation_dict["technical"]
                    if technical.shape[1] > 7:
                        sma_10 = float(technical[-1, 6])
                        sma_20 = float(technical[-1, 7])
                        if sma_20 > 0:
                            divergence = abs(sma_10 - sma_20) / sma_20
                            if divergence > 0.03:  # 3% divergence
                                return 0.75  # Increased from 0.70

            # TERMINATION CONDITION 4: Smooth sigmoid for time-based
            # Wrapper enforces min_duration, so start from step 0
            prob = sigmoid_termination_prob(step, min_duration=0, max_steps=self.max_wait, steepness=0.5)
            
            return min(prob, 0.50)  # Cap at 50% for gradual termination

        except (IndexError, ValueError, ZeroDivisionError):
            return 0.20  # Default low probability


class OptionsController(nn.Module):
    """High-level controller for option selection using learned policy.

    The OptionsController is a neural network that learns to select which trading
    option to execute based on the current market state. It consists of:

    1. Option Selector Network: Outputs logits for each option
    2. Option Value Network: Estimates Q-values for each option
    3. Initiation Set Masking: Ensures only valid options are selected
    4. State Tracking: Monitors current option and execution step

    The controller integrates with SAC by providing:
    - High-level action selection (which option to use)
    - Option-level value estimates (for temporal credit assignment)
    - Automatic option termination and switching
    """

    def __init__(
        self,
        state_dim: int,
        num_options: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        """Initialize OptionsController.

        Args:
            state_dim: Dimension of flattened state vector
            num_options: Number of available options (default 6)
            hidden_dim: Hidden layer dimension for networks
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.num_options = num_options
        self.state_dim = state_dim

        # Option selection network (policy over options)
        self.option_selector = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, num_options),
        )

        # Option value network (Q-values for options)
        self.option_value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_options),
        )

        # Initialize trading options
        self.options = [
            OpenLongOption(min_confidence=0.6, max_steps=10),
            OpenShortOption(min_confidence=0.6, max_steps=10, rsi_overbought_threshold=65.0),
            ClosePositionOption(profit_target=0.025, stop_loss=-0.015),
            TrendFollowOption(momentum_threshold=0.02),
            ScalpOption(profit_target=0.010, stop_loss=-0.005),
            WaitOption(max_wait_steps=20),
        ]

        # State tracking
        self.current_option: Optional[int] = None
        self.option_step: int = 0
        self._history_maxlen = 10_000
        self.option_history: deque[int] = deque(maxlen=self._history_maxlen)

        logger.info(
            "OptionsController initialized: state_dim=%d, num_options=%d, hidden_dim=%d",
            state_dim,
            num_options,
            hidden_dim,
        )

    def select_option(
        self,
        state: torch.Tensor,
        observation_dict: Optional[
            Union[Dict[str, np.ndarray], Sequence[Optional[Dict[str, np.ndarray]]]]
        ] = None,
        deterministic: bool = False,
        *,
        options_override: Optional[
            Union[Sequence[TradingOption], Sequence[Sequence[TradingOption]]]
        ] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select which option to execute based on current state.

        Args:
            state: Flattened state tensor (batch_size, state_dim)
            observation_dict: Original dict observation (optional)
            deterministic: If True, select argmax; if False, sample from distribution

        Returns:
            Tuple of (option_indices [batch_size], option_values [batch_size, num_options])
        """
        # Ensure state is 2D
        if state.dim() == 1:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]

        # Get option logits and values
        option_logits = self.option_selector(state)
        option_values = self.option_value(state)

        state_np = state.detach().cpu().numpy()

        # Normalize observation dictionaries to per-sample list
        if observation_dict is None:
            obs_dicts: List[Optional[Dict[str, np.ndarray]]] = [None] * batch_size
        elif isinstance(observation_dict, dict):
            obs_dicts = [observation_dict] * batch_size
        else:
            obs_dicts = list(observation_dict)
            if len(obs_dicts) != batch_size:
                raise ValueError(
                    "Observation dictionary batch size mismatch: "
                    f"expected {batch_size}, got {len(obs_dicts)}"
                )

        # Normalize per-environment option overrides
        if options_override is None:
            options_per_env: List[Sequence[TradingOption]] = [self.options] * batch_size
        else:
            override_seq = list(options_override)
            if not override_seq:
                options_per_env = [self.options] * batch_size
            elif isinstance(override_seq[0], TradingOption):
                options_per_env = [override_seq] * batch_size
            else:
                if len(override_seq) != batch_size:
                    raise ValueError(
                        "options_override batch size mismatch: "
                        f"expected {batch_size}, got {len(override_seq)}"
                    )
                options_per_env = [list(opts) for opts in override_seq]

        num_options = option_logits.shape[1]
        available_mask = torch.zeros((batch_size, num_options), dtype=torch.bool, device=state.device)
        fallback_indices: List[int] = []

        for sample_idx in range(batch_size):
            options_for_env = options_per_env[sample_idx]
            if len(options_for_env) != num_options:
                logger.error(
                    "Option count mismatch for sample %d: controller=%d, override=%d",
                    sample_idx,
                    num_options,
                    len(options_for_env),
                )
                options_for_env = self.options
            fallback_idx = max(0, len(options_for_env) - 1)
            fallback_indices.append(fallback_idx)

            obs_for_env = obs_dicts[sample_idx]
            state_for_env = state_np[sample_idx]

            for option_idx, option in enumerate(options_for_env):
                try:
                    can_init = option.initiation_set(state_for_env, obs_for_env)
                except Exception as exc:
                    logger.warning("Error checking initiation set for %s: %s", option.name, exc)
                    can_init = False
                available_mask[sample_idx, option_idx] = bool(can_init)

        masked_logits = option_logits.clone()
        no_available = ~available_mask.any(dim=1)
        if no_available.any():
            for row_idx in torch.nonzero(no_available, as_tuple=False).view(-1).tolist():
                fallback_idx = fallback_indices[row_idx]
                available_mask[row_idx, fallback_idx] = True
                masked_logits[row_idx] = -float("inf")
                masked_logits[row_idx, fallback_idx] = 0.0
                logger.warning("No options available for sample %d; defaulting to fallback index %d", row_idx, fallback_idx)

        masked_logits = masked_logits.masked_fill(~available_mask, -float("inf"))

        # Select option for each batch element
        if deterministic:
            option_indices = masked_logits.argmax(dim=-1)
        else:
            probs = torch.softmax(masked_logits, dim=-1)
            probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
            probs_sum = probs.sum(dim=-1, keepdim=True)
            probs = probs / (probs_sum + 1e-8)

            valid_probs = probs_sum.squeeze(-1) > 1e-6
            if not valid_probs.all():
                logger.warning("Invalid probability distribution detected; using argmax fallback")
                option_indices = masked_logits.argmax(dim=-1)
            else:
                try:
                    option_indices = torch.multinomial(probs, 1).squeeze(-1)
                except RuntimeError as exc:
                    logger.warning("Multinomial sampling failed (%s); using argmax", exc)
                    option_indices = masked_logits.argmax(dim=-1)

        # Ensure indices are within valid range (safety check for CUDA)
        option_indices = torch.clamp(option_indices, 0, num_options - 1)

        return option_indices, option_values

    def execute_option(
        self,
        state: np.ndarray,
        option_idx: int,
        observation_dict: Optional[Dict[str, np.ndarray]] = None,
        *,
        options_override: Optional[list[TradingOption]] = None,
    ) -> Tuple[float, bool, Dict[str, any]]:
        """Execute the selected option's policy and check termination.

        Args:
            state: Current state as numpy array
            option_idx: Index of option to execute
            observation_dict: Original dict observation (optional)

        Returns:
            Tuple of (continuous_action, should_terminate, info_dict)
        """
        # Validate option index first
        options = options_override if options_override is not None else self.options

        fallback_reason: Optional[str] = None
        if option_idx < 0 or option_idx >= len(options):
            fallback_idx = max(0, len(options) - 1)
            logger.warning("Invalid option index %d, defaulting to index %d", option_idx, fallback_idx)
            option_idx = fallback_idx
            fallback_reason = "invalid_index"

        try:
            option_name = OptionType(option_idx).name
        except ValueError:
            option_name = getattr(options[option_idx], "name", f"OPTION_{option_idx}")

        info = {
            "option_idx": option_idx,
            "option_name": option_name,
            "option_step": self.option_step,
        }
        if fallback_reason is not None:
            info["fallback"] = fallback_reason

        option = options[option_idx]

        try:
            # Get action from option's policy
            action = option.policy(state, self.option_step, observation_dict)
            info["raw_action"] = action

            # Clip action to valid range
            action = float(np.clip(action, -1.0, 1.0))
            info["clipped_action"] = action

            # Check termination
            term_prob = option.termination_probability(state, self.option_step, observation_dict)
            terminate = np.random.random() < term_prob

            info["termination_prob"] = term_prob
            info["terminated"] = terminate

            # Update step counter
            if terminate:
                self.option_step = 0
                option.reset()
                self.current_option = None
            else:
                self.option_step += 1
                self.current_option = option_idx

            # Track option usage
            self.option_history.append(option_idx)

        except Exception as e:
            logger.error("Error executing option %s: %s", option.name, e, exc_info=True)
            action = 0.0  # Default to HOLD
            terminate = True
            info["error"] = str(e)

        return action, terminate, info

    def reset(self) -> None:
        """Reset controller state (called on environment reset).
        
        CRITICAL FIX: Clear option_history to prevent memory accumulation across episodes.
        Without this, the deque grows unbounded and option statistics become stale.
        """
        self.current_option = None
        self.option_step = 0
        # CRITICAL: Clear history to prevent memory leak
        self.option_history.clear()
        for option in self.options:
            option.reset()

    def get_option_statistics(self) -> Dict[str, any]:
        """Get statistics on option usage for analysis.

        Returns:
            Dictionary with option usage counts and percentages
        """
        if not self.option_history:
            return {"total_selections": 0, "usage": {}}

        total = len(self.option_history)
        usage_counts = {}
        for option_idx in self.option_history:
            option_name = OptionType(option_idx).name
            usage_counts[option_name] = usage_counts.get(option_name, 0) + 1

        usage_pct = {name: count / total for name, count in usage_counts.items()}

        return {
            "total_selections": total,
            "usage_counts": usage_counts,
            "usage_percentages": usage_pct,
        }

    def forward(
        self, state: torch.Tensor, observation_dict: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for integration with SAC.

        Args:
            state: State tensor
            observation_dict: Optional dict observation

        Returns:
            Tuple of (option_logits, option_values)
        """
        option_logits = self.option_selector(state)
        option_values = self.option_value(state)
        return option_logits, option_values
