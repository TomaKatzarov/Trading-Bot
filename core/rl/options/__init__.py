"""Hierarchical Options Framework for Trading Strategies.

This module implements hierarchical reinforcement learning using the Options Framework
to enable multi-step coordinated trading strategies. Options allow the agent to select
high-level behaviors that execute sequences of primitive continuous actions.

Key Components:
    - TradingOption: Abstract base class for trading options
    - OptionsController: Neural network that selects which option to execute
    - Concrete Options: OpenLongOption, OpenShortOption, ClosePositionOption, 
                       TrendFollowOption, ScalpOption, WaitOption

Usage:
    >>> from core.rl.options import OptionsController, OptionType
    >>> controller = OptionsController(state_dim=512, num_options=6)
    >>> option_idx, option_values = controller.select_option(state, deterministic=False)
    >>> action, terminate = controller.execute_option(state_np, option_idx)
"""

from .trading_options import (
    ClosePositionOption,
    OpenLongOption,
    OpenShortOption,
    OptionsController,
    OptionType,
    ScalpOption,
    TradingOption,
    TrendFollowOption,
    WaitOption,
)

__all__ = [
    "TradingOption",
    "OptionType",
    "OpenLongOption",
    "OpenShortOption",
    "ClosePositionOption",
    "TrendFollowOption",
    "ScalpOption",
    "WaitOption",
    "OptionsController",
]
