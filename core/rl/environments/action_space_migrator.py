"""Utilities for migrating between discrete and continuous action spaces."""
from __future__ import annotations

from typing import Any

import numpy as np

from .continuous_trading_env import ContinuousTradingEnvironment, HybridActionEnvironment
from .trading_env import TradeAction, TradingConfig, TradingEnvironment


class ActionSpaceMigrator:
    """Helper for mapping discrete policies to the new continuous interface."""

    DISCRETE_TO_CONTINUOUS = {
        "HOLD": 0.0,
        "BUY_SMALL": 0.3,
        "BUY_MEDIUM": 0.6,
        "BUY_LARGE": 0.9,
        "SELL_PARTIAL": -0.5,
        "SELL_ALL": -1.0,
        "ADD_POSITION": 0.4,
    }

    @staticmethod
    def discrete_to_continuous(discrete_action: int) -> float:
        """Map a discrete action index to the representative continuous value."""

        try:
            name = TradeAction(discrete_action).name
        except ValueError:
            return 0.0
        return float(ActionSpaceMigrator.DISCRETE_TO_CONTINUOUS.get(name, 0.0))

    @staticmethod
    def create_hybrid_environment(config: TradingConfig, *, seed: int | None = None, mode: str = "continuous") -> TradingEnvironment:
        """Instantiate an environment that supports discrete, continuous, or hybrid operation."""

        normalized_mode = (mode or "continuous").lower()
        if normalized_mode == "discrete":
            return TradingEnvironment(config, seed)
        if normalized_mode == "hybrid":
            return HybridActionEnvironment(config, seed)
        return ContinuousTradingEnvironment(config, seed)

    @staticmethod
    def adapt_action(action: Any) -> np.ndarray:
        """Convert arbitrary action inputs into the continuous format expected by SAC-like agents."""

        if isinstance(action, (int, np.integer)):
            value = ActionSpaceMigrator.discrete_to_continuous(int(action))
            return np.asarray([value], dtype=np.float32)
        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr.astype(np.float32)
