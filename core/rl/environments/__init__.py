"""RL environment components for trading agents."""

from .feature_extractor import FeatureConfig, FeatureExtractor
from .regime_indicators import RegimeIndicators
from .portfolio_manager import PortfolioConfig, PortfolioManager, Position
from .reward_shaper import RewardConfig, RewardShaper

try:  # pragma: no cover - optional dependency on gymnasium
    from .trading_env import TradingConfig, TradingEnvironment
    from .vec_trading_env import (
        make_multi_symbol_vec_env,
        make_parallel_env,
        make_sequential_env,
        make_vec_trading_env,
    )
except ModuleNotFoundError:  # gymnasium not installed in minimal contexts
    TradingConfig = None  # type: ignore
    TradingEnvironment = None  # type: ignore
    make_multi_symbol_vec_env = None  # type: ignore
    make_parallel_env = None  # type: ignore
    make_sequential_env = None  # type: ignore
    make_vec_trading_env = None  # type: ignore

__all__ = [
    "FeatureExtractor",
    "FeatureConfig",
    "RegimeIndicators",
    "PortfolioManager",
    "PortfolioConfig",
    "Position",
    "RewardShaper",
    "RewardConfig",
]

if TradingEnvironment is not None and TradingConfig is not None:
    __all__.extend([
        "TradingEnvironment",
        "TradingConfig",
        "make_vec_trading_env",
        "make_multi_symbol_vec_env",
        "make_parallel_env",
        "make_sequential_env",
    ])
