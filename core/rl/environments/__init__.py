"""RL environment components for trading agents."""

from .feature_extractor import FeatureConfig, FeatureExtractor
from .regime_indicators import RegimeIndicators
from .portfolio_manager import PortfolioConfig, PortfolioManager, Position
from .reward_shaper import RewardConfig, RewardShaper

try:  # pragma: no cover - optional dependency on gymnasium
    from .trading_env import TradingConfig, TradingEnvironment
except ModuleNotFoundError:  # gymnasium not installed in minimal contexts
    TradingConfig = None  # type: ignore
    TradingEnvironment = None  # type: ignore

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
    __all__.extend(["TradingEnvironment", "TradingConfig"])
