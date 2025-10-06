"""Vectorized environment factories for trading reinforcement learning tasks."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from .trading_env import TradingConfig, TradingEnvironment

logger = logging.getLogger(__name__)

ENV_LOGGER_NAMES = [
    "core.rl.environments",
    "core.rl.environments.portfolio_manager",
    "core.rl.environments.reward_shaper",
    "core.rl.environments.trading_env",
    "core.rl.environments.feature_extractor",
    "core.rl.environments.regime_indicators",
]


def configure_env_loggers(level: Optional[int]) -> None:
    """Apply a logging level to environment loggers across processes."""

    if level is None:
        return

    for name in ENV_LOGGER_NAMES:
        env_logger = logging.getLogger(name)
        env_logger.setLevel(level)
        env_logger.propagate = True
        if not env_logger.handlers:
            env_logger.addHandler(logging.NullHandler())


def _normalize_sl_checkpoints(value: Optional[Mapping[str, Any]]) -> Dict[str, Path]:
    """Normalize checkpoint mapping to ``Dict[str, Path]`` for :class:`TradingConfig`."""

    if not value:
        return {}

    normalized: Dict[str, Path] = {}
    for name, path in value.items():
        normalized[name] = Path(path)
    return normalized


def _build_config(
    *,
    symbol: str,
    data_path: Path,
    sl_checkpoints: Optional[Mapping[str, Any]] = None,
    base_kwargs: Optional[Dict[str, Any]] = None,
) -> TradingConfig:
    """Materialize a :class:`TradingConfig` from provided arguments."""

    kwargs: Dict[str, Any] = dict(base_kwargs or {})
    kwargs.setdefault("symbol", symbol)
    kwargs.setdefault("data_path", data_path)
    kwargs.setdefault("sl_checkpoints", _normalize_sl_checkpoints(sl_checkpoints))

    # Prevent duplicates that would raise ``TypeError`` when instantiating the dataclass.
    for required_key in ("symbol", "data_path", "sl_checkpoints"):
        if required_key in kwargs:
            continue
        if required_key == "sl_checkpoints":
            kwargs[required_key] = {}
        elif required_key == "symbol":
            kwargs[required_key] = symbol
        elif required_key == "data_path":
            kwargs[required_key] = data_path

    return TradingConfig(**kwargs)


def _make_env_factory(
    *,
    symbol: str,
    rank: int,
    seed: int,
    data_dir: Path,
    env_kwargs: Dict[str, Any],
    env_log_level: Optional[int] = None,
) -> Callable[[], TradingEnvironment]:
    """Create a callable that instantiates :class:`TradingEnvironment`."""

    config_kwargs = dict(env_kwargs)
    data_path = Path(config_kwargs.pop("data_path", data_dir / f"{symbol}.parquet"))
    sl_checkpoints = config_kwargs.pop("sl_checkpoints", None)

    def _init() -> TradingEnvironment:
        configure_env_loggers(env_log_level)
        if not data_path.exists():
            raise FileNotFoundError(f"Data not found for symbol '{symbol}': {data_path}")

        config = _build_config(
            symbol=symbol,
            data_path=data_path,
            sl_checkpoints=sl_checkpoints,
            base_kwargs=config_kwargs,
        )

        env_seed = seed + rank
        env = TradingEnvironment(config, seed=env_seed)
        logger.debug("Created TradingEnvironment(rank=%s, seed=%s) for %s", rank, env_seed, symbol)
        return env

    return _init


def make_vec_trading_env(
    *,
    symbol: str,
    data_dir: Path,
    num_envs: int = 4,
    seed: int = 0,
    use_subprocess: bool = True,
    start_method: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    env_log_level: Optional[int] = None,
) -> VecEnv:
    """Create a Stable-Baselines3 compatible vectorized trading environment."""

    if num_envs <= 0:
        raise ValueError("num_envs must be positive")

    kwargs = dict(env_kwargs or {})
    factories: List[Callable[[], TradingEnvironment]] = [
        _make_env_factory(
            symbol=symbol,
            rank=rank,
            seed=seed,
            data_dir=data_dir,
            env_kwargs=kwargs,
            env_log_level=env_log_level,
        )
        for rank in range(num_envs)
    ]

    configure_env_loggers(env_log_level)

    if use_subprocess and num_envs > 1:
        logger.info(
            "Creating SubprocVecEnv(%s) for symbol %s (start_method=%s)",
            num_envs,
            symbol,
            start_method or "default",
        )
        return SubprocVecEnv(factories, start_method=start_method)

    logger.info("Creating DummyVecEnv(%s) for symbol %s", num_envs, symbol)
    return DummyVecEnv(factories)


def make_multi_symbol_vec_env(
    *,
    symbols: Sequence[str],
    data_dir: Path,
    envs_per_symbol: int = 2,
    seed: int = 0,
    use_subprocess: bool = True,
    start_method: Optional[str] = None,
    shared_env_kwargs: Optional[Dict[str, Any]] = None,
    env_log_level: Optional[int] = None,
) -> VecEnv:
    """Create a vectorized environment that cycles through multiple symbols."""

    if envs_per_symbol <= 0:
        raise ValueError("envs_per_symbol must be positive")

    if not symbols:
        raise ValueError("symbols cannot be empty")

    kwargs = dict(shared_env_kwargs or {})
    factories: List[Callable[[], TradingEnvironment]] = []
    rank = 0

    for symbol in symbols:
        symbol_path = Path(kwargs.get("data_path", data_dir / f"{symbol}.parquet"))
        if not symbol_path.exists():
            logger.warning("Skipping symbol %s because data is missing at %s", symbol, symbol_path)
            continue

        for _ in range(envs_per_symbol):
            factories.append(
                _make_env_factory(
                    symbol=symbol,
                    rank=rank,
                    seed=seed,
                    data_dir=data_dir,
                    env_kwargs=kwargs,
                    env_log_level=env_log_level,
                )
            )
            rank += 1

    if not factories:
        raise ValueError("No environments were created. Check symbols and data availability.")

    total_envs = len(factories)
    configure_env_loggers(env_log_level)
    if use_subprocess and total_envs > 1:
        logger.info(
            "Creating SubprocVecEnv(%s) for %s symbols (start_method=%s)",
            total_envs,
            len(symbols),
            start_method or "default",
        )
        return SubprocVecEnv(factories, start_method=start_method)

    logger.info("Creating DummyVecEnv(%s) across %s symbols", total_envs, len(symbols))
    return DummyVecEnv(factories)


def make_parallel_env(
    *,
    symbol: str,
    data_dir: Path,
    num_envs: int = 4,
    seed: int = 0,
    start_method: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    env_log_level: Optional[int] = None,
) -> VecEnv:
    """Convenience wrapper that always uses :class:`SubprocVecEnv`."""

    return make_vec_trading_env(
        symbol=symbol,
        data_dir=data_dir,
        num_envs=num_envs,
        seed=seed,
        use_subprocess=True,
        start_method=start_method,
        env_kwargs=env_kwargs,
        env_log_level=env_log_level,
    )


def make_sequential_env(
    *,
    symbol: str,
    data_dir: Path,
    num_envs: int = 1,
    seed: int = 0,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """Convenience wrapper that always uses :class:`DummyVecEnv`."""

    return make_vec_trading_env(
        symbol=symbol,
        data_dir=data_dir,
        num_envs=num_envs,
        seed=seed,
        use_subprocess=False,
        env_kwargs=env_kwargs,
    )


__all__ = [
    "make_vec_trading_env",
    "make_multi_symbol_vec_env",
    "make_parallel_env",
    "make_sequential_env",
]
