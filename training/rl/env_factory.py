"""Utilities for building trading environments from YAML configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from core.rl.environments import PortfolioConfig, RewardConfig
from core.rl.environments.trading_env import TradingConfig


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} must be a mapping")
    return data


def _apply_mapping(target: Any, updates: Mapping[str, Any]) -> Any:
    """Apply dictionary updates to an existing configuration object."""

    for key, value in updates.items():
        if value is None or not hasattr(target, key):
            continue
        current = getattr(target, key)
        try:
            if isinstance(current, bool):
                coerced = bool(value)
            elif isinstance(current, int) and not isinstance(current, bool):
                coerced = int(value)
            elif isinstance(current, float):
                coerced = float(value)
            else:
                coerced = value
        except Exception:  # Fallback to raw value if coercion fails
            coerced = value
        setattr(target, key, coerced)
    return target


def build_reward_config(settings: Optional[Mapping[str, Any]]) -> Optional[RewardConfig]:
    """Construct a :class:`RewardConfig` instance from a mapping."""

    if settings is None:
        return None
    reward_cfg = RewardConfig()
    return _apply_mapping(reward_cfg, settings)


def build_portfolio_config(settings: Optional[Mapping[str, Any]]) -> Optional[PortfolioConfig]:
    """Construct a :class:`PortfolioConfig` instance from a mapping."""

    if settings is None:
        return None
    portfolio_cfg = PortfolioConfig()
    return _apply_mapping(portfolio_cfg, settings)


def build_trading_config(settings: Mapping[str, Any]) -> TradingConfig:
    """Construct a :class:`TradingConfig` from configuration settings."""

    required_keys = {"symbol", "data_path"}
    missing = required_keys - settings.keys()
    if missing:
        raise KeyError(f"Trading configuration missing required keys: {', '.join(sorted(missing))}")

    data_path = Path(settings["data_path"]).expanduser()
    sl_checkpoints_raw = settings.get("sl_checkpoints", {}) or {}
    sl_checkpoints = {name: Path(path).expanduser() for name, path in sl_checkpoints_raw.items()}

    trading_cfg = TradingConfig(
        symbol=str(settings["symbol"]),
        data_path=data_path,
        sl_checkpoints=sl_checkpoints,
        sl_inference_device=settings.get("sl_inference_device"),
    )

    reward_settings = settings.get("reward_config")
    portfolio_settings = settings.get("portfolio_config")

    if reward_settings is not None:
        trading_cfg.reward_config = build_reward_config(reward_settings)
    if portfolio_settings is not None:
        trading_cfg.portfolio_config = build_portfolio_config(portfolio_settings)

    overrides = {
        key: value
        for key, value in settings.items()
        if key not in {"symbol", "data_path", "sl_checkpoints", "reward_config", "portfolio_config"}
    }
    _apply_mapping(trading_cfg, overrides)

    continuous_settings = settings.get("continuous_settings")
    if continuous_settings is not None:
        trading_cfg.continuous_settings = dict(continuous_settings)

    action_mode = settings.get("action_mode")
    if action_mode is not None:
        trading_cfg.action_mode = str(action_mode)

    disabled_actions = settings.get("disabled_actions")
    if disabled_actions is not None:
        trading_cfg.disabled_actions = list(disabled_actions)

    return trading_cfg
