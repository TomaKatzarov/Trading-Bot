import math
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from core.rl.environments.action_space_migrator import ActionSpaceMigrator
from core.rl.environments.continuous_trading_env import ContinuousTradingEnvironment
from core.rl.environments.trading_env import TradeAction, TradingConfig, TradingEnvironment
from core.rl.environments.portfolio_manager import PortfolioConfig


@pytest.fixture
def make_parquet(tmp_path: Path):
    def _builder():
        rows = 320
        timestamps = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
        base_close = 100 + np.linspace(0, 6, rows)

        df = pd.DataFrame({"timestamp": timestamps})
        df["close"] = base_close
        df["open"] = base_close * 0.999
        df["high"] = base_close * 1.001
        df["low"] = base_close * 0.998
        df["volume"] = 1_000_000
        df["vwap"] = df["close"]
        df["SMA_10"] = df["close"].rolling(10, min_periods=1).mean()
        df["SMA_20"] = df["close"].rolling(20, min_periods=1).mean()
        df["MACD_line"] = 0.0  # Fixed: was "MACD"
        df["MACD_signal"] = 0.0
        df["MACD_hist"] = 0.0
        df["RSI_14"] = 55.0
        df["Stoch_K"] = 60.0  # Fixed: was "Stochastic_K"
        df["Stoch_D"] = 58.0  # Fixed: was "Stochastic_D"
        df["ADX_14"] = 22.0
        df["ATR_14"] = 0.5
        df["BB_bandwidth"] = 0.04
        df["OBV"] = np.cumsum(np.where(np.diff(np.r_[df["close"].iloc[0], df["close"]]) >= 0, 1, -1) * 10_000)
        df["Volume_SMA_20"] = 1_000_000
        df["1h_return"] = df["close"].pct_change().fillna(0.0)  # Fixed: was "Return_1h"
        df["sentiment_score_hourly_ffill"] = 0.55

        day_angle = 2 * math.pi * timestamps.dayofweek / 7
        df["DayOfWeek_sin"] = np.sin(day_angle)
        df["DayOfWeek_cos"] = np.cos(day_angle)

        for col in TradingEnvironment.REGIME_FEATURE_CANDIDATES:
            df[col] = 0.5

        file_path = tmp_path / f"continuous_fixture_{uuid4().hex}.parquet"
        df.to_parquet(file_path, index=False)
        return file_path

    return _builder


@pytest.fixture
def continuous_config(make_parquet):
    def _factory():
        data_path = make_parquet()
        config = TradingConfig(
            symbol="TEST",
            data_path=data_path,
            sl_checkpoints={},
            episode_length=256,
            portfolio_config=PortfolioConfig(
                initial_capital=25_000.0,
                commission_rate=0.0002,
                slippage_bps=2.0,
                max_positions=1,
            ),
        )
        config.continuous_settings = {
            "hold_threshold": 0.05,
            "max_position_pct": 0.12,
            "smoothing_window": 5,
            "transaction_cost": 0.001,
            "min_trade_value": 1.0,
        }
        return config

    return _factory


def _reset_with_offset(env: ContinuousTradingEnvironment, config: TradingConfig, seed: int) -> None:
    env.reset(seed=seed, options={"start_idx": config.lookback_window + 10})


def test_continuous_action_interpretation(continuous_config):
    config = continuous_config()
    env = ContinuousTradingEnvironment(config)
    threshold = env.hold_threshold

    _reset_with_offset(env, config, 11)
    _, _, _, _, info = env.step(np.asarray([0.0], dtype=np.float32))
    assert info["continuous_action"]["trade_type"] == "hold"

    _reset_with_offset(env, config, 17)
    _, _, _, _, info = env.step(np.asarray([threshold + 0.2], dtype=np.float32))
    action_info = info["continuous_action"]
    assert action_info["trade_type"] == "buy"
    assert action_info["trade_value"] > 0.0

    _reset_with_offset(env, config, 23)
    env.step(np.asarray([threshold + 0.2], dtype=np.float32))
    info = {}
    for _ in range(6):
        _, _, _, _, info = env.step(np.asarray([-0.8], dtype=np.float32))
        if info["continuous_action"]["trade_type"] == "sell":
            break

    action_info = info["continuous_action"]
    assert action_info["trade_type"] == "sell"
    assert action_info["shares"] > 0.0


def test_action_smoothing_reduces_variance(continuous_config):
    env = ContinuousTradingEnvironment(continuous_config())
    rng = np.random.default_rng(7)
    raw_actions = rng.uniform(-1.0, 1.0, size=256)
    smoothed_actions = [env._smooth_action(value) for value in raw_actions]

    raw_std = float(np.std(raw_actions))
    smoothed_std = float(np.std(smoothed_actions))
    assert smoothed_std <= raw_std * 0.7


def test_continuous_env_cycle_runs(continuous_config):
    env = ContinuousTradingEnvironment(continuous_config())
    env.reset(seed=101)
    rng = np.random.default_rng(99)

    steps = 0
    while steps < 1000:
        action = np.asarray([float(np.sin(steps / 25.0) * 0.8)], dtype=np.float32)
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            env.reset(seed=steps + 101)

    assert steps == 1000


def test_migrator_accepts_discrete_actions(continuous_config):
    config = continuous_config()
    config.action_mode = "hybrid"
    env = ActionSpaceMigrator.create_hybrid_environment(config, seed=55, mode="hybrid")
    env.reset(seed=55)

    _, _, _, _, info_first = env.step(TradeAction.BUY_SMALL.value)
    assert info_first.get("executed_action_idx") == TradeAction.BUY_SMALL.value

    # Second step exercises partial close path and ensures discrete compatibility persists.
    _, _, _, _, info_second = env.step(TradeAction.SELL_ALL.value)
    assert info_second.get("executed_action_idx") in {
        TradeAction.SELL_ALL.value,
        TradeAction.HOLD.value,
    }
