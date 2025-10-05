import math
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from core.rl.environments.trading_env import (
    TradeAction,
    TradingConfig,
    TradingEnvironment,
)


@pytest.fixture
def make_parquet(tmp_path: Path):
    """Factory generating synthetic parquet datasets with required features."""

    default_cols = TradingEnvironment.DEFAULT_FEATURE_COLUMNS
    regime_cols = TradingEnvironment.REGIME_FEATURE_CANDIDATES

    def _builder(modifier=None) -> Path:
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
        df["MACD"] = 0.0
        df["MACD_signal"] = 0.0
        df["MACD_hist"] = 0.0
        df["RSI_14"] = 55.0
        df["Stochastic_K"] = 60.0
        df["Stochastic_D"] = 58.0
        df["ADX_14"] = 22.0
        df["ATR_14"] = 0.5
        df["BB_bandwidth"] = 0.04
        df["OBV"] = np.cumsum(np.where(np.diff(np.r_[df["close"].iloc[0], df["close"]]) >= 0, 1, -1) * 10_000)
        df["Volume_SMA_20"] = 1_000_000
        df["Return_1h"] = df["close"].pct_change().fillna(0.0)
        df["sentiment_score_hourly_ffill"] = 0.55

        day_angle = 2 * math.pi * timestamps.dayofweek / 7
        df["DayOfWeek_sin"] = np.sin(day_angle)
        df["DayOfWeek_cos"] = np.cos(day_angle)

        for col in regime_cols:
            df[col] = 0.5

        if modifier:
            modifier(df)

        missing = set(default_cols) - set(df.columns)
        if missing:
            raise AssertionError(f"Fixture missing columns: {missing}")

        file_path = tmp_path / f"dataset_{uuid4().hex}.parquet"
        df.to_parquet(file_path, index=False)
        return file_path

    return _builder


def _make_config(data_path: Path, **overrides) -> TradingConfig:
    return TradingConfig(
        symbol="TEST",
        data_path=data_path,
        sl_checkpoints={},
        episode_length=64,
        **overrides,
    )


def test_reset_provides_expected_shapes(make_parquet):
    data_path = make_parquet()
    config = _make_config(data_path)
    env = TradingEnvironment(config)

    observation, info = env.reset(seed=123, options={"start_idx": config.lookback_window + 5})

    assert observation["technical"].shape == (config.lookback_window, len(env.feature_cols))
    assert observation["sl_probs"].shape == (len(TradingEnvironment.SL_MODEL_ORDER),)
    assert observation["position"].shape == (5,)
    assert observation["portfolio"].shape == (8,)
    assert observation["regime"].shape == (10,)
    assert math.isclose(float(observation["sl_probs"].sum()), 1.0, rel_tol=1e-3)

    assert info["position"] is None
    assert "timestamp" in info and info["timestamp"].endswith("+00:00")
    assert env.config.stop_loss == pytest.approx(0.02)
    assert env.config.take_profit == pytest.approx(0.025)
    assert env.config.max_hold_hours == 8


def test_buy_and_sell_cycle_updates_portfolio(make_parquet):
    def modifier(df: pd.DataFrame) -> None:
        idx = 170
        df.loc[idx, "close"] = df.loc[idx - 1, "close"] * 1.02
        df.loc[idx, "high"] = df.loc[idx, "close"] * 1.001
        df.loc[idx, "low"] = df.loc[idx, "close"] * 0.999

    data_path = make_parquet(modifier=modifier)
    config = _make_config(data_path, commission_rate=0.0, slippage_bps=0.0)
    env = TradingEnvironment(config)

    env.reset(seed=7, options={"start_idx": 168})

    _, reward, terminated, truncated, info = env.step(TradeAction.BUY_MEDIUM.value)
    assert info["action_executed"] is True
    assert env.portfolio.positions.get("TEST") is not None
    assert not terminated and not truncated

    _, reward, terminated, truncated, info = env.step(TradeAction.SELL_ALL.value)
    assert info["position_closed"] is not None
    assert info["position_closed"].get("trigger") == "agent_full_close"
    assert env.portfolio.positions.get("TEST") is None
    assert env.portfolio.get_equity() >= config.initial_capital  # profited from price jump


def test_stop_loss_triggers_on_price_drawdown(make_parquet):
    start_index = 140
    drop_index = start_index + 1

    def modifier(df: pd.DataFrame) -> None:
        df.loc[drop_index, "close"] = df.loc[drop_index - 1, "close"] * 0.94
        df.loc[drop_index, "high"] = df.loc[drop_index, "close"] * 1.001
        df.loc[drop_index, "low"] = df.loc[drop_index, "close"] * 0.999

    data_path = make_parquet(modifier=modifier)
    config = _make_config(data_path, commission_rate=0.0, slippage_bps=0.0)
    env = TradingEnvironment(config)

    env.reset(seed=11, options={"start_idx": start_index})

    env.step(TradeAction.BUY_SMALL.value)
    _, reward, terminated, truncated, info = env.step(TradeAction.HOLD.value)

    assert info["position_closed"] is not None
    assert info["position_closed"]["trigger"] == "stop_loss"
    assert info["position_closed"]["realized_pnl"] < 0
    assert env.portfolio.positions.get("TEST") is None
    assert not terminated  # episode continues even after stop loss