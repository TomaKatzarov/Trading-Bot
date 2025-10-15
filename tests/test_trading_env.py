import logging
import math
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

import core.rl.environments.trading_env as trading_env_module
from core.rl.environments.trading_env import (
    TradeAction,
    TradingConfig,
    TradingEnvironment,
)
from core.rl.environments.portfolio_manager import PortfolioConfig, Position
from core.rl.environments.reward_shaper import RewardConfig


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
    params = {
        "symbol": "TEST",
        "data_path": data_path,
        "sl_checkpoints": {},
        "episode_length": 64,
    }
    params.update(overrides)
    return TradingConfig(**params)


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
    # Check that a position exists for TEST symbol (multi-position uses position_id as key)
    test_positions = [p for p in env.portfolio.positions.values() if p.symbol == "TEST"]
    assert len(test_positions) == 1
    assert not terminated and not truncated

    _, reward, terminated, truncated, info = env.step(TradeAction.SELL_ALL.value)
    assert info["position_closed"] is not None
    assert info["position_closed"].get("trigger") == "agent_full_close"
    # Position should be closed
    test_positions = [p for p in env.portfolio.positions.values() if p.symbol == "TEST"]
    assert len(test_positions) == 0
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
    # Position should be closed (multi-position uses position_id as key)
    test_positions = [p for p in env.portfolio.positions.values() if p.symbol == "TEST"]
    assert len(test_positions) == 0
    assert not terminated  # episode continues even after stop loss


def test_take_profit_triggers_on_price_spike(make_parquet):
    start_index = 150
    spike_index = start_index + 1

    def modifier(df: pd.DataFrame) -> None:
        df.loc[spike_index, "close"] = df.loc[spike_index - 1, "close"] * 1.05
        df.loc[spike_index, "high"] = df.loc[spike_index, "close"] * 1.002
        df.loc[spike_index, "low"] = df.loc[spike_index, "close"] * 0.998

    data_path = make_parquet(modifier=modifier)
    config = _make_config(data_path, commission_rate=0.0, slippage_bps=0.0)
    env = TradingEnvironment(config)

    env.reset(seed=13, options={"start_idx": start_index})

    env.step(TradeAction.BUY_SMALL.value)
    _, reward, terminated, truncated, info = env.step(TradeAction.HOLD.value)

    assert info["position_closed"] is not None
    assert info["position_closed"]["trigger"] == "take_profit"
    # Position should be closed (multi-position uses position_id as key)
    test_positions = [p for p in env.portfolio.positions.values() if p.symbol == "TEST"]
    assert len(test_positions) == 0
    assert not terminated and not truncated


def test_trading_config_prefers_provided_portfolio() -> None:
    dummy_path = Path("/tmp/nonexistent.parquet")
    portfolio_cfg = PortfolioConfig(initial_capital=250_000.0, max_positions=2)
    config = TradingConfig(symbol="TEST", data_path=dummy_path, sl_checkpoints={}, portfolio_config=portfolio_cfg)

    assert config.get_portfolio_config() is portfolio_cfg


def test_math_utilities_cover_edge_cases() -> None:
    assert trading_env_module._safe_divide(1.0, 0.0, default=7.5) == pytest.approx(7.5)
    assert trading_env_module._safe_divide(9.0, 3.0) == pytest.approx(3.0)
    assert trading_env_module._clip01(-0.2) == 0.0
    assert trading_env_module._clip01(1.7) == 1.0
    assert trading_env_module._timestamp_str(pd.NaT) == "NaT"


def test_logger_respects_existing_handlers(make_parquet):
    handler = logging.StreamHandler()
    trading_env_module.logger.addHandler(handler)
    trading_env_module.logger.setLevel(logging.WARNING)

    data_path = make_parquet()
    config = _make_config(data_path, log_level=logging.DEBUG)
    env = TradingEnvironment(config)

    try:
        assert trading_env_module.logger.level == logging.DEBUG
    finally:
        trading_env_module.logger.removeHandler(handler)
        env.close()


def test_step_handles_truncation_and_bankruptcy(make_parquet, monkeypatch, caplog):
    data_path = make_parquet()
    config = _make_config(data_path, episode_length=10)
    env = TradingEnvironment(config)
    env.reset(seed=3, options={"start_idx": config.lookback_window + 2})

    env.render_mode = "human"
    render_calls: list[Dict[str, Any]] = []
    monkeypatch.setattr(env, "_render_human", lambda info: render_calls.append(info))

    monkeypatch.setattr(env.portfolio, "get_equity", lambda: env.portfolio.config.initial_capital * 0.4)
    env.current_step = len(env.data) - 1
    env.episode_step = env.config.episode_length - 1

    with caplog.at_level(logging.WARNING, logger=trading_env_module.logger.name):
        observation, reward, terminated, truncated, info = env.step(TradeAction.HOLD.value)

    assert terminated is True and truncated is True
    assert "Bankruptcy condition triggered" in caplog.text
    assert render_calls, "_render_human should be invoked in human mode"
    assert "episode" in info and info["episode"]["equity_final"] == pytest.approx(env.portfolio.config.initial_capital * 0.4)
    assert observation["technical"].shape[0] == config.lookback_window

    env.close()


def test_close_invokes_portfolio_reset(make_parquet, monkeypatch):
    data_path = make_parquet()
    config = _make_config(data_path)
    env = TradingEnvironment(config)
    called = {"value": False}

    def fake_reset() -> None:
        called["value"] = True

    monkeypatch.setattr(env.portfolio, "reset", fake_reset)
    env.close()

    assert called["value"] is True


def test_load_data_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.parquet"
    config = TradingConfig(symbol="TEST", data_path=missing_path, sl_checkpoints={})

    with pytest.raises(FileNotFoundError):
        TradingEnvironment(config)


def test_load_data_requires_timestamp(tmp_path: Path) -> None:
    file_path = tmp_path / "no_timestamp.parquet"
    df = pd.DataFrame({"close": np.linspace(100, 102, 10)})
    df.to_parquet(file_path, index=False)

    config = TradingConfig(symbol="TEST", data_path=file_path, sl_checkpoints={})

    with pytest.raises(ValueError, match="timestamp"):
        TradingEnvironment(config)


def test_load_data_filters_training_range(make_parquet):
    data_path = make_parquet()
    config = _make_config(
        data_path,
        train_start="2024-01-05T00:00:00+00:00",
        train_end="2024-01-07T00:00:00+00:00",
    )
    env = TradingEnvironment(config)

    assert env.data["timestamp"].min() >= pd.Timestamp("2024-01-05T00:00:00+00:00")
    assert env.data["timestamp"].max() <= pd.Timestamp("2024-01-07T00:00:00+00:00")

    env.close()


def test_load_data_empty_after_filtering(make_parquet):
    data_path = make_parquet()
    config = _make_config(
        data_path,
        train_start="2026-01-01T00:00:00+00:00",
        train_end="2026-01-02T00:00:00+00:00",
    )

    with pytest.raises(ValueError, match="empty"):
        TradingEnvironment(config)


def test_add_position_gate_blocks_overexposure(make_parquet):
    data_path = make_parquet()
    reward_cfg = RewardConfig()
    reward_cfg.failed_action_penalty = -0.1

    config = _make_config(
        data_path,
        commission_rate=0.0,
        slippage_bps=0.0,
        reward_config=reward_cfg,
        stop_loss=1.0,
        take_profit=1.0,
        add_position_gate_enabled=True,
        add_position_gate_max_exposure_pct=0.12,
        add_position_gate_min_unrealized_pct=0.0,
        add_position_gate_base_penalty=0.25,
        add_position_gate_severity_multiplier=0.5,
        add_position_gate_penalty_cap=1.2,
        add_position_gate_violation_decay=5,
    )

    env = TradingEnvironment(config)
    try:
        env.reset(seed=17, options={"start_idx": config.lookback_window + 8})

        env.step(TradeAction.BUY_LARGE.value)
        _, reward1, _, _, info1 = env.step(TradeAction.ADD_POSITION.value)

        assert info1["action_info"]["reject_reason"] == "add_gate_exposure"
        gate_payload = info1.get("add_position_gate")
        assert gate_payload is not None
        assert gate_payload["reason"] == "add_gate_exposure"
        assert math.isclose(gate_payload["penalty"], -0.25, rel_tol=1e-6)
        assert reward1 - info1["reward_breakdown"]["total"] == pytest.approx(gate_payload["penalty"])

        _, reward2, _, _, info2 = env.step(TradeAction.ADD_POSITION.value)
        gate_payload_second = info2.get("add_position_gate")
        assert gate_payload_second is not None
        assert gate_payload_second["reason"] == "add_gate_exposure"
        assert gate_payload_second["streak"] >= 2
        assert abs(gate_payload_second["penalty"]) > abs(gate_payload["penalty"])
        assert reward2 - info2["reward_breakdown"]["total"] == pytest.approx(gate_payload_second["penalty"])
    finally:
        env.close()


def test_add_position_gate_blocks_negative_unrealized(make_parquet):
    start_idx = 180
    drop_index = start_idx + 1

    def modifier(df: pd.DataFrame) -> None:
        df.loc[drop_index, "close"] = df.loc[drop_index - 1, "close"] * 0.985
        df.loc[drop_index, "high"] = df.loc[drop_index, "close"] * 1.001
        df.loc[drop_index, "low"] = df.loc[drop_index, "close"] * 0.999

    data_path = make_parquet(modifier=modifier)
    reward_cfg = RewardConfig()
    reward_cfg.failed_action_penalty = -0.1

    config = _make_config(
        data_path,
        commission_rate=0.0,
        slippage_bps=0.0,
        reward_config=reward_cfg,
        stop_loss=1.0,
        take_profit=1.0,
        add_position_gate_enabled=True,
        add_position_gate_max_exposure_pct=0.5,
        add_position_gate_min_unrealized_pct=0.0,
        add_position_gate_base_penalty=0.25,
        add_position_gate_severity_multiplier=0.5,
        add_position_gate_penalty_cap=1.2,
        add_position_gate_violation_decay=5,
    )

    env = TradingEnvironment(config)
    try:
        env.reset(seed=23, options={"start_idx": start_idx})

        env.step(TradeAction.BUY_SMALL.value)
        _, reward, _, _, info = env.step(TradeAction.ADD_POSITION.value)

        assert info["action_info"]["reject_reason"] == "add_gate_unrealized"
        gate_payload = info.get("add_position_gate")
        assert gate_payload is not None
        assert gate_payload["reason"] == "add_gate_unrealized"
        assert gate_payload["metrics"]["unrealized_pnl_pct"] < 0.0
        assert math.isclose(
            reward - info["reward_breakdown"]["total"],
            gate_payload["penalty"],
            rel_tol=1e-6,
        )
    finally:
        env.close()


def test_load_sl_models_handles_outcomes(make_parquet, monkeypatch, tmp_path: Path):
    data_path = make_parquet()
    good_path = tmp_path / "good.chkpt"
    good_path.write_text("good")
    fail_path = tmp_path / "fail.chkpt"
    fail_path.write_text("fail")
    none_path = tmp_path / "none.chkpt"
    none_path.write_text("none")

    def fake_load(path: Path):
        if path == fail_path:
            raise RuntimeError("boom")
        if path == none_path:
            return None
        return {"model": str(path)}

    monkeypatch.setattr(trading_env_module, "load_sl_checkpoint", fake_load)
    config = _make_config(
        data_path,
        sl_checkpoints={"GOOD": good_path, "FAIL": fail_path, "NONE": none_path},
    )
    env = TradingEnvironment(config)

    assert "good" in env.sl_models
    assert "fail" not in env.sl_models
    assert "none" not in env.sl_models

    env.sl_models = None
    monkeypatch.setattr(trading_env_module, "load_sl_checkpoint", lambda _path: None)
    env._load_sl_models()
    assert env.sl_models == {}

    env.close()


def test_get_observation_requires_pipeline(make_parquet):
    data_path = make_parquet()
    config = _make_config(data_path)
    env = TradingEnvironment(config)

    env.feature_extractor = None
    env.regime_indicators = None

    with pytest.raises(RuntimeError):
        env._get_observation()

    env.close()


def test_get_observation_sl_prob_uniform_fallback(make_parquet, monkeypatch):
    data_path = make_parquet()
    config = _make_config(data_path)
    env = TradingEnvironment(config)
    env.reset(seed=5, options={"start_idx": config.lookback_window + 3})

    zeros = np.zeros(len(TradingEnvironment.SL_MODEL_ORDER), dtype=np.float32)
    monkeypatch.setattr(env.feature_extractor, "get_sl_predictions", lambda *args, **kwargs: zeros)

    observation = env._get_observation()
    expected = np.full_like(zeros, 1.0 / len(zeros))
    assert np.allclose(observation["sl_probs"], expected)

    env.close()


def test_execute_action_rejects_invalid_target(make_parquet, monkeypatch):
    data_path = make_parquet()
    config = _make_config(data_path)
    env = TradingEnvironment(config)
    env.reset(seed=17, options={"start_idx": config.lookback_window + 5})

    monkeypatch.setattr(env.portfolio, "get_equity", lambda: 0.0)
    success, info = env._execute_action(TradeAction.BUY_SMALL.value)

    assert success is False
    assert info["reject_reason"] == "invalid_target"

    env.close()


def test_execute_action_rejects_negative_share_count(make_parquet, monkeypatch):
    data_path = make_parquet()
    config = _make_config(data_path)
    env = TradingEnvironment(config)
    env.reset(seed=21, options={"start_idx": config.lookback_window + 6})

    monkeypatch.setattr(env.portfolio, "get_equity", lambda: env.portfolio.config.initial_capital)
    env.data.loc[env.current_step, "close"] = -abs(env.data.loc[env.current_step, "close"])
    success, info = env._execute_action(TradeAction.BUY_MEDIUM.value)

    assert success is False
    assert info["reject_reason"] == "zero_shares"

    env.close()


def test_execute_action_handles_portfolio_rejection(make_parquet, monkeypatch):
    data_path = make_parquet()
    config = _make_config(data_path)
    env = TradingEnvironment(config)
    env.reset(seed=27, options={"start_idx": config.lookback_window + 4})

    monkeypatch.setattr(env.portfolio, "get_equity", lambda: env.portfolio.config.initial_capital)

    def fake_open_position(**kwargs):
        return False, None

    monkeypatch.setattr(env.portfolio, "open_position", fake_open_position)
    success, info = env._execute_action(TradeAction.BUY_LARGE.value)

    assert success is False
    assert info["reject_reason"] == "portfolio_rejected"

    env.close()


def test_execute_action_sell_partial_requires_positive_shares(make_parquet):
    data_path = make_parquet()
    config = _make_config(data_path)
    env = TradingEnvironment(config)
    env.reset(seed=33, options={"start_idx": config.lookback_window + 7})

    env.portfolio.positions[env.config.symbol] = Position(
        symbol=env.config.symbol,
        shares=0.0,
        entry_price=env.data.loc[env.current_step, "close"],
    entry_time=datetime.now(UTC),
        entry_step=env.current_step,
        cost_basis=0.0,
    )

    success, info = env._execute_action(TradeAction.SELL_PARTIAL.value)
    assert success is False
    assert info["reject_reason"] == "zero_shares"

    env.close()


def test_execute_action_handles_unknown_action(make_parquet, monkeypatch):
    class ExtendedTradeAction(IntEnum):
        HOLD = 0
        BUY_SMALL = 1
        BUY_MEDIUM = 2
        BUY_LARGE = 3
        SELL_PARTIAL = 4
        SELL_ALL = 5
        ADD_POSITION = 6
        UNKNOWN = 7

        @classmethod
        def __len__(cls) -> int:
            return 8

    monkeypatch.setattr(trading_env_module, "TradeAction", ExtendedTradeAction)

    data_path = make_parquet()
    config = _make_config(data_path)
    env = trading_env_module.TradingEnvironment(config)
    env.reset(seed=41, options={"start_idx": config.lookback_window + 8})

    success, info = env._execute_action(ExtendedTradeAction.UNKNOWN.value)
    assert success is False
    assert info["reject_reason"] == "unknown_action"

    env.close()


def test_update_position_tags_risk_trades(make_parquet, monkeypatch):
    data_path = make_parquet()
    config = _make_config(data_path)
    env = TradingEnvironment(config)
    env.reset(seed=101, options={"start_idx": config.lookback_window + 12})

    env.portfolio.positions[env.config.symbol] = Position(
        symbol=env.config.symbol,
        shares=1.0,
        entry_price=env.data.loc[env.current_step, "close"],
    entry_time=datetime.now(UTC),
        entry_step=env.current_step,
        cost_basis=env.data.loc[env.current_step, "close"],
    )

    monkeypatch.setattr(
        env.portfolio,
        "enforce_risk_limits",
        lambda *args, **kwargs: [{"exit_reason": "risk_limit"}],
    )

    trades = env._update_position()
    assert trades and trades[0]["trigger"] == "risk_limit"
    assert trades[0]["closed"] is True

    env.close()


def test_compute_reward_logs_components(make_parquet, monkeypatch, caplog):
    data_path = make_parquet()
    config = _make_config(data_path)
    env = TradingEnvironment(config)
    env.reset(seed=77, options={"start_idx": config.lookback_window + 9})

    components = {
        "pnl": 0.1,
        "transaction_cost": 0.0,
        "time_efficiency": 0.0,
        "sharpe": 0.05,
    }

    monkeypatch.setattr(env.reward_shaper, "compute_reward", lambda **_: (0.1, components))
    trading_env_module.logger.setLevel(logging.DEBUG)

    with caplog.at_level(logging.DEBUG, logger=trading_env_module.logger.name):
        env._compute_reward(TradeAction.HOLD, False, 100.0, 100.0, False)

    assert "Total reward" in caplog.text
    env.close()


def test_action_name_handles_invalid_value(make_parquet):
    data_path = make_parquet()
    config = _make_config(data_path)
    env = TradingEnvironment(config)

    assert env._action_name(999) == "999"

    env.close()