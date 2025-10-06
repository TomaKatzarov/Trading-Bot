"""Tests for the vectorized trading environment wrappers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.env_checker import check_env

from core.rl.environments.trading_env import TradeAction, TradingEnvironment
from core.rl.environments.vec_trading_env import (
    make_multi_symbol_vec_env,
    make_parallel_env,
    make_sequential_env,
    make_vec_trading_env,
)


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory populated with synthetic symbol datasets."""

    directory = tmp_path / "data"
    directory.mkdir()
    for symbol in ("TEST1", "TEST2", "TEST3"):
        _write_dataset(directory / f"{symbol}.parquet", symbol)
    return directory


def _write_dataset(path: Path, symbol: str, rows: int = 512) -> None:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    base_close = 100 + np.cumsum(rng.normal(0.0, 0.4, size=rows))

    df = pd.DataFrame({"timestamp": timestamps})
    df["close"] = base_close
    df["open"] = base_close * 0.999
    df["high"] = base_close * 1.0015
    df["low"] = base_close * 0.9985
    df["volume"] = rng.integers(5_000, 50_000, size=rows)
    df["vwap"] = base_close
    df["SMA_10"] = df["close"].rolling(10, min_periods=1).mean()
    df["SMA_20"] = df["close"].rolling(20, min_periods=1).mean()
    df["MACD"] = rng.normal(0.0, 0.2, size=rows)
    df["MACD_signal"] = rng.normal(0.0, 0.1, size=rows)
    df["MACD_hist"] = rng.normal(0.0, 0.05, size=rows)
    df["RSI_14"] = rng.uniform(30, 70, size=rows)
    df["Stochastic_K"] = rng.uniform(20, 80, size=rows)
    df["Stochastic_D"] = rng.uniform(20, 80, size=rows)
    df["ADX_14"] = rng.uniform(10, 40, size=rows)
    df["ATR_14"] = rng.uniform(0.5, 2.0, size=rows)
    df["BB_bandwidth"] = rng.uniform(0.01, 0.06, size=rows)
    df["OBV"] = np.cumsum(rng.integers(-5_000, 5_000, size=rows))
    df["Volume_SMA_20"] = rng.integers(5_000, 50_000, size=rows)
    df["Return_1h"] = df["close"].pct_change().fillna(0.0)
    df["sentiment_score_hourly_ffill"] = rng.uniform(0.4, 0.6, size=rows)

    day_angle = 2 * math.pi * np.arange(rows) / 24.0
    df["DayOfWeek_sin"] = np.sin(day_angle)
    df["DayOfWeek_cos"] = np.cos(day_angle)

    missing = set(TradingEnvironment.DEFAULT_FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise AssertionError(f"Dataset missing expected columns: {sorted(missing)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


class TestVectorizedEnvironmentCreation:
    """Verify vectorized environment creation paths."""

    def test_create_sequential_vec_env(self, sample_data_dir: Path) -> None:
        vec_env = make_sequential_env(
            symbol="TEST1",
            data_dir=sample_data_dir,
            num_envs=4,
            env_kwargs={"episode_length": 100},
        )

        try:
            assert vec_env.num_envs == 4
            assert vec_env.action_space.n == len(TradeAction)
        finally:
            vec_env.close()

    def test_create_parallel_vec_env(self, sample_data_dir: Path) -> None:
        vec_env = make_parallel_env(
            symbol="TEST1",
            data_dir=sample_data_dir,
            num_envs=2,
            env_kwargs={"episode_length": 100},
        )

        try:
            assert vec_env.num_envs == 2
        finally:
            vec_env.close()

    def test_create_multi_symbol_env(self, sample_data_dir: Path) -> None:
        vec_env = make_multi_symbol_vec_env(
            symbols=["TEST1", "TEST2", "TEST3"],
            data_dir=sample_data_dir,
            envs_per_symbol=2,
            use_subprocess=False,
            shared_env_kwargs={"episode_length": 120},
        )

        try:
            assert vec_env.num_envs == 6
        finally:
            vec_env.close()


class TestVectorizedEnvironmentOperations:
    """Exercise vectorized observation and stepping behaviour."""

    def test_reset_returns_batched_obs(self, sample_data_dir: Path) -> None:
        num_envs = 4
        vec_env = make_sequential_env(
            symbol="TEST1",
            data_dir=sample_data_dir,
            num_envs=num_envs,
            env_kwargs={"episode_length": 128},
        )

        try:
            observation = vec_env.reset()
            lookback = vec_env.get_attr("config")[0].lookback_window
            feature_count = len(vec_env.get_attr("feature_cols")[0])

            assert observation["technical"].shape == (num_envs, lookback, feature_count)
            assert observation["sl_probs"].shape == (num_envs, len(TradingEnvironment.SL_MODEL_ORDER))
            assert observation["position"].shape == (num_envs, 5)
            assert observation["portfolio"].shape == (num_envs, 8)
            assert observation["regime"].shape[0] == num_envs
        finally:
            vec_env.close()

    def test_step_returns_batched_results(self, sample_data_dir: Path) -> None:
        num_envs = 3
        vec_env = make_sequential_env(
            symbol="TEST1",
            data_dir=sample_data_dir,
            num_envs=num_envs,
            env_kwargs={"episode_length": 128},
        )

        try:
            vec_env.reset()
            actions = np.array([0, 1, 2])
            observation, rewards, dones, infos = vec_env.step(actions)

            assert observation["technical"].shape[0] == num_envs
            assert rewards.shape == (num_envs,)
            assert dones.shape == (num_envs,)
            assert isinstance(infos, list) and len(infos) == num_envs
        finally:
            vec_env.close()

    def test_parallel_execution(self, sample_data_dir: Path) -> None:
        vec_env = make_parallel_env(
            symbol="TEST1",
            data_dir=sample_data_dir,
            num_envs=2,
            env_kwargs={"episode_length": 64},
        )

        try:
            vec_env.reset()
            for _ in range(10):
                actions = [vec_env.action_space.sample() for _ in range(vec_env.num_envs)]
                _, rewards, dones, infos = vec_env.step(actions)

                assert rewards.shape == (vec_env.num_envs,)
                assert dones.shape == (vec_env.num_envs,)
                assert isinstance(infos, (list, tuple))
        finally:
            vec_env.close()


class TestDeterminism:
    """Ensure seeding results in deterministic rollouts."""

    def test_deterministic_with_seed(self, sample_data_dir: Path) -> None:
        rewards_a, observations_a = _collect_rollout(sample_data_dir, seed=42)
        rewards_b, observations_b = _collect_rollout(sample_data_dir, seed=42)

        np.testing.assert_array_equal(observations_a["technical"], observations_b["technical"])
        for row_a, row_b in zip(rewards_a, rewards_b):
            np.testing.assert_array_equal(row_a, row_b)


def _collect_rollout(data_dir: Path, seed: int) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    vec_env = make_sequential_env(
        symbol="TEST1",
        data_dir=data_dir,
        num_envs=2,
        seed=seed,
        env_kwargs={"episode_length": 50},
    )

    try:
        observation = vec_env.reset()
        reward_batches: List[np.ndarray] = []
        for _ in range(10):
            actions = np.array([0, 1])
            _, rewards, _, _ = vec_env.step(actions)
            reward_batches.append(rewards.copy())
        return reward_batches, observation
    finally:
        vec_env.close()


class TestSB3Integration:
    """Validate Stable-Baselines3 environment compatibility."""

    def test_sb3_check_env(self, sample_data_dir: Path) -> None:
        vec_env = make_sequential_env(
            symbol="TEST1",
            data_dir=sample_data_dir,
            num_envs=1,
            env_kwargs={"episode_length": 128},
        )

        try:
            single_env = vec_env.envs[0]
            check_env(single_env)
        finally:
            vec_env.close()


class TestResourceManagement:
    """Confirm resources are released gracefully."""

    def test_proper_cleanup(self, sample_data_dir: Path) -> None:
        vec_env = make_parallel_env(
            symbol="TEST1",
            data_dir=sample_data_dir,
            num_envs=2,
            env_kwargs={"episode_length": 128},
        )

        vec_env.reset()
        vec_env.step(np.array([0, 1]))

        vec_env.close()
        vec_env.close()


def test_make_vec_trading_env_rejects_invalid_count(sample_data_dir: Path) -> None:
    with pytest.raises(ValueError, match="num_envs must be positive"):
        make_vec_trading_env(
            symbol="TEST1",
            data_dir=sample_data_dir,
            num_envs=0,
        )
