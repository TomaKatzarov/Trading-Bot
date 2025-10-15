"""Comprehensive integration tests for the TradingEnvironment stack.

These tests exercise end-to-end behaviour including episode rollouts, component
interactions, edge-case handling, and determinism guarantees. They operate on a
synthetic Parquet dataset generated per-test to avoid external data
dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow", reason="pyarrow is required for Parquet fixtures")
gym = pytest.importorskip(
    "gymnasium",
    reason="gymnasium is required for RL environment tests",
)

from core.rl.environments.trading_env import TradingConfig, TradingEnvironment


@pytest.fixture(scope="module")
def sample_data_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a deterministic Parquet dataset with rich technical features."""

    tmp_path = tmp_path_factory.mktemp("env_integration")

    n = 720
    dates = pd.date_range("2024-01-01", periods=n, freq="h")

    rng = np.random.default_rng(seed=42)
    base_price = 100 + rng.standard_normal(n).cumsum() * 0.5

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": base_price + rng.normal(0, 0.1, n),
            "high": base_price + np.abs(rng.normal(0, 0.6, n)),
            "low": base_price - np.abs(rng.normal(0, 0.6, n)),
            "close": base_price,
            "volume": rng.integers(1_000, 10_000, n),
            "vwap": base_price + rng.normal(0, 0.05, n),
            "SMA_10": pd.Series(base_price).rolling(10, min_periods=1).mean().to_numpy(),
            "SMA_20": pd.Series(base_price).rolling(20, min_periods=1).mean().to_numpy(),
            "MACD_line": rng.normal(0, 0.5, n),  # Fixed: was "MACD"
            "MACD_signal": rng.normal(0, 0.3, n),
            "MACD_hist": rng.normal(0, 0.2, n),
            "RSI_14": rng.uniform(30, 70, n),
            "Stoch_K": rng.uniform(20, 80, n),  # Fixed: was "Stochastic_K"
            "Stoch_D": rng.uniform(20, 80, n),  # Fixed: was "Stochastic_D"
            "ADX_14": rng.uniform(10, 40, n),
            "ATR_14": rng.uniform(0.5, 2.0, n),
            "BB_bandwidth": rng.uniform(0.01, 0.05, n),
            "OBV": rng.integers(-1000, 1000, n).cumsum(),
            "Volume_SMA_20": pd.Series(rng.integers(1_000, 10_000, n)).rolling(20, min_periods=1).mean().to_numpy(),
            "1h_return": rng.normal(0, 0.01, n),  # Fixed: was "Return_1h"
            "sentiment_score_hourly_ffill": rng.uniform(0.4, 0.6, n),
            "DayOfWeek_sin": np.sin(np.arange(n) * 2 * np.pi / 24 / 7),
            "DayOfWeek_cos": np.cos(np.arange(n) * 2 * np.pi / 24 / 7),
        }
    )

    path = tmp_path / "TEST.parquet"
    data.to_parquet(path)
    return path


def make_env(
    data_path: Path,
    *,
    overrides: Optional[Dict[str, object]] = None,
    seed: Optional[int] = 123,
) -> TradingEnvironment:
    """Helper to instantiate the environment with optional config overrides."""

    kwargs: Dict[str, object] = {
        "symbol": "TEST",
        "data_path": data_path,
        "sl_checkpoints": {},
        "episode_length": 128,
        "lookback_window": 24,
    }
    if overrides:
        kwargs.update(overrides)

    config = TradingConfig(**kwargs)
    env = TradingEnvironment(config, seed=seed)
    return env


class TestFullEpisodeRollouts:
    """End-to-end episode behaviour."""

    def test_full_episode_random_actions(self, sample_data_path: Path) -> None:
        env = make_env(sample_data_path, seed=42)
        observation, info = env.reset(seed=42)

        assert set(observation.keys()) == {"technical", "sl_probs", "position", "portfolio", "regime"}
        assert info["episode_step"] == 0

        total_reward = 0.0
        steps = 0

        for _ in range(env.config.episode_length):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        assert steps > 0
        assert np.isfinite(total_reward)
        assert "equity" in info and info["equity"] > 0
        assert "num_trades" in info

    def test_buy_and_hold_strategy(self, sample_data_path: Path) -> None:
        env = make_env(sample_data_path, overrides={"episode_length": 64}, seed=7)
        env.reset(seed=7)

        # Buy medium on first step then hold
        _, _, _, _, info = env.step(2)
        final_info = info
        for _ in range(63):
            _, _, terminated, truncated, info = env.step(0)
            final_info = info
            if terminated or truncated:
                break

        assert final_info["num_trades"] >= 1
        assert final_info["equity"] > 0

    def test_multiple_trades_episode(self, sample_data_path: Path) -> None:
        env = make_env(sample_data_path, overrides={"episode_length": 96, "max_hold_hours": 12}, seed=21)
        env.reset(seed=21)

        completed_trades = 0
        for step in range(96):
            if step % 24 == 0:
                action = 2  # BUY_MEDIUM
            elif step % 24 == 12:
                action = 5  # SELL_ALL
            else:
                action = 0  # HOLD

            _, _, terminated, truncated, info = env.step(action)
            if info.get("position_closed") is not None:
                completed_trades += 1
            elif info.get("action_info", {}).get("trade"):
                completed_trades += 1

            if terminated or truncated:
                break

        assert completed_trades >= 1
        assert info["num_trades"] >= completed_trades


class TestComponentIntegration:
    """Cross-component wiring and data flow."""

    def test_feature_extractor_integration(self, sample_data_path: Path) -> None:
        env = make_env(sample_data_path, seed=5)
        obs, _ = env.reset()

        assert isinstance(obs["technical"], np.ndarray)
        assert obs["technical"].shape == (
            env.config.lookback_window,
            env.feature_extractor.get_feature_count(),
        )
        assert obs["sl_probs"].shape == (3,)
        assert obs["position"].shape == env.observation_space["position"].shape
        assert obs["portfolio"].shape == env.observation_space["portfolio"].shape
        assert obs["regime"].shape == env.observation_space["regime"].shape

    def test_portfolio_manager_integration(self, sample_data_path: Path) -> None:
        env = make_env(sample_data_path, seed=11)
        env.reset()

        env.step(2)  # BUY_MEDIUM
        metrics_before = env.portfolio.get_portfolio_metrics()
        assert metrics_before["num_positions"] <= env.portfolio.config.max_positions
        assert env.portfolio.get_equity() > 0

        env.step(5)  # SELL_ALL
        assert len(env.portfolio.positions) == 0
        assert env.portfolio.total_trades >= 1

    def test_reward_shaper_integration(self, sample_data_path: Path) -> None:
        env = make_env(sample_data_path, seed=13)
        env.reset()

        info = {}
        for _ in range(10):
            _, _, _, _, info = env.step(env.action_space.sample())

        assert "reward_stats" in info
        stats = info["reward_stats"]
        assert "total_reward_sum" in stats
        assert stats["steps"] >= 1


class TestEdgeCases:
    """Edge case handling and graceful degradation."""

    def test_stop_loss_trigger(self, sample_data_path: Path) -> None:
        crash_df = pd.read_parquet(sample_data_path).copy()
        crash_df.loc[40:60, "close"] = crash_df.loc[40, "close"] * 0.85
        crash_path = sample_data_path.parent / "crash.parquet"
        crash_df.to_parquet(crash_path)

        env = make_env(
            crash_path,
            overrides={"episode_length": 80, "stop_loss": 0.02},
            seed=9,
        )
        env.reset(options={"start_idx": 30})

        env.step(2)  # BUY_MEDIUM
        forced_close = False
        last_info: Dict[str, object] = {}
        for _ in range(30):
            _, _, _, _, info = env.step(0)
            last_info = info
            if len(env.portfolio.positions) == 0:
                forced_close = True
                break

        assert forced_close
        closed_trade = last_info.get("position_closed")
        if closed_trade is None:
            action_info = last_info.get("action_info")
            forced_trades = []
            if isinstance(action_info, dict):
                forced_trades = action_info.get("forced_trades", [])  # type: ignore[arg-type]
            if forced_trades:
                closed_trade = forced_trades[0]

        assert closed_trade is not None
        assert closed_trade.get("trigger") in {"stop_loss", "agent_full_close", "max_hold_time"}

    def test_episode_length_limit(self, sample_data_path: Path) -> None:
        env = make_env(sample_data_path, overrides={"episode_length": 20}, seed=17)
        env.reset()

        truncated = False
        for _ in range(25):
            _, _, terminated, trunc, _ = env.step(0)
            if trunc:
                truncated = True
                break
        assert truncated
        assert not terminated

    def test_insufficient_data(self, tmp_path: Path) -> None:
        n = 30
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.linspace(100, 110, n)
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": np.ones(n) * 1000,
                "vwap": close,
                "SMA_10": close,
                "SMA_20": close,
                "MACD_line": np.zeros(n),  # Fixed: was "MACD"
                "MACD_signal": np.zeros(n),
                "MACD_hist": np.zeros(n),
                "RSI_14": np.ones(n) * 50,
                "Stoch_K": np.ones(n) * 50,  # Fixed: was "Stochastic_K"
                "Stoch_D": np.ones(n) * 50,  # Fixed: was "Stochastic_D"
                "ADX_14": np.ones(n) * 20,
                "ATR_14": np.ones(n),
                "BB_bandwidth": np.ones(n) * 0.02,
                "OBV": np.zeros(n),
                "Volume_SMA_20": np.ones(n) * 1000,
                "1h_return": np.zeros(n),  # Fixed: was "Return_1h"
                "sentiment_score_hourly_ffill": np.ones(n) * 0.5,
                "DayOfWeek_sin": np.zeros(n),
                "DayOfWeek_cos": np.ones(n),
            }
        )
        path = tmp_path / "small.parquet"
        df.to_parquet(path)

        env = make_env(path, overrides={"episode_length": 100})
        with pytest.raises(ValueError, match="Not enough data"):
            env.reset()


class TestPerformanceConsistency:
    """Repeatability and state management."""

    def test_deterministic_with_seed(self, sample_data_path: Path) -> None:
        env1 = make_env(sample_data_path, overrides={"episode_length": 40}, seed=101)
        obs1, _ = env1.reset(seed=101)
        rng = np.random.default_rng(7)
        actions = [int(rng.integers(0, env1.action_space.n)) for _ in range(40)]

        rewards1 = []
        for action in actions:
            _, reward, terminated, truncated, _ = env1.step(action)
            rewards1.append(reward)
            if terminated or truncated:
                break

        env2 = make_env(sample_data_path, overrides={"episode_length": 40}, seed=101)
        obs2, _ = env2.reset(seed=101)
        rewards2 = []
        for action in actions:
            _, reward, terminated, truncated, _ = env2.step(action)
            rewards2.append(reward)
            if terminated or truncated:
                break

        np.testing.assert_array_equal(obs1["technical"], obs2["technical"])
        assert rewards1 == rewards2

    def test_reset_clears_state(self, sample_data_path: Path) -> None:
        env = make_env(sample_data_path, seed=55)
        env.reset()
        for _ in range(20):
            env.step(env.action_space.sample())

        assert env.portfolio.total_trades >= 0
        assert len(env.reward_shaper.episode_rewards) > 0

        obs, info = env.reset()
        assert env.episode_step == 0
        assert len(env.portfolio.positions) == 0
        assert env.portfolio.total_trades == 0
        assert len(env.reward_shaper.episode_rewards) == 0
        assert info["episode_step"] == 0

    def test_action_rejection_for_sell_without_position(self, sample_data_path: Path) -> None:
        env = make_env(sample_data_path, seed=3)
        env.reset()

        _, _, _, _, info = env.step(5)  # SELL_ALL without position
        assert info["action_executed"] is False
        assert info["action_info"].get("reject_reason") == "no_position"

    def test_forced_close_on_max_hold_time(self, sample_data_path: Path) -> None:
        env = make_env(sample_data_path, overrides={"max_hold_hours": 4, "episode_length": 32}, seed=202)
        env.reset()

        env.step(2)  # BUY_MEDIUM
        forced_close = False
        for _ in range(16):
            _, _, _, _, info = env.step(0)
            closed_trade = info.get("position_closed")
            if closed_trade and closed_trade.get("trigger") == "max_hold_time":
                forced_close = True
                break
            action_info = info.get("action_info")
            forced_trades = []
            if isinstance(action_info, dict):
                forced_trades = action_info.get("forced_trades", [])  # type: ignore[arg-type]
            if forced_trades and forced_trades[0].get("trigger") == "max_hold_time":
                forced_close = True
                break
        assert forced_close
        assert len(env.portfolio.positions) == 0

    def test_reward_stats_present_after_episode(self, sample_data_path: Path) -> None:
        env = make_env(sample_data_path, overrides={"episode_length": 30}, seed=404)
        env.reset()

        final_info: Dict[str, object] = {}
        for _ in range(30):
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            final_info = info
            if terminated or truncated:
                break

        assert "episode" in final_info
        assert "reward_stats" in final_info
        assert final_info["reward_stats"]["steps"] >= 1

    def test_invalid_action_raises(self, sample_data_path: Path) -> None:
        env = make_env(sample_data_path, seed=808)
        env.reset()

        with pytest.raises(gym.error.InvalidAction):
            env.step(99)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
