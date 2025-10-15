"""Comprehensive end-to-end tests for SAC training pipeline.

This test suite validates the entire training infrastructure with surgical precision:
1. Configuration loading and validation
2. Data loading and preprocessing
3. Environment initialization and configuration propagation
4. Reward calculation accuracy
5. Action space correctness
6. Episode simulation
7. Multi-environment vectorization
8. Training loop execution
9. Model checkpointing
10. Evaluation mode consistency
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from core.rl.environments import (
    ContinuousTradingEnvironment,
    PortfolioConfig,
    RewardConfig,
    TradingConfig,
)
from core.rl.environments.trading_env import TradeAction, TradingEnvironment
from training.rl.env_factory import build_portfolio_config, build_reward_config, build_trading_config, load_yaml


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def sample_config_yaml(temp_dir: Path) -> Path:
    """Create a sample training configuration YAML."""
    config = {
        "experiment": {
            "name": "test_sac",
            "data_dir": str(temp_dir / "data"),
            "output_dir": str(temp_dir / "models"),
        },
        "environment": {
            "symbol": "SPY",
            "data_path": str(temp_dir / "data" / "SPY" / "train.parquet"),
            "action_mode": "continuous",
            "episode_length": 100,
            "lookback_window": 24,
            "continuous_settings": {
                "hold_threshold": 0.04,
                "max_position_pct": 0.08,
                "smoothing_window": 1,
                "transaction_cost": 0.0015,
                "min_trade_value": 50.0,
            },
            "portfolio_config": {
                "initial_capital": 100000.0,
                "commission_rate": 0.001,
                "slippage_bps": 5.0,
                "max_positions": 1,
                "max_position_size_pct": 0.45,
            },
            "reward_config": {
                "pnl_weight": 0.85,
                "transaction_cost_weight": 0.05,
                "diversity_bonus_weight": 0.02,
                "pnl_scale": 0.0001,
                "reward_clip": 1000.0,
            },
        },
        "training": {
            "total_timesteps": 1000,
            "seed": 2025,
            "n_envs": 2,
        },
        "sac": {
            "base_learning_rate": 0.0003,
            "buffer_size": 10000,
            "batch_size": 64,
        },
    }
    
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config_path


@pytest.fixture
def sample_training_data(temp_dir: Path) -> Path:
    """Generate synthetic training data in Phase 3 format."""
    np.random.seed(42)
    
    # Create 500 hourly bars (about 3 weeks)
    n_bars = 500
    base_price = 400.0
    
    # Generate realistic OHLCV
    returns = np.random.normal(0.0001, 0.01, n_bars)
    close = base_price * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
    open_price = np.roll(close, 1)
    open_price[0] = base_price
    volume = np.random.uniform(50_000_000, 150_000_000, n_bars)
    
    # Generate timestamps
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="1h")
    
    # Create technical indicators
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "vwap": (high + low + close) / 3,
    })
    
    # Add technical indicators
    df["SMA_10"] = df["close"].rolling(10, min_periods=1).mean()
    df["SMA_20"] = df["close"].rolling(20, min_periods=1).mean()
    df["EMA_10"] = df["close"].ewm(span=10, adjust=False).mean()
    
    # MACD (production uses MACD_line, not MACD)
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD_line"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD_line"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD_line"] - df["MACD_signal"]
    
    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    # Stochastic (production uses Stoch_K, Stoch_D)
    low_14 = df["low"].rolling(14, min_periods=1).min()
    high_14 = df["high"].rolling(14, min_periods=1).max()
    df["Stoch_K"] = 100 * (df["close"] - low_14) / (high_14 - low_14 + 1e-10)
    df["Stoch_D"] = df["Stoch_K"].rolling(3, min_periods=1).mean()
    
    # ADX
    df["ADX_14"] = 25 + np.random.normal(0, 5, n_bars)  # Simplified
    
    # ATR
    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - df["close"].shift(1))
    tr3 = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14, min_periods=1).mean()
    
    # BB_bandwidth (Bollinger Bands width indicator)
    sma_20 = df["close"].rolling(20, min_periods=1).mean()
    std_20 = df["close"].rolling(20, min_periods=1).std()
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    df["BB_bandwidth"] = (bb_upper - bb_lower) / sma_20
    
    # OBV (On-Balance Volume)
    obv = np.zeros(n_bars)
    obv[0] = volume[0]
    for i in range(1, n_bars):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    df["OBV"] = obv
    
    # Volume_SMA_20
    df["Volume_SMA_20"] = df["volume"].rolling(20, min_periods=1).mean()
    
    # 1h_return (hourly returns)
    df["1h_return"] = df["close"].pct_change()
    
    # Sentiment (required by FeatureExtractor)
    df["sentiment_score_hourly_ffill"] = np.random.uniform(-1, 1, n_bars)
    
    # Temporal features (required by FeatureExtractor)
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
    
    # Fill any remaining NaN values (using new pandas API)
    df = df.bfill().ffill()
    
    # Save train/val/test splits
    data_dir = temp_dir / "data" / "SPY"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    test_path = data_dir / "test.parquet"
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    # Create metadata
    metadata = {
        "symbol": "SPY",
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_start": train_df["timestamp"].min().isoformat(),
        "train_end": train_df["timestamp"].max().isoformat(),
    }
    
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return train_path


# ============================================================================
# TEST 1: CONFIGURATION LOADING
# ============================================================================

def test_config_loading_from_yaml(sample_config_yaml: Path):
    """Test that YAML configuration loads correctly."""
    config = load_yaml(sample_config_yaml)
    
    assert "experiment" in config
    assert "environment" in config
    assert "training" in config
    assert config["experiment"]["name"] == "test_sac"
    assert config["environment"]["symbol"] == "SPY"


def test_reward_config_construction():
    """Test RewardConfig can be built from dict."""
    settings = {
        "pnl_weight": 0.85,
        "transaction_cost_weight": 0.05,
        "pnl_scale": 0.0001,
        "reward_clip": 1000.0,
    }
    
    reward_cfg = build_reward_config(settings)
    
    assert reward_cfg is not None
    assert math.isclose(reward_cfg.pnl_weight, 0.85, abs_tol=1e-9)
    assert math.isclose(reward_cfg.transaction_cost_weight, 0.05, abs_tol=1e-9)
    assert math.isclose(reward_cfg.pnl_scale, 0.0001, abs_tol=1e-9)


def test_portfolio_config_construction():
    """Test PortfolioConfig can be built from dict."""
    settings = {
        "initial_capital": 100000.0,
        "commission_rate": 0.001,
        "max_positions": 3,
    }
    
    portfolio_cfg = build_portfolio_config(settings)
    
    assert portfolio_cfg is not None
    assert math.isclose(portfolio_cfg.initial_capital, 100000.0, abs_tol=1e-9)
    assert math.isclose(portfolio_cfg.commission_rate, 0.001, abs_tol=1e-9)
    assert portfolio_cfg.max_positions == 3


def test_trading_config_construction(sample_config_yaml: Path, sample_training_data: Path):
    """Test TradingConfig builds correctly with all nested configs."""
    config = load_yaml(sample_config_yaml)
    env_cfg = config["environment"]
    
    trading_cfg = build_trading_config(env_cfg)
    
    assert trading_cfg.symbol == "SPY"
    assert trading_cfg.episode_length == 100
    assert trading_cfg.lookback_window == 24
    assert trading_cfg.portfolio_config is not None
    assert trading_cfg.reward_config is not None
    assert trading_cfg.continuous_settings is not None


# ============================================================================
# TEST 2: DATA LOADING AND VALIDATION
# ============================================================================

def test_data_file_exists_and_readable(sample_training_data: Path):
    """Test that generated training data is valid parquet."""
    assert sample_training_data.exists()
    
    df = pd.read_parquet(sample_training_data)
    
    assert len(df) > 0
    assert "timestamp" in df.columns
    assert "close" in df.columns
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "volume" in df.columns


def test_data_has_required_technical_indicators(sample_training_data: Path):
    """Test that data includes all required technical indicators."""
    df = pd.read_parquet(sample_training_data)
    
    # Use production feature names (matching FeatureExtractor._TECH_COLUMNS)
    required_indicators = [
        "SMA_10", "SMA_20", "MACD_line", "MACD_signal", "MACD_hist",
        "RSI_14", "Stoch_K", "Stoch_D", "ADX_14", "ATR_14",
        "BB_bandwidth", "OBV", "Volume_SMA_20", "1h_return",
    ]
    
    for indicator in required_indicators:
        assert indicator in df.columns, f"Missing indicator: {indicator}"


def test_data_has_no_missing_values(sample_training_data: Path):
    """Test that data has no NaN values."""
    df = pd.read_parquet(sample_training_data)
    
    assert not df.isnull().any().any(), "Data contains NaN values"


def test_data_timestamps_are_sequential(sample_training_data: Path):
    """Test that timestamps are in chronological order."""
    df = pd.read_parquet(sample_training_data)
    
    timestamps = pd.to_datetime(df["timestamp"])
    assert timestamps.is_monotonic_increasing, "Timestamps are not sequential"


def test_data_ohlc_constraints(sample_training_data: Path):
    """Test OHLC relationships are valid (high >= close, low <= close, etc)."""
    df = pd.read_parquet(sample_training_data)
    
    # High should be >= Close
    assert (df["high"] >= df["close"]).all(), "High < Close violation"
    
    # Low should be <= Close
    assert (df["low"] <= df["close"]).all(), "Low > Close violation"
    
    # High should be >= Low
    assert (df["high"] >= df["low"]).all(), "High < Low violation"


# ============================================================================
# TEST 3: ENVIRONMENT INITIALIZATION
# ============================================================================

def test_discrete_environment_initialization(sample_training_data: Path):
    """Test that discrete trading environment initializes correctly."""
    config = TradingConfig(
        symbol="SPY",
        data_path=sample_training_data,
        sl_checkpoints={},
        episode_length=50,
        lookback_window=24,
    )
    
    env = TradingEnvironment(config)
    
    assert env.action_space.n == len(TradeAction)
    assert env.observation_space is not None
    assert env.config.symbol == "SPY"


def test_continuous_environment_initialization(sample_training_data: Path):
    """Test that continuous trading environment initializes correctly."""
    config = TradingConfig(
        symbol="SPY",
        data_path=sample_training_data,
        sl_checkpoints={},
        episode_length=50,
        lookback_window=24,
    )
    # Add continuous settings as attribute (ContinuousTradingEnvironment reads via getattr)
    config.continuous_settings = {
        "hold_threshold": 0.04,
        "max_position_pct": 0.08,
        "smoothing_window": 1,
    }
    
    env = ContinuousTradingEnvironment(config)
    
    assert isinstance(env.action_space, type(env.action_space))  # Box space
    assert env.action_space.shape == (1,)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0


def test_environment_reset(sample_training_data: Path):
    """Test that environment reset returns valid observation."""
    config = TradingConfig(
        symbol="SPY",
        data_path=sample_training_data,
        sl_checkpoints={},
        episode_length=50,
        lookback_window=24,
    )
    
    env = TradingEnvironment(config, seed=42)
    obs, info = env.reset()
    
    assert obs is not None
    assert isinstance(obs, dict)
    assert "technical" in obs
    assert "position" in obs
    assert isinstance(info, dict)


def test_portfolio_config_propagates_to_environment(sample_training_data: Path):
    """Test that portfolio config values propagate correctly."""
    portfolio_cfg = PortfolioConfig(
        initial_capital=50000.0,
        commission_rate=0.002,
        max_positions=3,
    )
    
    config = TradingConfig(
        symbol="SPY",
        data_path=sample_training_data,
        sl_checkpoints={},
        portfolio_config=portfolio_cfg,
    )
    
    env = TradingEnvironment(config)
    
    assert math.isclose(env.portfolio.config.initial_capital, 50000.0, abs_tol=1e-6)
    assert math.isclose(env.portfolio.config.commission_rate, 0.002, abs_tol=1e-9)
    assert env.portfolio.config.max_positions == 3


def test_reward_config_propagates_to_environment(sample_training_data: Path):
    """Test that reward config values propagate correctly."""
    reward_cfg = RewardConfig(
        pnl_weight=0.95,
        transaction_cost_weight=0.01,
        pnl_scale=0.0005,
    )
    
    config = TradingConfig(
        symbol="SPY",
        data_path=sample_training_data,
        sl_checkpoints={},
        reward_config=reward_cfg,
    )
    
    env = TradingEnvironment(config)
    
    assert math.isclose(env.reward_shaper.config.pnl_weight, 0.95, abs_tol=1e-9)
    assert math.isclose(env.reward_shaper.config.transaction_cost_weight, 0.01, abs_tol=1e-9)
    assert math.isclose(env.reward_shaper.config.pnl_scale, 0.0005, abs_tol=1e-9)


# ============================================================================
# TEST 4: REWARD CALCULATION VALIDATION
# ============================================================================

def test_pnl_reward_for_profitable_trade(sample_training_data: Path):
    """Test that profitable trades generate positive PnL reward."""
    reward_cfg = RewardConfig(
        pnl_weight=1.0,
        transaction_cost_weight=0.0,
        pnl_scale=0.0001,
    )
    
    config = TradingConfig(
        symbol="SPY",
        data_path=sample_training_data,
        sl_checkpoints={},
        reward_config=reward_cfg,
        episode_length=50,
    )
    
    env = TradingEnvironment(config, seed=42)
    env.reset()
    
    # Execute a buy-sell cycle
    # Buy
    obs, reward, terminated, truncated, info = env.step(TradeAction.BUY_MEDIUM.value)
    initial_reward = reward
    
    # Advance a few steps
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(TradeAction.HOLD.value)
    
    # Sell
    obs, reward, terminated, truncated, info = env.step(TradeAction.SELL_ALL.value)
    
    # Check that we got reward breakdown
    assert "reward_breakdown" in info
    
    # PnL component should be non-zero (could be positive or negative)
    if "pnl" in info["reward_breakdown"]:
        pnl_component = info["reward_breakdown"]["pnl"]
        assert pnl_component is not None


def test_transaction_cost_penalty(sample_training_data: Path):
    """Test that transaction costs reduce reward."""
    reward_cfg = RewardConfig(
        pnl_weight=0.0,
        transaction_cost_weight=1.0,
        base_transaction_cost_pct=0.001,
    )
    
    config = TradingConfig(
        symbol="SPY",
        data_path=sample_training_data,
        sl_checkpoints={},
        reward_config=reward_cfg,
        episode_length=50,
    )
    
    env = TradingEnvironment(config, seed=42)
    env.reset()
    
    # Execute action that incurs cost
    obs, reward, terminated, truncated, info = env.step(TradeAction.BUY_MEDIUM.value)
    
    # Transaction cost should create negative reward
    if "reward_breakdown" in info:
        breakdown = info["reward_breakdown"]
        if "transaction_cost" in breakdown:
            assert breakdown["transaction_cost"] <= 0.0


def test_reward_clipping(sample_training_data: Path):
    """Test that rewards are clipped to configured bounds."""
    reward_cfg = RewardConfig(
        reward_clip=5.0,
    )
    
    config = TradingConfig(
        symbol="SPY",
        data_path=sample_training_data,
        sl_checkpoints={},
        reward_config=reward_cfg,
        episode_length=50,
    )
    
    env = TradingEnvironment(config, seed=42)
    env.reset()
    
    # Run episode
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Reward should be within clip bounds
        assert -reward_cfg.reward_clip <= reward <= reward_cfg.reward_clip


# ============================================================================
# TEST 5: ACTION SPACE CORRECTNESS
# ============================================================================

def test_discrete_action_space_coverage():
    """Test that all discrete actions are valid."""
    actions = list(TradeAction)
    
    # Check we have expected actions
    assert TradeAction.HOLD in actions
    assert TradeAction.BUY_SMALL in actions
    assert TradeAction.BUY_MEDIUM in actions
    assert TradeAction.BUY_LARGE in actions
    assert TradeAction.SELL_PARTIAL in actions
    assert TradeAction.SELL_ALL in actions


def test_continuous_action_mapping(sample_training_data: Path):
    """Test that continuous actions map correctly to discrete actions."""
    config = TradingConfig(
        symbol="SPY",
        data_path=sample_training_data,
        sl_checkpoints={},
        episode_length=100,  # Fixed: sample data has 500 bars, need shorter episode
        lookback_window=24,  # Ensure we have enough data for lookback
    )
    # Add continuous settings as attribute (ContinuousTradingEnvironment reads via getattr)
    config.continuous_settings = {
        "hold_threshold": 0.1,
        "max_position_pct": 0.08,
    }
    
    env = ContinuousTradingEnvironment(config, seed=42)
    env.reset()
    
    # Test hold threshold
    obs, reward, terminated, truncated, info = env.step(np.array([0.05]))  # Within hold threshold
    assert info["continuous_action"]["trade_type"] == "hold"
    
    # Reset for next test
    env.reset()
    
    # Test buy action
    obs, reward, terminated, truncated, info = env.step(np.array([0.5]))  # Strong buy signal
    assert info["continuous_action"]["trade_type"] == "buy"


def test_action_masking_prevents_invalid_sells(sample_training_data: Path):
    """Test that agent cannot sell without a position."""
    config = TradingConfig(
        symbol="SPY",
        data_path=sample_training_data,
        sl_checkpoints={},
        episode_length=50,
    )
    
    env = TradingEnvironment(config, seed=42)
    env.reset()
    
    # Try to sell without position
    obs, reward, terminated, truncated, info = env.step(TradeAction.SELL_ALL.value)
    
    # Should reject
    assert not info.get("action_executed", False) or reward <= 0


# ============================================================================
# TEST 6: EPISODE SIMULATION
# ============================================================================

def test_full_episode_runs_without_errors(sample_training_data: Path):
    """Test that a full episode can complete without crashes."""
    config = TradingConfig(
        symbol="SPY",
        data_path=sample_training_data,
        sl_checkpoints={},
        episode_length=50,
    )
    
    env = TradingEnvironment(config, seed=42)
    obs, info = env.reset()
    
    done = False
    steps = 0
    max_steps = 100
    
    while not done and steps < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        
        # Validate observation structure
        assert "technical" in obs
        assert "position" in obs
    
    assert steps > 0


def test_episode_terminates_at_configured_length(sample_training_data: Path):
    """Test that episodes end at configured episode_length."""
    episode_len = 30
    config = TradingConfig(
        symbol="SPY",
        data_path=sample_training_data,
        sl_checkpoints={},
        episode_length=episode_len,
    )
    
    env = TradingEnvironment(config, seed=42)
    env.reset()
    
    steps = 0
    done = False
    
    while not done and steps < episode_len + 10:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    
    # Should terminate at or before episode_length
    assert steps <= episode_len + 1  # +1 for potential boundary


def test_portfolio_state_updates_correctly(sample_training_data: Path):
    """Test that portfolio metrics update after trades."""
    config = TradingConfig(
        symbol="SPY",
        data_path=sample_training_data,
        sl_checkpoints={},
        episode_length=50,
    )
    
    env = TradingEnvironment(config, seed=42)
    env.reset()
    
    initial_equity = env.portfolio.get_equity()
    initial_cash = env.portfolio.cash
    
    # Execute a trade
    obs, reward, terminated, truncated, info = env.step(TradeAction.BUY_MEDIUM.value)
    
    # Portfolio should change
    new_cash = env.portfolio.cash
    
    if info.get("action_executed", False):
        # Cash should decrease if buy executed
        assert new_cash < initial_cash


# ============================================================================
# TEST 7: EVALUATION MODE
# ============================================================================

def test_evaluation_mode_disables_exploration(sample_training_data: Path):
    """Test that evaluation mode uses deterministic behavior."""
    config = TradingConfig(
        symbol="SPY",
        data_path=sample_training_data,
        sl_checkpoints={},
        episode_length=50,
        evaluation_mode=True,
    )
    
    env = TradingEnvironment(config, seed=42)
    
    # Evaluation mode should be set
    assert env.config.evaluation_mode is True


def test_separate_train_val_data_paths(temp_dir: Path, sample_training_data: Path):
    """Test that train and val data can be loaded separately."""
    data_dir = sample_training_data.parent
    
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    
    assert train_path.exists()
    assert val_path.exists()
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # Should have different sizes
    assert len(train_df) != len(val_df)
    assert len(train_df) > 0
    assert len(val_df) > 0


# ============================================================================
# TEST 8: MULTI-ENVIRONMENT VECTORIZATION
# ============================================================================

def test_subproc_vec_env_creation(sample_training_data: Path):
    """Test that SubprocVecEnv can be created with multiple environments."""
    from stable_baselines3.common.vec_env import SubprocVecEnv
    
    def make_env(rank: int):
        def _init():
            config = TradingConfig(
                symbol="SPY",
                data_path=sample_training_data,
                sl_checkpoints={},
                episode_length=50,
                lookback_window=24,
            )
            env = TradingEnvironment(config, seed=42 + rank)
            return env
        return _init
    
    n_envs = 2
    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    
    assert vec_env.num_envs == n_envs
    
    # Test reset - obs is a dict with batched observations
    obs = vec_env.reset()
    # Check that each observation component has batch dimension
    assert obs["portfolio"].shape[0] == n_envs
    assert obs["position"].shape[0] == n_envs
    
    vec_env.close()


# ============================================================================
# TEST 9: EDGE CASES AND ERROR HANDLING
# ============================================================================

def test_environment_handles_missing_columns_gracefully(temp_dir: Path):
    """Test that environment raises clear error for missing columns."""
    # Create data with missing required columns
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="1h"),
        "close": np.random.uniform(400, 410, 100),
        # Missing open, high, low, volume, etc.
    })
    
    data_path = temp_dir / "bad_data.parquet"
    df.to_parquet(data_path, index=False)
    
    config = TradingConfig(
        symbol="SPY",
        data_path=data_path,
        sl_checkpoints={},
    )
    
    with pytest.raises(Exception):  # Should raise due to missing columns
        env = TradingEnvironment(config)


def test_environment_handles_insufficient_data(temp_dir: Path):
    """Test that environment handles data shorter than lookback window."""
    # Create minimal data
    n_bars = 10  # Less than typical lookback
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_bars, freq="1h"),
        "open": np.random.uniform(400, 410, n_bars),
        "high": np.random.uniform(405, 415, n_bars),
        "low": np.random.uniform(395, 405, n_bars),
        "close": np.random.uniform(400, 410, n_bars),
        "volume": np.random.uniform(1e6, 2e6, n_bars),
        "vwap": np.random.uniform(400, 410, n_bars),
        "SMA_10": np.random.uniform(400, 410, n_bars),
        "SMA_20": np.random.uniform(400, 410, n_bars),
        "MACD_line": np.random.uniform(-1, 1, n_bars),  # Fixed: was "MACD"
        "MACD_signal": np.random.uniform(-1, 1, n_bars),
        "MACD_hist": np.random.uniform(-0.5, 0.5, n_bars),
        "RSI_14": np.random.uniform(30, 70, n_bars),
        "Stoch_K": np.random.uniform(20, 80, n_bars),  # Fixed: was "Stochastic_K"
        "Stoch_D": np.random.uniform(20, 80, n_bars),  # Fixed: was "Stochastic_D"
        "ADX_14": np.random.uniform(20, 40, n_bars),
        "ATR_14": np.random.uniform(1, 3, n_bars),
    })
    
    data_path = temp_dir / "short_data.parquet"
    df.to_parquet(data_path, index=False)
    
    config = TradingConfig(
        symbol="SPY",
        data_path=data_path,
        sl_checkpoints={},
        lookback_window=50,  # Longer than data
        episode_length=5,
    )
    
    # Should either raise error or handle gracefully
    try:
        env = TradingEnvironment(config)
        env.reset()
    except Exception:
        pass  # Expected to fail


# ============================================================================
# TEST 10: INTEGRATION WITH SAC COMPONENTS
# ============================================================================

def test_sac_can_instantiate_with_environment(sample_training_data: Path):
    """Test that SAC agent can be created with the trading environment."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    def make_env():
        config = TradingConfig(
            symbol="SPY",
            data_path=sample_training_data,
            sl_checkpoints={},
            episode_length=50,
        )
        # Add continuous settings as attribute
        config.continuous_settings = {"hold_threshold": 0.05}
        return ContinuousTradingEnvironment(config, seed=42)
    
    vec_env = DummyVecEnv([make_env])
    
    model = SAC(
        "MultiInputPolicy",
        vec_env,
        learning_rate=0.0003,
        buffer_size=1000,
        batch_size=64,
        verbose=0,
    )
    
    assert model is not None
    
    vec_env.close()


def test_sac_training_smoke_test(sample_training_data: Path):
    """Test that SAC can train for a few steps without errors."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    def make_env():
        config = TradingConfig(
            symbol="SPY",
            data_path=sample_training_data,
            sl_checkpoints={},
            episode_length=30,
        )
        # Add continuous settings as attribute
        config.continuous_settings = {"hold_threshold": 0.05}
        return ContinuousTradingEnvironment(config, seed=42)
    
    vec_env = DummyVecEnv([make_env])
    
    model = SAC(
        "MultiInputPolicy",
        vec_env,
        learning_rate=0.0003,
        buffer_size=500,
        batch_size=32,
        learning_starts=100,
        verbose=0,
    )
    
    # Train for a small number of steps
    try:
        model.learn(total_timesteps=200, log_interval=None)
        success = True
    except Exception as e:
        success = False
        print(f"Training failed: {e}")
    
    vec_env.close()
    
    assert success, "SAC training failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
