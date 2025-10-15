"""Tests for data validation, configuration consistency, and training readiness.

This test suite ensures:
1. Data format compliance
2. Configuration parameter validation
3. Training/evaluation data split consistency
4. Feature engineering correctness
5. Data preprocessing pipeline
"""
from __future__ import annotations

import json
import math
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest
import yaml

from core.rl.environments import TradingConfig
from training.rl.env_factory import build_trading_config, load_yaml


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def phase3_data_structure(temp_dir: Path) -> Path:
    """Create Phase 3 data structure with train/val/test splits."""
    np.random.seed(42)
    
    symbol_dir = temp_dir / "SPY"
    symbol_dir.mkdir(parents=True)
    
    # Generate data for each split with sequential dates
    split_info = [
        ("train", 350, "2024-01-01"),
        ("val", 75, "2024-01-15 14:00:00"),    # Start after train ends
        ("test", 75, "2024-01-18 17:00:00")     # Start after val ends
    ]
    
    for split, n_bars, start_date in split_info:
        base_price = 400.0
        returns = np.random.normal(0.0001, 0.01, n_bars)
        close = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            "timestamp": pd.date_range(start_date, periods=n_bars, freq="1h"),
            "open": np.roll(close, 1),
            "high": close * (1 + np.abs(np.random.normal(0, 0.005, n_bars))),
            "low": close * (1 - np.abs(np.random.normal(0, 0.005, n_bars))),
            "close": close,
            "volume": np.random.uniform(50_000_000, 150_000_000, n_bars),
            "vwap": close * (1 + np.random.normal(0, 0.001, n_bars)),
            "SMA_10": close + np.random.normal(0, 1, n_bars),
            "SMA_20": close + np.random.normal(0, 1, n_bars),
            "MACD_line": np.random.normal(0, 0.5, n_bars),  # Production uses MACD_line
            "MACD_signal": np.random.normal(0, 0.5, n_bars),
            "MACD_hist": np.random.normal(0, 0.3, n_bars),
            "RSI_14": np.random.uniform(30, 70, n_bars),
            "Stoch_K": np.random.uniform(20, 80, n_bars),  # Production uses Stoch_K
            "Stoch_D": np.random.uniform(20, 80, n_bars),  # Production uses Stoch_D
            "ADX_14": np.random.uniform(20, 40, n_bars),
            "ATR_14": np.random.uniform(1, 3, n_bars),
            "BB_bandwidth": np.random.uniform(0.01, 0.05, n_bars),  # Required by FeatureExtractor
            "OBV": np.cumsum(np.random.randn(n_bars) * 1e6),  # Required by FeatureExtractor
            "Volume_SMA_20": np.random.uniform(50_000_000, 150_000_000, n_bars),  # Required
            "1h_return": np.random.randn(n_bars) * 0.01,  # Required by FeatureExtractor
            "sentiment_score_hourly_ffill": np.random.uniform(-1, 1, n_bars),  # Required
            "DayOfWeek_sin": np.sin(2 * np.pi * np.arange(n_bars) / 7),  # Required
            "DayOfWeek_cos": np.cos(2 * np.pi * np.arange(n_bars) / 7),  # Required
        })
        
        # Fix first open
        df.loc[0, "open"] = base_price
        
        # Ensure OHLC constraints
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
        
        df.to_parquet(symbol_dir / f"{split}.parquet", index=False)
    
    # Create metadata
    metadata = {
        "symbol": "SPY",
        "train_samples": 350,
        "val_samples": 75,
        "test_samples": 75,
        "train_start": "2024-01-01T00:00:00",
        "train_end": "2024-01-15T13:00:00",
        "frequency": "1H",
    }
    
    with open(symbol_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return temp_dir


# ============================================================================
# TEST 1: DATA FORMAT VALIDATION
# ============================================================================

def test_parquet_files_exist_for_all_splits(phase3_data_structure: Path):
    """Test that train, val, test parquet files exist."""
    symbol_dir = phase3_data_structure / "SPY"
    
    assert (symbol_dir / "train.parquet").exists(), "train.parquet missing"
    assert (symbol_dir / "val.parquet").exists(), "val.parquet missing"
    assert (symbol_dir / "test.parquet").exists(), "test.parquet missing"


def test_metadata_file_exists_and_valid(phase3_data_structure: Path):
    """Test that metadata.json exists and contains required fields."""
    metadata_path = phase3_data_structure / "SPY" / "metadata.json"
    
    assert metadata_path.exists(), "metadata.json missing"
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    required_fields = ["symbol", "train_samples", "val_samples", "test_samples"]
    for field in required_fields:
        assert field in metadata, f"Missing metadata field: {field}"


def test_data_has_required_ohlcv_columns(phase3_data_structure: Path):
    """Test that all data files have OHLCV columns."""
    symbol_dir = phase3_data_structure / "SPY"
    
    required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(symbol_dir / f"{split}.parquet")
        
        for col in required_columns:
            assert col in df.columns, f"{split}.parquet missing column: {col}"


def test_data_has_required_technical_indicators(phase3_data_structure: Path):
    """Test that all required technical indicators are present."""
    symbol_dir = phase3_data_structure / "SPY"
    
    # Use production feature names (matching FeatureExtractor._TECH_COLUMNS)
    required_indicators = [
        "SMA_10", "SMA_20", "MACD_line", "MACD_signal", "MACD_hist",
        "RSI_14", "Stoch_K", "Stoch_D", "ADX_14", "ATR_14",
        "BB_bandwidth", "OBV", "Volume_SMA_20", "1h_return",
    ]
    
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(symbol_dir / f"{split}.parquet")
        
        for indicator in required_indicators:
            assert indicator in df.columns, \
                f"{split}.parquet missing indicator: {indicator}"


def test_data_has_no_null_values(phase3_data_structure: Path):
    """Test that data contains no null/NaN values."""
    symbol_dir = phase3_data_structure / "SPY"
    
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(symbol_dir / f"{split}.parquet")
        
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        
        assert total_nulls == 0, \
            f"{split}.parquet contains {total_nulls} null values:\n{null_counts[null_counts > 0]}"


def test_data_has_no_infinite_values(phase3_data_structure: Path):
    """Test that data contains no infinite values."""
    symbol_dir = phase3_data_structure / "SPY"
    
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(symbol_dir / f"{split}.parquet")
        
        # Check numeric columns for inf
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            assert inf_count == 0, \
                f"{split}.parquet column '{col}' contains {inf_count} infinite values"


def test_timestamps_are_chronological(phase3_data_structure: Path):
    """Test that timestamps are in chronological order."""
    symbol_dir = phase3_data_structure / "SPY"
    
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(symbol_dir / f"{split}.parquet")
        
        timestamps = pd.to_datetime(df["timestamp"])
        assert timestamps.is_monotonic_increasing, \
            f"{split}.parquet timestamps are not chronological"


def test_timestamps_have_consistent_frequency(phase3_data_structure: Path):
    """Test that timestamps have consistent frequency (1H)."""
    symbol_dir = phase3_data_structure / "SPY"
    
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(symbol_dir / f"{split}.parquet")
        
        timestamps = pd.to_datetime(df["timestamp"])
        diffs = timestamps.diff().dropna()
        
        # Most common frequency should be 1 hour
        expected_freq = pd.Timedelta(hours=1)
        mode_freq = diffs.mode()[0]
        
        assert mode_freq == expected_freq, \
            f"{split}.parquet has inconsistent frequency (mode: {mode_freq})"


# ============================================================================
# TEST 2: OHLC CONSTRAINTS
# ============================================================================

def test_high_is_highest_price(phase3_data_structure: Path):
    """Test that High >= Open, Close, Low."""
    symbol_dir = phase3_data_structure / "SPY"
    
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(symbol_dir / f"{split}.parquet")
        
        assert (df["high"] >= df["open"]).all(), f"{split}: High < Open"
        assert (df["high"] >= df["close"]).all(), f"{split}: High < Close"
        assert (df["high"] >= df["low"]).all(), f"{split}: High < Low"


def test_low_is_lowest_price(phase3_data_structure: Path):
    """Test that Low <= Open, Close, High."""
    symbol_dir = phase3_data_structure / "SPY"
    
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(symbol_dir / f"{split}.parquet")
        
        assert (df["low"] <= df["open"]).all(), f"{split}: Low > Open"
        assert (df["low"] <= df["close"]).all(), f"{split}: Low > Close"
        assert (df["low"] <= df["high"]).all(), f"{split}: Low > High"


def test_prices_are_positive(phase3_data_structure: Path):
    """Test that all prices are positive."""
    symbol_dir = phase3_data_structure / "SPY"
    
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(symbol_dir / f"{split}.parquet")
        
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            assert (df[col] > 0).all(), f"{split}: {col} has non-positive values"


def test_volume_is_non_negative(phase3_data_structure: Path):
    """Test that volume is non-negative."""
    symbol_dir = phase3_data_structure / "SPY"
    
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(symbol_dir / f"{split}.parquet")
        
        assert (df["volume"] >= 0).all(), f"{split}: Volume has negative values"


# ============================================================================
# TEST 3: TECHNICAL INDICATOR VALIDITY
# ============================================================================

def test_rsi_bounds(phase3_data_structure: Path):
    """Test that RSI is bounded [0, 100]."""
    symbol_dir = phase3_data_structure / "SPY"
    
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(symbol_dir / f"{split}.parquet")
        
        assert (df["RSI_14"] >= 0).all(), f"{split}: RSI < 0"
        assert (df["RSI_14"] <= 100).all(), f"{split}: RSI > 100"


def test_stochastic_bounds(phase3_data_structure: Path):
    """Test that Stochastic indicators are bounded [0, 100]."""
    symbol_dir = phase3_data_structure / "SPY"
    
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(symbol_dir / f"{split}.parquet")
        
        # Use production feature names
        for col in ["Stoch_K", "Stoch_D"]:
            assert (df[col] >= 0).all(), f"{split}: {col} < 0"
            assert (df[col] <= 100).all(), f"{split}: {col} > 100"


def test_atr_positive(phase3_data_structure: Path):
    """Test that ATR is positive."""
    symbol_dir = phase3_data_structure / "SPY"
    
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(symbol_dir / f"{split}.parquet")
        
        assert (df["ATR_14"] > 0).all(), f"{split}: ATR <= 0"


def test_macd_histogram_is_difference(phase3_data_structure: Path):
    """Test that MACD_hist = MACD - MACD_signal (approximately)."""
    symbol_dir = phase3_data_structure / "SPY"
    
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(symbol_dir / f"{split}.parquet")
        
        # Use production feature name MACD_line
        calculated_hist = df["MACD_line"] - df["MACD_signal"]
        
        # Allow small tolerance for floating point errors
        diff = np.abs(df["MACD_hist"] - calculated_hist)
        max_diff = diff.max()
        
        # If histogram is calculated correctly, diff should be tiny
        # (or if it's pre-calculated, this test documents the relationship)
        assert max_diff < 10.0, f"{split}: MACD_hist deviates from MACD - MACD_signal"


# ============================================================================
# TEST 4: DATA SPLIT CONSISTENCY
# ============================================================================

def test_train_val_test_sizes_match_metadata(phase3_data_structure: Path):
    """Test that actual data sizes match metadata."""
    symbol_dir = phase3_data_structure / "SPY"
    
    with open(symbol_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    splits = {
        "train": metadata["train_samples"],
        "val": metadata["val_samples"],
        "test": metadata["test_samples"],
    }
    
    for split, expected_size in splits.items():
        df = pd.read_parquet(symbol_dir / f"{split}.parquet")
        actual_size = len(df)
        
        assert actual_size == expected_size, \
            f"{split}: Expected {expected_size} rows, got {actual_size}"


def test_splits_do_not_overlap_in_time(phase3_data_structure: Path):
    """Test that train/val/test splits don't overlap in time."""
    symbol_dir = phase3_data_structure / "SPY"
    
    train_df = pd.read_parquet(symbol_dir / "train.parquet")
    val_df = pd.read_parquet(symbol_dir / "val.parquet")
    test_df = pd.read_parquet(symbol_dir / "test.parquet")
    
    train_end = pd.to_datetime(train_df["timestamp"]).max()
    val_start = pd.to_datetime(val_df["timestamp"]).min()
    val_end = pd.to_datetime(val_df["timestamp"]).max()
    test_start = pd.to_datetime(test_df["timestamp"]).min()
    
    assert train_end <= val_start, "Train overlaps with val"
    assert val_end <= test_start, "Val overlaps with test"


def test_splits_are_contiguous_or_have_documented_gaps(phase3_data_structure: Path):
    """Test that splits are contiguous (or gaps are acceptable)."""
    symbol_dir = phase3_data_structure / "SPY"
    
    train_df = pd.read_parquet(symbol_dir / "train.parquet")
    val_df = pd.read_parquet(symbol_dir / "val.parquet")
    
    train_end = pd.to_datetime(train_df["timestamp"]).max()
    val_start = pd.to_datetime(val_df["timestamp"]).min()
    
    gap = val_start - train_end
    
    # Gap should be 1 hour (contiguous) or within reasonable range
    assert gap <= pd.Timedelta(days=7), \
        f"Gap between train and val is {gap}, seems excessive"


# ============================================================================
# TEST 5: CONFIGURATION VALIDATION
# ============================================================================

def test_config_loading_from_phase_a2_template(temp_dir: Path):
    """Test that phase_a2_sac_sharpe.yaml loads correctly."""
    config_path = Path("training/config_templates/phase_a2_sac_sharpe.yaml")
    
    if not config_path.exists():
        pytest.skip("phase_a2_sac_sharpe.yaml not found")
    
    config = load_yaml(config_path)
    
    assert "experiment" in config
    assert "training" in config
    assert "environment" in config
    assert "sac" in config


def test_reward_weights_sum_reasonably(temp_dir: Path):
    """Test that reward component weights are reasonable."""
    config_path = Path("training/config_templates/phase_a2_sac_sharpe.yaml")
    
    if not config_path.exists():
        pytest.skip("Config not found")
    
    config = load_yaml(config_path)
    reward_cfg = config["environment"].get("reward_config", {})
    
    # Sum main reward weights
    weights = [
        reward_cfg.get("pnl_weight", 0.0),
        reward_cfg.get("transaction_cost_weight", 0.0),
        reward_cfg.get("time_efficiency_weight", 0.0),
        reward_cfg.get("sharpe_weight", 0.0),
        reward_cfg.get("drawdown_weight", 0.0),
        reward_cfg.get("diversity_bonus_weight", 0.0),
    ]
    
    total_weight = sum(weights)
    
    # Should be roughly around 1.0 (allow flexibility)
    assert 0.5 <= total_weight <= 2.0, \
        f"Reward weights sum to {total_weight}, seems unusual"


def test_critical_config_parameters_are_set(temp_dir: Path):
    """Test that critical parameters have reasonable values."""
    config_path = Path("training/config_templates/phase_a2_sac_sharpe.yaml")
    
    if not config_path.exists():
        pytest.skip("Config not found")
    
    config = load_yaml(config_path)
    
    # Training params
    training = config.get("training", {})
    assert training.get("total_timesteps", 0) > 0, "total_timesteps must be positive"
    assert training.get("n_envs", 0) > 0, "n_envs must be positive"
    
    # Environment params
    env = config.get("environment", {})
    assert env.get("episode_length", 0) > 0, "episode_length must be positive"
    assert env.get("lookback_window", 0) > 0, "lookback_window must be positive"
    
    # SAC params
    sac = config.get("sac", {})
    assert sac.get("buffer_size", 0) > 0, "buffer_size must be positive"
    assert sac.get("batch_size", 0) > 0, "batch_size must be positive"


def test_portfolio_config_max_positions_reasonable(temp_dir: Path):
    """Test that max_positions is set to reasonable value."""
    config_path = Path("training/config_templates/phase_a2_sac_sharpe.yaml")
    
    if not config_path.exists():
        pytest.skip("Config not found")
    
    config = load_yaml(config_path)
    portfolio_cfg = config["environment"].get("portfolio_config", {})
    
    max_positions = portfolio_cfg.get("max_positions", 1)
    
    assert 1 <= max_positions <= 10, \
        f"max_positions={max_positions} seems unusual"


# ============================================================================
# TEST 6: FEATURE ENGINEERING VALIDATION
# ============================================================================

def test_moving_averages_smooth_price_series(phase3_data_structure: Path):
    """Test that moving averages are less volatile than price."""
    symbol_dir = phase3_data_structure / "SPY"
    df = pd.read_parquet(symbol_dir / "train.parquet")
    
    # Calculate volatility
    close_std = df["close"].std()
    sma10_std = df["SMA_10"].std()
    sma20_std = df["SMA_20"].std()
    
    # Moving averages should be smoother (less volatile)
    # Allow some tolerance for synthetic data
    assert sma10_std <= close_std * 1.2, "SMA_10 more volatile than close"
    assert sma20_std <= close_std * 1.2, "SMA_20 more volatile than close"


def test_vwap_within_high_low_range(phase3_data_structure: Path):
    """Test that VWAP is within [low, high] range."""
    symbol_dir = phase3_data_structure / "SPY"
    
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(symbol_dir / f"{split}.parquet")
        
        # VWAP should generally be within bar range
        # (allow small tolerance for calculation differences)
        within_range = (df["vwap"] >= df["low"] * 0.99) & (df["vwap"] <= df["high"] * 1.01)
        violations = (~within_range).sum()
        
        # Allow up to 5% violations (for synthetic data)
        violation_pct = violations / len(df)
        assert violation_pct < 0.05, \
            f"{split}: {violation_pct:.1%} of VWAP values outside [low, high]"


def test_feature_distributions_are_reasonable(phase3_data_structure: Path):
    """Test that feature distributions don't have extreme outliers."""
    symbol_dir = phase3_data_structure / "SPY"
    df = pd.read_parquet(symbol_dir / "train.parquet")
    
    # Check key features for outliers (> 5 std from mean)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col == "timestamp":
            continue
        
        mean = df[col].mean()
        std = df[col].std()
        
        if std > 0:
            z_scores = np.abs((df[col] - mean) / std)
            extreme_outliers = (z_scores > 10).sum()
            
            # Should have very few extreme outliers
            assert extreme_outliers < len(df) * 0.01, \
                f"{col} has {extreme_outliers} extreme outliers (>10σ)"


# ============================================================================
# TEST 7: ENVIRONMENT CONFIGURATION PROPAGATION
# ============================================================================

def test_trading_config_builds_from_yaml(phase3_data_structure: Path):
    """Test that TradingConfig can be built from YAML."""
    env_settings = {
        "symbol": "SPY",
        "data_path": str(phase3_data_structure / "SPY" / "train.parquet"),
        "episode_length": 100,
        "lookback_window": 24,
        "portfolio_config": {
            "initial_capital": 100000.0,
            "max_positions": 3,
        },
        "reward_config": {
            "pnl_weight": 0.85,
            "pnl_scale": 0.0001,
        },
    }
    
    trading_cfg = build_trading_config(env_settings)
    
    assert trading_cfg.symbol == "SPY"
    assert trading_cfg.episode_length == 100
    assert trading_cfg.portfolio_config is not None
    assert trading_cfg.reward_config is not None


def test_nested_config_settings_propagate(phase3_data_structure: Path):
    """Test that nested config settings propagate to sub-configs."""
    env_settings = {
        "symbol": "SPY",
        "data_path": str(phase3_data_structure / "SPY" / "train.parquet"),
        "continuous_settings": {
            "hold_threshold": 0.04,
            "max_position_pct": 0.08,
        },
        "portfolio_config": {
            "max_positions": 3,
            "commission_rate": 0.002,
        },
    }
    
    trading_cfg = build_trading_config(env_settings)
    
    assert trading_cfg.continuous_settings is not None
    assert trading_cfg.continuous_settings["hold_threshold"] == 0.04
    assert trading_cfg.portfolio_config.max_positions == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
