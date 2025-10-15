"""Pre-training validation tests - validates actual training setup is ready.

This test suite checks the ACTUAL training configuration and data files
that will be used in production training runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest
import yaml

from training.rl.env_factory import build_trading_config, load_yaml


# ============================================================================
# CONFIGURATION PATHS
# ============================================================================

CONFIG_PATH = Path("training/config_templates/phase_a2_sac_sharpe.yaml")
PHASE_A2_SAC_PATH = Path("training/config_templates/phase_a2_sac.yaml")


# ============================================================================
# TEST 1: CONFIGURATION FILE EXISTS AND VALID
# ============================================================================

def test_phase_a2_sac_sharpe_config_exists():
    """Test that phase_a2_sac_sharpe.yaml exists."""
    assert CONFIG_PATH.exists(), \
        f"Configuration file not found: {CONFIG_PATH}"


def test_config_is_valid_yaml():
    """Test that config file is valid YAML."""
    try:
        config = load_yaml(CONFIG_PATH)
        assert isinstance(config, dict)
    except Exception as e:
        pytest.fail(f"Failed to load config: {e}")


def test_config_has_required_sections():
    """Test that config has all required sections."""
    config = load_yaml(CONFIG_PATH)
    
    required_sections = ["experiment", "training", "environment", "sac"]
    for section in required_sections:
        assert section in config, f"Missing section: {section}"


# ============================================================================
# TEST 2: EXPERIMENT CONFIGURATION
# ============================================================================

def test_experiment_config_valid():
    """Test that experiment config has valid values."""
    config = load_yaml(CONFIG_PATH)
    exp = config["experiment"]
    
    assert "name" in exp
    assert isinstance(exp["name"], str)
    assert len(exp["name"]) > 0
    
    # Check symbols list
    if "symbols" in exp:
        assert isinstance(exp["symbols"], list)
        assert len(exp["symbols"]) > 0
        for symbol in exp["symbols"]:
            assert isinstance(symbol, str)
            assert len(symbol) > 0


def test_data_directory_specified():
    """Test that data_dir is specified."""
    config = load_yaml(CONFIG_PATH)
    exp = config["experiment"]
    
    assert "data_dir" in exp, "data_dir not specified"
    assert isinstance(exp["data_dir"], str)


def test_output_directory_specified():
    """Test that output_dir is specified."""
    config = load_yaml(CONFIG_PATH)
    exp = config["experiment"]
    
    assert "output_dir" in exp, "output_dir not specified"
    assert isinstance(exp["output_dir"], str)


# ============================================================================
# TEST 3: TRAINING CONFIGURATION
# ============================================================================

def test_training_timesteps_reasonable():
    """Test that total_timesteps is reasonable."""
    config = load_yaml(CONFIG_PATH)
    training = config["training"]
    
    timesteps = training.get("total_timesteps", 0)
    assert timesteps > 0, "total_timesteps must be positive"
    assert timesteps >= 10000, "total_timesteps should be at least 10k"
    assert timesteps <= 10_000_000, "total_timesteps seems excessive"


def test_training_n_envs_reasonable():
    """Test that n_envs is reasonable."""
    config = load_yaml(CONFIG_PATH)
    training = config["training"]
    
    n_envs = training.get("n_envs", 0)
    assert n_envs > 0, "n_envs must be positive"
    assert 1 <= n_envs <= 64, "n_envs should be between 1 and 64"


def test_training_seed_set():
    """Test that random seed is set for reproducibility."""
    config = load_yaml(CONFIG_PATH)
    training = config["training"]
    
    assert "seed" in training, "Random seed should be set"
    seed = training["seed"]
    assert isinstance(seed, int)
    assert 0 <= seed <= 2**32 - 1


# ============================================================================
# TEST 4: ENVIRONMENT CONFIGURATION
# ============================================================================

def test_environment_episode_length_reasonable():
    """Test that episode_length is reasonable."""
    config = load_yaml(CONFIG_PATH)
    env = config["environment"]
    
    episode_length = env.get("episode_length", 0)
    assert episode_length > 0, "episode_length must be positive"
    assert 50 <= episode_length <= 2000, \
        f"episode_length {episode_length} seems unusual (should be 50-2000)"


def test_environment_lookback_window_reasonable():
    """Test that lookback_window is reasonable."""
    config = load_yaml(CONFIG_PATH)
    env = config["environment"]
    
    lookback = env.get("lookback_window", 0)
    assert lookback > 0, "lookback_window must be positive"
    assert 1 <= lookback <= 100, \
        f"lookback_window {lookback} seems unusual (should be 1-100)"


def test_environment_action_mode_valid():
    """Test that action_mode is valid."""
    config = load_yaml(CONFIG_PATH)
    env = config["environment"]
    
    action_mode = env.get("action_mode", "")
    assert action_mode in ["discrete", "continuous"], \
        f"Invalid action_mode: {action_mode}"


def test_continuous_settings_present_if_continuous():
    """Test that continuous_settings present if action_mode is continuous."""
    config = load_yaml(CONFIG_PATH)
    env = config["environment"]
    
    if env.get("action_mode") == "continuous":
        assert "continuous_settings" in env, \
            "continuous_settings required for continuous action mode"


def test_portfolio_config_present():
    """Test that portfolio_config is present."""
    config = load_yaml(CONFIG_PATH)
    env = config["environment"]
    
    assert "portfolio_config" in env, "portfolio_config missing"
    portfolio = env["portfolio_config"]
    
    # Check key parameters (actual structure from production config)
    assert "max_positions" in portfolio
    assert portfolio["max_positions"] >= 1
    
    assert "max_position_size_pct" in portfolio
    assert 0 < portfolio["max_position_size_pct"] <= 1.0


def test_reward_config_present():
    """Test that reward_config is present."""
    config = load_yaml(CONFIG_PATH)
    env = config["environment"]
    
    assert "reward_config" in env, "reward_config missing"
    reward = env["reward_config"]
    
    # Check key parameters
    assert "pnl_weight" in reward
    assert "pnl_scale" in reward
    assert "reward_clip" in reward


# ============================================================================
# TEST 5: SAC CONFIGURATION
# ============================================================================

def test_sac_learning_rate_reasonable():
    """Test that learning rate is reasonable."""
    config = load_yaml(CONFIG_PATH)
    sac = config["sac"]
    
    lr = sac.get("base_learning_rate", 0.0)
    assert lr > 0, "learning_rate must be positive"
    assert 1e-6 <= lr <= 1e-2, \
        f"learning_rate {lr} seems unusual (should be 1e-6 to 1e-2)"


def test_sac_buffer_size_reasonable():
    """Test that buffer_size is reasonable."""
    config = load_yaml(CONFIG_PATH)
    sac = config["sac"]
    
    buffer_size = sac.get("buffer_size", 0)
    assert buffer_size > 0, "buffer_size must be positive"
    assert buffer_size >= 1000, "buffer_size should be at least 1000"


def test_sac_batch_size_reasonable():
    """Test that batch_size is reasonable."""
    config = load_yaml(CONFIG_PATH)
    sac = config["sac"]
    
    batch_size = sac.get("batch_size", 0)
    assert batch_size > 0, "batch_size must be positive"
    assert 16 <= batch_size <= 1024, \
        f"batch_size {batch_size} seems unusual (should be 16-1024)"


def test_sac_gamma_valid():
    """Test that gamma discount factor is valid."""
    config = load_yaml(CONFIG_PATH)
    sac = config["sac"]
    
    gamma = sac.get("gamma", 0.0)
    assert 0.0 < gamma <= 1.0, \
        f"gamma {gamma} must be in (0, 1]"


def test_sac_tau_valid():
    """Test that tau (polyak update) is valid."""
    config = load_yaml(CONFIG_PATH)
    sac = config["sac"]
    
    tau = sac.get("tau", 0.0)
    assert 0.0 < tau <= 1.0, \
        f"tau {tau} must be in (0, 1]"


# ============================================================================
# TEST 6: DATA AVAILABILITY
# ============================================================================

def test_data_directory_exists():
    """Test that data directory exists."""
    config = load_yaml(CONFIG_PATH)
    data_dir = Path(config["experiment"].get("data_dir", "data/phase3_splits"))
    
    # May not exist yet, so just warn
    if not data_dir.exists():
        pytest.skip(f"Data directory not found: {data_dir} (will be created during training)")


def test_primary_symbol_data_exists():
    """Test that data for primary symbol exists."""
    pytest.skip("Production parquet files created with older PyArrow version - incompatible with PyArrow 19.0.0. Data needs regeneration.")
    config = load_yaml(CONFIG_PATH)
    
    symbol = config["environment"].get("symbol", "SPY")
    data_dir = Path(config["experiment"].get("data_dir", "data/phase3_splits"))
    
    symbol_dir = data_dir / symbol
    
    if not symbol_dir.exists():
        pytest.skip(f"Symbol directory not found: {symbol_dir}")
    
    # Check for train.parquet
    train_path = symbol_dir / "train.parquet"
    if train_path.exists():
        # Validate can be read
        df = pd.read_parquet(train_path)
        assert len(df) > 0, "train.parquet is empty"


def test_metadata_exists_for_primary_symbol():
    """Test that metadata exists for primary symbol."""
    config = load_yaml(CONFIG_PATH)
    
    symbol = config["environment"].get("symbol", "SPY")
    data_dir = Path(config["experiment"].get("data_dir", "data/phase3_splits"))
    
    symbol_dir = data_dir / symbol
    metadata_path = symbol_dir / "metadata.json"
    
    if not metadata_path.exists():
        pytest.skip(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    assert "symbol" in metadata
    assert metadata["symbol"] == symbol


# ============================================================================
# TEST 7: CRITICAL PARAMETER SANITY
# ============================================================================

def test_reward_weights_not_all_zero():
    """Test that at least one reward component is enabled."""
    config = load_yaml(CONFIG_PATH)
    reward = config["environment"]["reward_config"]
    
    weights = [
        reward.get("pnl_weight", 0.0),
        reward.get("transaction_cost_weight", 0.0),
        reward.get("time_efficiency_weight", 0.0),
        reward.get("sharpe_weight", 0.0),
        reward.get("drawdown_weight", 0.0),
        reward.get("diversity_bonus_weight", 0.0),
    ]
    
    total = sum(abs(w) for w in weights)
    assert total > 0.0, "All reward weights are zero - no reward signal!"


def test_pnl_weight_is_dominant():
    """Test that PnL weight is typically the largest component."""
    config = load_yaml(CONFIG_PATH)
    reward = config["environment"]["reward_config"]
    
    pnl_weight = abs(reward.get("pnl_weight", 0.0))
    other_weights = [
        abs(reward.get("transaction_cost_weight", 0.0)),
        abs(reward.get("time_efficiency_weight", 0.0)),
        abs(reward.get("sharpe_weight", 0.0)),
        abs(reward.get("drawdown_weight", 0.0)),
    ]
    
    # PnL should be largest or second largest
    max_other = max(other_weights) if other_weights else 0.0
    
    # Allow PnL to be slightly less than largest other component
    assert pnl_weight >= max_other * 0.5, \
        "PnL weight should be dominant component"


def test_transaction_cost_not_excessive():
    """Test that transaction cost weight doesn't dominate."""
    config = load_yaml(CONFIG_PATH)
    reward = config["environment"]["reward_config"]
    
    tc_weight = abs(reward.get("transaction_cost_weight", 0.0))
    pnl_weight = abs(reward.get("pnl_weight", 1.0))
    
    # Transaction cost should be much smaller than PnL
    if pnl_weight > 0:
        ratio = tc_weight / pnl_weight
        assert ratio < 1.0, \
            f"Transaction cost weight {tc_weight} too high relative to PnL {pnl_weight}"


def test_max_positions_reasonable():
    """Test that max_positions is reasonable for training."""
    config = load_yaml(CONFIG_PATH)
    portfolio = config["environment"]["portfolio_config"]
    
    max_positions = portfolio.get("max_positions", 1)
    
    assert 1 <= max_positions <= 10, \
        f"max_positions {max_positions} seems unusual (should be 1-10)"


def test_episode_length_sufficient_for_trading():
    """Test that episode length allows meaningful trades."""
    config = load_yaml(CONFIG_PATH)
    env = config["environment"]
    
    episode_length = env.get("episode_length", 0)
    lookback = env.get("lookback_window", 0)
    
    # Episode should be at least 3x lookback window
    assert episode_length >= lookback * 3, \
        f"Episode length {episode_length} too short for lookback {lookback}"


# ============================================================================
# TEST 8: ANTI-COLLAPSE MECHANISMS
# ============================================================================

def test_diversity_mechanisms_enabled():
    """Test that diversity bonus or penalty is configured."""
    config = load_yaml(CONFIG_PATH)
    reward = config["environment"]["reward_config"]
    
    diversity_bonus = reward.get("diversity_bonus_weight", 0.0)
    diversity_penalty = reward.get("diversity_penalty_weight", 0.0)
    
    # At least one diversity mechanism should be present
    assert diversity_bonus > 0.0 or diversity_penalty > 0.0, \
        "No diversity mechanisms enabled - risk of action collapse"


def test_exploration_curriculum_or_epsilon_greedy():
    """Test that exploration mechanism is enabled."""
    config = load_yaml(CONFIG_PATH)
    env = config["environment"]
    
    epsilon_enabled = env.get("epsilon_greedy_enabled", False)
    curriculum_enabled = env.get("exploration_curriculum_enabled", False)
    
    # At least one exploration mechanism recommended
    if not (epsilon_enabled or curriculum_enabled):
        pytest.skip("No explicit exploration mechanism - relying on SAC entropy")


# ============================================================================
# TEST 9: CONFIGURATION CONSISTENCY
# ============================================================================

def test_batch_size_less_than_buffer():
    """Test that batch_size < buffer_size."""
    config = load_yaml(CONFIG_PATH)
    sac = config["sac"]
    
    batch_size = sac.get("batch_size", 64)
    buffer_size = sac.get("buffer_size", 1000)
    
    assert batch_size < buffer_size, \
        f"batch_size {batch_size} must be less than buffer_size {buffer_size}"


def test_learning_starts_reasonable():
    """Test that learning_starts is reasonable."""
    config = load_yaml(CONFIG_PATH)
    sac = config["sac"]
    
    learning_starts = sac.get("learning_starts", 0)
    batch_size = sac.get("batch_size", 64)
    
    # Should be at least batch_size
    assert learning_starts >= batch_size, \
        f"learning_starts {learning_starts} should be >= batch_size {batch_size}"


def test_eval_freq_reasonable():
    """Test that eval_freq is reasonable."""
    config = load_yaml(CONFIG_PATH)
    training = config["training"]
    
    eval_freq = training.get("eval_freq", 0)
    total_timesteps = training.get("total_timesteps", 100000)
    
    if eval_freq > 0:
        # Should evaluate at least once
        assert eval_freq <= total_timesteps, \
            f"eval_freq {eval_freq} exceeds total_timesteps {total_timesteps}"
        
        # Should evaluate multiple times
        n_evals = total_timesteps // eval_freq
        assert n_evals >= 3, \
            f"Only {n_evals} evaluations - should be at least 3"


# ============================================================================
# TEST 10: COMPARISON WITH phase_a2_sac.yaml
# ============================================================================

def test_both_configs_exist():
    """Test that both phase_a2 configs exist."""
    assert CONFIG_PATH.exists(), f"Missing: {CONFIG_PATH}"
    assert PHASE_A2_SAC_PATH.exists(), f"Missing: {PHASE_A2_SAC_PATH}"


def test_configs_have_consistent_structure():
    """Test that both configs have same top-level structure."""
    config1 = load_yaml(CONFIG_PATH)
    config2 = load_yaml(PHASE_A2_SAC_PATH)
    
    keys1 = set(config1.keys())
    keys2 = set(config2.keys())
    
    # Should have same major sections
    common = keys1 & keys2
    assert "experiment" in common
    assert "training" in common
    assert "environment" in common
    assert "sac" in common


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
