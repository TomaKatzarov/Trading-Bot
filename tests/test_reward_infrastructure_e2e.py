"""Validation tests for the reward configuration and shaper behaviour.

These checks ensure that the Phase A2 SAC Sharpe template keeps the reward
settings aligned with the fixes discussed in the training review:

- RewardConfig mirrors the YAML values (weights, scales, gates).
- RewardShaper components react correctly to common edge cases.
- Diversity guardrails (bonus + penalty) are wired into aggregation.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

from core.rl.environments.reward_shaper import RewardConfig, RewardShaper
from training.rl.env_factory import build_reward_config, load_yaml

_CONFIG_PATH = Path(__file__).parent.parent / "training" / "config_templates" / "phase_a2_sac_sharpe.yaml"


def _load_reward_config() -> RewardConfig:
    config_dict = load_yaml(_CONFIG_PATH)
    environment_cfg = config_dict["environment"]
    reward_cfg = environment_cfg.get("reward_config")
    assert reward_cfg is not None, "reward_config section missing from template"
    return build_reward_config(reward_cfg)  # type: ignore[return-value]


def _build_shaper() -> RewardShaper:
    return RewardShaper(_load_reward_config())


def test_reward_config_matches_template_values() -> None:
    """Verify reward config matches phase_a2_sac_sharpe.yaml values exactly."""
    cfg = _load_reward_config()

    # Match actual YAML config values
    assert math.isclose(cfg.pnl_weight, 0.95, rel_tol=0.0, abs_tol=1e-12), f"Expected pnl_weight=0.95, got {cfg.pnl_weight}"
    assert math.isclose(cfg.transaction_cost_weight, 0.001, rel_tol=0.0, abs_tol=1e-12), f"Expected transaction_cost_weight=0.001, got {cfg.transaction_cost_weight}"
    assert math.isclose(cfg.drawdown_weight, 0.0, rel_tol=0.0, abs_tol=1e-12), f"Expected drawdown_weight=0.0, got {cfg.drawdown_weight}"
    assert math.isclose(cfg.diversity_bonus_weight, 0.15, rel_tol=0.0, abs_tol=1e-12), f"Expected diversity_bonus_weight=0.15, got {cfg.diversity_bonus_weight}"
    assert math.isclose(cfg.diversity_penalty_weight, 0.0, rel_tol=0.0, abs_tol=1e-12), f"Expected diversity_penalty_weight=0.0, got {cfg.diversity_penalty_weight}"
    assert math.isclose(cfg.hold_penalty_weight, 0.040, rel_tol=0.0, abs_tol=1e-12), f"Expected hold_penalty_weight=0.040, got {cfg.hold_penalty_weight}"
    assert math.isclose(cfg.pnl_scale, 0.0001, rel_tol=0.0, abs_tol=1e-12), f"Expected pnl_scale=0.0001, got {cfg.pnl_scale}"
    assert math.isclose(cfg.reward_clip, 1250.0, rel_tol=0.0, abs_tol=1e-12), f"Expected reward_clip=1250.0, got {cfg.reward_clip}"
    assert not cfg.roi_multiplier_enabled, "ROI multiplier should be disabled"
    assert not cfg.sharpe_gate_enabled, "Sharpe gate should be disabled"
    assert cfg.sharpe_gate_min_self_trades == 240, f"Expected sharpe_gate_min_self_trades=240, got {cfg.sharpe_gate_min_self_trades}"
    assert math.isclose(cfg.roi_negative_scale, 1.0, rel_tol=0.0, abs_tol=1e-12), f"Expected roi_negative_scale=1.0, got {cfg.roi_negative_scale}"


def test_shaper_initial_gate_closed() -> None:
    shaper = _build_shaper()
    assert hasattr(shaper, "_sharpe_gate_open")
    assert shaper._sharpe_gate_open is False  # Sharpe gate should start closed


def test_reward_aggregation_uses_component_weights() -> None:
    """Verify reward aggregation applies component weights correctly."""
    shaper = _build_shaper()

    components: Dict[str, float] = {
        "pnl": 12.5,
        "transaction_cost": -0.3,
        "time_efficiency": 0.0,
        "sharpe": 0.0,
        "drawdown": -2.0,
        "sizing": 0.0,
        "hold_penalty": -0.1,
        "diversity_bonus": 0.4,
        "diversity_penalty": 0.0,  # disabled in config
        "action_repeat_penalty": 0.0,
        "intrinsic_action": 0.0,
        "equity_delta": 0.0,
    }

    total = shaper._aggregate_components(components)

    cfg = shaper.config
    expected = (
        cfg.pnl_weight * components["pnl"]
        + cfg.transaction_cost_weight * components["transaction_cost"]
        + cfg.time_efficiency_weight * components["time_efficiency"]
        + cfg.sharpe_weight * components["sharpe"]
        + cfg.drawdown_weight * components["drawdown"]
        + cfg.sizing_weight * components["sizing"]
        + cfg.hold_penalty_weight * components["hold_penalty"]
        + cfg.diversity_bonus_weight * components["diversity_bonus"]
        + cfg.diversity_penalty_weight * components["diversity_penalty"]
        + cfg.action_repeat_penalty_weight * components["action_repeat_penalty"]
        + cfg.intrinsic_action_reward * components["intrinsic_action"]
        + cfg.equity_delta_weight * components["equity_delta"]
    )

    assert math.isclose(total, expected, rel_tol=0.0, abs_tol=1e-9), f"Expected {expected}, got {total}"


def test_negative_roi_penalty_respects_configured_multipliers() -> None:
    shaper = _build_shaper()
    cfg = shaper.config

    trade_info = {
        "pnl_pct": -0.01,
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }

    penalty = shaper._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=99000.0,
        trade_info=trade_info,
        position_info=None,
        action=5,
    portfolio_state={"num_trades": 0},
    )

    expected_roi = shaper._apply_roi_shaping(trade_info["pnl_pct"], 0.0)
    expected = expected_roi / cfg.pnl_scale
    expected *= cfg.loss_penalty_multiplier
    expected *= cfg.position_size_medium_multiplier
    expected *= cfg.full_exit_multiplier
    expected *= cfg.closing_bonus_multiplier
    assert math.isclose(penalty, expected, rel_tol=0.0, abs_tol=1e-9)


def test_positive_pnl_reward_clipped_at_limit() -> None:
    shaper = _build_shaper()
    cfg = shaper.config

    trade_info = {
        "pnl_pct": 0.25,  # 25% profit should exceed the reward clip once amplified
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }

    reward = shaper._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=125000.0,
        trade_info=trade_info,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 50},
    )

    assert math.isclose(reward, cfg.reward_clip, rel_tol=0.0, abs_tol=1e-6)


def test_diversity_penalty_triggers_on_action_collapse() -> None:
    """Test that diversity penalty mechanism works (even if disabled in config)."""
    shaper = _build_shaper()

    diversity_info = {
        "action_diversity_window": [3] * 9 + [2],  # 9/10 same action → collapse
        "repeat_streak": 9,
    }

    penalty = shaper._compute_diversity_penalty(diversity_info)
    assert penalty < 0.0, "Diversity penalty should be negative for action collapse"

    # Note: diversity_penalty_weight is 0.0 in config, so total should be 0
    components = {
        "pnl": 0.0,
        "transaction_cost": 0.0,
        "time_efficiency": 0.0,
        "sharpe": 0.0,
        "drawdown": 0.0,
        "sizing": 0.0,
        "hold_penalty": 0.0,
        "diversity_bonus": 0.0,
        "diversity_penalty": penalty,
        "action_repeat_penalty": 0.0,
        "intrinsic_action": 0.0,
        "equity_delta": 0.0,
    }

    total = shaper._aggregate_components(components)
    expected = shaper.config.diversity_penalty_weight * penalty  # 0.0 * penalty = 0.0
    assert math.isclose(total, expected, rel_tol=0.0, abs_tol=1e-9), f"Expected {expected}, got {total}"


def test_small_positive_roi_generates_reward() -> None:
    shaper = _build_shaper()

    trade_info = {
        "pnl_pct": 0.001,  # 0.1% profit must remain positive after scaling
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }

    reward = shaper._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=100100.0,
        trade_info=trade_info,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 15},
    )

    assert reward > 0.0


def test_roi_negative_scale_reduces_loss_magnitude() -> None:
    """Test ROI negative scaling - but note config has roi_multiplier_enabled=False.
    
    The current phase_a2_sac_sharpe.yaml config has ROI multiplier disabled,
    so both scaled and baseline will produce the same results. This test
    documents the expected behavior when ROI scaling IS enabled.
    """
    cfg_scaled = _load_reward_config()
    cfg_baseline = _load_reward_config()
    
    # Enable ROI multiplier for this test
    cfg_scaled.roi_multiplier_enabled = True
    cfg_scaled.roi_negative_scale = 0.7  # From config
    cfg_baseline.roi_multiplier_enabled = True
    cfg_baseline.roi_negative_scale = 1.0  # Baseline: no scaling
    
    shaper_scaled = RewardShaper(cfg_scaled)
    shaper_baseline = RewardShaper(cfg_baseline)

    trade_info = {
        "pnl_pct": -0.01,  # Smaller loss to test scaling effect
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }

    penalty_scaled = shaper_scaled._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=99000.0,
        trade_info=trade_info,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 80},
    )

    penalty_baseline = shaper_baseline._compute_pnl_reward(
        prev_equity=100000.0,
        current_equity=99000.0,
        trade_info=trade_info,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 80},
    )

    # When ROI scaling is enabled, scaled should have smaller penalty magnitude
    assert abs(penalty_scaled) < abs(penalty_baseline), \
        f"ROI negative scaling should reduce loss magnitude: scaled={penalty_scaled}, baseline={penalty_baseline}"


def test_reward_clip_bounds_total_reward() -> None:
    """Verify reward clip bounds aggregate rewards."""
    shaper = _build_shaper()

    # Use exaggerated component signals to trigger clipping inside aggregation
    components = {
        "pnl": 5000.0,
        "transaction_cost": -0.5,
        "time_efficiency": 0.0,
        "sharpe": 0.0,
        "drawdown": 0.0,
        "sizing": 0.0,
        "hold_penalty": 0.0,
        "diversity_bonus": 0.0,
        "diversity_penalty": 0.0,
        "action_repeat_penalty": 0.0,
        "intrinsic_action": 0.0,
        "equity_delta": 0.0,
    }

    total = shaper._aggregate_components(components)
    # Should be clipped to reward_clip (1250.0)
    assert math.isclose(total, shaper.config.reward_clip, rel_tol=0.0, abs_tol=1e-6), f"Expected {shaper.config.reward_clip}, got {total}"


if __name__ == "__main__":
    # Allow quick local execution with `python tests/test_reward_infrastructure_e2e.py`.
    import pytest

    raise SystemExit(pytest.main([__file__]))
