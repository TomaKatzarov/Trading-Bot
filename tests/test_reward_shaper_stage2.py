import pytest

from core.rl.environments.reward_shaper import RewardConfig, RewardShaper


def _make_shaper(**overrides) -> RewardShaper:
    base = RewardConfig(
        pnl_weight=0.0,
        transaction_cost_weight=0.0,
        time_efficiency_weight=0.0,
        sharpe_weight=1.0,
        drawdown_weight=0.0,
        sizing_weight=0.0,
        hold_penalty_weight=0.0,
        sharpe_gate_enabled=True,  # ADDED: Enable gate
        sharpe_gate_window=3,
        sharpe_gate_min_self_trades=2,
        sharpe_gate_floor_scale=0.25,
        sharpe_gate_active_scale=1.0,
        roi_scale_factor=1.0,
        roi_gate_floor_scale=0.25,
        forced_exit_penalty=0.0,
        reward_clip=5.0,
        time_decay_threshold_hours=18.0,
        time_decay_penalty_per_hour=0.003,
        time_decay_max_penalty=0.05,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return RewardShaper(base)


def test_sharpe_gate_requires_voluntary_closes():
    """Test that the Sharpe gate only opens on voluntary (not forced) closes."""
    shaper = _make_shaper()

    portfolio_state = {"num_trades": 10, "sharpe_ratio": 1.5}

    # Gate should start closed
    assert not shaper._sharpe_gate_open

    # Forced exits should NOT open the gate (even if many occur)
    for _ in range(10):
        shaper.compute_reward(
            action=5,
            action_executed=True,
            prev_equity=100.0,
            current_equity=99.0,
            position_info=None,
            trade_info={
                "pnl_pct": -0.01,
                "holding_hours": 5,
                "action": "position_loss_limit",
                "forced_exit": True,
                "exit_type": "full",
                "entry_size": "small",
                "pyramid_count": 0,
            },
            portfolio_state=portfolio_state,
            diversity_info=None,
        )
    
    # Gate should still be closed after forced exits
    assert not shaper._sharpe_gate_open

    # Two voluntary closes SHOULD open the gate (min_self_trades=2)
    for _ in range(2):
        shaper.compute_reward(
            action=5,
            action_executed=True,
            prev_equity=100.0,
            current_equity=101.0,
            position_info=None,
            trade_info={
                "pnl_pct": 0.01,
                "holding_hours": 6,
                "action": "agent_full_close",
                "forced_exit": False,
                "exit_type": "full",
                "entry_size": "small",
                "pyramid_count": 0,
            },
            portfolio_state=portfolio_state,
            diversity_info=None,
        )

    # Gate should now be open
    assert shaper._sharpe_gate_open


def test_forced_exit_penalty_scales_with_loss():
    shaper = _make_shaper(
        time_efficiency_weight=1.0,
        sharpe_weight=0.0,
        reward_clip=5.0,
        forced_exit_base_penalty=0.2,
        forced_exit_loss_scale=3.0,
        forced_exit_penalty_cap=1.5,
    )

    penalty = shaper._compute_time_reward(
        trade_info={
            "pnl_pct": -0.04,
            "holding_hours": 10,
            "action": "position_loss_limit",
            "forced_exit": True,
            "exit_type": "full",
            "entry_size": "medium",
            "pyramid_count": 0,
        },
        position_info=None,
    )

    expected = -(shaper.config.forced_exit_base_penalty + 0.04 * shaper.config.forced_exit_loss_scale)
    expected = max(-shaper.config.forced_exit_penalty_cap, expected)
    assert penalty == pytest.approx(expected)


def test_time_decay_penalty_applies_after_threshold():
    shaper = _make_shaper(
        sharpe_weight=0.0,
        time_efficiency_weight=1.0,
        reward_clip=5.0,
        time_decay_threshold_hours=12.0,
        time_decay_penalty_per_hour=0.004,
        time_decay_max_penalty=0.05,
    )

    penalty = shaper._compute_time_reward(
        trade_info=None,
        position_info={"is_open": True, "duration": 18},
    )

    expected = -min(
        shaper.config.time_decay_max_penalty,
        (18 - shaper.config.time_decay_threshold_hours) * shaper.config.time_decay_penalty_per_hour,
    )
    assert penalty == pytest.approx(expected)

    # Below the threshold there should be no penalty
    no_penalty = shaper._compute_time_reward(
        trade_info=None,
        position_info={"is_open": True, "duration": 8},
    )
    assert no_penalty == 0.0
