"""
Debug reward calculation directly
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from core.rl.environments.reward_shaper import RewardShaper, RewardConfig
from training.rl.env_factory import build_reward_config

config_path = Path("training/config_templates/phase_a2_sac_sharpe.yaml")
with open(config_path) as f:
    yaml_config = yaml.safe_load(f)

reward_settings = yaml_config["environment"]["reward_config"]
reward_config = build_reward_config(reward_settings)
shaper = RewardShaper(reward_config)

print("="*80)
print(" REWARD CALCULATION DEBUG")
print("="*80)
print(f"\nConfig:")
print(f"  pnl_weight: {reward_config.pnl_weight}")
print(f"  pnl_scale: {reward_config.pnl_scale}")
print(f"  roi_neutral_zone: {reward_config.roi_neutral_zone}")
print(f"  roi_positive_scale: {reward_config.roi_positive_scale}")
print(f"  roi_negative_scale: {reward_config.roi_negative_scale}")
print(f"  win_bonus_multiplier: {reward_config.win_bonus_multiplier}")
print(f"  loss_penalty_multiplier: {reward_config.loss_penalty_multiplier}")

# Test with actual values from Test 5
test_cases = [
    {"pnl_pct": -0.0086, "desc": "Loss 0.86%"},
    {"pnl_pct": -0.0036, "desc": "Loss 0.36%"},
    {"pnl_pct": +0.0050, "desc": "Profit 0.50%"},
    {"pnl_pct": +0.0100, "desc": "Profit 1.00%"},
]

for case in test_cases:
    pnl_pct = case["pnl_pct"]
    desc = case["desc"]
    
    trade_info = {
        "pnl_pct": pnl_pct,
        "holding_hours": 10.0,
        "action": "SELL_ALL",
        "forced_exit": False,
        "exit_type": "full",
        "entry_size": "medium",
        "pyramid_count": 0,
    }
    
    pnl_reward = shaper._compute_pnl_reward(
        prev_equity=100000,
        current_equity=100000 * (1 + pnl_pct),
        trade_info=trade_info,
        position_info=None,
        action=5,
        portfolio_state={"num_trades": 5.0},
    )
    
    print(f"\n{desc}:")
    print(f"  Input pnl_pct: {pnl_pct*100:+.2f}%")
    print(f"  PnL reward: {pnl_reward:.6f}")
    
    # Manual calculation
    adj_roi = shaper._apply_roi_shaping(pnl_pct, 5.0)
    print(f"  After ROI shaping: {adj_roi:.6f}")
    normalized = adj_roi / reward_config.pnl_scale
    print(f"  After normalization (/{reward_config.pnl_scale}): {normalized:.6f}")
    
print("\n" + "="*80)
