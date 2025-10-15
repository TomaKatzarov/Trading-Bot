"""
Debug script: Test if PnL rewards are being calculated correctly with current config.
"""
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rl.environments.reward_shaper import RewardConfig, RewardShaper

# Load config
config_path = Path("training/config_templates/phase_a2_sac_sharpe.yaml")
with open(config_path) as f:
    yaml_config = yaml.safe_load(f)

# Build RewardConfig
reward_settings = yaml_config["environment"]["reward_config"]
reward_config = RewardConfig()

# Apply settings
for key, value in reward_settings.items():
    if hasattr(reward_config, key):
        setattr(reward_config, key, value)

print("="*80)
print("  REWARD CONFIG LOADED")
print("="*80)
print(f"pnl_weight: {reward_config.pnl_weight}")
print(f"pnl_scale: {reward_config.pnl_scale}")
print(f"win_bonus_multiplier: {reward_config.win_bonus_multiplier}")
print(f"loss_penalty_multiplier: {reward_config.loss_penalty_multiplier}")
print(f"realized_pnl_weight: {reward_config.realized_pnl_weight}")
print(f"roi_multiplier_enabled: {reward_config.roi_multiplier_enabled}")
print(f"roi_neutral_zone: {reward_config.roi_neutral_zone}")
print(f"roi_positive_scale: {reward_config.roi_positive_scale}")
print()

# Create RewardShaper
shaper = RewardShaper(reward_config)

print("="*80)
print("  SIMULATED TRADE: 1% PROFIT")
print("="*80)

# Simulate a profitable trade
trade_info = {
    "pnl_pct": 0.01,  # 1% profit
    "holding_hours": 2.0,
    "action": "SELL_ALL",
    "forced_exit": False,
    "exit_type": "full",
    "entry_size": "medium",
    "pyramid_count": 0,
}

portfolio_state = {
    "num_trades": 10.0,
    "sharpe_ratio": -0.5,
    "peak_equity": 100000.0,
}

position_info = None  # Position closed

prev_equity = 100000.0
current_equity = 101000.0  # 1% gain

# Call compute_reward
total_reward, components = shaper.compute_reward(
    action=5,  # SELL_ALL
    action_executed=True,
    prev_equity=prev_equity,
    current_equity=current_equity,
    position_info=position_info,
    trade_info=trade_info,
    portfolio_state=portfolio_state,
    diversity_info={"action_diversity_window": [1, 2, 3, 5], "episode_step": 100, "repeat_streak": 1},
)

print(f"\nTRADE INPUT:")
print(f"  realized_pnl_pct: {trade_info['pnl_pct']*100:.2f}%")
print(f"  prev_equity: ${prev_equity:,.0f}")
print(f"  current_equity: ${current_equity:,.0f}")
print(f"  entry_size: {trade_info['entry_size']}")
print(f"  exit_type: {trade_info['exit_type']}")

print(f"\nREWARD COMPONENTS (raw, before weighting):")
for key, value in sorted(components.items()):
    if value != 0:
        print(f"  {key:25s}: {value:>10.4f}")

print(f"\nWEIGHTED CONTRIBUTIONS:")
print(f"  pnl ({reward_config.pnl_weight:.2f}× raw):           {reward_config.pnl_weight * components.get('pnl', 0):>10.2f}")
print(f"  transaction_cost ({reward_config.transaction_cost_weight:.3f}× raw): {reward_config.transaction_cost_weight * components.get('transaction_cost', 0):>10.2f}")
print(f"  diversity ({reward_config.diversity_bonus_weight:.2f}× raw):       {reward_config.diversity_bonus_weight * components.get('diversity_bonus', 0):>10.2f}")

print(f"\nTOTAL REWARD: {total_reward:.2f}")

if components.get("pnl", 0) == 0:
    print("\n" + "="*80)
    print("  ⚠️  WARNING: PnL COMPONENT IS ZERO!")
    print("="*80)
    print("  This indicates the bug is still present!")
    print("  Check:")
    print("    1. roi_neutral_zone value")
    print("    2. _apply_roi_shaping() logic")
    print("    3. realized_pnl_weight value")
else:
    print("\n" + "="*80)
    print("  ✅ SUCCESS: PnL COMPONENT IS NON-ZERO!")
    print("="*80)
    print(f"  1% profit generated {components['pnl']:.2f} raw reward")
    print(f"  After weighting ({reward_config.pnl_weight}×): {reward_config.pnl_weight * components['pnl']:.2f}")
