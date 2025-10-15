"""
Minimal PnL test: Create environment, execute trades manually, check PnL rewards
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
from core.rl.environments.trading_env import TradingEnvironment, TradeAction
from training.rl.env_factory import build_trading_config

print("="*80)
print(" MINIMAL PNL TEST: Discrete Actions")
print("="*80)

# Load config
config_path = Path("training/config_templates/phase_a2_sac_sharpe.yaml")
with open(config_path) as f:
    yaml_config = yaml.safe_load(f)

env_settings = yaml_config["environment"]
env_settings["action_mode"] = "discrete"  # Use DISCRETE for simplicity
env_settings["epsilon_greedy_enabled"] = False  # No random exploration
trading_config = build_trading_config(env_settings)

env = TradingEnvironment(trading_config, seed=42)
obs, info = env.reset(seed=100)

print(f"\nStarting equity: ${env.portfolio.get_equity():,.2f}")
print(f"Starting step: {env.current_step}")

# Attempt 1: BUY_MEDIUM
print(f"\n--- Attempting BUY_MEDIUM ---")
obs, reward, terminated, truncated, info = env.step(TradeAction.BUY_MEDIUM.value)
print(f"  Action executed: {info.get('action_executed', False)}")
print(f"  Positions: {len(env.portfolio.positions)}")

if len(env.portfolio.positions) > 0:
    pos = list(env.portfolio.positions.values())[0]
    print(f"  Position: {pos.shares:.4f} shares @ ${pos.entry_price:.2f}")
    
    # HOLD for 10 steps
    print(f"\n--- HOLDing for 10 steps ---")
    for i in range(10):
        obs, reward, terminated, truncated, info = env.step(TradeAction.HOLD.value)
        if i % 3 == 0:
            current_price = env.data.iloc[env.current_step-1]["close"]
            unrealized = (current_price - pos.entry_price) / pos.entry_price
            print(f"  Step {i+1}: Price=${current_price:.2f}, Unrealized={unrealized*100:+.2f}%")
    
    # SELL_ALL
    print(f"\n--- Attempting SELL_ALL ---")
    obs, reward, terminated, truncated, info = env.step(TradeAction.SELL_ALL.value)
    print(f"  Action executed: {info.get('action_executed', False)}")
    print(f"  Positions after: {len(env.portfolio.positions)}")
    print(f"  Trade in info: {'trade' in info}")
    print(f"  Position closed in info: {info.get('position_closed') is not None}")
    
    components = env._last_reward_components
    pnl_component = components.get('pnl', 0)
    
    print(f"\n  Reward Components:")
    for key, val in sorted(components.items()):
        if abs(val) > 1e-9:
            print(f"    {key:25s}: {val:>10.6f}")
    
    print(f"\n  Total reward: {reward:.6f}")
    print(f"  PnL component: {pnl_component:.6f}")
    
    if abs(pnl_component) > 0.01:
        print(f"\n{'='*80}")
        print(f"  [SUCCESS] PnL reward = {pnl_component:+.6f}")
        print(f"{'='*80}")
        
        # Show trade details
        closed = env.portfolio.get_closed_positions()
        if closed:
            last_trade = closed[-1]
            print(f"\n  Trade details:")
            print(f"    PnL $: ${last_trade.get('realized_pnl', 0):.2f}")
            print(f"    PnL %: {last_trade.get('realized_pnl_pct', 0)*100:+.4f}%")
            print(f"    Exit reason: {last_trade.get('exit_reason', 'N/A')}")
        print("\n[CONFIRMED] PnL REWARD SYSTEM WORKS!")
    else:
        print(f"\n[FAIL] PnL component is ZERO")
        closed = env.portfolio.get_closed_positions()
        if closed:
            last_trade = closed[-1]
            print(f"  Last trade: PnL%={last_trade.get('realized_pnl_pct', 0)*100:+.4f}%, exit={last_trade.get('exit_reason')}")
else:
    print(f"  [FAIL] BUY was rejected: {info.get('reject_reason', 'unknown')}")

print("\n" + "="*80)
