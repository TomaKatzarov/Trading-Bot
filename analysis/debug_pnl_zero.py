"""
Deep dive: Why is PnL component still 0 even after trades execute?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
from core.rl.environments.continuous_trading_env import ContinuousTradingEnvironment
from training.rl.env_factory import build_trading_config

config_path = Path("training/config_templates/phase_a2_sac_sharpe.yaml")
with open(config_path) as f:
    yaml_config = yaml.safe_load(f)

env_settings = yaml_config["environment"]
trading_config = build_trading_config(env_settings)
env = ContinuousTradingEnvironment(trading_config, seed=123)

print("="*80)
print("  DEEP DIVE: PnL REWARD TRACKING")
print("="*80)

obs, info = env.reset()
print(f"\nStarting equity: ${env.portfolio.get_equity():,.2f}")

# Execute multiple buy-sell cycles and track PnL components
trade_count = 0
pnl_rewards_seen = []

for cycle in range(10):
    print(f"\n{'='*80}")
    print(f"  CYCLE {cycle + 1}")
    print(f"{'='*80}")
    
    # BUY with strong signal
    buy_action = np.array([0.9], dtype=np.float32)
    print(f"\n--- BUY Action ---")
    obs, reward, terminated, truncated, info = env.step(buy_action)
    
    if terminated or truncated:
        print("Episode ended during BUY")
        break
    
    buy_components = env._last_reward_components
    print(f"  Action executed: {info.get('action_executed', False)}")
    print(f"  Positions after BUY: {len(env.portfolio.positions)}")
    print(f"  PnL component: {buy_components.get('pnl', 0):.6f}")
    print(f"  Total reward: {reward:.6f}")
    
    if len(env.portfolio.positions) == 0:
        print("  ⚠️  No position opened, skipping cycle")
        continue
    
    # Get position details
    pos = list(env.portfolio.positions.values())[0]
    entry_price = pos.entry_price
    entry_step = env.current_step
    print(f"  Position opened: {pos.shares:.4f} shares @ ${entry_price:.2f}")
    
    # Hold for 5-10 steps
    hold_steps = np.random.randint(5, 11)
    print(f"\n--- HOLD for {hold_steps} steps ---")
    
    for h in range(hold_steps):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
        if terminated or truncated:
            print(f"  Episode ended at hold step {h+1}")
            break
    
    if terminated or truncated:
        break
    
    # Check unrealized PnL before sell
    if len(env.portfolio.positions) > 0:
        pos = list(env.portfolio.positions.values())[0]
        current_price = env.data.iloc[env.current_step]["close"]
        unrealized_pnl_pct = (current_price - entry_price) / entry_price
        print(f"  Current price: ${current_price:.2f}")
        print(f"  Unrealized PnL: {unrealized_pnl_pct*100:+.4f}%")
    else:
        print("  ⚠️  Position was closed during HOLD (forced exit?)")
        continue
    
    # SELL
    sell_action = np.array([-0.9], dtype=np.float32)
    print(f"\n--- SELL Action ---")
    obs, reward, terminated, truncated, info = env.step(sell_action)
    
    sell_components = env._last_reward_components
    
    print(f"  Action executed: {info.get('action_executed', False)}")
    print(f"  Trade in info: {'trade' in info}")
    print(f"  Positions after SELL: {len(env.portfolio.positions)}")
    
    # Check _last_closed_trade
    if hasattr(env, '_last_closed_trade'):
        print(f"  _last_closed_trade: {env._last_closed_trade is not None}")
        if env._last_closed_trade:
            print(f"    PnL %: {env._last_closed_trade.get('realized_pnl_pct', 0)*100:+.4f}%")
    
    print(f"\n  SELL Reward Components:")
    for key in ['pnl', 'transaction_cost', 'sharpe', 'diversity_bonus']:
        val = sell_components.get(key, 0)
        if abs(val) > 1e-9:
            print(f"    {key:25s}: {val:>10.6f}")
    
    print(f"  Total reward: {reward:.6f}")
    
    pnl_component = sell_components.get('pnl', 0)
    if abs(pnl_component) > 1e-9:
        trade_count += 1
        pnl_rewards_seen.append(pnl_component)
        print(f"  ✅ PnL reward recorded: {pnl_component:+.6f}")
    else:
        print(f"  ❌ PnL reward is ZERO!")
        
        # Debug why
        print(f"\n  DEBUG: Why is PnL zero?")
        
        # Check closed positions
        closed = env.portfolio.get_closed_positions()
        if closed:
            last_trade = closed[-1]
            print(f"    Last closed position:")
            print(f"      Realized PnL: ${last_trade.get('realized_pnl', 0):.2f}")
            print(f"      Realized PnL %: {last_trade.get('realized_pnl_pct', 0)*100:+.4f}%")
            print(f"      Exit reason: {last_trade.get('exit_reason', 'N/A')}")
            print(f"      Forced exit: {last_trade.get('forced_exit', False)}")
        else:
            print(f"    No closed positions found!")
    
    if terminated or truncated:
        break

print(f"\n{'='*80}")
print(f"  FINAL SUMMARY")
print(f"{'='*80}")
print(f"Trades with non-zero PnL: {trade_count}")
print(f"Total PnL rewards seen: {len(pnl_rewards_seen)}")
if pnl_rewards_seen:
    print(f"PnL rewards: {pnl_rewards_seen}")
    print(f"Total: {sum(pnl_rewards_seen):+.6f}")
else:
    print("❌ NO PnL REWARDS GENERATED!")

print(f"\nTotal closed positions: {len(env.portfolio.get_closed_positions())}")
if env.portfolio.get_closed_positions():
    print("\nAll closed trades:")
    for i, trade in enumerate(env.portfolio.get_closed_positions()[-5:], 1):  # Last 5
        print(f"  {i}. PnL%: {trade.get('realized_pnl_pct', 0)*100:+.2f}%, Reason: {trade.get('exit_reason', 'N/A')}")

print("="*80)
