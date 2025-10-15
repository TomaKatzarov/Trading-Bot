"""
Debug: Why are trades showing 0% profit?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
from core.rl.environments.continuous_trading_env import ContinuousTradingEnvironment
from training.rl.env_factory import build_trading_config

# Load config
config_path = Path("training/config_templates/phase_a2_sac_sharpe.yaml")
with open(config_path) as f:
    yaml_config = yaml.safe_load(f)

env_settings = yaml_config["environment"]
trading_config = build_trading_config(env_settings)
env = ContinuousTradingEnvironment(trading_config, seed=42)

print("="*80)
print("  TRADE EXECUTION DEBUG")
print("="*80)

obs, info = env.reset()
equity = env.portfolio.get_equity()
available_capital = env.portfolio.get_available_capital()
max_position_value = env.portfolio.get_max_position_value('SPY')

print(f"\nInitial equity: ${equity:,.2f}")
print(f"Initial cash: ${env.portfolio.cash:,.2f}")
print(f"Min position value (5%): ${equity * 0.05:,.2f}")
print(f"Max position value (PortfolioManager 10%): ${max_position_value:,.2f}")
print(f"Available capital: ${available_capital:,.2f}")

# Calculate expected trade value from continuous action
max_position_pct = 0.60  # From config
action_value = 0.8
expected_desired_value = action_value * max_position_pct * available_capital
print(f"\nExpected calculation for action {action_value}:")
print(f"  desired_value = {action_value} × {max_position_pct} × ${available_capital:,.2f}")
print(f"              = ${expected_desired_value:,.2f}")
print(f"  BUT: Max position from PortfolioManager = ${max_position_value:,.2f}")
print(f"  -> Will be clipped to ${min(expected_desired_value, max_position_value):,.2f}")

# Execute BUY
buy_action = np.array([0.8], dtype=np.float32)
print(f"\n--- Step 1: BUY Action ({buy_action[0]}) ---")
obs, reward, terminated, truncated, info = env.step(buy_action)

print(f"Action executed: {info.get('action_executed', False)}")
print(f"Reject reason: {info.get('reject_reason', 'N/A')}")
print(f"Reward: {reward:.6f}")
print(f"Equity: ${env.portfolio.get_equity():,.2f}")
print(f"Cash: ${env.portfolio.cash:,.2f}")
print(f"Positions: {len(env.portfolio.positions)}")

if env.portfolio.positions:
    for symbol, pos in env.portfolio.positions.items():
        print(f"\n  Position {symbol}:")
        print(f"    Shares: {pos.shares:.4f}")
        print(f"    Entry price: ${pos.entry_price:.2f}")
        print(f"    Current value: ${pos.current_value:,.2f}")
        print(f"    Cost basis: ${pos.cost_basis:,.2f}")
        print(f"    Unrealized PnL: ${pos.unrealized_pnl:,.2f} ({pos.unrealized_pnl_pct*100:+.2f}%)")

# Hold for several steps
print(f"\n--- Steps 2-11: HOLD Actions ---")
for step in range(10):
    obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
    
    if env.portfolio.positions:
        pos = list(env.portfolio.positions.values())[0]
        current_price = env.data.iloc[env.current_step]["close"]
        print(f"  Step {step+2}: Price=${current_price:.2f}, Unrealized PnL={pos.unrealized_pnl_pct*100:+.2f}%")
    
    if terminated or truncated:
        print(f"  Episode ended at step {step+2}")
        break

# SELL
if not (terminated or truncated) and env.portfolio.positions:
    sell_action = np.array([-0.9], dtype=np.float32)
    print(f"\n--- Step 12: SELL Action ({sell_action[0]}) ---")
    
    # Get position details before sell
    pos = list(env.portfolio.positions.values())[0]
    entry_price = pos.entry_price
    shares = pos.shares
    current_price = env.data.iloc[env.current_step]["close"]
    
    print(f"  Before SELL:")
    print(f"    Entry price: ${entry_price:.2f}")
    print(f"    Current price: ${current_price:.2f}")
    print(f"    Price change: {((current_price - entry_price) / entry_price * 100):+.2f}%")
    print(f"    Shares: {shares:.4f}")
    print(f"    Position value: ${pos.current_value:,.2f}")
    
    obs, reward, terminated, truncated, info = env.step(sell_action)
    
    print(f"\n  After SELL:")
    print(f"    Action executed: {info.get('action_executed', False)}")
    print(f"    Trade info exists: {'trade' in info}")
    
    if 'trade' in info:
        trade = info['trade']
        print(f"\n  Trade Details:")
        print(f"    Entry price: ${trade.get('entry_price', 0):.2f}")
        print(f"    Exit price: ${trade.get('exit_price', 0):.2f}")
        print(f"    Shares: {trade.get('shares', 0):.4f}")
        print(f"    Cost basis: ${trade.get('cost_basis', 0):,.2f}")
        print(f"    Proceeds: ${trade.get('proceeds', 0):,.2f}")
        print(f"    Realized PnL: ${trade.get('realized_pnl', 0):,.2f}")
        print(f"    Realized PnL %: {trade.get('realized_pnl_pct', 0)*100:+.4f}%")
        print(f"    Commission: ${trade.get('commission', 0):.2f}")
        print(f"    Slippage: ${trade.get('slippage', 0):.2f}")
    
    print(f"\n  Reward Breakdown:")
    print(f"    Total reward: {reward:.6f}")
    
    components = env._last_reward_components
    for key, value in sorted(components.items()):
        if abs(value) > 1e-9:
            print(f"    {key:25s}: {value:>10.6f}")
    
    print(f"\n  Portfolio Final State:")
    print(f"    Equity: ${env.portfolio.get_equity():,.2f}")
    print(f"    Cash: ${env.portfolio.cash:,.2f}")
    print(f"    Positions: {len(env.portfolio.positions)}")

print("\n" + "="*80)
print("  ANALYSIS")
print("="*80)

closed_positions = env.portfolio.get_closed_positions()
print(f"\nTotal closed positions: {len(closed_positions)}")

if closed_positions:
    print(f"\nAll Closed Trades:")
    for i, trade in enumerate(closed_positions, 1):
        realized_pnl = trade.get('realized_pnl', 0)
        realized_pnl_pct = trade.get('realized_pnl_pct', 0)
        print(f"  Trade {i}: PnL=${realized_pnl:.2f} ({realized_pnl_pct*100:+.4f}%), Reason={trade.get('exit_reason', 'N/A')}")

print("\n" + "="*80)
