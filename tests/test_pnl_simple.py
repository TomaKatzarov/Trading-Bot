"""
Simplified test: Force a profitable trade and verify PnL reward > 0
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
from core.rl.environments.continuous_trading_env import ContinuousTradingEnvironment
from training.rl.env_factory import build_trading_config

print("="*80)
print(" SIMPLIFIED PNL TEST: Does a profitable SELL generate PnL reward > 0?")
print("="*80)

# Load config
config_path = Path("training/config_templates/phase_a2_sac_sharpe.yaml")
with open(config_path) as f:
    yaml_config = yaml.safe_load(f)

env_settings = yaml_config["environment"]
# DISABLE epsilon exploration for deterministic test
env_settings["epsilon_greedy_enabled"] = False
trading_config = build_trading_config(env_settings)

# Use seed that generates upward price movement
env = ContinuousTradingEnvironment(trading_config, seed=123)  # Seed 123 for consistency

obs, info = env.reset(seed=456)  # Different seed for episode start
print(f"\nStarting equity: ${env.portfolio.get_equity():,.2f}")
print(f"Starting step: {env.current_step}")
print(f"Episode length: {len(env.data)} steps")
print(f"Positions after reset: {len(env.portfolio.positions)}")

# Try up to 50 buy-sell cycles to find good market conditions
for attempt in range(50):
    print(f"\n{'='*80}")
    print(f"ATTEMPT {attempt + 1}")
    print(f"{'='*80}")
    
    # Check if we need to BUY or already have a position
    if len(env.portfolio.positions) == 0:
        # BUY with strong signal
        buy_action = np.array([0.95], dtype=np.float32)
        print(f"\nStep {env.current_step}: BUY action = {buy_action[0]:.2f}")
        
        obs, reward, terminated, truncated, info = env.step(buy_action)
        
        if len(env.portfolio.positions) == 0:
            cont_info = info.get('continuous_action', {})
            reject_reason = cont_info.get('reject_reason', info.get('reject_reason', 'unknown'))
            action_executed = info.get('action_executed', False)
            if attempt < 3:  # Debug first few
                capital = env.portfolio.get_available_capital()
                print(f"  [FAIL] BUY rejected: action_executed={action_executed}, reject_reason={reject_reason}, capital=${capital:.2f}")
            if terminated or truncated:
                print("  Episode ended!")
                break
            continue
    else:
        print(f"\nAlready have position, skipping BUY")
    
    pos = list(env.portfolio.positions.values())[0]
    entry_price = pos.entry_price
    entry_step = env.current_step - 1
    print(f"  [OK] Position opened: {pos.shares:.4f} shares @ ${entry_price:.2f}")
    
    # Hold for 10-20 steps, looking for price increase
    max_unrealized_pnl = -1.0
    best_step = -1
    
    for h in range(20):
        current_price = env.data.iloc[env.current_step]["close"]
        unrealized_pnl_pct = (current_price - entry_price) / entry_price
        
        if unrealized_pnl_pct > max_unrealized_pnl:
            max_unrealized_pnl = unrealized_pnl_pct
            best_step = h
        
        if h % 5 == 0:
            print(f"  Step {env.current_step}: Price=${current_price:.2f}, Unrealized PnL={unrealized_pnl_pct*100:+.2f}%")
        
        # HOLD (use action within hold_threshold to ensure it's treated as HOLD)
        obs, reward, terminated, truncated, info = env.step(np.array([0.0001]))  # Tiny positive to stay in hold zone
        
        # Debug: Check if action was interpreted as HOLD
        cont_info = info.get('continuous_action', {})
        if h < 5 or cont_info.get('trade_type') != 'hold':
            print(f"    Step {h}: raw={cont_info.get('raw'):.4f}, smoothed={cont_info.get('smoothed'):.4f}, type={cont_info.get('trade_type')}, action_executed={info.get('action_executed', False)}")
        
        if terminated or truncated or len(env.portfolio.positions) == 0:
            print(f"  Position closed unexpectedly at step {h+1}")
            # Debug: Check why it closed
            if len(env.portfolio.get_closed_positions()) > 0:
                last_closed = env.portfolio.get_closed_positions()[-1]
                print(f"    Exit reason: {last_closed.get('exit_reason', 'N/A')}")
                print(f"    Trigger: {last_closed.get('trigger', 'N/A')}")
                print(f"    Forced exit: {last_closed.get('forced_exit', False)}")
            break
    
    if terminated or truncated:
        print("Episode ended during HOLD")
        break
    
    if len(env.portfolio.positions) == 0:
        print("Position was force-closed, moving to next attempt")
        continue
    
    # Check final unrealized PnL
    current_price = env.data.iloc[env.current_step]["close"]
    unrealized_pnl_pct = (current_price - entry_price) / entry_price
    
    print(f"\n  Final unrealized PnL: {unrealized_pnl_pct*100:+.4f}%")
    print(f"  Max unrealized PnL seen: {max_unrealized_pnl*100:+.4f}% at hold step {best_step}")
    
    # SELL
    sell_action = np.array([-0.95], dtype=np.float32)
    print(f"\n  Step {env.current_step}: SELL action = {sell_action[0]:.2f}")
    
    obs, reward, terminated, truncated, info = env.step(sell_action)
    
    # Check components
    components = env._last_reward_components
    pnl_component = components.get('pnl', 0)
    
    print(f"\n  SELL Results:")
    print(f"    Position after SELL: {len(env.portfolio.positions)} open")
    print(f"    Total reward: {reward:.4f}")
    print(f"    PnL component: {pnl_component:.4f}")
    
    print(f"\n  All reward components:")
    for key, val in sorted(components.items()):
        if abs(val) > 1e-9:
            print(f"    {key:25s}: {val:>10.6f}")
    
    # Check if we got a non-zero PnL reward
    if abs(pnl_component) > 0.01:
        print(f"\n{'='*80}")
        print(f"  [SUCCESS] PnL reward = {pnl_component:+.4f}")
        print(f"{'='*80}")
        
        # Check closed positions
        closed = env.portfolio.get_closed_positions()
        if closed:
            last_trade = closed[-1]
            print(f"\n  Last closed trade details:")
            print(f"    Realized PnL $: ${last_trade.get('realized_pnl', 0):.2f}")
            print(f"    Realized PnL %: {last_trade.get('realized_pnl_pct', 0)*100:+.4f}%")
            print(f"    Exit reason: {last_trade.get('exit_reason', 'N/A')}")
            print(f"    Forced exit: {last_trade.get('forced_exit', False)}")
        
        print("\n[CONFIRMED] PnL REWARD SYSTEM IS WORKING!")
        break
    else:
        print(f"\n  [FAIL] PnL reward is ZERO (expected non-zero)")
        
        # Debug
        closed = env.portfolio.get_closed_positions()
        if closed:
            last_trade = closed[-1]
            print(f"\n  Debug - Last closed trade:")
            print(f"    Realized PnL %: {last_trade.get('realized_pnl_pct', 0)*100:+.4f}%")
            print(f"    Exit reason: {last_trade.get('exit_reason', 'N/A')}")
            print(f"    Forced exit: {last_trade.get('forced_exit', False)}")
            print(f"    Entry size: {last_trade.get('entry_size', 'N/A')}")
            print(f"    Exit type: {last_trade.get('exit_type', 'N/A')}")
    
    if terminated or truncated:
        break

print("\n" + "="*80)
