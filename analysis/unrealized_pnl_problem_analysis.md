# Unrealized PnL Problem: Root Cause of Action Collapse

**Date:** October 8, 2025  
**Author:** Analysis based on user insight

## 🎯 THE CORE INSIGHT

**User's hypothesis:** "Profit and PnL should only count when a position is closed, like in real life. You're counting future takings as profit. This is why the sell action is not used at all."

## 📊 CURRENT IMPLEMENTATION (THE PROBLEM)

### How Rewards Are Calculated Now:

```python
# In trading_env.py step():
prev_equity = self.portfolio.get_equity()  # Cash + unrealized PnL
# ... execute action ...
current_equity = self.portfolio.get_equity()  # Cash + unrealized PnL

reward = compute_reward(
    prev_equity=prev_equity,
    current_equity=current_equity,
    # ...
)

# In portfolio_manager.py:
def get_equity(self) -> float:
    """Return total equity (cash + market value of open positions)."""
    position_value = sum(position.current_value for position in self.positions.values())
    return self.cash + position_value  # ← INCLUDES UNREALIZED PnL!
```

### The Problem:

**Every step, the agent gets rewarded for UNREALIZED gains!**

Example timeline:
- **Step 1:** BUY_SMALL @ $100 → No reward (just transaction cost)
- **Step 2:** HOLD, price → $101 → **+$1 reward** (unrealized)
- **Step 3:** HOLD, price → $102 → **+$1 reward** (unrealized)
- **Step 4:** HOLD, price → $103 → **+$1 reward** (unrealized)
- **Step 5:** HOLD, price → $104 → **+$1 reward** (unrealized)
- ... continues for 495 steps ...
- **Step 500:** Price drops to $95 → **-$9 penalty** but already collected +$400 in rewards!

**Total rewards collected: ~+$391 even though the position lost money!**

## 🔍 WHY THIS CAUSES COLLAPSE

### The Perverse Incentive Structure:

1. **BUY_SMALL is rewarded immediately** as price fluctuates up
2. **HOLD continues to accumulate rewards** from unrealized gains
3. **SELL actions are PUNISHED** because:
   - Selling realizes the gain (locks it in)
   - But agent already collected all the rewards during HOLD
   - Selling triggers transaction costs
   - After sell, no more unrealized PnL rewards!

### The Optimal Policy (Under Current System):

```
BUY_SMALL → HOLD → HOLD → HOLD → ... → HOLD (forever)
   ↑         ↑       ↑       ↑             ↑
 -$0.1    +$0.5   +$0.3   +$0.2  ... never sell!
(cost)   (unrealized rewards keep flowing)
```

**The agent learns:** "Never sell! Selling stops the reward stream!"

## 📈 EVIDENCE FROM TRAINING RESULTS

**Action Distribution:**
- BUY_SMALL: 90.2%
- HOLD: 9.8%
- SELL_PARTIAL: 0.0% ← **NEVER USED!**
- SELL_ALL: 0.0% ← **NEVER USED!**

**Why no sells?** Because selling ends the unrealized PnL reward stream!

## ✅ THE FIX: Realized PnL Only

### Proposed Implementation:

```python
def compute_reward(self, ...):
    """Compute reward based on REALIZED PnL only."""
    
    # Option 1: Only reward on closed positions
    if trade_info is not None:  # Position was closed
        realized_pnl = trade_info['pnl']
        reward = realized_pnl * weight
    else:  # Position open or no position
        reward = 0  # No reward until realized!
    
    # Option 2: Small step reward + big closing bonus
    if trade_info is not None:  # Position closed
        realized_pnl = trade_info['pnl']
        reward = realized_pnl * large_weight  # e.g., 10x
    else:  # Position open
        equity_change = current_equity - prev_equity
        reward = equity_change * small_weight  # e.g., 0.1x
```

### Benefits:

1. **Sells become valuable!** They're the ONLY way to collect big rewards
2. **Matches real trading:** You don't have profit until you sell
3. **Encourages full trade cycles:** BUY → HOLD → SELL → repeat
4. **Prevents infinite HOLD:** No reward for just sitting on positions
5. **Natural action diversity:** Need to use SELL to get paid!

## 🧮 EXPECTED IMPACT ON ACTION DISTRIBUTION

### Before (Current System):
```
BUY_SMALL: 90.2%
HOLD:       9.8%
SELL:       0.0%  ← Never used!
```

### After (Realized PnL Only):
```
BUY_SMALL:  ~30-40%  ← Still important to enter trades
HOLD:       ~20-30%  ← Position management
SELL_PARTIAL: ~15-25%  ← NOW VALUABLE! Locks in gains
SELL_ALL:   ~15-25%  ← NOW VALUABLE! Completes cycles
BUY_MEDIUM: ~5-10%   ← More aggressive sizing
BUY_LARGE:  ~2-5%    ← Occasionally used
```

**Expected entropy:** 1.2-1.6 (vs current 0.0098)

## 📋 IMPLEMENTATION PLAN

### Phase 1: Add Realized PnL Tracking
- [x] Already exists! `trade_info` contains closed position PnL
- [x] Portfolio manager tracks realized vs unrealized

### Phase 2: Modify Reward Calculation
- [ ] Change `_compute_pnl_reward()` to:
  - Large weight on `trade_info['pnl']` (closed positions)
  - Small/zero weight on `equity_change` (unrealized)
- [ ] Add configuration parameters:
  - `realized_pnl_weight: float = 0.8`  # 80% of reward from closes
  - `unrealized_pnl_weight: float = 0.2`  # 20% from unrealized (optional)

### Phase 3: Test Impact
- [ ] Run 100k training with realized-only rewards
- [ ] Compare action distributions
- [ ] Measure Sharpe and returns
- [ ] Check SELL action usage

### Phase 4: Tune Curriculum (If Needed)
- [ ] With better action diversity, may not need curriculum at all!
- [ ] Or relax to 3% minimum per action
- [ ] Or disable entirely if natural diversity emerges

## 🎲 ALTERNATIVE CONFIGURATIONS

### Config A: Pure Realized (Most Aggressive)
```yaml
realized_pnl_weight: 1.0    # 100% from closed trades
unrealized_pnl_weight: 0.0  # 0% from unrealized
```
**Effect:** Forces agent to close positions to get any reward

### Config B: Mixed (Balanced) ⭐ RECOMMENDED
```yaml
realized_pnl_weight: 0.8    # 80% from closed trades
unrealized_pnl_weight: 0.2  # 20% from unrealized (holding incentive)
```
**Effect:** Strong incentive to close, but not punished for holding good positions

### Config C: Unrealized Penalty
```yaml
realized_pnl_weight: 1.0     # 100% from closed trades
unrealized_pnl_weight: -0.1  # Penalize for not closing!
```
**Effect:** Maximum pressure to realize gains

## 🔬 HYPOTHESIS TESTING

### Hypothesis 1: Sell Actions Will Increase
**Test:** Count SELL_PARTIAL + SELL_ALL usage before/after
**Expected:** 0% → 30-40%

### Hypothesis 2: Action Entropy Will Improve
**Test:** Measure action entropy
**Expected:** 0.0098 → 1.2+

### Hypothesis 3: Sharpe May Improve
**Test:** Compare Sharpe ratio
**Expected:** +0.361 → +0.5+ (more complete cycles)

### Hypothesis 4: Curriculum May Become Unnecessary
**Test:** Disable curriculum, check action distribution
**Expected:** Natural diversity without forced penalties

## 🚀 NEXT STEPS

1. **Implement realized PnL weighting** in `reward_shaper.py`
2. **Add configuration parameters** to `phase3_ppo_baseline.yaml`
3. **Run comparative training:**
   - Baseline (current unrealized system)
   - Realized-only (100%)
   - Mixed (80/20)
4. **Analyze results:**
   - Action distributions
   - Sharpe ratios
   - Sell action usage
   - Episode completion rates
5. **Decide on curriculum:**
   - May not be needed if realized PnL creates natural diversity
   - Or can relax to much lower requirements (3% vs 8%)

## 💡 CONCLUSION

**The user is ABSOLUTELY CORRECT!**

The current system rewards unrealized PnL, which creates a perverse incentive to:
- Buy once
- Hold forever
- Never sell (selling stops the reward stream)

This explains:
- ✅ Why BUY_SMALL dominates (90.2%)
- ✅ Why HOLD is the only other action (9.8%)
- ✅ Why SELL actions are NEVER used (0.0%)
- ✅ Why curriculum penalties didn't help (fighting the wrong battle)

**The fix is simple:** Only reward REALIZED PnL (when positions close).

This will:
- Make SELL actions valuable (they unlock rewards!)
- Create natural buy-sell cycles
- Improve action diversity without forced penalties
- Match real trading behavior

**Expected outcome:** Natural action diversity, curriculum possibly unnecessary!
