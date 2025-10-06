# Phase 3: Validation Protocol & Success Criteria

**Document Version:** 1.0  
**Created:** October 6, 2025  
**Phase:** 3 - Prototype Training & Validation  
**Status:** Implementation Ready

---

## Executive Summary

This document defines the comprehensive validation methodology for Phase 3 prototype agents. It establishes rigorous success criteria, comparison frameworks against the catastrophic SL baseline, and validation workflows to ensure agents are production-ready before scaling to Phase 4 (143 agents).

**Critical Context:**
- **SL Baseline:** -88% to -93% total return (catastrophic failure)
- **Best SL Config:** -10.9% return @ threshold 0.80 (MLP)
- **RL Target:** ≥+12% return, ≥0.50 Sharpe, ≤25% drawdown
- **Validation Data:** 2025 Q1-Q2 (out-of-sample from training)

---

## 1. Validation Methodology

### 1.1 Three-Tier Validation Framework

**Tier 1: Training Validation (During Training)**
- **Frequency:** Every 5,000 steps
- **Purpose:** Monitor convergence, prevent overfitting
- **Data:** Validation split (2025 Q1-Q2)
- **Duration:** 10 episodes per evaluation

**Tier 2: Hold-Out Validation (Post-Training)**
- **Frequency:** Once after training complete
- **Purpose:** Unbiased performance estimate
- **Data:** Test split (2025 Aug-Oct)
- **Duration:** 20 episodes

**Tier 3: Walk-Forward Validation (Final Check)**
- **Frequency:** Once before Phase 4 approval
- **Purpose:** Test robustness across time periods
- **Data:** Rolling windows across full 2-year period
- **Duration:** 5 walks × 20 episodes each

---

### 1.2 Validation Data Splits

**Training Period:**
- **Dates:** 2023-10-02 to 2024-12-31
- **Hours:** ~12,700
- **Percentage:** 70%
- **Purpose:** Policy learning

**Validation Period (Tier 1):**
- **Dates:** 2025-01-01 to 2025-07-31
- **Hours:** ~5,100
- **Percentage:** 15%
- **Purpose:** Hyperparameter selection, early stopping
- **Market Regime:** Mixed (bull in Q1, consolidation in Q2)

**Test Period (Tier 2):**
- **Dates:** 2025-08-01 to 2025-10-01
- **Hours:** ~1,500
- **Percentage:** 15%
- **Purpose:** Final unbiased evaluation
- **Market Regime:** Recent data (most realistic for deployment)

---

### 1.3 Episode Configuration

**Episode Parameters:**
```python
episode_config = {
    "max_steps": 1000,           # ~1000 hours = 41 days
    "initial_capital": 100_000,  # Match SL baseline
    "commission": 0.001,         # 0.10% per trade
    "slippage_bps": 5.0,         # 5 basis points
    "max_position_pct": 0.10,    # 10% max position size
    "max_exposure_pct": 1.0,     # 100% max exposure
}
```

**Evaluation Settings:**
```python
eval_config = {
    "deterministic": True,        # No exploration during eval
    "num_episodes": 10,           # Default for during-training eval
    "record_video": False,        # Enable for debugging
    "save_trajectories": True,    # For analysis
}
```

---

## 2. Performance Metrics

### 2.1 Primary Metrics (Core Success Criteria)

**1. Sharpe Ratio**

```python
def compute_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.03) -> float:
    """Annualized Sharpe ratio."""
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Annualize (assuming hourly returns)
    annual_return = mean_return * 252 * 6.5  # 252 days × 6.5 hours/day
    annual_std = std_return * np.sqrt(252 * 6.5)
    
    sharpe = (annual_return - risk_free_rate) / annual_std
    return sharpe
```

**Target:** ≥0.50 (Phase 3), ≥0.80 (Stretch)

**SL Baseline:** -0.05 (MLP @ threshold 0.80)

**Interpretation:**
- <0.0: Worse than risk-free rate (FAIL)
- 0.0-0.3: Poor risk-adjusted return
- 0.3-0.5: Acceptable (Phase 3 minimum)
- 0.5-1.0: Good (Phase 3 target)
- >1.0: Excellent (rare in trading)

---

**2. Total Return**

```python
def compute_total_return(final_value: float, initial_value: float) -> float:
    """Total return over evaluation period."""
    return (final_value - initial_value) / initial_value
```

**Target:** ≥+12% (2-year period)

**SL Baseline:** -10.9% (MLP @ threshold 0.80)

**Annualized Equivalent:**
```python
def annualize_return(total_return: float, years: float) -> float:
    """Convert total to annualized return."""
    return (1 + total_return) ** (1 / years) - 1
```

**Target Annualized:** ≥+15% (from +12% total over 2 years)

---

**3. Maximum Drawdown**

```python
def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough decline."""
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peaks) / peaks
    max_dd = np.min(drawdowns)  # Most negative value
    return abs(max_dd)
```

**Target:** ≤25% (Phase 3), ≤20% (Stretch)

**SL Baseline:** 12.4% (but with -10.9% return, so meaningless)

**Risk Categories:**
- <10%: Conservative (excellent risk management)
- 10-20%: Moderate (good)
- 20-30%: Aggressive (acceptable if returns justify)
- >30%: Dangerous (likely FAIL unless exceptional returns)

---

**4. Win Rate**

```python
def compute_win_rate(trades: List[Trade]) -> float:
    """Percentage of profitable trades."""
    profitable = sum(1 for t in trades if t.pnl > 0)
    return profitable / len(trades) if trades else 0.0
```

**Target:** ≥52%

**SL Baseline:** 47.7%

**Interpretation:**
- <45%: Poor strategy (FAIL unless profit factor >2.0)
- 45-50%: Below average
- 50-55%: Average (acceptable with good profit factor)
- 55-60%: Good
- >60%: Excellent (but check for over-optimization)

---

**5. Profit Factor**

```python
def compute_profit_factor(trades: List[Trade]) -> float:
    """Gross profit / Gross loss."""
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    
    return gross_profit / gross_loss if gross_loss > 0 else float('inf')
```

**Target:** ≥1.30

**SL Baseline:** 0.82

**Interpretation:**
- <1.0: Losing strategy (FAIL)
- 1.0-1.2: Marginal (risky)
- 1.2-1.5: Acceptable
- 1.5-2.0: Good
- >2.0: Excellent (but verify not overfit)

---

### 2.2 Secondary Metrics (Risk Analysis)

**6. Sortino Ratio**

```python
def compute_sortino_ratio(returns: np.ndarray, target_return: float = 0.0) -> float:
    """Sharpe-like ratio using only downside volatility."""
    mean_return = np.mean(returns)
    downside_returns = returns[returns < target_return]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
    
    return (mean_return - target_return) / downside_std
```

**Target:** ≥0.7 (higher than Sharpe due to asymmetric risk)

---

**7. Calmar Ratio**

```python
def compute_calmar_ratio(annual_return: float, max_dd: float) -> float:
    """Annual return / Max drawdown."""
    return annual_return / max_dd if max_dd > 0 else 0.0
```

**Target:** ≥1.0 (15% return / 15% max DD)

---

**8. Trade Frequency**

```python
def compute_trade_frequency(trades: List[Trade], days: int) -> float:
    """Trades per year."""
    trades_per_day = len(trades) / days
    return trades_per_day * 252
```

**Target:** 500-1000 trades/year (for 10 agents over 2 years)

**Warning Thresholds:**
- <100/year: Too passive (missing opportunities)
- 100-500/year: Conservative (good for transaction costs)
- 500-1000/year: Moderate (Phase 3 target)
- 1000-2000/year: Active (watch transaction costs)
- >2000/year: Overtrading (likely FAIL due to costs)

---

**9. Average Trade Duration**

```python
def compute_avg_duration(trades: List[Trade]) -> float:
    """Average hold time in hours."""
    durations = [t.exit_time - t.entry_time for t in trades]
    return np.mean(durations) if durations else 0.0
```

**Target:** 4-12 hours (based on 8-hour max hold from environment)

---

**10. Risk-Adjusted Trade Quality**

```python
def compute_trade_quality(trades: List[Trade]) -> Dict:
    """Comprehensive trade quality metrics."""
    return {
        "avg_win": np.mean([t.pnl for t in trades if t.pnl > 0]),
        "avg_loss": np.mean([t.pnl for t in trades if t.pnl < 0]),
        "win_loss_ratio": avg_win / abs(avg_loss),
        "largest_win": max(t.pnl for t in trades),
        "largest_loss": min(t.pnl for t in trades),
        "consecutive_wins": max_consecutive_wins(trades),
        "consecutive_losses": max_consecutive_losses(trades),
    }
```

---

### 2.3 Action Distribution Metrics

**11. Action Diversity**

```python
def compute_action_diversity(actions: List[int]) -> Dict:
    """Measure policy richness."""
    from collections import Counter
    counts = Counter(actions)
    total = len(actions)
    
    return {
        "hold_pct": counts[0] / total,
        "buy_pct": sum(counts[i] for i in [1,2,3]) / total,
        "sell_pct": sum(counts[i] for i in [4,5,6]) / total,
        "entropy": -sum(p * np.log2(p) for p in 
                       [counts[i]/total for i in range(7)] if p > 0),
    }
```

**Target Entropy:** >1.0 (indicates diverse policy)

**Warning:** Entropy <0.5 suggests degenerate policy

---

## 3. SL Baseline Comparison Framework

### 3.1 Direct Comparison Protocol

**Step 1: Run SL Baseline on Same Data**

```python
# Run SL models on validation period
sl_results = {}
for model in ["MLP_trial72", "LSTM_trial62", "GRU_trial93"]:
    for threshold in [0.60, 0.70, 0.80]:
        result = backtest_sl_model(
            model=model,
            threshold=threshold,
            data=validation_data,
            config=episode_config,
        )
        sl_results[f"{model}_t{threshold}"] = result
```

**Step 2: Select Best SL Configuration**

Based on `docs/baseline_for_rl_comparison.md`:
- **Best Config:** MLP @ threshold 0.80
- **Metrics:** -10.9% return, -0.05 Sharpe, 12.4% DD, 47.7% win rate

**Step 3: Compare RL vs SL**

```python
comparison = {
    "rl_vs_sl_return": rl_return - sl_return,
    "rl_vs_sl_sharpe": rl_sharpe - sl_sharpe,
    "rl_improvement_pct": (rl_return - sl_return) / abs(sl_return) * 100,
}
```

---

### 3.2 Success Thresholds

**Minimum Viable (Phase 3 Pass):**
- RL return > SL return + 20% (e.g., RL +10% vs SL -10.9%)
- RL Sharpe > SL Sharpe + 0.5 (e.g., RL 0.5 vs SL -0.05)
- RL beats SL on at least 3/5 primary metrics

**Strong Success (Phase 3 Target):**
- RL return > +12% (vs SL -10.9%)
- RL Sharpe > 0.50 (vs SL -0.05)
- RL beats SL on all 5 primary metrics
- RL max DD ≤ 25% (vs SL 12.4% but with losses)

**Exceptional (Stretch Goal):**
- RL return > +20%
- RL Sharpe > 0.80
- RL profit factor > 1.5
- RL beats SL by >100% on return

---

### 3.3 Regime-Specific Comparison

**Bull Market (2024 Q1):**
```python
bull_comparison = compare_in_regime(
    rl_agent, sl_model, 
    start="2024-01-01", end="2024-03-31"
)
```

**Expected:** Both should profit, but RL should have higher Sharpe

---

**Bear Market (2024 Q3):**
```python
bear_comparison = compare_in_regime(
    rl_agent, sl_model,
    start="2024-07-01", end="2024-09-30"
)
```

**Expected:** RL should lose less than SL (better risk management)

---

**Sideways Market (2024 Q2):**
```python
sideways_comparison = compare_in_regime(
    rl_agent, sl_model,
    start="2024-04-01", end="2024-06-30"
)
```

**Expected:** RL should avoid overtrading (lower transaction costs)

---

## 4. Validation Workflows

### 4.1 During-Training Validation (Tier 1)

**Executed by:** `EvalCallback` in Stable-Baselines3

```python
eval_callback = EvalCallback(
    eval_env=validation_env,
    n_eval_episodes=10,
    eval_freq=5000,  # Every 5k training steps
    log_path="logs/eval/",
    best_model_save_path="models/best/",
    deterministic=True,
)
```

**Process:**
1. Pause training every 5,000 steps
2. Run 10 validation episodes (deterministic policy)
3. Compute metrics (Sharpe, return, DD)
4. Save checkpoint if best Sharpe so far
5. Resume training

**Early Stopping:**
- If Sharpe plateaus for 5 evaluations (25k steps) → stop
- If Sharpe exceeds 0.80 (stretch goal) → stop

---

### 4.2 Post-Training Validation (Tier 2)

**Executed manually after training complete**

```python
def validate_agent(agent_path: str, test_env: gym.Env) -> Dict:
    """Comprehensive validation on hold-out test set."""
    
    # Load best checkpoint
    agent = PPO.load(agent_path)
    
    # Run 20 episodes
    all_metrics = []
    for ep in range(20):
        obs = test_env.reset()
        done = False
        episode_data = []
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            episode_data.append(info)
        
        # Compute episode metrics
        metrics = compute_all_metrics(episode_data)
        all_metrics.append(metrics)
    
    # Aggregate
    return {
        "sharpe_mean": np.mean([m["sharpe"] for m in all_metrics]),
        "sharpe_std": np.std([m["sharpe"] for m in all_metrics]),
        "return_mean": np.mean([m["return"] for m in all_metrics]),
        "return_std": np.std([m["return"] for m in all_metrics]),
        "max_dd_mean": np.mean([m["max_dd"] for m in all_metrics]),
        # ... all other metrics
    }
```

**Success Criteria (Per Agent):**
- Sharpe_mean ≥ 0.30
- Return_mean ≥ +5%
- Max_DD_mean ≤ 30%
- No episodes with DD >50%

---

### 4.3 Walk-Forward Validation (Tier 3)

**Purpose:** Test temporal robustness

**Method:** Rolling window evaluation

```python
def walk_forward_validation(agent, data, window_months=6, step_months=3):
    """
    Rolling window validation.
    
    Args:
        window_months: Size of validation window
        step_months: How much to advance each step
    """
    
    results = []
    start_date = data.start_date
    end_date = data.end_date
    
    current_start = start_date
    while current_start + timedelta(days=window_months*30) <= end_date:
        current_end = current_start + timedelta(days=window_months*30)
        
        # Validate on this window
        window_data = data.slice(current_start, current_end)
        metrics = validate_agent(agent, window_data)
        
        results.append({
            "window_start": current_start,
            "window_end": current_end,
            **metrics
        })
        
        # Advance window
        current_start += timedelta(days=step_months*30)
    
    return results
```

**Analysis:**
- Plot Sharpe over time windows
- Check for degradation trend
- Identify regime-dependent performance

**Success Criteria:**
- Sharpe positive in ≥80% of windows
- No window with return <-20%
- Sharpe std across windows <0.3 (consistent)

---

## 5. Agent-Level Success Criteria

### 5.1 Minimum Viable Agent (MVA)

**Definition:** Agent that passes Phase 3 and can proceed to Phase 4

**Criteria:**
- [x] **Validation Sharpe ≥ 0.30** (critical)
- [x] **Test Sharpe ≥ 0.25** (within 20% of validation)
- [x] **Total Return > 0%** (at minimum, don't lose money)
- [x] **Max Drawdown ≤ 35%** (risk tolerance)
- [x] **Training Completed** (no crashes, NaN losses)
- [x] **No Degenerate Behavior** (action entropy >0.5)
- [x] **Beats SL Baseline** (on at least 2/5 primary metrics)

**Minimum to Proceed:** ≥5/10 agents meet MVA criteria

---

### 5.2 Target Agent (TA)

**Definition:** Agent that meets Phase 3 targets

**Criteria:**
- [x] **Validation Sharpe ≥ 0.50** (target)
- [x] **Total Return ≥ +12%** (beat SL by >20%)
- [x] **Max Drawdown ≤ 25%** (good risk management)
- [x] **Win Rate ≥ 52%** (better than SL 47.7%)
- [x] **Profit Factor ≥ 1.30** (positive expectancy)
- [x] **Trade Frequency** 50-100 trades/year (reasonable)
- [x] **Beats SL Baseline** (on all 5 primary metrics)

**Target:** ≥3/10 agents meet TA criteria

---

### 5.3 Exceptional Agent (EA)

**Definition:** Agent that exceeds expectations (stretch goal)

**Criteria:**
- [x] **Validation Sharpe ≥ 0.80**
- [x] **Total Return ≥ +20%**
- [x] **Max Drawdown ≤ 20%**
- [x] **Win Rate ≥ 55%**
- [x] **Profit Factor ≥ 1.50**
- [x] **Calmar Ratio ≥ 1.5**
- [x] **Consistent Across Regimes** (positive in all 3)

**Aspirational:** ≥1/10 agents meet EA criteria

---

## 6. Portfolio-Level Validation

### 6.1 Multi-Agent Portfolio

**Combine all 10 agents into portfolio:**

```python
def evaluate_portfolio(agents: List[Agent], env: gym.Env) -> Dict:
    """
    Evaluate portfolio of agents.
    
    Strategy: Equal-weight allocation across all 10 symbols.
    """
    
    portfolio_capital = 100_000 * 10  # $1M total
    allocation_per_symbol = 100_000   # $100k each
    
    # Run all agents in parallel
    observations = {symbol: env[symbol].reset() for symbol in symbols}
    portfolio_equity = portfolio_capital
    
    while not all_done:
        # Get actions from all agents
        actions = {
            symbol: agents[symbol].predict(observations[symbol])
            for symbol in symbols
        }
        
        # Step all environments
        for symbol in symbols:
            obs, reward, done, truncated, info = env[symbol].step(actions[symbol])
            observations[symbol] = obs
            portfolio_equity += info["equity_change"]
    
    # Compute portfolio metrics
    return compute_portfolio_metrics(portfolio_equity_curve)
```

**Portfolio-Level Targets:**
- **Portfolio Sharpe ≥ 0.60** (diversification benefit)
- **Portfolio Return ≥ +15%** (10 agents × average return)
- **Portfolio Max DD ≤ 20%** (less than individual agents)
- **Correlation Matrix** (agents should have <0.7 correlation)

---

### 6.2 Diversification Analysis

```python
def analyze_diversification(agents: List[Agent]) -> Dict:
    """Measure diversification benefits."""
    
    # Get return series for each agent
    returns = {symbol: agent.get_returns() for symbol, agent in agents.items()}
    
    # Correlation matrix
    import pandas as pd
    df = pd.DataFrame(returns)
    corr_matrix = df.corr()
    
    # Diversification ratio
    portfolio_std = df.sum(axis=1).std()
    weighted_avg_std = df.std().mean()
    diversification_ratio = weighted_avg_std / portfolio_std
    
    return {
        "correlation_matrix": corr_matrix,
        "avg_correlation": corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean(),
        "max_correlation": corr_matrix.values.max(),
        "diversification_ratio": diversification_ratio,
    }
```

**Targets:**
- Average correlation <0.5 (agents are differentiated)
- Diversification ratio >1.2 (portfolio benefits from diversification)

---

## 7. Reporting & Documentation

### 7.1 Validation Report Template

Create `analysis/reports/phase3_validation_report.md`:

```markdown
# Phase 3 Validation Report

**Date:** 2025-10-XX  
**Agents Validated:** 10  
**Validation Period:** 2025-01-01 to 2025-07-31  
**Test Period:** 2025-08-01 to 2025-10-01

## Executive Summary

- **Agents Passing MVA:** 8/10 (80%) ✅
- **Agents Achieving TA:** 4/10 (40%) ✅
- **Agents Achieving EA:** 1/10 (10%) ✅
- **Portfolio Sharpe:** 0.67 ✅
- **SL Comparison:** RL beats SL by 65% on return ✅

**Recommendation:** PROCEED to Phase 4 (143 agents)

## Individual Agent Results

### SPY Agent
- **Validation Sharpe:** 0.42 (TA) ✅
- **Test Sharpe:** 0.38 (-9% from validation) ✅
- **Total Return:** +14.2% ✅
- **Max Drawdown:** 18.3% ✅
- **Win Rate:** 53.1% ✅
- **Profit Factor:** 1.38 ✅
- **Classification:** Target Agent (TA)

[Repeat for all 10 agents]

## Portfolio Analysis

[Portfolio metrics, correlation matrix, diversification analysis]

## SL Baseline Comparison

[Side-by-side comparison tables and charts]

## Regime Analysis

[Performance breakdown by market regime]

## Risk Analysis

[Stress test results, failure modes, edge cases]

## Recommendations

[Phase 4 scaling strategy, hyperparameter finalization, risk controls]

## Appendices

[Detailed metrics, plots, trajectories]
```

---

### 7.2 Validation Dashboard

**MLflow Dashboard:**
- Comparison view: All 10 agents side-by-side
- Metric trends: Sharpe over training
- Best model tracking: Top 3 by Sharpe

**TensorBoard:**
- Validation curves
- Distribution plots (actions, rewards)
- Episode replays

**Custom Report:**
```python
import plotly.graph_objects as go

def create_validation_dashboard(results: Dict):
    """Generate interactive HTML dashboard."""
    
    fig = go.Figure()
    
    # Sharpe comparison
    fig.add_trace(go.Bar(
        name='RL Agents',
        x=list(results.keys()),
        y=[r['sharpe'] for r in results.values()],
    ))
    
    fig.add_trace(go.Bar(
        name='SL Baseline',
        x=list(results.keys()),
        y=[-0.05] * len(results),  # SL sharpe
    ))
    
    fig.update_layout(
        title='Phase 3 Validation: Sharpe Ratio Comparison',
        xaxis_title='Agent',
        yaxis_title='Sharpe Ratio',
        barmode='group',
    )
    
    fig.write_html('validation_dashboard.html')
```

---

## 8. Go/No-Go Decision Framework

### 8.1 Phase 3 → Phase 4 Approval

**Mandatory Requirements (ALL must pass):**

1. **✅ Agent Success Rate:** ≥5/10 agents achieve MVA
2. **✅ Target Achievement:** ≥3/10 agents achieve TA  
3. **✅ SL Improvement:** ≥1 agent beats SL by ≥20% on return
4. **✅ Training Stability:** ≥7/10 agents complete without crashes
5. **✅ No Catastrophic Failures:** No agent with DD >50% or return <-30%
6. **✅ Portfolio Viability:** Portfolio Sharpe ≥0.50
7. **✅ Hyperparameters Validated:** Final config tested on ≥3 agents

**Optional Success (Bonus, not required):**
- 🎯 ≥8/10 agents achieve MVA
- 🎯 Portfolio Sharpe ≥0.80
- 🎯 All agents beat SL baseline
- 🎯 ≥1 Exceptional Agent (EA)

---

### 8.2 Decision Matrix

| Scenario | Agent MVA | Agent TA | Portfolio Sharpe | Decision |
|----------|-----------|----------|------------------|----------|
| Best Case | 10/10 | 7/10 | >0.80 | **GO** - Excellent, scale aggressively |
| Target Case | 8/10 | 4/10 | 0.60-0.80 | **GO** - Solid, proceed as planned |
| Minimum Case | 5/10 | 2/10 | 0.50-0.60 | **CONDITIONAL GO** - Scale cautiously to 25-50 agents first |
| Below Minimum | <5/10 | <2/10 | <0.50 | **NO-GO** - Iterate on Phase 3 |

---

### 8.3 Conditional Go Strategy (Phase 3.5)

**If 5-7 agents succeed (borderline):**

1. **Identify Success Patterns:**
   - Which symbols succeeded? (SPY, QQQ, large-cap?)
   - Which hyperparameters worked best?
   - Common characteristics of successful agents?

2. **Intermediate Scaling (Phase 3.5):**
   - Scale to 25-50 agents (instead of 143)
   - Focus on symbols similar to successful prototypes
   - Validate at scale before full deployment

3. **Additional Validation:**
   - Longer training (150k steps instead of 100k)
   - More conservative hyperparameters
   - Enhanced monitoring

4. **Decision Point:**
   - If Phase 3.5 succeeds (≥20/25 agents MVA) → Phase 4
   - If Phase 3.5 fails → Return to architecture/reward iteration

---

## 9. Failure Analysis Protocol

### 9.1 Agent Failure Classification

**Type 1: Training Failure**
- NaN losses
- Crashes
- No convergence
- **Action:** Debug training loop, adjust hyperparameters

**Type 2: Performance Failure**
- Sharpe <0.0
- Large losses (return <-20%)
- **Action:** Review reward function, check environment

**Type 3: Degenerate Policy**
- All HOLD
- Random actions
- **Action:** Increase entropy, adjust action masking

**Type 4: Overfitting**
- Validation << Training performance
- **Action:** Reduce capacity, add regularization, more data

**Type 5: SL Baseline Failure**
- Can't beat SL despite decent metrics
- **Action:** May actually be OK if SL is flawed benchmark

---

### 9.2 Debugging Workflow

**Step 1: Isolate Failure Mode**
```python
# Check training logs
mlflow.search_runs(filter_string="metrics.sharpe < 0.1")

# Identify pattern
failed_agents = [...]
successful_agents = [...]

# Compare configurations
compare_configs(failed_agents, successful_agents)
```

**Step 2: Root Cause Analysis**
- Review TensorBoard curves
- Inspect episode replays
- Check reward component breakdown
- Analyze action distributions

**Step 3: Hypothesis & Fix**
- Formulate hypothesis (e.g., "entropy too low")
- Test fix on 1 agent
- Validate improvement
- Apply to remaining agents

---

## 10. Success Stories & Best Practices

### 10.1 Expected Success Profile

**Typical Strong Agent:**
- Converges by 60-80k steps
- Validation Sharpe 0.40-0.60
- Balanced action distribution (40% HOLD, 30% BUY, 30% SELL)
- Profit factor 1.2-1.4
- Trade frequency 50-80/year
- Consistent across regimes

**Common Patterns:**
- Large-cap tech (AAPL, MSFT) often succeed first
- Indices (SPY, QQQ) provide stable baseline
- High-volatility stocks (TSLA, NVDA) take longer to learn

---

### 10.2 Lessons from Supervised Learning Failure

**What SL Did Wrong (and RL Must Avoid):**
1. **Ignored Transaction Costs** → RL: 15% reward weight on costs
2. **Poor Timing** → RL: 15% reward weight on time efficiency  
3. **Overtraded** → RL: Entropy decay to reduce churn over time
4. **No Risk Management** → RL: 10% reward weight on drawdown
5. **Binary Decisions** → RL: 7 discrete actions for nuance

---

## Appendices

### A. Metrics Computation Library

```python
# See validation_metrics.py for full implementation
from validation_metrics import (
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_profit_factor,
    compute_trade_quality,
    generate_validation_report,
)
```

### B. Validation Script

```bash
# Run full validation suite
python scripts/validate_phase3_agents.py \
  --agents-dir training/rl/phase3_output/best/ \
  --validation-data data/historical/ \
  --output-dir analysis/reports/phase3_validation/
```

### C. Comparison with Other RL Studies

**Academic Benchmarks:**
- Deng et al. (2016): Sharpe 0.47 on portfolio
- Jiang et al. (2017): Sharpe 0.31 on crypto
- Zhang et al. (2020): Sharpe 0.62 on stocks

**Our Target:** Sharpe 0.50 (competitive with literature)

---

**Document Status:** Ready for Phase 3 Implementation  
**Next Update:** After Phase 3 completion with empirical validation results