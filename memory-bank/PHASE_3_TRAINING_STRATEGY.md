# Phase 3: Training Strategy & Optimization Guide

**Document Version:** 1.0  
**Created:** October 6, 2025  
**Phase:** 3 - Prototype Training & Validation  
**Status:** Implementation Ready

---

## Executive Summary

This document provides detailed training strategies, optimization techniques, and troubleshooting guidance for Phase 3 prototype training. It addresses the unique challenges of training 10 multi-agent RL systems that must overcome catastrophic SL baseline failures (-88% to -93% losses) while establishing a scalable foundation for Phase 4 (143 agents).

**Key Challenges Addressed:**
1. **SL Catastrophic Failure Context:** Models must learn cost-awareness, timing, and sequential decisions
2. **Multi-Agent Coordination:** 10 agents sharing encoder without interference
3. **Reward Engineering:** Balance 7-component reward to avoid hacking
4. **Sample Efficiency:** Achieve results with 100k steps (limited compute budget)
5. **Generalization:** Perform across bull/bear/sideways regimes

---

## 1. PPO Training Strategy

### 1.1 Why PPO for This Problem?

**PPO Advantages:**
- **Sample Efficiency:** On-policy with multiple epochs per batch
- **Stability:** Clipped objective prevents destructive updates
- **Proven:** State-of-the-art for continuous control and trading
- **SB3 Support:** Mature implementation with callbacks

**Alternative Algorithms Considered:**
- **SAC:** Better for continuous action spaces; our actions are discrete
- **TD3:** Same issue + more complex than needed
- **A2C:** Less stable than PPO, no clipping
- **DQN:** No policy gradient; harder to incorporate entropy

**Decision:** PPO is optimal for Phase 3 prototype.

### 1.2 Core Hyperparameters

**Learning Rate: 3e-4 (baseline)**

```python
learning_rate = 3e-4  # Standard PPO default

# Adaptive schedule (if baseline unstable):
def lr_schedule(progress: float) -> float:
    """Linear decay from 3e-4 to 1e-5"""
    return 3e-4 * (1 - 0.97 * progress)
```

**Rationale:**
- 3e-4 proven across many RL domains
- Too high (>1e-3): Training instability, NaN losses
- Too low (<1e-4): Slow convergence, may not reach target in 100k steps

**Tuning Priority:** HIGH - First parameter to adjust if issues arise

---

**Rollout Buffer (n_steps): 2048**

```python
n_steps = 2048  # Collect 2048 steps before update
# With 8 parallel envs: 2048 × 8 = 16,384 timesteps per update
```

**Rationale:**
- Larger buffers → better advantage estimates (more context)
- Smaller buffers → faster updates, better for non-stationary environments
- 2048 balances both for trading (regime shifts are slow)

**Memory Note:** 2048 × 8 envs × observation size (~500 floats) = ~60MB per agent

---

**PPO Clip Range: 0.2**

```python
clip_range = 0.2  # Clip policy ratio to [1-ε, 1+ε]
```

**Rationale:**
- 0.2 is standard, prevents too-large policy updates
- Lower (0.1): More conservative, slower learning
- Higher (0.3): Faster learning but risk instability

**Adaptive Clipping (if baseline fails):**
```python
def clip_schedule(progress: float) -> float:
    """Start aggressive (0.3), end conservative (0.1)"""
    return 0.3 - 0.2 * progress
```

---

**Entropy Coefficient: 0.01**

```python
ent_coef = 0.01  # Entropy bonus weight
```

**CRITICAL for avoiding degenerate policies!**

**Rationale:**
- Encourages exploration by penalizing deterministic policies
- Too low (<0.001): Premature convergence to suboptimal policy (all HOLD)
- Too high (>0.05): Agent never commits to strategy

**Warning Signs:**
- Entropy → 0 quickly: Increase to 0.05
- All actions = HOLD: Increase to 0.10 temporarily
- Random behavior after 50k steps: Decrease to 0.001

**Entropy Decay Schedule:**
```python
def ent_schedule(progress: float) -> float:
    """Start high (0.05), decay to low (0.001)"""
    return 0.05 * (1 - progress) + 0.001 * progress
```

---

**GAE Lambda: 0.95**

```python
gae_lambda = 0.95  # Generalized Advantage Estimation
```

**Rationale:**
- Controls bias-variance tradeoff in advantage estimates
- 0.95 balances near-term (credit assignment) and long-term (Sharpe) rewards
- Higher (0.98): More long-term bias, better for portfolio metrics
- Lower (0.90): Less bias, better when rewards are noisy

**Tuning Priority:** MEDIUM - Adjust if value function struggles

---

**Value Function Coefficient: 0.5**

```python
vf_coef = 0.5  # Value loss weight in total loss
```

**Rationale:**
- Balances policy loss and value loss
- Standard default works well for most problems
- Increase to 1.0 if explained variance is low (<0.2)

---

**Gradient Clipping: 0.5**

```python
max_grad_norm = 0.5  # Clip gradients to prevent explosions
```

**CRITICAL for transformer stability!**

**Rationale:**
- Transformers can have large gradients during early training
- 0.5 is conservative, prevents NaN losses
- If gradients consistently hit 0.5, consider:
  - Reducing LR
  - Increasing clip_range
  - Checking reward scaling

**Monitoring:**
```python
# Log in training loop
mlflow.log_metric("grad_norm", grad_norm, step=update)
mlflow.log_metric("grad_clipped_pct", clipped_pct, step=update)
```

---

### 1.3 Batch Size & Epochs

**Mini-Batch Size: 64**

```python
batch_size = 64  # From 16,384 rollout buffer
# Updates per epoch: 16,384 / 64 = 256
```

**Rationale:**
- Smaller batches: Noisier gradients, faster updates
- Larger batches: Stabler gradients, slower updates
- 64 balances both for RTX 5070 Ti memory

**GPU Memory Equation:**
```
Memory ≈ batch_size × (obs_size + action_size + advantage_size) × 4 bytes
       ≈ 64 × (500 + 10 + 10) × 4 ≈ 133KB per mini-batch
```

---

**Epochs Per Update: 10**

```python
n_epochs = 10  # Process rollout buffer 10 times
```

**Rationale:**
- More epochs → better sample efficiency, risk overfitting
- Fewer epochs → less overfitting, worse sample efficiency
- 10 is proven default for PPO

**KL Divergence Monitoring:**
```python
# If KL divergence > 0.015, stop early
if kl_div > 0.015:
    break  # Policy changed too much
```

---

### 1.4 Discount Factor (Gamma)

**Gamma: 0.99**

```python
gamma = 0.99  # Discount factor for future rewards
```

**Effective Horizon:**
```
H = 1 / (1 - γ) = 1 / (1 - 0.99) = 100 steps
```

With 1-hour bars, this is ~4 days of trading horizon.

**Rationale:**
- 0.99 standard for episodic tasks
- Lower (0.95): Myopic, focuses on immediate rewards (good for day trading)
- Higher (0.995): Far-sighted, better for Sharpe optimization

**Recommendation:** Start with 0.99, increase to 0.995 if Sharpe is primary metric.

---

## 2. Reward Engineering & Tuning

### 2.1 7-Component Reward Breakdown

From [`RewardShaper`](core/rl/environments/reward_shaper.py:1):

```python
reward = (
    0.40 * r_pnl +         # Equity growth
    0.15 * r_cost +        # Transaction cost penalty
    0.15 * r_time +        # Time efficiency
    0.05 * r_sharpe +      # Risk-adjusted return
    0.10 * r_drawdown +    # Drawdown penalty
    0.05 * r_sizing +      # Position sizing quality
    0.00 * r_hold          # Hold penalty (disabled)
)
```

**Design Rationale:**
- **PnL (40%):** Primary objective - make money
- **Cost (15%):** Addresses SL failure #1 - transaction cost blindness
- **Time (15%):** Addresses SL failure #2 - poor timing
- **Sharpe (5%):** Regularization toward risk-adjusted returns
- **Drawdown (10%):** Risk management, avoid catastrophic losses
- **Sizing (5%):** Encourages proper position sizing
- **Hold (0%):** Disabled to avoid discouraging patience

### 2.2 Component Analysis

**PnL Component (r_pnl):**

```python
def compute_pnl_reward(self, equity_change: float) -> float:
    """Normalized P&L reward."""
    return equity_change / self.initial_capital
```

**Range:** [-0.10, +0.10] typically (±10% equity change per step)

**Issues to Watch:**
- **Dominates other components:** Reduce weight to 0.30
- **Too volatile:** Add smoothing or clipping
- **Encourages reckless trades:** Increase cost/drawdown weights

---

**Cost Component (r_cost):**

```python
def compute_cost_reward(self, commission: float, slippage: float) -> float:
    """Penalize transaction costs."""
    total_cost = commission + slippage
    return -total_cost / self.initial_capital
```

**Critical for avoiding SL failure mode!**

**Range:** [-0.002, 0] (0% cost = 0 reward, 0.2% cost = -0.002)

**Tuning:**
- SL models ignored this → lost money to churn
- If agent trades too much (>1000 trades/year): Increase weight to 0.25
- If agent never trades: Decrease to 0.10

---

**Time Component (r_time):**

```python
def compute_time_reward(self, hold_duration: int, profitable: bool) -> float:
    """Reward quick profitable trades, penalize slow losers."""
    if profitable:
        return 1.0 / max(hold_duration, 1)  # Faster = better
    else:
        return -hold_duration / self.max_hold  # Slower loss = worse
```

**Addresses SL timing failure!**

**Range:** [0, 1.0] for wins, [-1.0, 0] for losses

**Issues:**
- **Agent exits winners too early:** Reduce weight to 0.05
- **Agent holds losers forever:** Increase weight to 0.25

---

**Sharpe Component (r_sharpe):**

```python
def compute_sharpe_reward(self, returns: List[float]) -> float:
    """Incremental Sharpe ratio contribution."""
    if len(returns) < 2:
        return 0.0
    mean_return = np.mean(returns)
    std_return = np.std(returns) + 1e-6
    return (mean_return / std_return) - self.target_sharpe
```

**Target Sharpe:** 1.0 (stretch goal)

**Range:** [-2.0, +2.0] typically

**Purpose:** Regularization toward risk-adjusted returns

---

**Drawdown Component (r_drawdown):**

```python
def compute_drawdown_penalty(self, current_dd: float) -> float:
    """Exponential penalty for large drawdowns."""
    if current_dd < 0.05:
        return 0.0  # No penalty for small DD
    elif current_dd < 0.15:
        return -current_dd  # Linear penalty
    else:
        return -current_dd ** 2  # Quadratic penalty
```

**Risk Management Focus:**

**Range:** [0, -0.25] (0% DD = 0, 50% DD = -0.25)

**Tuning:**
- If max DD exceeds 30%: Increase weight to 0.20
- If agent too conservative: Decrease to 0.05

---

### 2.3 Reward Scaling & Normalization

**Option 1: Running Mean/Std (Recommended)**

```python
from stable_baselines3.common.vec_env import VecNormalize

env = VecNormalize(
    env,
    norm_obs=True,      # Normalize observations
    norm_reward=True,   # Normalize rewards (running mean/std)
    clip_obs=10.0,      # Clip normalized obs to [-10, 10]
    clip_reward=10.0,   # Clip normalized reward to [-10, 10]
    gamma=0.99,         # For return normalization
)
```

**Advantages:**
- Automatic adaptation to reward magnitude
- Handles non-stationary rewards (regime shifts)
- Standard in SB3

**Disadvantages:**
- Can hide reward scale issues
- Needs warm-up period (first 1000 steps)

---

**Option 2: Manual Scaling**

```python
def scale_reward(raw_reward: float) -> float:
    """Scale reward to [-1, 1] range."""
    return np.tanh(raw_reward)  # Squash to [-1, 1]
```

**Use when:**
- Rewards have extreme outliers
- Want more control over scaling
- Debugging reward issues

---

### 2.4 Reward Debugging Protocol

**Step 1: Component Logging**

```python
# In training loop
self.logger.log({
    "reward/total": total_reward,
    "reward/pnl": r_pnl * 0.40,
    "reward/cost": r_cost * 0.15,
    "reward/time": r_time * 0.15,
    "reward/sharpe": r_sharpe * 0.05,
    "reward/drawdown": r_drawdown * 0.10,
    "reward/sizing": r_sizing * 0.05,
})
```

**Analyze:**
- Which component dominates? (Should be PnL ~40-50%)
- Are any components always zero? (Bug or design?)
- Do components correlate? (PnL and Sharpe should)

---

**Step 2: Reward Histogram**

```python
import matplotlib.pyplot as plt

plt.hist(all_rewards, bins=50)
plt.xlabel("Reward")
plt.ylabel("Frequency")
plt.title("Reward Distribution")
plt.savefig("reward_histogram.png")
```

**Expected:** Bell curve centered near 0, range [-1, 1]

**Red Flags:**
- Bimodal distribution → reward hacking
- All positive or all negative → imbalanced components
- Long tail → outliers need clipping

---

**Step 3: Reward-to-Go Analysis**

```python
def compute_reward_to_go(rewards: List[float], gamma: float) -> List[float]:
    """Discounted cumulative rewards."""
    rtg = []
    cumulative = 0
    for r in reversed(rewards):
        cumulative = r + gamma * cumulative
        rtg.append(cumulative)
    return list(reversed(rtg))
```

**Plot:** Reward-to-go over episode

**Expected:** Generally increasing for profitable episodes

**Issues:**
- Flat → agent not learning value function
- Erratic → reward too noisy, increase gamma

---

## 3. Training Dynamics & Convergence

### 3.1 Expected Learning Curves

**Phase 1: Exploration (0-20k steps)**
- Reward: Highly variable, near zero or negative
- Entropy: High (>1.5)
- Actions: Diverse, mostly HOLD
- Value Loss: High, explained variance near 0

**Normal:** Agent explores randomly, learns observation space

---

**Phase 2: Initial Learning (20k-50k steps)**
- Reward: Begins trending upward
- Entropy: Decreases (1.5 → 1.0)
- Actions: Emerges patterns (BUY when price drops)
- Value Loss: Decreasing, explained variance increasing

**Normal:** Agent discovers profitable strategies

---

**Phase 3: Refinement (50k-80k steps)**
- Reward: Continues improving, less variance
- Entropy: Stabilizes (0.8-1.0)
- Actions: Clear strategy visible
- Value Loss: Low, explained variance >0.5

**Normal:** Agent optimizes discovered strategy

---

**Phase 4: Convergence (80k-100k steps)**
- Reward: Plateaus at final level
- Entropy: Low but stable (0.5-0.8)
- Actions: Consistent policy
- Value Loss: Minimal, explained variance >0.7

**Normal:** Agent converged to near-optimal policy

---

### 3.2 Warning Signs & Interventions

**⚠️ NaN Losses**

**Symptoms:**
```
Step 15234: policy_loss = nan, value_loss = nan
```

**Causes:**
1. Gradient explosion (grad_norm >100)
2. Reward outliers (|reward| >1000)
3. Numerical instability in value function

**Fixes:**
```python
# 1. Reduce learning rate
learning_rate = 1e-4  # From 3e-4

# 2. Increase gradient clipping
max_grad_norm = 0.3  # From 0.5

# 3. Clip rewards
env = VecNormalize(env, clip_reward=5.0)

# 4. Add observation clipping
obs = np.clip(obs, -10, 10)
```

---

**⚠️ Premature Convergence (All HOLD)**

**Symptoms:**
```
Step 30000:
  - 99% of actions = HOLD
  - Entropy < 0.1
  - Reward = 0
```

**Causes:**
1. Entropy coefficient too low
2. Reward components favor inaction
3. Action masking too restrictive

**Fixes:**
```python
# 1. Increase entropy bonus
ent_coef = 0.05  # From 0.01

# 2. Add action diversity reward
if action != HOLD:
    reward += 0.01  # Small bonus for any action

# 3. Reduce cost penalty temporarily
cost_weight = 0.05  # From 0.15

# 4. Reduce time weight (may penalize HOLD)
time_weight = 0.05  # From 0.15
```

---

**⚠️ Exploding Rewards**

**Symptoms:**
```
Step 25000: mean_reward = 15.3 (was 0.5 at step 24000)
```

**Causes:**
1. Reward hacking (agent found exploit)
2. Reward scaling broken
3. Environment bug

**Diagnosis:**
```python
# Log detailed episode trajectory
mlflow.log_artifact("episode_replay.json")

# Check:
# - What actions did agent take?
# - Which reward component spiked?
# - Is this a legitimate strategy?
```

**Fixes:**
- If legitimate: Accept and monitor
- If hack: Fix reward component or add constraint
- If bug: Debug environment

---

**⚠️ Value Function Collapse**

**Symptoms:**
```
Explained variance < 0 (negative!)
Value loss not decreasing
```

**Causes:**
1. Reward distribution non-stationary
2. Value coefficient too low
3. Batch size too small

**Fixes:**
```python
# 1. Increase value coefficient
vf_coef = 1.0  # From 0.5

# 2. Increase batch size
batch_size = 128  # From 64

# 3. Add value loss clipping
value_loss = torch.clamp(value_loss, 0, 10)

# 4. Use VecNormalize for reward
env = VecNormalize(env, norm_reward=True)
```

---

### 3.3 Convergence Criteria

**Minimum Success (proceed to next agent):**
- [x] Validation Sharpe >0.3
- [x] No crashes or NaN losses
- [x] Explained variance >0.5
- [x] Entropy stable (0.5-1.5)

**Strong Success (excellent agent):**
- [x] Validation Sharpe >0.5
- [x] Beats SL baseline by 20%
- [x] Max drawdown <20%
- [x] Action diversity >3 different actions used

**Early Stopping:**
- If validation Sharpe plateaus for 10k steps → stop
- If agent beats stretch goal (Sharpe >0.8) → stop and save

---

## 4. Optimization Techniques

### 4.1 Vectorized Environments

**Baseline: 8 Parallel Environments**

```python
env = make_vec_trading_env(
    symbol="AAPL",
    num_envs=8,
    vec_env_cls=SubprocVecEnv,  # Separate processes
)
```

**Speedup:** ~6-7× (not quite 8× due to overhead)

**Memory:** 8 × environment size (~500MB per agent)

**Alternative: DummyVecEnv (single process)**
```python
vec_env_cls=DummyVecEnv  # For debugging
```

Use when:
- Debugging environment issues
- Profiling code
- Single-threaded simplicity

---

### 4.2 Mixed Precision Training

**RTX 5070 Ti supports FP16!**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop
with autocast():
    loss = compute_loss(obs, actions, advantages)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits:**
- ~1.5-2× faster training
- ~40% less memory
- Minimal accuracy loss

**Risks:**
- Numerical instability (monitor for NaN)
- Not all operations support FP16

**Recommendation:** Try on 1 agent, validate results match FP32

---

### 4.3 Gradient Accumulation

**Use when memory-limited:**

```python
accumulation_steps = 4
batch_size_effective = 64 * 4  # 256

for i, batch in enumerate(dataloader):
    loss = compute_loss(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Benefits:**
- Simulate larger batch size
- Stabler gradients
- Same memory as smaller batch

**Downsides:**
- Slower training (4× more forward passes)

---

### 4.4 Checkpoint Frequency

**Baseline Strategy:**

```python
checkpoint_freq = 10000  # Every 10k steps
eval_freq = 5000         # Every 5k steps
```

**Storage:** ~500MB per checkpoint × 10 checkpoints = ~5GB per agent

**Optimization:**
- Save only best 3 checkpoints (by validation Sharpe)
- Compress checkpoints (torch.save with compression)
- Save to fast SSD

---

### 4.5 Early Stopping

**Patience-Based:**

```python
class EarlyStoppingCallback:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = -float('inf')
        self.wait = 0
    
    def __call__(self, metric):
        if metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True  # Stop training
        return False
```

**Use when:**
- Validation Sharpe plateaus
- Training time limited
- Agent already exceeds targets

---

## 5. Troubleshooting Guide

### 5.1 Common Issues & Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Slow Training** | <100 steps/sec | Reduce num_envs, use DummyVecEnv, profile code |
| **OOM (Out of Memory)** | CUDA OOM error | Reduce batch_size, num_envs, or use gradient accumulation |
| **Degenerate Policy** | All HOLD | Increase ent_coef, reduce cost_weight, check reward balance |
| **Unstable Training** | Reward variance high | Reduce learning_rate, increase batch_size, use reward normalization |
| **Poor Generalization** | Val << Train performance | Increase dropout, reduce training time, check distribution shift |
| **Action Masking Errors** | Masked action selected | Debug ActionMasker logic, check observation correctness |

---

### 5.2 Debugging Workflow

**Step 1: Isolate the Problem**
```python
# Test environment alone
env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    assert not np.isnan(obs).any()
    assert not np.isnan(reward)
```

**Step 2: Test Agent Forward Pass**
```python
agent.eval()
with torch.no_grad():
    obs_tensor = torch.from_numpy(obs).float()
    actions, log_probs, values = agent(obs_tensor)
    assert not torch.isnan(actions).any()
```

**Step 3: Test PPO Update**
```python
# Single update with synthetic data
batch = create_synthetic_batch(batch_size=64)
loss = model.train_on_batch(batch)
assert not np.isnan(loss)
```

**Step 4: Full Training with Logging**
```python
# Extensive logging
mlflow.log_metric("grad_norm", grad_norm, step=update)
mlflow.log_metric("clip_fraction", clip_frac, step=update)
mlflow.log_metric("kl_divergence", kl_div, step=update)
```

---

## 6. Success Metrics & KPIs

### 6.1 Training Metrics

**Policy Metrics:**
- `policy_loss`: Should decrease over time
- `entropy`: Should start high (>1.5), end moderate (0.5-1.0)
- `clip_fraction`: Should be 0.1-0.3 (indicates PPO clipping active)
- `kl_divergence`: Should be <0.02 (policy not changing too fast)

**Value Metrics:**
- `value_loss`: Should decrease
- `explained_variance`: Should increase to >0.5
- `value_mean`: Should correlate with episode returns

**Gradient Metrics:**
- `grad_norm`: Should be <1.0 (gradients not exploding)
- `grad_clipped_pct`: Should be <20% (not clipping too often)

---

### 6.2 Environment Metrics

**Episode Metrics:**
- `episode_reward_mean`: Should increase
- `episode_length_mean`: Should stabilize
- `episode_success_rate`: % of profitable episodes

**Trading Metrics:**
- `sharpe_ratio`: Target >0.3 (validation)
- `max_drawdown`: Target <25%
- `total_trades`: 500-1000 per 2 years
- `win_rate`: Target >52%
- `profit_factor`: Target >1.3

---

### 6.3 Reporting Template

**Weekly Status Report:**

```markdown
# Phase 3 Training Status - Week 1

## Agents Completed: 3/10
- ✅ SPY: Sharpe 0.42 (PASS)
- ✅ AAPL: Sharpe 0.38 (PASS)
- ✅ QQQ: Sharpe 0.31 (PASS)
- ⏳ MSFT: Training (45k/100k steps)

## Training Stability: GOOD
- No NaN losses
- Entropy stable (0.7-1.2)
- Grad norms <0.8

## Issues Encountered:
1. NVDA: Degenerate policy (all HOLD) → Increased ent_coef to 0.05
2. TSLA: High variance → Reduced LR to 1e-4

## Next Steps:
- Complete remaining 7 agents
- Begin hyperparameter tuning on best 3
- Prepare validation backtest
```

---

## 7. Phase 3 → Phase 4 Transition

### 7.1 Hyperparameter Finalization

**Export Final Config:**

```yaml
# training/rl/configs/phase4_production.yaml
# Validated hyperparameters from Phase 3

ppo:
  learning_rate: 3.0e-4  # Or tuned value
  ent_coef: 0.01         # Or tuned value
  gae_lambda: 0.95       # Or tuned value
  # ... all other params

# Add Phase 3 validation results
validation_sharpe_mean: 0.42
validation_sharpe_std: 0.08
training_stability_score: 0.95  # % of runs without crashes
```

---

### 7.2 Scaling Strategy

**Phase 4 will scale to 143 agents:**

**Option 1: Sequential (like Phase 3)**
- Time: 143 agents × 100 hours = 14,300 hours (596 days) ❌
- Not feasible!

**Option 2: Parallel Training (Ray)**
- 8 GPUs × 18 agents each = 144 agents
- Time: ~100-120 hours (4-5 days) ✅

**Recommendation:** Use Ray for distributed training in Phase 4

---

### 7.3 Lessons Learned Documentation

Create `analysis/reports/phase3_lessons_learned.md`:

**Sections:**
1. What worked well
2. What didn't work
3. Hyperparameter insights
4. Environment issues discovered
5. Recommendations for Phase 4

---

## 8. Appendices

### A. Hyperparameter Tuning Grid

```python
optuna_config = {
    "learning_rate": (1e-4, 1e-3, "log"),
    "ent_coef": (1e-3, 5e-2, "log"),
    "gae_lambda": (0.90, 0.98, "uniform"),
    "clip_range": (0.1, 0.3, "uniform"),
    "n_epochs": (5, 15, "int"),
    "batch_size": [32, 64, 128],
}
```

### B. Monitoring Dashboard Layout

**TensorBoard:**
- Tab 1: Scalars (losses, metrics)
- Tab 2: Distributions (gradients, actions)
- Tab 3: Images (value heatmaps)

**MLflow:**
- Run comparison table
- Hyperparameter importance
- Best model tracking

### C. References

- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
- Andrychowicz et al. (2020): "What Matters in On-Policy RL"
- Engstrom et al. (2020): "Implementation Matters in RL"

---

**Document Status:** Ready for Phase 3 Implementation  
**Next Update:** After Phase 3 completion with empirical results