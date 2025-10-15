# Phase 3: Training Strategy & Optimization Guide

**Document Version:** 1.1  
**Created:** October 6, 2025  
**Updated:** October 9, 2025  
**Phase:** 3 - Prototype Training & Validation  
**Status:** Implementation Ready (Stage 5 validation pending)

---

## Executive Summary

This document provides detailed training strategies, optimization techniques, and troubleshooting guidance for Phase 3 prototype training. It addresses the unique challenges of training 10 multi-agent RL systems that must overcome catastrophic SL baseline failures (-88% to -93% losses) while establishing a scalable foundation for Phase 4 (143 agents).

**Key Challenges Addressed:**
1. **SL Catastrophic Failure Context:** Models must learn cost-awareness, timing, and sequential decisions
2. **Multi-Agent Coordination:** 10 agents sharing encoder without interference
3. **Reward Engineering:** Balance 7-component reward to avoid hacking
4. **Sample Efficiency:** Achieve results with 100k steps (limited compute budget)
5. **Generalization:** Perform across bull/bear/sideways regimes
6. **Policy Collapse Prevention (NEW 2025-10-08):** Environment-level constraints to prevent degenerate single-action policies

### Stage Progress Snapshot (2025-10-09)

| Stage | Focus | Status | Evidence / Baseline |
|-------|-------|--------|----------------------|
| Stage 1 | Exploration recovery (actor gain 0.1, entropy schedule, reward normalization disabled) | ✅ Complete | `analysis/validate_exploration_fix.py`; smoke target entropy ≥1.3 with ≥4 actions/120 steps |
| Stage 2 | Professional reward stack (realized-only PnL, exit multipliers) | ✅ Complete | 3 k-step benchmark: Sharpe +0.563, win rate 64.5%, max DD 0.45% |
| Stage 3 | Anti-collapse invariants (diversity bonus 0.07, action repeat penalty 0.05, SELL mask audit) | ✅ Complete | `tests/test_reward_shaper.py` 42/42 (2025-10-09); SELL availability log entries |
| Stage 4 | Telemetry + Stage 5 runner (entropy guard, voluntary trade telemetry) | ✅ Complete | `scripts/run_phase3_stage5_validation.py` instrumentation validated via dry run (skip-training) |
| Stage 5 | Short retrain (5–10 k steps per seed) + telemetry diff | 🚧 Pending | Command staged; baseline success criteria enumerated in Section 9 |

**CRITICAL UPDATE (2025-10-08 → 2025-10-09):**  
Anti-collapse improvements implemented after catastrophic policy collapse (99.88% BUY_SMALL, 0.007 entropy). Follow-up on 2025-10-09 introduced entropy guard automation, telemetry baselines, diversity/repeat reward components, and Stage 5 validation workflow. See Section 9 for full details.

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

**Entropy Stack (UPDATED 2025-10-09)**

```python
ent_coef = 0.08  # Initial exploration weight (paired with scheduler)
entropy_scheduler = dict(strategy="hold_then_linear", initial=0.08, final=0.03,
                         hold_steps=10_000, decay_steps=40_000, min=0.03)
entropy_bonus = dict(enabled=True, target_entropy=0.55, bonus_scale=0.35,
                     warmup_steps=4_000, decay_rate=0.12, max_multiplier=3.0,
                     floor=0.02)
action_entropy_guard = dict(enabled=True, threshold=0.22, warmup_steps=4_000,
                            boost_multiplier=1.7, max_multiplier=3.0,
                            cooldown_steps=6_000, patience=2)
```

**Why the stack matters:**
- `ent_coef` starts at 0.08 but is held flat for the first 10 k steps, then linearly decays to a 0.03 floor—enough exploration without overwhelming policy loss.
- Adaptive entropy bonus auto-scales the reward signal toward target entropy 0.55, compensating when policy entropy dips due to non-stationary rewards.
- Action entropy guard halts catastrophic collapse in real time. When mean policy entropy <0.22 after warmup, it boosts `ent_coef` (up to 3×) and can halt training if entropy fails to recover.

**Operational guidance:**
- Expect `policy_action_entropy_mean` ≈0.6–0.8 during the first 5 k steps, settling near 0.35–0.45 by 20 k steps.
- Guard triggers should be rare; more than two per 10 k-step window signals deeper reward or masking issues (investigate Section 9 checklists).
- Maintain synergy with Section 9 constraints (diversity bonus, repeat penalty, SELL mask). Disabling any component voids Stage 5 baselines.

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

### 2.1 Reward Component Stack (UPDATED 2025-10-09)

From [`RewardShaper`](core/rl/environments/reward_shaper.py:1):

```python
reward = (
    0.75 * r_pnl +               # Equity growth (ROI-scaled)
    0.10 * r_cost +              # Transaction cost penalty (keeps churn honest)
    0.00 * r_time +              # Time efficiency (still disabled)
    0.03 * r_sharpe +            # Risk-adjusted return (light regularizer)
    0.02 * r_drawdown +          # Drawdown penalty (light pressure)
    0.00 * r_sizing +            # Position sizing quality (disabled)
    0.00 * r_hold +              # Hold penalty (disabled)
    0.07 * r_diversity +         # Action diversity bonus (anti-collapse)
    0.05 * r_action_repeat +     # Penalty for exceeding repetition caps
    0.01 * r_intrinsic_action    # Tiny intrinsic reward for valid non-HOLD actions
)
```

**Design Rationale (Stage 5 Baseline):**
- **PnL (75%):** Profit remains the dominant signal, now boosted back up because diversity is enforced elsewhere.
- **Cost (10%):** Enough friction to curb churn without overwhelming PnL after diversity/repetition controls.
- **Time (0%):** Still disabled—time pressure conflicted with Stage 5 voluntary-trade goals.
- **Sharpe (3%):** Lightly nudges risk-adjusted behavior without destabilizing early learning.
- **Drawdown (2%):** Provides gentle downside awareness.
- **Sizing/Hold (0%):** Disabled to avoid biasing HOLD vs BUY decisions during collapse recovery.
- **Diversity (7%):** Incentivizes ≥3 actions per 50-step window; pairs with telemetry targets.
- **Action Repeat (5%):** Penalizes exceeding rolling repetition quotas; hits chronic BUY spam even if mask fails.
- **Intrinsic Action (1%):** Offsets transaction costs for valid, non-HOLD actions so the agent will probe alternatives.

**CRITICAL INSIGHT (2025-10-08):**  
Hyperparameter tuning (actor gain, entropy, reward weights) CANNOT fix environment design flaws. After 8 failed hyperparameter configurations, root cause identified as environment allowing infinite action repetition. Solution requires environment-level hard constraints (see Section 9).

### 2.2 Component Analysis

**PnL Component (r_pnl):**

```python
def compute_pnl_reward(self, equity_change: float) -> float:
    """Normalized P&L reward."""
    return equity_change / self.initial_capital
```

**Range:** [-0.10, +0.10] typically (±10% equity change per step)

**Issues to Watch:**
- **Dominates other components:** Step weight back to 0.60 temporarily.
- **Too volatile:** Add smoothing or clip component output to ±1.5.
- **Encourages reckless trades:** Increase cost (0.15) or action-repeat penalty (0.07) rather than gutting PnL.

---

**Cost Component (r_cost):**

```python
def compute_cost_reward(self, commission: float, slippage: float) -> float:
    """Penalize transaction costs."""
    total_cost = commission + slippage
    return -total_cost / self.initial_capital
```

**Critical for avoiding SL failure mode!**

**Range:** [-0.001, 0] (0% cost = 0 reward, 0.1% cost = -0.001)

**Tuning:**
- If agent trades too much (>1 200 trades/year): Increase weight to 0.15.
- If agent never trades: Decrease to 0.05 (but inspect voluntary-trade rate first).
- Combine adjustments with `failed_action_penalty` and action-repeat penalties before touching PnL weight.

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
- **Agent exits winners too early:** Reintroduce with 0.05 weight after Stage 5 baseline validated.
- **Agent holds losers forever:** Prefer using drawdown penalty or forced-exit scaling before increasing time weight.

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

**Action Repeat Penalty (r_action_repeat):**

```python
def compute_action_repeat_penalty(self, diversity_info: Dict[str, Any]) -> float:
    streak = diversity_info.get("consecutive_action_count", 0)
    limit = diversity_info.get("max_consecutive_actions", 3)
    if streak <= limit:
        return 0.0
    overflow = streak - limit
    severity = min(overflow / limit, 1.0)
    return -0.5 * severity
```

**Purpose:** Applies graded penalties when the executed action exceeds the permitted streak length (default 3). Severity escalates as the streak length doubles the limit, capped at -0.5 before weighting.

**Tuning:**
- Raise weight to 0.07 if repetition violations persist even after mask fixes.
- Lower to 0.03 if penalty dominates total reward or creates oscillation between HOLD/BUY.
- Inspect `charts/consecutive_action_violations` and `sanitizer_trade_delta` before altering—penalty should spur SELL usage, not blind randomness.

---

**Intrinsic Action Reward (r_intrinsic_action):**

```python
def compute_intrinsic_action_reward(self, diversity_info: Dict[str, Any]) -> float:
    if diversity_info.get("executed_action_valid", False):
        return 0.1
    return 0.0
```

**Purpose:** Provides a small positive reward (scaled by 0.01) whenever the agent executes a valid non-HOLD action. Offsets transaction costs during exploratory probes so the policy keeps sampling SELL/ADD opportunities.

**Tuning:**
- Reduce to 0.0 once voluntary trade rate stabilizes >15% (prevents overtrading late in training).
- Increase to 0.02 temporarily if agent refuses to trade after curriculum relaxes.
- Pair with `failed_action_penalty` adjustments to ensure invalid attempts remain costly.

---

### 2.3 Reward Scaling & Normalization

> **Stage 5 Baseline (2025-10-09):** Reward normalization is **disabled**. We keep VecNormalize for observations only to preserve raw reward magnitudes for the entropy guard and telemetry.

**Option 1: Running Mean/Std (Reference Only)**

```python
from stable_baselines3.common.vec_env import VecNormalize

env = VecNormalize(
    env,
    norm_obs=True,      # Normalize observations (enabled)
    norm_reward=False,  # KEEP DISABLED unless Stage 5 guard remains stable for multiple runs
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=0.99,
)
```

**Advantages:**
- Automatic adaptation to reward magnitude
- Handles non-stationary rewards (regime shifts)
- Standard in SB3

**Disadvantages:**
- Can hide reward scale issues
- Needs warm-up period (first 1000 steps)
- Breaks Stage 5 telemetry/guard assumptions if enabled too early

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
    "reward/pnl": r_pnl * 0.75,
    "reward/cost": r_cost * 0.10,
    "reward/sharpe": r_sharpe * 0.03,
    "reward/drawdown": r_drawdown * 0.02,
    "reward/diversity": r_diversity * 0.07,
    "reward/action_repeat": r_action_repeat * 0.05,
    "reward/intrinsic": r_intrinsic_action * 0.01,
})
```

**Analyze:**
- Which component dominates? (PnL should contribute ~70-80%)
- Are any components always zero? (Telemetries miswired or component disabled)
- Do components correlate? (PnL vs action-repeat penalty should move inversely when collapse risk grows)

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

## 9. Anti-Collapse Improvements (CRITICAL UPDATE 2025-10-08)

### 9.1 Problem: Catastrophic Policy Collapse

**Incident Summary:**
- **Date:** October 8, 2025
- **Symptom:** Agents converged to 99.88% BUY_SMALL action within 10k training steps
- **Entropy:** Dropped to 0.007 (from >1.5)
- **Root Cause:** Environment design allowed infinite repetition of locally optimal actions
- **Failed Solutions:** 8 hyperparameter configurations (actor gain, entropy coef, reward weights, transaction costs)

**Key Insight:**  
In discrete action spaces with locally optimal degenerate strategies (e.g., "always buy" in uptrending markets), **hyperparameters cannot prevent collapse**. The environment must enforce diversity through hard constraints.

---

### 9.2 Solution: 4 Environment-Level Constraints

All improvements implemented in:
- `core/rl/environments/trading_env.py`
- `core/rl/environments/reward_shaper.py`
- `core/rl/policies/symbol_agent.py`
- `training/config_templates/phase3_ppo_baseline.yaml`

**Full Documentation:** `docs/anti_collapse_improvements_2025-10-08.md`

---

### 9.3 Improvement #1: Action Repetition Limit

**Implementation:**
```python
# trading_env.py
self.max_consecutive_actions = 3  # Hard limit
self.consecutive_action_count = 0
self.action_history = []
self.action_diversity_window = deque(maxlen=50)

# In step()
if len(self.action_history) > 0 and self.action_history[-1] == action:
    self.consecutive_action_count += 1
else:
    self.consecutive_action_count = 1

# Enforce limit
if self.consecutive_action_count > self.max_consecutive_actions:
    logger.debug("Action repetition limit hit, forcing HOLD")
    action = TradeAction.HOLD
    self.consecutive_action_count = 1
```

**Impact:**
- Prevents 99%+ single-action collapse
- Forces minimum diversity of 3-5 actions
- Creates forced exploration when agent fixates

**Monitoring:**
```python
mlflow.log_metric("charts/consecutive_action_violations", violations)
```

---

### 9.4 Improvement #2: Action Diversity Bonus

**Implementation:**
```python
# reward_shaper.py
def _compute_diversity_bonus(self, diversity_info: Dict) -> float:
    """Reward using 3-5+ unique actions per 50 steps."""
    window = diversity_info.get("action_diversity_window", [])
    if len(window) < 10:
        return 0.0
    
    unique_actions = len(set(window))
    
    if unique_actions >= 5:
        return 0.3      # Excellent diversity
    elif unique_actions == 4:
        return 0.2      # Good diversity
    elif unique_actions == 3:
        return 0.1      # Acceptable diversity
    else:
        return 0.0      # Poor diversity
```

**Config:**
```yaml
reward_weights:
  diversity_bonus: 0.05  # 5% weight
```

**Impact:**
- Encourages natural exploration beyond forced minimum
- Rewards agents for using full action toolkit
- Synergizes with repetition limit

**Monitoring:**
```python
mlflow.log_metric("charts/diversity_bonus", avg_bonus)
mlflow.log_metric("charts/unique_actions_per_window", unique_count)
```

---

### 9.5 Improvement #3: ROI-Scaled PnL Rewards

**Problem:**  
Old system rewarded absolute profit equally regardless of capital deployed. $100 profit on $1,000 position (10% ROI) got same reward as $100 profit on $10,000 position (1% ROI).

**Implementation:**
```python
# reward_shaper.py
def _compute_pnl_reward(self, prev_equity, current_equity, 
                       trade_info, position_info=None):
    # ... compute base_reward ...
    
    # ROI Scaling
    if self.config.roi_multiplier_enabled and position_info:
        shares = position_info.get("shares", 0)
        entry_price = position_info.get("entry_price", 0)
        position_size = shares * entry_price
        
        if position_size > 0:
            roi = equity_change / position_size
            roi_multiplier = 1.0 + (roi * self.config.roi_scale_factor)
            roi_multiplier = max(0.5, min(roi_multiplier, 3.0))  # Clamp
            
            final_reward = base_reward * roi_multiplier
            return final_reward
    
    return base_reward
```

**Config:**
```yaml
reward_weights:
  roi_multiplier_enabled: true
  roi_scale_factor: 2.0  # 10% ROI → 1.2x reward
```

**Example:**
- **High ROI:** $100 profit on $1,000 position (10% ROI)
  - ROI multiplier: 1.0 + (0.10 × 2.0) = 1.2
  - Final reward: base × 1.2
  
- **Low ROI:** $100 profit on $10,000 position (1% ROI)
  - ROI multiplier: 1.0 + (0.01 × 2.0) = 1.02
  - Final reward: base × 1.02

**Impact:**
- Encourages capital-efficient trades
- Discourages oversized positions with low returns
- Aligns with real-world trading best practices

**Monitoring:**
```python
mlflow.log_metric("charts/roi_multiplier", avg_multiplier)
mlflow.log_metric("charts/position_size_vs_roi", scatter_plot)
```

---

### 9.6 Improvement #4: Stricter Action Masking

**Problem:**  
Agents could use BUY_SMALL/MEDIUM/LARGE when position already exists, accidentally doubling positions.

**Implementation:**
```python
# symbol_agent.py
class ActionMasker:
    def get_mask(self, observations):
        # ... existing logic ...
        
        if has_position.any():
            # Block BUY actions (indices 1, 2, 3)
            for idx in self._buy_indices.tolist():
                mask[has_position, idx] = False
            # ADD_POSITION (idx=6) remains allowed
            
            logger.debug("Blocked BUY actions for %d envs with positions",
                        has_position.sum().item())
```

**Impact:**
- Makes scaling-in intentional (must use ADD_POSITION)
- Prevents "spam BUY" strategies
- Encourages position management awareness

**Monitoring:**
```python
mlflow.log_metric("charts/action_masking_events", mask_count)
```

---

### 9.7 Configuration Summary

**Fees (Alpaca Realistic):**
```yaml
environment:
  commission_rate: 0.00002   # 0.002% = 2 bps
  slippage_pct: 0.0001       # 0.01% = 1 bp
```

**Entropy (Anti-Collapse):**
```yaml
ppo:
    ent_coef: 0.08
    entropy_scheduler:
        strategy: hold_then_linear
        initial: 0.08
        final: 0.03
        hold_steps: 10000
        decay_steps: 40000
        min: 0.03
    entropy_bonus:
        enabled: true
        target_entropy: 0.55
        bonus_scale: 0.35
        warmup_steps: 4000
        decay_rate: 0.12
        max_multiplier: 3.0
        floor: 0.02

action_entropy_guard:
    enabled: true
    threshold: 0.22
    warmup_steps: 4000
    boost_multiplier: 1.7
    max_multiplier: 3.0
    cooldown_steps: 6000
    halt_on_failure: true
```

**Reward Weights (Stage 5 Baseline):**
```yaml
reward_weights:
    pnl: 0.75
    cost: 0.10
    time: 0.0
    sharpe: 0.03
    drawdown: 0.02
    sizing: 0.0
    hold: 0.0
diversity_bonus: 0.07
action_repeat_penalty: 0.05
intrinsic_action_reward: 0.01
roi_multiplier_enabled: true
roi_scale_factor: 2.0
```

**Environment Constraints:**
```python
# trading_env.py (hardcoded)
max_consecutive_actions = 3      # Hard limit
action_diversity_window = 50     # Rolling window size
```

---

### 9.8 Testing & Validation

**Smoke Test (15k steps):**
```bash
python train_phase3_agents.py --symbols SPY --total-timesteps 15000
```

**Success Criteria:**
- ✅ Action diversity: Executed actions cover ≥4 unique actions per 120-step window (Stage 5 telemetry).
- ✅ `policy_action_entropy_mean`: ≥0.20 by step 6 k, no sustained dips <0.18.
- ✅ Repetition guard: No streaks >3 without penalty; guard events <5% of steps.
- ✅ Sharpe ratio: >-0.1 (better than -0.25 baseline) on smoke evaluation.
- ✅ Voluntary trade rate: ≥10% with sanitizer delta <15 pp.

**Monitoring Dashboards:**

*TensorBoard:*
- `charts/policy_action_entropy_mean` - Alert if <0.20 mid-run
- `charts/executed_vs_policy_trades` - Track sanitizer delta (<15 pp target)
- `charts/action_distribution` - Should show 4-5 actions sharing load (10-35%)
- `charts/consecutive_action_violations` - Rare blips (<5% of steps)
- `charts/diversity_bonus` - Average 0.08-0.15

*MLflow:*
- `policy_action_entropy_mean`
- `executed_action_entropy_mean`
- `voluntary_trade_rate_mean`
- `sanitizer_trade_delta_mean`
- `ep_total_trades_mean` - Should be 5-15 (not 0 or 100)
- `ep_unique_actions_mean` - Should be ≥4

**Validation Script:**
```bash
python validate_anti_collapse_improvements.py
```

Expected: All 5 tests pass ✅

---

### 9.9 Rollback Procedures

**If improvements cause new issues:**

**Disable diversity bonus:**
```yaml
diversity_bonus: 0.0
```

**Disable ROI scaling:**
```yaml
roi_multiplier_enabled: false
```

**Disable repetition limit (code change required):**
```python
# trading_env.py line 190
self.max_consecutive_actions = 999  # Effectively disabled
```

**Revert action masking (code change required):**
```python
# symbol_agent.py line 110-123
if has_position.any():
    pass  # Allow BUY actions
```

**Full rollback:**
```bash
git checkout HEAD -- training/config_templates/phase3_ppo_baseline.yaml
git checkout HEAD -- core/rl/environments/reward_shaper.py
git checkout HEAD -- core/rl/environments/trading_env.py
git checkout HEAD -- core/rl/policies/symbol_agent.py
```

---

### 9.10 Lessons Learned

**1. Hyperparameters Cannot Fix Environment Design Flaws**
- Tried 8 different configurations (actor gain 0.01→0.3, entropy 0.05→0.25, costs 0.01%→0.5%)
- NONE prevented policy collapse
- Root cause: Environment allowed infinite repetition of locally optimal action

**2. Discrete Action Spaces Need Explicit Diversity Mechanisms**
- Entropy regularization alone is insufficient
- Need BOTH:
  - Hard constraints (repetition limits)
  - Soft incentives (diversity bonuses)

**3. Transaction Cost Penalties Ineffective Against Profitable Degenerate Strategies**
- If "always buy" is genuinely profitable in uptrending markets, higher costs just make it slightly less profitable
- Need structural constraints to prevent learning the strategy

**4. Capital Efficiency Matters More Than Absolute Profit**
- ROI scaling aligns RL objectives with real-world trading goals
- Encourages better risk-adjusted returns
- Prevents agents from learning "big position = big profit" heuristic

---

### 9.11 Integration with Existing Training

**These improvements are NOW THE BASELINE for all Phase 3 training.**

**Updated Training Workflow:**

1. **Start Training:**
   ```bash
   python train_phase3_agents.py --config training/config_templates/phase3_ppo_baseline.yaml
   ```

2. **Monitor Anti-Collapse Metrics:**
   - Check `charts/action_distribution` every 5k steps
   - Verify entropy stays >0.8
   - Confirm diversity bonus averaging 0.1-0.2

3. **Early Warning System:**
   - If action distribution >80% single action → ALERT
   - If entropy <0.3 → ALERT
   - If consecutive violations >50/episode → ALERT

4. **Debugging Checklist:**
   - [ ] Verify `max_consecutive_actions = 3` in trading_env.py
   - [ ] Check `diversity_bonus_weight = 0.05` in config
   - [ ] Confirm `roi_multiplier_enabled = true` in config
   - [ ] Validate action masking logs: "blocked BUY actions"

**All agents trained from 2025-10-08 onward MUST use these constraints.**

---

### 9.12 References

- **Implementation:** `docs/anti_collapse_improvements_2025-10-08.md`
- **Quick Reference:** `ANTI_COLLAPSE_QUICK_REFERENCE.md`
- **Validation Script:** `validate_anti_collapse_improvements.py`
- **Root Cause Analysis:** Policy collapse incident (99.88% BUY_SMALL, 0.007 entropy)
- **Design Decision:** Environment constraints > hyperparameter tuning

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