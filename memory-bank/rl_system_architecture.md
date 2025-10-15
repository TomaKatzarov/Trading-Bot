# Multi-Agent Reinforcement Learning Trading System Architecture

**Document Version:** 1.2  
**Date:** 2025-10-05 (Updated: 2025-10-09)  
**Status:** Approved for Implementation Planning

**CRITICAL UPDATE (2025-10-08):** Anti-collapse improvements added to prevent catastrophic policy failures. See Section "Anti-Collapse Architecture" for environment-level constraints that are now mandatory for all agent training.

## Executive Summary

This document defines the architecture for a hierarchical multi-agent reinforcement learning (RL) trading system designed to replace the underperforming supervised learning (SL) stack. The architecture integrates pre-trained SL classifiers as auxiliary signals, a three-tiered agent hierarchy for coordinated decision-making, and a multi-objective reward structure that internalizes transaction costs, drawdowns, and capital efficiency. The design targets sustainable profitability, adaptive regime management, and controlled risk exposure across a 143-symbol universe.

**Anti-Collapse Update (2025-10-08 → 2025-10-09):** After experiencing catastrophic policy collapse (99.88% single-action convergence within 10k steps), the architecture now mandates environment-level diversity constraints that supersede hyperparameter-only approaches. Follow-up work on 2025-10-09 added action-voluntary telemetry, entropy guards, and reward balancing baselines required for Stage 5 validation.

## Phase 3 Stage Rollup (as of 2025-10-09)

| Stage | Focus | Status | Baseline / Verification |
|-------|-------|--------|-------------------------|
| Stage 1 | Exploration recovery (actor gain 0.1, entropy coefficient schedule, reward normalization disabled) | ✅ Completed 2025-10-08 | `analysis/validate_exploration_fix.py` green; smoke-run requirement tracked in training monitor: action entropy ≥1.3 with ≥4 distinct actions per 120-step window (last verified 2025-10-08) |
| Stage 2 | Professional reward stack (realized PnL only, exit multipliers, pyramiding guards) | ✅ Completed 2025-10-08 | Professional strategy metrics in `rl_training_guide.md` (Sharpe +0.563, win rate 64.5%, max DD 0.45%) |
| Stage 3 | Anti-collapse hard constraints (diversity bonus, action-repeat penalty, ROI scaling, SELL mask audit) | ✅ Completed 2025-10-08 | `tests/test_reward_shaper.py` 42/42 pass on 2025-10-09; SELL availability confirmed via symbol-agent mask diagnostics |
| Stage 4 | Telemetry & validation harness (policy vs executed trades, entropy guard, Stage 5 CLI) | ✅ Completed 2025-10-09 | `scripts/run_phase3_stage5_validation.py` emits new metrics (`policy_action_entropy_mean`, `voluntary_trade_rate`, sanitizer deltas) and enforces entropy threshold 0.22 |
| Stage 5 | Cross-seed short retrain + telemetry diff (5–10 k steps per seed 42/1337/9001) | 🚧 Pending execution (scheduled next) | Command staged; success criteria: mid-run `policy_action_entropy_mean ≥ 0.20`, voluntary trades ≥ 10%, sanitizer deltas converging |

## Strategic Objectives

- **Restore Positive Expectancy:** Achieve $>15\%$ total return with Sharpe $>0.8$ in the first production release.
- **Reduce Drawdowns:** Maintain max drawdown below $25\%$ while sustaining risk-adjusted returns.
- **Leverage Existing Assets:** Reuse SL checkpoints as feature priors to accelerate RL convergence.
- **Enable Portfolio Coordination:** Manage symbol-level autonomy under portfolio-level risk constraints.
- **Support Continuous Learning:** Provide modular subsystems for frequent retraining and regime adaptation.

## Design Principles

1. **Hierarchy over Monolith:** Decompose the trading task into portfolio allocation and symbol-level execution to exploit specialization while preserving centralized risk control.
2. **Shared Intelligence:** Use shared encoders and hybrid SL+RL signals to transfer knowledge across agents and symbols.
3. **Cost-Aware Optimization:** Embed transaction costs, slippage, and drawdown penalties directly in the reward.
4. **Vectorized Efficiency:** Build a highly parallel training environment capable of simulating tens of thousands of steps per hour across GPUs.
5. **Observability First:** Instrument every agent and subsystem for introspection, explainability, and post-trade forensics.
6. **Diversity by Design (NEW 2025-10-08):** Enforce action diversity through hard environment constraints (repetition limits, masking rules) to prevent degenerate single-action policies that hyperparameters alone cannot address.

## System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│ Tier 1: Master Agent (Portfolio Manager)                             │
│  • Regime adaptation, capital allocation, global risk constraints     │
└───────────────┬─────────────────────────────┬────────────────────────┘
                │                             │
┌───────────────▼─────────────┐   ...   ┌─────▼───────────────┐
│ Tier 2: Symbol Agent (AAPL) │         │ Tier 2: Symbol ...  │
│  • Entry/exit, sizing, exits │         │  (MSFT, etc.)       │
└───────────────┬─────────────┘         └──────────┬──────────┘
                │                                    │
           ┌────▼────────────────────────────────────▼────┐
           │ Tier 3: Shared Feature Encoders              │
           │  • Transformer, LSTM, CNN, Graph encoders    │
           │  • Hybrid SL+RL observation fusion           │
           └──────────────────────────────────────────────┘
```

## Component Detailing

### Tier 1: Master Agent

- **Role:** Portfolio-level coordinator that sets global risk posture, position limits, and capital deployment levers.
- **Observation Channels:** Aggregated symbol agent intents, portfolio telemetry, market regime diagnostics.
- **Action Space:** Discrete parameter grid controlling risk multiplier, per-position caps, open position limits, and regime modes (`defensive`, `neutral`, `aggressive`).
- **Learning Algorithm:** Multi-Agent Proximal Policy Optimization (MAPPO) with centralized value function on portfolio state.

```python
class MasterAgent(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU()
        )
        self.actor = nn.Linear(128, 4 * 4 * 3)  # flattened discrete grid
        self.critic = nn.Linear(128, 1)
```

### Tier 2: Symbol Agents

- **Population:** 143 independent agents, one per tradable symbol.
- **Shared Backbone:** Transformer encoder (multi-head attention) pre-trained on unlabeled historical sequences with masked forecasting objective.
- **Head Architecture:** Symbol-specific actor-critic heads operating on the shared embedding.
- **Action Space:** Seven discrete actions (`HOLD`, `BUY_SMALL`, `BUY_MEDIUM`, `BUY_LARGE`, `SELL_PARTIAL`, `SELL_ALL`, `ADD_TO_WINNER`).
- **Coordination:** Symbol agents receive master directives (risk multiplier, caps) as part of their observation.

```python
class SymbolAgent(nn.Module):
    def __init__(self, shared_encoder, action_dim=7):
        super().__init__()
        self.encoder = shared_encoder
        self.actor_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, obs):
        embedding = self.encoder(obs)
        logits = self.actor_head(embedding)
        value = self.critic_head(embedding)
```

### Tier 3: Shared Feature Encoders

- **Modality Fusion:**
  - **Temporal module:** Bidirectional LSTM for micro-pattern capture.
  - **Transformer module:** Multi-scale attention across 24×23 technical inputs.
  - **CNN module:** 1-D convolutions emphasizing local momentum signatures.
  - **Graph module:** Message passing across correlation-based asset graph to propagate cross-symbol signals.
- **Hybrid SL+RL Inputs:** Appends probabilities from `MLP72`, `LSTM62`, `GRU93` SL checkpoints; RL learns to interpret them as priors.
- **Output:** 256-dimensional embedding consumed by symbol agents and master agent summarizers.

### Observation Model

```
Observation Dim ≈ 512
├── Technical Window (24h × 23 features)
├── Sentiment + Macro Indicators
├── SL Probability Triplet (MLP/LSTM/GRU)
├── Position State (entry, P&L, duration, size)
├── Portfolio Telemetry (cash %, drawdown, Sharpe)
├── Market Regime Features (VIX, trend, breadth, correlation)
├── Master Directives (risk multiplier, caps)
```

### Reward Design

Reward integrates micro- and macro-level targets via the configurable `RewardShaper`
module (`core/rl/environments/reward_shaper.py`). Component weights are defined in
`RewardConfig` and validated each episode using `scripts/analyze_reward_signals.py`
to prevent dominance or collapse. For symbol agent $i$ at time $t$:

$$
\begin{aligned}
R^{(i)}_t &= 10 \cdot r^{(i)}_{\text{pnl}}
+ 5 \cdot r^{(i)}_{\text{cost}}
+ 2 \cdot r^{(i)}_{\text{timing}}
+ 3 \cdot r^{(i)}_{\text{sharpe}}
+ 5 \cdot r^{(i)}_{\text{drawdown}}
+ 0.5 \cdot r^{(i)}_{\text{util}}
+ 1 \cdot r^{(i)}_{\text{diversification}}
\end{aligned}
$$

Where:
- $r^{(i)}_{\text{pnl}}$ normalizes realized P&L by initial capital.
- $r^{(i)}_{\text{cost}}$ penalizes transaction costs at $15$ bps per round trip.
- $r^{(i)}_{\text{drawdown}}$ applies multiplicative penalties when intra-trade drawdown exceeds $10\%$.

Master agent reward emphasizes risk-adjusted return, diversification, and excess performance over SPY benchmark.

### Environment Architecture

- **Core Module:** `core/rl/trading_env.py`
  - Implements OpenAI Gymnasium API with vectorized `reset()` and `step()`.
  - Supports batched symbol execution and GPU-accelerated feature stacking.
  - Integrates transaction cost model, borrow fees, and slippage simulator.
- **Vectorized Wrapper:** `core/rl/vector_env.py` enabling $N$ parallel market scenarios for faster sample collection.
- **Data Sources:** Consolidated historical parquet datasets, sentiment attachments, macro features, and pre-generated SL signal caches.

### Data & Model Flow

```
Historical Market Data ─┐
Sentiment & Macro ------┼─► Feature Fusion Layer ─► Shared Encoder ─► Agent Policies
SL Model Probabilities ┘

Agent Actions ─► Trading Simulator ─► Rewards & State Updates ─► Replay Buffers

Replay Buffers ─► PPO/MAPPO Learners ─► Policy Weights ─► Checkpoints & Serving Artifacts
```

## Hybrid SL + RL Integration

- **Inference Cache:** SL checkpoints now staged under `models/sl_checkpoints/<model>/` with verified <0.2 ms/sample inference (`reports/sl_inference_benchmarks.json`); probabilities are refreshed hourly and cached as feature tensors for RL consumption.
- **Auxiliary Losses:** During RL training, add Kullback–Leibler divergence regularizer encouraging alignment with SL predictions early in training, annealed over epochs.
- **Warm Starts:** Initialize actor heads with regression weights that approximate SL decision boundaries to reduce exploration burn-in.

## Scalability & Performance

- **Training Cluster:** 8× GPUs with distributed rollout workers (Ray RLlib or CleanRL with custom vector env).
- **Throughput Target:** 128 parallel environments × 2048-step rollouts ⇒ ~262k transitions per update cycle.
- **Latency Targets:** <20 ms inference latency for symbol agents; <50 ms for master agent adjustments.

## Risk & Mitigation

| Risk | Description | Mitigation |
|------|-------------|------------|
| Reward Misalignment | Over- or under-weighted reward terms could reintroduce churn or excessive risk. | Conduct reward coefficient ablations; monitor component-level contributions via `RewardShaper.get_episode_stats()` and `scripts/analyze_reward_signals.py`; integrate human-in-the-loop review for early episodes. |
| **Policy Collapse (CRITICAL)** | **Agents converge to degenerate single-action policies (>95% one action) within early training, immune to hyperparameter fixes.** | **MANDATORY: Environment-level constraints (max 3 consecutive actions, diversity bonuses, ROI-scaled rewards, stricter masking). See Anti-Collapse Architecture section below.** |
| Exploration Blowup | 143 agents exploring simultaneously may trigger destabilizing trades. | Use entropy annealing, action masking under master directives, and staged symbol onboarding (curriculum). |
| Regime Non-Stationarity | Market regime shifts could invalidate policies. | Implement regime detection module feeding master agent; schedule quarterly fine-tunes; maintain rapid retraining pipeline. |
| Computational Overhead | MAPPO with 143 agents may strain resources. | Employ parameter sharing, mixed precision training, and prioritized rollout batching; scale horizontally with Ray cluster. |
| Compliance & Risk Limits | Autonomous agents may breach risk constraints. | Hard constraints enforced in environment (max leverage, position caps); monitor via real-time guardrails and kill-switch. |

---

## Anti-Collapse Architecture (MANDATORY - Added 2025-10-08)

### Context: Policy Collapse Incident

**Date:** October 8, 2025  
**Symptom:** Agents converged to 99.88% BUY_SMALL action within 10k steps, entropy collapsed to 0.007  
**Root Cause:** Discrete action environments with locally optimal degenerate strategies (e.g., "always buy" in uptrends) cannot be stabilized by hyperparameters alone  
**Failed Approaches:** 8 hyperparameter configurations (actor gain 0.01→0.3, entropy 0.05→0.25, costs 0.01%→0.5%)  
**Solution:** Environment-level hard constraints that enforce diversity independent of learning dynamics

### Mandatory Architectural Components

All RL training environments MUST implement the following 4 constraints:

#### 1. Action Repetition Limiter (Trading Environment)

**Implementation Location:** `core/rl/environments/trading_env.py`

```python
class TradingEnvironment(gym.Env):
    def __init__(self, ...):
        # MANDATORY: Action diversity tracking
        self.max_consecutive_actions = 3  # Hard limit
        self.consecutive_action_count = 0
        self.action_history = []
        self.action_diversity_window = deque(maxlen=50)
    
    def step(self, action):
        # MANDATORY: Enforce repetition limit
        if len(self.action_history) > 0 and self.action_history[-1] == action:
            self.consecutive_action_count += 1
        else:
            self.consecutive_action_count = 1
        
        # Override to HOLD if limit exceeded
        if self.consecutive_action_count > self.max_consecutive_actions:
            logger.warning("Action repetition limit violated, forcing HOLD")
            action = TradeAction.HOLD
            self.consecutive_action_count = 1
        
        # Update tracking
        self.action_history.append(action)
        self.action_diversity_window.append(action)
```

**Rationale:** Prevents agents from fixating on single action regardless of profitability. Creates forced exploration when convergence detected.

#### 2. Diversity Bonus Reward Component (Reward Shaper)

**Implementation Location:** `core/rl/environments/reward_shaper.py`

```python
class RewardShaper:
    def _compute_diversity_bonus(self, diversity_info: Dict) -> float:
        """MANDATORY: Reward using 3-5+ unique actions per 50-step window."""
        window = diversity_info.get("action_diversity_window", [])
        if len(window) < 10:
            return 0.0
        
        unique_actions = len(set(window))
        
        # Tiered bonuses
        if unique_actions >= 5:
            return 0.3      # Excellent diversity
        elif unique_actions == 4:
            return 0.2      # Good diversity
        elif unique_actions == 3:
            return 0.1      # Acceptable diversity
        else:
            return 0.0      # Collapse warning
```

**Configuration:** Set `diversity_bonus_weight: 0.05` (5% of total reward)

**Rationale:** Soft incentive to complement hard repetition limit. Rewards natural exploration.

#### 3. ROI-Scaled PnL Rewards (Reward Shaper)

**Implementation Location:** `core/rl/environments/reward_shaper.py`

```python
def _compute_pnl_reward(self, prev_equity, current_equity, 
                       trade_info, position_info):
    # Compute base reward
    base_reward = equity_change / self.pnl_scale
    
    # MANDATORY: ROI scaling for capital efficiency
    if self.config.roi_multiplier_enabled and position_info:
        position_size = position_info["shares"] * position_info["entry_price"]
        if position_size > 0:
            roi = equity_change / position_size
            roi_multiplier = 1.0 + (roi * self.config.roi_scale_factor)
            roi_multiplier = np.clip(roi_multiplier, 0.5, 3.0)
            return base_reward * roi_multiplier
    
    return base_reward
```

**Configuration:**
```yaml
roi_multiplier_enabled: true
roi_scale_factor: 2.0  # 10% ROI → 1.2x reward
```

**Rationale:** Discourages "big position = big reward" heuristic. Aligns with real-world capital efficiency goals.

#### 4. Stricter Action Masking (Policy Network)

**Implementation Location:** `core/rl/policies/symbol_agent.py`

```python
class ActionMasker(nn.Module):
    def get_mask(self, observations):
        # ... existing logic ...
        
        # MANDATORY: Block BUY actions when position exists
        if has_position.any():
            for idx in [1, 2, 3]:  # BUY_SMALL, BUY_MEDIUM, BUY_LARGE
                mask[has_position, idx] = False
            # Only ADD_POSITION (idx=6) allowed for pyramiding
```

**Rationale:** Makes position scaling intentional. Prevents "spam BUY" collapse mode.

### Integration Requirements

**All new RL environments MUST:**
1. Inherit `TradingEnvironment` with repetition tracking
2. Use `RewardShaper` with diversity bonus enabled
3. Use `SymbolAgent` with stricter action masking
4. Activate the Stage 5 exploration stack: entropy scheduler (initial 0.08 → floor 0.03), adaptive entropy bonus, and action-entropy guard (threshold 0.22, warmup 4 k steps)

**Configuration Template:**
```yaml
# training/config_templates/*.yaml
ppo:
    ent_coef: 0.08        # Initial exploration weight (Stage 5 baseline)
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
    action_entropy_guard:
        enabled: true
        threshold: 0.22
        warmup_steps: 4000

reward_weights:
    pnl: 0.75             # Profit remains dominant but not overwhelming
    cost: 0.10            # Keeps churn in check without crushing reward
    sharpe: 0.03
    drawdown: 0.02
diversity_bonus: 0.07   # MANDATORY (Stage 3 baseline)
action_repeat_penalty: 0.05
intrinsic_action_reward: 0.01
roi_multiplier_enabled: true
roi_scale_factor: 2.0
```

### Validation Checklist

Before deploying any RL agent to training:

- [ ] `validate_anti_collapse_improvements.py` (or Stage 5 subset) passes all active tests
- [ ] Action masking logs show "blocked BUY actions" or "SELL restored" events during warmup
- [ ] Diversity bonus averages ≥0.08 over a 5 k-step smoke run
- [ ] `policy_action_entropy_mean` ≥0.20 by step 6 k, never <0.18 afterwards without triggering guard intervention
- [ ] Voluntary trade rate ≥0.10 and sanitizer deltas <15 pp difference in Stage 5 evaluation sweeps
- [ ] No single executed action exceeds 60% distribution in smoke run histograms

### Monitoring & Alerts

**TensorBoard Dashboards (MANDATORY):**
- `charts/policy_action_entropy_mean` - Alert if <0.20 mid-run
- `charts/executed_vs_policy_trades` - Track sanitizer delta (target <15 pp)
- `charts/action_distribution` - Alert if any action >80%
- `charts/consecutive_action_violations` - Expect light activity (≤5% of steps)
- `charts/diversity_bonus` - Target average 0.08-0.15

**MLflow Metrics (MANDATORY):**
- `policy_action_entropy_mean`
- `executed_action_entropy_mean`
- `voluntary_trade_rate_mean`
- `sanitizer_trade_delta_mean`
- `ep_unique_actions_mean`
- `ep_diversity_bonus_mean`

### References

- **Full Documentation:** `docs/anti_collapse_improvements_2025-10-08.md`
- **Quick Reference:** `ANTI_COLLAPSE_QUICK_REFERENCE.md`
- **Training Guide Update:** `memory-bank/PHASE_3_TRAINING_STRATEGY.md` Section 9
- **Validation Script:** `validate_anti_collapse_improvements.py`

---

## Interfaces & Telemetry

- **Event Bus:** Kafka topics for agent actions, state snapshots, and master directives during live trading.
- **Metrics:** Per-agent reward, Sharpe, turnover, drawdown; aggregated dashboards in Grafana.
- **Stage 5 Telemetry (NEW 2025-10-09):** Environment now surfaces `policy_action_entropy_mean`, `executed_action_entropy_mean`, `voluntary_trade_rate`, `sanitizer_trade_delta`, and per-episode counts for selected vs executed voluntary trades. Logged through Stage 5 validation runner and MLflow autologging hooks.
- **Diagnostics:** Saliency maps for encoder attention, SL vs RL decision comparisons, rule-based anomaly detectors.

## Alignment with Roadmap

| Phase | Deliverable | Architectural Focus |
|-------|-------------|---------------------|
| Week 1-2 | Environment MVP | Implement `TradingEnvironment`, vectorized rollouts, reward plumbing |
| Week 3-6 | Symbol Agent PPO | Shared encoder pre-training, policy head training, curriculum scheduler |
| Week 7-8 | Master Agent MAPPO | Aggregation pipelines, centralized value function |
| Week 9-10 | Validation Suite | Walk-forward evaluation, stress testing toolkit |
| Week 11-12 | Deployment Prep | Serving stack, monitoring integrations, risk guardrails |

## Go / No-Go Criteria

- **Go:** Vectorized environment validated; symbol agents achieve Sharpe $>0.6$ on sandbox walk-forward; reward diagnostics stable; master agent reduces portfolio drawdown $>15\%$ vs baseline.
- **No-Go:** Reward gradients collapse; agents ignore cost penalties; MAPPO training diverges; compute costs exceed budget without meeting performance targets.

## References

- Schulman et al., "Proximal Policy Optimization Algorithms"
- Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
- Zhang & Yang, "An Overview of Multi-Agent Reinforcement Learning from Game Theoretical Perspective"
