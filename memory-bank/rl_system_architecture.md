# Multi-Agent Reinforcement Learning Trading System Architecture

**Document Version:** 1.0  
**Date:** 2025-10-05  
**Status:** Approved for Implementation Planning

## Executive Summary

This document defines the architecture for a hierarchical multi-agent reinforcement learning (RL) trading system designed to replace the underperforming supervised learning (SL) stack. The architecture integrates pre-trained SL classifiers as auxiliary signals, a three-tiered agent hierarchy for coordinated decision-making, and a multi-objective reward structure that internalizes transaction costs, drawdowns, and capital efficiency. The design targets sustainable profitability, adaptive regime management, and controlled risk exposure across a 143-symbol universe.

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
| Exploration Blowup | 143 agents exploring simultaneously may trigger destabilizing trades. | Use entropy annealing, action masking under master directives, and staged symbol onboarding (curriculum). |
| Regime Non-Stationarity | Market regime shifts could invalidate policies. | Implement regime detection module feeding master agent; schedule quarterly fine-tunes; maintain rapid retraining pipeline. |
| Computational Overhead | MAPPO with 143 agents may strain resources. | Employ parameter sharing, mixed precision training, and prioritized rollout batching; scale horizontally with Ray cluster. |
| Compliance & Risk Limits | Autonomous agents may breach risk constraints. | Hard constraints enforced in environment (max leverage, position caps); monitor via real-time guardrails and kill-switch. |

## Interfaces & Telemetry

- **Event Bus:** Kafka topics for agent actions, state snapshots, and master directives during live trading.
- **Metrics:** Per-agent reward, Sharpe, turnover, drawdown; aggregated dashboards in Grafana.
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
