# Research Report: Strategic Pivot from Supervised Learning to Multi-Agent RL

**Document Version:** 1.0  
**Date:** 2025-10-04  
**Author:** Trading Intelligence Team

## Executive Summary

Backtests conducted on the supervised learning (SL) ensemble (MLP72, LSTM62, GRU93) over 2023-10-02 → 2025-10-01 produced catastrophic losses (-88% to -93%) with Sharpe ratios near zero and drawdowns exceeding 91%. The root causes stem from structural limitations of static classifiers: label myopia, cost blindness, sequential decision gaps, and severe regime overfitting. This report formalizes the negative expectancy, diagnoses failure modes, and defines the remedial strategy—a hierarchical multi-agent reinforcement learning (RL) system that coordinates portfolio risk and symbol-level execution while leveraging the existing SL checkpoints as contextual priors.

## Failure Analysis

### Quantitative Outcomes

| Metric | SL Ensemble (Backtest) | SPY Buy-Hold |
|--------|------------------------|--------------|
| Total Return | -88% to -93% | +59.9% |
| Sharpe Ratio | -0.02 to -0.04 | 1.47 |
| Max Drawdown | >91% | -17% |
| Win Rate | 44% | — |
| Profit Factor | 0.70 | — |
| Trades | 8.5k – 11.4k | ~500 (passive) |

### Root Causes

1. **Label Quality Illusion:** Binary labels captured directional bias but ignored timing, certainty, and regime context. Models chased stale signals without regard for price path.
2. **Transaction Cost Devastation:** High frequency of trades under 15 bps round-trip costs yielded structural losses. Expected per-trade value:

$$
\mathbb{E}[r] = 0.44 \times 0.022 - 0.56 \times 0.021 - 0.0015 = -0.00358.
$$

Across 10,000 trades, the expected loss is $-35.8\%$ before slippage—consistent with observed drawdowns.
3. **Sequential Decision Gap:** SL treated each timestamp independently, incapable of managing hold durations, scale-ins, exits, or cross-symbol allocation.
4. **Distribution Shift:** Training distribution skewed to bull markets; models memorized patterns that failed in neutral/bear regimes.
5. **Risk Neglect:** No notion of capital allocation, drawdown control, or diversification existed in the SL inference layer.

## Why RL Solves the Problem

| Challenge | SL Limitation | RL Remedy |
|-----------|---------------|-----------|
| Timing | Static threshold decisions | Policies optimize sequences with temporal credit assignment |
| Sizing | Fixed post-threshold sizing | Action space includes position scaling and pyramiding |
| Costs | Post-hoc penalties | Reward function embeds transaction cost penalties |
| Regime Adaptation | No context awareness | Master agent learns regime-conditioned risk overlays |
| Portfolio Coordination | Independent symbol calls | Hierarchical agents coordinate capital and diversification |
| Learning Signal | Binary labels | Dense multi-objective reward incorporating P&L, risk, utilization |

## Proposed RL System (Summary)

- **Hybrid Observations:** Integrate SL probabilities with raw features, macro indicators, and portfolio context.
- **Symbol Agents:** PPO-based actor-critics with shared encoders, discrete action space for buy/sell sizing decisions.
- **Master Agent:** MAPPO coordinator adjusting risk multipliers, position caps, and regime modes.
- **Reward Shaping:** Multi-term objective balancing P&L, cost penalties, drawdown control, timing efficiency, and diversification.
- **Training Roadmap:** 12-week plan covering environment construction, symbol PPO training, master MAPPO training, validation, and deployment.

## Implementation Roadmap

| Phase | Timeline | Milestones |
|-------|----------|------------|
| Phase 1 | Weeks 1-2 | Gym-compatible environment, vectorized rollouts, smoke tests |
| Phase 2 | Weeks 3-6 | Shared encoder pre-training, symbol agent PPO training with curriculum |
| Phase 3 | Weeks 7-8 | Master agent MAPPO, coordinated fine-tuning |
| Phase 4 | Weeks 9-10 | Walk-forward validation, stress testing, benchmark comparisons |
| Phase 5 | Weeks 11-12 | Deployment hardening, monitoring, risk controls |

Dependencies: GPU cluster provisioning, feature fusion pipeline, MLflow + Optuna infrastructure, data quality assurance.

## Innovations Detailed

1. **Hybrid SL+RL Feature Interface:** Reuse SL classifiers as prior signals; RL agents learn when to amplify, ignore, or counteract them.
2. **Shared Encoder with Symbol Heads:** Transfer temporal and cross-symbol knowledge while allowing symbol-specific tactics.
3. **Multi-Objective Reward Shaping:** Embeds costs, timing efficiency, drawdown penalties, and diversification incentives to counter prior failure modes.
4. **Hierarchical Master Agent:** Portfolio-level MAPPO ensures coordinated risk posture and diversification, resolving sequential decision gaps.
5. **Curriculum & Regime Training:** Stage agents through increasing complexity, mixing regime scenarios to combat distribution shift.

## Risk Analysis & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Reward Mis-specification | Agents optimize unintended behaviors | Iterative calibration, component monitoring, human review of trajectories |
| Overfitting to Backtest | RL memorizes historical quirks | Walk-forward validation, synthetic stress scenarios, regular retraining |
| Computational Bottlenecks | Training stalls due to scale | Ray-based distributed rollouts, mixed precision, experience replay compression |
| Live Trading Failures | Unexpected agent actions | Action masking, kill-switch guardrails, human-in-the-loop oversight |
| Data Drift | Feature distributions shift post-deployment | Real-time drift detectors, scheduled retraining, hybrid fallback strategies |

## Alternative Quick Wins (Parallel Track)

1. **Threshold Optimization:** Optimize SL thresholds over 0.60–0.90 range using rapid backtests.
2. **Top Decile Filter:** Trade only top 10% confidence signals to curb churn and improve precision.
3. **Regime-Aware Filters:** Disable trading during high-VIX or bear regimes.

These experiments run in parallel, offering potential mitigation (e.g., reducing losses to -50% → -30%) while RL system matures. They also provide baselines to measure RL uplift.

## Expected Outcomes

| Scenario | Total Return (2 yrs) | Sharpe | Max DD | Win Rate | Profit Factor |
|----------|----------------------|--------|--------|----------|---------------|
| Minimum Viable | ≥ +15% | >0.8 | <30% | >48% | >1.1 |
| Target | ≥ +30% | >1.2 | <25% | >52% | >1.4 |
| Stretch | ≥ +50% | >1.5 | <20% | >55% | >1.6 |

## Go / No-Go Criteria

- **Go:** RL prototype achieves target metrics on walk-forward validation; master agent reduces drawdown vs symbol-only baseline; deployment guardrails and monitoring validated.
- **No-Go:** RL fails to beat SL quick wins, reward instability persists, or transaction costs remain unmanageable. Trigger redesign of reward structure, expand quick-win heuristics, or explore alternative RL algorithms (e.g., SAC, distributional RL).

## Lessons Learned

- **Labels ≠ Actions:** Financial decision-making requires sequential context; single-step classifications are insufficient.
- **Costs Must Be First-Class:** Without embedded cost awareness, accurate predictions still fail.
- **Hybridization Matters:** Reusing SL models accelerates RL training and preserves prior investments.
- **Risk Coordination is Essential:** Portfolio-level control mechanisms guard against agent proliferation and drawdown cascades.

## Next Steps

1. Approve RL architecture and training guide for implementation kickoff (Week 1).
2. Allocate GPU resources and engineering bandwidth for RL environment build.
3. Launch quick-win SL experiments for interim mitigation and benchmarking.
4. Schedule weekly steering reviews to monitor progress and adjust roadmap.
