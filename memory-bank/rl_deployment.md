# Multi-Agent RL Deployment Guide

**Document Version:** 1.0  
**Date:** 2025-10-04  
**Status:** Draft for Production Planning (Phase 5)

## Overview

This guide outlines the procedures, infrastructure, and operational safeguards required to deploy the hierarchical multi-agent reinforcement learning (RL) trading system into production. It covers serving architecture, real-time data integration, execution stack, monitoring, risk controls, and playbooks for incident response.

## Deployment Objectives

- **Low-Latency Decisions:** Maintain end-to-end action latency <150 ms for symbol agents and <300 ms for master adjustments.
- **Deterministic Reproducibility:** Ensure that production policies are versioned, checksum-verified, and reproducible from source artifacts.
- **Operational Safety:** Enforce strict risk guardrails and automated kill switches across all layers.
- **Observability:** Provide full-stack telemetry from data ingestion to trade execution and post-trade analytics.

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Data Plane                                                                 │
│  • Market Data Ingestors (WebSocket + REST backfill)                        │
│  • Sentiment + Macro Feeds (batch every hour)                               │
└───────────┬────────────────────────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────────────┐
│ Feature Service                                          │
│  • Real-time feature vector assembly                     │
│  • SL signal refresh (every bar)                         │
│  • Regime detector (VIX, breadth, trend)                 │
└───────────┬──────────────────────────────────────────────┘
            │ gRPC
┌───────────▼────────────────────────┐   ┌───────────────────┐
│ Inference Cluster (GPU)            │   │ Policy Registry    │
│  • Symbol Agent Serving Pods       │   │  • Model catalog   │
│  • Master Agent Service            │   │  • Version control │
└───────────┬────────────────────────┘   └─────────┬─────────┘
            │ REST / gRPC                            │
┌───────────▼────────────────────────┐              │
│ Execution Gateway                  │◄─────────────┘
│  • Order router (FIX/Alpaca API)   │
│  • Risk & compliance checks        │
└───────────┬────────────────────────┘
            │
┌───────────▼────────────────────────┐
│ Post-Trade & Monitoring            │
│  • Prometheus / Grafana dashboards │
│  • Trade reconciliation            │
│  • Alerting (PagerDuty, Slack)     │
└────────────────────────────────────┘
```

## Serving Stack

### Model Registry
- Store TorchScript (or ONNX) exports under `models/rl/production/<date>/<agent>.ptc`.
- Maintain metadata: git commit hash, training dataset identifiers, evaluation metrics, signature schema.
- Use MLflow Model Registry or custom Postgres-backed registry with checksum validation.

### Inference Services
- **Symbol Agent Service:**
  - Deployed as horizontally scalable pods (Kubernetes or Ray Serve) with GPU affinity.
  - Batch incoming feature requests per timestep to leverage parallel GPU execution.
  - Responds with discrete action probabilities and recommended action.

- **Master Agent Service:**
  - Stateful service updating risk directives every 5 minutes or when regime shifts are detected.
  - Communicates directives via Redis pub/sub or Kafka topic `master-directives`.

### Feature Service
- Aggregates market data, sentiment, macro indicators, portfolio state, and cached SL signals.
- Implements `HybridRLObservation` pipeline to ensure observation parity with training.
- Includes feature drift detection (Kolmogorov–Smirnov tests, mean/variance tracking).

## Execution Pipeline

1. **Feature Assembly:** At each decision interval, `feature_service` builds observation tensors per symbol.
2. **Symbol Decisions:** Observation passes to symbol agent service; action chosen with risk masks applied.
3. **Master Overlay:** Master agent directives adjust or veto symbol actions (e.g., enforce `max_position_size`).
4. **Risk Checks:** Execution gateway applies deterministic checks (leverage, capital, compliance rules).
5. **Order Routing:** Approved orders dispatched to broker/exchange via FIX/REST with transaction cost estimates.
6. **Post-Trade Logging:** Executions appended to audit trail; real fills update portfolio state.

## Risk Management

| Layer | Control | Description |
|-------|---------|-------------|
| Data | Feed validation | Reject stale or inconsistent data; redundant providers with quorum vote |
| Feature | Drift alarms | Halt trading if key feature distributions deviate >4σ from training |
| Agent | Action masking | Disallow trades violating risk directives or portfolio constraints |
| Portfolio | Hard guardrails | Enforce max gross leverage 1.5×, max cash drawdown 30%, per-symbol cap 10% equity |
| Execution | Kill switch | Immediate disable if latency spikes, P&L drawdown triggers, or compliance breach |
| Oversight | Human supervision | Real-time dashboards + manual override console |

## Monitoring & Alerts

### Metrics
- **Latency:** P50/P99 inference latency per agent; action-to-order latency.
- **Performance:** Rolling Sharpe, win rate, profit factor, turnover.
- **Risk:** Drawdown, exposure by sector, divergence from SPY benchmark.
- **Data Quality:** Missing bar counts, feature drift scores, SL prediction parity.

### Alert Thresholds
- Latency >250 ms (symbol agent) or >400 ms (master) for >3 consecutive intervals.
- Portfolio drawdown exceeds 15% intraday.
- Transaction cost ratio >40% of gross profit over trailing 1k trades.
- Feature drift score >0.8 (possible regime change or data issue).

Alerts routed via PagerDuty + Slack; include auto-generated incident context (recent trades, agent decisions).

## Deployment Process

1. **Staging Validation**
   - Deploy policies to staging cluster using replay data (latest 30 days).
   - Verify parity between training and serving observations via checksum logs.
   - Execute paper trading session for 5 trading days; review metrics.

2. **Change Control**
   - Submit deployment RFC including evaluation metrics, risk assessment, rollback plan.
   - Obtain approval from risk committee before production cutover.

3. **Production Rollout**
   - Perform canary deployment on 10% of symbols; monitor for 2 trading days.
   - If stable, scale to full universe; keep fallback to SL threshold strategy for emergency reversion.

4. **Post-Deployment Review**
   - Capture performance summary after 2 weeks; adjust policies if performance drifts.
   - Archive logs and metrics snapshots for audit.

## Incident Response Playbooks

| Scenario | Immediate Action | Follow-Up |
|----------|------------------|-----------|
| Latency spike | Switch to safe-mode (halt new orders, close positions gradually); investigate GPU nodes | Root cause analysis (hardware, load); update capacity planning |
| Cost explosion | Engage master agent defensive mode; tighten action masks; review spreads | Recalibrate transaction cost model; adjust reward weights |
| Drawdown breach | Trigger kill switch; unwind positions; alert stakeholders | Post-mortem with reward/agent analysis; revise guardrails |
| Feature drift | Pause trading; validate data feeds; retrain agents with latest distribution | Update feature normalization; monitor drift detectors |
| Model anomaly | Revert to last known-good checkpoint; run anomaly detection on decisions | Expand unit/integration tests; add explainability hooks |

## Security & Compliance

- Role-based access control for policy registry and execution gateway.
- Signed model artifacts with SHA256; verify signature before loading.
- Encrypt traffic between services (mTLS).
- Log retention compliant with regulatory requirements (e.g., 7 years for order/trade data).

## Continuous Improvement

- **Shadow Deployment:** Run next-generation policies in parallel (no orders) to gather comparative metrics.
- **Feedback Loop:** Feed live trading data into retraining sets; schedule monthly RL fine-tunes.
- **A/B Testing:** Rotate subsets of symbols between policy variants to measure incremental gains.

## Go / No-Go Checklist

- ✅ Policies achieve minimum success metrics on validation (Sharpe >0.8, max DD <30%, PF >1.1).
- ✅ Staging paper trading passes with positive net returns and controlled turnover.
- ✅ Monitoring dashboards operational with alerting tested.
- ✅ Risk committee has signed off on guardrails and incident playbooks.
- ✅ Rollback plan rehearsed (switch to SL thresholds or close positions).
- ❌ If any condition fails, delay deployment and remediate.

## Appendices

### A. Reference Services
- `services/feature_service/`
- `services/symbol_agent_server/`
- `services/master_agent_server/`
- `services/execution_gateway/`

### B. Key Configuration Files
- `config/rl/serving.yaml`
- `config/rl/risk_overlays.yaml`
- `config/rl/feature_drift_thresholds.json`

### C. Glossary
- **HybridRLObservation:** Feature vector combining raw market data, SL predictions, and portfolio context.
- **MAPPO:** Multi-Agent Proximal Policy Optimization.
- **Herfindahl Index:** Concentration measure used for diversification monitoring.
