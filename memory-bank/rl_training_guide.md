# Multi-Agent RL Training Guide

**Document Version:** 1.1  
**Date:** 2025-10-05  
**Status:** Ready for Execution (Phase 1-3)

## Purpose

This guide describes the operational procedures for developing, training, and validating the hierarchical multi-agent reinforcement learning (RL) trading system. It covers environment setup, training workflows for symbol and master agents, hyperparameter management, evaluation protocols, and troubleshooting practices for the 12-week roadmap.

## Prerequisites

### Hardware
- 1× NVIDIA Ada/Ampere GPUs (=16 GB VRAM)
- 96 GB system RAM per training node
- High-throughput NVMe storage (≥2 GB/s)
- Optional: Dedicated evaluation node with CPU focus for backtesting

### Software Stack
- Python 3.12.10 (project `.venv`)
- PyTorch 2.8.0.dev+cu128 (CUDA 12.8)
- Gymnasium 1.1.1, Stable-Baselines3 2.7.0, Ray 2.49.2 (`ray[rllib]` extras)
- Optuna 3.x (bundled via `requirements_rl.txt`)
- MLflow 2.15 (tracking server)
- Redis / PostgreSQL (experience buffer + Optuna storage)

Install & verify:

```bash
pip install -r requirements.txt
pip install -r requirements_rl.txt
python scripts/verify_rl_environment.py
python scripts/verify_rl_libraries.py
python scripts/test_gpu_rl_readiness.py
```

### Data Assets
- Historical OHLCV + technical feature parquet files (`data/historical/<SYMBOL>/1Hour/data.parquet`)
- Sentiment overlays (`data/sentiment/<SYMBOL>/daily_sentiment.parquet`)
- Macro/regime features (generated via `analysis/generate_model_index.py`)
- Supervised learning checkpoints staged in `models/sl_checkpoints/<model>/model.pt` with `metadata.json` and `scalers.joblib`
- Consolidated SL probability tensors (`data/precomputed/sl_signals/*.npy`)
- Data quality report + remediation tracker (`data/validation_report.json`, `docs/data_quality_report_rl.md`)

## Environment Setup

0. **Preflight Checks**
   - Confirm GPU readiness via `scripts/test_gpu_rl_readiness.py` (logs → `docs/gpu_readiness_report.txt`).
   - Validate RL data coverage with `scripts/validate_rl_data_readiness.py`; review `data/validation_report.json` for outstanding symbol gaps.
   - Smoke-test SL checkpoints with `scripts/test_sl_checkpoint_loading.py` and capture baselines via `scripts/benchmark_sl_inference.py` before integrating probabilities into RL observations.

1. **Feature Fusion Pipeline**
   - Run `scripts/generate_combined_training_data.py --include-rl-features` to produce RL-ready tensors (technical, sentiment, macro, SL signals).
   - Output directory: `data/training_data_rl/` containing `train_rl.npz`, `val_rl.npz`, `test_rl.npz`.

2. **RL Environment Module**
   - Implement `core/rl/trading_env.py` (Phase 1 Task 1.1) with Gymnasium interface.
   - Implement `core/rl/vector_env.py` for batched rollouts and GPU acceleration.

3. **Configuration Templates**
   - `training/config_templates/rl_symbol_agent.yaml`
   - `training/config_templates/rl_master_agent.yaml`
   - `training/config_templates/rl_curriculum.yaml`

4. **Experiment Tracking**
   - Launch MLflow tracking server (`mlflow server --host 127.0.0.1 --port 8080`).
   - Configure environment variable `MLFLOW_TRACKING_URI=http://127.0.0.1:8080`.

## Training Workflow Overview

| Phase | Timeline | Objectives | Key Outputs |
|-------|----------|------------|-------------|
| Phase 1 | Weeks 1-2 | Environment, reward shaping, vectorization | `core/rl/trading_env.py`, smoke tests |
| Phase 2 | Weeks 3-6 | Symbol agent PPO training w/ shared encoder | Agent checkpoints, PPO logs |
| Phase 3 | Weeks 7-8 | Master agent MAPPO training | Portfolio policy checkpoints |
| Phase 4 | Weeks 9-10 | Walk-forward + stress validation | `reports/rl_validation_*.json` |
| Phase 5 | Weeks 11-12 | Deployment hardening, integration | Production-ready models |

## Phase 1: Environment Validation

1. **Unit Tests**
   - Create `tests/test_trading_env.py` to verify reset/step signatures and reward invariants.
   - Validate transaction cost penalties and drawdown penalties with crafted scenarios.

2. **Smoke Rollouts**
   - Run random policy rollouts (`scripts/smoke_test_env.py`) for 10k steps.
   - Inspect reward component logging for expected magnitudes.

3. **Reward Coefficient Calibration**
   - Conduct grid search on reward coefficients; log contributions in MLflow to ensure no single component dominates (>60%).
   - After each calibration run, export episode traces (`reward_shaper.component_history`) and invoke `python scripts/analyze_reward_signals.py --episode-data <path>` to confirm healthy component balance and signal-to-noise levels.
   - Generate post-run diagnostics with `python scripts/monitor_environment_performance.py --episode-data <episode.json> --output-dir analysis/environment/<run>` to inspect action diversity, reward quality, portfolio KPIs, and risk events; archive the resulting `performance_report.txt` alongside MLflow artifacts.

## Phase 2: Symbol Agent Training (PPO)

### Encoder Pre-Training
- Objective: Masked next-step prediction on historical sequences.
- Command:

```bash
python training/pretrain_shared_encoder.py \
  --data data/training_data_rl/train_rl.npz \
  --epochs 25 --batch-size 1024 --lr 1e-3
```

### PPO Configuration

```yaml
learning_rate: 3e-4
batch_size: 256
num_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_ratio: 0.2
value_coef: 0.5
entropy_coef: 0.01
max_grad_norm: 0.5
rollout_steps: 2048
num_envs: 128
reward_normalization: running_mean
```

### Training Loop Skeleton

```python
for curriculum_stage in curriculum.schedule:
    env = build_vector_env(stage=curriculum_stage)
    for update in range(stage_updates):
        batch = rollout_collector.collect(env, policy, steps=2048)
        advantages = compute_gae(batch.rewards, batch.values)
        metrics = ppo_update(policy, batch, advantages)
        log_metrics(metrics, symbol=curriculum_stage.symbol_group)
```

### Curriculum Strategy
- **Stage 0:** Top 10 liquid symbols, low volatility periods.
- **Stage 1:** Expand to top 50 symbols, include moderate volatility windows.
- **Stage 2:** Full 143-symbol universe with regime-mixed samples.

### Checkpoints & Validation
- Save checkpoints every 10 updates to `models/rl/symbol_agents/<symbol>/epoch_XXX.pt`.
- Evaluate on validation set using deterministic policy; target Sharpe $>0.6$ and turnover < 8k trades/year.

## Phase 3: Master Agent Training (MAPPO)

### Setup
- Freeze symbol agent encoders and actor heads initially; allow value heads to adapt.
- Observation aggregation: mean/max pooling of symbol signals, portfolio telemetry, regime indicators.

### MAPPO Hyperparameters

```yaml
learning_rate: 1.5e-4
clip_ratio: 0.15
central_value_coef: 1.0
entropy_coef: 0.005
gamma: 0.995
gae_lambda: 0.9
rollout_steps: 1024
num_envs: 64
communication_interval: 4  # master updates every 4 symbol steps
```

### Joint Fine-Tuning
- After master agent stabilizes, unfreeze symbol agent actor heads with low learning rate (1e-5) for coordinated fine-tuning.
- Use population-based training (PBT) for risk multiplier exploration.

## Phase 4: Validation & Backtesting

1. **Walk-Forward Evaluation**
   - Rolling 30-day train/10-day test windows from 2023-10-01 to 2025-10-01.
   - Metrics: Sharpe, Sortino, max drawdown, hit rate, turnover, SPY benchmark excess.

2. **Stress Testing**
   - Replay high-volatility periods (2020 COVID crash, 2022 bear market if available).
   - Inject synthetic slippage shocks (+50% transaction cost) to assess robustness.

3. **A/B Comparisons**
   - Compare RL policy vs SL thresholds, top-decile filter, regime-aware heuristics from quick wins.

4. **Reporting**
   - Generate `reports/rl_validation_summary.json` plus Markdown summary for memory-bank.

## Phase 5: Production Readiness

- Convert policies to TorchScript for low-latency inference.
- Implement evaluation harness `scripts/validate_rl_strategy.py` with CLI options for dataset slices.
- Execute shadow-trading in paper environment for 4 weeks prior to go-live.

## Monitoring & Logging

| Metric | Symbol Agent | Master Agent | Notes |
|--------|--------------|--------------|-------|
| Average reward (per 1k steps) | ✔ | ✔ | Monitor for drift |
| Turnover (trades/day) | ✔ | ✔ | Hard cap at 50 trades/day per symbol |
| Drawdown | ✔ | ✔ | Alert if < -25% (symbol) or -20% (portfolio) |
| Cost ratio (gross vs net) | ✔ | ✔ | Flag if costs exceed 35% of gross P&L |
| Diversification index | — | ✔ | Herfindahl index < 0.2 |

## Hyperparameter Management

- Track all experiments in MLflow with tags: `agent_type`, `curriculum_stage`, `reward_version`.
- Automate sweeps with Optuna using `training/run_rl_hpo.py` (extends HPO infrastructure to RL).
- Store best configurations under `configs/rl/best_params/<agent>/<date>.yaml`.

## Troubleshooting Playbook

| Symptom | Likely Cause | Resolution |
|---------|--------------|------------|
| Reward collapses to negative values | Reward weights mis-scaled; PPO clipping too tight | Rebalance reward scales; increase `clip_ratio` to 0.3; normalize rewards |
| Excessive churn (>15k trades/year) | Cost penalty insufficient; exploration too high | Increase cost penalty weight; raise entropy decay rate; add action masking |
| Master agent oscillates risk modes | Observation lag; insufficient smoothing | Add exponential moving averages to aggregated signals; reduce learning rate |
| Divergent value loss | Central critic underfitting | Increase critic capacity; apply gradient clipping at 0.3 |
| GPU OOM | Batch size too large; replay buffer retention | Reduce `num_envs`; enable gradient checkpointing; offload buffers to CPU |

## Documentation & Checkpoints

- **Daily Logs:** Automated via MLflow; include seed, git hash, environment snapshot.
- **Weekly Reports:** Summaries of reward trends, validation metrics, anomalies.
- **Checkpoint Retention:** Keep top-3 checkpoints per agent based on validation Sharpe.

## Go / No-Go Gates

- **Phase 2 Exit:** ≥70% of symbol agents reach Sharpe $>0.6$ on validation along with turnover <8k trades/year.
- **Phase 3 Exit:** Master policy improves portfolio Sharpe by ≥0.2 and reduces max drawdown by ≥15% vs independent agents.
- **Phase 4 Exit:** Combined system satisfies minimum success criteria (Sharpe >0.8, max DD <30%, win rate >48%, PF >1.1) on out-of-sample.

## Appendices

### A. Sample Curriculum Configuration

```yaml
stages:
  - name: stage0_liquid_low_vol
    symbols: [AAPL, MSFT, NVDA, SPY, QQQ, AMZN, TSLA, META, GOOGL, JPM]
    volatility_filter: [0, 0.25]
    duration: 20  # PPO updates
  - name: stage1_mid_cap
    symbols: symbol_universe_top50
    volatility_filter: [0, 0.35]
    duration: 30
  - name: stage2_full
    symbols: full_universe
    volatility_filter: [0, 0.6]
    duration: 40
```

### B. Evaluation Metrics

- **Sharpe Ratio:** $\text{Sharpe} = \frac{\mathbb{E}[r_t]}{\sigma[r_t]} \sqrt{252}$
- **Profit Factor:** $\text{PF} = \frac{\sum \text{gross profit}}{\sum \text{gross loss}}$
- **Hit Rate:** Fraction of trades with positive net P&L.
- **Cost Ratio:** Transaction cost / gross profit.
