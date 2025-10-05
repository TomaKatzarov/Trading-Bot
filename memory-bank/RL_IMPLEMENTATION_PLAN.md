# Multi-Agent RL Trading System Implementation Plan

## Executive Summary
The Trading Bot program will evolve into a production-ready multi-agent reinforcement learning (RL) trading platform over an 18-week horizon (≈4.5 months). The roadmap is segmented into eight progressive phases—from foundational setup through production readiness—each with explicit deliverables, success criteria, quality gates, and rollback strategies. By Week 18 the team targets a validated, risk-managed portfolio orchestrated by 143 symbol agents and a master agent that collectively outperform existing supervised learning (SL) baselines and the SPY benchmark.

**Core goals for this initiative**
- Replace the current single-model SL pipeline with a scalable multi-agent RL architecture.
- Achieve portfolio-level Sharpe ratio > 1.0, max drawdown < 30%, and >5% annualized outperformance vs. SPY buy-and-hold.
- Deliver repeatable training, evaluation, and deployment workflows with full observability, risk controls, and documentation.

**High-level timeline**
- **Weeks 1-2:** Phase 0 — Foundational environment, data, checkpoints, baseline documentation.
- **Weeks 3-4:** Phase 1 — Trading environment creation, validation, and benchmarking.
- **Weeks 5-6:** Phase 2 — Agent architectures, shared encoders, policy modules.
- **Weeks 7-8:** Phase 3 — Prototype training on 10 symbols with analysis and tuning.
- **Weeks 9-12:** Phase 4 — Scale training to 143 agents with distributed infrastructure.
- **Weeks 13-14:** Phase 5 — Master agent development and (optional) joint fine-tuning.
- **Weeks 15-16:** Phase 6 — Comprehensive validation, stress, and sensitivity testing.
- **Weeks 17-18:** Phase 7 — Production hardening, monitoring, documentation, GO/NO-GO decision.

Progression between phases requires meeting well-defined success metrics, passing quality gates, and completing deliverables. Rollback plans ensure safe recovery from phase-level failures without jeopardizing the overall schedule.

## Timeline & Milestone Overview

| Phase | Duration | Calendar Weeks | Primary Objectives | Key Deliverables |
| --- | --- | --- | --- | --- |
| 0. Foundation & Setup | 2 weeks | 1-2 | Environment, data, SL checkpoints, baseline docs | RL-ready env, data validation report, SL inference benchmarks, baseline report |
| 1. Trading Environment | 2 weeks | 3-4 | Gymnasium-compatible trading env, feature engineering, reward shaping | `core/rl/environments/`, feature pipeline, reward modules, tests, benchmarks |
| 2. Agent Architecture | 2 weeks | 5-6 | Transformer encoder, symbol & master policy scaffolding | `core/rl/agents/`, `core/rl/policies/`, architecture diagrams |
| 3. Prototype Training | 2 weeks | 7-8 | Train 10 symbol agents, tune PPO, analyze performance | `train_symbol_agents.py`, tuned hyperparams, analysis report |
| 4. Full Symbol Training | 4 weeks | 9-12 | Train all 143 agents with distributed infrastructure | 143 agent checkpoints, training logs, validation metrics |
| 5. Master Agent | 2 weeks | 13-14 | Portfolio-level environment and PPO training | Master agent weights, portfolio backtests |
| 6. Comprehensive Validation | 2 weeks | 15-16 | Walk-forward, stress, sensitivity, comparison testing | Validation + stress reports, comparison analyses |
| 7. Production Readiness | 2 weeks | 17-18 | Deployment artifacts, monitoring, risk controls, documentation | ONNX exports, serving API, monitoring dashboards, GO/NO-GO memo |

## Phase Breakdown & Task Decomposition
Each phase includes detailed tasks, dependencies, resources, success criteria, quality gates, and rollback steps. Responsible roles assume **ML Engineer (MLE)** full-time ownership with optional contributors: **RL Specialist (RLS)** and **Risk Manager (RM)**.

### Phase 0: Foundation & Setup (Weeks 1-2)
**Goal:** Establish infrastructure, validate data, organize SL checkpoints, and capture baseline performance.

**Milestones**
- Completed environment setup documentation.
- Verified data readiness across 143 symbols.
- SL checkpoints organized and benchmarked.
- Baseline report published.

**Task Breakdown**

| Task ID | Description | Outputs | Dependencies | Owner | Est. Effort |
| --- | --- | --- | --- | --- | --- |
| 0.1 | Environment Audit & Setup | Verified tooling & docs | None | MLE | 2 days |
| 0.1.1 | Verify Python ≥3.10, PyTorch ≥2.0, CUDA ≥12.1 | Version report | None | MLE | 0.5 day |
| 0.1.2 | Install `stable-baselines3`, `gymnasium`, `ray[rllib]` | Requirements update | 0.1.1 | MLE | 0.5 day |
| 0.1.3 | Test GPU & CUDA availability | GPU validation log | 0.1.1 | MLE | 0.5 day |
| 0.1.4 | Create `trading_rl_env` virtualenv | `.venv` activation guide | 0.1.1 | MLE | 0.5 day |
| 0.1.5 | Document setup (`docs/setup_rl_environment.md`) | Markdown guide | 0.1.2-0.1.4 | MLE | 0.5 day |
| 0.2 | Data Validation & Preparation | Data readiness report + script | None | MLE | 3 days |
| 0.2.1 | Verify historical data (Oct 2023 - Oct 2025) | Inventory checklist | None | MLE | 0.5 day |
| 0.2.2 | Confirm 143 symbols coverage | Coverage matrix | 0.2.1 | MLE | 0.5 day |
| 0.2.3 | Validate technical indicators | Indicator QA logs | 0.2.1 | MLE | 0.5 day |
| 0.2.4 | Confirm sentiment scores attached | Sentiment QA logs | 0.2.1 | MLE | 0.5 day |
| 0.2.5 | Produce data quality report | `data/validation_report.json` update | 0.2.2-0.2.4 | MLE | 0.5 day |
| 0.2.6 | Script `scripts/validate_rl_data_readiness.py` | Executable script | 0.2.5 | MLE | 0.5 day |
| 0.3 | SL Model Checkpoint Organization | Structured checkpoints, metadata, benchmarks | 0.1 complete | MLE | 3 days |
| 0.3.1 | Copy HPO checkpoints to `models/sl_checkpoints/` | Organized directory | None | MLE | 0.5 day |
| 0.3.2 | Create metadata JSON files | Metadata per checkpoint | 0.3.1 | MLE | 0.5 day |
| 0.3.3 | Test checkpoint loading & inference | Smoke test notebook or script | 0.3.1 | MLE | 0.5 day |
| 0.3.4 | Benchmark inference (<1ms/prediction) | `scripts/benchmark_sl_inference.py` output | 0.3.3 | MLE | 1 day |
| 0.3.5 | Author benchmarking script | Script + README snippet | 0.3.4 | MLE | 0.5 day |
| 0.4 | Project Structure Creation | RL directory scaffolding + gitignore | 0.1 complete | MLE | 2 days |
| 0.4.1 | Create `core/rl/` subpackages | Package skeleton | 0.4 | MLE | 1 day |
| 0.4.2 | Create `training/rl/` structure | Training artifacts directories | 0.4 | MLE | 0.5 day |
| 0.4.3 | Update `.gitignore` for RL artifacts | Gitignore entries | 0.4.1-0.4.2 | MLE | 0.5 day |
| 0.5 | Baseline Measurement | SL baseline metrics & report | 0.2, 0.3 complete | MLE | 3 days |
| 0.5.1 | Re-run backtesting with thresholds 0.60-0.80 | Backtest runs | 0.2 | MLE | 1 day |
| 0.5.2 | Document best SL performance | Summary table | 0.5.1 | MLE | 0.5 day |
| 0.5.3 | Set RL performance targets | Target metrics | 0.5.2 | MLE + RM | 0.5 day |
| 0.5.4 | Write baseline report (`docs/baseline_for_rl_comparison.md`) | Markdown report | 0.5.1-0.5.3 | MLE | 1 day |

#### Phase 0 Progress Tracker

- [x] 0.1.1: Verify Python environment ✅ (Python 3.12.10, PyTorch 2.8.0.dev20250415+cu128, CUDA 12.8, 1 GPU)
- [x] 0.1.2: Install RL libraries ✅ (Gymnasium 1.1.1, Stable-Baselines3 2.7.0, Ray 2.49.2)
- [x] 0.1.3: Test GPU availability ✅ (Matrix mult 10×5000² = 0.187s, NN forward = 0.061s, 15.92 GB VRAM)
- [x] 0.1.4: Create `trading_rl_env` virtual environment ✅ (venv provisioned with project + RL deps; activation scripts added)
- [x] 0.1.5: Document environment setup ✅ (`docs/setup_rl_environment.md` published with procedures)
- [x] 0.2.1: Verify historical data availability ✅ (86/162 symbols pass 2023-10-02 → 2025-10-01 coverage; remediation needed for remaining 76)
- [x] 0.2.2: Check symbol coverage ✅ (config expects 162; data provides 154 directories, 19 symbols missing parquet files)
- [x] 0.2.3: Validate technical indicators are pre-computed ✅ (0 missing columns, 0 NaNs in trailing 168 bars)
- [x] 0.2.4: Confirm sentiment scores are attached ✅ (`sentiment_score_hourly_ffill` present and within [0,1])
- [x] 0.2.5: Create data quality report ✅ (`docs/data_quality_report_rl.md` issued with remediation plan)
- [x] 0.2.6: Script `scripts/validate_rl_data_readiness.py` ✅ (outputs `data/validation_report.json` + log)
- [x] 0.3.1: Copy HPO checkpoints to `models/sl_checkpoints/` ✅ (MLP/LSTM/GRU checkpoints staged with scalers)
- [x] 0.3.2: Create checkpoint metadata JSON files ✅ (`metadata.json` populated with hyperparameters & provenance)
- [x] 0.3.3: Test checkpoint loading and inference ✅ (`scripts/test_sl_checkpoint_loading.py` smoke-tests all checkpoints)
- [x] 0.3.4: Benchmark inference speed (<1ms per prediction) ✅ (`reports/sl_inference_benchmarks.json` shows <0.1 ms/pred on GPU)
- [x] 0.3.5: Script `scripts/benchmark_sl_inference.py` ✅ (full benchmarking harness with logging & JSON output)
- [x] 0.4.1: Create RL project structure (`core/rl/...`) ✅ (packages scaffolded under `core/rl/` with placeholders)
- [x] 0.4.2: Create training directory structure (`training/rl/...`) ✅ (README and subdirs committed)
- [x] 0.4.3: Update `.gitignore` for RL artifacts ✅ (ignores checkpoints, logs, and Ray outputs)
- [x] 0.5.1: Re-run backtesting with optimal thresholds (0.60-0.80) ✅ (`backtesting/results/threshold_sweep/` JSONs dated 2025-10-05)
- [x] 0.5.2: Document best SL-only performance achievable ✅ (`docs/baseline_for_rl_comparison.md` tables, Section "Baseline Summary")
- [x] 0.5.3: Set RL performance targets (must beat best SL) ✅ (targets recorded in `docs/baseline_for_rl_comparison.md`)
- [x] 0.5.4: Create baseline report (`docs/baseline_for_rl_comparison.md`) ✅ (see new baseline report)

**Phase Dependencies & Prerequisites**
- Must have repository access, data storage credentials, and GPU hardware.
- Ensure prior SL models and HPO results are accessible in `models/`.

**Resource Requirements**
- 1× MLE full-time (Weeks 1-2).
- Hardware: 1 GPU-enabled workstation, ≥64GB RAM, 500GB storage for data copies.

**Success Criteria**
- ✅ All dependencies installed and validated via scripted checks.
- ✅ Data coverage confirmed for 143 symbols with documented gaps (if any) addressed.
- ✅ SL checkpoints load, infer, and benchmark <1ms per prediction.
- ✅ RL-specific directory scaffolding committed with `.gitignore` updates.
- ✅ Baseline performance documented with target thresholds.

**Quality Gate to exit Phase 0**
- All scripts (`scripts/validate_rl_data_readiness.py`, `scripts/benchmark_sl_inference.py`) executed successfully with documented outputs.
- Setup and baseline documentation peer-reviewed (MLE + RLS).

**Rollback Plan**
- If environment setup fails: revert to previous Python/CUDA versions, engage DevOps for compatibility, block RL work until resolution.
- If data gaps discovered: escalate to data engineering, populate missing segments before Phase 1 starts.
- If SL checkpoints unusable: re-export from HPO artifacts or revisit SL training pipeline.

---

### Phase 1: Trading Environment Development (Weeks 3-4)
**Goal:** Build and validate Gymnasium-compatible trading environments with supporting feature, reward, and portfolio components.

**Milestones**
- `core/rl/environments/` package implemented with `TradingEnvironment`, `PortfolioEnvironment`, and vectorized wrappers.
- Feature extraction pipeline integrates SL predictions and market regimes.
- Reward shaping module finalized and benchmarked.
- Tests covering reset, step, rewards, and edge cases with performance benchmarks.

**Task Breakdown**

| Task ID | Description | Outputs | Dependencies | Owner | Est. Effort |
| --- | --- | --- | --- | --- | --- |
| 1.1 | Core Environment Class | `TradingEnvironment` implementation | Phase 0 deliverables | MLE + RLS | 5 days |
| 1.1.1 | Create `TradingEnvironment` inheriting `gymnasium.Env` | `trading_env.py` | 0.4 | MLE | 1 day |
| 1.1.2 | Define observation space as specified | Observation schema | 1.1.1 | MLE | 0.5 day |
| 1.1.3 | Define action space (7 discrete actions) | Action schema | 1.1.1 | MLE | 0.5 day |
| 1.1.4 | Implement `reset()` | Reset returns initial observation | 1.1.2-1.1.3 | MLE | 0.5 day |
| 1.1.5 | Implement `step()` with transaction costs | State transition logic | 1.1.4 | MLE | 1 day |
| 1.1.6 | Implement `render()` for debugging | Console/plot output | 1.1.4 | MLE | 0.5 day |
| 1.1.7 | Add comprehensive logging | Structured logs | 1.1.5 | MLE | 0.5 day |
| 1.2 | Feature Engineering Pipeline | `FeatureExtractor` class | 0.2, 0.3 | MLE | 4 days |
| 1.2.1 | Create `FeatureExtractor` | `feature_encoder.py` partial | 1.1 | MLE | 0.5 day |
| 1.2.2 | Technical feature computation | Technical tensors | 1.2.1 | MLE | 1 day |
| 1.2.3 | Integrate SL inference | SL probability feed | 1.2.2, 0.3 | MLE | 0.5 day |
| 1.2.4 | Add regime detection (VIX proxy, trend, breadth) | Regime features | 1.2.2 | MLE | 1 day |
| 1.2.5 | Implement feature normalization | Normalized tensors | 1.2.2 | MLE | 0.5 day |
| 1.2.6 | Add feature caching | Performance optimization | 1.2.5 | MLE | 0.5 day |
| 1.2.7 | Test extraction speed (<10ms) | Benchmark report | 1.2.6 | MLE | 0.5 day |
| 1.3 | Reward Function Implementation | `RewardShaper` class | 1.1 | MLE + RLS | 3 days |
| 1.3.1 | Create `RewardShaper` | `reward_shaper.py` | 1.1 | MLE | 0.5 day |
| 1.3.2-1.3.6 | Implement components (P&L, costs, time, Sharpe, drawdown) | Configurable reward components | 1.3.1 | MLE | 1.5 days |
| 1.3.7 | Add configurable weights | Config object | 1.3.6 | MLE | 0.5 day |
| 1.3.8 | Test reward ranges (-1 to +1) | Unit test results | 1.3.7 | MLE | 0.5 day |
| 1.4 | Portfolio State Management | `Portfolio` class | 1.1.5 | MLE | 3 days |
| 1.4.1-1.4.7 | Implement cash/positions, entry/exit logic, metrics, risk limits, transaction costs | Portfolio utilities | 1.1, 1.3 | MLE | 3 days |
| 1.5 | Environment Testing | Test suite & benchmarks | 1.1-1.4 | MLE | 3 days |
| 1.5.1-1.5.5 | `tests/test_trading_env.py` (reset, step, rewards, edge cases) | Passing tests | 1.1-1.4 | MLE | 2 days |
| 1.5.6 | Benchmark performance (>1000 steps/sec) | Benchmark logs | 1.5.5 | MLE | 0.5 day |
| 1.5.7 | Memory profiling (100k steps) | Profiling report | 1.5.5 | MLE | 0.5 day |
| 1.6 | Vectorized Environment | `vectorized_env.py` | 1.1, 1.5 | MLE + RLS | 2 days |
| 1.6.1 | Implement `VectorizedTradingEnv` using `gymnasium.vector` | Vector env class | 1.5 | MLE | 1 day |
| 1.6.2 | Parallel training support | Config examples | 1.6.1 | MLE | 0.5 day |
| 1.6.3 | Test multi-processing stability | Stress test report | 1.6.1 | MLE | 0.25 day |
| 1.6.4 | Benchmark speedup (target 8× on 8 workers) | Benchmark logs | 1.6.3 | MLE | 0.25 day |

#### Phase 1 Progress Tracker
- [x] 1.1.1: ✅ TradingEnvironment created (2025-10-05)
- [x] 1.1.2: Observation space schema implemented with multi-tensor dict
- [x] 1.1.3: Seven-action discrete policy space wired with validation guards
- [x] 1.1.4: `reset()` delivers reproducible seeding and configurable start offsets
- [x] 1.1.5: `step()` integrates transaction costs, stop/limit enforcement, reward shaping
- [x] 1.1.6: Human render pathway added for debugging summaries
- [x] 1.1.7: Structured logging and trade journaling enabled via `logging`
- [x] 1.2.1: ✅ FeatureExtractor base class created (2025-10-05)
	- File: `core/rl/environments/feature_extractor.py`
	- Features: LRU caching, zscore/minmax/robust normalization, batch extraction helper
	- Performance: <0.1 ms per window extraction once cached (local benchmark)
- [x] 1.2.2: ✅ Regime indicator extraction (2025-10-05)
	- File: `core/rl/environments/regime_indicators.py`
	- Indicators: 10 normalized market regime features (volatility, trend, momentum, volume)
	- Integration: Regime vectors exposed via `RegimeIndicators.get_regime_vector()` for environment use
- [x] 1.2.3: ✅ SL model prediction integration (2025-10-05)
	- Method: `FeatureExtractor.get_sl_predictions()`
	- Supports batch-style inference per window with neutral fallbacks
	- Leverages `scripts/sl_checkpoint_utils.run_inference` with exception guards
- [x] 1.2.3.a: ✅ RL environment tests executed inside `trading_rl_env` (2025-10-05)
	- Command: `source trading_rl_env/Scripts/activate && python -m pytest tests/test_trading_env.py`
	- Ensures Gymnasium dependencies are resolved via RL-specific virtualenv
	- `pytest` pinned in `requirements_rl.txt` for reproducible future runs
- [x] 1.2.4: ✅ Feature extraction test suite (2025-10-05)
	- File: `tests/test_feature_extractor.py` (~15 cases)
	- Coverage: initialization, window/batch extraction, z-score / min-max / robust normalization, caching, SL fallback, regime indicators
	- Status: `python -m pytest tests/test_feature_extractor.py` passing inside `trading_rl_env`
- [x] 1.2.5: ✅ Performance benchmarks (2025-10-05)
	- Script: `scripts/benchmark_feature_extraction.py`
	- Results: P95 latency 0.5–2.0 ms across normalization modes; cache hit rate ≥90% after warm-up
	- Status: ✅ Exceeds <10 ms target and cache efficiency goal
- [x] 1.2.6: ✅ Integration with TradingEnvironment (2025-10-05)
	- Modified: `core/rl/environments/trading_env.py`
	- Changes: TradingEnvironment now uses `FeatureExtractor` and `RegimeIndicators` for observations
	- Tests: `python -m pytest tests/test_trading_env.py` passing in `.venv`
- [x] 1.3.1: ✅ RewardShaper base class created (2025-10-05)
	- File: `core/rl/environments/reward_shaper.py`
	- Components: 7 (PnL, cost, time, Sharpe, drawdown, sizing, hold penalty)
	- Features: Configurable weights, episode tracking, component analysis
	- Addresses SL failures: Transaction cost awareness, timing rewards
- [x] 1.3.2: ✅ Reward function test suite (2025-10-05)
	- File: `tests/test_reward_shaper.py`
	- Coverage: All 7 components, edge cases, episode tracking
	- Tests: 40+ assertions covering component interactions and statistics
- [x] 1.3.3: ✅ RewardShaper integration (2025-10-05)
	- Modified: `core/rl/environments/trading_env.py`
	- Changes: Wired RewardShaper, updated reward computation, added stats tracking
	- Tests: ✅ `tests/test_reward_shaper.py`; trading env suite requires gymnasium (pending in CI env)
- [x] 1.3.4: ✅ Reward analysis utilities (2025-10-05)
	- Script: `scripts/analyze_reward_signals.py`
	- Features: Component stats, balance checks, SNR & correlation analysis, visualization export
	- Usage: `python scripts/analyze_reward_signals.py --episode-data <json>` (run inside RL env for matplotlib)
- **Task 1.2 Summary:** Feature engineering stack (FeatureExtractor + RegimeIndicators + SL prob integration) fully operational, benchmarks validated, and environment wiring complete with all unit tests green (`python -m pytest tests/test_feature_extractor.py tests/test_trading_env.py`).

> Implementation located at `core/rl/environments/trading_env.py` with:
> - Multi-component observations (technical, SL predictions, position, portfolio, regime)
> - Action execution covering buys, scaling, and exits with compliance checks
> - Reward aggregation balancing equity growth, drawdown penalties, and risk regularization

**Additional Dependencies**
- Phase 0 baseline and data scripts complete.
- RL libraries installed and validated inside `trading_rl_env`.

**Resource Requirements**
- 1× MLE full-time, RLS part-time for reward design review.
- Hardware: 2 GPUs for performance testing, profiling tools.

**Success Criteria**
- ✅ All environment unit tests pass with >90% coverage for `core/rl/environments`.
- ✅ Single-threaded environment achieves >1000 steps/sec; vectorized >5× speedup on 8 workers (target 8×).
- ✅ No memory leaks detected in 100k-step run.
- ✅ SB3/RLlib compatibility confirmed via smoke training script.

**Quality Gate to exit Phase 1**
- Merge request approved with code reviews from RLS.
- Benchmarks captured in `analysis/reports/rl_env_benchmarks.md`.

**Rollback Plan**
- If performance targets missed: profile bottlenecks, simplify observation space, or reduce logging verbosity; extend phase by ≤3 days with approval.
- If reward shaping unstable: revert to minimal reward schema and iterate while halting Phase 2 start.

---

### Phase 2: Agent Architecture Development (Weeks 5-6)
**Goal:** Implement shared transformer encoder, symbol agent, master agent scaffold, and weight sharing mechanisms.

**Milestones**
- Transformer-based `FeatureEncoder` operational.
- Actor-critic implementations for symbol and master agents with forward-pass tests.
- Parameter budgets documented and below thresholds.
- Action masking validated.

**Task Breakdown**

| Task ID | Description | Outputs | Dependencies | Owner | Est. Effort |
| --- | --- | --- | --- | --- | --- |
| 2.1 | Shared Feature Encoder | Transformer encoder module | Phase 1 complete | MLE + RLS | 5 days |
| 2.1.1-2.1.7 | Design + implement 4-layer, 256-dim, 8-head transformer with positional encodings, residuals, unit tests | `core/rl/policies/attention_policy.py` | 1.2, 1.3 | MLE | 5 days |
| 2.2 | Symbol Agent Policy Network | `SymbolAgent` with actor & critic | 2.1 | MLE | 4 days |
| 2.2.1-2.2.7 | Implement actor/critic, action masking, entropy regularization, forward tests, parameter count | `symbol_agent.py`, metrics in docs | 2.1 | MLE | 4 days |
| 2.3 | Master Agent Architecture | Portfolio-level encoder & heads | 2.1, 2.2 | MLE + RLS | 4 days |
| 2.3.1-2.3.5 | Aggregate symbol states, design actor/critic, synthetic data tests | `master_agent.py` | 2.2 | MLE | 4 days |
| 2.4 | Weight Initialization & Transfer | Weight sharing & pretraining hooks | 2.1-2.3 | MLE | 3 days |
| 2.4.1-2.4.4 | Xavier/He init, optional pretraining, shared encoder weights, convergence experiments | Updated modules + notebooks | 2.1-2.3 | MLE | 3 days |

**Dependencies**
- Phase 1 feature encoder interfaces stable.
- Access to SL inference outputs for optional pretraining.

**Resource Requirements**
- 1× MLE full-time, RLS part-time review.
- Hardware: 2 GPUs for experimentation, 128GB RAM recommended for synthetic batching.

**Success Criteria**
- ✅ Encoder forward pass verified on batched synthetic data (no NaNs).
- ✅ Symbol agent parameters <10M; total <50M.
- ✅ Action masking unit tests cover cash/position constraints.
- ✅ PPO compatibility confirmed by running dummy rollout.

**Quality Gate to exit Phase 2**
- Architecture diagram committed to `docs/rl_architecture_diagram.pdf`.
- API docs generated for agents/policies.

**Rollback Plan**
- If transformer fails performance, fallback to LSTM baseline while retaining interface; revisit attention after prototype feedback.
- If action masking unreliable, temporarily disable certain actions and document limitation before Phase 3.

---

### Phase 3: Symbol Agent Training – Prototype (Weeks 7-8)
**Goal:** Demonstrate viability by training 10 symbol agents and beating SL baseline on selected symbols.

**Milestones**
- Training pipeline operational with MLflow, TensorBoard, checkpointing.
- First 10 agents trained to convergence thresholds.
- Hyperparameters tuned; analysis report delivered.

**Task Breakdown**

| Task ID | Description | Outputs | Dependencies | Owner | Est. Effort |
| --- | --- | --- | --- | --- | --- |
| 3.1 | Training Pipeline Setup | `train_symbol_agents.py`, configs | Phase 2 | MLE | 3 days |
| 3.1.1-3.1.6 | Integrate MLflow, configure PPO, checkpointing, TensorBoard, early stopping | Scripts, configs | 1.6, 2.x | MLE | 3 days |
| 3.2 | Curriculum Learning Design | Ranked symbol list, training schedule | 3.1 | MLE + RLS | 2 days |
| 3.2.1-3.2.3 | Rank symbols, select first 10 (SPY, QQQ, AAPL, MSFT, NVDA, AMZN, META, TSLA, JPM, XOM), schedule progression | Curriculum doc | 0.5, 3.1 | MLE | 2 days |
| 3.3 | Prototype Training Run | 10 trained agent checkpoints | 3.1-3.2 | MLE | 6 days |
| 3.3.1-3.3.6 | Train 100k steps each, monitor metrics, validation on 2025 Q2, compare to SL baseline, log issues | Checkpoints + MLflow runs | 3.2 | MLE | 6 days |
| 3.4 | Hyperparameter Tuning | Tuning experiments & results | 3.3 results | MLE | 3 days |
| 3.4.1-3.4.4 | Adjust LR, entropy coef, GAE lambda, reward scaling; grid/random search; select best config | Hyperparam report | 3.3 | MLE | 3 days |
| 3.5 | Analysis & Debugging | Insights and report | 3.3-3.4 | MLE + RLS | 2 days |
| 3.5.1-3.5.5 | Visualize policies, reward components, check degeneracy, regime robustness, produce report | `analysis/reports/rl_prototype_analysis.md` | 3.4 | MLE | 2 days |

**Dependencies**
- Phase 2 agents integrated; training data validated.
- MLflow server accessible.

**Resource Requirements**
- 2 GPUs (dedicated) for training runs.
- 1× MLE full-time; RLS for review of analysis.

**Success Criteria**
- ✅ ≥5/10 agents achieve validation Sharpe > 0.3.
- ✅ Agents outperform SL baseline on selected symbols.
- ✅ No NaN losses or gradient explosions observed.
- ✅ Training of 10 agents completes within 3 days on 8 GPUs or equivalent time-sliced hardware.

**Quality Gate to exit Phase 3**
- Prototype analysis reviewed with stakeholders; go/no-go recorded.
- Hyperparameter configuration exported to `training/rl/configs/prototype.yaml`.

**Rollback Plan**
- If agents fail to meet Sharpe targets: revisit reward shaping (Phase 1) or architecture (Phase 2) before scaling.
- If training instabilities persist: escalate to RLS, run ablation (reduced action space), or extend Phase 3 by 1 week with approval.

---

### Phase 4: Full Symbol Agent Training (Weeks 9-12)
**Goal:** Scale training to all 143 symbols with distributed infrastructure, ensuring quality and resilience.

**Milestones**
- Distributed training orchestrated via Ray with fault tolerance.
- All symbol agents trained, validated, and meeting minimum Sharpe > 0.2 (≥90%).
- Ensemble analysis identifies redundancy and diversification opportunities.

**Task Breakdown**

| Task ID | Description | Outputs | Dependencies | Owner | Est. Effort |
| --- | --- | --- | --- | --- | --- |
| 4.1 | Parallel Training Infrastructure | Ray setup + scripts | Phase 3 success | MLE + DevOps | 5 days |
| 4.1.1-4.1.4 | Configure Ray multi-GPU, distribute agents, implement checkpoint resume, allocate resources | `training/rl/cluster_config.yaml` | 3.x | MLE | 5 days |
| 4.2 | Batch Training Execution | 143 trained agents | 4.1 | MLE | 15 days |
| 4.2.1-4.2.5 | Launch training, monitor dashboards, flag strugglers, targeted interventions, manage timeline | MLflow runs, logs | 3.x | MLE | 15 days |
| 4.3 | Validation & Quality Control | Validation metrics per agent | 4.2 | MLE + RM | 5 days (overlap) |
| 4.3.1-4.3.5 | Validate on held-out periods, compute metrics, flag underperformers (<0.2 Sharpe), retrain, ensure >90% pass | Validation reports | 4.2 | MLE | 5 days |
| 4.4 | Agent Ensemble Analysis | Portfolio-level evaluation | 4.3 | MLE + RLS | 3 days |
| 4.4.1-4.4.4 | Portfolio test, correlation analysis, redundancy detection, diversification optimization | Analysis notebook/report | 4.3 | MLE | 3 days |

**Dependencies**
- Prototype hyperparameters finalized.
- Access to 8 GPUs (or cloud equivalent).

**Resource Requirements**
- Compute: 8× NVIDIA RTX 5070 Ti (or better), 128GB+ RAM, 1TB storage for checkpoints/logs.
- Personnel: MLE full-time; DevOps support for Ray cluster; RM part-time for validation thresholds.

**Success Criteria**
- ✅ ≥130 agents (90%) achieve validation Sharpe > 0.2.
- ✅ No agent loses >50% during validation.
- ✅ Distributed infrastructure stable for 4-week execution without critical downtime.
- ✅ Checkpoints versioned and reproducible.

**Quality Gate to exit Phase 4**
- Validation dashboard (`reports/agent_validation_dashboard.xlsx` or similar) signed off by RM.
- Ensemble analysis recommendations documented before master agent work begins.

**Rollback Plan**
- If infrastructure unstable: throttle to smaller batches, leverage cloud resources, or reschedule training windows.
- If large subset fails validation: pause scaling, revisit curriculum or rewards, potentially re-run Phase 3 on difficult symbols.

---

### Phase 5: Master Agent Development (Weeks 13-14)
**Goal:** Implement and train a portfolio-level master agent coordinating symbol agents.

**Milestones**
- Portfolio environment functional with aggregated state and reward definitions.
- Master agent trained with frozen symbol agents meeting portfolio targets.
- Optional joint fine-tuning assessed.

**Task Breakdown**

| Task ID | Description | Outputs | Dependencies | Owner | Est. Effort |
| --- | --- | --- | --- | --- | --- |
| 5.1 | Portfolio Environment | `PortfolioEnvironment` | Phase 4 | MLE | 4 days |
| 5.1.1-5.1.5 | Aggregate symbol envs, define master action space, implement portfolio reward, test with frozen agents | `portfolio_env.py` updates | 4.x | MLE | 4 days |
| 5.2 | Master Agent Training | Master agent checkpoints | 5.1 | MLE | 6 days |
| 5.2.1-5.2.6 | Initialize agent, load symbol agents, PPO training, monitor metrics, validation | Checkpoints + logs | 4.x | MLE | 6 days |
| 5.3 | Joint Fine-Tuning (Optional) | Fine-tuned agents | 5.2 | MLE + RLS | 4 days |
| 5.3.1-5.3.5 | Unfreeze agents, train jointly with low LR, monitor catastrophic forgetting, compare results | Comparative report | 5.2 | MLE | 4 days |

**Dependencies**
- Phase 4 agents stable and versioned.
- Portfolio reward weights chosen in consultation with RM.

**Resource Requirements**
- 4 GPUs dedicated for master agent PPO runs.
- Personnel: MLE; RLS for reward review; RM for drawdown oversight.

**Success Criteria**
- ✅ Master agent increases portfolio Sharpe vs. fixed allocation baseline.
- ✅ Validation Sharpe > 0.8 and drawdown < 30%.
- ✅ No degenerate strategies (all cash/all-in) observed.

**Quality Gate to exit Phase 5**
- Portfolio backtest report comparing frozen vs joint fine-tuning scenarios.
- Decision log on whether to proceed with joint fine-tuning in later phases.

**Rollback Plan**
- If master agent underperforms: revert to portfolio-level heuristics, revisit reward shaping, postpone joint fine-tuning.
- If catastrophic forgetting occurs during joint training: immediately revert symbol agents to frozen weights and disable joint training path.

---

### Phase 6: Comprehensive Validation (Weeks 15-16)
**Goal:** Stress-test and verify the RL system across temporal, market, and configuration variations.

**Milestones**
- Walk-forward validation completed over Oct 2023 - Oct 2025.
- Stress/sensitivity tests executed with documented resilience.
- Comparison against baselines with clear outperformance.
- Out-of-sample testing confirms robustness.

**Task Breakdown**

| Task ID | Description | Outputs | Dependencies | Owner | Est. Effort |
| --- | --- | --- | --- | --- | --- |
| 6.1 | Walk-Forward Validation | Rolling window results | Phases 3-5 complete | MLE + RM | 5 days |
| 6.1.1-6.1.5 | Window splits, train/validate cycles, result aggregation, trend analysis | Validation report | 5.2 | MLE | 5 days |
| 6.2 | Stress Testing | Regime-specific evaluation | 6.1 | MLE + RM | 3 days |
| 6.2.1-6.2.5 | Test corrections, rallies, sideways markets, measure drawdowns, ensure <40% | Stress report | 6.1 | MLE | 3 days |
| 6.3 | Comparison Backtesting | Benchmark comparison report | 6.1 | MLE + RM | 3 days |
| 6.3.1-6.3.3 | Full backtest vs SL, SPY, equal-weight, random | Comparison report | 6.1 | MLE | 3 days |
| 6.4 | Sensitivity Analysis | Parameter robustness report | 6.1 | MLE | 3 days |
| 6.4.1-6.4.4 | Vary transaction costs, slippage, position limits, document results | Sensitivity tables | 6.1 | MLE | 3 days |
| 6.5 | Out-of-Sample Testing | Unseen data evaluation | 6.1 | MLE + RM | 2 days |
| 6.5.1-6.5.3 | Evaluate Q4 2025 (if available), compare degradation, baseline comparison | OOS report | 6.1 | MLE | 2 days |

**Resource Requirements**
- CPU-heavy backtesting cluster; GPUs optional for inference acceleration.
- Personnel: MLE full-time; RM for validation oversight.

**Success Criteria**
- ✅ Walk-forward results stable with no major degradation; aggregated Sharpe > 0.8.
- ✅ Stress tests show drawdowns ≤40%; portfolio remains profitable.
- ✅ RL system outperforms SL and SPY benchmarks materially.
- ✅ Sensitivity tests confirm robustness to market frictions.
- ✅ Out-of-sample Sharpe > 0.6.

**Quality Gate to exit Phase 6**
- Validation dossier approved by RM and stakeholders (PDF/slide deck for review).

**Rollback Plan**
- If validation fails: identify weakest components (agents, rewards, master agent) and loop back to relevant phase for remediation before proceeding to production readiness.

---

### Phase 7: Production Readiness (Weeks 17-18)
**Goal:** Finalize deployment artifacts, monitoring, risk controls, documentation, and GO/NO-GO decision for paper trading.

**Milestones**
- All agents exported to ONNX with serving APIs benchmarked.
- Risk management overrides and monitoring dashboards operational.
- Documentation updated and runbooks prepared.
- GO/NO-GO meeting conducted with decision recorded.

**Task Breakdown**

| Task ID | Description | Outputs | Dependencies | Owner | Est. Effort |
| --- | --- | --- | --- | --- | --- |
| 7.1 | Model Artifacts | ONNX exports + serving pipeline | Phase 6 success | MLE + DevOps | 4 days |
| 7.1.1-7.1.5 | Export 143 symbol agents + master agent, implement API, benchmark latency (<10 ms), load tests | `models/onnx/`, API service | 4.x, 5.x | MLE | 4 days |
| 7.2 | Risk Management System | Enforcement scripts/dashboards | 7.1 | MLE + RM | 4 days |
| 7.2.1-7.2.5 | Hard stop-loss, drawdown breaker, position limits, daily loss limit, monitoring dashboard | `core/rl/risk/` modules | 7.1 | MLE | 4 days |
| 7.3 | Monitoring & Logging | Real-time observability setup | 7.1 | MLE + DevOps | 3 days |
| 7.3.1-7.3.4 | Performance tracking, alerts, trade logging, live portfolio dashboard | Monitoring stack | 7.2 | MLE | 3 days |
| 7.4 | Documentation | Updated knowledge base | Ongoing | MLE + RLS | 3 days |
| 7.4.1-7.4.4 | Update memory-bank docs, deployment runbook, model version matrix, troubleshooting guide | Docs | All prior phases | MLE | 3 days |
| 7.5 | Final Decision | GO/NO-GO memo | 7.1-7.4 complete | RM + Stakeholders | 2 days |
| 7.5.1-7.5.5 | Review validation, risk committee, decision, initiate paper trading if GO, else document NO-GO actions | Decision memo | 6.x, 7.x | RM | 2 days |

**Resource Requirements**
- Production-like environment for serving tests.
- Personnel: MLE, DevOps, RM, stakeholders.

**Success Criteria**
- ✅ Serving pipeline meets latency <10 ms per decision.
- ✅ Risk systems tested with simulated failures (stop-loss, drawdown, etc.).
- ✅ Monitoring and alerting functioning with test signals.
- ✅ Documentation complete and accessible.
- ✅ GO/NO-GO decision documented; if GO, paper trading plan ready.

**Quality Gate to exit Phase 7**
- Sign-off from RM and leadership on deployment readiness.
- Paper trading start date scheduled (if GO).

**Rollback Plan**
- If latency unacceptable: optimize ONNX runtime, scale hardware, or limit concurrency.
- If risk monitoring fails: block deployment, fix control scripts, re-test before reconsidering GO.

---

## Dependencies & Prerequisites Summary
- **Global prerequisites:** Access to historical and real-time data feeds, secure credentials (see `Credential.env`), GPU infrastructure, and MLflow tracking server.
- **Data dependencies:** Continuous availability of Oct 2023 - Oct 2025 data with complete indicator and sentiment coverage per symbol.
- **Software stack:** Python ≥3.10, PyTorch ≥2.0, CUDA ≥12.1, `stable-baselines3`, `gymnasium`, `ray[rllib]`, MLflow, Ray cluster tooling.
- **People dependencies:** Dedicated MLE, on-call DevOps for cluster setup, RM for validation gates, optional RLS for architecture and RL-specific guidance.

## Resource Requirements

### Computing Resources
- **Minimum baseline:** 8× NVIDIA RTX 5070 Ti (or A100 equivalents in cloud bursts), 128GB RAM, 1TB SSD dedicated to RL artifacts.
- **Augmentation options:** Leverage AWS/GCP GPU instances during Phase 4 heavy training; use spot instances with checkpoint resume.

### Data Resources
- Historical OHLCV + features spanning Oct 2023 - Oct 2025 for 143 symbols.
- Real-time feed for paper trading (Phase 7) with low-latency access.
- Pre-computed technical indicators and sentiment features stored in `data/prepared_training/`.

### Personnel
- **MLE (Full-time, Weeks 1-18):** Responsible for implementation, experiments, and reporting.
- **DevOps (As-needed, Phases 4 & 7):** Ray cluster provisioning, serving infrastructure.
- **RLS (Part-time, Phases 1-5):** Architecture/reward reviews and troubleshooting.
- **RM (Part-time, Phases 3-7):** Validation, risk metric oversight, GO/NO-GO involvement.

## Risk Assessment & Mitigation

| Risk | Probability | Impact | Description | Mitigation | Owner | Detection Signals |
| --- | --- | --- | --- | --- | --- | --- |
| Training instability | High | High | PPO may diverge or collapse | Curriculum learning, reward shaping iterations, gradient clipping, early stopping | MLE + RLS | NaN losses, plateaued rewards |
| Underperforming agents | Medium | High | RL agents fail to beat SL baseline | Hyperparameter tuning, hybrid reward components, fallback to SL improvements | MLE + RM | Validation Sharpe below targets |
| Compute bottlenecks | Medium | High | 143 agents strain GPU/CPU resources | Ray autoscaling, cloud burst plan, staggered training windows | DevOps | Training queue delays |
| Data quality issues | Low | High | Missing features or sentiment data | Phase 0 validation, automated data checks before training | MLE | Data validation alerts |
| Reward hacking | Medium | Medium | Agents exploit loopholes (e.g., minimal trading) | Reward auditing, policy visualization, degenerate behavior tests | MLE + RLS | Degenerate action distributions |
| Overfitting | Medium | Medium | Good in-sample but poor out-of-sample | Walk-forward validation, regularization, dropout, randomization | MLE + RM | OOS performance drop |
| Monitoring gaps | Low | High | Production incidents unnoticed | Implement comprehensive logging, alerts, rehearsed drills | DevOps + RM | Missing alert coverage |

## Success Criteria

### Phase-Level Success Gates
- **Phase 0:** Environment, data, SL baselines ready; scripts executed successfully.
- **Phase 1:** Environments validated with tests, performance, and compatibility benchmarks.
- **Phase 2:** Architectures implemented with forward-pass tests and documented budgets.
- **Phase 3:** ≥5/10 prototype agents exceed Sharpe thresholds and stability achieved.
- **Phase 4:** ≥90% of 143 agents meet validation Sharpe >0.2 with reliable infrastructure.
- **Phase 5:** Master agent lifts portfolio Sharpe >0.8 with drawdown <30%.
- **Phase 6:** System passes walk-forward, stress, comparison, sensitivity, and OOS tests.
- **Phase 7:** Serving, risk, monitoring, docs ready; GO/NO-GO decision made.

### Final Success Metrics (Week 18)
- Portfolio Sharpe > 1.0 (validation), max drawdown < 30%, win rate > 50%, profit factor > 1.3.
- Outperforms SPY by >5% annualized and best SL configuration.
- Robust under stress and sensitivity variations.
- Fully documented, monitored, and ready for paper trading.

### MVP Fallback Targets
- ≥50 symbol agents performing well.
- Portfolio Sharpe > 0.6 with positive returns and controlled risk.

## Quality Gates & Validation Checkpoints
- **Gate 0→1:** Environment/data scripts signed off; baseline report approved.
- **Gate 1→2:** Environment unit tests + benchmarks verified; architecture interfaces stable.
- **Gate 2→3:** Encoder/policy tests green; PPO training dry-run completed.
- **Gate 3→4:** Prototype analysis accepted; hyperparameters frozen.
- **Gate 4→5:** Validation dashboard shows ≥90% agents meeting criteria.
- **Gate 5→6:** Master agent backtest confirms Sharpe/drawdown targets.
- **Gate 6→7:** Validation dossier approved by RM and leadership.
- **Post Phase 7:** GO/NO-GO meeting documented.

## Rollback Strategies by Phase
- **Phase 0:** Revert to prior environment snapshots; prioritize fixing environment/data before progressing.
- **Phase 1:** Simplify observation space; rollback to minimal reward; postpone vectorization if necessary.
- **Phase 2:** Swap transformer encoder for simpler MLP/LSTM temporarily; maintain interface to avoid downstream breakage.
- **Phase 3:** Pause scaling, revisit hyperparameters, consider reward adjustments; extend timeline ≤1 week.
- **Phase 4:** Suspend failing agent groups, re-run with adjusted configs; use cloud burst to recover schedule.
- **Phase 5:** Roll back to frozen-symbol baseline; disable joint fine-tuning if destabilizing.
- **Phase 6:** Identify weak windows/regimes, retrain specific agents or re-weight portfolio; do not proceed to production until resolved.
- **Phase 7:** Delay deployment, iterate on latency or risk controls; preserve validated models until issues resolved.

## Monitoring, Reporting, and Communication Cadence
- **Weekly status reports:** Progress, blockers, metrics per phase, shared with stakeholders every Friday.
- **Phase transition reviews:** 1-hour meeting to review success criteria, quality gate artifacts, and approve progression.
- **Incident channel:** Dedicated Slack/Teams channel for training incidents, with documented escalation path.
- **Documentation updates:** Memory-bank docs refreshed at the end of each phase, ensuring knowledge continuity.

## Deliverables Checklist
- `docs/setup_rl_environment.md` (Phase 0).
- `data/validation_report.json` (updated) and `scripts/validate_rl_data_readiness.py`.
- `models/sl_checkpoints/` with metadata JSON and `scripts/benchmark_sl_inference.py` results.
- RL directory scaffolding (`core/rl`, `training/rl`) with `.gitignore` updates.
- `docs/baseline_for_rl_comparison.md` baseline report.
- `core/rl/environments/`, `core/rl/utils/`, `core/rl/rewards/`, `core/rl/agents/`, `core/rl/policies/` modules with tests.
- Benchmark and profiling reports under `analysis/reports/`.
- `train_symbol_agents.py`, MLflow experiment configurations, tensorboard logs.
- Hyperparameter configuration files (`training/rl/configs/`).
- 143 agent checkpoints + validation reports + ensemble analysis.
- Master agent checkpoints and portfolio backtests.
- Validation dossier (walk-forward, stress, comparison, sensitivity, OOS).
- ONNX exports, serving API code, risk control modules, monitoring dashboards.
- Updated memory-bank documentation, deployment runbook, troubleshooting guide.
- GO/NO-GO decision memo and (if GO) paper trading plan.

## Iteration & Adaptation Guidelines
- **Behind schedule:** Reduce agent scope (e.g., train top 50 symbols first), skip joint fine-tuning, or extend timeline with stakeholder approval while preserving quality gates.
- **Ahead of schedule:** Explore advanced architectures (GNNs), alternative RL algorithms (SAC, TD3), or accelerated paper trading entry.
- **If outcomes disappointing:** Reinvest in SL enhancements (threshold tuning, regime filters), adopt hybrid strategies (RL timing + SL selection), or engage external RL expertise.

## Reporting Templates & Tools
- **Status Template:** Summary, accomplishments, risks, mitigations, next steps.
- **Quality Gate Checklist:** Artifact list, test results, approvals.
- **Risk Log:** Maintained in `docs/risk_register_rl.xlsx`, updated weekly.
- **MLflow Experiment Tags:** Phase, symbol count, hyperparameter version, reward version.

---

With this plan, the team has a detailed, actionable roadmap for implementing, validating, and operationalizing the multi-agent RL trading system over the next 18 weeks, complete with contingencies, quality safeguards, and clear deliverables.