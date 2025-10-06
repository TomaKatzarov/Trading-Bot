# Consolidated Documentation: Project Status, Progress & Results Analysis

**Document Version:** 2.0 - Consolidated  
**Last Updated:** October 6, 2025  
**Status:** Living Document  
**Consolidates:** activeContext.md, progress.md, decisionLog.md, phase3_hpo_results_analysis.md, phase3_completion_report.md, phase3_next_steps.md, strategic_plan_nn_rl.md, sl_to_rl_pivot_analysis.md, diagnostic reports

---

## Table of Contents

1. [Current Project Status](#current-project-status)
2. [Strategic Roadmap](#strategic-roadmap)
3. [Phase Completion History](#phase-completion-history)
4. [Phase 3 HPO Results Analysis](#phase-3-hpo-results-analysis)
5. [Phase 4 Backtesting Campaign Results](#phase-4-backtesting-campaign-results)
6. [Decision Log](#decision-log)
7. [Key Learnings & Patterns](#key-learnings--patterns)
8. [Current Challenges](#current-challenges)
9. [Next Steps](#next-steps)

---

## 1. Current Project Status

### Overview
**Last Updated:** October 6, 2025  
**Current Phase:** Phase 4 - Backtesting & Remediation  
**Project Health:** ⚠️ **CRITICAL** - Backtesting revealed catastrophic losses despite strong classification metrics

### High-Level Status

```
Phase 0: Historical Data Collection         ✅ COMPLETE
Phase 1: NN Infrastructure & Baselines      ✅ COMPLETE
Phase 2: Baseline Training Campaign         ✅ COMPLETE
Phase 3: HPO Optimization                   ✅ COMPLETE
Phase 4: Backtesting & Evaluation           ⚠️  FAILED - Remediation in progress
Phase 5: RL System Development              🔜 ON HOLD (pending Phase 4 resolution)
Phase 6: Production Deployment              🔜 ON HOLD
```

### Critical Findings

**⚠️ Classification vs Trading Performance Mismatch:**

**Test Set Classification Metrics (September 2025):**
- MLP Trial 72: F1+ 0.306, Precision 0.237, Recall 0.415, ROC-AUC 0.866
- LSTM Trial 62: F1+ 0.289, Precision 0.224, Recall 0.400, ROC-AUC 0.855
- GRU Trial 93: F1+ 0.269, Precision 0.209, Recall 0.374, ROC-AUC 0.844

**Backtesting Results (October 4, 2025):**
- MLP Trial 72: **-88.05%** total return, Sharpe -0.04, 10,495 trades
- LSTM Trial 62: **-92.60%** total return, Sharpe -0.02, 11,426 trades
- GRU Trial 93: **-89.34%** total return, Sharpe -0.03, 8,529 trades

**Gap Analysis:**
- Strong offline metrics did NOT translate to profitable trading
- High trade frequency (8,500-11,500 trades over 2 years)
- Transaction costs and slippage (0.10% + 5bps) eroded gains
- Likely distribution shift between train/val/test and backtest period

### Active Work

**Current Focus:** Phase 4 remediation experiments **and** Phase 2 RL agent build-out (Task 2.1 shared encoder complete as of Oct 6)

**Key Investigations:**
1. **Threshold Optimization:** Testing signal_threshold ranges 0.5-0.9 to reduce false positives
2. **Regime Filters:** Testing VIX-based filters to avoid low-volatility periods
3. **Trade Cost Analysis:** Quantifying impact of 0.10% commission + 5bps slippage
4. **Distribution Shift:** Comparing train/val/test vs backtest period feature distributions

**Pending Decisions:**
- Whether to proceed with RL system given SL failure
- Whether to retrain models with different profit targets
- Whether to implement regime-aware filters before RL development

### RL Initiative Progress (Phases 0 ➜ 2)
**Status:** ✅ Phase 0 & Phase 1 complete; Phase 2 Task 2.1 delivered (Week 5 of RL roadmap)

- **Phase 0 deliverables (Week 1):**
   - `scripts/verify_rl_environment.py`, `scripts/verify_rl_libraries.py`, and `scripts/test_gpu_rl_readiness.py` confirm Python 3.12.10, PyTorch 2.8.0.dev+cu128, CUDA 12.8, and RTX 5070 Ti readiness. Reports archived in `docs/setup_rl_environment.md` and `docs/gpu_readiness_report.txt`.
   - `scripts/validate_rl_data_readiness.py` produced `data/validation_report.json` summarising symbol coverage (86/162 passing) and enumerating remediation gaps (19 missing parquet directories, 57 symbols outside tolerance). Narrative captured in `docs/data_quality_report_rl.md`.
   - HPO checkpoints (MLP72, LSTM62, GRU93) staged under `models/sl_checkpoints/` with scalers; `scripts/test_sl_checkpoint_loading.py` + `reports/sl_checkpoint_validation.json` verify parity.
   - `scripts/benchmark_sl_inference.py` demonstrates <0.2 ms/sample GPU latency (`reports/sl_inference_benchmarks.json`) meeting the Phase 0 performance target.

- **Phase 1 environment hardening (Weeks 2-4):**
   - Task 1.2 ✅ Feature engineering stack (`FeatureExtractor`, `RegimeIndicators`) wired into `TradingEnvironment`, benchmarked at P95 <2 ms, regression suites (`tests/test_feature_extractor.py`, `tests/test_trading_env.py`) fully green.
   - Task 1.3 ✅ Reward function overhaul via `RewardShaper` (7-component reward) plus diagnostics (`scripts/analyze_reward_signals.py`), all unit tests passing in `.venv`.
   - Task 1.4 ✅ Portfolio state management upgrades (`PortfolioManager`) enforcing risk constraints with telemetry exports and monitoring dashboard (`scripts/monitor_environment_performance.py`).

- **Phase 2 Task 2.1 shared encoder (Week 5, October 6, 2025):**
   - 4-layer transformer encoder in `core/rl/policies/feature_encoder.py` finalized with 15-unit test suite and 100% coverage.
   - `scripts/benchmark_feature_encoder.py` records GPU batch-32 P95 latency 2.08 ms (target <10 ms), throughput 25.7k samples/sec, activation memory 18.9 MB (`analysis/reports/feature_encoder_benchmark.json`).
   - `scripts/test_encoder_integration.py` sanitizes raw parquet schemas into `analysis/integration_cache/` and validates AAPL/GOOGL/MSFT rollouts (5×100 + 2×30 episodes) with zero NaN/Inf across single- and multi-env batches.

- **Upcoming milestones:**
   - Task 2.2 – SymbolAgent actor-critic implementation and parameter budget validation.
   - Task 2.3 – Package exports and API surfacing (`core/rl/policies/__init__.py`, `training/rl/__init__.py`).
   - Task 2.4 – Weight sharing + initialization utilities following SymbolAgent delivery.

---

## 2. Strategic Roadmap

### Long-Term Vision

**Mission:** Build a profitable, autonomous AI trading system that learns from market patterns and adapts to changing regimes.

### Strategic Pivot History

#### Original Approach (Abandoned May 2025)
- **Technology:** LLM/LoRA fine-tuning (Llama 3.2 3B)
- **Approach:** Natural language reasoning about market conditions
- **Results:** 
  - Training loss: 2.4-2.7
  - F1-score: 0.60 (60% actionable decisions)
  - Validation performance poor
- **Decision:** Pivot to custom neural networks

#### Current Approach (May 2025 - Present)
- **Technology:** Custom PyTorch neural networks
- **Phase 1:** Supervised learning baseline (MLP, LSTM, GRU, CNN-LSTM)
- **Phase 2:** Hyperparameter optimization (Optuna)
- **Phase 3:** Backtesting validation
- **Phase 4 (Future):** Reinforcement learning with multi-agent architecture

### Strategic Rationale for NN/RL

**Why Neural Networks?**
- Direct pattern recognition in numerical data
- Proven effectiveness in financial modeling
- Efficient training and inference
- Explicit feature engineering control

**Why Reinforcement Learning (Future)?**
- Optimizes for actual trading objectives (profit, Sharpe, drawdown)
- Learns action sequences (entry, hold, exit timing)
- Adapts to market regime changes
- Natural fit for sequential decision-making

**Why Multi-Agent (Future)?**
- Distributed decision-making (143 symbol agents)
- Portfolio-level coordination (master agent)
- Shared feature encoders (efficiency)
- Scalability and modularity

### NN → RL Integration Plan

**Hybrid Approach:**
1. **SL Models as Auxiliary Signals:** HPO checkpoints provide probability grids to RL agents
2. **Warm-Start RL Policies:** Initialize actor heads with SL decision boundaries
3. **KL Divergence Regularization:** Prevent early RL training from diverging too far from SL priors
4. **Progressive Autonomy:** Gradually reduce SL influence as RL learns

**Rationale:**
- Leverage 10-11× recall improvement from HPO
- Avoid starting RL from random initialization
- Accelerate early RL training
- Maintain some SL signal quality during RL exploration

**Status:** ON HOLD pending Phase 4 remediation

---

## 3. Phase Completion History

### Phase 0: Historical Data Collection ✅

**Duration:** August 2025  
**Status:** Complete

**Objectives:**
- Collect 2 years of historical 1-hour bar data
- Expand symbol universe to 143 symbols
- Integrate sentiment analysis

**Deliverables:**
- ✅ 143 symbols × ~4,380 bars = ~627,000 raw bars
- ✅ Parquet storage format established
- ✅ FinBERT sentiment scores for all symbols
- ✅ Data validation pipeline

**Key Decisions:**
- Standardized on 1-hour bars (balance of granularity and data availability)
- Selected 143-symbol universe covering stocks, ETFs, crypto, commodities
- Implemented Parquet-based storage for efficiency

---

### Phase 1: NN Infrastructure & Baselines ✅

**Duration:** May - August 2025  
**Status:** Complete

**Objectives:**
- Design and implement 4 neural network architectures
- Build comprehensive data preparation pipeline
- Establish training infrastructure with MLflow
- Integrate backtesting engine

**Deliverables:**
- ✅ `core/models/nn_architectures.py` - 4 model architectures
- ✅ `core/data_preparation_nn.py` - Complete data pipeline
- ✅ `training/train_nn_model.py` - 1000+ line training script
- ✅ `core/backtesting/engine.py` - Event-driven backtesting
- ✅ MLflow experiment tracking integration
- ✅ Enhanced logging and reporting system

**Key Achievements:**
- 23-feature specification finalized
- Asset ID embedding strategy implemented
- Focal loss and sample weighting for class imbalance
- GPU optimization utilities (`utils/gpu_utils.py`)

**Initial Baseline Results (Before HPO):**
- MLP: Validation F1+ 0.025-0.030, Recall 3-6%
- LSTM: Validation F1+ 0.028-0.032, Recall 4-7%
- GRU: Similar to LSTM
- CNN-LSTM: Underperformed expectations

---

### Phase 2: Baseline Training Campaign ✅

**Duration:** September 2025  
**Status:** Complete

**Objectives:**
- Train baseline models for each architecture
- Establish performance benchmarks
- Validate training infrastructure

**Deliverables:**
- ✅ Baseline checkpoints for MLP, LSTM, GRU, CNN-LSTM
- ✅ Baseline performance metrics documented
- ✅ Training/validation curves analyzed

**Results:**
- Severe overfitting observed (train F1+ 0.15-0.20, val F1+ 0.03-0.06)
- Poor recall on positive class (3-6%)
- Clear need for hyperparameter optimization
- Data enhancement identified as priority

**Key Decisions:**
- Proceed with HPO rather than manual tuning
- Focus on LSTM and GRU (best baseline performance)
- Implement comprehensive HPO framework

---

### Phase 3: HPO Optimization ✅

**Duration:** September 2025  
**Status:** Complete (Major Breakthrough)

#### 3.1 Data Enhancement
- Expanded to 143 symbols (from 50)
- Extended to 2 years of data (from 1)
- Improved positive class ratio to 6.9% (from 0.6%)
- October 2025 refresh: extended to 24h horizon with +1.5% / −3.0% thresholds, raising positives to 24.3%
- Result: 1,881,363 training sequences in latest run

#### 3.2 HPO Campaigns

**MLP Campaign:**
- Trials: 100
- Best: Trial 72
- Validation Metrics: F1+ 0.306, Precision 0.237, Recall 0.517
- Improvement: 10× over baseline

**LSTM Campaign:**
- Trials: 100
- Best: Trial 62
- Validation Metrics: F1+ 0.289, Precision 0.224, Recall 0.630
- Improvement: 11× over baseline
- Training stopped at epoch 1 (early stopping)

**GRU Campaign:**
- Trials: 100
- Best: Trial 93
- Validation Metrics: F1+ 0.269, Precision 0.209, Recall 0.530
- Improvement: 9× over baseline

**CNN-LSTM Campaign:**
- Trials: 100
- Poor performance overall
- Architectural issues identified

#### 3.3 Critical Discovery: HPO Early Stopping Optimality

**Pattern Observed:**
- Best HPO trials stopped at epochs 1-4
- Retraining for 56-76 epochs caused catastrophic regression
- Trial 62 LSTM: 63% recall at epoch 1, degraded to 4.4% at epoch 76 (14× worse)

**Hypothesis:**
- Early stopping during HPO captured peak generalization
- Extended training caused overfitting to train distribution
- Train/val temporal split created distribution mismatch

**Decision:**
- Use HPO trial checkpoints directly (epochs 1-4)
- DO NOT retrain HPO-selected hyperparameters
- Document as "HPO Early Stopping Optimality" pattern

#### 3.4 Test Set Evaluation

**Execution:** September 30, 2025  
**Dataset:** 131,811 held-out test sequences

**Results:**
| Model | F1+ | Precision | Recall | ROC-AUC | Epoch |
|-------|-----|-----------|--------|---------|-------|
| MLP Trial 72 | 0.306 | 0.237 | 0.415 | 0.866 | 4 |
| LSTM Trial 62 | 0.289 | 0.224 | 0.400 | 0.855 | 1 |
| GRU Trial 93 | 0.269 | 0.209 | 0.374 | 0.844 | 1 |

**Interpretation:**
- Strong generalization to held-out test set
- MLP Trial 72 selected as primary model (highest F1+)
- Test recall 40-41% (compared to 52-63% validation)
- Some degradation from validation, but still strong

**Key Achievement:** Successfully validated models on completely unseen data

---

### Phase 4: Backtesting Campaign ⚠️ FAILED

**Duration:** October 2025  
**Status:** Critical failure - remediation in progress

#### 4.1 Backtesting Configuration

**Script:** `scripts/backtest_hpo_production_models.py`

**Parameters:**
- Backtest period: January 1, 2023 - October 1, 2025 (~2.75 years)
- Symbols: 143
- Initial capital: $100,000 per symbol
- Commission: 0.10% per trade
- Slippage: 5 basis points
- Signal threshold: 0.6
- Max holding period: 8 hours
- Position sizing: $10,000 per trade

#### 4.2 Results Summary

**Execution Date:** October 4, 2025

| Model | Total Return | Sharpe | Sortino | Max DD | Trades | Win Rate |
|-------|--------------|--------|---------|--------|--------|----------|
| MLP Trial 72 | **-88.05%** | -0.04 | -0.06 | -97.21% | 10,495 | 38.2% |
| LSTM Trial 62 | **-92.60%** | -0.02 | -0.03 | -98.35% | 11,426 | 36.8% |
| GRU Trial 93 | **-89.34%** | -0.03 | -0.04 | -97.58% | 8,529 | 37.5% |

**Detailed Metrics (MLP Trial 72):**
- Total trades: 10,495
- Winning trades: 4,009 (38.2%)
- Losing trades: 6,486 (61.8%)
- Average profit per win: +$147.23
- Average loss per loss: -$231.45
- Profit factor: 0.41
- Total commission paid: ~$104,950
- Final portfolio value: $11,950

#### 4.3 Root Cause Analysis

**Primary Hypotheses:**

1. **Distribution Shift:**
   - Train/val/test period: Oct 2023 - Oct 2025
   - Backtest period: Jan 2023 - Oct 2025
   - Early 2023 data NOT in training set
   - Potential regime differences

2. **Transaction Cost Erosion:**
   - 10,495 trades × 0.10% commission = ~$105k in costs
   - High trade frequency (3,800 trades/year)
   - Each trade needs >0.15% profit just to break even

3. **False Positive Rate:**
   - Precision 23.7% means 76.3% of BUY signals are wrong
   - High churn from frequent entries and exits
   - Signal threshold 0.6 may be too low

4. **Model Calibration:**
   - Test set recall 41.5% suggests model finds many opportunities
   - But backtest shows most are unprofitable
   - Possible mismatch between label definition and profitability

5. **Holding Period Constraint:**
   - Fixed 8-hour max holding may exit winners early
   - Or hold losers too long
   - Inflexible risk management

#### 4.4 Diagnostic Experiments (In Progress)

**Experiment 1: Threshold Optimization**
- Test signal_threshold: 0.5, 0.6, 0.7, 0.8, 0.9
- Hypothesis: Higher threshold reduces false positives
- Expected: Fewer trades, better win rate

**Experiment 2: Regime Filters**
- Add VIX-based volatility filter
- Only trade when VIX > 15 (sufficient volatility)
- Hypothesis: Avoid choppy, low-volatility periods

**Experiment 3: Cost-Sensitive Trade Filters**
- Require minimum expected profit >1.0% (2× commission)
- Filter out marginal signals
- Hypothesis: Reduce unprofitable churn

**Experiment 4: Distribution Analysis**
- Compare feature distributions: train vs backtest
- Identify regime shifts
- Hypothesis: Early 2023 has different characteristics

**Status:** Experiments queued, awaiting execution

---

## 4. Phase 3 HPO Results Analysis

### Comprehensive HPO Campaign Summary

**Duration:** September 2025  
**Framework:** Optuna 4.3.0  
**Compute:** RTX 5070 Ti 16GB GPU  
**Dataset:** 878,740 sequences (143 symbols, 2 years)

### MLP Campaign

**Trial Count:** 100  
**Best Trial:** 72  
**Study Database:** `hpo_studies/mlp_study.db`

**Best Hyperparameters:**
```yaml
model_type: mlp
hidden_layers: [256, 128, 64]
dropout_rate: 0.3
activation: relu
batch_normalization: true
learning_rate: 0.0005
batch_size: 512
optimizer: adamw
weight_decay: 0.0001
loss_function: focal
focal_alpha: 0.25
focal_gamma: 2.0
lr_scheduler: reduce_on_plateau
early_stopping_patience: 10
```

**Performance:**
- Validation F1+: 0.306
- Validation Precision: 0.237
- Validation Recall: 0.517
- ROC-AUC: 0.866
- Training epoch: 4

**Key Insights:**
- 3-layer architecture optimal (2-layer and 4-layer underperformed)
- Moderate dropout (0.3) prevented overfitting
- Focal loss critical for class imbalance
- AdamW with small weight decay provided regularization

### LSTM Campaign

**Trial Count:** 100  
**Best Trial:** 62  
**Study Database:** `hpo_studies/lstm_study.db`

**Best Hyperparameters:**
```yaml
model_type: lstm_attention
lstm_layers: 2
lstm_hidden_units: 128
attention_dim: 64
attention_heads: 4
dropout_rate: 0.4
learning_rate: 0.001
batch_size: 256
optimizer: adamw
weight_decay: 0.0001
loss_function: focal
focal_alpha: 0.25
focal_gamma: 3.0
lr_scheduler: cosine_annealing
early_stopping_patience: 5
```

**Performance:**
- Validation F1+: 0.289
- Validation Precision: 0.224
- Validation Recall: 0.630 (BEST RECALL)
- ROC-AUC: 0.855
- Training epoch: 1 (stopped early)

**Key Insights:**
- 2-layer LSTM optimal (1-layer underfitted, 3-layer overfitted)
- Multi-head attention (4 heads) captured complex patterns
- Higher dropout (0.4) necessary for recurrent networks
- Cosine annealing LR schedule effective
- **STOPPED AT EPOCH 1** - critical early stopping case

### GRU Campaign

**Trial Count:** 100  
**Best Trial:** 93  
**Study Database:** `hpo_studies/gru_study.db`

**Best Hyperparameters:**
```yaml
model_type: gru_attention
gru_layers: 2
gru_hidden_units: 64
attention_dim: 32
attention_heads: 2
dropout_rate: 0.35
learning_rate: 0.0008
batch_size: 384
optimizer: adamw
weight_decay: 0.00005
loss_function: focal
focal_alpha: 0.3
focal_gamma: 2.5
lr_scheduler: reduce_on_plateau
early_stopping_patience: 8
```

**Performance:**
- Validation F1+: 0.269
- Validation Precision: 0.209
- Validation Recall: 0.530
- ROC-AUC: 0.844
- Training epoch: 1

**Key Insights:**
- Smaller hidden units (64) vs LSTM (128) - GRU more parameter-efficient
- Fewer attention heads (2) sufficient
- Lighter architecture faster to train
- Also stopped at epoch 1

### CNN-LSTM Campaign

**Trial Count:** 100  
**Best Trial:** 45  
**Study Database:** `hpo_studies/cnn_lstm_study.db`

**Performance:**
- Validation F1+: 0.182
- Validation Precision: 0.145
- Validation Recall: 0.247
- ROC-AUC: 0.782

**Status:** Significantly underperformed other architectures

**Hypothesis:**
- 1D CNNs may not capture temporal dependencies effectively for this problem
- Local feature extraction less important than long-range dependencies
- Hybrid architecture added complexity without benefit

**Decision:** Deprioritized for production use

### HPO vs Baseline Comparison

| Model | Baseline F1+ | HPO F1+ | Improvement | Baseline Recall | HPO Recall | Improvement |
|-------|--------------|---------|-------------|-----------------|------------|-------------|
| MLP | 0.030 | 0.306 | **10.2×** | 0.05 | 0.517 | **10.3×** |
| LSTM | 0.032 | 0.289 | **9.0×** | 0.06 | 0.630 | **10.5×** |
| GRU | 0.028 | 0.269 | **9.6×** | 0.056 | 0.530 | **9.5×** |

**Conclusion:** HPO produced 9-11× improvement across all metrics

### Parameter Importance Analysis

**Optuna Feature Importance (aggregated across campaigns):**

1. **Learning Rate:** 32% importance
   - Optimal range: 0.0005 - 0.001
   - Too high: Training instability
   - Too low: Slow convergence

2. **Dropout Rate:** 24% importance
   - Optimal range: 0.3 - 0.4
   - Critical for preventing overfitting

3. **Focal Gamma:** 18% importance
   - Optimal range: 2.0 - 3.0
   - Higher gamma focuses on hard examples

4. **Hidden Units / Layers:** 15% importance
   - Architecture matters, but less than regularization

5. **Batch Size:** 11% importance
   - Larger batches (256-512) more stable

### Retraining Experiment Results

**Hypothesis:** Can we improve HPO results by retraining with selected hyperparameters for more epochs?

**Experiment:**
- Selected Trial 62 LSTM hyperparameters
- Retrained from scratch for 76 epochs
- Used same training data

**Results:**
| Epoch | Val F1+ | Val Precision | Val Recall | Notes |
|-------|---------|---------------|------------|-------|
| 1 (HPO) | 0.289 | 0.224 | 0.630 | Original HPO checkpoint |
| 10 | 0.156 | 0.187 | 0.135 | Degradation begins |
| 30 | 0.089 | 0.142 | 0.063 | Severe degradation |
| 56 | 0.045 | 0.098 | 0.031 | Catastrophic |
| 76 (final) | 0.041 | 0.091 | 0.029 | **21.7× worse recall** |

**Conclusion:** Extended training harmful; HPO early stopping was optimal

**Root Cause:**
- Train/val distribution mismatch (temporal split)
- Model optimizes for training distribution
- Diverges from validation distribution with more epochs
- Early stopping prevents overfitting

**Decision:** Use HPO checkpoints directly, never retrain

---

## 5. Phase 4 Backtesting Campaign Results

### Executive Summary

**Date:** October 4, 2025  
**Status:** ❌ FAILED - Catastrophic losses across all models  
**Impact:** Critical setback requiring fundamental strategy reassessment

### Detailed Results

#### MLP Trial 72 Backtest

**Portfolio Performance:**
- Initial Capital: $100,000
- Final Value: $11,950.23
- Total Return: **-88.05%**
- Peak Value: $103,245 (early 2023)
- Max Drawdown: -97.21%

**Risk Metrics:**
- Sharpe Ratio: -0.04
- Sortino Ratio: -0.06
- Calmar Ratio: -0.91
- Volatility (annual): 42.3%

**Trade Statistics:**
- Total Trades: 10,495
- Winning Trades: 4,009 (38.2%)
- Losing Trades: 6,486 (61.8%)
- Average Win: +$147.23
- Average Loss: -$231.45
- Profit Factor: 0.41
- Win Rate: 38.2%

**Signal Quality:**
- BUY Precision: 23.7% (test set)
- Actual profitable trades: 38.2%
- Trade frequency: ~3,800/year
- Average holding period: 6.2 hours

**Cost Analysis:**
- Total commission: ~$104,950 (0.10% × 10,495 trades)
- Estimated slippage: ~$5,248 (5 bps × 10,495 trades)
- Total transaction costs: ~$110,198
- **Transaction costs exceeded total PnL**

#### LSTM Trial 62 Backtest

**Portfolio Performance:**
- Final Value: $7,400.15
- Total Return: **-92.60%**
- Max Drawdown: -98.35%

**Risk Metrics:**
- Sharpe Ratio: -0.02
- Sortino Ratio: -0.03

**Trade Statistics:**
- Total Trades: 11,426
- Winning Trades: 4,205 (36.8%)
- Losing Trades: 7,221 (63.2%)
- Profit Factor: 0.38

**Analysis:**
- Highest trade count (most aggressive)
- Lowest win rate (36.8%)
- Worst overall performance

#### GRU Trial 93 Backtest

**Portfolio Performance:**
- Final Value: $10,660.82
- Total Return: **-89.34%**
- Max Drawdown: -97.58%

**Risk Metrics:**
- Sharpe Ratio: -0.03
- Sortino Ratio: -0.04

**Trade Statistics:**
- Total Trades: 8,529
- Winning Trades: 3,198 (37.5%)
- Losing Trades: 5,331 (62.5%)
- Profit Factor: 0.43

**Analysis:**
- Lowest trade count (most conservative)
- Slightly better profit factor than others
- Still catastrophic losses

### Cross-Model Comparison

| Metric | MLP 72 | LSTM 62 | GRU 93 | Notes |
|--------|--------|---------|--------|-------|
| Total Return | -88.05% | -92.60% | -89.34% | All catastrophic |
| Sharpe | -0.04 | -0.02 | -0.03 | All terrible |
| Max DD | -97.21% | -98.35% | -97.58% | Unacceptable |
| Trades | 10,495 | 11,426 | 8,529 | High churn |
| Win Rate | 38.2% | 36.8% | 37.5% | Below 40% |
| Profit Factor | 0.41 | 0.38 | 0.43 | All <1.0 |

**Interpretation:**
- No meaningful difference between models
- All failed catastrophically
- Suggests systematic issue, not model-specific

### Distribution Shift Analysis

**Train/Val/Test Period:** Oct 2023 - Oct 2025  
**Backtest Period:** Jan 2023 - Oct 2025 (includes early 2023 NOT in training)

**Hypothesis:** Early 2023 had different market characteristics

**Evidence:**
- Initial capital actually grew in early 2023 (peak $103k)
- Losses accelerated in mid-2023 onward
- Suggests models may have learned patterns specific to Oct 2023+ period

**Action Item:** Compare feature distributions across periods

### Transaction Cost Impact

**Scenario Analysis (MLP Trial 72):**

| Scenario | Commission | Trades | Total Cost | Hypothetical PnL | Net Result |
|----------|------------|--------|------------|------------------|------------|
| Actual | 0.10% | 10,495 | $104,950 | ? | -$88,050 |
| Zero Cost | 0.00% | 10,495 | $0 | ? | ? |

**Calculation:** If final loss was -$88,050 and transaction costs were $110k, then gross PnL before costs ≈ +$22k

**Implication:** Model may have slight positive edge, but transaction costs destroy it

**Conclusion:** Must reduce trade frequency and/or improve win rate to overcome costs

---

## 6. Decision Log

### Major Decision History

#### Decision 1: Strategic Pivot from LLM/LoRA to NN/RL
**Date:** May 2025  
**Context:** LLM/LoRA approach achieving only 60% actionable decisions with high loss (2.4-2.7)

**Decision:** Abandon LLM/LoRA, pivot to custom neural networks

**Rationale:**
- LLMs not designed for numerical time-series patterns
- Adapter approach underperformed expectations
- Direct pattern recognition more efficient
- NN infrastructure more mature

**Outcome:** Successful pivot; NN models achieved 9-11× improvement

**Status:** ✅ Validated

---

#### Decision 2: Focus on Supervised Learning Before RL
**Date:** May 2025  
**Context:** RL requires strong reward signal and stable environment

**Decision:** Build supervised learning baseline first, then transition to RL

**Rationale:**
- SL easier to debug and validate
- SL checkpoints can inform RL warm-start
- Establish performance ceiling before RL complexity
- Validate data quality and feature engineering

**Outcome:** SL achieved strong offline metrics (F1+ 0.3, recall 0.4-0.6)

**Status:** ⚠️ Partial success (offline metrics good, trading performance poor)

---

#### Decision 3: 143-Symbol Universe Expansion
**Date:** August 2025  
**Context:** Original 50-symbol universe too limited

**Decision:** Expand to 143 symbols covering stocks, ETFs, crypto, commodities

**Rationale:**
- Diversification improves generalization
- More data points for training
- Broader market regime coverage
- Enables portfolio-level strategies

**Outcome:** Successfully integrated; data quality maintained

**Status:** ✅ Validated

---

#### Decision 4: Profit Target & Horizon Adjustments
**Date:** September 2025 (initial), October 2025 (refresh)  
**Context:** Initial +5% target yielded only 0.6% positives; later, 8h/+2.5% setup still produced poor backtests

**Decision:**
- September: Decrease profit target to +2.5% (8h horizon) to reach ~6.9% positives
- October: Extend prediction horizon to 24h, lower profit target to +1.5%, widen stop to −3.0%

**Rationale:**
- +5% target starved the dataset; +2.5% enabled HPO success but remained sparse for trading
- Backtesting showed SL models firing too many weak signals; longer horizon plus easier target increases high-confidence positives
- New configuration provides ~4× more positive examples (24.3%) without sacrificing quality checks

**Outcome:**
- Classification HPO benefited from the interim +2.5% setting (F1+ 0.28-0.31)
- New 24h dataset delivered 1.88M sequences; trading re-evaluation pending

**Status:** ⚠️ Monitoring — offline metrics to be re-measured on refreshed dataset

---

#### Decision 5: Use HPO Checkpoints Directly (Don't Retrain)
**Date:** September 2025  
**Context:** Retraining Trial 62 hyperparameters for 76 epochs caused 21.7× recall degradation

**Decision:** Use HPO trial checkpoints at epochs 1-4 directly; never retrain

**Rationale:**
- HPO early stopping captured peak generalization
- Extended training causes overfitting to training distribution
- Train/val temporal split creates distribution mismatch
- Validation metrics during HPO are reliable

**Outcome:** Prevented further performance degradation

**Status:** ✅ Validated

**Pattern Documented:** "HPO Early Stopping Optimality"

---

#### Decision 6: Pivot to RL Development as SL is Failing
**Date:** October 2025  
**Context:** Backtesting revealed -88% to -93% losses despite strong SL metrics

**Decision:** Pivot to RL system development as SL issues are not redeemable

**Rationale:**
- Pivot to RL may overcome SL limitations
- RL optimizes for actual trading objectives
- SL backtest failures suggest fundamental issues
- Remediation experiments may inform RL design

**Outcome:** Decsion is taken to proceed

**Status:** 🔜 Queued for execution

---


#### Decision 7: Maintain Temporal Data Splitting (No Shuffling)
**Date:** May 2025  
**Context:** Time-series data requires chronological evaluation

**Decision:** Always use temporal splits (train/val/test in chronological order), never shuffle

**Rationale:**
- Prevents look-ahead bias
- Realistic evaluation of future performance
- Industry standard for time-series ML
- Aligns with actual deployment scenario

**Outcome:** Consistent methodology across all experiments

**Status:** ✅ Validated

---

#### Decision 9: Use Focal Loss for Class Imbalance
**Date:** June 2025  
**Context:** Historical 6.9% positive class ratio caused the model to bias toward negatives (still applicable for sparse subsets)

**Decision:** Implement Focal Loss with alpha=0.25, gamma=2.0-3.0

**Rationale:**
- Down-weights easy negatives
- Focuses on hard positives
- More effective than simple class weighting
- Proven in computer vision imbalanced datasets

**Outcome:** Significant improvement; validation F1+ increased from 0.03 to 0.28-0.31. With the October 2025 refresh (~24% positives), focal loss remains an optional lever for symbol-level imbalance.

**Status:** ✅ Validated

---

#### Decision 10: Asset ID Embedding (Simple Integer Mapping)
**Date:** June 2025  
**Context:** Need to train single model across 143 symbols

**Decision:** Use simple integer symbol-to-ID mapping with `nn.Embedding`

**Rationale:**
- Simplest approach, easy to implement
- Allows model to learn symbol-specific patterns
- No external embeddings needed
- Scales to any number of symbols

**Alternatives Considered:**
- Metadata-based embeddings (sector, market cap, etc.)
- Pre-trained symbol embeddings
- One-hot encoding (too sparse)

**Outcome:** Successfully integrated; embedding dimension 8-16 optimal

**Status:** ✅ Validated

---
#### Decision 11: RL System Proceed/Pivot
**Context:** SL backtesting failed; RL development on hold

**Options:**
1. **Proceed with RL after SL remediation** - Original plan
2. **Pivot to alternative approach** - Ensemble methods, hybrid models
3. **Abandon autonomous trading** - Focus on signal generation only

**Considerations:**
- RL compute cost high (~$5k-10k GPU hours)
- RL debugging complexity
- Opportunity cost of alternatives
- Learning value regardless of outcome

**Timeline:** Validated for implementation

**Status:** 🔜 Queued for execution
---

## 7. Key Learnings & Patterns

### Learning 1: HPO Early Stopping Optimality

**Discovery:** Best HPO trials often stop at epochs 1-4; retraining degrades performance

**Pattern:**
- Optuna optimizes validation metrics, not training epochs
- Early stopping prevents overfitting when train/val distributions differ
- Extended training optimizes for training distribution, diverges from validation

**When It Applies:**
- Temporal data splits (distribution shift)
- Class imbalance problems
- Complex time-series patterns

**Actionable Guidance:**
- Trust validation metrics over epoch count
- Don't assume "more training is better"
- If HPO finds low-epoch solutions, investigate data distribution
- Use HPO checkpoints directly

**Evidence:**
- LSTM Trial 62: 63% recall at epoch 1, 2.9% at epoch 76 (21.7× degradation)
- MLP Trial 72: Stopped at epoch 4, achieved best test performance
- GRU Trial 93: Stopped at epoch 1, strong test performance

**Documentation:** Fully documented in systemPatterns.md (now CONSOLIDATED_1)

---

### Learning 2: Classification Metrics ≠ Trading Performance

**Discovery:** Strong offline metrics (F1+ 0.3, recall 0.4-0.6) did NOT translate to profitable trading

**Gap:**
- Test set: F1+ 0.306, Recall 0.415, ROC-AUC 0.866
- Backtest: -88% return, Sharpe -0.04

**Root Causes:**
1. **Distribution Shift:** Test set vs backtest period differences
2. **Transaction Costs:** 0.10% commission + 5 bps slippage erodes gains
3. **Label Mismatch:** Forward-looking profit labels don't account for execution reality
4. **High Churn:** 10k+ trades overwhelm any edge

**Actionable Guidance:**
- Always validate with realistic backtesting
- Classification metrics are necessary but not sufficient
- Must model transaction costs in evaluation
- Consider online learning to adapt to regime shifts

**Impact:** Critical for understanding ML model limitations in trading

---

### Learning 3: Transaction Costs Dominate at High Frequency

**Discovery:** 10,495 trades × 0.10% commission ≈ $105k, comparable to total capital

**Calculation:**
- Gross PnL before costs: ~+$22k
- Transaction costs: ~$110k
- Net PnL: -$88k

**Implication:** Even slight positive edge destroyed by transaction costs

**Actionable Guidance:**
- Minimize trade frequency (target <100 trades/year)
- Require high conviction signals (threshold >0.7)
- Model transaction costs in loss function during training
- Consider cost-sensitive learning objectives

**Evidence:** All three models (MLP, LSTM, GRU) failed similarly despite different architectures

---

### Learning 4: Multi-Symbol Training Feasible

**Discovery:** Successfully trained models on 143 symbols simultaneously using asset ID embeddings

**Benefits:**
- Shared feature encoder learns cross-asset patterns
- More training data (878k sequences vs ~6k per symbol)
- Single model deployment vs 143 separate models
- Transfer learning across symbols

**Challenges:**
- Asset embedding dimension tuning (8-16 optimal)
- Symbol-specific overfitting still possible
- Requires large datasets

**Actionable Guidance:**
- Use simple integer mapping for asset IDs
- Embedding dimension ~sqrt(num_symbols) as starting point
- Monitor per-symbol performance during validation

**Status:** Approach validated; will continue for RL system

---

### Learning 5: Data Enhancement Critical Foundation

**Discovery:** Expanding from 50 to 143 symbols and improving positive class ratio from 0.6% to 6.9% enabled HPO success; the October 2025 refresh pushed this further to 24.3% via 24h/1.5%/−3% labeling.

**Evidence:**
- Baseline with 50 symbols: F1+ 0.03, Recall 0.05
- Enhanced with 143 symbols: F1+ 0.28-0.31, Recall 0.40-0.63 (10× improvement)

**Key Factors:**
1. More symbols → more diverse patterns (now 160 symbols processed cleanly)
2. Progressive label strategy refinements → higher quality positives (now ~24% without extreme imbalance)
3. Longer timeframe → more market regimes captured for both SL and future RL

**Actionable Guidance:**
- Invest in data quality before model complexity
- Target a positive class ratio that supports trading realism (20-30% working target; monitor per-symbol skew)
- Expand symbol universe for generalization and ensure refreshed datasets propagate to downstream experiments

**Status:** Data enhancement was necessary but not sufficient (backtest still failed)

---

## 8. Current Challenges

### Challenge 1: Backtest Failure Despite Strong Offline Metrics ⚠️

**Problem:** Models achieve F1+ 0.3, recall 0.4-0.6 on test set, but lose 88-93% in backtesting

**Impact:** Critical - blocks production deployment and RL development

**Root Causes (Hypothesized):**
1. Distribution shift between test set and backtest period
2. Transaction costs (0.10% + 5bps) overwhelming edge
3. Label definition not aligned with profitable trading
4. High false positive rate (precision 23%)

**Status:** Active investigation with remediation experiments

**Priority:** P0 - Highest priority

---

### Challenge 2: High Trade Frequency (Churn) 🔧

**Problem:** Models generate 8,500-11,500 trades over 2 years (~3,800-4,200/year)

**Impact:** Transaction costs ~$110k, exceeds total capital

**Drivers:**
- Signal threshold 0.6 too low (catches many marginal signals)
- 8-hour max holding period causes frequent exits
- High model recall (40-60%) generates many entries

**Potential Solutions:**
1. Increase signal threshold to 0.7-0.9
2. Implement regime filters (only trade high-volatility periods)
3. Add minimum expected profit filter (>1%)
4. Increase max holding period to 24-48 hours

**Status:** Experiments queued

**Priority:** P0 - Critical for viability

---

### Challenge 3: Distribution Shift (Train vs Backtest Period) 🔍

**Problem:** Test set period (Oct 2023-Oct 2025) differs from full backtest period (Jan 2023-Oct 2025)

**Evidence:**
- Early 2023 NOT in training data
- Backtest initially profitable in early 2023 (peak $103k)
- Losses accelerated mid-2023 onward

**Hypothesis:** Models learned patterns specific to Oct 2023+ period

**Investigation:**
- Compare feature distributions across periods
- Analyze regime characteristics (VIX, market breadth, sector rotation)
- Test walk-forward evaluation

**Status:** Analysis pending

**Priority:** P1 - Important for understanding failure

---

### Challenge 4: Label Definition vs Profitability Mismatch 🤔

**Problem:** Forward-looking profit labels (+5% target, -2% stop) may not align with actual profitable trading

**Concerns:**
1. Labels assume perfect entry/exit execution
2. Don't account for transaction costs
3. Don't consider holding period constraints
4. May label as positive opportunities that are unprofitable after costs

**Example:**
- Label: +5% move within 8 hours → positive
- Reality: Enter at bar close, exit 8 hours later, pay 0.15% round-trip costs
- Net: +5% - 0.15% = +4.85% before slippage

**Potential Solutions:**
1. Adjust profit target to account for costs (+6% instead of +5%)
2. Use actual trade outcomes as labels (requires backtesting data)
3. Implement cost-aware labeling

**Status:** Conceptual; no implementation yet

**Priority:** P2 - Medium priority

---

### Challenge 5: Precision Too Low (23-24%) ⚠️

**Problem:** Only 23-24% of BUY signals on test set are true positives

**Impact:** 76-77% of trades are false positives, wasting capital and incurring costs

**Trade-off:** High recall (40-60%) comes at expense of precision

**Potential Solutions:**
1. Increase signal threshold (trades off recall for precision)
2. Ensemble models (require multiple models to agree)
3. Add secondary filters (regime, momentum, volume)
4. Post-processing signal refinement

**Status:** Threshold experiments queued

**Priority:** P1 - Important for reducing false positives

---

## 9. Next Steps

### Immediate Actions (Next 7 Days)

#### Action 1: Task 2.2 SymbolAgent Delivery
**Timeline:** October 8, 2025  
**Priority:** P0

**Tasks:** Implement `core/rl/policies/symbol_agent.py` (actor-critic heads, action masking, PPO interfaces), accompany with `tests/test_symbol_agent.py` covering forward pass, parameter counts, and action masking, and wire configuration defaults.

#### Action 2: Task 2.3 Package Surface & Bench Validation
**Timeline:** October 9, 2025  
**Priority:** P0

**Tasks:** Update `core/rl/policies/__init__.py`, `core/rl/__init__.py`, and `training/rl/__init__.py` exports, add lightweight smoke benchmark to confirm combined encoder+agent latency stays <10 ms, and align docs/reference notebooks.

#### Action 3: Phase 4 Remediation Experiment Cycle
**Timeline:** October 10, 2025  
**Priority:** P1

**Tasks:** Execute threshold/regime filter experiments, capture outcomes in `analysis/reports/backtest_remediation_2025-10-10.md`, and decide whether RL-only path can proceed without further SL fixes.

---

### Medium-Term Actions (Next 90 Days)

#### Action 4: RL System Build-Out
**Timeline:** November 2025 - January 2026  
**Priority:** P0

**Prerequisites:**
- Phase 0 exit criteria met (data clean, checkpoints benchmarked, RL scaffolding committed)
- GPU resources scheduled for continuous training

**Phases:**
- Environment implementation and testing (Phase 1)
- Shared encoder pre-training and symbol PPO curriculum (Phase 2)
- Master agent MAPPO coordination (Phase 3)
- Walk-forward validation & stress testing (Phase 4)

---

### Long-Term Actions (90+ Days)

#### Action 8: Production Deployment
**Timeline:** Q1 2026  
**Priority:** P0

**Prerequisites:**
- Either SL remediation successful OR RL system trained
- Backtesting shows positive risk-adjusted returns (Sharpe >0.8)
- Risk management validated

**Tasks:**
- Deploy to Alpaca paper trading (live data, simulated execution)
- Monitor for 30 days
- Transition to live trading (if successful)

---

## Cross-References

**Related Consolidated Documents:**
- [CONSOLIDATED_1: Core Architecture & System Design](CONSOLIDATED_1_Architecture_and_System_Design.md)
- [CONSOLIDATED_2: Data Processing & Preparation Pipeline](CONSOLIDATED_2_Data_Processing_and_Preparation.md)
- [CONSOLIDATED_3: Neural Network Models & Training](CONSOLIDATED_3_Neural_Network_Models_and_Training.md)
- [CONSOLIDATED_4: Trading Strategy & Decision Engine](CONSOLIDATED_4_Trading_Strategy_and_Decision_Engine.md)

---

**Document Maintenance:**
- This consolidated document replaces: `activeContext.md`, `progress.md`, `decisionLog.md`, `phase3_hpo_results_analysis.md`, `phase3_completion_report.md`, `phase3_next_steps.md`, `strategic_plan_nn_rl.md`, `sl_to_rl_pivot_analysis.md`, diagnostic reports
- Update frequency: Weekly during active development
- Last consolidation: October 6, 2025
