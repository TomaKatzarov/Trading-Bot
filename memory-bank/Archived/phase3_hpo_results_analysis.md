# Phase 3: HPO Campaign Results & Retraining Analysis

**Date**: October 4, 2025  
**Status**: CRITICAL – Severe Backtesting Losses with HPO Checkpoints

## Executive Summary

Phase 3 HPO campaigns achieved breakthrough recall improvements (10-11x vs baseline), but subsequent retraining attempts revealed catastrophic performance regression. The first production-scale backtesting campaign (2023-10-02 → 2025-10-01) now confirms that despite strong classification metrics, all checkpoints and ensembles lose 88-93% of capital after realistic costs, exposing critical gaps in generalization, threshold calibration, and risk alignment.

## HPO Campaign Results (100 trials each)

### Best Trial Performance

**MLP (Trial 72):**
- Validation F1: 0.355 (35.5%)
- Validation Recall: 0.517 (51.7%)
- Validation Precision: 0.270 (27.0%)
- ROC-AUC: 0.749
- Best Epoch: 3
- Learning Rate: 0.0001073

**LSTM (Trial 62):**
- Validation F1: 0.329 (32.9%)
- Validation Recall: 0.630 (63.0%) ← HIGHEST
- Validation Precision: 0.222 (22.2%)
- ROC-AUC: 0.727
- Best Epoch: 1
- Learning Rate: 0.0000087

**GRU (Trial 93):**
- Validation F1: 0.334 (33.4%)
- Validation Recall: 0.549 (54.9%)
- Validation Precision: 0.240 (24.0%)
- ROC-AUC: 0.729
- Best Epoch: 4
- Learning Rate: 0.0003220

### Key Achievement
- **10-11x recall improvement** over Phase 2 baselines (from 5.6% to 51.7-63.0%)
- All models exceeded Phase 1 success criteria (Val F1 ≥ 0.25)

## Retraining Attempt Results

### Observation: "Premature" Stopping Pattern
All best trials stopped at epochs 1-4, which initially appeared problematic. Retraining was attempted with:
- Higher learning rates (0.0005-0.001)
- Extended training (min 15 epochs, patience 20)
- Stronger regularization

### Catastrophic Regression Results

| Model | HPO Val F1 | HPO Val Recall | Retrain Val F1 | Retrain Val Recall | Change |
|-------|-----------|---------------|---------------|-------------------|--------|
| GRU   | 0.334     | 0.549         | 0.209         | 0.178             | **-37% F1, -68% Recall** |
| LSTM  | 0.329     | 0.630         | 0.198         | 0.159             | **-40% F1, -75% Recall** |
| MLP   | 0.355     | 0.517         | 0.089         | 0.036             | **-75% F1, -93% Recall** |

### Retraining Metrics Detail

**GRU (Best Retrain Performance):**
- Best Epoch: 56
- Best Val F1: 0.209 (vs HPO: 0.334)
- Val Recall: 0.178 (vs HPO: 0.549)
- Train F1: 0.832 → **4.1x overfitting gap**
- Final LR: 0.0000625 (scheduler reduced it)

**LSTM:**
- Best Epoch: 76
- Best Val F1: 0.198 (vs HPO: 0.329)
- Val Recall: 0.159 (vs HPO: 0.630)
- Train F1: 0.904 → **4.7x overfitting gap**
- Final LR: 0.0000625

**MLP:**
- Best Epoch: 76
- Best Val F1: 0.089 (vs HPO: 0.355)
- Val Recall: 0.036 (vs HPO: 0.517)
- Train F1: 0.173 → **2.6x overfitting gap**
- Final LR: 0.0000625

## Root Cause Analysis

### Why HPO Models Were Better

1. **Optimal Early Stopping**: The 1-4 epoch "premature" stopping actually captured **peak generalization** before overfitting began

2. **Appropriate Learning Rates**: HPO-selected very low LRs (especially LSTM's 8.67e-6) prevented rapid overfitting

3. **Validation-Driven Optimization**: Optuna's TPE sampler found configurations that maximized validation metrics, not training metrics

### Why Retraining Failed

1. **Learning Rate Scheduler Failure**:
   - Started at 0.0005-0.001 (higher than HPO)
   - ReduceOnPlateau reduced to 0.0000625
   - Too high initially → overfitting
   - Too low finally → stuck in local minimum

2. **Extended Training Harmful**:
   - More epochs = more overfitting
   - 56-76 epochs vs optimal 1-4 epochs
   - Regularization insufficient to prevent memorization

3. **Fundamental Data Issue**:
   - Severe train/val distribution mismatch
   - Models memorize training patterns that don't generalize
   - Possible temporal drift (train = older data)

## Critical Discovery: Data Generalization Problem

### Evidence

**Overfitting Severity:**
- All models achieve 80-90% train F1
- But only 9-21% val F1
- 2.6x-4.7x performance gaps

**Pattern:**
- Quick training convergence (high train metrics in <20 epochs)
- Poor validation performance throughout
- No improvement with more training

### Hypotheses

1. **Train/Val Distribution Shift**:
   - Temporal split: train (Oct 2023-May 2025) vs val (May-Aug 2025)
   - Market regime changes between periods
   - Feature distributions differ

2. **Label Quality Issues**:
   - 2.5% profit target still too aggressive?
   - False positives in training labels
   - Validation period has different market conditions

3. **Feature Engineering Gaps**:
   - Current features capture training patterns but not generalizable signals
   - Need market regime indicators
   - Missing cross-asset relationships

## Decision: Use HPO Trial Checkpoints

### Rationale

The original HPO trial checkpoints (epochs 1-4) achieved:
- 3x-14x better validation performance than retrained models
- Better precision-recall balance
- Evidence of proper generalization (not overfitting)

### Recommendation

**DO NOT retrain. Use original HPO trial checkpoints as production models.**

Specifically:
- LSTM Trial 62 (epoch 1): 63.0% recall, 22.2% precision
- GRU Trial 93 (epoch 4): 54.9% recall, 24.0% precision
- MLP Trial 72 (epoch 3): 51.7% recall, 27.0% precision

## Test Set Evaluation (Completed October 4, 2025)

**Dataset**: `data/training_data_v2_final` test split (131,811 sequences, 24×23 feature tensor, asset IDs synchronized with production map)

**Evaluation Script**: `scripts/evaluate_hpo_models_on_test.py`

**Summary (Threshold = 0.50 unless noted)**

| Model (Trial) | Test F1⁺ | Test Recall⁺ | Test Precision⁺ | ROC-AUC | PR-AUC | Log-Loss | Optimal Threshold | F1⁺ (Optimal) | Notes |
|---------------|----------|---------------|------------------|---------|--------|----------|-------------------|---------------|-------|
| MLP (72)      | 0.306    | 0.415         | 0.242            | 0.866   | 0.239  | 0.2944   | 0.55              | 0.316         | Recall drop ~0.13 vs val; best overall balance |
| LSTM (62)     | 0.289    | 0.333         | 0.256            | 0.831   | 0.196  | 0.4032   | 0.50              | 0.289         | Holds precision, larger recall drop (~0.30) |
| GRU (93)      | 0.269    | 0.411         | 0.200            | 0.838   | 0.201  | 0.4849   | 0.50              | 0.269         | Recall resilient, precision lowest |

**Findings**

1. **Ordering preserved** – MLP > LSTM > GRU on F1⁺, consistent with validation results.
2. **Recall compression** – All models lose ~0.12–0.21 recall vs validation; however, positive-class F1 stays above 0.26.
3. **Calibration** – ROC-AUC between 0.83–0.87; PR-AUC between 0.20–0.24; log-loss indicates MLP retains best calibration.
4. **Threshold sweet spot** – MLP benefits from slightly higher threshold (0.55) yielding balanced precision/recall; LSTM/GRU optimum remains at default 0.50.
5. **Artifacts recorded** – JSON reports include full threshold grid, PR curve samples (≤500 points), inference speed, and metadata per checkpoint.

## Backtesting Campaign (Completed October 4, 2025)

**Run Command**: `python scripts/backtest_hpo_production_models.py --models mlp lstm gru ensemble_weighted_val ensemble_weighted_test --num-workers 6 --output-dir backtesting/results/full_campaign`

**Configuration Highlights**

- **Period**: 2023-10-02T00:00:00+00:00 → 2025-10-01T00:00:00+00:00 (two-year rolling window)
- **Universe**: 143 symbols (auto-filtered to symbols with complete hourly parquet coverage)
- **Capital & Sizing**: $100,000 initial equity, equal-weight entries capped at 10% each, max 20 concurrent positions
- **Execution Costs**: 0.10% commission per side plus 5bps adverse slippage baked into fills
- **Strategy Logic**: Long-only exposure using checkpoint signal probabilities with default entry threshold 0.50 and exit rules based on signal decay or max holding horizon

### Performance Summary

```
MODEL                          Total Return   Ann. Return   Sharpe   Sortino   Max DD   Calmar   Trades   Win %   Profit Factor
--------------------------------------------------------------------------------------------------------------------------------
MLP Trial 72                   -88.02%        -65.56%       -0.03    -0.13     -91.57%  -0.72    8,517    44.55   0.71
LSTM Trial 62                  -92.95%        -73.61%       -0.02    -0.13     -94.84%  -0.78   11,394    43.29   0.69
GRU Trial 93                   -92.53%        -72.84%       -0.02    -0.13     -94.57%  -0.77   11,096    43.25   0.71
Ensemble Weighted (Val F1)     -91.34%        -70.74%       -0.04    -0.15     -91.78%  -0.77    9,176    44.06   0.71
Ensemble Weighted (Test F1)    -91.53%        -71.06%       -0.04    -0.15     -91.92%  -0.77    9,031    43.83   0.71
SPY Buy-Hold (Baseline)        +59.89%        +26.45%       +1.47       —       -20.21%     —        —       —        —
```

### Key Findings

1. **Catastrophic Drawdowns** – Every model suffered >91% peak-to-trough drawdowns with negligible risk-adjusted performance (Sharpe ≈ -0.02 to -0.04).
2. **High Trade Volume Without Edge** – 8.5k–11.4k trades per model at <45% win rate and profit factors near 0.70 confirm negative expectancy after costs.
3. **Uniform Trade Distribution** – Best/worst trade magnitudes clustered around ±2.2%, indicating signal probabilities failed to differentiate trade quality or sizing.
4. **Ensembles Underperformed** – Weighted ensembles marginally lagged single checkpoints and inherited the same drawdown profile.
5. **Baseline Outperformance** – Passive SPY buy-hold returned +59.9% with Sharpe 1.47, underscoring the severity of model underperformance.

### Risk-Adjusted Ranking (Sharpe Ratio)

1. GRU Trial 93: -0.02 *(least negative Sharpe)*
2. LSTM Trial 62: -0.02
3. MLP Trial 72: -0.03
4. Ensemble Weighted (Test F1): -0.04
5. Ensemble Weighted (Val F1): -0.04

### Recommendation & Next Moves

- **Provisional Production Candidate**: GRU Trial 93, as the least bad performer, to anchor remediation experiments (threshold tuning, regime filters, risk overlays) before any deployment decision.
- **Strategic Pause**: Halt live deployment until strategy-level fixes address the negative expectancy and drawdown exposure.
- **Baseline Benchmark**: Retain SPY buy-hold as control for future backtesting comparisons.

### Artifacts & Logging

- Metrics JSON, trade-level CSV, and equity curve plots saved under `backtesting/results/full_campaign/` (timestamped 20251004_*`).
- Detailed execution logs recorded in `backtesting/results/full_campaign/backtest.log`, including data availability warnings and per-symbol instrumentation.

## Next Phase Requirements

### Critical Data Analysis

1. **Train/Val Distribution Comparison**:
   - Feature distributions (KS tests, histograms)
   - Positive class characteristics
   - Temporal patterns

2. **Label Quality Audit**:
   - Sample validation of profit targets
   - Check for look-ahead bias
   - Verify stop-loss logic

3. **Feature Importance Analysis**:
   - Identify overfitting features
   - Find generalizable signals
   - Consider feature selection

### Production Deployment Path

1. Extract HPO trial checkpoints for production use
2. Comprehensive test set evaluation
3. Backtesting with all three models
4. Ensemble strategy evaluation
5. Threshold optimization

### Alternative Exploration

1. **Profit Target Experimentation**: Test 3.0%, 3.5%, 4.0% targets
2. **Walk-Forward Validation**: Replace simple temporal split
3. **Cross-Validation**: K-fold with temporal awareness
4. **Feature Engineering V2**: Add market regime, volatility regime, cross-asset features

## Files Created During Retraining

- `training/config_templates/best_mlp_retrain.yaml`
- `training/config_templates/best_lstm_retrain.yaml`
- `training/config_templates/best_gru_retrain.yaml`
- `scripts/retrain_best_hpo_models.py`

**Status**: Deprecated - retrained models perform worse than HPO checkpoints

## Lessons Learned

1. **Trust the optimization process**: HPO found better solutions than manual intervention
2. **Early stopping isn't always "premature"**: Short training can capture peak generalization
3. **More training ≠ better models**: Especially with distribution shift
4. **Validation metrics are truth**: Don't dismiss low-epoch models with good validation performance
5. **Data quality > model complexity**: Current bottleneck is data, not architecture

## Action Items

- [x] Extract HPO trial checkpoints for production reuse
- [ ] Perform comprehensive data distribution analysis *(diagnose backtest failures, regime drift, cost sensitivity)*
- [x] Test set evaluation with HPO checkpoints
- [x] Backtesting campaign with all three models + ensembles
- [ ] Document deployment decision and rationale *(include remediation roadmap for GRU Trial 93)*
- [ ] Design and test mitigation plan (threshold re-optimization, trade filters, risk overlays) prior to any production go-live
