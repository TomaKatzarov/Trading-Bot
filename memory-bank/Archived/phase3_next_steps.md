# Phase 4: Production Model Deployment & Data Analysis

**Status**: READY TO EXECUTE  
**Priority**: HIGH  
**Blockers**: None

## Immediate Actions (Phase 4a: Model Extraction & Evaluation)

### Task 1: Extract HPO Trial Checkpoints

**Purpose**: Retrieve the optimal models from HPO campaigns for production use

**Steps**:
1. Query MLflow for best trials:
   - Trial 62 (LSTM): val_f1=0.329, val_recall=0.630
   - Trial 72 (MLP): val_f1=0.355, val_recall=0.517
   - Trial 93 (GRU): val_f1=0.334, val_recall=0.549

2. Download artifacts from MLflow:
   ```python
   import mlflow

   # LSTM Trial 62
   lstm_run = mlflow.get_run("<trial_62_run_id>")
   lstm_model_path = mlflow.artifacts.download_artifacts(
       run_id="<trial_62_run_id>",
       artifact_path="model"
   )

   # Similar for MLP and GRU
   ```

3. Copy to production directory:

```
models/production/
├── lstm_trial62_epoch1/
│   ├── model.pt
│   ├── config.yaml
│   ├── scaler.pkl
│   └── metrics.json
├── mlp_trial72_epoch3/
└── gru_trial93_epoch4/
```

4. Document model provenance:
   - HPO trial number
   - Training epoch
   - Hyperparameters
   - Validation metrics

### Task 2: Validate Checkpoint Integrity

**Purpose**: Ensure downloaded artifacts load correctly and produce expected metrics

**Steps**:
- Load each model with `training/train_nn_model.py --resume-from-checkpoint`
- Run quick validation pass on the stored validation split
- Compare metrics to MLflow logs (allow ±0.5% tolerance)
- Store verification report under `reports/phase4/model_validation/`

### Task 3: Deploy Models to Backtest Harness

**Purpose**: Generate production-grade evaluation for each checkpoint

**Steps**:
- Update `scripts/run_dummy_backtest.py` to accept production checkpoint paths
- Execute backtests across core validation symbols (AAPL, MSFT, NVDA, SPY, BTCUSD)
- Capture precision/recall/F1 along with trading KPIs
- Export consolidated results to `reports/phase4/backtesting_summary.json`

**Status (2025-10-04)**: ✅ Completed via `scripts/backtest_hpo_production_models.py` full-universe run (143 symbols, 2023-10-02 → 2025-10-01). Outcome: -88% to -93% total returns across all checkpoints and ensembles; see `backtesting/results/full_campaign/` and `memory-bank/phase3_hpo_results_analysis.md` for detailed analysis.

## Data Diagnostics (Phase 4b: Generalization Root Cause)

### Task 4: Train/Validation Distribution Analysis

**Purpose**: Quantify distribution shift causing overfitting gaps

**Steps**:
- Generate histograms/KS tests for top 20 features across train vs val
- Analyze positive-class feature distributions separately
- Produce temporal drift plots for critical indicators
- Publish findings in `reports/phase4/data_shift_analysis.md`

### Task 5: Label Quality Audit

**Purpose**: Confirm target definitions remain valid in recent regimes

**Steps**:
- Sample 100 trades per split; manually inspect price paths vs +5%/-2% criteria
- Validate absence of look-ahead bias (timestamp alignment, future leakage)
- Evaluate alternative profit targets (3.0%, 3.5%, 4.0%) on validation data
- Summarize outcomes and recommendations

### Task 6: Feature Importance Review

**Purpose**: Identify non-generalizing features and candidates for enhancement

**Steps**:
- Run SHAP/permutation analysis on each production checkpoint
- Compare top features across train vs validation data
- Flag features driving overfitting (high train impact, low val impact)
- Propose feature engineering V2 backlog items

## Evaluation & Deployment (Phase 4c: Production Readiness)

### Task 7: Test Set Evaluation ✅ (Completed 2025-10-04 18:08 UTC)

**Outcome**: `scripts/evaluate_hpo_models_on_test.py` executes full evaluation workflow against the enhanced Phase 3 test split (131,811 sequences, 23 features, 24-step windows).

- Checkpoints evaluated: `mlp_trial72`, `lstm_trial62`, `gru_trial93` from `models/hpo_derived`.
- Metrics recorded: accuracy, macro/positive precision/recall/F1, ROC-AUC, PR-AUC, log-loss, confusion matrix.
- Threshold sweep (0.05 increments + optimal selection) and PR curve snapshots exported per model.
- Artifacts stored under `reports/phase4/test_set_evaluation/` (latest `test_evaluation_20251004_180804.json`).
- Key results: MLP remains top candidate (F1⁺ 0.306 @0.50; optimal 0.316 @ threshold 0.55); recall drops ~0.13 vs validation but preserves ordering.

**Next linked tasks**: Feed metrics into Task 8 (threshold optimization) and Task 9 (backtesting campaign).

### Task 8: Threshold Optimization & Ensemble Strategy

**Purpose**: Maximize deployment performance while controlling risk

**Steps**:
- Sweep decision thresholds (0.10–0.90) for each model on validation
- Evaluate ensemble combinations (majority vote, weighted recall)
- Select primary + backup threshold configuration
- Document decision in `memory-bank/decisionLog.md`

**Status (2025-10-04)**: 🔄 In progress – re-scope required to incorporate full trading cost model, drawdown controls, and regime filters discovered during backtest post-mortem.

### Task 9: Backtesting Campaign

**Purpose**: Validate financial impact using historical data

**Steps**:
- Execute full backtesting with `scripts/run_baseline_training_campaign.py` adapted for checkpoints
- Include transaction costs, slippage, and risk metrics
- Summarize PnL, drawdown, trade distribution across symbols
- Prepare executive-ready report for stakeholders

**Status (2025-10-04)**: ✅ Completed via `scripts/backtest_hpo_production_models.py` (multi-model, ensemble, 143-symbol universe). Result: catastrophic losses (-88% to -93% total return) → triggered remediation plan.

## Documentation & Governance (Phase 4d)

### Task 10: Update Operational Playbooks

**Purpose**: Align runbooks with checkpoint-based deployment approach

**Steps**:
- Refresh `docs/baseline_training_campaign_guide.md` with checkpoint workflow
- Create `docs/production_model_playbook.md` covering deployment lifecycle
- Record rollback strategy based on MLflow artifact restoration

### Task 11: Action Item Tracking

**Purpose**: Ensure accountability for Phase 4 execution

**Steps**:
- Populate checklist in `memory-bank/phase3_hpo_results_analysis.md`
- Create project board entries (Kanban) per task
- Schedule weekly review cadence until Phase 4 complete

### Task 12: Stakeholder Communication

**Purpose**: Keep team informed of critical findings and next steps

**Steps**:
- Draft executive summary email referencing `phase3_hpo_results_analysis.md`
- Attach top-level metrics and decision highlights
- Present findings during weekly strategy sync
