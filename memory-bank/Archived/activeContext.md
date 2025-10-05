# Active Context

This file tracks the project's current status, including recent changes, current goals, and open questions.

The detailed implementation plan is in `progress.md`.

## Current Focus (October 4, 2025)

- **Phase 4 / Post-Backtest Remediation Window**
  - Tasks 7–9 complete: test evaluation + full-universe backtesting executed with HPO checkpoints and ensembles.
  - **Immediate priority:** Diagnose catastrophic backtesting losses (88–93% drawdowns) and design mitigation layers before any deployment approval.
  - Reference artifacts: `backtesting/results/full_campaign/` (20251004_*) and `reports/phase4/test_set_evaluation/test_evaluation_20251004_180804.json`.
- **Operational Requirements:**
  - Quantify train/val/test distribution drift vs backtest regime to isolate signal decay causes.
  - Re-run threshold/position-sizing sweeps under realistic cost assumptions.
  - Restore MLflow availability (previous `mlflow server` attempt failed with exit code 1) for logging remediation experiments.

## Fresh Updates

- **2025-10-04 21:30 UTC** – Full Backtesting Campaign Completed (Losses)
  - Executed `scripts/backtest_hpo_production_models.py` across MLP/LSTM/GRU + weighted ensembles over 2023-10-02 → 2025-10-01.
  - Outcome: -88% to -93% total return per strategy, Sharpe -0.02 to -0.04, >91% max drawdowns, profit factor ≈0.70.
  - Recommendation: Treat GRU Trial 93 as provisional focus for remediation, pause deployment, initiate regime/threshold diagnostics.
- **2025-10-04 18:08 UTC** – Test Set Evaluation Script Delivered
  - Added `scripts/evaluate_hpo_models_on_test.py` (batch inference, full metric suite, threshold sweep, PR curve sampling, JSON reporting).
  - Evaluated checkpoints: `mlp_trial72`, `lstm_trial62`, `gru_trial93` from `models/hpo_derived`.
  - Outputs include console summary + JSON artifacts; results confirm validation ordering and quantify recall drop.
- **2025-10-04 18:10 UTC** – Memory Bank refreshed with evaluation outcomes (see `progress.md`, `phase3_hpo_results_analysis.md`, `decisionLog.md`).

## Near-Term Action Items

1. **Backtest Post-Mortem & Data Drift Analysis**
  - Compare feature/label distributions across train/val/test vs 2023-2025 backtest horizon.
  - Quantify market regime shifts and label quality degradation.
  - Produce diagnostic report (`reports/phase4/data_shift_analysis.md`).
2. **Strategy Remediation Experiments**
  - Re-run threshold/position-sizing sweeps with transaction costs and signal decay filters.
  - Prototype risk overlays (volatility gating, drawdown caps) using GRU Trial 93 as control.
  - Log outcomes in `decisionLog.md` before proposing deployment go/no-go.
3. **MLflow Availability**
  - Previous `mlflow server` start failed (Exit Code 1). Investigate logs, restart before logging remediation experiments.

## Reference Systems & Tooling

- **Advanced Experiment Management** (`core/experiment_management/`)
  - ConfigurationManager, EnhancedMLflowLogger, ExperimentOrganizer, ExperimentReporter all in place.
  - MLflow tracking captures architecture, data context, environment metadata.
  - Usage guide: `docs/experiment_management.md`.
- **HPO Framework** (`training/run_hpo.py`, `training/run_hpo_quick_start.py`)
  - Optuna integration ready for follow-up studies if distribution fixes are needed.
- **Baseline Training Campaign Infrastructure** (completed 2025-09-30)
  - `scripts/run_baseline_training_campaign.py` orchestrates full baseline runs with standardized configs.
  - Documentation: `docs/baseline_training_campaign_guide.md`.

*(Earlier context archived for brevity.)*
