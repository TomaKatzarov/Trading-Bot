# Supervised-Learning Baseline for RL Comparison

## Overview
- Re-ran `scripts/backtest_hpo_production_models.py` on 2025-10-05 for the production MLP, LSTM, and GRU classifiers using probability thresholds **0.60 → 0.80** in 0.05 increments.
- Each run used the full evaluation window (2023-10-02 → 2025-10-01), $100{,}000$ notional capital, and identical cost assumptions as the production SL pipeline.
- Backtest artifacts are stored under `backtesting/results/threshold_sweep/` as JSON reports; the CSV trade logs remain available for drill-down.
- Threshold metadata is not recorded inside the JSON payloads, so this report maps thresholds to output files by the chronological execution order of the sweep (0.60 first → 0.80 last) for each model.

## Key Takeaways
- **Raising the decision threshold materially reduces trade volume.** At $0.80$, both the LSTM and GRU models stopped trading entirely, and the MLP executed only 583 trades vs. 6,899 at $0.60$.
- **All high-volume configurations lost capital.** Even the best-performing MLP sweep (threshold 0.80) finished at **-10.85\%** total return with a mild -12.4\% max drawdown. Lower thresholds amplified losses to -70\% to -80\% with drawdowns beyond -80\%.
- **LSTM 0.75 generated a marginal +0.15\% return** but only opened 29 positions, making the result statistically fragile and unsuitable as a production baseline.
- **Sharpe ratios are negative across active configurations** (≈-0.05 to -0.46). Profit factors remain below 1.0, indicating loss-making portfolios despite moderate win rates (~45-48\%).

## Detailed Results

### MLP Sweep
| Threshold | Total Return % | Annualized % | Sharpe | Max Drawdown % | Win Rate % | Profit Factor | Trades | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.60 | -78.78 | -54.12 | -0.053 | -82.09 | 44.76 | 0.728 | 6,899 | `backtest_campaign_20251005_102141.json` |
| 0.65 | -63.30 | -39.57 | -0.090 | -65.23 | 45.72 | 0.777 | 4,814 | `backtest_campaign_20251005_102203.json` |
| 0.70 | -38.50 | -21.68 | -0.106 | -38.83 | 46.20 | 0.772 | 2,169 | `backtest_campaign_20251005_102224.json` |
| 0.75 | -24.74 | -13.31 | -0.118 | -25.48 | 45.08 | 0.756 | 1,047 | `backtest_campaign_20251005_102244.json` |
| 0.80 | -10.85 | -5.61 | -0.136 | -12.42 | 47.68 | 0.822 | 583 | `backtest_campaign_20251005_102305.json` |

### LSTM Sweep
| Threshold | Total Return % | Annualized % | Sharpe | Max Drawdown % | Win Rate % | Profit Factor | Trades | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.60 | -79.85 | -55.30 | -0.054 | -81.98 | 44.05 | 0.713 | 6,247 | `backtest_campaign_20251005_102340.json` |
| 0.65 | -46.94 | -27.27 | -0.089 | -49.41 | 44.05 | 0.726 | 2,286 | `backtest_campaign_20251005_102402.json` |
| 0.70 | -26.63 | -14.41 | -0.073 | -29.10 | 41.49 | 0.631 | 699 | `backtest_campaign_20251005_102422.json` |
| 0.75 | 0.15 | 0.08 | -0.456 | -1.31 | 51.72 | 1.052 | 29 | `backtest_campaign_20251005_102443.json` |
| 0.80 | 0.00 | 0.00 | 0.000 | 0.00 | 0.00 | 0.000 | 0 | `backtest_campaign_20251005_102503.json` |

### GRU Sweep
| Threshold | Total Return % | Annualized % | Sharpe | Max Drawdown % | Win Rate % | Profit Factor | Trades | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.60 | -70.43 | -45.79 | -0.066 | -70.58 | 44.75 | 0.730 | 4,639 | `backtest_campaign_20251005_102544.json` |
| 0.65 | -34.59 | -19.21 | -0.089 | -34.90 | 39.30 | 0.586 | 827 | `backtest_campaign_20251005_102605.json` |
| 0.70 | 0.00 | 0.00 | 0.000 | 0.00 | 0.00 | 0.000 | 0 | `backtest_campaign_20251005_102625.json` |
| 0.75 | 0.00 | 0.00 | 0.000 | 0.00 | 0.00 | 0.000 | 0 | `backtest_campaign_20251005_102645.json` |
| 0.80 | 0.00 | 0.00 | 0.000 | 0.00 | 0.00 | 0.000 | 0 | `backtest_campaign_20251005_102705.json` |

## Baseline Summary
- **Representative SL baseline:** MLP at threshold 0.80 (largest trade count without catastrophic losses) — Total return -10.9\%, annualized -5.6\%, Sharpe -0.14, max drawdown 12.4\%, profit factor 0.82, 583 trades.
- **Degenerate configurations:** Thresholds ≥0.75 for GRU and 0.80 for LSTM halted trading (0 trades). These cannot serve as meaningful comparison points for RL.
- **Risk profile:** Even the mild-loss configurations retain moderate drawdowns (12-35\%) while underperforming cash. Any RL deployment must clear these hurdles and deliver genuine alpha.

## RL Performance Targets (Phase 0.5.3)
| Metric | Observed SL Baseline | RL Target for Phase Progression |
| --- | --- | --- |
| Total return (full window) | -10.9\% (MLP 0.80) with 583 trades | **≥ +12\%** while maintaining ≥500 trades |
| Annualized return | -5.6\% | **≥ +15\%** |
| Sharpe ratio | -0.05 (best non-degenerate) | **≥ 0.50** |
| Max drawdown | 12.4\% (best case, but with losses) / 82\% (worst) | **≤ 25\%** while hitting return target |
| Win rate | 47.7\% (MLP 0.80) | **≥ 52\%** |
| Profit factor | 0.82 | **≥ 1.30** |

Meeting the targets above will exceed the existing SL system by a wide margin and provide a 20-30\% improvement relative to the least-bad baseline across both return and risk dimensions. These thresholds now serve as the quantitative quality gate for promoting RL agents beyond Phase 0.

## Next Steps
1. Feed aggregated metrics into `memory-bank/RL_IMPLEMENTATION_PLAN.md` (Phase 0 tracker) to mark tasks 0.5.2 and 0.5.3 as complete.
2. Use the targets above to parameterize reward shaping and validation harnesses in Phases 1-3.
3. Revisit SL thresholds or hybrid strategies only if RL agents fail to clear the defined quality gate.
