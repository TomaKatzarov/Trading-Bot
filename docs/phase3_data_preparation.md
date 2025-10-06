# Phase 3 Data Preparation Guide

## Overview

This guide summarizes the data preparation workflow that feeds Phase 3 reinforcement learning experiments. It covers dataset validation, chronological splitting, feature scaling, and supervised-learning (SL) baseline caching so symbol agents can train against clean, reproducible inputs.

## Sentiment Signal

The pipeline preserves the supervised-learning sentiment feature as `sentiment_score_hourly_ffill`. During preparation the column is forward/back filled to cover minor gaps, then standardized (fit on the training split only) alongside the technical indicators. All Phase 3 datasets and metadata explicitly include this column so agents can condition on news-driven sentiment in the same way the SL models do.

## 10-Symbol Portfolio

**Selected Symbols:**
- ETFs: SPY, QQQ
- Tech: AAPL, MSFT, NVDA, AMZN, META
- Other: TSLA, JPM, XOM

**Rationale:**
- Sector diversity across index, technology, automotive, finance, and energy exposures
- Mix of mega-cap and large-cap equities with consistent liquidity
- Broad volatility spectrum to stress-test agent behavior
- Each symbol validated for two years of trading-hour coverage and indicator completeness

## Data Splits

**Split Ratios:**
- Training: 70% (Oct 2023 → mid-Mar 2025)
- Validation: 15% (mid-Mar 2025 → early Jul 2025)
- Test: 15% (early Jul 2025 → Oct 2025)

Splits are generated per symbol via `scripts/prepare_phase3_data.py` using fixed timestamps to prevent leakage. Median training window length is 6,439 bars (post-lookback) and each split preserves temporal order.

**Critical:** Scalers are always fit on the training split only before being applied to validation/test partitions.

## Feature Scaling

**Scaled Features:** 23 technical indicators plus the normalized sentiment signal `sentiment_score_hourly_ffill`, matching the supervised-learning feature contract defined in `FEATURE_COLUMNS`.

**Method:** StandardScaler (z-score normalization).

**Location:** `data/phase3_splits/{symbol}/scaler.joblib`

Each symbol directory also includes `metadata.json` documenting row counts, split boundaries, and inferred trading hours.

## SL Baseline

**Cached for Comparison:**
- MLP Trial 72 predictions
- LSTM Trial 62 predictions
- GRU Trial 93 predictions
- Decision threshold: 0.80 (current best SL configuration)

**Artifacts:** `data/phase3_splits/{symbol}/sl_baseline_cache.npz`

Each cache bundles probability arrays and boolean signals per split so RL runs can benchmark against historical SL behavior without re-running inference.

## Usage

```python
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

symbol = "AAPL"
data_dir = Path("data/phase3_splits") / symbol

train = pd.read_parquet(data_dir / "train.parquet")
val = pd.read_parquet(data_dir / "val.parquet")
test = pd.read_parquet(data_dir / "test.parquet")

scaler = joblib.load(data_dir / "scaler.joblib")

sl_cache = np.load(data_dir / "sl_baseline_cache.npz")
mlp_test_signals = sl_cache["mlp_test_signals"]
```

## Validation Checks

Run the validation workflow at any time to confirm coverage:

```bash
python scripts/validate_phase3_symbols.py
```

Automated checks include:
- Coverage window: 2023-10-01 → 2025-10-31 (trading hours inferred per symbol)
- Row sufficiency: ≥ 6,500 trading-hour observations per symbol (actual average ≈ 7,854)
- Missing data: < 0.6% gap rate after skipping non-trading sessions
- Feature completeness: all 23 technical features + sentiment column present
- Maximum intraday gap: ≤ 3 hours for the trading sessions considered

## Output Artifacts

### Scripts
- `scripts/validate_phase3_symbols.py`
- `scripts/prepare_phase3_data.py`
- `scripts/cache_sl_baseline.py`

### Data Directory

```
phase3_splits/
├── phase3_metadata.json
├── SPY/
│   ├── train.parquet
│   ├── val.parquet
│   ├── test.parquet
│   ├── scaler.joblib
│   ├── metadata.json
│   └── sl_baseline_cache.npz
├── QQQ/
│   └── … (same layout)
└── … eight more symbol folders
```

### Reports
- `analysis/reports/phase3_symbol_validation.csv`
- `analysis/reports/phase3_sl_cache_summary.json`

### Documentation
- Updated `memory-bank/RL_IMPLEMENTATION_PLAN.md`
- This guide (`docs/phase3_data_preparation.md`)

## Next Steps

With Task 3.1 complete, the data pipeline is ready for Phase 3 training runs. Proceed to Task 3.2 to wire PPO training infrastructure, configure MLflow/TensorBoard logging, and schedule the first 10-agent training sweep.

For reproducibility, regenerate artifacts by following the testing instructions in the Task 3.1 success criteria: activate `trading_rl_env`, run the three scripts in order, and verify the directory structure before launching RL experiments.
