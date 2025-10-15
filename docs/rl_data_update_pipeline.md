# RL Data Update Pipeline Audit

_Last updated: 2025-10-14_

## 1. Pipeline Stage Map

| Stage | Script Entry Point | Primary Inputs | Outputs | Incremental Strategy |
| --- | --- | --- | --- | --- |
| Historical download | `DataUpdatePipeline._download_historical_data` → `HistoricalDataLoader.load_all_symbols` | `config/symbols.json`, Alpaca/yfinance APIs | `data/historical/{SYMBOL}/1Hour/data.parquet` (OHLCV + derived base features) | Appends from the last persisted timestamp by default; switches to full refresh only when `--force-refresh` or no prior parquet exists. |
| Indicator recompute | `DataUpdatePipeline._recompute_technical_indicators` → `TechnicalIndicatorCalculator.process_symbol` | Latest hourly parquet | Overwrites parquet in-place with 14 technical indicators + temporal features | Always recomputes on top of existing bars; idempotent because calculations derive from the stored series. |
| Sentiment attachment | `DataUpdatePipeline._attach_sentiment` → `SentimentAttacher.process_symbol` | Hourly parquet, `data/sentiment/{SYMBOL}/daily_sentiment.parquet` | Parquet enriched with `sentiment_score_hourly_ffill` | Forward-fills daily scores without truncating history; skips symbols lacking sentiment source data. |
| RL validation | `scripts/validate_rl_data_readiness.py` | Enriched parquet files | `data/validation_report.json` | Read-only pass that flags coverage gaps; no mutations. |
| Remediation | `scripts/remediate_rl_data_gaps.py` | Validation report, data/sentiment | Repairs/creates hourly parquet + sentiment + indicators | Downloads only symbols flagged missing/incomplete; respects `--dry-run` to avoid disk writes. |
| Sample verification | `scripts/verify_data_update_success.py` | Hourly parquet | Console summary for Oct-2025 coverage | Pure read; intended as daily smoke check. |
| Phase 3 manifest snapshot *(new)* | `DataUpdatePipeline._snapshot_phase3_manifest` | `data/phase3_splits/**` | Step metrics listing symbol directories and missing artifacts | Read-only; surfaces stale or partial symbol folders prior to model-training steps. |
| Combined training dataset | `scripts/generate_combined_training_data.py` | Enriched hourly parquet | `data/training_data_v2_final` (`*_X.npy`, metadata, scalers) | Regenerates full global dataset; lookback/prediction + label settings configurable via CLI. |
| Training data validation | `DataUpdatePipeline._validate_training_dataset` | Generated numpy artifacts + metadata | Step metrics + basic integrity checks | Guard rails on minimum sample count and feature count; optional positive-ratio bounds can be supplied via CLI when needed.

## 2. Incremental Data Handling

- `HistoricalDataLoader` inspects each symbol's `data.parquet` for the most recent timestamp and automatically switches to append-mode (start = last_ts + 1s) unless `--force-refresh` is supplied.
- Append range can be overridden with `--start-date/--end-date`. When provided, loader sets `append=True` ensuring only the chosen window is fetched.
- Indicator and sentiment passes are safe to rerun because they overwrite / forward-fill on top of the persisted parquet. They do not truncate existing bars.
- Remediator inherits the same append semantics but explicitly targets validation-failing symbols; neutral sentiment seeding preserves exploration inputs when upstream news data is missing.

## 3. Feature Coverage Confirmation

| Feature Group | Source | Guarantee |
| --- | --- | --- |
| OHLCV + derived base (`VWAP`, `Returns`, `HL_diff`, `OHLC_avg`) | `HistoricalDataLoader._process_data` and `_persist_symbol_dataset` (remediator fallback) | Always written before indicator stage; loader merges with on-disk parquet without deleting existing history. |
| 14 technical indicators | `TechnicalIndicatorCalculator.calculate_all_indicators` | Recomputed for every symbol during indicator step and during remediation; checks raise when required columns are absent. |
| Temporal features | `TechnicalIndicatorCalculator.calculate_day_of_week_features` and `prepare_phase3_data.enrich_features` | Ensures sine/cosine features persist even if upstream parquet lacked them. |
| Sentiment (`sentiment_score_hourly_ffill`) | `SentimentAttacher.forward_fill_sentiment` | Missing values forward/back-filled; remediator seeds neutral 0.5 curves when newsroom data is absent. |
| SL baseline probabilities (Phase 3 only) | `prepare_phase3_data._attach_sl_probs` | Uses global scaler + per-symbol StandardScaler to emit `sl_prob_{mlp,lstm,gru}` columns for RL curriculum. |

A quick spot check on `data/phase3_splits/SPY/train.parquet` confirmed 37 columns covering all engineered features plus SL probabilities; no label filtering is applied in the RL splits.

## 4. Validator & Remediator Flow

1. `validate_rl_data_readiness` enumerates all symbols from `config/symbols.json`, ensuring each parquet spans Oct-2023 → Oct-2025, contains the full feature suite, and that sentiment stays within \[0, 1]. Results land in `data/validation_report.json` with per-symbol statuses (`ok`, `incomplete`, `missing`).
2. The remediator consumes that report. When invoked without `--dry-run` it will:
   - Download missing symbols via Alpaca loader or fall back to yfinance aliases.
   - Rebuild the canonical hourly grid, recompute indicators, seed sentiment if required, and reattach hourly scores via `SentimentAttacher`.
   - Optionally rerun the validator (`--run-validation`) so the report reflects repairs.
3. In the pipeline, the remediation step now passes along metrics noting whether missing/incomplete cohorts existed and whether dry-run mode was used. If nothing required fixing, it proactively confirms coverage by scanning all symbols.

## 5. Phase 3 Split Management

- The Phase 3 splitter (`scripts/prepare_phase3_data.py`) now accepts `--symbols` so we can regenerate a single asset (e.g., `python scripts/prepare_phase3_data.py --symbols TSLA`) without touching the rest of the portfolio.
- `DataUpdatePipeline._snapshot_phase3_manifest` records the active symbol directories, `phase3_metadata.json` timestamp, and any missing artifacts (per-symbol `train/val/test.parquet`, `metadata.json`, `scaler.joblib`). This runs on every pipeline execution to surface stale folders before model training kicks off.
- Each split file preserves the full time series (no positive-label pruning) and carries SL baseline probabilities expected by SAC modules.

## 6. Positive-Label Monitoring

- Positive label ratios are now informational only; pass/fail gating is disabled unless you explicitly provide `--positive-ratio-min`/`--positive-ratio-max` to the pipeline.
- Current production metadata (`data/training_data_v2_final/metadata.json`) shows a 19.37% positive ratio and 2.8M total sequences, confirming the RL dataset keeps the full label distribution.

## 7. Verification Checklist (Dry-Run Friendly)

1. **Quick manifest & validation sweep**
   ```bash
   python scripts/run_full_data_update.py --skip-download --skip-indicators --skip-sentiment \
       --skip-training --skip-training-validation --remediation-dry-run
   ```
   - Confirms validator/remediator can execute without touching data.
   - Produces the Phase 3 manifest snapshot in the summary block.
2. **Append-only smoke test**
   ```bash
   python scripts/run_full_data_update.py --start-date 2025-09-01 --skip-training-validation
   ```
   - Appends ~6 weeks of data, rebuilds features, refreshes combined dataset.
3. **Phase 3 single-symbol refresh**
   ```bash
   python scripts/prepare_phase3_data.py --symbols TSLA --verbose
   ```
   - Regenerates TSLA splits only; inspect `data/phase3_splits/TSLA/metadata.json` afterwards.
4. **Manifest spot check**
   ```bash
   python - <<'PY'
      import json
      from pathlib import Path
      root = Path('data/phase3_splits')
      manifest = {child.name: sorted(p.name for p in child.glob('*.parquet'))
                         for child in root.iterdir() if child.is_dir()}
   print(json.dumps(manifest, indent=2))
   PY
   ```
   - Ensures expected symbol folders and parquet files are present.
5. **Positive-ratio sanity check**
   ```bash
   python - <<'PY'
      import json
      from pathlib import Path
   data = json.loads(Path('data/training_data_v2_final/metadata.json').read_text())
   print({k: data[k] for k in ('positive_ratio_train', 'positive_ratio_val', 'positive_ratio_test')})
   PY
   ```

## 8. Usage Quick Reference

- Full end-to-end refresh (default 2-year lookback, full remediation):
  ```bash
  python scripts/run_full_data_update.py
  ```
- Regenerate combined dataset only (after manual data tweaks):
  ```bash
  python scripts/run_full_data_update.py --skip-download --skip-indicators --skip-sentiment \
      --skip-validation --skip-remediation --skip-sample-verification
  ```
- Rebuild Phase 3 splits for the default 10-symbol basket:
  ```bash
  python scripts/prepare_phase3_data.py
  ```
- Rebuild Phase 3 splits for a subset:
  ```bash
  python scripts/prepare_phase3_data.py --symbols SPY QQQ
  ```

This document should serve as the central reference for maintaining the RL data pipeline and confirming that each guarantee remains intact across future updates.
