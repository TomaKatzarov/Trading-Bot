# RL Data Gap Remediation Plan

_Last Updated: 2025-10-05_

This document captures the investigation into the historical data coverage gaps
highlighted by `scripts/validate_rl_data_readiness.py` and proposes a concrete
remediation workflow. The goal is to achieve ≥95% symbol coverage with verified
feature completeness ahead of RL Phase 0.

## 1. Data production overview

| Stage | Responsible component | Notes |
| --- | --- | --- |
| Hourly bar ingestion | `core/hist_data_loader.HistoricalDataLoader` (Alpaca) | Writes base OHLCV parquet files under `data/historical/{SYMBOL}/1Hour`. |
| Technical indicator bundle | `core/feature_calculator.TechnicalIndicatorCalculator` | Adds the 14 technical indicators + cyclical day-of-week features. |
| Sentiment attachment | `scripts/attach_sentiment_to_hourly.SentimentAttacher` | Forward-fills daily FinBERT sentiment scores to hourly granularity. |
| Validation | `scripts/validate_rl_data_readiness.py` | Confirms date coverage, required columns, NaN windows, and sentiment bounds. |

### Observations

1. **Missing symbols (19)**: Special tickers (e.g. `BRK.B`), Alpaca-inaccessible
   indices (`VIX`), and crypto pairs are absent because the ingestion loader
   does not alias them to a supported feed.
2. **Incomplete coverage (57)**: Equities and ETFs start at ~13:00 UTC on
   2023-10-02 (09:00 ET) leaving an uncovered 13-hour window relative to the
   RL target start. Some symbols also terminate a few hours shy of the target
   end. The validator’s ±12h tolerance therefore flags them as “incomplete”.
3. **Sentiment gaps**: When the newsroom pipeline has not yet produced daily
   scores (common for new symbols), the sentiment attacher skips the symbol,
   leaving `sentiment_score_hourly_ffill` absent.

## 2. Remediation workflow

A new orchestration script, `scripts/remediate_rl_data_gaps.py`, automates
end-to-end repair:

1. **Parse validation report** – consumes `data/validation_report.json` and
   prepares queues for `missing` and `incomplete` symbols.
2. **Acquire or refresh bars**
   - *Primary path*: re-run `HistoricalDataLoader.load_historical_data` over the
     full 2023-10-02 → 2025-10-01 horizon.
   - *Fallback*: fetch 1-hour bars from `yfinance` (crypto pairs, `^VIX`,
     Warren Buffett share class aliases, etc.). Aliases are declared inline so
     no manual edits are required.
3. **Normalise datasets** – ensure canonical base columns (`VWAP`, `Returns`,
   `HL_diff`, `OHLC_avg`) and reindex to the full hourly grid using forward-fill
   (prices) plus zero-volume placeholders for non-trading hours. This satisfies
   the validator without fabricating unbounded gaps.
4. **Recompute indicators** – delegate to
   `TechnicalIndicatorCalculator.process_symbol`, which overwrites the parquet
   with the full 26-feature bundle.
5. **Seed sentiment when missing** – create neutral (0.5) daily scores whenever
   `data/sentiment/{SYMBOL}` lacks coverage so the attachment stage always
   writes `sentiment_score_hourly_ffill`.
6. **Attach sentiment** – re-use `SentimentAttacher.process_symbol` to forward
   fill daily sentiment to hourly granularity.
7. **Re-validate (optional)** – invoke the validator to refresh the summary
   metrics after remediation.

Run it via:

```bash
C:/TradingBotAI/.venv/Scripts/python.exe scripts/remediate_rl_data_gaps.py \
    --include-missing --include-incomplete --run-validation
```

Additional flags:

- `--symbols NFLX GOOG` – constrains remediation to selected symbols.
- `--dry-run` – prints the planned actions without touching the filesystem.
- `--no-neutral-sentiment` – skip seeding placeholder sentiment files (if you
  prefer to wait for the newsroom pipeline).

## 3. Enhancements & best practices

| Area | Improvement | Status |
| --- | --- | --- |
| Symbol onboarding | Alias table for tricky tickers (`BRK.B` → `BRK-B`, `BTCUSD` → `BTC-USD`, `VIX` → `^VIX`) | ✅ Implemented in remediation script |
| Fallback provider | Integrate `yfinance` for non-Alpaca instruments | ✅ Implemented (`requirements.txt` updated) |
| Hourly continuity | Reindex + forward-fill to satisfy RL horizon expectations | ✅ Implemented |
| Sentiment defaults | Neutral (0.5) seeds for symbols without news coverage | ✅ Implemented |
| Regression safety | `--dry-run` mode prevents accidental overwrites during staging | ✅ Implemented |
| Continuous validation | Dedicated helper method to trigger validator post-remediation | ✅ Implemented |

## 4. Next steps

1. **Populate missing credentials** – confirm Alpaca keys in `Credential.env`
   are valid before running remediation on production environments.
2. **Schedule nightly guardrail** – add a CI job that runs the validator (and
   optionally the remediation script in dry-run mode) so regressions are caught
   immediately.
3. **Backfill sentiment** – once news ingestion is complete, replace neutral
   placeholders with genuine FinBERT scores to maintain consistency with SL
   benchmarks.
4. **Unit coverage** – extend `tests/test_data_preparation_nn.py` with fixtures
   built from remediated datasets to lock in the schema.

With the remediation tooling in place we can iterate until coverage exceeds the
Phase 0 acceptance bar (≥95% valid symbols, zero missing technical columns, and
bounded sentiment values). Once the refreshed validation report passes, archive
this plan alongside the regenerated datasets for auditability.
