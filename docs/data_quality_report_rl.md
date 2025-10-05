# RL Data Quality Report

Generated: 2025-10-05

## Executive Summary

| Metric | Value |
| --- | --- |
| Symbols expected (config/asset_id_mapping.json) | 162 |
| Symbols present in `data/historical/` | 154 |
| Symbols passing all checks | 86 (53.1%) |
| Symbols with coverage gaps | 57 |
| Symbols missing entirely or lacking 1Hour parquet | 19 |
| Technical indicator columns missing | 0 |
| Sentiment out-of-range issues | 0 |

Outcome: **FAIL** (coverage <95%). Remediation required before RL training can proceed.

## Validation Procedure

Validation executed with:

```bash
C:/TradingBotAI/.venv/Scripts/python.exe scripts/validate_rl_data_readiness.py > docs/rl_data_validation.log
```

Detailed JSON report written to `data/validation_report.json`.

## Missing Symbols (No Directory or No Parquet)

19 symbols lack a usable `1Hour/data.parquet` file:

```
AAVEUSD, ADAUSD, ADP, AVGO, BRK.B, BTCUSD, CMCSA, COMPUSD, DIS,
DOTUSD, ETHUSD, GOOG, MDLZ, MKRUSD, NFLX, SOLUSD, UNIUSD, VIX, YFIUSD
```

- **Directory missing entirely:** ADP, AVGO, BRK.B, CMCSA, DIS, GOOG, MDLZ, NFLX
- **Directory exists but file missing/empty:** Remaining symbols above (primarily crypto + VIX)

## Coverage Gaps

57 symbols have incomplete temporal coverage despite existing files.

- **Start gap only (26 symbols):** e.g., ABBV, ABT, AMGN, CCI, COP (data begins ~13:00 UTC on 2023-10-02)
- **Start + end gap (31 symbols):** e.g., AEP, AMT, APD, AVB, BKNG (data stops several hours before 2025-10-01 00:00 UTC)

These gaps fall within ±4 hours of market open/close for many equities but exceed the ±12 hour tolerance for a clean pass. Recommend extending source extraction to include full session coverage per day.

## Feature Completeness

- All required columns (26 features including OHLCV, technical indicators, day-of-week encodings, sentiment) exist for every loaded dataset.
- No NaN values detected within the most recent 168 observations for any required column.

## Sentiment Attachment

- `sentiment_score_hourly_ffill` present and constrained to [0, 1] for all datasets read successfully.
- No forward-fill gaps detected in the sampled windows.

## Recommended Actions

1. **Acquire missing datasets** for the 19 symbols listed above. Prioritize core equities/indices (e.g., NFLX, GOOG, BRK.B) before cryptos.
2. **Extend market-hour coverage** to capture full trading day for incomplete symbols—ensure extraction starts by 2023-10-02 00:00 UTC and ends at/after 2025-10-01 00:00 UTC.
3. **Re-run validation script** after remediation and target ≥95% pass rate.
4. **Version control the cleaned dataset** so RL experiments can rely on stable artifacts.

## Artifacts

- `docs/rl_data_validation.log` – console summary
- `data/validation_report.json` – structured detail per symbol
- `scripts/validate_rl_data_readiness.py` – validation utility

Once remediation is complete, update this report with the new statistics and archive dated copies for traceability.
