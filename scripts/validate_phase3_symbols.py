#!/usr/bin/env python3
"""Phase 3 symbol portfolio validation utility.

This script verifies historical data readiness for the 10-symbol Phase 3
prototype portfolio. It checks date coverage, required feature availability,
missing data ratios, and gap statistics while accounting for the observed
trading hours per symbol.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


PHASE3_SYMBOLS: Sequence[str] = (
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "TSLA",
    "JPM",
    "XOM",
)

TARGET_START = pd.Timestamp("2023-10-01 00:00:00", tz="UTC")
TARGET_END = pd.Timestamp("2025-10-31 23:00:00", tz="UTC")
TIMEFRAME = "1Hour"

# Expected data quality thresholds (aligned with extended-hours U.S. equities data)
MAX_MISSING_PCT = 10.0  # percent of missing trading hours or NaNs tolerated
MAX_GAP_HOURS = 16  # maximum tolerated missing trading hours within a single day
MAX_MISSING_BUSINESS_DAYS = 50
START_TOLERANCE_HOURS = 168
END_TOLERANCE_HOURS = 800
MIN_EXPECTED_ROWS = 6500

REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "VWAP",
    "Returns",
    "HL_diff",
    "OHLC_avg",
    "SMA_10",
    "SMA_20",
    "MACD_line",
    "MACD_signal",
    "MACD_hist",
    "RSI_14",
    "Stoch_K",
    "Stoch_D",
    "ADX_14",
    "ATR_14",
    "BB_bandwidth",
    "OBV",
    "Volume_SMA_20",
    "Return_1h",
    "DayOfWeek_sin",
    "DayOfWeek_cos",
    "sentiment_score_hourly_ffill",
]


@dataclass
class SymbolValidation:
    symbol: str
    passed: bool
    rows: int
    expected_rows: int
    missing_pct: float
    max_gap_hours: int
    trading_hours: List[int]
    start_date: str | None
    end_date: str | None
    issues: List[str]
    missing_columns: List[str]
    null_columns: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/historical"),
        help="Root directory containing symbol/hourly parquet files.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="*",
        default=list(PHASE3_SYMBOLS),
        help="Optional subset of symbols to validate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/reports/phase3_symbol_validation.csv"),
        help="Destination CSV report path.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("analysis/reports/phase3_symbol_validation.json"),
        help="Optional JSON summary output path.",
    )
    return parser.parse_args()


def load_symbol_frame(symbol: str, data_root: Path) -> pd.DataFrame:
    file_path = data_root / symbol / TIMEFRAME / "data.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found for {symbol}: {file_path}")

    df = pd.read_parquet(file_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing 'timestamp' column in {file_path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


def determine_trading_hours(index: pd.DatetimeIndex, presence_threshold: float = 0.95) -> List[int]:
    """Infer the dominant trading hours (UTC) for a symbol."""

    weekday_index = index[index.weekday < 5]
    if weekday_index.empty:
        return []

    unique_days = pd.DatetimeIndex(weekday_index.normalize().unique())
    total_days = len(unique_days) or 1

    hour_day_counts: Dict[int, int] = {}
    hour_seen: Dict[int, set] = {}
    for ts in weekday_index:
        hour = int(ts.hour)
        day = ts.normalize()
        day_set = hour_seen.setdefault(hour, set())
        if day not in day_set:
            day_set.add(day)
            hour_day_counts[hour] = hour_day_counts.get(hour, 0) + 1

    core_hours = [
        hour
        for hour, count in hour_day_counts.items()
        if (count / total_days) >= presence_threshold
    ]
    if not core_hours:
        core_hours = list(hour_day_counts.keys())
    return sorted(core_hours)


def compute_missing_trading_hours(
    df: pd.DataFrame,
    trading_hours: List[int],
    min_presence_ratio: float = 0.5,
) -> Tuple[int, int, int, int]:
    """Calculate missing trading hours while skipping weekends and partial days.

    Returns total missing hour occurrences, maximum consecutive missing hours
    within a day, number of trading days considered, and the expected row count
    implied by the inferred trading hours.
    """

    if not trading_hours or df.empty:
        return 0, 0, 0, 0

    expected_hours = set(trading_hours)
    threshold = max(int(len(expected_hours) * min_presence_ratio), 1)
    total_missing = 0
    max_missing = 0
    trading_days_considered = 0

    for day, group in df.groupby(df.index.normalize()):
        if day.weekday() >= 5:  # skip weekends
            continue
        if len(group) < threshold:
            # treat heavily truncated sessions (holidays, partial data) as
            # non-trading days so they do not skew missing ratios
            continue

        trading_days_considered += 1
        hours_present = set(int(ts.hour) for ts in group.index)
        missing = expected_hours.difference(hours_present)
        count = len(missing)
        total_missing += count
        if count > max_missing:
            max_missing = count

    expected_rows = trading_days_considered * len(expected_hours)
    return total_missing, max_missing, trading_days_considered, expected_rows


def validate_symbol(symbol: str, data_root: Path) -> SymbolValidation:
    issues: List[str] = []
    missing_columns: List[str] = []
    null_columns: List[str] = []

    try:
        df = load_symbol_frame(symbol, data_root)
    except Exception as exc:
        return SymbolValidation(
            symbol=symbol,
            passed=False,
            rows=0,
            expected_rows=0,
            missing_pct=100.0,
            max_gap_hours=999,
            trading_hours=[],
            start_date=None,
            end_date=None,
            issues=[str(exc)],
            missing_columns=[],
            null_columns=[],
        )

    df = df[(df.index >= TARGET_START) & (df.index <= TARGET_END)]
    rows = int(df.shape[0])
    start_date = df.index.min().isoformat() if not df.empty else None
    end_date = df.index.max().isoformat() if not df.empty else None

    trading_hours = determine_trading_hours(df.index)
    if not trading_hours:
        issues.append("Unable to infer trading hours (no weekday data)")

    # Column presence & null checks
    available_columns: List[str] = []
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            missing_columns.append(column)
        else:
            available_columns.append(column)
            if df[column].tail(168).isna().any():
                null_columns.append(column)

    if missing_columns:
        issues.append(f"Missing columns: {missing_columns}")
    if null_columns:
        issues.append(f"Recent NaNs detected in columns: {null_columns}")

    if available_columns:
        missing_pct_nan = float(
            df[available_columns]
            .isna()
            .any(axis=1)
            .mean()
            * 100.0
        )
    else:
        missing_pct_nan = 100.0

    (
        missing_hours_total,
        max_missing_hours,
        trading_days_considered,
        expected_rows,
    ) = compute_missing_trading_hours(df, trading_hours)
    observed_days = pd.DatetimeIndex(df.index.normalize().unique())
    expected_days = pd.bdate_range(TARGET_START.normalize(), TARGET_END.normalize(), tz="UTC")
    total_expected_hours = max(expected_rows, 1)
    missing_pct = (missing_hours_total / total_expected_hours) * 100.0 if total_expected_hours else 100.0

    missing_days = expected_days.difference(observed_days)
    if len(missing_days) > MAX_MISSING_BUSINESS_DAYS:
        issues.append(
            f"Missing {len(missing_days)} business days exceeds threshold {MAX_MISSING_BUSINESS_DAYS}"
        )

    if rows < MIN_EXPECTED_ROWS:
        issues.append(f"Only {rows} rows (< {MIN_EXPECTED_ROWS} minimum)")
    if missing_pct_nan > MAX_MISSING_PCT:
        issues.append(f"Rows with NaNs {missing_pct_nan:.2f}% exceeds {MAX_MISSING_PCT}% threshold")
    if missing_pct > MAX_MISSING_PCT:
        issues.append(f"Missing trading hours {missing_pct:.2f}% exceeds {MAX_MISSING_PCT}% threshold")
    if max_missing_hours > MAX_GAP_HOURS:
        issues.append(f"Max intraday gap {max_missing_hours}h exceeds {MAX_GAP_HOURS}h threshold")

    start_delta = (df.index.min() - TARGET_START).total_seconds() / 3600 if start_date else float("inf")
    end_delta = (TARGET_END - df.index.max()).total_seconds() / 3600 if end_date else float("inf")
    if start_delta > START_TOLERANCE_HOURS:
        issues.append(
            f"Start {start_date} lags target by {start_delta:.1f}h (>{START_TOLERANCE_HOURS}h tolerance)"
        )
    if end_delta > END_TOLERANCE_HOURS:
        issues.append(
            f"End {end_date} leads target by {end_delta:.1f}h (>{END_TOLERANCE_HOURS}h tolerance)"
        )

    passed = len(issues) == 0

    return SymbolValidation(
        symbol=symbol,
        passed=passed,
        rows=rows,
        expected_rows=expected_rows or rows,
        missing_pct=float(max(missing_pct, missing_pct_nan)),
    max_gap_hours=int(max_missing_hours),
        trading_hours=trading_hours,
        start_date=start_date,
        end_date=end_date,
        issues=issues,
        missing_columns=missing_columns,
        null_columns=null_columns,
    )


def render_summary(results: List[SymbolValidation]) -> None:
    print("=" * 70)
    print("PHASE 3 SYMBOL PORTFOLIO VALIDATION")
    print("=" * 70)
    print(f"\nValidating {len(results)} symbols...")
    print(f"Target period: {TARGET_START.date()} to {TARGET_END.date()}")
    print(f"Trading-day hours evaluated: weekday business hours inferred per symbol")

    passes = 0
    for result in results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        date_span = (
            f"{result.start_date[:10]} to {result.end_date[:10]}"
            if result.start_date and result.end_date
            else "n/a"
        )
        print(
            f"\nValidating {result.symbol}... {status}"
            f" ({result.rows:,} rows, expected ≈ {result.expected_rows:,}, period {date_span})"
        )
        if result.trading_hours:
            print(f"  Trading hours (UTC): {result.trading_hours}")
        if result.issues:
            for issue in result.issues:
                print(f"  - {issue}")
        else:
            passes += 1

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    print(f"\nPassed: {passed}/{total} symbols")
    print(f"Failed: {failed}/{total} symbols")
    if failed == 0:
        print("\n✅ ALL SYMBOLS VALIDATED - Ready for Phase 3 training!")
    else:
        failed_symbols = ", ".join(r.symbol for r in results if not r.passed)
        print("\n❌ VALIDATION FAILED - Review issues above")
        print(f"Failed symbols: {failed_symbols}")


def save_reports(results: List[SymbolValidation], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([asdict(result) for result in results])
    df.to_csv(csv_path, index=False)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_start": TARGET_START.isoformat(),
        "target_end": TARGET_END.isoformat(),
        "timeframe": TIMEFRAME,
        "thresholds": {
            "max_missing_pct": MAX_MISSING_PCT,
            "max_gap_hours": MAX_GAP_HOURS,
            "min_expected_rows": MIN_EXPECTED_ROWS,
        },
        "symbols": [asdict(result) for result in results],
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nValidation report saved: {csv_path}")
    print(f"JSON summary saved: {json_path}")


def main() -> None:
    args = parse_args()
    symbols = list(dict.fromkeys(args.symbols))  # preserve order & remove dupes

    results = [validate_symbol(symbol, args.data_root) for symbol in symbols]
    render_summary(results)
    save_reports(results, args.output, args.json)


if __name__ == "__main__":  # pragma: no cover
    main()
