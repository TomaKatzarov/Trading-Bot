"""Validate RL data readiness across historical symbol files."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd


TARGET_START = datetime(2023, 10, 2, tzinfo=timezone.utc)
TARGET_END = datetime(2025, 10, 1, tzinfo=timezone.utc)
START_TOLERANCE_HOURS = 12
END_TOLERANCE_HOURS = 12
DATA_BASE = Path("data/historical")
TIMEFRAME = "1Hour"

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

SENTIMENT_COLUMN = "sentiment_score_hourly_ffill"


@dataclass
class SymbolReport:
    symbol: str
    status: str
    data_start: str | None = None
    data_end: str | None = None
    rows: int | None = None
    missing_columns: List[str] | None = None
    null_columns: List[str] | None = None
    sentiment_out_of_bounds: bool | None = None
    notes: List[str] | None = None


def flatten_symbols(config_path: Path) -> List[str]:
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    symbols: Set[str] = set()

    def collect(values: Iterable):
        for item in values:
            if isinstance(item, list):
                collect(item)
            elif isinstance(item, dict):
                collect(item.values())
            else:
                symbols.add(str(item))

    collect(config.get("sectors", {}).values())
    collect(config.get("indices", {}).values())
    collect(config.get("etfs", {}).values())
    collect(config.get("crypto", {}).values())

    return sorted(symbols)


def validate_symbol(symbol: str) -> SymbolReport:
    report = SymbolReport(symbol=symbol, status="ok", notes=[])

    file_path = DATA_BASE / symbol / TIMEFRAME / "data.parquet"
    if not file_path.exists():
        report.status = "missing"
        report.notes = [f"File not found: {file_path.as_posix()}"]
        return report

    try:
        df = pd.read_parquet(file_path)
    except Exception as exc:  # pragma: no cover - IO errors
        report.status = "error"
        report.notes = [f"Error reading parquet: {exc}"]
        return report

    if "timestamp" not in df.columns:
        report.status = "error"
        report.notes = ["Missing timestamp column"]
        return report

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    data_start = df["timestamp"].min()
    data_end = df["timestamp"].max()

    report.data_start = data_start.isoformat()
    report.data_end = data_end.isoformat()
    report.rows = int(df.shape[0])

    date_notes: List[str] = []
    if data_start > TARGET_START:
        date_notes.append(
            f"Starts at {data_start.isoformat()} (target {TARGET_START.isoformat()})"
        )
    if data_end < TARGET_END:
        date_notes.append(
            f"Ends at {data_end.isoformat()} (target {TARGET_END.isoformat()})"
        )

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        report.status = "incomplete"
        report.missing_columns = missing_columns

    null_columns = [
        col for col in REQUIRED_COLUMNS if col in df.columns and df[col].tail(168).isnull().any()
    ]
    if null_columns:
        report.status = "incomplete"
        report.null_columns = null_columns

    sentiment_out_of_bounds = False
    if SENTIMENT_COLUMN in df.columns:
        sentiment_max = float(df[SENTIMENT_COLUMN].max())
        sentiment_min = float(df[SENTIMENT_COLUMN].min())
        if sentiment_min < 0.0 or sentiment_max > 1.0:
            sentiment_out_of_bounds = True
            report.status = "incomplete"
    else:
        sentiment_out_of_bounds = True
        missing_columns = report.missing_columns or []
        if SENTIMENT_COLUMN not in missing_columns:
            missing_columns.append(SENTIMENT_COLUMN)
        report.missing_columns = missing_columns
        report.status = "incomplete"

    report.sentiment_out_of_bounds = sentiment_out_of_bounds

    if date_notes:
        start_diff = (data_start - TARGET_START).total_seconds() / 3600
        end_diff = (TARGET_END - data_end).total_seconds() / 3600

        within_tolerance = (
            (start_diff <= START_TOLERANCE_HOURS) if start_diff > 0 else True
        ) and (
            (end_diff <= END_TOLERANCE_HOURS) if end_diff > 0 else True
        )

        if not within_tolerance:
            report.status = "incomplete"
            report.notes = (report.notes or []) + date_notes
        else:
            if report.notes is None:
                report.notes = []
            report.notes.extend(date_notes)
            report.notes.append(
                "Within tolerance despite minor start/end deviations"
            )

    return report


def main() -> None:
    config_path = Path("config/symbols.json")
    symbols = flatten_symbols(config_path)

    reports: List[SymbolReport] = []
    status_counter: defaultdict[str, int] = defaultdict(int)

    for symbol in symbols:
        report = validate_symbol(symbol)
        reports.append(report)
        status_counter[report.status] += 1

    total = len(symbols)
    valid = status_counter["ok"]
    coverage_pct = (valid / total) * 100 if total else 0.0

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_start": TARGET_START.isoformat(),
        "target_end": TARGET_END.isoformat(),
        "timeframe": TIMEFRAME,
        "total_symbols": total,
        "valid_symbols": valid,
        "coverage_pct": coverage_pct,
        "status_counts": status_counter,
        "symbols": [asdict(report) for report in reports],
    }

    output_path = Path("data/validation_report.json")
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 80)
    print("HISTORICAL DATA VALIDATION REPORT")
    print("=" * 80)
    print(f"Total symbols expected : {total}")
    print(f"Valid symbols          : {valid}")
    print(f"Coverage percentage    : {coverage_pct:.2f}%")
    print(f"Incomplete / missing   : {total - valid}")
    print("" )

    print("Top outstanding issues:")
    outstanding = [r for r in reports if r.status != "ok"][:10]
    if not outstanding:
        print("- None")
    else:
        for report in outstanding:
            note = ", ".join(report.notes or []) if report.notes else "No notes"
            print(f"- {report.symbol}: {report.status} ({note})")

    overall_pass = coverage_pct >= 95.0 and status_counter["incomplete"] == 0 and status_counter["missing"] == 0
    print("" )
    print("RESULT: " + ("PASS" if overall_pass else "FAIL"))


if __name__ == "__main__":
    main()