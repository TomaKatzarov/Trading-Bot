#!/usr/bin/env python3
"""Remediation utilities for closing RL data gaps.

This script orchestrates historical price ingestion, feature regeneration, and
sentiment attachment for symbols flagged by ``scripts/validate_rl_data_readiness.py``.
It is deliberately modular so it can be invoked from CI, notebooks, or the CLI.

Key capabilities
================

1. Read the latest validation report (``data/validation_report.json``) and
   categorise symbols as *missing* or *incomplete*.
2. Attempt automated remediation for each symbol category:
   • Missing symbols → fetch hourly bars via Alpaca (if credentials exist) or a
     fallback provider (``yfinance``). The fetched dataset is normalised to the
     canonical schema and aligned to the RL date range.
   • Incomplete symbols → force-refresh bars via Alpaca, align to the canonical
     clock, and regenerate indicators.
3. Recompute the technical indicator bundle using
   ``core.feature_calculator.TechnicalIndicatorCalculator``.
4. Ensure a sentiment series exists (seeded with neutral 0.5 scores when the
   newsroom pipeline has not produced coverage yet) and attach it to the hourly
   parquet through ``scripts.attach_sentiment_to_hourly.SentimentAttacher``.
5. Optionally re-run the validator so the updated coverage statistics are
   captured in ``data/validation_report.json``.

The script does **not** make network calls when invoked with ``--dry-run`` and
will simply print the actions it *would* perform.

Example usage
-------------

.. code-block:: bash

   C:/TradingBotAI/.venv/Scripts/python.exe scripts/remediate_rl_data_gaps.py \
       --include-missing --include-incomplete --run-validation

   # Remediate only a specific subset without touching the filesystem
   C:/TradingBotAI/.venv/Scripts/python.exe scripts/remediate_rl_data_gaps.py \
       --symbols NFLX GOOG --dry-run

Dependencies
============

• ``yfinance`` (optional) – used as a fallback data source for symbols that the
  Alpaca API does not cover (crypto pairs, synthetic indices, special tickers).
  The dependency is declared in ``requirements.txt``.
• Alpaca credentials – required if you would like the remediation utility to
  use ``HistoricalDataLoader`` for re-downloading incomplete datasets.

"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import sys

try:  # pragma: no cover - optional dependency
    import yfinance as yf  # type: ignore

    YFINANCE_AVAILABLE = True
except Exception:  # pragma: no cover - yfinance is optional
    YFINANCE_AVAILABLE = False

# Local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.feature_calculator import TechnicalIndicatorCalculator
from core.hist_data_loader import HistoricalDataLoader
from scripts.attach_sentiment_to_hourly import SentimentAttacher
import scripts.validate_rl_data_readiness as data_validator

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
HISTORICAL_ROOT = DATA_ROOT / "historical"
SENTIMENT_ROOT = DATA_ROOT / "sentiment"
SYMBOL_CONFIG_PATH = PROJECT_ROOT / "config" / "symbols.json"
VALIDATION_REPORT_PATH = DATA_ROOT / "validation_report.json"

TARGET_START = datetime(2023, 10, 2, tzinfo=timezone.utc)
TARGET_END = datetime(2025, 10, 1, tzinfo=timezone.utc)
TIMEFRAME = "1Hour"

BASE_COLUMNS = [
    "timestamp",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "VWAP",
    "Returns",
    "HL_diff",
    "OHLC_avg",
]

YFINANCE_ALIAS: Dict[str, str] = {
    "BRK.B": "BRK-B",
    "BRK.BR": "BRK-B",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "ADAUSD": "ADA-USD",
    "SOLUSD": "SOL-USD",
    "DOTUSD": "DOT-USD",
    "UNIUSD": "UNI-USD",
    "AAVEUSD": "AAVE-USD",
    "COMPUSD": "COMP-USD",
    "MKRUSD": "MKR-USD",
    "YFIUSD": "YFI-USD",
    "VIX": "^VIX",
}

# ---------------------------------------------------------------------------
# Helper data classes
# ---------------------------------------------------------------------------

@dataclass
class SymbolMetadata:
    """Container describing a trading symbol."""

    symbol: str
    asset_classes: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    @property
    def primary_asset_class(self) -> str:
        """Return the dominant asset class for downstream decisions."""
        preferred_order = ["crypto", "etf", "equity"]
        for candidate in preferred_order:
            if candidate in self.asset_classes:
                return candidate
        return self.asset_classes[0] if self.asset_classes else "equity"


@dataclass
class RemediationResult:
    """Record the outcome of processing a symbol."""

    symbol: str
    action: str
    status: str
    notes: List[str] = field(default_factory=list)

    def add_note(self, note: str) -> None:
        self.notes.append(note)


# ---------------------------------------------------------------------------
# Core remediation class
# ---------------------------------------------------------------------------

class RLDataGapRemediator:
    """Encapsulates the logic required to repair RL data gaps."""

    def __init__(
        self,
        dry_run: bool = False,
        create_neutral_sentiment: bool = True,
    ) -> None:
        self.dry_run = dry_run
        self.create_neutral_sentiment = create_neutral_sentiment
        self.loader = HistoricalDataLoader()
        self.indicator_calculator = TechnicalIndicatorCalculator(DATA_ROOT / "historical")
        self.sentiment_attacher = SentimentAttacher()
        self.symbol_catalog = self._build_symbol_catalog()

    # ------------------------------------------------------------------
    # Public orchestration methods
    # ------------------------------------------------------------------
    def remediate(
        self,
        include_missing: bool = True,
        include_incomplete: bool = True,
        symbols: Optional[Iterable[str]] = None,
        validation_report_path: Path = VALIDATION_REPORT_PATH,
    ) -> List[RemediationResult]:
        """Run remediation across the requested symbol cohorts."""
        report = self._load_validation_report(validation_report_path)
        requested = {s.upper() for s in symbols} if symbols else None

        queue: List[Tuple[str, str]] = []
        for entry in report.get("symbols", []):
            symbol = entry["symbol"].upper()
            if requested and symbol not in requested:
                continue
            status = entry.get("status")
            if status == "missing" and include_missing:
                queue.append((symbol, "missing"))
            elif status == "incomplete" and include_incomplete:
                queue.append((symbol, "incomplete"))

        LOGGER.info("Prepared remediation queue with %s items", len(queue))

        results: List[RemediationResult] = []
        for symbol, issue in queue:
            if issue == "missing":
                results.append(self._remediate_missing_symbol(symbol))
            else:
                results.append(self._remediate_incomplete_symbol(symbol))

        return results

    def run_validator(self) -> None:
        """Re-run the official validation script to refresh the report."""
        LOGGER.info("Re-running RL data readiness validator...")
        data_validator.main()
        LOGGER.info("Validation complete. Updated report saved to %s", VALIDATION_REPORT_PATH)

    # ------------------------------------------------------------------
    # Missing symbol remediation
    # ------------------------------------------------------------------
    def _remediate_missing_symbol(self, symbol: str) -> RemediationResult:
        result = RemediationResult(symbol=symbol, action="missing", status="skipped")

        if self.dry_run:
            result.status = "dry-run"
            result.add_note("Would fetch full history and build dataset")
            return result

        LOGGER.info("Attempting to remediate missing symbol %s", symbol)

        # First, try to fetch with yfinance if available. This covers crypto,
        # synthetic symbols, or tickers that Alpaca does not expose.
        df: Optional[pd.DataFrame] = None
        if YFINANCE_AVAILABLE:
            df = self._fetch_from_yfinance(symbol, result)
        else:
            result.add_note("yfinance unavailable; skipping fallback provider")

        if df is None:
            # As a fallback, try HistoricalDataLoader (requires Alpaca credentials).
            df = self._fetch_with_loader(symbol, result)

        if df is None:
            result.status = "failed"
            result.add_note("No data retrieved from any provider")
            return result

        if not self._persist_symbol_dataset(symbol, df, result):
            return result

        result.status = "ok"
        return result

    # ------------------------------------------------------------------
    # Incomplete symbol remediation
    # ------------------------------------------------------------------
    def _remediate_incomplete_symbol(self, symbol: str) -> RemediationResult:
        result = RemediationResult(symbol=symbol, action="incomplete", status="skipped")

        if self.dry_run:
            result.status = "dry-run"
            result.add_note("Would force refresh from Alpaca and align hourly grid")
            return result

        LOGGER.info("Repairing incomplete coverage for %s", symbol)

        # Force-refresh via HistoricalDataLoader
        refreshed_df = self._fetch_with_loader(symbol, result, force_refresh=True)
        if refreshed_df is None:
            result.add_note("Loader refresh returned no data; attempting yfinance fallback")
            if YFINANCE_AVAILABLE:
                refreshed_df = self._fetch_from_yfinance(symbol, result)

        if refreshed_df is None:
            existing_df = self._load_existing_symbol(symbol)
            if existing_df is None:
                result.status = "failed"
                result.add_note("Existing parquet missing and no new data acquired")
                return result
            refreshed_df = existing_df

        if not self._persist_symbol_dataset(symbol, refreshed_df, result):
            return result

        result.status = "ok"
        return result

    # ------------------------------------------------------------------
    # Data acquisition helpers
    # ------------------------------------------------------------------
    def _fetch_with_loader(
        self,
        symbol: str,
        result: RemediationResult,
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Download bars using the HistoricalDataLoader (Alpaca-based)."""
        try:
            LOGGER.debug("Fetching %s via HistoricalDataLoader", symbol)
            start_date = TARGET_START
            end_date = TARGET_END
            df = self.loader.load_historical_data(
                symbol=symbol,
                timeframe="hour",
                years=2,
                start_date=start_date,
                end_date=end_date,
                verbose=False,
                append=False,
            )
            if df is not None and not df.empty:
                result.add_note("Downloaded via HistoricalDataLoader")
                return df

            if force_refresh:
                # When force refreshing fails, we still return None so fallback
                # mechanisms can attempt remediation.
                result.add_note("Loader returned no rows during force refresh")
            return None
        except Exception as exc:  # pragma: no cover - network dependent
            result.add_note(f"Loader exception: {exc}")
            LOGGER.debug("HistoricalDataLoader failed for %s", symbol, exc_info=True)
            return None

    def _fetch_from_yfinance(
        self, symbol: str, result: RemediationResult
    ) -> Optional[pd.DataFrame]:  # pragma: no cover - requires network
        if not YFINANCE_AVAILABLE:
            return None

        ticker = self._resolve_yfinance_symbol(symbol)
        try:
            LOGGER.debug("Downloading %s from yfinance (%s)", symbol, ticker)
            target_end = min(TARGET_END, datetime.now(timezone.utc))
            target_start = TARGET_START

            if target_end <= target_start:
                result.add_note("Target window not positive; skipping yfinance fetch")
                return None

            max_window = pd.Timedelta(days=729)  # yfinance hourly hard limit ~730 days
            target_end_ts = pd.Timestamp(target_end)
            cursor = pd.Timestamp(target_start)
            frames: List[pd.DataFrame] = []

            while cursor < target_end_ts:
                proposed_end = cursor + max_window
                chunk_end = proposed_end if proposed_end <= target_end_ts else target_end_ts
                start_arg = cursor.to_pydatetime().replace(tzinfo=None)
                # yfinance treats end as exclusive; add one hour to capture final bar
                end_arg = (chunk_end + pd.Timedelta(hours=1)).to_pydatetime().replace(tzinfo=None)
                LOGGER.debug("yfinance chunk %s → %s", start_arg, end_arg)

                chunk = yf.download(
                    ticker,
                    start=start_arg,
                    end=end_arg,
                    interval="1h",
                    auto_adjust=False,
                    progress=False,
                    prepost=True,
                    threads=False,
                )

                if chunk is None or chunk.empty:
                    result.add_note(
                        f"yfinance chunk returned no data for alias {ticker} \u2013 range {start_arg.date()} to {chunk_end.date()}"
                    )
                else:
                    frames.append(chunk)

                # Advance cursor to next hour after chunk_end to avoid overlap
                cursor = chunk_end + pd.Timedelta(hours=1)

            if not frames:
                result.add_note(f"yfinance returned no data for alias {ticker}")
                return None

            raw = pd.concat(frames).sort_index()
            raw = raw[~raw.index.duplicated(keep="last")].copy()
            raw.index = pd.to_datetime(raw.index, utc=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col not in raw.columns:
                    result.add_note(f"yfinance dataset missing column {col}")
                    return None

            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.reset_index(inplace=True)
            df.rename(columns={"index": "timestamp", "Datetime": "timestamp"}, inplace=True)
            result.add_note(f"Fetched {len(df):,} rows via yfinance alias {ticker} in {len(frames)} chunk(s)")
            return df
        except Exception as exc:
            result.add_note(f"yfinance exception for {ticker}: {exc}")
            LOGGER.debug("yfinance download failed for %s", symbol, exc_info=True)
            return None

    def _resolve_yfinance_symbol(self, symbol: str) -> str:
        if symbol in YFINANCE_ALIAS:
            return YFINANCE_ALIAS[symbol]
        if symbol.endswith("USD") and symbol not in YFINANCE_ALIAS:
            # Convert e.g. BTCUSD → BTC-USD
            return f"{symbol[:-3]}-USD"
        if "." in symbol:
            return symbol.replace(".", "-")
        return symbol

    # ------------------------------------------------------------------
    # Data persistence & enrichment
    # ------------------------------------------------------------------
    def _persist_symbol_dataset(
        self, symbol: str, df: pd.DataFrame, result: RemediationResult
    ) -> bool:
        symbol_meta = self.symbol_catalog.get(symbol, SymbolMetadata(symbol=symbol))
        result.add_note(f"Asset class: {symbol_meta.primary_asset_class}")

        try:
            enriched_df = self._prepare_base_dataframe(symbol, df, symbol_meta.primary_asset_class)
        except Exception as exc:
            result.status = "failed"
            result.add_note(f"Failed to normalise dataset: {exc}")
            LOGGER.debug("Normalisation failed for %s", symbol, exc_info=True)
            return False

        if enriched_df is None:
            result.status = "failed"
            result.add_note("Failed to create base dataframe")
            return False

        if self.dry_run:
            result.status = "dry-run"
            result.add_note("Dry-run: skipping disk writes")
            return False

        file_path = HISTORICAL_ROOT / symbol / TIMEFRAME / "data.parquet"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        enriched_df.to_parquet(file_path, index=False)
        result.add_note(f"Wrote base parquet with {len(enriched_df):,} rows")

        # Recompute indicators (this overwrites the parquet with full feature set)
        self.indicator_calculator.process_symbol(symbol)
        result.add_note("Technical indicators regenerated")

        # Ensure sentiment coverage
        if self.create_neutral_sentiment:
            self._ensure_sentiment(seed_symbol=symbol, asset_class=symbol_meta.primary_asset_class)
        attach_summary = self.sentiment_attacher.process_symbol(symbol)
        result.add_note(f"Sentiment attachment status: {attach_summary.get('status', 'unknown')}")
        return True

    def _prepare_base_dataframe(
        self, symbol: str, df: pd.DataFrame, asset_class: str
    ) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None

        working = df.copy()
        if "timestamp" not in working.columns:
            if working.index.name in {"timestamp", None} or isinstance(working.index, pd.DatetimeIndex):
                working = working.reset_index()
                working.rename(columns={working.columns[0]: "timestamp"}, inplace=True)
            else:
                raise ValueError("DataFrame must include a timestamp column")

        if "timestamp" not in working.columns:
            raise ValueError("Unable to determine timestamp column after normalization")

        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
        working = working.sort_values("timestamp").drop_duplicates(subset="timestamp")

        # Compute derived base features prior to alignment
        working["VWAP"] = (working["High"] + working["Low"] + working["Close"]) / 3
        working["OHLC_avg"] = (working["Open"] + working["High"] + working["Low"] + working["Close"]) / 4
        working["Returns"] = working["Close"].pct_change().fillna(0.0)
        working["HL_diff"] = working["High"] - working["Low"]
        working["Volume"] = working["Volume"].fillna(0.0)

        aligned = self._align_to_hourly_grid(working, asset_class)
        aligned = aligned[BASE_COLUMNS]
        aligned["timestamp"] = aligned["timestamp"].dt.tz_convert(timezone.utc)
        return aligned

    def _align_to_hourly_grid(self, df: pd.DataFrame, asset_class: str) -> pd.DataFrame:
        hourly_index = pd.date_range(TARGET_START, TARGET_END, freq="1h", tz=timezone.utc, inclusive="left")

        base = df.set_index("timestamp").reindex(hourly_index)
        # Forward-fill price-like columns, retaining the latest observed value
        for col in ["Open", "High", "Low", "Close", "VWAP", "OHLC_avg"]:
            base[col] = base[col].ffill().bfill()

        if asset_class == "crypto":
            base["Volume"] = base["Volume"].fillna(0.0)
        else:
            # Equities/ETFs: treat non-trading hours as zero volume but keep price constant
            base["Volume"] = base["Volume"].fillna(0.0)

        base["Returns"] = base["Close"].pct_change().fillna(0.0)
        base["HL_diff"] = base["High"] - base["Low"]
        base.reset_index(inplace=True)
        base.rename(columns={"index": "timestamp"}, inplace=True)
        return base

    # ------------------------------------------------------------------
    # Sentiment helpers
    # ------------------------------------------------------------------
    def _ensure_sentiment(self, seed_symbol: str, asset_class: str) -> None:
        sentiment_path = SENTIMENT_ROOT / seed_symbol / "daily_sentiment.parquet"
        if sentiment_path.exists():
            return

        sentiment_path.parent.mkdir(parents=True, exist_ok=True)
        if asset_class == "crypto":
            date_range = pd.date_range(TARGET_START.date(), TARGET_END.date(), freq="D")
        else:
            date_range = pd.bdate_range(TARGET_START.date(), TARGET_END.date())

        placeholder = pd.DataFrame(
            {
                "date": date_range,
                "sentiment_score": np.full(len(date_range), 0.5, dtype=float),
                "news_count": np.zeros(len(date_range), dtype=int),
                "model_used": "neutral_seed",
            }
        )
        placeholder.to_parquet(sentiment_path, index=False)
        LOGGER.info("Seeded neutral sentiment series for %s (%s rows)", seed_symbol, len(placeholder))

    # ------------------------------------------------------------------
    # Support functions
    # ------------------------------------------------------------------
    def _build_symbol_catalog(self) -> Dict[str, SymbolMetadata]:
        if not SYMBOL_CONFIG_PATH.exists():
            LOGGER.warning("symbols.json not found; returning empty catalog")
            return {}

        with SYMBOL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        catalog: Dict[str, SymbolMetadata] = {}

        def register(sym: str, asset_class: str, category: str) -> None:
            record = catalog.setdefault(sym.upper(), SymbolMetadata(sym.upper()))
            if asset_class not in record.asset_classes:
                record.asset_classes.append(asset_class)
            if category not in record.categories:
                record.categories.append(category)

        for sector, symbols in config.get("sectors", {}).items():
            for sym in symbols:
                register(sym, "equity", f"sector:{sector}")

        for index_name, symbols in config.get("indices", {}).items():
            for sym in symbols:
                register(sym, "equity", f"index:{index_name}")

        for group, symbols in config.get("etfs", {}).items():
            for sym in symbols:
                register(sym, "etf", f"etf:{group}")

        for group, symbols in config.get("crypto", {}).items():
            for sym in symbols:
                register(sym, "crypto", f"crypto:{group}")

        return catalog

    def _load_validation_report(self, path: Path) -> Dict[str, object]:
        if not path.exists():
            raise FileNotFoundError(
                f"Validation report not found at {path}. Run validate_rl_data_readiness.py first."
            )
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_existing_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        file_path = HISTORICAL_ROOT / symbol / TIMEFRAME / "data.parquet"
        if not file_path.exists():
            return None
        df = pd.read_parquet(file_path)
        if "timestamp" not in df.columns:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Remediate RL historical data gaps")
    parser.add_argument("--symbols", nargs="*", help="Subset of symbols to process")
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Attempt remediation for symbols marked as missing",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Attempt remediation for symbols marked as incomplete",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Describe actions without writing to disk",
    )
    parser.add_argument(
        "--no-neutral-sentiment",
        action="store_true",
        help="Do not seed neutral sentiment files when missing",
    )
    parser.add_argument(
        "--run-validation",
        action="store_true",
        help="Re-run the validation script after remediation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if not args.include_missing and not args.include_incomplete:
        parser.error("At least one of --include-missing or --include-incomplete must be provided")

    _configure_logging(args.verbose)

    remediator = RLDataGapRemediator(
        dry_run=args.dry_run,
        create_neutral_sentiment=not args.no_neutral_sentiment,
    )

    results = remediator.remediate(
        include_missing=args.include_missing,
        include_incomplete=args.include_incomplete,
        symbols=args.symbols,
    )

    if not results:
        LOGGER.info("No symbols matched the requested criteria")
    else:
        for item in results:
            LOGGER.info(
                "[%s] %s → %s | %s",
                item.action.upper(),
                item.symbol,
                item.status,
                "; ".join(item.notes),
            )

    if args.run_validation and not args.dry_run:
        remediator.run_validator()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
