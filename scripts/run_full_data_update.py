#!/usr/bin/env python3
"""Unified end-to-end data update pipeline.

This orchestration script refreshes historical data, regenerates derived
features, validates RL readiness, runs automated remediation for gaps, and
rebuilds the supervised learning training dataset in ``data/training_data_v2_final``.

High-level steps
----------------
1. Download/refresh hourly historical data for all configured symbols.
2. Recompute the technical-indicator feature bundle.
3. Forward-fill sentiment scores to hourly granularity.
4. Run the RL data readiness validator and optionally trigger remediation.
5. Re-run validation to confirm coverage improvements.
6. Run lightweight spot checks for October 2025 coverage.
7. Regenerate the combined training dataset with the latest labels.
8. Validate the regenerated dataset (feature count, class balance, volume).

Example
-------
Run the full pipeline with default settings:

    python scripts/run_full_data_update.py

Append only the most recent month of data and skip remediation:

    python scripts/run_full_data_update.py --start-date 2025-09-01 \
        --max-workers 6 --skip-remediation

The script prints a JSON-style summary at the end so it can be monitored from CI
or higher-level schedulers.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Local imports
from core.feature_calculator import TechnicalIndicatorCalculator
from core.hist_data_loader import HistoricalDataLoader
from scripts import generate_combined_training_data as training_data_script
from scripts.attach_sentiment_to_hourly import SentimentAttacher
from scripts.remediate_rl_data_gaps import RLDataGapRemediator
from scripts.validate_rl_data_readiness import main as run_rl_validator
from scripts.verify_data_update_success import verify_data_updates

LOGGER = logging.getLogger("DataUpdatePipeline")
VALIDATION_REPORT_PATH = PROJECT_ROOT / "data" / "validation_report.json"
HISTORICAL_DATA_ROOT = PROJECT_ROOT / "data" / "historical"


@dataclass
class StepResult:
    """Container describing the outcome of a pipeline stage."""

    status: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: str | None = None
    skipped: bool = False

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "status": self.status,
            "metrics": self.metrics,
            "skipped": self.skipped,
        }
        if self.details:
            payload["details"] = self.details
        return payload


class DataUpdatePipeline:
    """Execute the multi-stage data refresh workflow."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_dir = (PROJECT_ROOT / args.output_dir).resolve()
        self.output_dir_relative = Path(args.output_dir)
        self.max_workers = args.max_workers
        self.years = args.years
        self.force_refresh = args.force_refresh
        self.remediation_dry_run = args.remediation_dry_run
        self.min_total_samples = args.min_total_samples
        self.positive_ratio_min = args.positive_ratio_min
        self.positive_ratio_max = args.positive_ratio_max
        self.feature_count_min = args.feature_count_min

        self.start_dt: Optional[datetime] = self._parse_date(args.start_date)
        self.end_dt: Optional[datetime] = self._parse_date(args.end_date)
        if self.start_dt and not self.end_dt:
            self.end_dt = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError as exc:  # pragma: no cover - defensive programming
            raise ValueError(f"Invalid date format '{value}'. Use YYYY-MM-DD.") from exc

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        summary: Dict[str, Dict[str, Any]] = {}
        LOGGER.info("Starting full data update pipeline")

        summary["historical_download"] = self._execute_step(
            name="historical_download",
            skip=self.args.skip_download,
            runner=self._download_historical_data,
        )

        summary["technical_indicators"] = self._execute_step(
            name="technical_indicators",
            skip=self.args.skip_indicators,
            runner=self._recompute_technical_indicators,
        )

        summary["sentiment_attachment"] = self._execute_step(
            name="sentiment_attachment",
            skip=self.args.skip_sentiment,
            runner=self._attach_sentiment,
        )

        pre_validation = self._execute_step(
            name="rl_validation_pre",
            skip=self.args.skip_validation,
            runner=self._run_rl_validation,
        )
        summary["rl_validation_pre"] = pre_validation

        summary["remediation"] = self._execute_step(
            name="remediation",
            skip=self.args.skip_remediation,
            runner=lambda: self._run_remediation(pre_validation.get("metrics")),
        )

        if not self.args.skip_validation:
            summary["rl_validation_post"] = self._execute_step(
                name="rl_validation_post",
                skip=self.args.skip_validation,
                runner=self._run_rl_validation,
            )

        summary["sample_verification"] = self._execute_step(
            name="sample_verification",
            skip=self.args.skip_sample_verification,
            runner=self._run_sample_verification,
        )

        summary["training_data_generation"] = self._execute_step(
            name="training_data_generation",
            skip=self.args.skip_training,
            runner=self._generate_training_dataset,
        )

        summary["training_data_validation"] = self._execute_step(
            name="training_data_validation",
            skip=self.args.skip_training_validation,
            runner=self._validate_training_dataset,
        )

        LOGGER.info("Data update pipeline complete")
        return summary

    # ------------------------------------------------------------------
    def _execute_step(self, name: str, skip: bool, runner) -> Dict[str, Any]:
        if skip:
            LOGGER.info("Step %s skipped via CLI flag", name)
            return StepResult(status="skipped", skipped=True).as_dict()
        LOGGER.info("Running step: %s", name)
        try:
            result = runner()
            if isinstance(result, StepResult):
                payload = result.as_dict()
            elif isinstance(result, dict):
                payload = result
            else:
                payload = StepResult(status="ok", metrics={"result": result}).as_dict()
            LOGGER.info("Step %s finished with status %s", name, payload.get("status"))
            return payload
        except Exception as exc:
            LOGGER.exception("Step %s failed", name)
            return StepResult(status="error", details=str(exc)).as_dict()

    # ------------------------------------------------------------------
    def _download_historical_data(self) -> StepResult:
        loader = HistoricalDataLoader()
        kwargs = {
            "timeframe": "hour",
            "verbose": True,
            "max_workers": self.max_workers,
            "force_refresh": self.force_refresh,
        }
        if self.start_dt and self.end_dt:
            kwargs.update({
                "start_date": self.start_dt,
                "end_date": self.end_dt,
                "append": True,
            })
            LOGGER.info(
                "Downloading historical data in append mode from %s to %s",
                self.start_dt.date(),
                self.end_dt.date(),
            )
        else:
            kwargs["years"] = self.years
            LOGGER.info("Downloading the last %.2f years of historical data", self.years)

        loader.load_all_symbols(**kwargs)
        return StepResult(status="ok", metrics={"mode": "append" if "append" in kwargs else "full"})

    # ------------------------------------------------------------------
    def _available_symbols(self) -> List[str]:
        if not HISTORICAL_DATA_ROOT.exists():
            return []
        symbols: List[str] = []
        for symbol_dir in HISTORICAL_DATA_ROOT.iterdir():
            data_path = symbol_dir / "1Hour" / "data.parquet"
            if data_path.exists():
                symbols.append(symbol_dir.name)
        symbols.sort()
        return symbols

    # ------------------------------------------------------------------
    def _recompute_technical_indicators(self) -> StepResult:
        symbols = self._available_symbols()
        if not symbols:
            return StepResult(status="error", details="No symbols with historical data found")

        calculator = TechnicalIndicatorCalculator(str(HISTORICAL_DATA_ROOT))
        success = 0
        for symbol in symbols:
            if calculator.process_symbol(symbol):
                success += 1
        metrics = {
            "symbols": len(symbols),
            "processed": success,
            "failed": len(symbols) - success,
        }
        status = "ok" if success == len(symbols) else "warning"
        return StepResult(status=status, metrics=metrics)

    # ------------------------------------------------------------------
    def _attach_sentiment(self) -> StepResult:
        symbols = self._available_symbols()
        if not symbols:
            return StepResult(status="error", details="No symbols available for sentiment attachment")
        attacher = SentimentAttacher()
        summary = attacher.process_all_symbols(symbols)
        metrics = {
            "total_symbols": summary.get("total_symbols"),
            "processed": summary.get("processed"),
            "skipped": summary.get("skipped"),
            "failed": summary.get("failed"),
        }
        status = "ok" if summary.get("failed", 0) == 0 else "warning"
        return StepResult(status=status, metrics=metrics)

    # ------------------------------------------------------------------
    def _run_rl_validation(self) -> StepResult:
        LOGGER.info("Executing RL data readiness validator")
        run_rl_validator()
        if not VALIDATION_REPORT_PATH.exists():
            return StepResult(status="error", details="Validation report not generated")
        with VALIDATION_REPORT_PATH.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        metrics = {
            "coverage_pct": report.get("coverage_pct"),
            "valid_symbols": report.get("valid_symbols"),
            "total_symbols": report.get("total_symbols"),
            "status_counts": report.get("status_counts", {}),
        }
        status = "ok" if metrics["status_counts"].get("missing", 0) == 0 and metrics["status_counts"].get("incomplete", 0) == 0 else "warning"
        return StepResult(status=status, metrics=metrics)

    # ------------------------------------------------------------------
    def _run_remediation(self, validation_metrics: Optional[Dict[str, Any]]) -> StepResult:
        status_counts = (validation_metrics or {}).get("status_counts", {})
        include_missing = status_counts.get("missing", 0) > 0
        include_incomplete = status_counts.get("incomplete", 0) > 0
        if not include_missing and not include_incomplete:
            LOGGER.info("No missing/incomplete symbols flagged; remediator will run to confirm coverage.")
            include_missing = True
            include_incomplete = True

        remediator = RLDataGapRemediator(dry_run=self.remediation_dry_run)
        results = remediator.remediate(
            include_missing=include_missing,
            include_incomplete=include_incomplete,
        )
        if not self.remediation_dry_run:
            remediator.run_validator()
        outstanding = status_counts.get("missing", 0) + status_counts.get("incomplete", 0)
        metrics = {
            "processed": len(results),
            "dry_run": self.remediation_dry_run,
            "include_missing": include_missing,
            "include_incomplete": include_incomplete,
            "outstanding_before": outstanding,
        }
        if outstanding == 0 and not results:
            status = "ok"
        else:
            status = "ok" if results else "warning"
        return StepResult(status=status, metrics=metrics)

    # ------------------------------------------------------------------
    def _run_sample_verification(self) -> StepResult:
        success = verify_data_updates()
        status = "ok" if success else "warning"
        return StepResult(status=status, metrics={"sample_success": success})

    # ------------------------------------------------------------------
    def _generate_training_dataset(self) -> StepResult:
        available_symbols = training_data_script.get_available_symbols()
        if not available_symbols:
            return StepResult(status="error", details="No symbols available for training data generation")

        config = training_data_script.create_training_config(available_symbols)
        config["lookback_window"] = self.args.lookback
        config["prediction_horizon_hours"] = self.args.prediction_horizon
        config["prediction_horizon"] = self.args.prediction_horizon
        config["profit_target"] = self.args.profit_target
        config["stop_loss"] = self.args.stop_loss
        config["stop_loss_target"] = self.args.stop_loss

        LOGGER.info(
            "Generating combined training data → %s (symbols=%s, lookback=%s, horizon=%s)",
            self.output_dir_relative,
            len(available_symbols),
            config["lookback_window"],
            config["prediction_horizon_hours"],
        )

        data_preparer = training_data_script.NNDataPreparer(config)
        data_splits = data_preparer.get_prepared_data_for_training()
        data_splits["symbols_processed"] = available_symbols
        data_splits["lookback_window"] = config["lookback_window"]
        data_splits["prediction_horizon_hours"] = config["prediction_horizon_hours"]
        data_splits["prediction_horizon"] = config["prediction_horizon"]
        data_splits["profit_target"] = config["profit_target"]
        data_splits["stop_loss"] = config["stop_loss"]

        training_data_script.save_training_data(data_splits, str(self.output_dir_relative))

        metrics = {
            "output_dir": str(self.output_dir_relative),
            "symbols": len(available_symbols),
            "train_samples": int(data_splits["train"]["X"].shape[0]) if "train" in data_splits else 0,
            "val_samples": int(data_splits["val"]["X"].shape[0]) if "val" in data_splits else 0,
            "test_samples": int(data_splits["test"]["X"].shape[0]) if "test" in data_splits else 0,
            "feature_count": int(data_splits["train"]["X"].shape[2]) if "train" in data_splits else 0,
        }
        return StepResult(status="ok", metrics=metrics)

    # ------------------------------------------------------------------
    def _validate_training_dataset(self) -> StepResult:
        dataset_dir = self.output_dir
        if not dataset_dir.exists():
            return StepResult(status="error", details=f"Dataset directory {dataset_dir} not found")

        metadata_path = dataset_dir / "metadata.json"
        if not metadata_path.exists():
            return StepResult(status="error", details="metadata.json missing from dataset directory")

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        metrics: Dict[str, Any] = {
            "feature_count": metadata.get("feature_count"),
            "total_sequences": metadata.get("total_sequences"),
            "positive_ratio_train": metadata.get("positive_ratio_train"),
            "positive_ratio_val": metadata.get("positive_ratio_val"),
            "positive_ratio_test": metadata.get("positive_ratio_test"),
        }

        total_sequences = metadata.get("total_sequences", 0)
        feature_count = metadata.get("feature_count", 0)
        positive_ratio = metadata.get("positive_ratio_train", 0.0)

        # Load splits to validate presence
        splits = {}
        for split in ("train", "val", "test"):
            x_path = dataset_dir / f"{split}_X.npy"
            y_path = dataset_dir / f"{split}_y.npy"
            if not x_path.exists() or not y_path.exists():
                return StepResult(status="error", details=f"Missing files for {split} split")
            X = np.load(x_path)
            y = np.load(y_path)
            splits[split] = {"samples": int(len(y)), "positive_ratio": float(y.mean()) if len(y) else 0.0}
        metrics["splits"] = splits

        checks_passed = (
            total_sequences is not None
            and total_sequences >= self.min_total_samples
            and self.positive_ratio_min <= positive_ratio <= self.positive_ratio_max
            and feature_count >= self.feature_count_min
        )

        metrics.update({
            "min_total_samples": self.min_total_samples,
            "positive_ratio_bounds": [self.positive_ratio_min, self.positive_ratio_max],
            "feature_count_min": self.feature_count_min,
            "checks_passed": checks_passed,
        })

        status = "ok" if checks_passed else "warning"
        details = None if checks_passed else "Dataset checks did not meet configured thresholds"
        return StepResult(status=status, metrics=metrics, details=details)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the end-to-end data update pipeline")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD) for append mode", default=None)
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD) for append mode", default=None)
    parser.add_argument("--years", type=float, default=2.0, help="Historical lookback in years if no start date provided")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum parallel workers for data download")
    parser.add_argument("--force-refresh", action="store_true", help="Force-refresh the most recent 24h for each symbol")
    parser.add_argument("--profit-target", type=float, default=0.015, help="Profit target used in label generation")
    parser.add_argument("--stop-loss", type=float, default=0.03, help="Stop loss used in label generation")
    parser.add_argument("--lookback", type=int, default=24, help="Lookback window (hours) for training sequences")
    parser.add_argument("--prediction-horizon", type=int, default=24, help="Prediction horizon for label generation")
    parser.add_argument("--output-dir", type=str, default="data/training_data_v2_final", help="Directory for the generated training dataset")
    parser.add_argument("--min-total-samples", type=int, default=500_000, help="Minimum total sequences expected in the dataset")
    parser.add_argument("--positive-ratio-min", type=float, default=0.01, help="Lower bound for positive label ratio in training split")
    parser.add_argument("--positive-ratio-max", type=float, default=0.15, help="Upper bound for positive label ratio in training split")
    parser.add_argument("--feature-count-min", type=int, default=22, help="Minimum feature count expected in the dataset")

    # Skip flags
    parser.add_argument("--skip-download", action="store_true", help="Skip historical data download step")
    parser.add_argument("--skip-indicators", action="store_true", help="Skip technical indicator recomputation")
    parser.add_argument("--skip-sentiment", action="store_true", help="Skip sentiment attachment step")
    parser.add_argument("--skip-validation", action="store_true", help="Skip RL validation steps")
    parser.add_argument("--skip-remediation", action="store_true", help="Skip RL data gap remediation")
    parser.add_argument("--skip-sample-verification", action="store_true", help="Skip sample October 2025 verification")
    parser.add_argument("--skip-training", action="store_true", help="Skip training dataset generation")
    parser.add_argument("--skip-training-validation", action="store_true", help="Skip training dataset validation step")

    parser.add_argument("--remediation-dry-run", action="store_true", help="Run remediation without writing to disk")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    pipeline = DataUpdatePipeline(args)
    summary = pipeline.run()

    print("\n" + "=" * 80)
    print("DATA UPDATE PIPELINE SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
