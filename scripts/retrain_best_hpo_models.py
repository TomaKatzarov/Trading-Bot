#!/usr/bin/env python3
"""Retrain the top-performing HPO trials with full-length training runs.

This orchestration script rebuilds the production-ready MLP, LSTM, and GRU models
identified by the HPO campaign using safer learning rates and extended early-stopping
patience. It ensures each retraining run uses the enhanced Phase 3 dataset and
completes a substantial number of epochs before model selection.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "training_data_v2_final"
TRAINING_SCRIPT = PROJECT_ROOT / "training" / "train_nn_model.py"

MODELS: Dict[str, Dict[str, object]] = {
    "mlp": {
        "config": PROJECT_ROOT / "training" / "config_templates" / "best_mlp_retrain.yaml",
        "hpo_trial": "trial_72",
        "expected_duration_hours": 3.5,
    },
    "lstm": {
        "config": PROJECT_ROOT / "training" / "config_templates" / "best_lstm_retrain.yaml",
        "hpo_trial": "trial_62",
        "expected_duration_hours": 5.5,
    },
    "gru": {
        "config": PROJECT_ROOT / "training" / "config_templates" / "best_gru_retrain.yaml",
        "hpo_trial": "trial_93",
        "expected_duration_hours": 5.5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain the best HPO discoveries with production-ready settings."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODELS.keys()),
        help="Subset of models to retrain (default: all).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for retraining (default: current interpreter).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the retraining commands without executing them.",
    )
    parser.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Skip verifying that the enhanced dataset is present.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments appended to each training invocation.",
    )
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


def retrain_model(
    model_name: str,
    model_info: Dict[str, object],
    python_executable: str,
    dry_run: bool = False,
    extra_args: Iterable[str] | None = None,
) -> bool:
    config_path = Path(model_info["config"]).resolve()
    if not config_path.exists():
        print(f"❌ Config not found for {model_name}: {config_path}")
        return False

    cmd = [
        python_executable,
        str(TRAINING_SCRIPT),
        "--config",
        str(config_path),
        "--use-pregenerated",
        "--pregenerated-path",
        str(DEFAULT_DATASET),
    ]

    if extra_args:
        cmd.extend(extra_args)

    banner = "=" * 80
    print(f"\n{banner}")
    print(f"RETRAINING {model_name.upper()} (HPO {model_info['hpo_trial']})")
    print(f"Expected duration: ~{model_info['expected_duration_hours']:.1f} hours")
    print(banner)

    if dry_run:
        print("DRY-RUN →", " ".join(cmd))
        return True

    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\n⚠️  Retraining interrupted by user.")
        return False

    duration = time.time() - start_time
    if result.returncode == 0:
        print(
            f"\n✅ {model_name.upper()} retraining completed in {duration/3600:.2f} hours"
            f" ({format_duration(duration)})."
        )
        return True

    print(f"\n❌ {model_name.upper()} retraining failed (exit code {result.returncode}).")
    return False


def main() -> int:
    args = parse_args()

    if not args.skip_data_check and not DEFAULT_DATASET.exists():
        print("❌ Enhanced dataset not found at", DEFAULT_DATASET)
        print(
            "   Run: python scripts/generate_combined_training_data.py --output-dir",
            DEFAULT_DATASET,
        )
        return 1

    target_models = args.models or list(MODELS.keys())
    extra_args = args.extra_args or []

    print("\n" + "=" * 80)
    print("BEST HPO MODELS – PRODUCTION RETRAINING CAMPAIGN")
    print("Addressing premature stopping by enforcing 15+ epochs and stable learning rates")
    print("=" * 80 + "\n")

    results: Dict[str, bool] = {}
    campaign_start = time.time()

    for model_name in target_models:
        info = MODELS[model_name]
        success = retrain_model(
            model_name,
            info,
            python_executable=args.python,
            dry_run=args.dry_run,
            extra_args=extra_args,
        )
        results[model_name] = success

    total_duration = time.time() - campaign_start

    print("\n" + "=" * 80)
    print("RETRAINING CAMPAIGN SUMMARY")
    print("=" * 80)
    print(f"Total wall-clock time: {total_duration/3600:.2f} hours ({format_duration(total_duration)})\n")

    for model, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model.upper():12} {status}")

    successful = sum(1 for outcome in results.values() if outcome)
    total = len(results)
    print(f"\n{successful}/{total} models retrained successfully")
    if successful == total:
        print("\n✅ All models retrained – ready for evaluation & backtesting")

    return 0 if successful == total else 1


if __name__ == "__main__":
    sys.exit(main())
