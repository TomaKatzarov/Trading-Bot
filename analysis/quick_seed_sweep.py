"""Quick multi-seed PPO sweep runner for Phase 3 agents.

This utility orchestrates short PPO runs across multiple random seeds using the
existing training pipeline. It adjusts output directories to avoid overwriting
baseline checkpoints, aggregates final evaluation metrics, and optionally
persists the summary for rapid inspection before launching full 100k-step jobs.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_phase3_agents import (
    TrainingResult,
    close_mlflow,
    load_config,
    prepare_mlflow,
    train_symbol_agent,
)


def _ensure_sequence(values: Iterable[int]) -> List[int]:
    return list(dict.fromkeys(int(v) for v in values))


def _aggregate_metrics(results: Sequence[TrainingResult]) -> Dict[str, Dict[str, Any]]:
    aggregated: Dict[str, Dict[str, List[float]]] = {}
    fingerprints: Dict[str, str] = {}

    for result in results:
        if result.final_summary is None:
            continue
        symbol_bucket = aggregated.setdefault(result.symbol, {})
        for metric, value in result.final_summary.items():
            if not isinstance(value, (int, float)):
                continue
            symbol_bucket.setdefault(metric, []).append(float(value))
        if result.config_fingerprint:
            fingerprints[result.symbol] = result.config_fingerprint

    summary: Dict[str, Dict[str, Any]] = {}
    for symbol, metrics in aggregated.items():
        symbol_summary: Dict[str, Any] = {}
        for metric, values in metrics.items():
            array = np.array(values, dtype=np.float64)
            symbol_summary[metric] = {
                "values": values,
                "mean": float(np.mean(array)),
                "std": float(np.std(array, ddof=0)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
            }
        if symbol in fingerprints:
            symbol_summary["config_fingerprint"] = fingerprints[symbol]
        summary[symbol] = symbol_summary
    return summary


def _configure_run(
    base_config: Dict[str, Any],
    *,
    seed: int,
    total_timesteps: int,
    n_envs: int | None,
    eval_freq: int | None,
    save_freq: int | None,
    run_tag: str,
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)

    experiment_cfg = config.setdefault("experiment", {})
    training_cfg = config.setdefault("training", {})

    base_name = experiment_cfg.get("name", "phase3")
    experiment_cfg["name"] = f"{base_name}_{run_tag}"

    base_checkpoint_dir = Path(experiment_cfg.get("checkpoint_dir", "models/phase3_checkpoints"))
    base_log_dir = Path(experiment_cfg.get("log_dir", "logs/phase3_training"))

    suffix = f"seed_sweep/{run_tag}/seed_{seed}"
    experiment_cfg["checkpoint_dir"] = str(base_checkpoint_dir / suffix)
    experiment_cfg["log_dir"] = str(base_log_dir / suffix)

    training_cfg["total_timesteps"] = int(total_timesteps)
    if n_envs is not None:
        training_cfg["n_envs"] = max(1, int(n_envs))
    if eval_freq is not None:
        training_cfg["eval_freq"] = max(1, int(eval_freq))
    if save_freq is not None:
        training_cfg["save_freq"] = max(1, int(save_freq))

    training_cfg["seed"] = int(seed)

    return config


def run_sweep(
    *,
    config_path: Path,
    symbols: Sequence[str],
    seeds: Sequence[int],
    total_timesteps: int,
    n_envs: int | None,
    eval_freq: int | None,
    save_freq: int | None,
    device: str | None,
    run_tag: str,
) -> List[TrainingResult]:
    base_config = load_config(config_path)
    results: List[TrainingResult] = []

    for seed in seeds:
        config = _configure_run(
            base_config,
            seed=seed,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            eval_freq=eval_freq,
            save_freq=save_freq,
            run_tag=run_tag,
        )

        training_cfg = config.setdefault("training", {})
        if device is not None:
            training_cfg["device"] = device

        for symbol in symbols:
            try:
                prepare_mlflow(config, symbol)
                result = train_symbol_agent(symbol, config, resume=False)
                results.append(result)
            finally:
                close_mlflow()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short multi-seed PPO sweeps for Phase 3 agents")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=["SPY", "QQQ"],
        help="Symbols to train (space separated)",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        default=[101, 202, 303],
        help="Random seeds to sweep",
    )
    parser.add_argument("--total-timesteps", type=int, default=25000, help="Training steps per run")
    parser.add_argument("--n-envs", type=int, default=None, help="Override number of vector envs")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluation frequency")
    parser.add_argument("--save-freq", type=int, default=25000, help="Checkpoint frequency")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu, cuda, etc.)")
    parser.add_argument("--tag", type=str, default="quick", help="Tag suffix used for experiment/log dirs")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--no-rich", action="store_true", help="Disable Rich live status display")

    args = parser.parse_args()

    if args.no_rich:
        os.environ["PHASE3_DISABLE_RICH"] = "1"

    seeds = _ensure_sequence(args.seeds)
    results = run_sweep(
        config_path=args.config,
        symbols=args.symbols,
        seeds=seeds,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        device=args.device,
        run_tag=args.tag,
    )

    aggregate = _aggregate_metrics(results)

    payload = {
        "config": str(args.config.resolve()),
        "symbols": args.symbols,
        "seeds": seeds,
        "total_timesteps": args.total_timesteps,
        "results": [result.__dict__ for result in results],
        "aggregate": aggregate,
    }

    print(json.dumps(payload, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Aggregate summary written to {args.output}")


if __name__ == "__main__":
    main()
