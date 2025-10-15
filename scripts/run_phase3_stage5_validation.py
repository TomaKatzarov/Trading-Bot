#!/usr/bin/env python3
"""Stage 5 validation pipeline for Phase 3 PPO agents.

This script orchestrates the Stage 5 validation protocol described in the
anti-collapse mitigation plan. It can optionally launch short cross-seed
training sweeps, evaluate the resulting checkpoints on out-of-sample data,
and aggregate quality-gate metrics such as policy entropy and forced-exit
rates.

Typical usage (short cross-seed validation):
    python scripts/run_phase3_stage5_validation.py \
        --config training/config_templates/phase3_ppo_baseline.yaml \
        --symbols SPY \
        --seeds 42 1337 9001 \
        --total-timesteps 20000 \
        --evaluation-episodes 5

To skip training and only evaluate existing checkpoints (e.g., after a full
run completes) pass ``--skip-training``. By default the script writes an
aggregate JSON report under ``reports/stage5_validation`` and exits with a
non-zero code if any quality gate is violated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rl.environments.trading_env import TradeAction  # noqa: E402
from core.rl.environments.vec_trading_env import make_vec_trading_env  # noqa: E402
from training import train_phase3_agents as trainer  # noqa: E402


@dataclass
class RunResult:
    symbol: str
    seed: int
    training_summary: Optional[Dict[str, Any]]
    best_summary: Optional[Dict[str, Any]]
    evaluation: Optional[Dict[str, Any]]
    thresholds: Dict[str, float]
    status: str
    violations: List[str]
    artifacts_dir: Path


ACTION_NAMES: List[str] = [action.name for action in TradeAction]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5 validation pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to Phase 3 PPO YAML config")
    parser.add_argument("--symbols", nargs="*", default=None, help="Subset of symbols to validate (default: config list)")
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[42, 1337, 9001],
        help="Training seeds to evaluate (default: 42 1337 9001)",
    )
    parser.add_argument("--total-timesteps", type=int, default=20_000, help="Training timesteps override for quick sweeps")
    parser.add_argument("--n-envs", type=int, default=2, help="Number of parallel envs during training runs")
    parser.add_argument("--eval-freq", type=int, default=5_000, help="Evaluation frequency override during training")
    parser.add_argument(
        "--save-freq",
        type=int,
        default=None,
        help="Checkpoint save frequency override (default: same as total timesteps)",
    )
    parser.add_argument("--evaluation-episodes", type=int, default=5, help="Episodes per evaluation sweep on the test split")
    parser.add_argument(
        "--evaluation-split",
        choices=["val", "test"],
        default="test",
        help="Dataset split used for out-of-sample evaluation",
    )
    parser.add_argument("--entropy-threshold", type=float, default=0.45, help="Minimum mean policy entropy")
    parser.add_argument("--forced-exit-threshold", type=float, default=0.10, help="Maximum mean forced exit ratio")
    parser.add_argument("--voluntary-trade-threshold", type=float, default=0.10, help="Minimum voluntary trade rate")
    parser.add_argument("--output-dir", type=str, default="reports/stage5_validation", help="Directory for validation artifacts")
    parser.add_argument("--skip-training", action="store_true", help="Skip training runs and only evaluate existing checkpoints")
    parser.add_argument("--device", type=str, default="auto", help="Device override for evaluation (auto/cpu/cuda[:idx])")
    parser.add_argument(
        "--run-pytest",
        action="store_true",
        help="Run pytest suite as part of Stage 5 quality gates",
    )
    parser.add_argument(
        "--pytest-args",
        type=str,
        default=None,
        help="Extra argument string forwarded to pytest (e.g. '--maxfail=1 tests/test_stage5_validation.py')",
    )
    parser.add_argument(
        "--lint-command",
        type=str,
        default=None,
        help="Optional lint command to execute as an additional quality gate (e.g. 'ruff check .')",
    )
    return parser.parse_args(argv)


def run_training(
    config_path: Path,
    symbol: str,
    seed: int,
    total_timesteps: int,
    n_envs: int,
    eval_freq: int,
    save_freq: Optional[int],
) -> None:
    """Launch ``train_phase3_agents.py`` for a given seed/symbol."""

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "training" / "train_phase3_agents.py"),
        "--config",
        str(config_path),
        "--symbols",
        symbol,
        "--seed",
        str(seed),
        "--total-timesteps",
        str(total_timesteps),
        "--n-envs",
        str(n_envs),
        "--eval-freq",
        str(eval_freq),
    ]
    if save_freq is None:
        save_freq = total_timesteps
    cmd.extend(["--save-freq", str(save_freq)])

    print(f"\n[Stage5] Training symbol={symbol} seed={seed} timesteps={total_timesteps} n_envs={n_envs}")
    process = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if process.returncode != 0:
        raise RuntimeError(f"Training failed for {symbol} seed={seed} (exit {process.returncode})")


def copy_artifacts(source_dir: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "final_model.zip",
        "best_model.zip",
        "final_evaluation_summary.json",
        "best_evaluation_summary.json",
        "vecnormalize.pkl",
        "training_monitor.json",
    ]:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, destination_dir / name)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_device(preference: str) -> str:
    preference = preference.lower()
    if preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if preference.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return preference
    if preference == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device preference: {preference}")


def build_vecnorm_kwargs(config: Mapping[str, Any], symbol: str) -> Tuple[bool, Dict[str, Any]]:
    training_cfg = config.get("training", {})
    norm_cfg = training_cfg.get("reward_normalization", {})
    use_vecnormalize = bool(norm_cfg.get("enabled", False))
    vecnorm_kwargs = {
        "norm_obs": bool(norm_cfg.get("norm_obs", False)),
        "norm_reward": bool(norm_cfg.get("norm_reward", True)),
        "clip_reward": float(norm_cfg.get("clip_reward", 5.0)),
        "gamma": float(norm_cfg.get("gamma", config.get("ppo", {}).get("gamma", 0.99))),
    }
    if use_vecnormalize:
        print(
            f"[Stage5] VecNormalize enabled for {symbol} (norm_obs={vecnorm_kwargs['norm_obs']},"
            f" norm_reward={vecnorm_kwargs['norm_reward']}, clip_reward={vecnorm_kwargs['clip_reward']},"
            f" gamma={vecnorm_kwargs['gamma']})"
        )
    return use_vecnormalize, vecnorm_kwargs


def build_eval_env(
    symbol: str,
    split: str,
    config: Mapping[str, Any],
    seed: int,
    vecnorm_stats_path: Path,
    use_vecnormalize: bool,
    vecnorm_kwargs: Mapping[str, Any],
) -> VecEnv:
    env_kwargs = trainer.make_env_kwargs(symbol, split, config)
    env = make_vec_trading_env(
        symbol=symbol,
        data_dir=Path(config["experiment"]["data_dir"]) / symbol,
        num_envs=1,
        seed=seed,
        use_subprocess=False,
        env_kwargs=env_kwargs,
        env_log_level=env_kwargs.get("log_level"),
    )

    if use_vecnormalize:
        env = VecNormalize(env, **vecnorm_kwargs)
        env.training = False
        if vecnorm_stats_path.exists():
            env = VecNormalize.load(str(vecnorm_stats_path), env)
            env.training = False
            env.norm_reward = bool(vecnorm_kwargs.get("norm_reward", False))  # type: ignore[attr-defined]
        else:
            print(f"[Stage5] Warning: VecNormalize stats missing at {vecnorm_stats_path}")
    return env


def _truncate_output(text: Optional[str], *, limit: int = 4000) -> str:
    base = text or ""
    if len(base) <= limit:
        return base
    remainder = len(base) - limit
    return base[:limit] + f"\n... (truncated {remainder} characters)"


def run_quality_gate_command(name: str, command: Sequence[str]) -> Dict[str, Any]:
    normalized_command = [str(part) for part in command]
    print(f"\n[Stage5] Running quality gate '{name}': {' '.join(normalized_command)}")
    start_time = time.time()
    try:
        process = subprocess.run(
            normalized_command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        duration = time.time() - start_time
        message = str(exc)
        print(f"[Stage5] Quality gate '{name}' failed to start: {message}")
        return {
            "name": name,
            "command": normalized_command,
            "status": "error",
            "returncode": -1,
            "duration": duration,
            "stdout": "",
            "stderr": message,
        }

    duration = time.time() - start_time
    status = "passed" if process.returncode == 0 else "failed"
    if status == "passed":
        print(f"[Stage5] Quality gate '{name}' passed in {duration:.2f}s")
    else:
        print(f"[Stage5] Quality gate '{name}' failed (returncode={process.returncode}) in {duration:.2f}s")

    return {
        "name": name,
        "command": normalized_command,
        "status": status,
        "returncode": process.returncode,
        "duration": duration,
        "stdout": _truncate_output(process.stdout),
        "stderr": _truncate_output(process.stderr),
    }


def _distribution_from_counter(counter: Counter[int]) -> Dict[str, float]:
    total = sum(counter.values())
    distribution: Dict[str, float] = {}
    for action in TradeAction:
        count = counter.get(int(action.value), 0)
        distribution[action.name] = float(count / total) if total > 0 else 0.0
    return distribution


def _entropy(values: Iterable[float]) -> float:
    entropy = 0.0
    for value in values:
        if value > 0:
            entropy -= float(value) * math.log(float(value))
    return float(entropy)


def evaluate_checkpoint(
    model_path: Path,
    symbol: str,
    split: str,
    config: Mapping[str, Any],
    seed: int,
    vecnorm_stats_path: Path,
    use_vecnormalize: bool,
    vecnorm_kwargs: Mapping[str, Any],
    episodes: int,
    device: str,
) -> Dict[str, Any]:
    env = build_eval_env(symbol, split, config, seed, vecnorm_stats_path, use_vecnormalize, vecnorm_kwargs)
    model = PPO.load(str(model_path), env=env, device=device)
    model.policy.to(device)

    results: List[Dict[str, Any]] = []
    for episode_idx in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        action_counter: Counter[int] = Counter()
        policy_counter: Counter[int] = Counter()
        step_count = 0
        state = None

        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            action_idx = int(np.asarray(action).reshape(-1)[0])
            policy_counter[action_idx] += 1
            obs, reward, done_array, info_array = env.step(action)
            executed_idx = info_array[0].get("executed_action_idx")
            executed_idx = int(executed_idx) if executed_idx is not None else action_idx
            action_counter[executed_idx] += 1
            total_reward += float(reward[0])
            step_count += 1
            done = bool(done_array[0])

        base_env = env.envs[0]
        portfolio = base_env.portfolio
        metrics = portfolio.get_portfolio_metrics()
        closed_positions = list(portfolio.get_closed_positions())
        forced_exits = sum(1 for trade in closed_positions if trade.get("forced_exit"))
        trade_count = len(closed_positions)
        forced_ratio = forced_exits / trade_count if trade_count > 0 else 0.0
        voluntary_rate = (trade_count - forced_exits) / trade_count if trade_count > 0 else 0.0

        action_dist = _distribution_from_counter(action_counter)
        policy_dist = _distribution_from_counter(policy_counter)

        results.append(
            {
                "episode": episode_idx,
                "total_reward": total_reward,
                "episode_length": step_count,
                "trade_count": trade_count,
                "forced_exit_count": forced_exits,
                "forced_exit_ratio": forced_ratio,
                "voluntary_trade_rate": voluntary_rate,
                "total_return": float(metrics.get("total_return", 0.0)),
                "total_return_pct": float(metrics.get("total_return_pct", 0.0)),
                "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "win_rate": float(metrics.get("win_rate", 0.0)),
                "profit_factor": float(metrics.get("total_pnl", 0.0)),
                "action_distribution": action_dist,
                "policy_action_distribution": policy_dist,
                "action_entropy": _entropy(action_dist.values()),
                "policy_action_entropy": _entropy(policy_dist.values()),
            }
        )

    env.close()

    summary: Dict[str, Any] = {
        "episodes": len(results),
        "per_episode": results,
    }

    def _mean_std(key: str) -> Tuple[float, float]:
        values = [entry[key] for entry in results]
        if not values:
            return 0.0, 0.0
        return float(np.mean(values)), float(np.std(values))

    scalar_keys = [
        "total_reward",
        "episode_length",
        "trade_count",
        "forced_exit_ratio",
        "voluntary_trade_rate",
        "total_return",
        "total_return_pct",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "policy_action_entropy",
        "action_entropy",
    ]

    for key in scalar_keys:
        mean, std = _mean_std(key)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std

    summary["forced_exit_rate"] = summary.get("forced_exit_ratio_mean", 0.0)
    summary["voluntary_trade_rate_mean"] = summary.get("voluntary_trade_rate_mean", 0.0)
    return summary


def evaluate_quality_gates(summary: Mapping[str, Any], thresholds: Mapping[str, float]) -> Tuple[bool, List[str]]:
    violations: List[str] = []
    policy_entropy = float(summary.get("policy_action_entropy_mean", 0.0))
    if policy_entropy < thresholds["entropy"]:
        violations.append(
            f"policy_action_entropy {policy_entropy:.3f} < threshold {thresholds['entropy']:.3f}"
        )

    forced_exit_rate = float(summary.get("forced_exit_rate", 0.0))
    if forced_exit_rate > thresholds["forced_exit"]:
        violations.append(
            f"forced_exit_rate {forced_exit_rate:.3f} > threshold {thresholds['forced_exit']:.3f}"
        )

    voluntary_rate = float(summary.get("voluntary_trade_rate_mean", 0.0))
    if voluntary_rate < thresholds["voluntary"]:
        violations.append(
            f"voluntary_trade_rate {voluntary_rate:.3f} < threshold {thresholds['voluntary']:.3f}"
        )

    return not violations, violations


def stage5_validation(args: argparse.Namespace) -> Tuple[List[RunResult], Dict[str, Any]]:
    config_path = (PROJECT_ROOT / args.config).resolve() if not os.path.isabs(args.config) else Path(args.config)
    config = trainer.load_config(config_path)

    symbols: List[str]
    if args.symbols:
        symbols = list(args.symbols)
    else:
        symbols = list(config.get("experiment", {}).get("symbols", []))
        if not symbols:
            raise ValueError("No symbols specified in config or CLI")

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_root = Path(config["experiment"]["checkpoint_dir"]).resolve()

    thresholds = {
        "entropy": float(args.entropy_threshold),
        "forced_exit": float(args.forced_exit_threshold),
        "voluntary": float(args.voluntary_trade_threshold),
    }

    results: List[RunResult] = []
    quality_gate_results: Dict[str, Any] = {}

    for seed in args.seeds:
        print(f"\n========== Stage 5 Validation :: Seed {seed} ==========")
        for symbol in symbols:
            symbol_checkpoint_dir = checkpoint_root / symbol
            if not args.skip_training:
                run_training(
                    config_path,
                    symbol,
                    seed,
                    args.total_timesteps,
                    args.n_envs,
                    args.eval_freq,
                    args.save_freq,
                )

            seed_artifacts_dir = output_dir / f"seed_{seed}" / symbol
            if symbol_checkpoint_dir.exists():
                copy_artifacts(symbol_checkpoint_dir, seed_artifacts_dir)

            final_summary = read_json(seed_artifacts_dir / "final_evaluation_summary.json") or read_json(
                symbol_checkpoint_dir / "final_evaluation_summary.json"
            )
            best_summary = read_json(seed_artifacts_dir / "best_evaluation_summary.json") or read_json(
                symbol_checkpoint_dir / "best_evaluation_summary.json"
            )

            if final_summary is None:
                print(f"[Stage5] Warning: final summary missing for {symbol} seed {seed}")

            final_model_path = seed_artifacts_dir / "final_model.zip"
            if not final_model_path.exists():
                final_model_path = symbol_checkpoint_dir / "final_model.zip"
            if not final_model_path.exists():
                print(f"[Stage5] Warning: final_model.zip not found for {symbol} seed {seed}; skipping evaluation")
                evaluation_summary = None
                status = "failed"
                violations = ["missing_model"]
            else:
                vecnorm_stats_path = seed_artifacts_dir / "vecnormalize.pkl"
                if not vecnorm_stats_path.exists():
                    vecnorm_stats_path = symbol_checkpoint_dir / "vecnormalize.pkl"

                use_vecnormalize, vecnorm_kwargs = build_vecnorm_kwargs(config, symbol)
                device = resolve_device(args.device)
                evaluation_summary = evaluate_checkpoint(
                    final_model_path,
                    symbol,
                    args.evaluation_split,
                    config,
                    seed + 99,
                    vecnorm_stats_path,
                    use_vecnormalize,
                    vecnorm_kwargs,
                    args.evaluation_episodes,
                    device,
                )

                passed, violations = evaluate_quality_gates(evaluation_summary, thresholds)
                status = "passed" if passed else "failed"

            results.append(
                RunResult(
                    symbol=symbol,
                    seed=seed,
                    training_summary=final_summary,
                    best_summary=best_summary,
                    evaluation=evaluation_summary,
                    thresholds=thresholds,
                    status=status,
                    violations=violations,
                    artifacts_dir=seed_artifacts_dir,
                )
            )

    if args.run_pytest:
        pytest_cmd: List[str] = [sys.executable, "-m", "pytest"]
        if args.pytest_args:
            pytest_cmd.extend(shlex.split(args.pytest_args))
        quality_gate_results["pytest"] = run_quality_gate_command("pytest", pytest_cmd)

    if args.lint_command:
        lint_cmd = shlex.split(args.lint_command)
        if lint_cmd:
            quality_gate_results["lint"] = run_quality_gate_command("lint", lint_cmd)

    aggregate_report = build_report(results, config_path, args, quality_gate_results)
    report_path = output_dir / "stage5_validation_summary.json"
    report_path.write_text(json.dumps(aggregate_report, indent=2), encoding="utf-8")
    print(f"\n[Stage5] Summary report written to {report_path}")
    return results, aggregate_report


def build_report(
    results: Sequence[RunResult],
    config_path: Path,
    args: argparse.Namespace,
    quality_gates: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "config": str(config_path),
        "symbols": sorted({result.symbol for result in results}),
        "seeds": sorted({result.seed for result in results}),
        "thresholds": results[0].thresholds if results else {},
        "runs": [],
    }

    normalized_quality_gates = dict(quality_gates or {})
    payload["quality_gates"] = normalized_quality_gates

    failed_runs = []
    policy_entropies: List[float] = []
    forced_exit_rates: List[float] = []
    voluntary_rates: List[float] = []

    for result in results:
        run_payload: Dict[str, Any] = {
            "symbol": result.symbol,
            "seed": result.seed,
            "status": result.status,
            "violations": result.violations,
            "artifacts_dir": str(result.artifacts_dir),
        }
        if result.training_summary is not None:
            run_payload["training_summary"] = result.training_summary
        if result.best_summary is not None:
            run_payload["best_summary"] = result.best_summary
        if result.evaluation is not None:
            run_payload["evaluation"] = result.evaluation
            policy_entropies.append(float(result.evaluation.get("policy_action_entropy_mean", 0.0)))
            forced_exit_rates.append(float(result.evaluation.get("forced_exit_rate", 0.0)))
            voluntary_rates.append(float(result.evaluation.get("voluntary_trade_rate_mean", 0.0)))
        payload["runs"].append(run_payload)
        if result.status != "passed":
            failed_runs.append({"symbol": result.symbol, "seed": result.seed, "violations": result.violations})

    quality_gate_failures = [
        {
            "gate": name,
            "status": data.get("status"),
            "returncode": data.get("returncode"),
        }
        for name, data in normalized_quality_gates.items()
        if data.get("status") not in {"passed", "skipped"}
    ]

    overall_status = "passed"
    if failed_runs or quality_gate_failures:
        overall_status = "failed"

    payload["overall"] = {
        "status": overall_status,
        "failed_runs": failed_runs,
        "quality_gate_failures": quality_gate_failures,
        "policy_action_entropy_mean": float(np.mean(policy_entropies)) if policy_entropies else 0.0,
        "forced_exit_rate_mean": float(np.mean(forced_exit_rates)) if forced_exit_rates else 0.0,
        "voluntary_trade_rate_mean": float(np.mean(voluntary_rates)) if voluntary_rates else 0.0,
    }

    payload["command"] = {
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "evaluation_split": args.evaluation_split,
        "evaluation_episodes": args.evaluation_episodes,
        "skip_training": args.skip_training,
    }
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    results, aggregate_report = stage5_validation(args)
    for name, gate in aggregate_report.get("quality_gates", {}).items():
        status = gate.get("status", "unknown")
        returncode = gate.get("returncode", "n/a")
        print(f"[Stage5] Quality gate '{name}': {status.upper()} (returncode={returncode})")
    if aggregate_report["overall"]["status"] != "passed":
        print("\n[Stage5] Validation FAILED. See report for details.")
        return 1
    print("\n[Stage5] Validation passed all quality gates.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
