"""Trading environment validation harness.

This utility performs a battery of checks against the TradingEnvironment prior
to reinforcement-learning training runs. It validates observation/action
spaces, exercises full episodes to ensure stability, inspects reward signal
quality, and benchmarks step/reset performance. Optional helpers collect
additional diagnostics such as reward component breakdowns and trade
statistics. The script is designed to run inside the RL virtual environment
(`trading_rl_env`) where Gymnasium and the RL dependency stack are installed.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from core.rl.environments.trading_env import TradingConfig, TradingEnvironment

COLUMN_ALIASES: Mapping[str, str] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "VWAP": "vwap",
    "MACD_line": "MACD",
    "Stoch_K": "Stochastic_K",
    "Stoch_D": "Stochastic_D",
}

LOWERCASE_BASE_COLUMNS = {"open", "high", "low", "close", "volume", "vwap"}

# ---------------------------------------------------------------------------
# Data discovery helpers
# ---------------------------------------------------------------------------


def discover_data_file(symbol: str, data_root: Path, timeframe: str) -> Path:
    """Return the parquet file for the requested symbol/timeframe.

    Project convention: ``data/historical/<SYMBOL>/<TIMEFRAME>/data.parquet``.
    """

    candidate = data_root / symbol / timeframe / "data.parquet"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Historical parquet not found for symbol '{symbol}' at {candidate}."
        )
    return candidate


def discover_sl_checkpoints(sl_root: Optional[Path]) -> Dict[str, Path]:
    """Locate supervised-learning checkpoints if available.

    The TradingEnvironment expects a mapping of lowercase model names to
    checkpoint paths. We scan directories beneath ``sl_root`` and capture any
    ``model.pt`` files, normalizing keys to the directory name (lowercase).
    """

    if not sl_root:
        return {}
    if not sl_root.exists():
        raise FileNotFoundError(f"SL checkpoint directory not found: {sl_root}")

    checkpoints: Dict[str, Path] = {}
    for subdir in sl_root.iterdir():
        if not subdir.is_dir():
            continue
        model_file = subdir / "model.pt"
        if model_file.exists():
            checkpoints[subdir.name.lower()] = model_file
    return checkpoints


def prepare_dataset(path: Path) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    """Ensure dataset column names align with environment expectations.

    If the incoming parquet already satisfies requirements, the original path
    is returned with ``None`` tempdir. Otherwise, a sanitized copy is written
    to a temporary directory whose handle must be retained by the caller to
    avoid premature cleanup.
    """

    df = pd.read_parquet(path)
    original_columns = list(df.columns)

    # Apply explicit alias mapping first.
    df = df.rename(columns=COLUMN_ALIASES)

    # Normalize base OHLCV columns to lowercase if necessary.
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        lower = col.lower()
        if lower in LOWERCASE_BASE_COLUMNS and col != lower:
            rename_map[col] = lower
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure timestamp column is datetime and sorted.
    if "timestamp" not in df.columns:
        raise ValueError("Dataset is missing required 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.sort_values("timestamp").reset_index(drop=True)

    expected_features = set(TradingEnvironment.DEFAULT_FEATURE_COLUMNS)
    missing = expected_features - set(df.columns)
    if missing:
        raise ValueError(
            "Dataset is missing required feature columns: "
            + ", ".join(sorted(missing))
        )

    modified = original_columns != list(df.columns)

    if not modified:
        return path, None

    tmpdir = tempfile.TemporaryDirectory()
    sanitized_path = Path(tmpdir.name) / "data.parquet"
    df.to_parquet(sanitized_path)
    return sanitized_path, tmpdir


# ---------------------------------------------------------------------------
# Validation routines
# ---------------------------------------------------------------------------


def validate_spaces(env: TradingEnvironment) -> bool:
    """Validate observation and action spaces."""

    print("\n" + "=" * 60)
    print("SPACE VALIDATION")
    print("=" * 60)

    assert env.action_space.n == 7, "Action space should have 7 actions"
    print("✓ Action space: 7 discrete actions")

    obs, _ = env.reset()

    required_keys = {"technical", "sl_probs", "position", "portfolio", "regime"}
    missing = required_keys - set(obs.keys())
    if missing:
        raise KeyError(f"Observation missing keys: {missing}")

    print("✓ Observation space: Dict with 5 components")
    print(f"  - technical: {obs['technical'].shape}")
    print(f"  - sl_probs: {obs['sl_probs'].shape}")
    print(f"  - position: {obs['position'].shape}")
    print(f"  - portfolio: {obs['portfolio'].shape}")
    print(f"  - regime: {obs['regime'].shape}")

    return True


def _run_single_episode(env: TradingEnvironment) -> Tuple[int, float, Dict[str, float]]:
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    for _ in range(env.config.episode_length):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    metrics = {
        "equity": float(info.get("equity", 0.0)),
        "return_pct": float(info.get("total_return_pct", 0.0)),
        "num_trades": float(info.get("num_trades", 0.0)),
    }
    reward_stats = info.get("reward_stats")
    if isinstance(reward_stats, Mapping):
        metrics["reward_mean"] = float(reward_stats.get("total_reward_mean", 0.0))
        metrics["reward_std"] = float(reward_stats.get("total_reward_std", 0.0))
    return steps, total_reward, metrics


def validate_episode_rollout(env: TradingEnvironment, num_episodes: int = 5) -> bool:
    """Run multiple episodes and validate aggregate metrics."""

    print("\n" + "=" * 60)
    print("EPISODE ROLLOUT VALIDATION")
    print("=" * 60)

    episode_lengths: List[int] = []
    episode_returns: List[float] = []
    episode_rewards: List[float] = []

    for ep in range(num_episodes):
        steps, total_reward, metrics = _run_single_episode(env)
        episode_lengths.append(steps)
        episode_returns.append(metrics.get("return_pct", 0.0))
        episode_rewards.append(total_reward)

        print(
            f"Episode {ep + 1}: {steps} steps | Return {metrics.get('return_pct', 0.0):.2f}% | "
            f"Trades {metrics.get('num_trades', 0)} | Reward Σ {total_reward:.4f}"
        )

    print(f"\nAverage episode length: {statistics.mean(episode_lengths):.1f} steps")
    print(f"Average return: {statistics.mean(episode_returns):.2f}%")
    print(f"Average total reward: {statistics.mean(episode_rewards):.4f}")

    return True


def validate_reward_signals(env: TradingEnvironment, num_steps: int = 100) -> bool:
    """Validate reward signal quality."""

    print("\n" + "=" * 60)
    print("REWARD SIGNAL VALIDATION")
    print("=" * 60)

    env.reset()
    rewards: List[float] = []
    reward_components: Dict[str, List[float]] = {}

    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(reward)

        breakdown = info.get("reward_breakdown")
        if isinstance(breakdown, Mapping):
            for key, value in breakdown.items():
                reward_components.setdefault(key, []).append(float(value))

        if term or trunc:
            env.reset()

    rewards_np = np.array(rewards, dtype=np.float64)

    print(f"Reward statistics ({len(rewards_np)} samples):")
    print(f"  Mean: {np.mean(rewards_np):.4f}")
    print(f"  Std:  {np.std(rewards_np):.4f}")
    print(f"  Min:  {np.min(rewards_np):.4f}")
    print(f"  Max:  {np.max(rewards_np):.4f}")
    print(f"  Median: {np.median(rewards_np):.4f}")

    if np.any(np.isnan(rewards_np)):
        print("❌ WARNING: NaN rewards detected")
        return False
    if np.any(np.isinf(rewards_np)):
        print("❌ WARNING: Infinite rewards detected")
        return False

    print("✓ No NaN/Inf rewards")

    if np.std(rewards_np) == 0:
        print("❌ WARNING: Zero reward variance")
        return False

    print("✓ Reward signal has variance")

    if reward_components:
        print("\nReward component means:")
        for key, values in reward_components.items():
            mean = statistics.mean(values)
            std_dev = statistics.pstdev(values) if len(values) > 1 else 0.0
            print(f"  - {key}: mean {mean:.4f}, std {std_dev:.4f}")

    return True


def benchmark_performance(env: TradingEnvironment, num_steps: int = 1000) -> bool:
    """Benchmark environment performance."""

    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARKING")
    print("=" * 60)

    env.reset()

    step_times: List[float] = []
    reset_times: List[float] = []

    for i in range(num_steps):
        action = env.action_space.sample()
        start = time.perf_counter()
        obs, reward, term, trunc, info = env.step(action)
        step_times.append(time.perf_counter() - start)

        if term or trunc or i % 100 == 99:
            start = time.perf_counter()
            env.reset()
            reset_times.append(time.perf_counter() - start)

    if step_times:
        step_ms = np.array(step_times) * 1000
        print(f"Step performance ({len(step_ms)} samples):")
        print(f"  Mean: {np.mean(step_ms):.3f} ms")
        print(f"  Median: {np.median(step_ms):.3f} ms")
        print(f"  P95: {np.percentile(step_ms, 95):.3f} ms")
        print(f"  P99: {np.percentile(step_ms, 99):.3f} ms")
        if np.percentile(step_ms, 95) > 50:
            print("⚠ WARNING: Step P95 latency >50ms")
        else:
            print("✓ Step performance acceptable")

    if reset_times:
        reset_ms = np.array(reset_times) * 1000
        print(f"\nReset performance ({len(reset_ms)} samples):")
        print(f"  Mean: {np.mean(reset_ms):.3f} ms")
        print(f"  Median: {np.median(reset_ms):.3f} ms")

    return True


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------


@dataclass
class ValidationConfig:
    symbol: str
    data_root: Path
    timeframe: str
    num_episodes: int
    benchmark_steps: int
    seed: int
    sl_root: Optional[Path]
    episode_length: int


def build_environment(config: ValidationConfig) -> TradingEnvironment:
    original_path = discover_data_file(config.symbol, config.data_root, config.timeframe)
    data_path, tmpdir = prepare_dataset(original_path)
    sl_checkpoints = discover_sl_checkpoints(config.sl_root)

    trading_config = TradingConfig(
        symbol=config.symbol,
        data_path=data_path,
        sl_checkpoints=sl_checkpoints,
        episode_length=config.episode_length,
    )
    env = TradingEnvironment(trading_config, seed=config.seed)
    if tmpdir is not None:
        setattr(env, "_validation_tmpdir", tmpdir)
    return env


def parse_args(argv: Optional[Iterable[str]] = None) -> ValidationConfig:
    parser = argparse.ArgumentParser(description="Validate trading environment")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Symbol to validate")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/historical",
        help="Root directory containing historical parquet data",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1Hour",
        help="Subdirectory timeframe (e.g. 1Hour, 4Hour)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes for rollout validation",
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=1000,
        help="Number of steps to use for performance benchmarking",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment seeding",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=200,
        help="Episode length override for validation runs",
    )
    parser.add_argument(
        "--sl-root",
        type=str,
        default="models/sl_checkpoints",
        help="Directory containing supervised learning checkpoints",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    data_root = Path(args.data_root)
    sl_root = Path(args.sl_root) if args.sl_root else None

    return ValidationConfig(
        symbol=args.symbol,
        data_root=data_root,
        timeframe=args.timeframe,
        num_episodes=args.num_episodes,
        benchmark_steps=args.benchmark_steps,
        seed=args.seed,
        sl_root=sl_root,
        episode_length=args.episode_length,
    )


def main(argv: Optional[Iterable[str]] = None) -> bool:
    print("=" * 60)
    print("TRADING ENVIRONMENT VALIDATION")
    print("=" * 60)

    cfg = parse_args(argv)

    try:
        env = build_environment(cfg)
    except Exception as exc:
        print(f"❌ Environment initialization failed: {exc}")
        return False

    validations = [
        ("Space validation", lambda: validate_spaces(env)),
        (
            "Episode rollout validation",
            lambda: validate_episode_rollout(env, cfg.num_episodes),
        ),
        ("Reward signal validation", lambda: validate_reward_signals(env)),
        ("Performance benchmarking", lambda: benchmark_performance(env, cfg.benchmark_steps)),
    ]

    all_passed = True
    for label, func in validations:
        try:
            result = func()
            all_passed &= bool(result)
        except Exception as exc:  # pragma: no cover - defensive logging
            all_passed = False
            print(f"❌ {label} raised an exception: {exc}")
            import traceback

            traceback.print_exc()

    env.close()

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    if all_passed:
        print("✅ All validations PASSED")
        print("Environment is ready for training")
    else:
        print("❌ Some validations FAILED")
        print("Review warnings above")

    return all_passed


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    success = main()
    sys.exit(0 if success else 1)
