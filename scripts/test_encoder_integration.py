"""Feature Encoder Integration Testing.

Validates the shared feature encoder with REAL `TradingEnvironment` observations
across multiple symbols and batch scenarios.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.rl.environments import TradingConfig, TradingEnvironment  # noqa: E402
from core.rl.policies.feature_encoder import EncoderConfig, FeatureEncoder  # noqa: E402


EXPECTED_FEATURES = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "SMA_10",
    "SMA_20",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "RSI_14",
    "Stochastic_K",
    "Stochastic_D",
    "ADX_14",
    "ATR_14",
    "BB_bandwidth",
    "OBV",
    "Volume_SMA_20",
    "Return_1h",
    "sentiment_score_hourly_ffill",
    "DayOfWeek_sin",
    "DayOfWeek_cos",
}

COLUMN_ALIASES = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "VWAP": "vwap",
    "MACD_line": "MACD",
    "MACD_signal": "MACD_signal",
    "MACD_hist": "MACD_hist",
    "Stoch_K": "Stochastic_K",
    "Stoch_D": "Stochastic_D",
}

INTEGRATION_CACHE_DIR = ROOT_DIR / "analysis" / "integration_cache"


@dataclass
class IntegrationStats:
    """Container for encoder integration test results."""

    symbol: str
    num_episodes: int = 0
    total_steps: int = 0
    nan_count: int = 0
    inf_count: int = 0
    errors: List[str] = None
    output_ranges: List[List[float]] = None
    output_means: List[float] = None
    output_stds: List[float] = None
    episode_lengths: List[int] = None

    def __post_init__(self) -> None:
        self.errors = [] if self.errors is None else self.errors
        self.output_ranges = [] if self.output_ranges is None else self.output_ranges
        self.output_means = [] if self.output_means is None else self.output_means
        self.output_stds = [] if self.output_stds is None else self.output_stds
        self.episode_lengths = [] if self.episode_lengths is None else self.episode_lengths

    @property
    def passed(self) -> bool:
        return (
            self.errors == []
            and self.nan_count == 0
            and self.inf_count == 0
            and self.num_episodes > 0
        )


def resolve_raw_data(symbol: str, root: Path) -> Optional[Path]:
    """Locate the raw parquet file for a symbol within ``root``."""

    symbol_root = root / symbol
    if not symbol_root.exists():
        return None

    parquet_files = sorted(symbol_root.rglob("*.parquet"))
    if not parquet_files:
        return None

    # Prefer 1Hour timeframe if available, otherwise first result
    parquet_files.sort(key=lambda p: ("1Hour" not in p.parts, len(p.parts)))
    return parquet_files[0]


def prepare_data_file(symbol: str, data_root: Path) -> Optional[Path]:
    """Create (or reuse) a sanitized parquet file with expected feature columns."""

    raw_path = resolve_raw_data(symbol, data_root)
    if raw_path is None:
        return None

    INTEGRATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target_path = INTEGRATION_CACHE_DIR / f"{symbol}.parquet"

    if target_path.exists() and target_path.stat().st_mtime >= raw_path.stat().st_mtime:
        return target_path

    df = pd.read_parquet(raw_path)
    df.columns = [col.strip() for col in df.columns]

    rename_map = {col: COLUMN_ALIASES.get(col, col) for col in df.columns}
    df = df.rename(columns=rename_map)

    # Ensure base OHLCV columns exist (case-sensitive after renaming)
    required_columns = set(EXPECTED_FEATURES)
    required_columns.add("timestamp")
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Prepared dataframe for {symbol} is missing required columns: {sorted(missing)}"
        )

    df = df.sort_values("timestamp").reset_index(drop=True)

    keep_columns = ["timestamp"] + [col for col in df.columns if col in EXPECTED_FEATURES]
    df = df[keep_columns]

    df.to_parquet(target_path, index=False)
    return target_path


def to_torch_batch(obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Convert an observation dict of numpy arrays to a torch batch."""

    return {key: torch.from_numpy(value).float().unsqueeze(0) for key, value in obs.items()}


def to_torch_from_batch(batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Convert a batched numpy observation dictionary to torch tensors."""

    return {key: torch.from_numpy(value).float() for key, value in batch.items()}


def verify_shapes(batch: Dict[str, torch.Tensor]) -> None:
    """Ensure observation tensors match expected encoder input shapes."""

    expected_shapes = {
        "technical": (1, 24, 23),
        "sl_probs": (1, 3),
        "position": (1, 5),
        "portfolio": (1, 8),
        "regime": (1, 10),
    }
    for key, expected in expected_shapes.items():
        actual = tuple(batch[key].shape)
        if actual != expected:
            raise AssertionError(f"{key} shape mismatch: expected {expected}, got {actual}")


def run_single_symbol_test(
    symbol: str,
    *,
    num_episodes: int,
    steps_per_episode: int,
    encoder: FeatureEncoder,
    data_root: Path,
    seed_offset: int = 0,
) -> Optional[IntegrationStats]:
    """Run integration test for a single symbol."""

    print("\n" + "=" * 70)
    print(f"Testing Feature Encoder with Real Environment: {symbol}")
    print("=" * 70)

    try:
        data_path = prepare_data_file(symbol, data_root)
    except ValueError as exc:
        print(f"❌ Failed to prepare data for {symbol}: {exc}")
        return None

    if data_path is None:
        print(f"⚠️  Warning: No parquet data found for {symbol} under {data_root}")
        return None

    config = TradingConfig(
        symbol=symbol,
        data_path=data_path,
        sl_checkpoints={},
        episode_length=steps_per_episode,
        log_level=logging.ERROR,
    )

    try:
        env = TradingEnvironment(config, seed=42 + seed_offset)
    except Exception as exc:  # pragma: no cover - runtime integration safegaurd
        print(f"❌ Failed to initialize TradingEnvironment for {symbol}: {exc}")
        return None

    stats = IntegrationStats(symbol=symbol)

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        try:
            obs, _ = env.reset()
            torch_obs = to_torch_batch(obs)
            verify_shapes(torch_obs)

            with torch.no_grad():
                encoded = encoder(torch_obs)

            if torch.isnan(encoded).any():
                stats.nan_count += 1
                print("  ⚠️  NaN detected in encoded output (reset)")
            if torch.isinf(encoded).any():
                stats.inf_count += 1
                print("  ⚠️  Inf detected in encoded output (reset)")

            stats.output_ranges.append([encoded.min().item(), encoded.max().item()])
            stats.output_means.append(encoded.mean().item())
            stats.output_stds.append(encoded.std().item())

            episode_steps = 0
            for step in range(steps_per_episode):
                obs, _, terminated, truncated, _ = env.step(0)
                torch_obs = to_torch_batch(obs)

                with torch.no_grad():
                    encoded = encoder(torch_obs)

                if torch.isnan(encoded).any():
                    stats.nan_count += 1
                    if stats.nan_count <= 3:
                        print(f"  ⚠️  NaN detected at step {step + 1}")
                if torch.isinf(encoded).any():
                    stats.inf_count += 1
                    if stats.inf_count <= 3:
                        print(f"  ⚠️  Inf detected at step {step + 1}")

                stats.output_ranges.append([encoded.min().item(), encoded.max().item()])
                stats.output_means.append(encoded.mean().item())
                stats.output_stds.append(encoded.std().item())

                episode_steps += 1
                stats.total_steps += 1

                if terminated or truncated:
                    break

            stats.episode_lengths.append(episode_steps)
            stats.num_episodes += 1
            print(f"  ✅ Episode completed: {episode_steps} steps")
        except Exception as exc:  # pragma: no cover - runtime integration safeguard
            stats.errors.append(str(exc))
            print(f"  ❌ Error during episode {episode + 1}: {exc}")
            break

    env.close()
    summarize_stats(stats, target_episodes=num_episodes)
    return stats


def summarize_stats(stats: IntegrationStats, *, target_episodes: int) -> None:
    """Print summary statistics for a single-symbol test."""

    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)
    print(f"\nSymbol: {stats.symbol}")
    print(f"Episodes completed: {stats.num_episodes}/{target_episodes}")
    print(f"Total steps: {stats.total_steps}")
    print(f"NaN detections: {stats.nan_count}")
    print(f"Inf detections: {stats.inf_count}")
    print(f"Errors: {len(stats.errors)}")

    if stats.output_ranges:
        ranges = np.asarray(stats.output_ranges)
        mean_values = np.asarray(stats.output_means)
        std_values = np.asarray(stats.output_stds)
        print("\nOutput Statistics:")
        print(f"  Min value: {ranges[:, 0].min():.4f}")
        print(f"  Max value: {ranges[:, 1].max():.4f}")
        print(f"  Mean: {mean_values.mean():.4f} ± {mean_values.std():.4f}")
        print(f"  Std: {std_values.mean():.4f} ± {std_values.std():.4f}")

    verdict = "✅ Integration test PASSED" if stats.passed else "❌ Integration test FAILED"
    print(f"\n{verdict} for {stats.symbol}")
    if stats.errors:
        print("\nErrors encountered:")
        for idx, error in enumerate(stats.errors, 1):
            print(f"  {idx}. {error}")


def test_multi_symbol_integration(
    symbols: Iterable[str],
    *,
    steps_per_episode: int,
    encoder: FeatureEncoder,
    data_root: Path,
) -> Dict[str, Optional[IntegrationStats]]:
    """Run integration tests across multiple symbols."""

    print("\n" + "=" * 70)
    print("MULTI-SYMBOL INTEGRATION TEST")
    print("=" * 70)
    print(f"\nTesting symbols: {', '.join(symbols)}")

    results: Dict[str, Optional[IntegrationStats]] = {}
    for idx, symbol in enumerate(symbols):
        stats = run_single_symbol_test(
            symbol,
            num_episodes=2,
            steps_per_episode=steps_per_episode,
            encoder=encoder,
            data_root=data_root,
            seed_offset=idx + 1,
        )
        results[symbol] = stats

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    successful = sum(1 for s in results.values() if s and s.passed)
    total = sum(1 for s in results.values() if s is not None)
    print(f"\nSuccessful: {successful}/{total} symbols")

    for symbol, stats in results.items():
        if stats is None:
            print(f"  {symbol}: ⚠️  Skipped (data not available)")
        elif stats.passed:
            print(f"  {symbol}: ✅ PASS ({stats.total_steps} steps)")
        else:
            print(
                f"  {symbol}: ❌ FAIL (NaN: {stats.nan_count}, Inf: {stats.inf_count}, "
                f"Errors: {len(stats.errors)})"
            )

    if successful == total and total > 0:
        print("\n" + "=" * 70)
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("⚠️  SOME TESTS FAILED - Investigation needed")
        print("=" * 70)

    return results


def test_batch_encoding(
    symbol: str,
    *,
    batch_size: int,
    encoder: FeatureEncoder,
    data_root: Path,
    steps_per_episode: int,
) -> bool:
    """Validate that the encoder supports batched environment observations."""

    print("\n" + "=" * 70)
    print("BATCH ENCODING TEST")
    print("=" * 70)

    try:
        data_path = prepare_data_file(symbol, data_root)
    except ValueError as exc:
        print(f"❌ Failed to prepare data for batch test ({symbol}): {exc}")
        return False

    if data_path is None:
        print(f"⚠️  Data file not found for {symbol}, skipping batch test")
        return False

    envs = []
    try:
        for i in range(batch_size):
            config = TradingConfig(
                symbol=symbol,
                data_path=data_path,
                sl_checkpoints={},
                episode_length=steps_per_episode,
                log_level=logging.ERROR,
            )
            env = TradingEnvironment(config, seed=99 + i)
            envs.append(env)
    except Exception as exc:  # pragma: no cover - runtime integration safeguard
        print(f"❌ Failed to create environments for batch test: {exc}")
        for env in envs:
            env.close()
        return False

    observations: List[Dict[str, np.ndarray]] = []
    for env in envs:
        obs, _ = env.reset()
        observations.append(obs)

    batch_obs: Dict[str, np.ndarray] = {}
    for key in observations[0].keys():
        batch_obs[key] = np.stack([obs[key] for obs in observations], axis=0)

    torch_batch = to_torch_from_batch(batch_obs)

    print("\nBatch shapes:")
    for key, tensor in torch_batch.items():
        print(f"  {key}: {tuple(tensor.shape)}")

    with torch.no_grad():
        encoded = encoder(torch_batch)

    print(f"\nEncoded batch shape: {tuple(encoded.shape)}")
    print("Expected: ({}, 256)".format(batch_size))

    assert encoded.shape == (batch_size, 256), f"Batch encoding failed: {tuple(encoded.shape)}"
    assert not torch.isnan(encoded).any(), "NaN detected in batch encoding output"
    assert not torch.isinf(encoded).any(), "Inf detected in batch encoding output"

    for env in envs:
        env.close()

    print("\n✅ Batch encoding test PASSED")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature Encoder Integration Testing")
    parser.add_argument("--data-root", type=str, default="data/historical")
    parser.add_argument("--single-symbol", type=str, default="AAPL")
    parser.add_argument("--multi-symbols", nargs="*", default=["AAPL", "GOOGL", "MSFT"])
    parser.add_argument("--single-episodes", type=int, default=5)
    parser.add_argument("--single-steps", type=int, default=100)
    parser.add_argument("--multi-steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--batch-steps", type=int, default=50)
    args = parser.parse_args()

    print("=" * 70)
    print("FEATURE ENCODER INTEGRATION TESTING")
    print("=" * 70)
    print("\nThis script validates the Feature Encoder with REAL TradingEnvironment data.")
    print("Tests include:")
    print("  1. Single-symbol integration ({})".format(args.single_symbol))
    print("  2. Multi-symbol integration ({})".format(", ".join(args.multi_symbols)))
    print("  3. Batch encoding (size = {})".format(args.batch_size))

    encoder = FeatureEncoder(EncoderConfig())
    encoder.eval()

    data_root = Path(args.data_root)

    single_stats = run_single_symbol_test(
        args.single_symbol,
        num_episodes=args.single_episodes,
        steps_per_episode=args.single_steps,
        encoder=encoder,
        data_root=data_root,
    )

    multi_results = test_multi_symbol_integration(
        args.multi_symbols,
        steps_per_episode=args.multi_steps,
        encoder=encoder,
        data_root=data_root,
    )

    batch_passed = test_batch_encoding(
        args.single_symbol,
        batch_size=args.batch_size,
        encoder=encoder,
        data_root=data_root,
        steps_per_episode=args.batch_steps,
    )

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    all_passed = bool(
        (single_stats and single_stats.passed)
        and all(stats and stats.passed for stats in multi_results.values())
        and batch_passed
    )

    if all_passed:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("\nFeature Encoder successfully validated with:")
        print("  ✅ Real environment observations")
        print("  ✅ Multiple trading symbols")
        print("  ✅ Batch processing")
        print("  ✅ Multi-episode consistency")
        print("\nEncoder is READY for Symbol Agent integration (Task 2.2)!")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("\nReview the detailed output above and resolve issues before proceeding.")


if __name__ == "__main__":
    main()
