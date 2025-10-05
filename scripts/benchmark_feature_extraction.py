"""Benchmark feature extraction performance for the RL trading environment.

This utility measures the latency of the feature engineering pipeline under
various normalization schemes and reports cache efficiency statistics. It also
benchmarks the accompanying market regime indicator helper to ensure the entire
observation pipeline meets the real-time constraints required by the
multi-agent RL system.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rl.environments.feature_extractor import FeatureConfig, FeatureExtractor
from core.rl.environments.regime_indicators import RegimeIndicators

DATA_ROOT = Path("data")
DEFAULT_SYMBOL = "AAPL"
DEFAULT_TIMEFRAME = "1Hour"

_COLUMN_SYNONYMS: Dict[str, str] = {
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


@dataclass
class BenchmarkResult:
    """Container for feature extraction benchmark statistics."""

    init_time_s: float
    mean_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    cache_hit_rate: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "init_time_s": self.init_time_s,
            "mean_time_ms": self.mean_time_ms,
            "median_time_ms": self.median_time_ms,
            "p95_time_ms": self.p95_time_ms,
            "p99_time_ms": self.p99_time_ms,
            "cache_hit_rate": self.cache_hit_rate,
        }


def _resolve_symbol_path(symbol: str, timeframe: str) -> Optional[Path]:
    """Return the most likely parquet path for the requested symbol."""

    candidates: List[Path] = [
        DATA_ROOT / "historical_data_parquet" / f"{symbol}.parquet",
        DATA_ROOT / "prepared_training" / f"{symbol}.parquet",
        DATA_ROOT / "training_data_v2_final" / f"{symbol}.parquet",
    ]

    legacy_dir = DATA_ROOT / "historical" / symbol
    if legacy_dir.exists():
        candidates.append(legacy_dir / timeframe / "data.parquet")
        candidates.extend(sorted(legacy_dir.glob("**/*.parquet")))

    for path in candidates:
        if path.exists():
            return path
    return None


def _load_symbol_data(symbol: str, timeframe: str) -> pd.DataFrame:
    path = _resolve_symbol_path(symbol, timeframe)
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Unable to locate parquet data for symbol '{symbol}'. Checked under {DATA_ROOT}"
        )

    data = pd.read_parquet(path)
    if _COLUMN_SYNONYMS:
        data = data.rename(columns=_COLUMN_SYNONYMS)
    if not isinstance(data.index, pd.DatetimeIndex):
        if "timestamp" in data.columns:
            data = data.copy()
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data.set_index("timestamp", inplace=True)
        else:
            data.index = pd.to_datetime(data.index)

    print(f"Loaded {len(data):,} timesteps from {path}")
    return data


def _generate_indices(extractor: FeatureExtractor, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    start, end = extractor.get_valid_range()
    if end - start <= 0:
        raise RuntimeError(
            "FeatureExtractor reports no valid indices. Ensure lookback window and dataset length are compatible."
        )

    if n_samples > (end - start) * 5:
        # Clamp to avoid excessive duplicates when the dataset is small.
        n_samples = (end - start) * 5

    return rng.integers(start, end, size=n_samples)


def _measure_pass(
    extractor: FeatureExtractor,
    indices: Iterable[int],
    normalize: bool = True,
) -> List[float]:
    times: List[float] = []
    for idx in indices:
        start = time.perf_counter()
        extractor.extract_window(int(idx), normalize=normalize)
        times.append(time.perf_counter() - start)
    return times


def _compute_delta_stats(before: Dict[str, float], after: Dict[str, float]) -> Tuple[int, int, float]:
    delta_hits = int(after.get("hits", 0) - before.get("hits", 0))
    delta_misses = int(after.get("misses", 0) - before.get("misses", 0))
    total = max(delta_hits + delta_misses, 1)
    hit_rate = delta_hits / total * 100.0
    return delta_hits, delta_misses, hit_rate


def benchmark_feature_extraction(
    symbol: str = DEFAULT_SYMBOL,
    lookback_window: int = 24,
    n_samples: int = 1000,
    seed: int = 7,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> Dict[str, BenchmarkResult]:
    """Benchmark the feature extraction pipeline for a single symbol."""

    rng = np.random.default_rng(seed)
    data = _load_symbol_data(symbol, timeframe)

    configs = {
        "no_norm": FeatureConfig(normalize_method="none"),
        "zscore": FeatureConfig(normalize_method="zscore"),
        "minmax": FeatureConfig(normalize_method="minmax"),
        "robust": FeatureConfig(normalize_method="robust"),
    }

    results: Dict[str, BenchmarkResult] = {}

    for name, config in configs.items():
        print(f"\nBenchmarking configuration: {name}")

        start_init = time.time()
        extractor = FeatureExtractor(data, config, lookback_window=lookback_window)
        init_time = time.time() - start_init

        indices = _generate_indices(extractor, n_samples=n_samples, rng=rng)

        # Warm the cache to simulate sequential training access patterns.
        extractor.prefetch(indices)
        stats_before = extractor.get_cache_stats()

        # Measure two passes to ensure cached performance dominates.
        times_pass1 = _measure_pass(extractor, indices)
        times_pass2 = _measure_pass(extractor, indices)

        stats_after = extractor.get_cache_stats()
        _, _, hit_rate = _compute_delta_stats(stats_before, stats_after)

        combined = times_pass1 + times_pass2

        result = BenchmarkResult(
            init_time_s=init_time,
            mean_time_ms=float(np.mean(combined) * 1000.0),
            median_time_ms=float(np.median(combined) * 1000.0),
            p95_time_ms=float(np.percentile(combined, 95) * 1000.0),
            p99_time_ms=float(np.percentile(combined, 99) * 1000.0),
            cache_hit_rate=hit_rate,
        )
        results[name] = result

        print(f"  Init time:  {result.init_time_s:.3f}s")
        print(f"  Mean:       {result.mean_time_ms:.4f} ms")
        print(f"  Median:     {result.median_time_ms:.4f} ms")
        print(f"  P95:        {result.p95_time_ms:.4f} ms")
        print(f"  P99:        {result.p99_time_ms:.4f} ms")
        print(f"  Cache hit:  {result.cache_hit_rate:.1f}%")

    print("\nBenchmarking regime indicator extraction…")
    start_regime = time.time()
    regime = RegimeIndicators(data)
    regime_init_time = time.time() - start_regime

    regime_times: List[float] = []
    min_index = max(252, lookback_window)
    for _ in range(min(len(data) - min_index, 1000)):
        idx = rng.integers(min_index, len(data))
        start = time.perf_counter()
        regime.get_regime_vector(int(idx))
        regime_times.append(time.perf_counter() - start)

    print(f"  Init time:  {regime_init_time:.3f}s")
    if regime_times:
        print(f"  Mean:       {np.mean(regime_times) * 1000:.4f} ms")
        print(f"  P95:        {np.percentile(regime_times, 95) * 1000:.4f} ms")
    else:
        print("  Insufficient data to benchmark regime vectors.")

    fastest = min(results.items(), key=lambda item: item[1].p95_time_ms)
    meets_latency = fastest[1].p95_time_ms < 10.0
    meets_cache = all(result.cache_hit_rate >= 80.0 for result in results.values())

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        print(
            f"- {name:<8} | mean {result.mean_time_ms:6.3f} ms | "
            f"P95 {result.p95_time_ms:6.3f} ms | cache {result.cache_hit_rate:5.1f}%"
        )

    print(f"\nFastest configuration: {fastest[0]} (P95 {fastest[1].p95_time_ms:.3f} ms)")
    print(f"Latency target <10 ms (P95): {'✅ PASS' if meets_latency else '❌ FAIL'}")
    print(f"Cache hit target ≥80%:       {'✅ PASS' if meets_cache else '❌ FAIL'}")

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL, help="Symbol to benchmark (default: AAPL)")
    parser.add_argument("--lookback", type=int, default=24, help="Lookback window size for feature extraction")
    parser.add_argument("--samples", type=int, default=1000, help="Number of random samples to benchmark")
    parser.add_argument("--seed", type=int, default=7, help="Seed for RNG when sampling indices")
    parser.add_argument(
        "--timeframe",
        type=str,
        default=DEFAULT_TIMEFRAME,
        help="Timeframe folder when using legacy directory structure (default: 1Hour)",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        benchmark_feature_extraction(
            symbol=args.symbol,
            lookback_window=args.lookback,
            n_samples=args.samples,
            seed=args.seed,
            timeframe=args.timeframe,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except Exception as exc:  # pragma: no cover - benchmarking convenience
        print(f"❌ Benchmark failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
