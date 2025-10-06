#!/usr/bin/env python3
"""Cache supervised-learning baseline signals for Phase 3 datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PHASE3_SYMBOLS = (
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

SL_MODELS = ("mlp", "lstm", "gru")
DEFAULT_THRESHOLD = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-root", type=Path, default=Path("data/phase3_splits"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--symbols", type=str, nargs="*", default=list(PHASE3_SYMBOLS))
    parser.add_argument("--summary", type=Path, default=Path("analysis/reports/phase3_sl_cache_summary.json"))
    return parser.parse_args()


def load_split(symbol_dir: Path, split_name: str) -> pd.DataFrame:
    file_path = symbol_dir / f"{split_name}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")
    return pd.read_parquet(file_path)


def cache_symbol(symbol: str, root: Path, threshold: float) -> Dict[str, Dict[str, int]]:
    symbol_dir = root / symbol
    if not symbol_dir.exists():
        raise FileNotFoundError(f"Symbol directory not found: {symbol_dir}")

    splits = {name: load_split(symbol_dir, name) for name in ("train", "val", "test")}
    missing_models: List[str] = []
    cached: Dict[str, np.ndarray] = {}
    stats: Dict[str, Dict[str, int]] = {}

    for model in SL_MODELS:
        col = f"sl_prob_{model}"
        if not all(col in df.columns for df in splits.values()):
            missing_models.append(model)
            continue

        stats[model] = {}
        for split_name, df in splits.items():
            probs = df[col].to_numpy(dtype=np.float32)
            signals = (probs > threshold).astype(np.int8)
            cached[f"{model}_{split_name}_probabilities"] = probs
            cached[f"{model}_{split_name}_signals"] = signals
            stats[model][split_name] = int(signals.sum())

    if missing_models:
        raise ValueError(f"Missing SL probability columns for {symbol}: {missing_models}")

    output_path = symbol_dir / "sl_baseline_cache.npz"
    np.savez_compressed(output_path, **cached)
    return stats


def main() -> None:
    args = parse_args()
    args.splits_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SL BASELINE CACHING")
    print("=" * 70)
    print(f"\nCaching SL predictions for {len(args.symbols)} symbols...")
    print(f"Models: {', '.join(SL_MODELS)}")
    print(f"Decision threshold: {args.threshold:.2f}")

    summary: Dict[str, Dict[str, Dict[str, int]]] = {}

    for symbol in args.symbols:
        print(f"\nProcessing {symbol}...", end=" ")
        stats = cache_symbol(symbol, args.splits_root, args.threshold)
        summary[symbol] = stats
        print("✅ cached")

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("SL CACHING COMPLETE")
    print("=" * 70)
    print(f"Summary saved to: {args.summary}")


if __name__ == "__main__":  # pragma: no cover
    main()
