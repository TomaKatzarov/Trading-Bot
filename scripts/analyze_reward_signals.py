"""Reward signal analysis utilities for RL trading experiments.

This script inspects episode-level reward breakdowns emitted by the
`TradingEnvironment`/`RewardShaper` pipeline. It surfaces statistics about
component balance, stability, and their relationship to trading outcomes, and
produces visualizations to aid debugging reward shaping issues.

Typical JSON input structure (per episode):

```
{
  "rewards": [0.12, ...],
  "components": [
    {"pnl": 0.4, "transaction_cost": -0.1, ...},
    ...
  ],
  "equity_curve": [100000.0, 100120.0, ...],  # optional
  "realized_pnl": [0.0, 120.0, ...]           # optional cumulative profits
}
```

The script accepts either a single-episode JSON object or a list of objects.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Reward components traced by RewardShaper
COMPONENT_KEYS: Tuple[str, ...] = (
    "pnl",
    "transaction_cost",
    "time_efficiency",
    "sharpe",
    "drawdown",
    "sizing",
)


@dataclass
class RewardAnalysisResult:
    """Container for reward analysis outputs."""

    component_stats: Dict[str, Dict[str, float]]
    balance_status: Dict[str, str]
    reward_summary: Dict[str, float]
    correlations: Dict[str, float]

    def to_serializable(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict (floats only)."""
        return {
            "component_stats": _convert_nested(self.component_stats),
            "balance_status": self.balance_status,
            "reward_summary": _convert_nested(self.reward_summary),
            "correlations": _convert_nested(self.correlations),
        }


def _convert_nested(data: Any) -> Any:
    """Recursively cast numpy scalars/arrays to Python types."""
    if isinstance(data, dict):
        return {k: _convert_nested(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_convert_nested(v) for v in data]
    if isinstance(data, (np.generic, np.ndarray)):
        return float(np.asarray(data).item())
    return data


def analyze_reward_components(components: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute descriptive statistics for each reward component."""
    if not components:
        return {key: {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "sum": 0.0, "contribution_pct": 0.0} for key in COMPONENT_KEYS}

    comp_arrays: Dict[str, np.ndarray] = {}
    for key in COMPONENT_KEYS:
        comp_arrays[key] = np.asarray([comp.get(key, 0.0) for comp in components], dtype=float)

    total_contribution = sum(abs(arr.sum()) for arr in comp_arrays.values())
    if total_contribution == 0:
        total_contribution = 1.0  # avoid divide-by-zero

    stats: Dict[str, Dict[str, float]] = {}
    for name, values in comp_arrays.items():
        stats[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "sum": float(np.sum(values)),
            "contribution_pct": float(abs(np.sum(values)) / total_contribution * 100.0),
            "signal_to_noise": float(_safe_signal_to_noise(values)),
        }
    return stats


def summarize_rewards(rewards: Sequence[float]) -> Dict[str, float]:
    """Return general statistics for the total reward sequence."""
    rewards_arr = np.asarray(rewards, dtype=float)
    if rewards_arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "sum": 0.0, "signal_to_noise": 0.0}

    return {
        "mean": float(np.mean(rewards_arr)),
        "std": float(np.std(rewards_arr)),
        "min": float(np.min(rewards_arr)),
        "max": float(np.max(rewards_arr)),
        "sum": float(np.sum(rewards_arr)),
        "signal_to_noise": float(_safe_signal_to_noise(rewards_arr)),
    }


def compute_correlations(
    rewards: Sequence[float],
    components: Sequence[Dict[str, float]],
    equity_curve: Optional[Sequence[float]] = None,
    realized_pnl: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Compute correlations between rewards/components and profit metrics."""
    corr_results: Dict[str, float] = {}

    rewards_arr = np.asarray(rewards, dtype=float)
    profits = _derive_profits(equity_curve=equity_curve, realized_pnl=realized_pnl, rewards_len=len(rewards_arr))

    if profits.size == rewards_arr.size and profits.size > 1:
        corr_results["reward_vs_profit"] = float(_safe_corr(rewards_arr, profits))
        for key in COMPONENT_KEYS:
            component_values = np.asarray([comp.get(key, 0.0) for comp in components], dtype=float)
            if component_values.size == profits.size:
                corr_results[f"{key}_vs_profit"] = float(_safe_corr(component_values, profits))
    else:
        corr_results["reward_vs_profit"] = float("nan")

    return corr_results


def check_reward_balance(stats: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """Flag components that dominate or contribute too little."""
    status: Dict[str, str] = {}
    for component, metrics in stats.items():
        pct = metrics.get("contribution_pct", 0.0)
        if pct > 60:
            status[component] = "ERROR: Dominating signal"
        elif pct > 40:
            status[component] = "WARNING: High contribution"
        elif pct < 5:
            status[component] = "WARNING: Low contribution"
        else:
            status[component] = "OK"
    return status


def plot_component_contributions(
    stats: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Render a bar chart of component contribution percentages."""
    components = list(stats.keys())
    contributions = [stats[c]["contribution_pct"] for c in components]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(components, contributions, color="#4C78A8")
    ax.set_ylabel("Contribution (%)")
    ax.set_title("Reward Component Contributions")
    ax.set_ylim(0, 100)
    ax.axhline(40, color="orange", linestyle="--", linewidth=1, label="High contribution threshold")
    ax.axhline(5, color="red", linestyle=":", linewidth=1, label="Low contribution threshold")
    ax.legend()
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info("Saved component contribution plot to %s", save_path)

    if show and not save_path:
        plt.show()

    plt.close(fig)


def _derive_profits(
    *,
    equity_curve: Optional[Sequence[float]],
    realized_pnl: Optional[Sequence[float]],
    rewards_len: int,
) -> np.ndarray:
    """Derive per-step profit series from available signals."""
    if realized_pnl is not None:
        pnl_arr = np.asarray(realized_pnl, dtype=float)
        if pnl_arr.size == rewards_len:
            return pnl_arr
        if pnl_arr.size > rewards_len:
            return pnl_arr[:rewards_len]

    if equity_curve is not None:
        eq_arr = np.asarray(equity_curve, dtype=float)
        if eq_arr.size >= rewards_len:
            # Use per-step deltas aligned to rewards length
            deltas = np.diff(eq_arr, prepend=eq_arr[0])
            return deltas[:rewards_len]

    return np.asarray([], dtype=float)


def _safe_signal_to_noise(values: np.ndarray) -> float:
    std = np.std(values)
    if std < 1e-9:
        return 0.0
    return float(np.mean(values) / std)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ensure_backend(non_interactive: bool) -> None:
    """Switch matplotlib backend to Agg when running headless."""
    if non_interactive:
        import matplotlib

        matplotlib.use("Agg")


def load_episode_payload(path: Path) -> List[Dict[str, Any]]:
    """Load JSON file that may contain a single episode or list of episodes."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "episodes" in payload:
        episodes = payload["episodes"]
    elif isinstance(payload, dict):
        episodes = [payload]
    elif isinstance(payload, list):
        episodes = payload
    else:
        raise ValueError("Unsupported episode data format")

    if not episodes:
        raise ValueError("No episode entries found in data")

    return episodes


def analyze_episode(episode_entry: Dict[str, Any]) -> RewardAnalysisResult:
    """Perform full analysis on a single episode entry."""
    rewards = episode_entry.get("rewards", [])
    components = episode_entry.get("components", [])
    equity_curve = episode_entry.get("equity_curve")
    realized_pnl = episode_entry.get("realized_pnl")

    component_stats = analyze_reward_components(components)
    reward_summary = summarize_rewards(rewards)
    correlations = compute_correlations(
        rewards=rewards,
        components=components,
        equity_curve=equity_curve,
        realized_pnl=realized_pnl,
    )
    balance_status = check_reward_balance(component_stats)

    return RewardAnalysisResult(
        component_stats=component_stats,
        balance_status=balance_status,
        reward_summary=reward_summary,
        correlations=correlations,
    )


def render_report(index: int, result: RewardAnalysisResult) -> None:
    """Prints a textual summary for the episode analysis."""
    print("=" * 70)
    print(f"EPISODE {index + 1} REWARD ANALYSIS")
    print("=" * 70)

    print("\nComponent Statistics:")
    for component, metrics in result.component_stats.items():
        print(f"\n{component}:")
        print(f"  Mean:            {metrics['mean']:.4f}")
        print(f"  Std Dev:         {metrics['std']:.4f}")
        print(f"  Sum:             {metrics['sum']:.4f}")
        print(f"  Contribution %:  {metrics['contribution_pct']:.2f}")
        print(f"  SNR:             {metrics['signal_to_noise']:.3f}")

    print("\nReward Summary:")
    for key, value in result.reward_summary.items():
        print(f"  {key.capitalize():<14}{value:.4f}")

    if result.correlations:
        print("\nReward/Profit Correlations:")
        for name, value in result.correlations.items():
            print(f"  {name:<20}{value:.4f}")

    print("\nBalance Check:")
    for component, status in result.balance_status.items():
        print(f"  {component:<18}{status}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze reward signals from RL episodes")
    parser.add_argument("--episode-data", type=str, required=True, help="Path to episode data JSON")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/rewards",
        help="Directory for generated plots and stats",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display plots interactively instead of saving",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity",
    )

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    output_dir = Path(args.output_dir)
    _ensure_backend(non_interactive=not args.show)

    episodes = load_episode_payload(Path(args.episode_data))

    all_results: List[RewardAnalysisResult] = []
    for idx, episode_entry in enumerate(episodes):
        result = analyze_episode(episode_entry)
        render_report(idx, result)
        all_results.append(result)

        plot_path = output_dir / f"episode_{idx + 1}_component_contributions.png"
        plot_component_contributions(result.component_stats, save_path=plot_path if not args.show else None, show=args.show)

    # Persist aggregated stats
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_payload = [res.to_serializable() for res in all_results]
    stats_path = output_dir / "reward_analysis.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats_payload, handle, indent=2)
    logger.info("Saved analysis summary to %s", stats_path)


if __name__ == "__main__":
    main()
