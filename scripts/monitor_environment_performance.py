"""Environment Performance Monitoring Dashboard.

This script analyzes single-episode rollouts collected from the trading
environment and produces:

- Action distribution statistics (counts, percentages, entropy/diversity)
- Reward quality diagnostics (mean/std ratios, positive/negative balance)
- Portfolio performance metrics (returns, drawdowns, Sharpe/Sortino, trade stats)
- Position lifecycle insights (holding periods, exit reasons, P&L distributions)
- Risk control effectiveness summaries (stop-loss hits, emergency liquidations)
- Visualization dashboard saved as PNG files (equity curve, actions, rewards, P&L)
- Optional CSV/JSON exports for downstream aggregations
- Human-readable text report for quick review by researchers

Example usage::

    python scripts/monitor_environment_performance.py \
        --episode-data analysis/episodes/sample_episode.json \
        --output-dir analysis/environment/episode_001

Episode JSON Schema (flexible):
    {
        "actions": [0, 1, 0, ...],
        "rewards": [0.02, -0.01, ...],
        "equity_curve": [100000, 100050, ...],
        "cash_curve": [100000, 95000, ...],             # optional
        "exposure_curve": [0.0, 0.25, ...],             # optional (0-1)
        "drawdown_curve": [0.0, 0.01, ...],             # optional (0-1)
        "trades": [
            {
                "entry_step": 42,
                "exit_step": 84,
                "holding_period": 42,
                "realized_pnl": 250.0,
                "realized_pnl_pct": 0.025,
                "exit_reason": "take_profit",
                "symbol": "BTC-USD"
            },
            ...
        ],
        "risk_events": [
            {
                "step": 120,
                "type": "portfolio_drawdown",
                "severity": "critical",
                "value": 0.21,
                "limit": 0.2,
                "action_required": "close_all_positions"
            }
        ],
        "info": {
            "episode_length": 512,
            "symbol": "BTC-USD",
            "config": {...}
        }
    }

The script is resilient to missing keys and will emit warnings if certain
sections cannot be computed.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from textwrap import indent
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.figsize": (14, 8),
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

LOGGER = logging.getLogger("env_monitor")
ACTION_LABELS = [
    "HOLD",
    "BUY_SMALL",
    "BUY_MED",
    "BUY_LARGE",
    "SELL_PARTIAL",
    "SELL_ALL",
    "ADD_POSITION",
]

# ---------------------------------------------------------------------------
# Utility dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AnalysisArtifact:
    """Container for storing derived statistics."""

    metrics: Dict[str, Any]
    warnings: List[str]

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
        LOGGER.warning(message)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_episode_data(episode_path: Path) -> Dict[str, Any]:
    """Load episode data from JSON file and return a dictionary.

    Parameters
    ----------
    episode_path: Path
        Location of the JSON file created after an environment rollout.
    """

    if not episode_path.exists():
        raise FileNotFoundError(f"Episode file not found: {episode_path}")

    with episode_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, Mapping):
        raise ValueError("Episode JSON must contain an object at the top level")

    return dict(data)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _safe_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return array
    return array[np.isfinite(array)]


def analyze_action_distribution(episode_data: Mapping[str, Any]) -> AnalysisArtifact:
    actions = episode_data.get("actions", [])
    artifact = AnalysisArtifact(metrics={}, warnings=[])

    if not actions:
        artifact.add_warning("No actions found in episode data; skipping distribution analysis")
        return artifact

    total_actions = len(actions)
    counts: Dict[int, int] = {}
    for action in actions:
        counts[action] = counts.get(action, 0) + 1

    percentages = {idx: cnt / total_actions * 100 for idx, cnt in counts.items()}
    probs = np.array(list(percentages.values()), dtype=float) / 100
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    max_entropy = float(np.log(len(ACTION_LABELS)))
    diversity = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    artifact.metrics.update(
        {
            "action_counts": counts,
            "action_percentages": percentages,
            "diversity_score": diversity,
            "entropy": entropy,
            "total_actions": total_actions,
            "unique_actions": len(counts),
        }
    )
    return artifact


def analyze_reward_quality(episode_data: Mapping[str, Any]) -> AnalysisArtifact:
    rewards = episode_data.get("rewards", [])
    artifact = AnalysisArtifact(metrics={}, warnings=[])

    if not rewards:
        artifact.add_warning("No rewards found in episode data; reward diagnostics skipped")
        return artifact

    reward_array = _safe_array(rewards)
    if reward_array.size == 0:
        artifact.add_warning("Rewards contained no finite values; reward diagnostics skipped")
        return artifact

    positive = reward_array[reward_array > 0]
    negative = reward_array[reward_array < 0]
    zero = reward_array[reward_array == 0]

    pos_ratio = float(len(positive) / reward_array.size)
    neg_ratio = float(len(negative) / reward_array.size)
    zero_ratio = float(len(zero) / reward_array.size)

    cumulative = reward_array.cumsum()
    rolling_mean = pd.Series(reward_array).rolling(window=20, min_periods=1).mean().tolist()

    artifact.metrics.update(
        {
            "mean": float(np.mean(reward_array)),
            "median": float(np.median(reward_array)),
            "std": float(np.std(reward_array)),
            "max": float(np.max(reward_array)),
            "min": float(np.min(reward_array)),
            "positive_ratio": pos_ratio,
            "negative_ratio": neg_ratio,
            "zero_ratio": zero_ratio,
            "cumulative_rewards": cumulative.tolist(),
            "rolling_mean": rolling_mean,
        }
    )
    return artifact


def analyze_portfolio_performance(episode_data: Mapping[str, Any]) -> AnalysisArtifact:
    artifact = AnalysisArtifact(metrics={}, warnings=[])

    equity_curve = episode_data.get("equity_curve", [])
    if not equity_curve:
        artifact.add_warning("No equity curve found; cannot compute portfolio metrics")
        return artifact

    equity = _safe_array(equity_curve)
    if equity.size < 2:
        artifact.add_warning("Equity curve must contain at least two points")
        return artifact

    initial_capital = float(equity[0])
    final_equity = float(equity[-1])
    total_return = float((final_equity - initial_capital) / initial_capital) if initial_capital != 0 else 0.0

    returns = np.diff(equity) / equity[:-1]
    sharpe = 0.0
    if returns.size > 1 and np.std(returns) > 1e-12:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

    downside = returns[returns < 0]
    sortino = 0.0
    if downside.size > 0 and np.std(downside) > 1e-12:
        sortino = float(np.mean(returns) / np.std(downside) * np.sqrt(252))

    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.maximum(peak, 1e-12)
    max_drawdown = float(np.max(drawdown))

    trades = episode_data.get("trades", [])
    winning = [t for t in trades if t.get("realized_pnl", 0.0) > 0]
    losing = [t for t in trades if t.get("realized_pnl", 0.0) < 0]

    avg_win = float(np.mean([t["realized_pnl"] for t in winning])) if winning else 0.0
    avg_loss = float(np.mean([t["realized_pnl"] for t in losing])) if losing else 0.0
    profit_factor = float(abs(avg_win / avg_loss)) if avg_loss != 0 else float("inf") if avg_win > 0 else 0.0

    artifact.metrics.update(
        {
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "num_trades": len(trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": float(len(winning) / len(trades)) if trades else 0.0,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "equity_curve": equity.tolist(),
            "returns": returns.tolist(),
            "drawdown_curve": drawdown.tolist(),
        }
    )
    return artifact


def analyze_position_lifecycle(episode_data: Mapping[str, Any]) -> AnalysisArtifact:
    artifact = AnalysisArtifact(metrics={}, warnings=[])
    trades = episode_data.get("trades", [])
    if not trades:
        artifact.add_warning("No trades found; position lifecycle analysis skipped")
        return artifact

    holding_periods = _safe_array([t.get("holding_period", 0) for t in trades])
    pnl_pcts = _safe_array([t.get("realized_pnl_pct", 0) * 100 for t in trades])
    exit_reasons: Dict[str, int] = {}
    for trade in trades:
        reason = trade.get("exit_reason", "unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    artifact.metrics.update(
        {
            "avg_holding_period": float(np.mean(holding_periods)) if holding_periods.size else 0.0,
            "median_holding_period": float(np.median(holding_periods)) if holding_periods.size else 0.0,
            "min_holding_period": float(np.min(holding_periods)) if holding_periods.size else 0.0,
            "max_holding_period": float(np.max(holding_periods)) if holding_periods.size else 0.0,
            "avg_pnl_pct": float(np.mean(pnl_pcts)) if pnl_pcts.size else 0.0,
            "median_pnl_pct": float(np.median(pnl_pcts)) if pnl_pcts.size else 0.0,
            "exit_reasons": exit_reasons,
            "pnl_distribution": pnl_pcts.tolist(),
            "holding_periods": holding_periods.tolist(),
        }
    )
    return artifact


def analyze_risk_controls(episode_data: Mapping[str, Any]) -> AnalysisArtifact:
    artifact = AnalysisArtifact(metrics={}, warnings=[])
    risk_events = episode_data.get("risk_events", [])
    if not risk_events:
        artifact.add_warning("No risk events found; assuming no violations occurred")
        return artifact

    severity_counts: Dict[str, int] = {}
    type_counts: Dict[str, int] = {}
    for event in risk_events:
        severity = event.get("severity", "unknown")
        event_type = event.get("type", "unknown")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        type_counts[event_type] = type_counts.get(event_type, 0) + 1

    artifact.metrics.update(
        {
            "risk_event_count": len(risk_events),
            "severity_counts": severity_counts,
            "type_counts": type_counts,
            "events": risk_events,
        }
    )
    return artifact


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_dashboard(
    episode_data: Mapping[str, Any],
    action_stats: Mapping[str, Any],
    reward_stats: Mapping[str, Any],
    portfolio_stats: Mapping[str, Any],
    position_stats: Mapping[str, Any],
    output_dir: Path,
) -> None:
    _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    # Equity
    equity_curve = portfolio_stats.get("equity_curve") or episode_data.get("equity_curve", [])
    if equity_curve:
        axes[0, 0].plot(equity_curve, linewidth=2.0, color="#1f77b4")
        axes[0, 0].set_title("Equity Curve")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Equity ($)")
        axes[0, 0].axhline(equity_curve[0], linestyle="--", color="red", alpha=0.4, label="Initial Capital")
        axes[0, 0].legend()

    drawdown_curve = portfolio_stats.get("drawdown_curve") or episode_data.get("drawdown_curve", [])
    if drawdown_curve:
        dd_pct = np.array(drawdown_curve) * 100
        axes[1, 0].plot(dd_pct, linewidth=1.8, color="#d62728")
        axes[1, 0].fill_between(range(len(dd_pct)), dd_pct, color="#d62728", alpha=0.25)
        axes[1, 0].set_title("Drawdown (%)")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Drawdown (%)")

    # Actions bar
    actions = episode_data.get("actions", [])
    if actions:
        counts = [action_stats.get("action_counts", {}).get(i, 0) for i in range(len(ACTION_LABELS))]
        axes[0, 1].bar(ACTION_LABELS, counts, color="#2ca02c")
        axes[0, 1].set_title("Action Counts")
        axes[0, 1].tick_params(axis="x", rotation=35)

    # Reward histogram
    rewards = episode_data.get("rewards", [])
    if rewards:
        axes[1, 1].hist(rewards, bins=40, alpha=0.8, color="#9467bd", edgecolor="black")
        axes[1, 1].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[1, 1].set_title("Reward Distribution")
        axes[1, 1].set_xlabel("Reward")

    # Trade PnL histogram
    trades = episode_data.get("trades", [])
    if trades:
        pnl_pct = [t.get("realized_pnl_pct", 0) * 100 for t in trades]
        axes[0, 2].hist(pnl_pct, bins=30, alpha=0.8, color="#ff7f0e", edgecolor="black")
        axes[0, 2].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[0, 2].set_title("Trade P&L (%)")
        axes[0, 2].set_xlabel("Realized P&L (%)")

    # Exposure curve if available
    exposure_curve = episode_data.get("exposure_curve", [])
    if exposure_curve:
        axes[1, 2].plot([val * 100 for val in exposure_curve], color="#17becf", linewidth=1.8)
        axes[1, 2].set_title("Portfolio Exposure (%)")
        axes[1, 2].set_xlabel("Step")
        axes[1, 2].set_ylabel("Exposure (%)")

    fig.suptitle("Environment Monitoring Dashboard", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    dashboard_path = output_dir / "dashboard.png"
    fig.savefig(dashboard_path, dpi=150)
    LOGGER.info("Dashboard saved to %s", dashboard_path)
    plt.close(fig)

    # Rolling reward plot
    if reward_stats.get("rolling_mean"):
        fig2, ax2 = plt.subplots(figsize=(16, 6))
        ax2.plot(reward_stats["rolling_mean"], color="#1f77b4", linewidth=2.0, label="Rolling Mean (window=20)")
        ax2.plot(reward_stats["cumulative_rewards"], color="#ff7f0e", alpha=0.6, label="Cumulative Reward")
        ax2.set_title("Reward Diagnostics")
        ax2.set_xlabel("Step")
        ax2.legend()
        reward_path = output_dir / "rewards.png"
        fig2.tight_layout()
        fig2.savefig(reward_path, dpi=150)
        LOGGER.info("Reward diagnostics saved to %s", reward_path)
        plt.close(fig2)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _format_percentage(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def generate_report(
    action_stats: Mapping[str, Any],
    reward_stats: Mapping[str, Any],
    portfolio_stats: Mapping[str, Any],
    position_stats: Mapping[str, Any],
    risk_stats: Mapping[str, Any],
    output_dir: Path,
) -> None:
    _ensure_output_dir(output_dir)
    report_path = output_dir / "performance_report.txt"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("=" * 100 + "\n")
        handle.write("ENVIRONMENT PERFORMANCE REPORT\n")
        handle.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        handle.write("=" * 100 + "\n\n")

        handle.write("ACTION STATISTICS\n")
        handle.write("-" * 100 + "\n")
        handle.write(f"Total Actions: {action_stats.get('total_actions', 0)}\n")
        handle.write(f"Unique Actions: {action_stats.get('unique_actions', 0)}\n")
        handle.write(f"Diversity Score: {action_stats.get('diversity_score', 0):.3f}\n")
        handle.write("Action Distribution (percent):\n")
        for idx, pct in sorted(action_stats.get("action_percentages", {}).items()):
            label = ACTION_LABELS[idx] if idx < len(ACTION_LABELS) else f"Action {idx}"
            handle.write(f"  {label:<15} {pct:6.2f}%\n")
        handle.write("\n")

        handle.write("REWARD SIGNAL QUALITY\n")
        handle.write("-" * 100 + "\n")
        if reward_stats:
            handle.write(f"Mean Reward: {reward_stats.get('mean', 0):.6f}\n")
            handle.write(f"Std Dev: {reward_stats.get('std', 0):.6f}\n")
            handle.write(f"Median Reward: {reward_stats.get('median', 0):.6f}\n")
            handle.write(f"Max Reward: {reward_stats.get('max', 0):.6f}\n")
            handle.write(f"Min Reward: {reward_stats.get('min', 0):.6f}\n")
            handle.write(f"Positive Ratio: {_format_percentage(reward_stats.get('positive_ratio', 0))}\n")
            handle.write(f"Negative Ratio: {_format_percentage(reward_stats.get('negative_ratio', 0))}\n")
            handle.write(f"Zero Ratio: {_format_percentage(reward_stats.get('zero_ratio', 0))}\n")
        else:
            handle.write("Reward statistics unavailable.\n")
        handle.write("\n")

        handle.write("PORTFOLIO PERFORMANCE\n")
        handle.write("-" * 100 + "\n")
        if portfolio_stats:
            handle.write(f"Initial Capital: ${portfolio_stats.get('initial_capital', 0):,.2f}\n")
            handle.write(f"Final Equity: ${portfolio_stats.get('final_equity', 0):,.2f}\n")
            handle.write(f"Total Return: {portfolio_stats.get('total_return_pct', 0):.2f}%\n")
            handle.write(f"Sharpe Ratio: {portfolio_stats.get('sharpe_ratio', 0):.3f}\n")
            handle.write(f"Sortino Ratio: {portfolio_stats.get('sortino_ratio', 0):.3f}\n")
            handle.write(f"Max Drawdown: {portfolio_stats.get('max_drawdown_pct', 0):.2f}%\n")
            handle.write(f"Total Trades: {portfolio_stats.get('num_trades', 0)}\n")
            handle.write(f"Winning Trades: {portfolio_stats.get('winning_trades', 0)}\n")
            handle.write(f"Losing Trades: {portfolio_stats.get('losing_trades', 0)}\n")
            handle.write(f"Win Rate: {_format_percentage(portfolio_stats.get('win_rate', 0))}\n")
            handle.write(f"Average Win: ${portfolio_stats.get('avg_win', 0):,.2f}\n")
            handle.write(f"Average Loss: ${portfolio_stats.get('avg_loss', 0):,.2f}\n")
            pf = portfolio_stats.get("profit_factor")
            pf_display = "∞" if pf == float("inf") else f"{pf:.2f}"
            handle.write(f"Profit Factor: {pf_display}\n")
        else:
            handle.write("Portfolio statistics unavailable.\n")
        handle.write("\n")

        handle.write("POSITION LIFECYCLE\n")
        handle.write("-" * 100 + "\n")
        if position_stats and not position_stats.get("warnings"):
            handle.write(f"Average Holding Period: {position_stats.get('avg_holding_period', 0):.2f} steps\n")
            handle.write(f"Median Holding Period: {position_stats.get('median_holding_period', 0):.2f} steps\n")
            handle.write(f"Average Realized P&L: {position_stats.get('avg_pnl_pct', 0):.2f}%\n")
            handle.write("Exit Reasons:\n")
            for reason, count in position_stats.get("exit_reasons", {}).items():
                handle.write(f"  {reason:<20} {count}\n")
        else:
            handle.write("Position statistics unavailable.\n")
        handle.write("\n")

        handle.write("RISK EVENTS\n")
        handle.write("-" * 100 + "\n")
        if risk_stats and risk_stats.get("risk_event_count", 0) > 0:
            handle.write(f"Total Risk Events: {risk_stats.get('risk_event_count')}\n")
            handle.write("By Severity:\n")
            for severity, count in risk_stats.get("severity_counts", {}).items():
                handle.write(f"  {severity:<12} {count}\n")
            handle.write("By Type:\n")
            for event_type, count in risk_stats.get("type_counts", {}).items():
                handle.write(f"  {event_type:<20} {count}\n")
        else:
            handle.write("No risk events recorded.\n")
        handle.write("\n")

        handle.write("WARNINGS\n")
        handle.write("-" * 100 + "\n")
        warnings = []
        for stats in (action_stats, reward_stats, portfolio_stats, position_stats, risk_stats):
            if isinstance(stats, AnalysisArtifact):
                warnings.extend(stats.warnings)
        if warnings:
            for warn in warnings:
                handle.write(f"  - {warn}\n")
        else:
            handle.write("  None\n")

    LOGGER.info("Report saved to %s", report_path)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def export_statistics_to_json(
    action_stats: Mapping[str, Any],
    reward_stats: Mapping[str, Any],
    portfolio_stats: Mapping[str, Any],
    position_stats: Mapping[str, Any],
    risk_stats: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    _ensure_output_dir(output_dir)
    stats_path = output_dir / "statistics.json"
    payload = {
        "action_stats": getattr(action_stats, "metrics", action_stats),
        "reward_stats": getattr(reward_stats, "metrics", reward_stats),
        "portfolio_stats": getattr(portfolio_stats, "metrics", portfolio_stats),
        "position_stats": getattr(position_stats, "metrics", position_stats),
        "risk_stats": getattr(risk_stats, "metrics", risk_stats),
    }
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    LOGGER.info("Statistics saved to %s", stats_path)
    return stats_path


def export_timelines_to_csv(episode_data: Mapping[str, Any], output_dir: Path) -> None:
    _ensure_output_dir(output_dir)
    timeline_keys = [
        "equity_curve",
        "cash_curve",
        "exposure_curve",
        "drawdown_curve",
        "rewards",
    ]
    timeline_data: Dict[str, List[float]] = {}
    length = 0
    for key in timeline_keys:
        series = episode_data.get(key, [])
        if series:
            length = max(length, len(series))
            timeline_data[key] = list(series)
    if not timeline_data:
        LOGGER.warning("No timeline curves found; skipping CSV export")
        return

    # Normalize lengths with NaNs
    for key, values in timeline_data.items():
        if len(values) < length:
            values.extend([np.nan] * (length - len(values)))
    frame = pd.DataFrame(timeline_data)
    csv_path = output_dir / "timelines.csv"
    frame.index.name = "step"
    frame.to_csv(csv_path)
    LOGGER.info("Timelines exported to %s", csv_path)


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor environment performance from episode JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episode-data", type=str, required=True, help="Path to episode JSON file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/environment",
        help="Directory where reports and plots will be saved",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation (useful for headless CI environments)",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export time-series curves (equity, exposure, rewards) to CSV",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    episode_path = Path(args.episode_data)
    output_dir = Path(args.output_dir)

    LOGGER.info("Loading episode data from %s", episode_path)
    episode_data = load_episode_data(episode_path)

    LOGGER.info("Running analyses...")
    action_stats = analyze_action_distribution(episode_data)
    reward_stats = analyze_reward_quality(episode_data)
    portfolio_stats = analyze_portfolio_performance(episode_data)
    position_stats = analyze_position_lifecycle(episode_data)
    risk_stats = analyze_risk_controls(episode_data)

    # Convert to raw metrics for downstream functions
    action_metrics = getattr(action_stats, "metrics", action_stats)
    reward_metrics = getattr(reward_stats, "metrics", reward_stats)
    portfolio_metrics = getattr(portfolio_stats, "metrics", portfolio_stats)
    position_metrics = getattr(position_stats, "metrics", position_stats)
    risk_metrics = getattr(risk_stats, "metrics", risk_stats)

    LOGGER.info("Generating textual report...")
    generate_report(action_metrics, reward_metrics, portfolio_metrics, position_metrics, risk_metrics, output_dir)

    LOGGER.info("Exporting statistics to JSON...")
    export_statistics_to_json(action_stats, reward_stats, portfolio_stats, position_stats, risk_stats, output_dir)

    if args.export_csv:
        LOGGER.info("Exporting timeline curves to CSV...")
        export_timelines_to_csv(episode_data, output_dir)

    if not args.no_plots:
        LOGGER.info("Generating visualization dashboard...")
        plot_dashboard(episode_data, action_metrics, reward_metrics, portfolio_metrics, position_metrics, output_dir)
    else:
        LOGGER.info("Plot generation disabled via --no-plots")

    LOGGER.info("Analysis complete; artifacts saved to %s", output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
