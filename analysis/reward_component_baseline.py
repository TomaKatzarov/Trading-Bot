"""Quick reward-component sanity check.

This utility runs a handful of random-policy episodes against the configured
Phase 3 trading environment so we can inspect the raw reward component ranges
before (or after) adjusting scaling parameters. The output highlights mean and
standard deviation for each component along with aggregated portfolio metrics,
which helps ensure the weights in the YAML translate into the magnitudes we
expect.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rl.environments.vec_trading_env import make_vec_trading_env
from training.train_phase3_agents import load_config, make_env_kwargs


@dataclass
class ComponentStats:
    values: List[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.values.append(float(value))

    def summary(self) -> Dict[str, float]:
        array = np.array(self.values, dtype=np.float64)
        if array.size == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }


def collect_reward_statistics(
    *,
    config_path: Path,
    symbol: str,
    split: str,
    episodes: int,
    seed: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    config = load_config(config_path)
    env_kwargs = make_env_kwargs(symbol, split, config)

    data_dir = Path(config["experiment"]["data_dir"]) / symbol
    vec_env = make_vec_trading_env(
        symbol=symbol,
        data_dir=data_dir,
        num_envs=1,
        seed=seed,
        use_subprocess=False,
        env_kwargs=env_kwargs,
        env_log_level=env_kwargs.get("log_level"),
    )

    try:
        component_keys = config["environment"].get("reward_weights", {}).keys()
        if not component_keys:
            component_keys = [
                "pnl",
                "cost",
                "time",
                "sharpe",
                "drawdown",
                "sizing",
                "hold",
            ]
        components: Dict[str, ComponentStats] = {
            key: ComponentStats() for key in component_keys
        }
        portfolio_metrics: Dict[str, ComponentStats] = {}

        for episode_idx in range(episodes):
            obs = vec_env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            while not done:
                sampled_action = vec_env.action_space.sample()
                if isinstance(sampled_action, np.ndarray):
                    action = sampled_action
                else:
                    action = np.array([sampled_action])
                obs, reward, dones, infos = vec_env.step(action)
                total_reward += float(reward[0])
                steps += 1
                done = bool(dones[0])
                if done:
                    info = infos[0]
                    reward_stats = info.get("terminal_reward_stats") or {}
                    for key, stats in reward_stats.items():
                        if not key.endswith("_mean"):
                            continue
                        base_key = key.replace("_mean", "")
                        if base_key in components:
                            components[base_key].add(stats)
                    metrics = info.get("terminal_metrics") or {}
                    for metric_key, value in metrics.items():
                        bucket = portfolio_metrics.setdefault(metric_key, ComponentStats())
                        bucket.add(value)
            # Optional: print per-episode summary for debugging
            print(
                f"Episode {episode_idx + 1}/{episodes}: reward={total_reward:.4f}, steps={steps}"
            )

        component_summary = {
            key: stats.summary() for key, stats in components.items()
        }
        metric_summary = {
            key: summary
            for key, summary in ((k, v.summary()) for k, v in portfolio_metrics.items())
        }
        return component_summary, metric_summary
    finally:
        vec_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect reward component magnitudes.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to phase3 YAML config",
    )
    parser.add_argument("--symbol", type=str, default="SPY", help="Symbol to analyze")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to sample",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of random episodes to sample",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON summary",
    )

    args = parser.parse_args()

    component_summary, metric_summary = collect_reward_statistics(
        config_path=args.config,
        symbol=args.symbol,
        split=args.split,
        episodes=args.episodes,
        seed=args.seed,
    )

    payload: Dict[str, Any] = {
        "config": str(args.config.resolve()),
        "symbol": args.symbol,
        "split": args.split,
        "episodes": args.episodes,
        "components": component_summary,
        "portfolio_metrics": metric_summary,
    }

    print(json.dumps(payload, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Summary written to {args.output}")


if __name__ == "__main__":
    main()
