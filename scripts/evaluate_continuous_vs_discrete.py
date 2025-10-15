"""Evaluate discrete PPO vs continuous SAC trading agents."""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import mlflow
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC

from core.rl.environments.action_space_migrator import ActionSpaceMigrator
from training.rl.env_factory import build_trading_config, load_yaml

LOGGER = logging.getLogger("evaluation.action_space")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s")


@dataclass
class EvaluationResult:
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    mean_trades_per_episode: float
    action_diversity: float
    total_return: float
    sharpe_ratio: float
    action_space: str
    algorithm: str

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "mean_episode_length": self.mean_episode_length,
            "mean_trades_per_episode": self.mean_trades_per_episode,
            "action_diversity": self.action_diversity,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "action_space": self.action_space,
            "algorithm": self.algorithm,
        }


class ActionSpaceComparator:
    """Compare discrete and continuous policies under identical market data."""

    def __init__(self, discrete_cfg: Dict[str, any], continuous_cfg: Dict[str, any], *, seed: int = 123) -> None:
        self.discrete_cfg = discrete_cfg
        self.continuous_cfg = continuous_cfg
        self.seed = seed

    def _make_env(self, cfg_dict: Dict[str, any], *, mode: str) -> any:
        trading_cfg = build_trading_config(cfg_dict)
        return ActionSpaceMigrator.create_hybrid_environment(trading_cfg, seed=self.seed, mode=mode)

    def _run_evaluation(self, model, cfg_dict: Dict[str, any], *, mode: str, n_episodes: int) -> EvaluationResult:
        env = self._make_env(cfg_dict, mode=mode)
        rewards = []
        lengths = []
        trade_counts = []
        diversities = []

        try:
            for episode in range(n_episodes):
                obs, _ = env.reset(seed=self.seed + episode)
                done = False
                episode_reward = 0.0
                episode_length = 0
                actions = []
                trades = 0

                while not done:
                    action, _state = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)

                    episode_reward += float(reward)
                    episode_length += 1

                    if isinstance(info, dict):
                        executed = info.get("executed_action_name") or info.get("interpreted_action")
                        if executed is not None:
                            actions.append(str(executed))
                        if info.get("executed_action_name") and info["executed_action_name"] != "HOLD":
                            trades += 1

                rewards.append(episode_reward)
                lengths.append(episode_length)
                trade_counts.append(trades)
                if actions:
                    unique_actions = len(set(actions))
                    diversities.append(unique_actions / max(1, len(actions)))
        finally:
            env.close()

        rewards_array = np.asarray(rewards, dtype=np.float64)
        mean_reward = float(np.mean(rewards_array)) if rewards else 0.0
        std_reward = float(np.std(rewards_array)) if rewards else 0.0
        mean_length = float(np.mean(lengths)) if lengths else 0.0
        mean_trades = float(np.mean(trade_counts)) if trade_counts else 0.0
        action_diversity = float(np.mean(diversities)) if diversities else 0.0
        total_return = float(np.sum(rewards_array))
        sharpe_ratio = float(mean_reward / (std_reward + 1e-8)) if rewards else 0.0

        return EvaluationResult(
            mean_reward=mean_reward,
            std_reward=std_reward,
            mean_episode_length=mean_length,
            mean_trades_per_episode=mean_trades,
            action_diversity=action_diversity,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            action_space="continuous" if mode == "continuous" else "discrete",
            algorithm="SAC" if mode == "continuous" else "PPO",
        )

    def compare(self, sac_model_path: Path, ppo_model_path: Path, *, episodes: int) -> pd.DataFrame:
        sac_model = SAC.load(str(sac_model_path))
        ppo_model = PPO.load(str(ppo_model_path))

        continuous_result = self._run_evaluation(
            sac_model,
            self.continuous_cfg,
            mode="continuous",
            n_episodes=episodes,
        )
        discrete_result = self._run_evaluation(
            ppo_model,
            self.discrete_cfg,
            mode="discrete",
            n_episodes=episodes,
        )

        df = pd.DataFrame(
            {
                "continuous": continuous_result.to_dict(),
                "discrete": discrete_result.to_dict(),
            }
        ).T

        improvements = {}
        for key in ["mean_reward", "std_reward", "mean_episode_length", "mean_trades_per_episode", "action_diversity", "total_return", "sharpe_ratio"]:
            base = df.loc["discrete", key]
            value = df.loc["continuous", key]
            if isinstance(base, str) or isinstance(value, str):
                continue
            improvements[f"improvement_%_{key}"] = 100.0 * ((value - base) / (abs(base) + 1e-8))
        improvement_series = pd.Series(improvements, name="improvement")
        df = pd.concat([df, improvement_series.to_frame().T], axis=0)
        return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare discrete PPO vs continuous SAC")
    parser.add_argument("--continuous-config", required=True, type=Path, help="Continuous environment YAML config")
    parser.add_argument("--discrete-config", required=True, type=Path, help="Discrete environment YAML config")
    parser.add_argument("--sac-model", required=True, type=Path, help="Path to trained SAC model")
    parser.add_argument("--ppo-model", required=True, type=Path, help="Path to trained PPO model")
    parser.add_argument("--episodes", type=int, default=25, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for evaluation")
    parser.add_argument("--mlflow-uri", type=str, help="Optional MLflow tracking URI")
    parser.add_argument("--mlflow-experiment", type=str, default="action-space-comparison", help="MLflow experiment name")
    return parser.parse_args()


def maybe_start_mlflow(args: argparse.Namespace) -> Optional[str]:
    if not args.mlflow_uri:
        return None
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    run = mlflow.start_run(run_name=f"action-comparison-{Path(args.sac_model).stem}")
    return run.info.run_id


def main() -> None:
    args = parse_args()
    continuous_cfg = load_yaml(args.continuous_config)
    discrete_cfg = load_yaml(args.discrete_config)

    comparator = ActionSpaceComparator(
        discrete_cfg=discrete_cfg.get("environment", discrete_cfg),
        continuous_cfg=continuous_cfg.get("environment", continuous_cfg),
        seed=args.seed,
    )

    run_id = maybe_start_mlflow(args)
    try:
        df = comparator.compare(args.sac_model, args.ppo_model, episodes=args.episodes)
        LOGGER.info("Evaluation results:\n%s", df)
        output_path = Path("reports") / "continuous_vs_discrete.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=True)
        LOGGER.info("Saved comparison report to %s", output_path.resolve())

        if run_id:
            mlflow.log_artifact(str(output_path))
            mlflow.log_metric("action_diversity_improvement", float(df.loc["improvement", "improvement_%_action_diversity"]))
    finally:
        if run_id:
            mlflow.end_run()


if __name__ == "__main__":
    main()
