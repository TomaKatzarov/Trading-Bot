"""Utility to evaluate a trained PPO policy and extract action distributions."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rl.environments.trading_env import TradeAction
from core.rl.environments.vec_trading_env import make_vec_trading_env
from training.train_phase3_agents import (
    extract_episode_metrics,
    load_config,
    make_env_kwargs,
    summarize_episode_metrics,
)


@dataclass
class EvaluationResult:
    symbol: str
    model_path: Path
    episodes: List[Dict[str, float]]
    action_counts: Dict[str, int]
    total_steps: int

    @property
    def summary(self) -> Dict[str, float]:  # pragma: no cover - simple proxy
        return summarize_episode_metrics(self.episodes)

    @property
    def action_distribution(self) -> Dict[str, float]:
        if self.total_steps <= 0:
            return {action: 0.0 for action in self.action_counts}
        return {action: count / self.total_steps for action, count in self.action_counts.items()}


def _resolve_vecnormalize(config: Dict[str, Dict], checkpoint_dir: Path) -> Tuple[bool, Dict[str, float], Path]:
    norm_cfg = config.get("training", {}).get("reward_normalization", {})
    use_vecnormalize = bool(norm_cfg.get("enabled", False))
    vecnorm_kwargs = {
        "norm_obs": bool(norm_cfg.get("norm_obs", False)),
        "norm_reward": bool(norm_cfg.get("norm_reward", True)),
        "clip_reward": float(norm_cfg.get("clip_reward", 5.0)),
        "gamma": float(norm_cfg.get("gamma", config.get("ppo", {}).get("gamma", 0.99))),
    }
    vec_stats_path = checkpoint_dir / "vecnormalize.pkl"
    return use_vecnormalize, vecnorm_kwargs, vec_stats_path


def _load_environment(
    config: Dict[str, Dict],
    symbol: str,
    split: str,
    *,
    use_vecnormalize: bool,
    vecnorm_kwargs: Dict[str, float],
    vec_stats_path: Path,
) -> VecEnv:
    experiment_cfg = config["experiment"]
    env_kwargs = make_env_kwargs(symbol, split, config)

    env = make_vec_trading_env(
        symbol=symbol,
        data_dir=Path(experiment_cfg["data_dir"]) / symbol,
        num_envs=1,
        seed=config.get("training", {}).get("seed", 42) + 999,
        use_subprocess=False,
        env_kwargs=env_kwargs,
        env_log_level=env_kwargs.get("log_level"),
    )

    if use_vecnormalize:
        env = VecNormalize(env, **vecnorm_kwargs)
        env.training = False
        env.norm_reward = bool(vecnorm_kwargs["norm_reward"])  # type: ignore[attr-defined]
        if vec_stats_path.exists():
            env = VecNormalize.load(str(vec_stats_path), env)
            env.training = False
            env.norm_reward = bool(vecnorm_kwargs["norm_reward"])  # type: ignore[attr-defined]

    return env


def evaluate_policy(
    model: PPO,
    env: VecEnv,
    episodes: int,
    *,
    symbol: str,
    model_path: Path,
    deterministic: bool = True,
) -> EvaluationResult:
    action_counts: Dict[str, int] = {action.name: 0 for action in TradeAction}
    total_steps = 0
    collected: List[Dict[str, float]] = []

    for _ in range(max(1, episodes)):
        obs = env.reset()
        state = None
        done = False
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            action_idx = int(np.squeeze(action))
            action_name = TradeAction(action_idx).name if action_idx in TradeAction._value2member_map_ else str(action_idx)
            action_counts.setdefault(action_name, 0)
            action_counts[action_name] += 1
            total_steps += 1

            obs, rewards, dones, infos = env.step(action)
            done = bool(dones[0])
            if done:
                base_env = env.envs[0] if hasattr(env, "envs") else None
                if base_env is None:
                    raise RuntimeError("Evaluation env does not expose base environment")
                collected.append(extract_episode_metrics(base_env, infos[0]))

    return EvaluationResult(
        symbol=symbol,
        model_path=model_path,
        episodes=collected,
        action_counts=action_counts,
        total_steps=total_steps,
    )


def run_evaluation(
    config_path: Path,
    symbol: str,
    *,
    model_path: Path,
    episodes: int,
    split: str = "val",
    deterministic: bool = True,
) -> Dict[str, object]:
    config = load_config(config_path)
    checkpoint_dir = Path(config["experiment"]["checkpoint_dir"]) / symbol
    use_vecnormalize, vecnorm_kwargs, vec_stats_path = _resolve_vecnormalize(config, checkpoint_dir)

    env = _load_environment(
        config,
        symbol,
        split,
        use_vecnormalize=use_vecnormalize,
        vecnorm_kwargs=vecnorm_kwargs,
        vec_stats_path=vec_stats_path,
    )

    model = PPO.load(str(model_path), env=env)
    result = evaluate_policy(
        model,
        env,
        episodes,
        symbol=symbol,
        model_path=model_path,
        deterministic=deterministic,
    )

    payload = {
        "symbol": symbol,
        "model_path": str(model_path),
        "episodes": len(result.episodes),
        "total_steps": result.total_steps,
        "action_counts": result.action_counts,
        "action_distribution": result.action_distribution,
        "summary": result.summary,
        "episodes_detail": result.episodes,
    }

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract action distribution from evaluation runs")
    parser.add_argument("--config", required=True, type=Path, help="Path to training configuration YAML")
    parser.add_argument("--symbol", required=True, help="Symbol to evaluate")
    parser.add_argument("--model-path", type=Path, help="Path to model checkpoint (default final_model)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate (train/val/test)")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions")
    parser.add_argument("--output", type=Path, help="Optional path to write JSON payload")

    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    checkpoint_dir = Path(config["experiment"]["checkpoint_dir"]) / args.symbol

    model_path = args.model_path or (checkpoint_dir / "final_model.zip")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    payload = run_evaluation(
        config_path=config_path,
        symbol=args.symbol,
        model_path=model_path,
        episodes=args.episodes,
        split=args.split,
        deterministic=args.deterministic,
    )

    output = args.output or (checkpoint_dir / f"action_distribution_{model_path.stem}.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
