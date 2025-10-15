"""Validation utilities for Phase A continuous action integration."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import A2C, PPO, SAC

from core.rl.curiosity.icm import ICMConfig, TradingICM
from core.rl.environments.action_space_migrator import ActionSpaceMigrator
from training.rl.env_factory import build_trading_config, load_yaml
from training.train_sac_continuous import TradingSAC, prepare_vec_env


# ---------------------------------------------------------------------------
# Dataclasses & helpers
# ---------------------------------------------------------------------------


@dataclass
class EpisodeMetrics:
    reward: float
    sharpe: float
    return_pct: float
    trades: int
    length: int


@dataclass
class EvaluationSummary:
    episodes: List[EpisodeMetrics]
    action_values: List[float]
    executed_actions: List[str]
    mode: str

    def to_dict(self) -> Dict[str, float]:
        sharpes = [ep.sharpe for ep in self.episodes if math.isfinite(ep.sharpe)]
        returns = [ep.return_pct for ep in self.episodes if math.isfinite(ep.return_pct)]
        trades = [ep.trades for ep in self.episodes]
        lengths = [ep.length for ep in self.episodes]

        entropy = compute_entropy(self.action_values)
        action_share = compute_max_action_share(self.executed_actions)
        trade_mean = float(np.mean(trades)) if trades else float("nan")
        sharpe_mean = float(np.mean(sharpes)) if sharpes else float("nan")
        sharpe_min = float(np.min(sharpes)) if sharpes else float("nan")
        return_mean = float(np.mean(returns)) if returns else float("nan")
        length_mean = float(np.mean(lengths)) if lengths else float("nan")
        trade_rate = (trade_mean / length_mean) if (np.isfinite(trade_mean) and np.isfinite(length_mean) and length_mean > 0) else float("nan")
        action_coverage = compute_action_coverage(self.action_values)

        return {
            "mode": self.mode,
            "episodes": len(self.episodes),
            "mean_sharpe": sharpe_mean,
            "min_sharpe": sharpe_min,
            "mean_return_pct": return_mean,
            "mean_trades_per_episode": trade_mean,
            "mean_episode_length": length_mean,
            "trade_execution_rate": trade_rate,
            "action_entropy": entropy,
            "action_range_coverage": action_coverage,
            "max_executed_action_share": action_share,
        }


@dataclass
class EnvironmentValidation:
    steps: int
    nan_observations: int
    nan_rewards: int
    action_min: float
    action_max: float
    unique_actions: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "steps": self.steps,
            "nan_observations": self.nan_observations,
            "nan_rewards": self.nan_rewards,
            "action_min": self.action_min,
            "action_max": self.action_max,
            "unique_actions": self.unique_actions,
        }


@dataclass
class ValidationReport:
    environment: EnvironmentValidation
    sac_model_ok: bool
    icm_ok: bool
    continuous_eval: EvaluationSummary
    discrete_eval: Optional[EvaluationSummary] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "environment": self.environment.to_dict(),
            "sac_model_ok": self.sac_model_ok,
            "icm_ok": self.icm_ok,
            "continuous_eval": self.continuous_eval.to_dict(),
        }
        if self.discrete_eval is not None:
            report["discrete_eval"] = self.discrete_eval.to_dict()
            report["metrics"] = self.metrics
        else:
            report["metrics"] = self.metrics
        return report


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def compute_entropy(values: Iterable[float], bins: int = 20) -> float:
    data = list(values)
    if not data:
        return float("nan")
    counts, _ = np.histogram(np.asarray(data, dtype=np.float64), bins=bins, range=(-1.0, 1.0))
    total = counts.sum()
    if total == 0:
        return float("nan")
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def compute_max_action_share(actions: Iterable[str]) -> float:
    filtered = [act for act in actions if act]
    if not filtered:
        return float("nan")
    counter = Counter(filtered)
    total = sum(counter.values())
    if total == 0:
        return float("nan")
    return float(max(counter.values()) / total)


def compute_action_coverage(values: Iterable[float]) -> float:
    data = list(values)
    if not data:
        return float("nan")
    min_val = min(data)
    max_val = max(data)
    coverage = (max_val - min_val) / 2.0  # range of [-1,1] has width 2
    return float(max(0.0, min(1.0, coverage)))


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def _has_nan(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_has_nan(v) for v in value.values())
    arr = np.asarray(value)
    try:
        return bool(np.isnan(arr).any())
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# Model loading utilities
# ---------------------------------------------------------------------------


def _resolve_policy_artifact(path: Path) -> Optional[Path]:
    if path.suffix == ".pt" and path.exists():
        return path

    candidates: List[Path] = [path.with_name(f"{path.stem}_policy.pt")]
    if path.suffix == ".zip":
        stem = path.stem
        if stem.endswith("_final"):
            candidates.append(path.with_name(f"{stem[:-6]}_policy.pt"))
        candidates.append(path.with_suffix(".pt"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    fallback = sorted(path.parent.glob("*policy.pt")) if path.parent.exists() else []
    return fallback[0] if fallback else None


def _load_continuous_sac(config: Dict[str, Any], model_path: Path, seed: int) -> SAC:
    if model_path.suffix == ".zip" and model_path.exists() and model_path.stat().st_size > 0:
        model = SAC.load(str(model_path))
        model.policy.set_training_mode(False)
        return model

    policy_path = _resolve_policy_artifact(model_path)
    if policy_path is None or not policy_path.exists():
        raise FileNotFoundError(f"No usable policy checkpoint found alongside {model_path}")

    env_cfg = config.get("environment", config)
    vec_env = prepare_vec_env(env_cfg, seed=seed, mode="continuous", num_envs=1)
    trainer = TradingSAC(vec_env, vec_env, config)
    model = trainer.model

    try:
        state_dict = torch.load(policy_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(policy_path, map_location="cpu")

    load_info = model.policy.load_state_dict(state_dict, strict=False)
    missing = getattr(load_info, "missing_keys", [])
    unexpected = getattr(load_info, "unexpected_keys", [])
    if missing or unexpected:
        print(
            json.dumps(
                {
                    "policy_load_warning": {
                        "missing_keys": sorted(missing),
                        "unexpected_keys": sorted(unexpected),
                    }
                }
            )
        )
    model.policy.to(torch.device("cpu"))
    model.policy.set_training_mode(False)
    return model


def _load_discrete_model(model_path: Path):
    errors: List[str] = []
    for algo in (PPO, A2C):
        try:
            model = algo.load(str(model_path))
            model.policy.set_training_mode(False)
            return model
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{algo.__name__}: {exc}")
    raise RuntimeError(f"Unable to load discrete policy from {model_path}: {'; '.join(errors)}")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def run_environment_validation(config: Dict[str, Any], *, seed: int, steps: int = 512) -> EnvironmentValidation:
    trading_cfg = build_trading_config(config)
    env = ActionSpaceMigrator.create_hybrid_environment(trading_cfg, seed=seed, mode="continuous")

    try:
        obs, _ = env.reset(seed=seed)
        nan_obs = 0
        nan_rewards = 0
        action_values: List[float] = []

        for step in range(steps):
            if _has_nan(obs):
                nan_obs += 1
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _info = env.step(action)
            action_values.append(float(np.asarray(action).squeeze()))
            if np.isnan(reward):
                nan_rewards += 1
            if terminated or truncated:
                obs, _ = env.reset(seed=seed + step + 1)

        unique_actions = len({round(val, 3) for val in action_values})
        return EnvironmentValidation(
            steps=steps,
            nan_observations=nan_obs,
            nan_rewards=nan_rewards,
            action_min=float(min(action_values) if action_values else 0.0),
            action_max=float(max(action_values) if action_values else 0.0),
            unique_actions=unique_actions,
        )
    finally:
        env.close()


def evaluate_policy(
    config: Dict[str, Any],
    model,
    *,
    trading_cfg,
    mode: str,
    episodes: int,
    seed: int,
    stochastic_continuous: bool = True,
    continuous_temperature: float = 1.0,
) -> EvaluationSummary:
    env = ActionSpaceMigrator.create_hybrid_environment(trading_cfg, seed=seed, mode=mode)

    episode_metrics: List[EpisodeMetrics] = []
    action_values: List[float] = []
    executed_actions: List[str] = []

    try:
        if mode == "continuous" and stochastic_continuous and continuous_temperature <= 0:
            raise ValueError("continuous_temperature must be positive for continuous evaluations")

        policy = getattr(model, "policy", None)
        actor = getattr(policy, "actor", None) if policy is not None else None

        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            done = False
            episode_reward = 0.0
            episode_trades = 0
            episode_length = 0

            while not done:
                deterministic = True
                if mode == "continuous" and stochastic_continuous:
                    deterministic = False

                if not deterministic:
                    use_temperature = (
                        mode == "continuous"
                        and actor is not None
                        and not math.isclose(continuous_temperature, 1.0, rel_tol=1e-6, abs_tol=1e-6)
                    )
                else:
                    use_temperature = False

                if use_temperature:
                    obs_tensor, _ = policy.obs_to_tensor(observation)  # type: ignore[union-attr]
                    with torch.no_grad():
                        mean_actions, log_std, kwargs = actor.get_action_dist_params(obs_tensor)
                        scaled_log_std = log_std + math.log(continuous_temperature)
                        action_dist = actor.action_dist
                        if hasattr(action_dist, "sample_weights"):
                            try:
                                action_dist.sample_weights(scaled_log_std)
                            except TypeError:
                                action_dist.sample_weights(scaled_log_std, batch_size=1)
                            action_dist = action_dist.proba_distribution(mean_actions, scaled_log_std, **kwargs)
                        else:
                            action_dist = action_dist.proba_distribution(mean_actions, scaled_log_std)
                        action_tensor = action_dist.get_actions(deterministic=False)
                    action = action_tensor.detach().cpu().numpy()
                    if np.ndim(action) >= 2 and action.shape[0] == 1:
                        action = action[0]
                else:
                    action, _ = model.predict(observation, deterministic=deterministic)
                adapted_action = ActionSpaceMigrator.adapt_action(action)
                step_action = adapted_action if mode == "continuous" else int(np.asarray(action).squeeze())
                observation, reward, terminated, truncated, info = env.step(step_action)
                done = bool(terminated or truncated)
                episode_reward += float(reward)
                episode_length += 1

                if mode == "continuous":
                    action_block = info.get("continuous_action", {})
                    if action_block:
                        raw = action_block.get("raw", action_block.get("smoothed"))
                        if raw is not None:
                            action_values.append(float(raw))
                        executed_flag = bool(info.get("action_executed"))
                        if not executed_flag:
                            executed_flag = bool(info.get("action_info", {}).get("action_executed"))
                        if executed_flag:
                            discrete_label = action_block.get("discrete_action")
                            if discrete_label is not None:
                                executed_actions.append(str(discrete_label))
                            episode_trades += 1
                else:
                    # Discrete action evaluation
                    action_scalar = int(np.asarray(action).squeeze())
                    action_values.append(ActionSpaceMigrator.discrete_to_continuous(action_scalar))
                    action_name = info.get("executed_action_name")
                    if action_name is not None:
                        executed_actions.append(str(action_name))
                        if action_name != "HOLD":
                            episode_trades += 1

                if done:
                    metrics = info.get("terminal_metrics", {})
                    episode_metrics.append(
                        EpisodeMetrics(
                            reward=episode_reward,
                            sharpe=float(metrics.get("sharpe_ratio", float("nan"))),
                            return_pct=float(metrics.get("total_return_pct", float("nan"))),
                            trades=episode_trades,
                            length=episode_length,
                        )
                    )
    finally:
        env.close()
        try:
            model_env = model.get_env()
            if model_env is not None:
                model_env.close()
        except Exception:  # noqa: BLE001
            pass

    return EvaluationSummary(episodes=episode_metrics, action_values=action_values, executed_actions=executed_actions, mode=mode)


def validate_icm() -> bool:
    icm = TradingICM(ICMConfig())
    batch_size = 32
    states = torch.randn(batch_size, icm.config.state_dim)
    next_states = torch.randn(batch_size, icm.config.state_dim)
    actions = torch.randn(batch_size, icm.config.action_dim)

    intrinsic_reward, losses = icm(states, next_states, actions)
    reward_ok = torch.isfinite(intrinsic_reward).all().item()
    loss_ok = all(torch.isfinite(value).item() for value in losses.values())
    non_negative = losses["forward_loss"] >= 0 and losses["inverse_loss"] >= 0 and losses["total_loss"] >= 0
    return bool(reward_ok and loss_ok and non_negative)


def compare_entropy_improvement(continuous: EvaluationSummary, discrete: EvaluationSummary) -> Tuple[float, float, float]:
    cont_entropy = continuous.to_dict().get("action_entropy", float("nan"))
    disc_entropy = discrete.to_dict().get("action_entropy", float("nan"))
    if not np.isfinite(cont_entropy) or not np.isfinite(disc_entropy):
        return float("nan"), cont_entropy, disc_entropy
    if disc_entropy == 0:
        return float("inf"), cont_entropy, disc_entropy
    improvement = (cont_entropy - disc_entropy) / abs(disc_entropy)
    return float(improvement), cont_entropy, disc_entropy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase A continuous action integration")
    parser.add_argument("--config", type=Path, required=True, help="Path to SAC training config YAML")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained SAC model (.zip or .pt)")
    parser.add_argument("--discrete-model", type=Path, help="Reference discrete model for comparison")
    parser.add_argument("--episodes", type=int, default=None, help="Number of evaluation episodes to simulate")
    parser.add_argument("--seed", type=int, default=2025, help="Base random seed for evaluation")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/reports/phase_a_validation_report.json"),
        help="Path to store JSON summary",
    )
    parser.add_argument(
        "--continuous-temperature",
        type=float,
        default=None,
        help="Scale factor for SAC stochastic action standard deviation (values > 1 widen coverage)",
    )
    parser.add_argument(
        "--deterministic-continuous",
        action="store_true",
        help="Disable stochastic sampling for the continuous SAC policy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    eval_cfg = config.get("evaluation", {})

    if args.episodes is None:
        args.episodes = int(eval_cfg.get("episodes", 10))

    if args.continuous_temperature is None:
        args.continuous_temperature = float(eval_cfg.get("continuous_temperature", 1.0))

    if not args.deterministic_continuous and eval_cfg.get("stochastic_continuous") is False:
        args.deterministic_continuous = True

    env_cfg = config.get("environment", config)
    trading_cfg = build_trading_config(env_cfg)

    environment_validation = run_environment_validation(env_cfg, seed=args.seed)

    sac_model = _load_continuous_sac(config, args.model, seed=args.seed)

    test_env = ActionSpaceMigrator.create_hybrid_environment(trading_cfg, seed=args.seed, mode="continuous")
    try:
        sample_obs, _ = test_env.reset(seed=args.seed)
    finally:
        test_env.close()

    sac_actions: List[float] = []
    for _ in range(5):
        action, _ = sac_model.predict(sample_obs, deterministic=False)
        sac_actions.append(float(np.asarray(action).squeeze()))
    sac_model_ok = all(-1.01 <= act <= 1.01 for act in sac_actions)

    icm_ok = validate_icm()

    continuous_eval = evaluate_policy(
        config,
        sac_model,
        trading_cfg=trading_cfg,
        mode="continuous",
        episodes=args.episodes,
        seed=args.seed,
        stochastic_continuous=not args.deterministic_continuous,
        continuous_temperature=args.continuous_temperature,
    )

    discrete_eval: Optional[EvaluationSummary] = None
    metrics: Dict[str, Any] = {}

    if args.discrete_model:
        discrete_model = _load_discrete_model(args.discrete_model)
        discrete_config_path = Path("training/config_templates/phase3_ppo_baseline.yaml")
        if discrete_config_path.exists():
            discrete_config = load_yaml(discrete_config_path)
            discrete_env_cfg = discrete_config.get("environment", discrete_config)
            try:
                discrete_trading_cfg = build_trading_config(discrete_env_cfg)
            except KeyError as exc:
                print(f"[warn] Discrete config incomplete ({exc}); reusing continuous trading config")
                discrete_trading_cfg = trading_cfg
        else:
            discrete_trading_cfg = trading_cfg
        discrete_eval = evaluate_policy(
            discrete_config if 'discrete_config' in locals() else config,
            discrete_model,
            trading_cfg=discrete_trading_cfg,
            mode="discrete",
            episodes=args.episodes,
            seed=args.seed,
        )
        improvement, cont_entropy, disc_entropy = compare_entropy_improvement(continuous_eval, discrete_eval)
        metrics.update(
            {
                "entropy_improvement": improvement,
                "continuous_entropy": cont_entropy,
                "discrete_entropy": disc_entropy,
            }
        )

    report = ValidationReport(
        environment=environment_validation,
        sac_model_ok=sac_model_ok,
        icm_ok=icm_ok,
        continuous_eval=continuous_eval,
        discrete_eval=discrete_eval,
        metrics=metrics,
    )

    payload = report.to_payload()

    continuous_metrics = continuous_eval.to_dict()
    quality = {
        "environment_clean": environment_validation.nan_observations == 0 and environment_validation.nan_rewards == 0,
        "sac_model_outputs_in_range": sac_model_ok,
        "icm_forward_pass_ok": icm_ok,
        "continuous_entropy_gt_0_6": continuous_metrics.get("action_entropy", float("nan")) > 0.6,
        "trade_execution_rate_gt_0_05": continuous_metrics.get("trade_execution_rate", float("nan")) > 0.05,
        "action_range_coverage_gt_0_9": continuous_metrics.get("action_range_coverage", float("nan")) > 0.9,
    }

    if discrete_eval is not None:
        quality["entropy_improvement_gt_20pct"] = metrics.get("entropy_improvement", float("nan")) > 0.20

    payload["quality"] = quality

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))

    if not all(value for value in quality.values() if isinstance(value, bool)):
        raise SystemExit("One or more quality gates failed")


if __name__ == "__main__":
    main()
