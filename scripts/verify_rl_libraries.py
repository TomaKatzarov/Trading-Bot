"""Verify installation of key RL libraries and basic environment sanity checks."""

from __future__ import annotations

import json
import sys


def main() -> None:
    messages: list[str] = []

    try:
        import gymnasium as gym  # type: ignore
        from stable_baselines3 import PPO  # type: ignore
        from stable_baselines3 import __version__ as sb3_version  # type: ignore
        from stable_baselines3.common.env_checker import check_env  # type: ignore
        import ray  # type: ignore
        from ray import tune  # noqa: F401  # type: ignore
        from ray.rllib.algorithms.ppo import PPOConfig  # type: ignore

        messages.append("All RL libraries imported successfully")
        messages.append(f"Gymnasium version: {gym.__version__}")
        messages.append(f"Stable-Baselines3 version: {sb3_version}")
        messages.append(f"Ray version: {ray.__version__}")

        # Minimal environment sanity check
        env = gym.make("CartPole-v1")
        check_env(env)
        messages.append("Gymnasium environment check passed (CartPole-v1)")

        # Simple PPO config instantiation to ensure RLlib compatibility
        _ = PPOConfig().environment(env="CartPole-v1")
        messages.append("RLlib PPOConfig instantiated successfully")

    except Exception as exc:  # pragma: no cover - release diagnostics
        print("RL library verification failed")
        print(exc)
        sys.exit(1)

    print("\n".join(messages))


if __name__ == "__main__":
    main()