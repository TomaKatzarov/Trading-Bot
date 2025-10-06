"""Validate SymbolAgent compatibility with Stable-Baselines3 PPO interface."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Ensure project root is on the import path when executing the script directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.rl.policies.feature_encoder import EncoderConfig
from core.rl.policies.symbol_agent import SymbolAgent, SymbolAgentConfig


def test_ppo_interface() -> None:
    """Execute a lightweight validation of PPO-facing methods."""

    print("=" * 70)
    print("PPO INTERFACE VALIDATION")
    print("=" * 70)

    encoder_config = EncoderConfig()
    config = SymbolAgentConfig(encoder_config=encoder_config)
    agent = SymbolAgent(config)
    agent.eval()

    batch_size = 16
    observations = {
        "technical": torch.randn(batch_size, 24, 23),
        "sl_probs": torch.rand(batch_size, 3),
        "position": torch.zeros(batch_size, 5),
        "portfolio": torch.tensor(
            [[10000.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        ).repeat(batch_size, 1),
        "regime": torch.randn(batch_size, 10),
    }

    print("\n1. Testing forward() method...")
    actions, log_probs, values = agent(observations)
    print(f"   ✅ Actions shape: {tuple(actions.shape)}")
    print(f"   ✅ Log probs shape: {tuple(log_probs.shape)}")
    print(f"   ✅ Values shape: {tuple(values.shape)}")

    print("\n2. Testing get_value() method...")
    values_only = agent.get_value(observations)
    print(f"   ✅ Values shape: {tuple(values_only.shape)}")

    print("\n3. Testing evaluate_actions() method...")
    test_actions = torch.randint(0, config.action_dim, (batch_size,))
    eval_log_probs, eval_values, entropy = agent.evaluate_actions(observations, test_actions)
    print(f"   ✅ Log probs shape: {tuple(eval_log_probs.shape)}")
    print(f"   ✅ Values shape: {tuple(eval_values.shape)}")
    print(f"   ✅ Entropy shape: {tuple(entropy.shape)}")

    print("\n4. Testing action masking...")
    obs_with_position = {key: value.clone() for key, value in observations.items()}
    obs_with_position["position"] = torch.tensor(
        [[100.0, 5.0, 50.0, 10.0, 1.0]], dtype=torch.float32
    ).repeat(batch_size, 1)
    masked_actions, _, _ = agent(obs_with_position)

    disallowed_buys = torch.tensor([1, 2, 3], device=masked_actions.device)
    assert not torch.isin(masked_actions, disallowed_buys).any(), (
        "BUY actions should be masked when a position is already held!"
    )
    print("   ✅ Action masking works correctly")

    print("\n" + "=" * 70)
    print("🎉 ALL PPO INTERFACE TESTS PASSED!")
    print("=" * 70)
    print("\nSymbol Agent is compatible with Stable-Baselines3 PPO")
    print("Ready for Phase 3 training!")


if __name__ == "__main__":
    test_ppo_interface()
