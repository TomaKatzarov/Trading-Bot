"""Utilities for managing shared encoder weights across symbol agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch.nn as nn


@dataclass
class ParameterReport:
    """Summary of parameter counts for shared encoder deployments."""

    total_params: int
    encoder_params: int
    agent_params: int
    num_agents: int
    shared_params: int
    unique_params_per_agent: int
    memory_mb: float
    savings_percent: float


class SharedEncoderManager:
    """Manage a collection of symbol agents that share a single encoder instance."""

    def __init__(self, encoder: nn.Module) -> None:
        self.encoder = encoder
        self.agents: List[nn.Module] = []
        self.encoder_params = self._count_parameters(encoder)

    def register_agent(self, agent: nn.Module) -> None:
        """Register a symbol agent that must reference the managed encoder."""

        if not hasattr(agent, "encoder"):
            raise AttributeError("Agent must expose an 'encoder' attribute")
        if getattr(agent, "encoder") is not self.encoder:
            raise ValueError("Agent must use the shared encoder instance")
        self.agents.append(agent)

    def _count_parameters(self, module: nn.Module) -> int:
        return sum(param.numel() for param in module.parameters() if param.requires_grad)

    def get_parameter_report(self) -> ParameterReport:
        """Generate a formal savings report for the registered agents."""

        if not self.agents:
            raise ValueError("At least one agent must be registered before reporting")

        sample_agent = self.agents[0]
        if not hasattr(sample_agent, "actor") or not hasattr(sample_agent, "critic"):
            raise AttributeError("Agents must expose 'actor' and 'critic' attributes")

        actor_params = self._count_parameters(sample_agent.actor)
        critic_params = self._count_parameters(sample_agent.critic)
        agent_unique_params = actor_params + critic_params

        num_agents = len(self.agents)
        total_with_sharing = self.encoder_params + num_agents * agent_unique_params
        total_without_sharing = num_agents * (self.encoder_params + agent_unique_params)

        memory_mb = (total_with_sharing * 4) / (1024 ** 2)
        savings_percent = 100.0 * (1 - (total_with_sharing / total_without_sharing))

        return ParameterReport(
            total_params=total_with_sharing,
            encoder_params=self.encoder_params,
            agent_params=agent_unique_params,
            num_agents=num_agents,
            shared_params=self.encoder_params,
            unique_params_per_agent=agent_unique_params,
            memory_mb=memory_mb,
            savings_percent=savings_percent,
        )

    def verify_sharing(self) -> Dict[str, bool]:
        """Confirm that every registered agent references the shared encoder."""

        results: Dict[str, bool] = {
            "all_share_encoder": True,
            "encoder_trainable": self.encoder.training,
            "num_agents": len(self.agents),
        }

        for index, agent in enumerate(self.agents):
            shares = getattr(agent, "encoder", None) is self.encoder
            results[f"agent_{index}_shares"] = shares
            if not shares:
                results["all_share_encoder"] = False

        return results

    def print_report(self) -> None:
        """Print a human-readable summary highlighting savings from sharing."""

        report = self.get_parameter_report()
        total_without = report.num_agents * (report.encoder_params + report.unique_params_per_agent)
        memory_without = (total_without * 4) / (1024 ** 2)

        print("=" * 70)
        print("WEIGHT SHARING PARAMETER REPORT")
        print("=" * 70)
        print(f"Shared encoder parameters: {report.encoder_params:,}")
        print(f"Per-agent unique parameters (actor+critic): {report.unique_params_per_agent:,}")
        print(f"Number of agents: {report.num_agents}")
        print("\n--- WITH SHARING ---")
        print(f"Total parameters: {report.total_params:,}")
        print(f"Memory usage: {report.memory_mb:.2f} MB")
        print("\n--- WITHOUT SHARING (Hypothetical) ---")
        print(f"Total parameters: {total_without:,}")
        print(f"Memory usage: {memory_without:.2f} MB")
        print("\n--- SAVINGS ---")
        print(f"Parameter reduction: {report.savings_percent:.1f}%")
        print(f"Memory saved: {memory_without - report.memory_mb:.2f} MB")
        print("=" * 70)


def calculate_savings_for_agents(
    encoder_params: int,
    agent_params: int,
    num_agents: int,
) -> Tuple[int, int, float]:
    """Return parameter totals and savings percentage for shared encoders."""

    if num_agents <= 0:
        raise ValueError("num_agents must be positive")
    if encoder_params < 0 or agent_params < 0:
        raise ValueError("Parameter counts must be non-negative")

    total_with = encoder_params + num_agents * agent_params
    total_without = num_agents * (encoder_params + agent_params)
    savings_percent = 100.0 * (1 - (total_with / total_without)) if total_without else 0.0

    return total_with, total_without, savings_percent


__all__ = [
    "ParameterReport",
    "SharedEncoderManager",
    "calculate_savings_for_agents",
]
