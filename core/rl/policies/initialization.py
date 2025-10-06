"""Weight Initialization Strategies for RL Trading Agents.

Provides proven initialization methods that preserve variance and gradient
flow based on Glorot & Bengio (2010), He et al. (2015), and Saxe et al. (2014).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


InitStrategy = Literal[
    "xavier_uniform",
    "xavier_normal",
    "he_uniform",
    "he_normal",
    "orthogonal",
]

_LINEAR_OR_CONV = (nn.Linear, nn.Conv1d, nn.Conv2d)
_NORM_LAYERS = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def _zero_bias(module: nn.Module) -> None:
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)  # type: ignore[arg-type]


def xavier_uniform_init(module: nn.Module, gain: float = 1.0) -> None:
    """Apply Xavier/Glorot uniform initialization to linear and conv layers."""

    if isinstance(module, _LINEAR_OR_CONV):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        _zero_bias(module)


def xavier_normal_init(module: nn.Module, gain: float = 1.0) -> None:
    """Apply Xavier/Glorot normal initialization to linear and conv layers."""

    if isinstance(module, _LINEAR_OR_CONV):
        nn.init.xavier_normal_(module.weight, gain=gain)
        _zero_bias(module)


def he_uniform_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    """Apply He/Kaiming uniform initialization tailored for ReLU activations."""

    if isinstance(module, _LINEAR_OR_CONV):
        nn.init.kaiming_uniform_(module.weight, a=0, mode="fan_in", nonlinearity="relu")
        if gain != math.sqrt(2.0):
            scale = gain / math.sqrt(2.0)
            module.weight.data.mul_(scale)
        _zero_bias(module)


def he_normal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    """Apply He/Kaiming normal initialization tailored for ReLU activations."""

    if isinstance(module, _LINEAR_OR_CONV):
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in", nonlinearity="relu")
        if gain != math.sqrt(2.0):
            scale = gain / math.sqrt(2.0)
            module.weight.data.mul_(scale)
        _zero_bias(module)


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    """Apply orthogonal initialization with the specified gain."""

    if isinstance(module, _LINEAR_OR_CONV):
        nn.init.orthogonal_(module.weight, gain=gain)
        _zero_bias(module)


_INIT_FUNCTIONS = {
    "xavier_uniform": xavier_uniform_init,
    "xavier_normal": xavier_normal_init,
    "he_uniform": he_uniform_init,
    "he_normal": he_normal_init,
    "orthogonal": orthogonal_init,
}


def _apply_norm_defaults(module: nn.Module) -> None:
    if isinstance(module, _NORM_LAYERS):
        if getattr(module, "weight", None) is not None:
            nn.init.ones_(module.weight)  # type: ignore[arg-type]
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)  # type: ignore[arg-type]


def init_encoder(encoder: nn.Module, strategy: InitStrategy = "xavier_uniform", gain: float = 1.0) -> None:
    """Initialize transformer encoder modules using the provided strategy."""

    if strategy not in _INIT_FUNCTIONS:
        raise ValueError(f"Unknown initialization strategy: {strategy}")

    init_fn = _INIT_FUNCTIONS[strategy]

    for module in encoder.modules():
        init_fn(module, gain=gain)
        _apply_norm_defaults(module)


def init_actor(
    actor: nn.Module,
    strategy: InitStrategy = "orthogonal",
    gain: float = 0.01,
    hidden_gain: float = math.sqrt(2.0),
) -> None:
    """Initialize an actor head with orthogonal defaults to encourage exploration."""

    if strategy not in _INIT_FUNCTIONS:
        raise ValueError(f"Unknown initialization strategy: {strategy}")

    init_fn = _INIT_FUNCTIONS[strategy]
    linear_layers = [module for module in actor.modules() if isinstance(module, _LINEAR_OR_CONV)]

    for layer in linear_layers[:-1]:
        init_fn(layer, gain=hidden_gain)
    if linear_layers:
        init_fn(linear_layers[-1], gain=gain)

    for module in actor.modules():
        _apply_norm_defaults(module)


def init_critic(
    critic: nn.Module,
    strategy: InitStrategy = "orthogonal",
    gain: float = 1.0,
    hidden_gain: float = math.sqrt(2.0),
) -> None:
    """Initialize a critic head with orthogonal defaults for stable value estimates."""

    if strategy not in _INIT_FUNCTIONS:
        raise ValueError(f"Unknown initialization strategy: {strategy}")

    init_fn = _INIT_FUNCTIONS[strategy]
    linear_layers = [module for module in critic.modules() if isinstance(module, _LINEAR_OR_CONV)]

    for layer in linear_layers[:-1]:
        init_fn(layer, gain=hidden_gain)
    if linear_layers:
        init_fn(linear_layers[-1], gain=gain)

    for module in critic.modules():
        _apply_norm_defaults(module)


def verify_initialization(
    module: nn.Module,
    strategy: InitStrategy,
    tolerance: float = 0.1,
    gain: float = 1.0,
) -> dict:
    """Verify initialization statistics against theoretical expectations."""

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    results = {
        "passed": True,
        "checks": [],
        "mean": 0.0,
        "std": 0.0,
        "has_nan": False,
        "has_inf": False,
    }

    weights = []

    for name, param in module.named_parameters():
        if param.data.numel() == 0 or "weight" not in name:
            continue

        tensor = param.data.detach().float()
        weights.append(tensor.view(-1))

        mean = tensor.mean().item()
        std = tensor.std(unbiased=False).item()

        if tensor.dim() >= 2 and abs(mean) > tolerance:
            results["passed"] = False
            results["checks"].append(f"{name}: mean {mean:.4f} exceeds tolerance {tolerance:.4f}")

        expected_std = None
        if tensor.dim() >= 2:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

            if strategy in {"xavier_uniform", "xavier_normal"}:
                expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
            elif strategy in {"he_uniform", "he_normal"}:
                expected_std = gain / math.sqrt(fan_in)

            if expected_std is not None:
                if expected_std <= 0:
                    raise ValueError("expected standard deviation must be positive")
                if abs(std - expected_std) > tolerance * expected_std:
                    results["passed"] = False
                    results["checks"].append(
                        f"{name}: std {std:.4f} differs from expected {expected_std:.4f}"
                    )

        if strategy == "orthogonal" and tensor.dim() == 2:
            rows, cols = tensor.shape
            qt = tensor @ tensor.t()
            identity = torch.eye(rows, device=tensor.device, dtype=tensor.dtype)
            orthogonality_error = torch.norm(qt - (gain ** 2) * identity).item()
            if orthogonality_error > tolerance:
                results["passed"] = False
                results["checks"].append(
                    f"{name}: orthogonality error {orthogonality_error:.4f} exceeds tolerance {tolerance:.4f}"
                )

        if torch.isnan(tensor).any():
            results["passed"] = False
            results["has_nan"] = True
            results["checks"].append(f"{name}: contains NaN values")
        if torch.isinf(tensor).any():
            results["passed"] = False
            results["has_inf"] = True
            results["checks"].append(f"{name}: contains Inf values")

    if weights:
        concatenated = torch.cat(weights)
        results["mean"] = concatenated.mean().item()
        results["std"] = concatenated.std(unbiased=False).item()
        results["has_nan"] = results["has_nan"] or torch.isnan(concatenated).any().item()
        results["has_inf"] = results["has_inf"] or torch.isinf(concatenated).any().item()

    return results


__all__ = [
    "InitStrategy",
    "xavier_uniform_init",
    "xavier_normal_init",
    "he_uniform_init",
    "he_normal_init",
    "orthogonal_init",
    "init_encoder",
    "init_actor",
    "init_critic",
    "verify_initialization",
]
