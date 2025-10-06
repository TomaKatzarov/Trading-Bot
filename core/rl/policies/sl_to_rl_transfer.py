"""Experimental infrastructure for transferring SL checkpoints to RL encoders.

⚠️  WARNING: Supervised learning models failed backtesting (−88% to −93%).
Use these utilities strictly for controlled A/B experiments and abandon
transfer if RL performance drops by more than 10% after 20k steps.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from .initialization import init_encoder


class SLTransferWarning(UserWarning):
    """Custom warning emitted whenever SL-to-RL transfer helpers are invoked."""


def load_sl_checkpoint(checkpoint_path: Path) -> Dict[str, Optional[dict]]:
    """Load a supervised learning checkpoint with strong cautionary warnings."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SL checkpoint not found: {checkpoint_path}")

    warnings.warn(
        "Loading SL checkpoint. Historical SL models produced −88% to −93% backtesting losses. "
        "Use only for initialization experiments and monitor RL closely.",
        SLTransferWarning,
        stacklevel=2,
    )

    model_path = checkpoint_path / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model state file: {model_path}")

    metadata_path = checkpoint_path / "metadata.json"
    metadata = None
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    model_state = torch.load(model_path, map_location="cpu")

    return {
        "model_state": model_state,
        "metadata": metadata,
    }


def transfer_sl_features_to_encoder(
    encoder: nn.Module,
    sl_checkpoint: Dict[str, Optional[dict]],
    freeze_transferred: bool = False,
) -> int:
    """Attempt to transfer compatible SL weights into the RL encoder."""

    warnings.warn(
        "Executing SL-to-RL transfer. If RL underperforms the random baseline by >10% after 20k steps, abandon this path.",
        SLTransferWarning,
        stacklevel=2,
    )

    sl_state = sl_checkpoint.get("model_state")
    if sl_state is None:
        raise ValueError("Checkpoint does not contain a model_state entry")

    encoder_state = encoder.state_dict()
    transferred_params = 0

    for name, tensor in encoder_state.items():
        if name in sl_state and sl_state[name].shape == tensor.shape:
            encoder_state[name] = sl_state[name].clone()
            transferred_params += tensor.numel()

    encoder.load_state_dict(encoder_state)

    if freeze_transferred:
        for name, param in encoder.named_parameters():
            if name in sl_state and sl_state[name].shape == param.shape:
                param.requires_grad = False

    return transferred_params


def create_sl_transfer_experiment(
    encoder: nn.Module,
    sl_checkpoint_path: Path,
    freeze: bool = False,
) -> Dict[str, object]:
    """Create baseline and SL-transfer encoder variants for A/B testing."""

    warnings.warn(
        "Preparing SL transfer experiment. Historical SL agents were catastrophic; expect the baseline to outperform.",
        SLTransferWarning,
        stacklevel=2,
    )

    checkpoint = load_sl_checkpoint(sl_checkpoint_path)

    encoder_cls = encoder.__class__
    if not hasattr(encoder, "config"):
        raise AttributeError("Encoder must expose a 'config' attribute for re-instantiation")

    baseline_encoder = encoder_cls(encoder.config)
    init_encoder(baseline_encoder, strategy="xavier_uniform", gain=1.0)

    transfer_encoder = encoder_cls(encoder.config)
    init_encoder(transfer_encoder, strategy="xavier_uniform", gain=1.0)
    transferred = transfer_sl_features_to_encoder(transfer_encoder, checkpoint, freeze_transferred=freeze)

    return {
        "baseline_encoder": baseline_encoder,
        "transfer_encoder": transfer_encoder,
        "metadata": {
            "sl_checkpoint": str(sl_checkpoint_path),
            "transferred_params": transferred,
            "freeze": freeze,
            "warning": "SL models failed backtesting (−88% to −93%). Monitor RL closely and abandon if performance lags >10%.",
        },
    }


__all__ = [
    "SLTransferWarning",
    "load_sl_checkpoint",
    "transfer_sl_features_to_encoder",
    "create_sl_transfer_experiment",
]
