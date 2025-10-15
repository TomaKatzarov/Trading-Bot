"""Phase 3 Agent Training Script.

Trains the 10-symbol Phase 3 prototype agents with PPO while tracking
experiments through MLflow and TensorBoard. The script wires together
vectorised trading environments, Stable-Baselines3 PPO, and rich monitoring
callbacks that surface both learning and trading metrics.

Usage
-----
    # Train all 10 agents with baseline configuration
    python training/train_phase3_agents.py --config training/config_templates/phase3_ppo_baseline.yaml

    # Train a subset of symbols
    python training/train_phase3_agents.py --config training/config_templates/phase3_ppo_baseline.yaml --symbols AAPL MSFT

    # Resume a particular symbol from its last best checkpoint
    python training/train_phase3_agents.py --config training/config_templates/phase3_ppo_baseline.yaml --resume AAPL
"""

from __future__ import annotations

# Fix Windows UTF-8 encoding for Rich library (2025-10-08)
import sys
if sys.platform == 'win32':
    import io
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding='utf-8')
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding='utf-8')

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import time
import types
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from functools import wraps
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO, __version__ as SB3_VERSION
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.utils import set_random_seed, update_learning_rate
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from rich.align import Align
from rich.console import Console, Group
from rich.errors import LiveError
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

from core.rl.environments import PortfolioConfig, RewardConfig
from core.rl.environments.trading_env import TradeAction
from core.rl.environments.vec_trading_env import make_vec_trading_env

# --------------------------------------------------------------------------------------
# Global configuration
# --------------------------------------------------------------------------------------
LOGGER = logging.getLogger("training.phase3")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s")

# Improve CuDNN determinism unless explicitly disabled
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

ALL_TRADE_ACTION_NAMES = [action.name for action in TradeAction]


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------

def _compute_file_fingerprint(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {config_path} is not a mapping")
    LOGGER.info("Loaded configuration from %s", config_path.resolve())
    fingerprint = _compute_file_fingerprint(config_path)
    meta = config.setdefault("meta", {})
    meta["config_path"] = str(config_path.resolve())
    meta["fingerprint"] = fingerprint
    LOGGER.info("Configuration fingerprint (sha256): %s", fingerprint)
    return config


def ensure_directory(path: Path) -> Path:
    """Ensure the directory exists and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_device(preference: Optional[str]) -> Tuple[str, Optional[str]]:
    """Resolve device preference into an explicit torch device string."""

    preference_normalized = (preference or "auto").strip().lower()

    def _cuda_device_string(pref: str) -> str:
        if ":" in pref:
            _, index_str = pref.split(":", 1)
            index = int(index_str)
        else:
            index = 0
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("CUDA device requested but no CUDA devices are available.")
        if index >= device_count:
            raise RuntimeError(f"Requested CUDA device index {index} but only {device_count} device(s) present.")
        return f"cuda:{index}"

    if preference_normalized == "auto":
        if torch.cuda.is_available():
            return _cuda_device_string("cuda"), None
        return "cpu", "CUDA not available; using CPU fallback. Install a CUDA-enabled PyTorch build to enable GPU training."

    if preference_normalized == "cpu":
        return "cpu", None

    if preference_normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False. Ensure GPU drivers and CUDA-enabled PyTorch are installed.")
        return _cuda_device_string(preference_normalized), None

    raise ValueError(f"Unsupported device preference: {preference}")


def resolve_activation(name: str) -> Any:
    """Resolve an activation function name into a torch.nn module."""

    name = name.lower()
    if name == "relu":
        return torch.nn.ReLU
    if name == "gelu":
        return torch.nn.GELU
    if name == "elu":
        return torch.nn.ELU
    if name == "tanh":
        return torch.nn.Tanh
    raise ValueError(f"Unsupported activation function: {name}")


def resolve_policy_kwargs(policy_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert YAML friendly policy kwargs into SB3 compatible objects."""

    if not policy_kwargs:
        return {}

    resolved: Dict[str, Any] = dict(policy_kwargs)

    activation_name = resolved.get("activation_fn")
    if isinstance(activation_name, str):
        resolved["activation_fn"] = resolve_activation(activation_name)

    # Ensure net_arch is either list or dict with lists
    net_arch = resolved.get("net_arch")
    if net_arch is not None:
        if isinstance(net_arch, dict):
            resolved["net_arch"] = {
                key: [int(v) for v in values] for key, values in net_arch.items()
            }
        elif isinstance(net_arch, list):
            resolved["net_arch"] = [int(layer) for layer in net_arch]
        else:
            raise ValueError("policy_kwargs.net_arch must be list or dict")

    return resolved


def create_lr_schedule(initial: float, schedule_type: str, lr_min: float) -> Any:
    """Create a learning rate schedule compatible with SB3."""

    schedule_type = (schedule_type or "constant").lower()
    lr_min = max(0.0, float(lr_min))

    if schedule_type == "constant":
        return float(initial)

    initial = float(initial)

    def linear_schedule(progress_remaining: float) -> float:
        return lr_min + (initial - lr_min) * progress_remaining

    if schedule_type == "linear":
        return linear_schedule

    if schedule_type == "cosine":
        def cosine_schedule(progress_remaining: float) -> float:
            progress = 1.0 - progress_remaining
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            value = lr_min + (initial - lr_min) * cosine
            return float(value)

        return cosine_schedule

    raise ValueError(f"Unsupported lr_schedule: {schedule_type}")


def create_entropy_schedule(initial: float, decay: float, minimum: float, *, total_timesteps: int, n_steps: int, n_envs: int) -> Any:
    """Create an entropy coefficient schedule anchored to training progress."""

    if decay <= 0 or math.isclose(decay, 1.0):
        return max(minimum, initial)

    initial = float(initial)
    minimum = float(minimum)
    decay = float(decay)
    updates = max(1, int(math.ceil(total_timesteps / max(1, n_steps * n_envs))))

    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        exponent = progress * updates
        value = initial * (decay ** exponent)
        return float(max(minimum, value))

    return schedule


def create_piecewise_entropy_schedule(
    *,
    initial: float,
    final: float,
    hold_steps: int,
    decay_steps: Optional[int],
    minimum: float,
    total_timesteps: int,
    strategy: str = "hold_then_linear",
) -> Callable[[float], float]:
    """Create a piecewise entropy schedule with an initial hold phase."""

    initial = float(initial)
    final = float(final)
    minimum = float(minimum)
    hold_steps = max(0, int(hold_steps))
    total_timesteps = max(1, int(total_timesteps))
    if decay_steps is None:
        decay_steps = max(1, total_timesteps - hold_steps)
    else:
        decay_steps = max(1, int(decay_steps))

    strategy = (strategy or "hold_then_linear").lower()

    def schedule(progress_remaining: float) -> float:
        progress_remaining = float(max(0.0, min(1.0, progress_remaining)))
        current_step = (1.0 - progress_remaining) * total_timesteps

        if current_step <= hold_steps:
            return initial

        decay_progress = min(1.0, (current_step - hold_steps) / decay_steps)

        if strategy == "hold_then_exponential":
            value = initial * ((final / max(initial, 1e-12)) ** decay_progress)
        else:  # default linear
            value = initial + (final - initial) * decay_progress

        return float(max(minimum, value))

    return schedule


def ensure_schedule_callable(value: Any) -> Callable[[float], float]:
    """Ensure the provided value behaves like an SB3-compatible schedule."""

    if callable(value):
        return value  # type: ignore[return-value]

    constant = float(value)

    def schedule(_: float, constant_value: float = constant) -> float:
        return float(constant_value)

    return schedule


class ScaledSchedule:
    """Wrap a schedule to allow dynamic scaling and enforce a floor."""

    def __init__(self, base_schedule: Callable[[float], float], *, min_value: float = 0.0) -> None:
        self.base_schedule = base_schedule
        self.min_value = float(max(0.0, min_value))
        self.multiplier = 1.0

    def __call__(self, progress_remaining: float) -> float:
        base = float(self.base_schedule(progress_remaining))
        if not np.isfinite(base):
            base = 0.0
        scaled = base * self.multiplier
        return float(max(self.min_value, scaled))

    def base_value(self, progress_remaining: float) -> float:
        base = float(self.base_schedule(progress_remaining))
        if not np.isfinite(base):
            return 0.0
        return base

    def current_value(self, progress_remaining: float) -> float:
        return self(progress_remaining)

    def decay(self, factor: float) -> float:
        factor = float(max(0.0, factor))
        self.multiplier *= factor
        return self.multiplier

    def set_multiplier(self, multiplier: float) -> float:
        self.multiplier = float(max(0.0, multiplier))
        return self.multiplier

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"ScaledSchedule(multiplier={self.multiplier:.6f}, min_value={self.min_value})"


class WarmupSchedule:
    """Apply a linear warmup phase to an existing schedule."""

    def __init__(
        self,
        base_schedule: Callable[[float], float],
        *,
        warmup_steps: int,
        total_timesteps: int,
        floor: float = 0.0,
    ) -> None:
        self.base_schedule = base_schedule
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_timesteps = max(1, int(total_timesteps))
        self.floor = float(max(0.0, floor))

    def __call__(self, progress_remaining: float) -> float:
        base_value = float(self.base_schedule(progress_remaining))
        if self.warmup_steps <= 0:
            return base_value

        progress_remaining = float(max(0.0, min(1.0, progress_remaining)))
        current_step = (1.0 - progress_remaining) * self.total_timesteps
        if current_step <= self.warmup_steps:
            ratio = current_step / max(1, self.warmup_steps)
            warmed = self.floor + (base_value - self.floor) * ratio
            return float(warmed)
        return base_value


class TrainingStabilityGuard:
    """Monitor training metrics and mitigate catastrophic PPO updates."""

    def __init__(
        self,
        *,
        symbol: str,
        lr_schedule: ScaledSchedule,
        total_timesteps: int,
        approx_kl_threshold: float,
        clip_fraction_threshold: float,
        value_loss_threshold: float,
        trigger_patience: int = 1,
        cooldown_steps: int = 5000,
        lr_decay_factor: float = 0.5,
        min_lr: float = 0.0,
        logger_tag: str = "stability",
        max_mitigations: Optional[int] = None,
    ) -> None:
        self.symbol = symbol
        self.lr_schedule = lr_schedule
        self.total_timesteps = max(1, int(total_timesteps))
        self.approx_kl_threshold = float(max(0.0, approx_kl_threshold))
        self.clip_fraction_threshold = float(max(0.0, clip_fraction_threshold))
        self.value_loss_threshold = float(max(0.0, value_loss_threshold))
        self.trigger_patience = max(1, int(trigger_patience))
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.lr_decay_factor = float(np.clip(lr_decay_factor, 1e-6, 1.0))
        self.min_lr = float(max(0.0, min_lr))
        self.logger_tag = logger_tag
        self.max_mitigations = None if max_mitigations in (None, 0) else max(1, int(max_mitigations))

        self._cooldown_until = 0
        self._consecutive_triggers = 0
        self._mitigation_count = 0
        self._halt_training = False

    @property
    def mitigation_count(self) -> int:
        return self._mitigation_count

    @property
    def should_halt(self) -> bool:
        return self._halt_training

    def _resolve_progress(self, model: PPO, num_timesteps: int) -> float:
        progress_remaining = getattr(model, "_current_progress_remaining", None)
        if progress_remaining is None:
            progress_remaining = 1.0 - float(num_timesteps) / float(self.total_timesteps)
        return float(max(0.0, min(1.0, progress_remaining)))

    def sync_optimizer(self, model: PPO, num_timesteps: int) -> None:
        progress = self._resolve_progress(model, num_timesteps)
        lr_value = self.lr_schedule(progress)
        if lr_value < self.min_lr:
            base = self.lr_schedule.base_value(progress)
            if base > 1e-12:
                self.lr_schedule.set_multiplier(self.min_lr / base)
                lr_value = self.lr_schedule(progress)
            else:
                lr_value = self.min_lr
        update_learning_rate(model.policy.optimizer, lr_value)
        if model.logger is not None:
            model.logger.record(f"{self.logger_tag}/lr", float(lr_value))

    def evaluate(self, model: PPO, num_timesteps: int) -> bool:
        if model.logger is None:
            return True

        metrics = dict(model.logger.name_to_value)
        approx_kl = float(metrics.get("train/approx_kl", 0.0))
        clip_fraction = float(metrics.get("train/clip_fraction", 0.0))
        value_loss = float(metrics.get("train/value_loss", 0.0))

        if not np.isfinite(approx_kl):
            approx_kl = float("inf")
        if not np.isfinite(clip_fraction):
            clip_fraction = float("inf")
        if not np.isfinite(value_loss):
            value_loss = float("inf")
        else:
            value_loss = abs(value_loss)

        anomalies: List[Tuple[str, float, float]] = []
        if self.approx_kl_threshold > 0 and approx_kl > self.approx_kl_threshold:
            anomalies.append(("approx_kl", approx_kl, self.approx_kl_threshold))
        if self.clip_fraction_threshold > 0 and clip_fraction > self.clip_fraction_threshold:
            anomalies.append(("clip_fraction", clip_fraction, self.clip_fraction_threshold))
        if self.value_loss_threshold > 0 and value_loss > self.value_loss_threshold:
            anomalies.append(("value_loss", value_loss, self.value_loss_threshold))

        if not anomalies:
            self._consecutive_triggers = 0
            return True

        self._consecutive_triggers += 1
        if self._consecutive_triggers < self.trigger_patience:
            return True

        if num_timesteps < self._cooldown_until:
            return True

        self._consecutive_triggers = 0
        if self.cooldown_steps > 0:
            self._cooldown_until = num_timesteps + self.cooldown_steps

        self._apply_mitigation(model, anomalies, num_timesteps)

        if self.max_mitigations is not None and self._mitigation_count >= self.max_mitigations:
            self._halt_training = True
            LOGGER.warning(
                "[%s] Stability guard reached mitigation limit (%s); halting training",
                self.symbol,
                self._mitigation_count,
            )
            return False

        return True

    def _apply_mitigation(self, model: PPO, anomalies: List[Tuple[str, float, float]], num_timesteps: int) -> None:
        self._mitigation_count += 1
        progress = self._resolve_progress(model, num_timesteps)
        previous_multiplier = self.lr_schedule.multiplier

        if self.lr_decay_factor < 1.0:
            self.lr_schedule.decay(self.lr_decay_factor)

        lr_value = self.lr_schedule(progress)
        if lr_value < self.min_lr:
            base = self.lr_schedule.base_value(progress)
            if base > 1e-12:
                self.lr_schedule.set_multiplier(self.min_lr / base)
                lr_value = self.lr_schedule(progress)
            else:
                lr_value = self.min_lr

        update_learning_rate(model.policy.optimizer, lr_value)

        anomaly_summary = ", ".join(f"{name}={value:.3g} (>{threshold:.3g})" for name, value, threshold in anomalies)
        LOGGER.warning(
            "[%s] Stability guard mitigation #%d at %d steps: %s | lr multiplier %.4f -> %.4f (lr=%.3e)",
            self.symbol,
            self._mitigation_count,
            num_timesteps,
            anomaly_summary,
            previous_multiplier,
            self.lr_schedule.multiplier,
            lr_value,
        )

        if model.logger is not None:
            model.logger.record(f"{self.logger_tag}/event_count", float(self._mitigation_count))
            model.logger.record(f"{self.logger_tag}/lr", float(lr_value))
            model.logger.record(f"{self.logger_tag}/multiplier", float(self.lr_schedule.multiplier))
            for name, value, _threshold in anomalies:
                model.logger.record(f"{self.logger_tag}/{name}", float(value))

        if mlflow.active_run() is not None:
            payload = {f"{self.logger_tag}/{name}": float(value) for name, value, _threshold in anomalies}
            payload[f"{self.logger_tag}/lr"] = float(lr_value)
            payload[f"{self.logger_tag}/event_count"] = float(self._mitigation_count)
            mlflow.log_metrics(payload, step=num_timesteps)


class TrainingHalted(RuntimeError):
    """Raised when the stability guard decides to halt PPO training."""


_TRAIN_GUARD_REGISTRY: Dict[int, TrainingStabilityGuard] = {}
_GUARD_PATCHED = False
_ORIGINAL_PPO_TRAIN: Optional[Callable[..., Any]] = None


def _install_guarded_train() -> None:
    global _GUARD_PATCHED, _ORIGINAL_PPO_TRAIN
    if _GUARD_PATCHED:
        return

    _ORIGINAL_PPO_TRAIN = PPO.train

    @wraps(PPO.train)
    def guarded_train(self: PPO, *args: Any, **kwargs: Any) -> Any:
        result = _ORIGINAL_PPO_TRAIN(self, *args, **kwargs)  # type: ignore[misc]
        guard = _TRAIN_GUARD_REGISTRY.get(id(self))
        if guard is not None:
            should_continue = guard.evaluate(self, self.num_timesteps)
            if not should_continue:
                raise TrainingHalted(f"Training halted by stability guard for {guard.symbol}")
        return result

    PPO.train = guarded_train  # type: ignore[assignment]
    _GUARD_PATCHED = True


def attach_training_guard(model: PPO, guard: Optional[TrainingStabilityGuard]) -> None:
    """Register stability guard for a PPO instance."""

    if guard is None:
        return

    _install_guarded_train()
    _TRAIN_GUARD_REGISTRY[id(model)] = guard
    guard.sync_optimizer(model, model.num_timesteps)


def detach_training_guard(model: PPO) -> None:
    _TRAIN_GUARD_REGISTRY.pop(id(model), None)


def _debug_pickle_failures(model: PPO) -> None:
    """Log unpicklable attributes to aid debugging."""

    try:
        import cloudpickle  # type: ignore import
    except Exception:  # pragma: no cover - best effort debugging aid
        LOGGER.exception("cloudpickle unavailable while debugging pickle failure")
        return

    excluded = set(model._excluded_save_params())  # type: ignore[attr-defined]
    problem_keys: List[str] = []
    for key, value in model.__dict__.items():
        if key in excluded:
            continue
        try:
            cloudpickle.dumps(value)
        except TypeError as err:
            LOGGER.error("Unpicklable attribute '%s': %s (%s)", key, err, type(value))
            print(f"[pickle-debug] {key}: {err} ({type(value)})")
            problem_keys.append(key)
        except Exception as err:  # pragma: no cover - diagnostic only
            LOGGER.error("Error while pickling attribute '%s': %s", key, err)
            print(f"[pickle-debug] {key}: {err} ({type(value)})")
            problem_keys.append(key)

    if not problem_keys:
        LOGGER.error("Pickle debugging found no problematic attributes; inspect external dependencies")
def build_reward_config(weights: Dict[str, float], base: Optional[RewardConfig] = None) -> RewardConfig:
    """Convert reward weight dictionary into a RewardConfig instance."""

    config = RewardConfig() if base is None else base

    primary_mapping = {
        "pnl_weight": weights.get("pnl", config.pnl_weight),
        "transaction_cost_weight": weights.get("cost", config.transaction_cost_weight),
        "time_efficiency_weight": weights.get("time", config.time_efficiency_weight),
        "sharpe_weight": weights.get("sharpe", config.sharpe_weight),
        "drawdown_weight": weights.get("drawdown", config.drawdown_weight),
        "sizing_weight": weights.get("sizing", config.sizing_weight),
        "hold_penalty_weight": weights.get("hold", config.hold_penalty_weight),
        "diversity_bonus_weight": weights.get("diversity_bonus", config.diversity_bonus_weight),  # NEW (2025-10-08)
    "action_repeat_penalty_weight": weights.get("action_repeat_penalty", config.action_repeat_penalty_weight),
        "base_transaction_cost_pct": weights.get("transaction_cost_pct", config.base_transaction_cost_pct),
    }

    for attr, value in primary_mapping.items():
        setattr(config, attr, float(value))

    extended_mapping = {
        "pnl_scale": weights.get("pnl_scale"),
        "target_sharpe": weights.get("target_sharpe"),
        "max_drawdown_threshold": weights.get("max_drawdown_threshold"),
        "win_bonus_multiplier": weights.get("win_bonus_multiplier"),
        "loss_penalty_multiplier": weights.get("loss_penalty_multiplier"),
        "quick_win_bonus": weights.get("quick_win_bonus"),
        "early_stop_bonus": weights.get("early_stop_bonus"),
        "manual_exit_bonus": weights.get("manual_exit_bonus"),
        "forced_exit_penalty": weights.get("forced_exit_penalty"),
    "intrinsic_action_reward": weights.get("intrinsic_action_reward"),
        "failed_action_penalty": weights.get("failed_action_penalty"),
        "reward_clip": weights.get("reward_clip"),
        "severe_loss_penalty": weights.get("severe_loss_penalty"),
        "max_single_loss": weights.get("max_single_loss"),
        "min_trades_for_sharpe": weights.get("min_trades_for_sharpe"),
        "neutral_exposure_pct": weights.get("neutral_exposure_pct"),
        "sizing_optimal_low": weights.get("sizing_optimal_low"),
        "sizing_optimal_high": weights.get("sizing_optimal_high"),
        "sizing_positive_bonus": weights.get("sizing_positive_bonus"),
        "sizing_moderate_bonus": weights.get("sizing_moderate_bonus"),
        "sizing_penalty_high": weights.get("sizing_penalty_high"),
        "roi_multiplier_enabled": weights.get("roi_multiplier_enabled"),  # NEW (2025-10-08)
        "roi_scale_factor": weights.get("roi_scale_factor"),              # NEW (2025-10-08)
    "roi_gate_floor_scale": weights.get("roi_gate_floor_scale"),      # Stage 2 (2025-10-10)
        "realized_pnl_weight": weights.get("realized_pnl_weight"),        # NEW (2025-10-08 FIX)
        "unrealized_pnl_weight": weights.get("unrealized_pnl_weight"),    # NEW (2025-10-08 FIX)
        "closing_bonus_multiplier": weights.get("closing_bonus_multiplier"),  # NEW (2025-10-08 FIX)
        # V3.1: Position sizing and exit strategy (2025-10-08)
        "position_size_small_multiplier": weights.get("position_size_small_multiplier"),
        "position_size_medium_multiplier": weights.get("position_size_medium_multiplier"),
        "position_size_large_multiplier": weights.get("position_size_large_multiplier"),
        "partial_exit_multiplier": weights.get("partial_exit_multiplier"),
        "full_exit_multiplier": weights.get("full_exit_multiplier"),
        "staged_exit_bonus": weights.get("staged_exit_bonus"),
        # V3.1: ADD_POSITION pyramiding (2025-10-08)
        "add_position_enabled": weights.get("add_position_enabled"),
        "add_position_min_profit_pct": weights.get("add_position_min_profit_pct"),
        "add_position_confidence_threshold": weights.get("add_position_confidence_threshold"),
        "add_position_immediate_reward": weights.get("add_position_immediate_reward"),
        "add_position_pyramid_bonus": weights.get("add_position_pyramid_bonus"),
        "add_position_max_adds": weights.get("add_position_max_adds"),
        "add_position_invalid_penalty": weights.get("add_position_invalid_penalty"),
    # Stage 2 reward hygiene (2025-10-10)
    "forced_exit_base_penalty": weights.get("forced_exit_base_penalty"),
    "forced_exit_loss_scale": weights.get("forced_exit_loss_scale"),
    "forced_exit_penalty_cap": weights.get("forced_exit_penalty_cap"),
    "sharpe_gate_enabled": weights.get("sharpe_gate_enabled"),
    "sharpe_gate_window": weights.get("sharpe_gate_window"),
    "sharpe_gate_min_self_trades": weights.get("sharpe_gate_min_self_trades"),
    "sharpe_gate_floor_scale": weights.get("sharpe_gate_floor_scale"),
    "sharpe_gate_active_scale": weights.get("sharpe_gate_active_scale"),
    "time_decay_threshold_hours": weights.get("time_decay_threshold_hours"),
    "time_decay_penalty_per_hour": weights.get("time_decay_penalty_per_hour"),
    "time_decay_max_penalty": weights.get("time_decay_max_penalty"),
    }

    int_attrs = {"min_trades_for_sharpe", "add_position_max_adds", "sharpe_gate_window", "sharpe_gate_min_self_trades"}
    bool_attrs = {"roi_multiplier_enabled", "add_position_enabled", "sharpe_gate_enabled"}  # NEW (2025-10-08)

    for attr, value in extended_mapping.items():
        if value is None:
            continue
        if attr in int_attrs:
            setattr(config, attr, int(value))
        elif attr in bool_attrs:
            setattr(config, attr, bool(value))
        else:
            setattr(config, attr, float(value))

    return config


def validate_anti_collapse_config(reward_config: RewardConfig, symbol: str) -> None:
    """
    Validate that anti-collapse improvements are properly configured (2025-10-08).
    
    V10 UPDATE: Relaxed validation - epsilon-greedy exploration replaces diversity bonus.
    
    Args:
        reward_config: The reward configuration to validate
        symbol: Symbol being trained (for logging)
    
    Raises:
        AssertionError: If any critical anti-collapse parameter is missing or invalid
    """
    errors = []
    
    # V10: diversity_bonus is now OPTIONAL (epsilon-greedy handles exploration)
    if not hasattr(reward_config, 'diversity_bonus_weight'):
        errors.append("diversity_bonus_weight attribute missing from RewardConfig")
    # Allow diversity_bonus = 0 if using epsilon-greedy
    
    # Check ROI multiplier enabled
    if not hasattr(reward_config, 'roi_multiplier_enabled'):
        errors.append("roi_multiplier_enabled attribute missing from RewardConfig")
    elif reward_config.roi_multiplier_enabled != True:
        errors.append(f"roi_multiplier_enabled must be True! Got: {reward_config.roi_multiplier_enabled}")
    
    # Check ROI scale factor
    if not hasattr(reward_config, 'roi_scale_factor'):
        errors.append("roi_scale_factor attribute missing from RewardConfig")
    elif reward_config.roi_scale_factor <= 0:
        errors.append(f"roi_scale_factor must be >0! Got: {reward_config.roi_scale_factor}")
    
    if errors:
        LOGGER.error(f"❌ ANTI-COLLAPSE CONFIG VALIDATION FAILED for {symbol}:")
        for error in errors:
            LOGGER.error(f"   - {error}")
        LOGGER.error("")
        LOGGER.error("This indicates anti-collapse improvements are NOT properly loaded!")
        LOGGER.error("See docs/anti_collapse_improvements_2025-10-08.md for details.")
        LOGGER.error("")
        raise AssertionError(f"Anti-collapse validation failed: {'; '.join(errors)}")
    
    # Log successful validation
    LOGGER.info(f"✅ Anti-collapse config validated for {symbol}:")
    LOGGER.info(f"   - diversity_bonus_weight: {reward_config.diversity_bonus_weight}")
    LOGGER.info(f"   - roi_multiplier_enabled: {reward_config.roi_multiplier_enabled}")
    LOGGER.info(f"   - roi_scale_factor: {reward_config.roi_scale_factor}")
    
    # Log realized PnL fix (2025-10-08)
    if hasattr(reward_config, 'realized_pnl_weight'):
        LOGGER.info(f"✅ Realized PnL fix ENABLED:")
        LOGGER.info(f"   - realized_pnl_weight: {reward_config.realized_pnl_weight}")
        LOGGER.info(f"   - unrealized_pnl_weight: {reward_config.unrealized_pnl_weight}")
        LOGGER.info(f"   - closing_bonus_multiplier: {reward_config.closing_bonus_multiplier}")


def build_portfolio_config(env_cfg: Dict[str, Any]) -> PortfolioConfig:
    """Create a PortfolioConfig from environment configuration."""

    portfolio = PortfolioConfig(
        initial_capital=float(env_cfg.get("initial_capital", 100_000.0)),
        commission_rate=float(env_cfg.get("commission_rate", 0.001)),
        slippage_bps=float(env_cfg.get("slippage_pct", 0.0005)) * 10_000.0,
        max_position_size_pct=float(env_cfg.get("max_position_pct", 0.2)),
        max_total_exposure_pct=float(env_cfg.get("max_portfolio_exposure", 0.9)),
        max_positions=1,
        max_position_loss_pct=float(env_cfg.get("max_position_loss", 0.08)),
        max_portfolio_loss_pct=float(env_cfg.get("max_portfolio_drawdown", 0.30)),
    )
    portfolio.validate()
    return portfolio


def make_env_kwargs(symbol: str, split: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Construct keyword arguments for TradingEnvironment instantiation."""

    env_cfg = config.get("environment", {})
    data_dir = Path(config["experiment"]["data_dir"]) / symbol
    data_path = data_dir / f"{split}.parquet"

    reward_config = build_reward_config(env_cfg.get("reward_weights", {}))
    
    # Validate anti-collapse improvements are properly configured (2025-10-08)
    validate_anti_collapse_config(reward_config, symbol)
    
    portfolio_config = build_portfolio_config(env_cfg)

    # CRITICAL FIX (2025-10-08): Properly handle None/null for stop_loss and take_profit
    # BUG: `None or 0.02` evaluates to 0.02, enabling autopilot exits when config says disabled!
    # FIX: Use explicit None check, set to 999.0 to effectively disable (no position hits +99,900%)
    stop_loss_cfg = env_cfg.get("stop_loss_pct")
    take_profit_cfg = env_cfg.get("take_profit_pct")
    
    stop_loss_value = float(stop_loss_cfg) if stop_loss_cfg is not None else 999.0
    take_profit_value = float(take_profit_cfg) if take_profit_cfg is not None else 999.0
    
    # Log warnings when automatic exits are disabled (agent must learn to close positions)
    if stop_loss_cfg is None:
        LOGGER.warning(f"⚠️  [{symbol}] stop_loss DISABLED (null in config) - agent must learn to close positions")
    if take_profit_cfg is None:
        LOGGER.warning(f"⚠️  [{symbol}] take_profit DISABLED (null in config) - agent must learn to close positions")

    env_kwargs: Dict[str, Any] = {
        "symbol": symbol,
        "data_path": data_path,
        "sl_checkpoints": env_cfg.get("sl_checkpoints", {}),
        "sl_inference_device": env_cfg.get("sl_inference_device"),
        "lookback_window": int(env_cfg.get("lookback_window", 24)),
        "episode_length": int(env_cfg.get("episode_length", 168)),
        "initial_capital": float(env_cfg.get("initial_capital", 100_000.0)),
        "commission_rate": float(env_cfg.get("commission_rate", 0.001)),
        "slippage_bps": float(env_cfg.get("slippage_pct", 0.0005)) * 10_000.0,
        "stop_loss": stop_loss_value,
        "take_profit": take_profit_value,
        "max_hold_hours": int(env_cfg.get("max_hold_hours", 8)),
        "reward_config": reward_config,
        "portfolio_config": portfolio_config,
        "log_trades": bool(env_cfg.get("log_trades", False)),
    }

    continuous_settings = env_cfg.get("continuous_settings")
    if continuous_settings is not None:
        env_kwargs["continuous_settings"] = dict(continuous_settings)

    action_mode = env_cfg.get("action_mode")
    if action_mode is not None:
        env_kwargs["action_mode"] = str(action_mode)

    # Exploration curriculum (2025-10-08 Anti-Collapse v4)
    curriculum_cfg = env_cfg.get("exploration_curriculum", {})
    LOGGER.info(f"Curriculum config retrieved: {curriculum_cfg}")
    if curriculum_cfg.get("enabled", False):
        LOGGER.info("Exploration curriculum ENABLED in training config")
        env_kwargs["exploration_curriculum_enabled"] = True
        env_kwargs["exploration_phase1_end_step"] = int(curriculum_cfg.get("phase1_end_step", 20000))
        env_kwargs["exploration_phase1_min_action_pct"] = float(curriculum_cfg.get("phase1_min_action_pct", 0.10))
        env_kwargs["exploration_phase1_penalty"] = float(curriculum_cfg.get("phase1_penalty", -5.0))
        env_kwargs["exploration_phase2_end_step"] = int(curriculum_cfg.get("phase2_end_step", 50000))
        env_kwargs["exploration_phase2_min_action_pct"] = float(curriculum_cfg.get("phase2_min_action_pct", 0.05))
        env_kwargs["exploration_phase2_penalty"] = float(curriculum_cfg.get("phase2_penalty", -2.0))
        env_kwargs["exploration_evaluation_window"] = int(curriculum_cfg.get("evaluation_window", 100))
        env_kwargs["exploration_excluded_actions"] = curriculum_cfg.get("excluded_actions", ["HOLD"])
        # V9: SELL-specific enforcement
        env_kwargs["exploration_require_sell_actions"] = bool(curriculum_cfg.get("require_sell_actions", False))
        env_kwargs["exploration_min_sell_pct"] = float(curriculum_cfg.get("min_sell_pct", 0.05))
        env_kwargs["exploration_sell_penalty_multiplier"] = float(curriculum_cfg.get("sell_penalty_multiplier", 5.0))
        env_kwargs["exploration_require_buy_actions"] = bool(curriculum_cfg.get("require_buy_actions", False))
        env_kwargs["exploration_min_buy_pct"] = float(curriculum_cfg.get("min_buy_pct", 0.05))
        env_kwargs["exploration_buy_penalty_multiplier"] = float(curriculum_cfg.get("buy_penalty_multiplier", 3.0))
    else:
        LOGGER.info(f"Exploration curriculum DISABLED (enabled={curriculum_cfg.get('enabled', False)})")

    # Epsilon-Greedy Exploration (2025-10-08 v10)
    epsilon_cfg = env_cfg.get("epsilon_greedy", {})
    if epsilon_cfg.get("enabled", False):
        LOGGER.info("Epsilon-greedy exploration ENABLED")
        env_kwargs["epsilon_greedy_enabled"] = True
        env_kwargs["epsilon_start"] = float(epsilon_cfg.get("epsilon_start", 0.5))
        env_kwargs["epsilon_end"] = float(epsilon_cfg.get("epsilon_end", 0.01))
        env_kwargs["epsilon_decay_steps"] = int(epsilon_cfg.get("epsilon_decay_steps", 50000))
        LOGGER.info(f"  ε decay: {epsilon_cfg.get('epsilon_start', 0.5):.2f} → {epsilon_cfg.get('epsilon_end', 0.01):.2f} over {epsilon_cfg.get('epsilon_decay_steps', 50000)} steps")
    else:
        LOGGER.info("Epsilon-greedy exploration DISABLED")
    
    # Action Restrictions (2025-10-08 v10 RADICAL FIX)
    disabled_actions = env_cfg.get("disabled_actions", [])
    if disabled_actions:
        env_kwargs["disabled_actions"] = disabled_actions
        LOGGER.info(f"⛔ Disabled actions: {disabled_actions}")

    add_gate_cfg = env_cfg.get("add_position_gate", {})
    if add_gate_cfg.get("enabled", False):
        env_kwargs["add_position_gate_enabled"] = True
        env_kwargs["add_position_gate_max_exposure_pct"] = float(add_gate_cfg.get("max_exposure_pct", 0.12))
        env_kwargs["add_position_gate_min_unrealized_pct"] = float(add_gate_cfg.get("min_unrealized_pnl_pct", 0.0))
        env_kwargs["add_position_gate_base_penalty"] = float(add_gate_cfg.get("base_penalty", 0.25))
        env_kwargs["add_position_gate_severity_multiplier"] = float(add_gate_cfg.get("severity_multiplier", 0.5))
        env_kwargs["add_position_gate_penalty_cap"] = float(add_gate_cfg.get("penalty_cap", 1.2))
        env_kwargs["add_position_gate_violation_decay"] = int(add_gate_cfg.get("violation_decay_steps", 2000))
        LOGGER.info(
            "ADD_POSITION gate enabled (max_exposure=%.2f%%, min_unrealized=%.2f%%, base_penalty=%.3f)",
            env_kwargs["add_position_gate_max_exposure_pct"] * 100,
            env_kwargs["add_position_gate_min_unrealized_pct"] * 100,
            env_kwargs["add_position_gate_base_penalty"],
        )

    log_level = env_cfg.get("log_level")
    if isinstance(log_level, str):
        env_kwargs["log_level"] = getattr(logging, log_level.upper(), logging.WARNING)
    elif isinstance(log_level, int):
        env_kwargs["log_level"] = log_level

    return env_kwargs


# --------------------------------------------------------------------------------------
# Rich status monitor utilities
# --------------------------------------------------------------------------------------


def _format_float(value: float, precision: int = 2, *, allow_signed: bool = True) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    fmt = f"{{:{'+' if allow_signed else ''}.{precision}f}}"
    return fmt.format(float(value))


def _format_duration(seconds: float) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "--"
    seconds_int = int(seconds)
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 999:
        hours = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _text_kv(label: str, value: str, *, value_style: str = "bold") -> Text:
    text = Text()
    text.append(f"{label}: ", style="dim")
    text.append(value, style=value_style)
    return text


def _text_metrics_row(pairs: Sequence[Tuple[str, str]]) -> Text:
    text = Text()
    for index, (label, value) in enumerate(pairs):
        if index:
            text.append(" | ", style="dim")
        text.append(f"{label}: ", style="dim")
        text.append(value, style="bold")
    return text


class RichTrainingMonitor:
    """Interactive Rich-based status board for Phase 3 training."""

    def __init__(
        self,
        symbol: str,
        total_timesteps: int,
        *,
        console: Optional[Console] = None,
        refresh_per_second: int = 4,
        device: Optional[str] = None,
        system_info: Optional[Sequence[str]] = None,
    ) -> None:
        self.symbol = symbol
        self.total_timesteps = max(1, int(total_timesteps))
        self.console = console or Console(highlight=False)
        self.refresh_per_second = max(1, int(refresh_per_second))
        disable_live = os.environ.get("PHASE3_DISABLE_RICH") or os.environ.get("RICH_NO_LIVE")
        if disable_live is None:
            disable_live = os.environ.get("CI")
        console_interactive = getattr(self.console, "is_terminal", False) and getattr(self.console, "is_interactive", False)
        self.enabled = console_interactive and not (
            isinstance(disable_live, str) and disable_live.strip().lower() in {"1", "true", "yes", "on"}
        )
        self.status: str = "Initializing"
        self.current_step: int = 0
        self.metrics: Dict[str, Any] = {
            "device": device or "--",
            "train_reward_mean": float("nan"),
            "train_reward_std": float("nan"),
            "train_episode_length": float("nan"),
            "eval_reward_mean": float("nan"),
            "eval_reward_std": float("nan"),
            "eval_episode_length": float("nan"),
            "fps": float("nan"),
            "eta_seconds": float("nan"),
            "lr": float("nan"),
            "entropy_coef": float("nan"),
            "best_sharpe": float("nan"),
            "last_eval_sharpe": float("nan"),
            "eval_return_pct": float("nan"),
            "last_eval_step": None,
        }
        if system_info is None:
            start_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            system_info = [
                f"Run start: {start_label}",
                f"Host: {platform.node()} ({platform.system()} {platform.release()})",
                f"Python {platform.python_version()} • Torch {torch.__version__} • SB3 {SB3_VERSION}",
            ]
            if device:
                system_info.append(f"Device: {device}")
        self.system_info_messages: List[str] = [str(msg).strip() for msg in (system_info or []) if str(msg).strip()]
        self._start_time = time.perf_counter()
        self._paused_since = self._start_time
        self._total_paused = 0.0
        self._last_render = 0.0
        self._initial_logged = False
        self._final_logged = False
        self.progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("[dim]{task.fields[fps]} FPS", justify="right"),
            console=self.console,
            transient=False,
            refresh_per_second=self.refresh_per_second,
        )
        self._task_id = self.progress.add_task(self._progress_description(), total=self.total_timesteps)
        self.live: Optional[Live] = None

    def __enter__(self) -> "RichTrainingMonitor":
        if not self.enabled:
            self._emit_initial_info()
            return self
        try:
            live = Live(
                self.render(),
                console=self.console,
                refresh_per_second=self.refresh_per_second,
                transient=False,
            )
            self.live = live.__enter__()
        except LiveError as exc:
            self.enabled = False
            self.live = None
            LOGGER.warning("Rich live monitor disabled; falling back to log-only updates: %s", exc)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:  # noqa: ANN001
        if not self.enabled:
            return
        try:
            if exc_type is not None:
                self.status = "Error"
            self.refresh(force=True)
        finally:
            if self.live is not None:
                # Ensure Live properly restores console state even when exceptions occur.
                self.live.__exit__(exc_type, exc, exc_tb)
                self.live = None
        self.resume_timer()
        merged = dict(self.metrics)
        merged.setdefault("status", self.status)
        self._emit_summary(merged)

    def _progress_description(self) -> str:
        return f"{self.symbol} • {self.status}"

    def _emit_initial_info(self) -> None:
        if self._initial_logged or not self.system_info_messages:
            return
        info_text = Text()
        for index, message in enumerate(self.system_info_messages):
            info_text.append("• ", style="dim")
            info_text.append(message, style="cyan")
            if index < len(self.system_info_messages) - 1:
                info_text.append("\n")
        panel = Panel(Align.left(info_text), border_style="bright_blue", title="[bold cyan]System info[/]")
        self.console.print(panel)
        self._initial_logged = True

    def _emit_summary(self, final_metrics: Mapping[str, Any]) -> None:
        if self._final_logged:
            return
        elapsed = self.training_elapsed()
        summary_pairs = [
            ("Status", str(final_metrics.get("status", self.status))),
            ("Best Sharpe", _format_float(final_metrics.get("best_sharpe"), precision=3)),
            ("Eval Sharpe", _format_float(final_metrics.get("last_eval_sharpe"), precision=3)),
            ("Eval Return %", _format_float(final_metrics.get("eval_return_pct"), precision=2)),
            ("Eval Reward μ", _format_float(final_metrics.get("eval_reward_mean"))),
            ("Eval Reward σ", _format_float(final_metrics.get("eval_reward_std"))),
            ("Eval Episode μ", _format_float(final_metrics.get("eval_episode_length"))),
            ("FPS", _format_float(final_metrics.get("fps"), precision=0, allow_signed=False)),
            ("Elapsed", _format_duration(elapsed)),
            ("ETA", _format_duration(final_metrics.get("eta_seconds"))),
        ]

        text = Text()
        for index, (label, value) in enumerate(summary_pairs):
            text.append(f"{label}: ", style="dim")
            text.append(value, style="bold")
            if index < len(summary_pairs) - 1:
                text.append("\n")

        panel = Panel(Align.left(text), border_style="bright_magenta", title=f"[bold bright_white]Phase 3 Summary • {self.symbol}[/]")
        self.console.print(panel)
        self._final_logged = True

    def pause_timer(self) -> None:
        if self._paused_since is None:
            self._paused_since = time.perf_counter()

    def resume_timer(self) -> None:
        if self._paused_since is not None:
            self._total_paused += time.perf_counter() - self._paused_since
            self._paused_since = None

    def training_elapsed(self) -> float:
        now = time.perf_counter()
        paused = self._total_paused
        if self._paused_since is not None:
            paused += now - self._paused_since
        return max(0.0, now - self._start_time - paused)

    def _build_system_info_panel(self) -> Optional[Panel]:
        if not self.system_info_messages:
            return None

        info_text = Text()
        for index, message in enumerate(self.system_info_messages):
            info_text.append("• ", style="dim")
            info_text.append(message, style="cyan")
            if index < len(self.system_info_messages) - 1:
                info_text.append("\n")

        return Panel(Align.left(info_text), border_style="bright_blue", title="[bold cyan]System info[/]")

    def set_system_info(self, messages: Sequence[str]) -> None:
        self.system_info_messages = [str(msg).strip() for msg in messages if str(msg).strip()]
        self.refresh(force=True)

    def render(self) -> Panel:
        if self.enabled:
            fps_for_task = _format_float(self.metrics.get("fps"), precision=0, allow_signed=False)
            elapsed_for_task = _format_duration(self.training_elapsed())
            eta_for_task = _format_duration(self.metrics.get("eta_seconds"))
            self.progress.update(
                self._task_id,
                completed=self.current_step,
                description=self._progress_description(),
                fps=fps_for_task,
                elapsed=elapsed_for_task,
                eta=eta_for_task,
            )

        renderables: List[Any] = []
        system_panel = self._build_system_info_panel()
        if system_panel is not None:
            renderables.append(system_panel)

        info_table = Table.grid(expand=True)
        info_table.add_column(ratio=1, justify="left")
        info_table.add_column(ratio=1, justify="right")

        elapsed = self.training_elapsed()
        fps = self.metrics.get("fps")
        eta_value = self.metrics.get("eta_seconds")
        if not (isinstance(eta_value, (int, float)) and np.isfinite(float(eta_value)) and float(eta_value) >= 0.0):
            eta_value = None
            if np.isfinite(fps) and fps and fps > 0:
                remaining = max(0.0, self.total_timesteps - self.current_step)
                eta_value = remaining / fps

        def _format_step_value(step: Any) -> str:
            if isinstance(step, (int, np.integer)):
                return f"{int(step):,}"
            if isinstance(step, float) and np.isfinite(step):
                return f"{int(step):,}"
            return "--"

        task = self.progress.tasks[self._task_id]
        task_speed = task.speed if task.speed is not None else float("nan")
        task_eta = task.time_remaining if task.time_remaining is not None else None

        if (not np.isfinite(self.metrics.get("fps", float("nan")))) and np.isfinite(task_speed):
            self.metrics["fps"] = float(task_speed)
        if (eta_value is None or not np.isfinite(float(eta_value))) and task_eta is not None:
            eta_value = float(task_eta)
        if eta_value is not None and np.isfinite(float(eta_value)):
            self.metrics["eta_seconds"] = float(eta_value)

        info_table.add_row(
            _text_kv("Status", str(self.status)),
            _text_kv("Device", str(self.metrics.get("device", "--"))),
        )
        info_table.add_row(
            _text_kv(
                "Train Reward μ / σ",
                f"{_format_float(self.metrics.get('train_reward_mean'))} / {_format_float(self.metrics.get('train_reward_std'))}",
            ),
            _text_kv("Train Episode μ", _format_float(self.metrics.get("train_episode_length"))),
        )
        info_table.add_row(
            _text_kv(
                "Eval Reward μ / σ",
                f"{_format_float(self.metrics.get('eval_reward_mean'))} / {_format_float(self.metrics.get('eval_reward_std'))}",
            ),
            _text_kv("Eval Episode μ", _format_float(self.metrics.get("eval_episode_length"))),
        )
        info_table.add_row(
            _text_kv("Best Sharpe", _format_float(self.metrics.get("best_sharpe"), precision=3)),
            _text_kv("Last Eval Sharpe", _format_float(self.metrics.get("last_eval_sharpe"), precision=3)),
        )
        info_table.add_row(
            _text_kv("Eval Return %", _format_float(self.metrics.get("eval_return_pct"), precision=2)),
            _text_kv("Last Eval Step", _format_step_value(self.metrics.get("last_eval_step"))),
        )

        lr_value = self.metrics.get("lr")
        lr_text = "--"
        if isinstance(lr_value, (int, float)) and np.isfinite(lr_value):
            lr_text = f"{float(lr_value):.2e}"

        info_table.add_row(
            _text_kv("LR", lr_text),
            _text_kv("Entropy coef", _format_float(self.metrics.get("entropy_coef"), precision=4, allow_signed=False)),
        )
        info_table.add_row(
            _text_metrics_row(
                [
                    ("FPS", _format_float(self.metrics.get("fps"), precision=0, allow_signed=False)),
                    ("Elapsed", _format_duration(elapsed)),
                    ("ETA", _format_duration(eta_value)),
                ]
            ),
            Text(),
        )

        renderables.append(self.progress)
        renderables.append(info_table)

        content = Group(*renderables)
        title = f"[bold bright_white]Phase 3 Training • {self.symbol}[/]"
        return Panel(content, border_style="bright_magenta", title=title)

    def update(self, *, step: Optional[int] = None, status: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None) -> None:
        if step is not None:
            self.current_step = max(0, min(int(step), self.total_timesteps))
        if status is not None:
            self.status = status
        if metrics:
            self.metrics.update(metrics)
        self.refresh()

    def update_evaluation(
        self,
        *,
        summary: Optional[Dict[str, float]] = None,
        best_metric: Optional[float] = None,
        step: Optional[int] = None,
    ) -> None:
        metrics: Dict[str, Any] = {}
        if summary:
            metrics["last_eval_sharpe"] = summary.get("sharpe_ratio_mean")
            metrics["eval_return_pct"] = summary.get("total_return_pct_mean")
            metrics["eval_reward_mean"] = summary.get("episode_reward_mean")
            metrics["eval_reward_std"] = summary.get("episode_reward_std")
            metrics["eval_episode_length"] = summary.get("episode_length_mean")
        if best_metric is not None and np.isfinite(best_metric):
            metrics["best_sharpe"] = best_metric
        if step is not None:
            metrics["last_eval_step"] = int(step)
        if metrics:
            self.update(metrics=metrics)

    def complete(self, final_status: str = "Complete", metrics: Optional[Dict[str, Any]] = None) -> None:
        final_metrics = metrics or {}
        final_metrics.setdefault("eta_seconds", 0.0)
        self.update(status=final_status, step=self.total_timesteps, metrics=final_metrics)
        if not self.enabled:
            self.resume_timer()
            merged = dict(self.metrics)
            merged["status"] = final_status
            self._emit_summary(merged)

    def refresh(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if not force and now - self._last_render < 1.0 / self.refresh_per_second:
            return
        self._last_render = now
        if self.live is not None:
            self.live.update(self.render())


class RichStatusCallback(BaseCallback):
    """Callback that feeds live training metrics into the Rich status monitor."""

    def __init__(
        self,
        monitor: RichTrainingMonitor,
        *,
        total_timesteps: int,
        refresh_steps: int = 512,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.total_timesteps = max(1, int(total_timesteps))
        self.refresh_steps = max(1, int(refresh_steps))
        self._last_update_step = 0
        self._last_update_time = monitor.training_elapsed()
        self._smoothed_fps: float = float("nan")

    def _on_training_start(self) -> None:
        self.monitor.resume_timer()
        self._last_update_time = self.monitor.training_elapsed()
        self._smoothed_fps = float("nan")
        if self.monitor.enabled:
            device = getattr(self.model, "device", None)
            self.monitor.update(metrics={"device": str(device) if device is not None else "--"}, status="Training")

    def _on_step(self) -> bool:
        if not self.monitor.enabled:
            return True
        if self.num_timesteps - self._last_update_step < self.refresh_steps and self.num_timesteps < self.total_timesteps:
            return True
        current_training_elapsed = self.monitor.training_elapsed()
        delta_time = max(0.0, current_training_elapsed - self._last_update_time)
        delta_steps = max(0, self.num_timesteps - self._last_update_step)
        if delta_steps <= 0:
            delta_steps = max(0, self.num_timesteps - self.monitor.current_step)
        instant_fps = float(delta_steps / delta_time) if delta_time > 0 else float("nan")
        if np.isfinite(instant_fps):
            if np.isfinite(self._smoothed_fps):
                fps = (self._smoothed_fps * 0.7) + (instant_fps * 0.3)
            else:
                fps = instant_fps
            self._smoothed_fps = fps
        else:
            fps = self._smoothed_fps if np.isfinite(self._smoothed_fps) else float("nan")
        self._last_update_step = self.num_timesteps
        self._last_update_time = current_training_elapsed

        remaining = max(0, self.total_timesteps - self.num_timesteps)
        eta_seconds = float(remaining / fps) if np.isfinite(fps) and fps > 1e-9 else float("nan")

        ep_rewards = [info.get("r", 0.0) for info in self.model.ep_info_buffer][-10:]
        ep_lengths = [info.get("l", 0.0) for info in self.model.ep_info_buffer][-10:]

        reward_mean = float(np.mean(ep_rewards)) if ep_rewards else float("nan")
        reward_std = float(np.std(ep_rewards)) if ep_rewards else float("nan")
        episode_length_mean = float(np.mean(ep_lengths)) if ep_lengths else float("nan")

        lr = float(self.model.policy.optimizer.param_groups[0].get("lr", float("nan")))
        ent_coef = self.model.ent_coef
        if hasattr(ent_coef, "item"):
            ent_coef = ent_coef.item()
        try:
            entropy_coef = float(ent_coef)
        except (TypeError, ValueError):
            entropy_coef = float("nan")

        self.monitor.update(
            step=self.num_timesteps,
            metrics={
                "train_reward_mean": reward_mean,
                "train_reward_std": reward_std,
                "train_episode_length": episode_length_mean,
                "fps": fps,
                "eta_seconds": eta_seconds,
                "lr": lr,
                "entropy_coef": entropy_coef,
            },
        )
        return True

    def _on_training_end(self) -> None:
        if self.monitor.enabled:
            self.monitor.update(step=self.num_timesteps)


class TemporaryLogLevel:
    """Temporarily set logging level (and disabled state) for selected loggers."""

    def __init__(self, logger_names: Iterable[str], level: int) -> None:
        self.logger_names = list(logger_names)
        self.level = level
        self.original_states: Dict[str, Tuple[int, bool]] = {}

    def __enter__(self) -> "TemporaryLogLevel":
        for name in self.logger_names:
            logger = logging.getLogger(name)
            self.original_states[name] = (logger.level, logger.disabled)
            logger.setLevel(self.level)
            if self.level > logging.CRITICAL:
                logger.disabled = True
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:  # noqa: ANN001
        for name, (level, disabled) in self.original_states.items():
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.disabled = disabled


# --------------------------------------------------------------------------------------
# Metric extraction & evaluation helpers
# --------------------------------------------------------------------------------------

def _safe_mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else 0.0


def _safe_std(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.std(finite)) if finite else 0.0


def _compute_profit_factor(trades: Sequence[Dict[str, Any]]) -> float:
    if not trades:
        return float("nan")
    gross_profit = sum(float(trade.get("realized_pnl", 0.0)) for trade in trades if float(trade.get("realized_pnl", 0.0)) > 0)
    gross_loss = sum(-float(trade.get("realized_pnl", 0.0)) for trade in trades if float(trade.get("realized_pnl", 0.0)) < 0)
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else float("nan")
    return float(gross_profit / gross_loss)


def _compute_average_trade_duration(trades: Sequence[Dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    durations = [float(trade.get("holding_period", 0.0)) for trade in trades]
    return float(np.mean(durations))


def _resolve_action_name(action_idx: int) -> str:
    try:
        return TradeAction(int(action_idx)).name
    except (ValueError, TypeError):
        return str(action_idx)


def _normalise_action_counts(counts: Dict[int, int], action_space_n: int) -> Tuple[Dict[str, int], Dict[str, float]]:
    total_actions = max(0, int(sum(int(v) for v in counts.values())))
    known_indices = set(range(max(0, action_space_n))) | set(int(k) for k in counts.keys())
    named_counts: Dict[str, int] = {}
    distribution: Dict[str, float] = {}

    for action_idx in sorted(known_indices):
        name = _resolve_action_name(action_idx)
        action_count = int(counts.get(action_idx, 0))
        named_counts[name] = action_count
        distribution[name] = float(action_count / total_actions) if total_actions > 0 else 0.0

    # Ensure canonical trade actions are present even if unseen
    for action_name in ALL_TRADE_ACTION_NAMES:
        named_counts.setdefault(action_name, 0)
        distribution.setdefault(action_name, 0.0)

    return named_counts, distribution


def _compute_action_entropy(probabilities: Iterable[float]) -> float:
    values = [float(p) for p in probabilities if isinstance(p, (int, float)) and p > 0.0]
    if not values:
        return 0.0
    entropy = -sum(p * math.log(p) for p in values if p > 0.0)
    return float(entropy)


def extract_episode_metrics(base_env: Any, info: Dict[str, Any]) -> Dict[str, Any]:
    """Extract trading metrics from a finished episode."""

    portfolio = base_env.portfolio

    metrics = info.get("terminal_metrics")
    if metrics is None:
        metrics = portfolio.get_portfolio_metrics()
    else:
        metrics = dict(metrics)

    trades = info.get("terminal_trades")
    if trades is None:
        trades = portfolio.get_closed_positions()

    reward_stats = info.get("terminal_reward_stats", info.get("reward_stats", {}))
    reward_contrib = info.get("reward_contributions", {})

    episode_metrics: Dict[str, Any] = {
        "total_return": float(metrics.get("total_return", 0.0)),
        "total_return_pct": float(metrics.get("total_return_pct", 0.0)),
        "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
        "sortino_ratio": float(metrics.get("sortino_ratio", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
        "win_rate": float(metrics.get("win_rate", 0.0)),
        "num_trades": int(metrics.get("total_trades", 0)),
        "profit_factor": _compute_profit_factor(trades),
        "avg_trade_duration": _compute_average_trade_duration(trades),
        "episode_reward": float(info.get("episode", {}).get("r", 0.0)),
        "episode_length": int(info.get("episode", {}).get("l", base_env.episode_step)),
        "reward_stats": reward_stats,
        "reward_contributions": reward_contrib,
    }

    if "action_distribution" in info:
        action_dist = info["action_distribution"]
        episode_metrics["action_distribution"] = {
            str(k): float(v) for k, v in action_dist.items()
        }

    if "action_counts" in info:
        action_counts = info["action_counts"]
        episode_metrics["action_counts"] = {
            str(k): int(v) for k, v in action_counts.items()
        }

    if "policy_action_counts" in info:
        policy_counts = info["policy_action_counts"]
        episode_metrics["policy_action_counts"] = {
            str(k): int(v) for k, v in policy_counts.items()
        }

    if "action_entropy" in info:
        entropy_value = info["action_entropy"]
        if isinstance(entropy_value, (int, float)) and np.isfinite(entropy_value):
            episode_metrics["action_entropy"] = float(entropy_value)

    return episode_metrics


def summarize_episode_metrics(episodes: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate per-episode metrics into summary statistics."""

    if not episodes:
        return {}

    summary: Dict[str, float] = {"episodes": float(len(episodes))}
    numeric_keys = [
        "total_return",
        "total_return_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "max_drawdown_pct",
        "win_rate",
        "profit_factor",
        "avg_trade_duration",
        "episode_reward",
        "episode_length",
    ]

    for key in numeric_keys:
        values = [episode.get(key) for episode in episodes]
        values = [float(v) for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
        if not values:
            continue
        summary[f"{key}_mean"] = _safe_mean(values)
        summary[f"{key}_std"] = _safe_std(values)

    action_entropy_values: List[float] = []
    aggregate_action_counts: Counter[str] = Counter()
    observed_actions: set[str] = set()
    aggregate_policy_counts: Counter[str] = Counter()
    observed_policy_actions: set[str] = set()

    for episode in episodes:
        entropy_value = episode.get("action_entropy")
        if isinstance(entropy_value, (int, float)) and np.isfinite(entropy_value):
            action_entropy_values.append(float(entropy_value))

        counts = episode.get("action_counts")
        if isinstance(counts, dict):
            cleaned_counts = {str(k): int(v) for k, v in counts.items()}
            aggregate_action_counts.update(cleaned_counts)
            observed_actions.update(cleaned_counts.keys())

        policy_counts = episode.get("policy_action_counts")
        if isinstance(policy_counts, dict):
            cleaned_policy = {str(k): int(v) for k, v in policy_counts.items()}
            aggregate_policy_counts.update(cleaned_policy)
            observed_policy_actions.update(cleaned_policy.keys())

    if action_entropy_values:
        summary["action_entropy_mean"] = _safe_mean(action_entropy_values)
        summary["action_entropy_std"] = _safe_std(action_entropy_values)

    if aggregate_action_counts:
        total_actions = sum(aggregate_action_counts.values())
        if total_actions > 0:
            distribution: Dict[str, float] = {}
            for name in ALL_TRADE_ACTION_NAMES:
                distribution[name] = float(aggregate_action_counts.get(name, 0) / total_actions)
            for name in sorted(observed_actions):
                if name not in distribution:
                    distribution[name] = float(aggregate_action_counts.get(name, 0) / total_actions)
            summary["action_distribution"] = distribution
            summary["action_entropy"] = _compute_action_entropy(distribution.values())

    if aggregate_policy_counts:
        total_policy = sum(aggregate_policy_counts.values())
        if total_policy > 0:
            policy_distribution: Dict[str, float] = {}
            for name in ALL_TRADE_ACTION_NAMES:
                policy_distribution[name] = float(aggregate_policy_counts.get(name, 0) / total_policy)
            for name in sorted(observed_policy_actions):
                if name not in policy_distribution:
                    policy_distribution[name] = float(aggregate_policy_counts.get(name, 0) / total_policy)
            summary["policy_action_distribution"] = policy_distribution
            summary["policy_action_entropy"] = _compute_action_entropy(policy_distribution.values())

    return summary


def run_evaluation(model: PPO, eval_env: VecEnv, n_episodes: int, deterministic: bool = True) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Run evaluation episodes and return summary plus per-episode metrics."""

    episodes: List[Dict[str, Any]] = []
    base_env = eval_env.envs[0] if hasattr(eval_env, "envs") else None
    if base_env is None:
        raise RuntimeError("Evaluation environment does not expose underlying environment")

    action_space = getattr(base_env, "action_space", getattr(eval_env, "action_space", None))
    if action_space is None or not hasattr(action_space, "n"):
        raise RuntimeError("Evaluation environment requires a discrete action space with 'n'")
    action_space_n = int(getattr(action_space, "n"))

    for _ in range(n_episodes):
        obs = eval_env.reset()
        state = None
        episode_reward = 0.0
        action_counter: Counter = Counter()
        while True:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            try:
                action_idx = int(np.asarray(action).reshape(-1)[0])
            except Exception:  # noqa: BLE001
                action_idx = int(action[0] if isinstance(action, (list, tuple)) else action)
            obs, rewards, dones, infos = eval_env.step(action)
            info_step = infos[0] if infos else {}
            executed_idx = info_step.get("executed_action_idx")
            if executed_idx is None:
                executed_idx = action_idx
            else:
                executed_idx = int(executed_idx)
            action_counter[executed_idx] += 1
            episode_reward += float(rewards[0])
            if dones[0]:
                info = infos[0]
                info.setdefault("episode", {})
                info["episode"].setdefault("r", episode_reward)
                named_counts, distribution = _normalise_action_counts(dict(action_counter), action_space_n)
                action_entropy = _compute_action_entropy(distribution.values())
                info["action_counts"] = named_counts
                info["action_distribution"] = distribution
                info["action_entropy"] = action_entropy
                episode_metrics = extract_episode_metrics(base_env, info)
                episodes.append(episode_metrics)
                break

    summary = summarize_episode_metrics(episodes)
    return summary, episodes


def log_evaluation_to_logger(logger: Optional[Logger], summary: Dict[str, float], step: int) -> None:
    if logger is None or not summary:
        return
    for key, value in summary.items():
        if key.endswith("_std"):
            continue
        if isinstance(value, (int, float)) and np.isfinite(value):
            logger.record(f"eval/{key}", float(value))
    logger.dump(step)


# --------------------------------------------------------------------------------------
# Custom callbacks
# --------------------------------------------------------------------------------------


class MLflowCallback(BaseCallback):
    """Periodically push training metrics to MLflow."""

    def __init__(self, log_freq: int = 100):
        super().__init__()
        self.log_freq = max(1, int(log_freq))

    def _on_step(self) -> bool:
        if mlflow.active_run() is None:
            return True
        if self.n_calls % self.log_freq != 0:
            return True

        metrics: Dict[str, float] = {"train/num_timesteps": float(self.num_timesteps)}

        if len(self.model.ep_info_buffer) > 0:
            rewards = [info.get("r", 0.0) for info in self.model.ep_info_buffer]
            lengths = [info.get("l", 0.0) for info in self.model.ep_info_buffer]
            metrics["train/episode_reward_mean"] = float(np.mean(rewards))
            metrics["train/episode_length_mean"] = float(np.mean(lengths))

        if self.model.logger is not None:
            for key, value in metrics.items():
                if key == "train/num_timesteps":
                    continue
                self.model.logger.record(key, value)

        mlflow.log_metrics(metrics, step=self.num_timesteps)
        return True


class RewardComponentLogger(BaseCallback):
    """Log reward component statistics and trading KPIs from env infos."""

    def __init__(self, component_keys: Iterable[str], log_freq: int = 1000):
        super().__init__()
        self.component_keys = list(component_keys)
        self.log_freq = max(1, int(log_freq))
        self._buffer: Dict[str, List[float]] = {k: [] for k in self.component_keys}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) if isinstance(self.locals, dict) else []
        for info in infos:
            reward_stats = info.get("reward_stats") or {}
            for key in self.component_keys:
                stat_key = f"{key}_mean"
                if stat_key in reward_stats:
                    self._buffer[key].append(float(reward_stats[stat_key]))

            if info.get("episode") is not None and mlflow.active_run() is not None:
                trading_metrics = {
                    "train/equity_return_pct": float(info.get("total_return_pct", 0.0)),
                    "train/sharpe_ratio": float(info.get("sharpe_ratio", 0.0)),
                    "train/max_drawdown_pct": float(info.get("max_drawdown_pct", 0.0)),
                    "train/win_rate": float(info.get("win_rate", 0.0)),
                    "train/num_trades": float(info.get("num_trades", 0.0)),
                }
                mlflow.log_metrics(trading_metrics, step=self.num_timesteps)
                if self.model.logger is not None:
                    for k, v in trading_metrics.items():
                        self.model.logger.record(k, v)

        if self.n_calls % self.log_freq == 0 and mlflow.active_run() is not None:
            aggregate = {
                f"reward/{key}_mean": float(np.mean(values))
                for key, values in self._buffer.items()
                if values
            }
            if aggregate:
                mlflow.log_metrics(aggregate, step=self.num_timesteps)
                if self.model.logger is not None:
                    for k, v in aggregate.items():
                        self.model.logger.record(k, v)
            self._buffer = {k: [] for k in self.component_keys}

        return True


class EntropyScheduleCallback(BaseCallback):
    """Dynamically update the entropy coefficient using a schedule."""

    def __init__(self, schedule: Callable[[float], float], log_freq: int = 100):
        super().__init__()
        self.schedule = schedule
        self.log_freq = max(1, int(log_freq))
        self._last_value: float = 0.0

    def _on_step(self) -> bool:
        progress_remaining = getattr(self.model, "_current_progress_remaining", None)
        if progress_remaining is None:
            total = float(getattr(self.model, "_total_timesteps", 1) or 1)
            progress_remaining = 1.0 - float(self.num_timesteps) / total
        progress_remaining = float(max(0.0, min(1.0, progress_remaining)))

        entropy_value = float(self.schedule(progress_remaining))
        self.model.ent_coef = entropy_value
        self._last_value = entropy_value

        if self.model.logger is not None and self.n_calls % self.log_freq == 0:
            self.model.logger.record("train/entropy_coef", entropy_value)

        if mlflow.active_run() is not None and self.n_calls % self.log_freq == 0:
            mlflow.log_metric("train/entropy_coef", entropy_value, step=self.num_timesteps)

        return True

    def boost(self, multiplier: float, *, max_multiplier: float | None = None, floor: float | None = None) -> Optional[float]:
        """Increase the schedule multiplier to encourage exploration."""

        schedule = getattr(self.schedule, "set_multiplier", None)
        if not callable(schedule):
            return None

        current_multiplier = getattr(self.schedule, "multiplier", 1.0)
        new_multiplier = current_multiplier * max(1.0, float(multiplier))
        if max_multiplier is not None:
            new_multiplier = min(float(max_multiplier), new_multiplier)
        self.schedule.set_multiplier(new_multiplier)

        progress_remaining = getattr(self.model, "_current_progress_remaining", None)
        if progress_remaining is None:
            total = float(getattr(self.model, "_total_timesteps", 1) or 1)
            progress_remaining = 1.0 - float(self.num_timesteps) / total
        progress_remaining = float(max(0.0, min(1.0, progress_remaining)))

        boosted_value = float(self.schedule(progress_remaining))
        if floor is not None and boosted_value < floor:
            base = getattr(self.schedule, "base_value", None)
            if callable(base):
                base_value = float(self.schedule.base_value(progress_remaining))
                if base_value > 0:
                    required_multiplier = float(floor / base_value)
                    if max_multiplier is not None:
                        required_multiplier = min(required_multiplier, float(max_multiplier))
                    self.schedule.set_multiplier(required_multiplier)
                    boosted_value = float(self.schedule(progress_remaining))

        self.model.ent_coef = boosted_value
        self._last_value = boosted_value
        return boosted_value


class PolicyTelemetryCallback(BaseCallback):
    """Capture policy diagnostics (logits, entropy, covariance, curriculum penalties)."""

    def __init__(
        self,
        *,
        action_names: Sequence[str],
        max_samples: int = 4096,
        histogram_bins: int = 20,
        log_every: int = 1,
    ) -> None:
        super().__init__()
        self.action_names = list(action_names)
        self.max_samples = max(1, int(max_samples))
        self.histogram_bins = max(5, int(histogram_bins))
        self.log_every = max(1, int(log_every))
        self._rollout_counter = 0
        self._curriculum_penalties: List[float] = []
        self._forced_exit_count: int = 0
        self._forced_exit_pnls: List[float] = []
        self._latest_rollout_metrics: Dict[str, Any] = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) if isinstance(self.locals, dict) else []
        for info in infos:
            curriculum = info.get("curriculum") or {}
            penalty = float(curriculum.get("penalty", 0.0)) if curriculum else 0.0
            if penalty != 0.0:
                self._curriculum_penalties.append(penalty)

            action_info = info.get("action_info") or {}
            forced_trades = action_info.get("forced_trades") or []
            for trade in forced_trades:
                self._forced_exit_count += 1
                pnl_value = trade.get("realized_pnl")
                if pnl_value is not None and np.isfinite(pnl_value):
                    self._forced_exit_pnls.append(float(pnl_value))
        return True

    @staticmethod
    def _flatten_array(array: np.ndarray) -> np.ndarray:
        if array.ndim >= 2:
            leading = array.shape[0] * array.shape[1]
            trailing = array.shape[2:]
            return array.reshape(leading, *trailing)
        return array.reshape(-1, *array.shape[1:]) if array.ndim > 1 else array.reshape(-1)

    def _prepare_observations(self, observations: Any) -> Tuple[Any, int]:
        if isinstance(observations, dict):
            flattened: Dict[str, np.ndarray] = {}
            sample_count = None
            for key, value in observations.items():
                arr = np.asarray(value)
                flat = self._flatten_array(arr)
                flattened[key] = flat
                if sample_count is None:
                    sample_count = flat.shape[0]
            return flattened, int(sample_count or 0)

        arr = np.asarray(observations)
        flat = self._flatten_array(arr)
        return flat, int(flat.shape[0])

    def _compute_policy_stats(self, buffer: Any) -> Optional[Dict[str, Any]]:
        obs_raw = getattr(buffer, "observations", None)
        advantages = getattr(buffer, "advantages", None)
        if obs_raw is None or advantages is None:
            return None

        flattened_obs, total_samples = self._prepare_observations(obs_raw)
        if total_samples == 0:
            return None

        sample_count = min(total_samples, self.max_samples)
        if isinstance(flattened_obs, dict):
            sample_obs = {k: v[:sample_count] for k, v in flattened_obs.items()}
        else:
            sample_obs = flattened_obs[:sample_count]

        with torch.no_grad():
            obs_tensor, _ = self.model.policy.obs_to_tensor(sample_obs)
            distribution = self.model.policy.get_distribution(obs_tensor)
            logits = distribution.distribution.logits
            probs = torch.softmax(logits, dim=-1)

        logits_np = logits.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        entropy_np = (-probs_np * np.log(np.clip(probs_np, 1e-12, 1.0))).sum(axis=1)

        advantages_np = np.asarray(advantages)
        advantages_np = self._flatten_array(advantages_np)[:sample_count]
        advantages_np = advantages_np.astype(np.float64, copy=False)

        stats: Dict[str, Any] = {
            "logits": logits_np,
            "probs": probs_np,
            "entropy": entropy_np,
            "advantages": advantages_np,
        }
        return stats

    def _log_histograms(self, logger: Logger, logits: np.ndarray) -> None:
        for idx, action_name in enumerate(self.action_names):
            action_logits = logits[:, idx]
            if action_logits.size == 0:
                continue
            min_val = float(np.min(action_logits))
            max_val = float(np.max(action_logits))
            if np.isclose(min_val, max_val):
                min_val -= 0.5
                max_val += 0.5
            bins = np.linspace(min_val, max_val, self.histogram_bins + 1)
            counts, _ = np.histogram(action_logits, bins=bins, density=False)
            normalised = counts / max(1, action_logits.size)
            for bin_idx, value in enumerate(normalised):
                logger.record(
                    f"telemetry/logits_hist/{action_name}/bin_{bin_idx}",
                    float(value),
                    exclude=("stdout",),
                )

    def _on_rollout_end(self) -> bool:
        self._rollout_counter += 1
        if self._rollout_counter % self.log_every != 0:
            self._curriculum_penalties.clear()
            self._forced_exit_count = 0
            self._forced_exit_pnls.clear()
            self._latest_rollout_metrics = {}
            return True

        buffer = getattr(self.model, "rollout_buffer", None)
        if buffer is None:
            return True

        stats = self._compute_policy_stats(buffer)
        logger = self.model.logger
        if stats is None or logger is None:
            self._curriculum_penalties.clear()
            self._forced_exit_count = 0
            self._forced_exit_pnls.clear()
            self._latest_rollout_metrics = {}
            return True

        logits = stats["logits"]
        probs = stats["probs"]
        entropy = stats["entropy"]
        advantages = stats["advantages"]

        logits_mean = logits.mean(axis=0)
        logits_std = logits.std(axis=0)
        logits_min = logits.min(axis=0)
        logits_max = logits.max(axis=0)

        entropy_mean = float(np.mean(entropy))
        entropy_std = float(np.std(entropy))
        entropy_min = float(np.min(entropy))
        entropy_max = float(np.max(entropy))

        adv_centered = advantages - advantages.mean() if advantages.size > 0 else advantages
        adv_variance = float(np.mean(adv_centered ** 2)) if advantages.size > 0 else 0.0

        covariance_map: Dict[str, float] = {}
        correlation_map: Dict[str, float] = {}

        for idx, action_name in enumerate(self.action_names):
            logger.record(f"telemetry/logits_mean/{action_name}", float(logits_mean[idx]))
            logger.record(f"telemetry/logits_std/{action_name}", float(logits_std[idx]))
            logger.record(f"telemetry/logits_min/{action_name}", float(logits_min[idx]))
            logger.record(f"telemetry/logits_max/{action_name}", float(logits_max[idx]))

            action_probs = probs[:, idx]
            prob_centered = action_probs - action_probs.mean()
            prob_variance = float(np.mean(prob_centered ** 2))
            covariance = float(np.mean(prob_centered * adv_centered)) if advantages.size > 0 else 0.0
            denom = np.sqrt(max(prob_variance, 1e-12) * max(adv_variance, 1e-12))
            correlation = float(covariance / denom) if denom > 0 else 0.0

            logger.record(f"telemetry/covariance/{action_name}", covariance)
            logger.record(f"telemetry/correlation/{action_name}", correlation)
            covariance_map[action_name] = covariance
            correlation_map[action_name] = correlation

        logger.record("telemetry/entropy_mean", entropy_mean)
        logger.record("telemetry/entropy_std", entropy_std)
        logger.record("telemetry/entropy_min", entropy_min)
        logger.record("telemetry/entropy_max", entropy_max)

        if self._curriculum_penalties:
            penalty_mean = float(np.mean(self._curriculum_penalties))
            penalty_sum = float(np.sum(self._curriculum_penalties))
            logger.record("telemetry/curriculum_penalty_mean", penalty_mean)
            logger.record("telemetry/curriculum_penalty_sum", penalty_sum)
        else:
            logger.record("telemetry/curriculum_penalty_mean", 0.0)
            logger.record("telemetry/curriculum_penalty_sum", 0.0)

        logger.record("telemetry/forced_exit_count", float(self._forced_exit_count))
        if self._forced_exit_pnls:
            logger.record("telemetry/forced_exit_pnl_mean", float(np.mean(self._forced_exit_pnls)))
            logger.record("telemetry/forced_exit_pnl_min", float(np.min(self._forced_exit_pnls)))
            logger.record("telemetry/forced_exit_pnl_max", float(np.max(self._forced_exit_pnls)))
        else:
            logger.record("telemetry/forced_exit_pnl_mean", 0.0)

        self._log_histograms(logger, logits)

        self._latest_rollout_metrics = {
            "entropy_mean": entropy_mean,
            "entropy_min": entropy_min,
            "entropy_max": entropy_max,
            "entropy_std": entropy_std,
            "covariance": covariance_map,
            "correlation": correlation_map,
            "logits_mean": logits_mean,
            "logits_std": logits_std,
        }

        self._curriculum_penalties.clear()
        self._forced_exit_count = 0
        self._forced_exit_pnls.clear()
        return True

    def get_latest_rollout_metrics(self) -> Dict[str, Any]:
        return dict(self._latest_rollout_metrics)


class EntropyAdaptiveController(BaseCallback):
    """Boost entropy coefficient when exploration collapses; guard covariance spikes."""

    def __init__(
        self,
        *,
        schedule: ScaledSchedule,
        telemetry: PolicyTelemetryCallback,
        total_timesteps: int,
        target_entropy: float,
        bonus_scale: float,
        decay_rate: float = 0.1,
        warmup_steps: int = 0,
        max_multiplier: Optional[float] = None,
        floor: Optional[float] = None,
        covariance_guard: Optional[Dict[str, Any]] = None,
        logger_tag: str = "entropy",
    ) -> None:
        super().__init__()
        self.schedule = schedule
        self.telemetry = telemetry
        self.total_timesteps = max(1, int(total_timesteps) or 1)
        self.target_entropy = float(max(0.0, target_entropy))
        self.bonus_scale = float(max(0.0, bonus_scale))
        self.decay_rate = float(max(0.0, min(decay_rate, 1.0)))
        self.warmup_steps = max(0, int(warmup_steps))
        self.max_multiplier = float(max_multiplier) if max_multiplier is not None else None
        self.floor = float(floor) if floor is not None else None
        self.covariance_guard_cfg = dict(covariance_guard or {})
        self.logger_tag = logger_tag

        self._last_boost: float = 1.0
        self._last_penalty: float = 1.0

    def _current_step(self) -> float:
        progress_remaining = getattr(self.model, "_current_progress_remaining", None)
        if progress_remaining is None:
            total = float(getattr(self.model, "_total_timesteps", 1) or 1)
            progress_remaining = 1.0 - float(self.num_timesteps) / total
        progress_remaining = float(max(0.0, min(1.0, progress_remaining)))
        return (1.0 - progress_remaining) * self.total_timesteps

    def _apply_decay(self) -> None:
        if self.decay_rate <= 0:
            return
        current = getattr(self.schedule, "multiplier", 1.0)
        if current <= 1.0:
            return
        decayed = 1.0 + (current - 1.0) * (1.0 - self.decay_rate)
        decayed = max(1.0, decayed)
        self.schedule.set_multiplier(decayed)

    def _apply_entropy_boost(self, entropy_mean: float) -> None:
        if self.target_entropy <= 0.0 or self.bonus_scale <= 0.0:
            self._last_boost = 1.0
            return

        gap = self.target_entropy - float(entropy_mean)
        if gap <= 0.0:
            self._last_boost = 1.0
            return

        relative_gap = gap / max(self.target_entropy, 1e-6)
        multiplier = 1.0 + self.bonus_scale * relative_gap
        current = getattr(self.schedule, "multiplier", 1.0)
        new_multiplier = current * multiplier
        if self.max_multiplier is not None:
            new_multiplier = min(new_multiplier, self.max_multiplier)
        self.schedule.set_multiplier(new_multiplier)
        if self.floor is not None:
            progress_remaining = getattr(self.model, "_current_progress_remaining", 0.0)
            value = float(self.schedule(progress_remaining))
            if value < self.floor:
                base_value = float(self.schedule.base_value(progress_remaining))
                if base_value > 0:
                    required = self.floor / base_value
                    if self.max_multiplier is not None:
                        required = min(required, self.max_multiplier)
                    self.schedule.set_multiplier(required)
        self._last_boost = multiplier

    def _apply_covariance_penalty(self, covariance: Dict[str, float]) -> None:
        if not covariance:
            self._last_penalty = 1.0
            return

        enabled = bool(self.covariance_guard_cfg.get("enabled", True))
        if not enabled:
            self._last_penalty = 1.0
            return

        threshold = float(self.covariance_guard_cfg.get("threshold", 0.25))
        penalty_scale = float(self.covariance_guard_cfg.get("penalty_scale", 0.5))
        min_multiplier = float(self.covariance_guard_cfg.get("min_multiplier", 0.5))

        max_cov = max(abs(float(v)) for v in covariance.values())
        if max_cov <= threshold:
            self._last_penalty = 1.0
            return

        severity = (max_cov - threshold) / max(max_cov, 1e-6)
        penalty = max(0.0, 1.0 - penalty_scale * severity)
        current = getattr(self.schedule, "multiplier", 1.0)
        new_multiplier = max(min_multiplier, current * penalty)
        self.schedule.set_multiplier(new_multiplier)
        self._last_penalty = penalty

    def _on_rollout_end(self) -> bool:
        current_step = self._current_step()
        metrics = self.telemetry.get_latest_rollout_metrics()
        if not metrics:
            self._apply_decay()
            return True

        entropy_mean = float(metrics.get("entropy_mean", 0.0))
        covariance = metrics.get("covariance", {}) or {}

        if current_step >= self.warmup_steps:
            self._apply_entropy_boost(entropy_mean)
            self._apply_covariance_penalty({k: float(v) for k, v in covariance.items()})
        else:
            self._last_boost = 1.0
            self._last_penalty = 1.0

        self._apply_decay()

        logger = getattr(self.model, "logger", None)
        if logger is not None:
            logger.record(f"{self.logger_tag}/multiplier", float(getattr(self.schedule, "multiplier", 1.0)))
            logger.record(f"{self.logger_tag}/last_boost", float(self._last_boost))
            logger.record(f"{self.logger_tag}/last_penalty", float(self._last_penalty))
            logger.record(f"{self.logger_tag}/entropy_mean", entropy_mean)
        return True

    def _on_step(self) -> bool:
        """No-op per-step hook required by :class:`BaseCallback`."""
        return True


@dataclass
class EvaluationHistoryEntry:
    step: int
    summary: Dict[str, float]
    episodes: List[Dict[str, Any]]


class Phase3EvaluationCallback(BaseCallback):
    """Custom evaluation callback with early stopping and best-checkpoint saving."""

    def __init__(
        self,
        *,
        eval_env: VecEnv,
        eval_freq: int,
        n_eval_episodes: int,
        patience: int,
        min_delta: float,
        monitor_metric: str,
        checkpoint_on_best: bool,
        checkpoint_dir: Path,
        performance_guard: Optional[Dict[str, Any]] = None,
        action_entropy_guard: Optional[Dict[str, Any]] = None,
        success_thresholds: Optional[Dict[str, float]] = None,
        deterministic: bool = True,
        verbose: int = 0,
        monitor: Optional[RichTrainingMonitor] = None,
        entropy_controller: Optional[EntropyScheduleCallback] = None,
        total_timesteps: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = max(1, int(n_eval_episodes))
        self.patience = max(0, int(patience))
        self.min_delta = float(min_delta)
        self.monitor_metric = monitor_metric
        self.checkpoint_on_best = checkpoint_on_best
        self.checkpoint_dir = checkpoint_dir
        self.success_thresholds = success_thresholds or {}
        self.deterministic = deterministic
        self.monitor = monitor
        self.performance_guard_cfg = dict(performance_guard or {})
        self.action_entropy_guard_cfg = dict(action_entropy_guard or {})
        self.entropy_controller = entropy_controller
        self.total_timesteps = max(1, int(total_timesteps) or 1)

        self.history = []
        self.best_metric = -float("inf")
        self.best_summary: Optional[Dict[str, float]] = None
        self.no_improvement_steps = 0
        self._last_eval_step = 0
        self._performance_guard_counter = 0
        self._action_entropy_guard_counter = 0
        self._entropy_guard_interventions = 0
        cooldown_default = int(self.action_entropy_guard_cfg.get("cooldown_steps", 0) or 0)
        self._last_entropy_intervention_step = -cooldown_default
        self.halt_reason: Optional[str] = None

    @staticmethod
    def _candidate_metric_keys(metric_name: str) -> Tuple[str, ...]:
        metric = (metric_name or "").strip().lower()
        alias_map = {
            "sharpe": "sharpe_ratio",
            "sharpe_ratio": "sharpe_ratio",
            "sharpe_ratio_mean": "sharpe_ratio_mean",
            "return": "total_return_pct",
            "total_return": "total_return",
            "total_return_pct": "total_return_pct",
            "max_drawdown": "max_drawdown_pct",
            "drawdown": "max_drawdown_pct",
            "max_drawdown_pct": "max_drawdown_pct",
            "win_rate": "win_rate",
            "profit_factor": "profit_factor",
        }

        canonical = alias_map.get(metric, metric)
        candidates: List[str] = []
        if canonical:
            if canonical.endswith("_mean"):
                candidates.append(canonical)
                base = canonical.rsplit("_mean", 1)[0]
                if base:
                    candidates.append(base)
            else:
                candidates.append(f"{canonical}_mean")
                candidates.append(canonical)

        # Preserve order while removing duplicates
        return tuple(dict.fromkeys(candidates))

    def _resolve_metric_value(self, summary: Dict[str, float], metric_name: str) -> Tuple[Optional[float], Optional[str]]:
        if not metric_name:
            return None, None

        metric_name_normalized = metric_name.strip().lower()
        if metric_name_normalized in {"action_entropy", "entropy"}:
            candidate_keys: Tuple[str, ...] = ("action_entropy", "action_entropy_mean")
        else:
            candidate_keys = self._candidate_metric_keys(metric_name)

        for key in candidate_keys:
            value = summary.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                return float(value), key
        return None, None

    def _check_performance_guard(self, summary: Dict[str, float]) -> bool:
        cfg = self.performance_guard_cfg
        if not cfg.get("enabled", False):
            return False

        warmup_steps = int(cfg.get("warmup_steps", 0) or 0)
        if self.num_timesteps < warmup_steps:
            self._performance_guard_counter = 0
            return False

        threshold = float(cfg.get("threshold", 0.0))
        patience = max(1, int(cfg.get("patience", 1)))
        metric_name = cfg.get("metric", "sharpe")
        metric_value, resolved_key = self._resolve_metric_value(summary, metric_name)

        if metric_value is None:
            return False

        if metric_value < threshold:
            self._performance_guard_counter += 1
            if self._performance_guard_counter >= patience:
                metric_label = resolved_key or metric_name
                self.halt_reason = f"performance_guard:{metric_label}"
                LOGGER.warning(
                    "Performance guard triggered: %s=%.4f below threshold %.4f at %s steps",
                    metric_label,
                    metric_value,
                    threshold,
                    self.num_timesteps,
                )
                if mlflow.active_run() is not None:
                    mlflow.log_metric("guard/performance_trigger_step", self.num_timesteps, step=self.num_timesteps)
                    mlflow.log_metric("guard/performance_metric", metric_value, step=self.num_timesteps)
                if self.monitor and self.monitor.enabled:
                    self.monitor.update(status="Performance guard halt", metrics={metric_label: metric_value})
                return True
        else:
            self._performance_guard_counter = 0

        return False

    def _check_action_entropy_guard(self, summary: Dict[str, float]) -> bool:
        cfg = self.action_entropy_guard_cfg
        if not cfg.get("enabled", False):
            return False

        warmup_steps = int(cfg.get("warmup_steps", 0) or 0)
        if self.num_timesteps < warmup_steps:
            self._action_entropy_guard_counter = 0
            return False

        threshold = float(cfg.get("threshold", 0.0))
        patience = max(1, int(cfg.get("patience", 1)))
        entropy_value, resolved_key = self._resolve_metric_value(summary, "action_entropy")

        if entropy_value is None:
            self._action_entropy_guard_counter = 0
            return False

        metric_label = resolved_key or "action_entropy"

        if entropy_value < threshold:
            self._action_entropy_guard_counter += 1
            if self._action_entropy_guard_counter < patience:
                return False

            boost_multiplier = float(cfg.get("boost_multiplier", 1.5))
            max_multiplier_raw = cfg.get("max_multiplier")
            boost_floor_raw = cfg.get("boost_floor")
            cooldown_steps = int(cfg.get("cooldown_steps", 0) or 0)
            max_interventions_raw = cfg.get("max_interventions")
            halt_on_failure = bool(cfg.get("halt_on_failure", True))
            halt_on_cooldown = bool(cfg.get("halt_on_cooldown", False))
            halt_on_limit = bool(cfg.get("halt_on_limit", False))

            max_multiplier = float(max_multiplier_raw) if max_multiplier_raw is not None else None
            boost_floor = float(boost_floor_raw) if boost_floor_raw is not None else None
            max_interventions = int(max_interventions_raw) if max_interventions_raw is not None else None

            since_last = self.num_timesteps - self._last_entropy_intervention_step
            cooldown_active = cooldown_steps > 0 and since_last < cooldown_steps
            intervention_limit_reached = max_interventions is not None and self._entropy_guard_interventions >= max_interventions

            if mlflow.active_run() is not None:
                mlflow.log_metric("guard/action_entropy_trigger_step", self.num_timesteps, step=self.num_timesteps)
                mlflow.log_metric("guard/action_entropy_value", entropy_value, step=self.num_timesteps)

            if self.monitor and self.monitor.enabled:
                self.monitor.update(status="Entropy guard trigger", metrics={metric_label: entropy_value})

            should_halt = False
            failure_reason: Optional[str] = None
            boosted_value: Optional[float] = None

            if self.entropy_controller is None:
                should_halt = True
                failure_reason = "no_entropy_controller"
            elif intervention_limit_reached:
                should_halt = halt_on_limit
                failure_reason = "intervention_limit"
            elif cooldown_active:
                should_halt = halt_on_cooldown
                failure_reason = "cooldown_active"
            else:
                boosted_value = self.entropy_controller.boost(
                    boost_multiplier,
                    max_multiplier=max_multiplier,
                    floor=boost_floor,
                )
                if boosted_value is not None:
                    self._entropy_guard_interventions += 1
                    self._last_entropy_intervention_step = self.num_timesteps
                    self._action_entropy_guard_counter = 0
                    LOGGER.warning(
                        "Action entropy guard intervention #%s: boosted entropy coef to %.6f (value=%.4f, threshold=%.4f)",
                        self._entropy_guard_interventions,
                        boosted_value,
                        entropy_value,
                        threshold,
                    )
                    if self.model.logger is not None:
                        self.model.logger.record("guard/action_entropy_interventions", float(self._entropy_guard_interventions))
                        self.model.logger.record("guard/action_entropy_value", float(entropy_value))
                        self.model.logger.record("train/entropy_coef", float(boosted_value))
                    if mlflow.active_run() is not None:
                        mlflow.log_metric("guard/action_entropy_interventions", self._entropy_guard_interventions, step=self.num_timesteps)
                        mlflow.log_metric("guard/action_entropy_boosted_coef", boosted_value, step=self.num_timesteps)
                    if self.monitor and self.monitor.enabled:
                        self.monitor.update(
                            status="Entropy boost applied",
                            metrics={metric_label: entropy_value, "entropy_coef": boosted_value},
                        )
                    return False
                else:
                    should_halt = halt_on_failure
                    failure_reason = "boost_failed"

            if should_halt:
                self.halt_reason = f"action_entropy_guard:{metric_label}"
                LOGGER.warning(
                    "Action entropy guard halt (%s): %s=%.4f below threshold %.4f at %s steps",
                    failure_reason or "triggered",
                    metric_label,
                    entropy_value,
                    threshold,
                    self.num_timesteps,
                )
                if mlflow.active_run() is not None:
                    mlflow.log_metric("guard/action_entropy_halt", 1, step=self.num_timesteps)
                if self.monitor and self.monitor.enabled:
                    self.monitor.update(status="Entropy guard halt", metrics={metric_label: entropy_value})
                return True

            LOGGER.info(
                "Action entropy guard triggered (%s) but continuing without halt (reason=%s, interventions=%s)",
                metric_label,
                failure_reason or "handled",
                self._entropy_guard_interventions,
            )
            self._action_entropy_guard_counter = 0
            if self.monitor and self.monitor.enabled:
                self.monitor.update(status="Entropy guard cooldown", metrics={metric_label: entropy_value})
            return False
        else:
            self._action_entropy_guard_counter = 0

        return False

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True

        self._last_eval_step = self.num_timesteps

        if self.monitor and self.monitor.enabled:
            self.monitor.update(status=f"Evaluating @ {self.num_timesteps:,} steps")

        if self.monitor is not None:
            self.monitor.pause_timer()
        try:
            summary, episodes = run_evaluation(
                self.model,
                self.eval_env,
                self.n_eval_episodes,
                deterministic=self.deterministic,
            )
        finally:
            if self.monitor is not None:
                self.monitor.resume_timer()
        self.history.append(EvaluationHistoryEntry(self.num_timesteps, summary, episodes))

        if self.monitor and self.monitor.enabled:
            self.monitor.update_evaluation(summary=summary, step=self.num_timesteps)

        log_evaluation_to_logger(self.model.logger, summary, self.num_timesteps)

        if mlflow.active_run() is not None:
            mlflow.log_metrics(
                {f"eval/{k}": float(v) for k, v in summary.items() if isinstance(v, (int, float))},
                step=self.num_timesteps,
            )

        metric_value: Optional[float] = None
        metric_key: Optional[str] = None
        for candidate_key in self._candidate_metric_keys(self.monitor_metric):
            value = summary.get(candidate_key)
            if value is None:
                continue
            if isinstance(value, (int, float)) and np.isfinite(value):
                metric_key = candidate_key
                metric_value = float(value)
                break

        if metric_value is not None and np.isfinite(metric_value):
            if metric_value > self.best_metric + self.min_delta:
                self.best_metric = float(metric_value)
                self.best_summary = dict(summary)
                self.no_improvement_steps = 0

                if self.checkpoint_on_best:
                    ensure_directory(self.checkpoint_dir)
                    best_path = self.checkpoint_dir / "best_model.zip"
                    try:
                        self.model.save(best_path)
                    except TypeError:
                        LOGGER.exception("Failed to save best model due to pickling error; attempting diagnostics")
                        _debug_pickle_failures(self.model)
                        raise
                    log_key = metric_key or self.monitor_metric
                    LOGGER.info("New best %s=%.4f at %s timesteps. Saved %s", log_key, metric_value, self.num_timesteps, best_path)
                    if mlflow.active_run() is not None:
                        mlflow.log_metric("eval/best_metric", self.best_metric, step=self.num_timesteps)
                if self.monitor and self.monitor.enabled:
                    self.monitor.update_evaluation(best_metric=self.best_metric, step=self.num_timesteps)
            else:
                self.no_improvement_steps += 1
                if self.patience and self.no_improvement_steps >= self.patience:
                    LOGGER.warning("Early stopping triggered after %s evaluations without improvement.", self.patience)
                    if mlflow.active_run() is not None:
                        mlflow.log_metric("training/early_stop_step", self.num_timesteps, step=self.num_timesteps)
                    if self.monitor and self.monitor.enabled:
                        self.monitor.update(status="Early stopping", metrics={"best_sharpe": self.best_metric})
                    return False

        if self._check_performance_guard(summary):
            return False

        if self._check_action_entropy_guard(summary):
            return False

        if self.monitor and self.monitor.enabled:
            self.monitor.update(status="Training")

        return True


# --------------------------------------------------------------------------------------
# Training orchestration
# --------------------------------------------------------------------------------------

def configure_sb3_logger(log_dir: Path) -> Logger:
    """Configure Stable-Baselines3 logger outputs."""

    ensure_directory(log_dir)
    return configure(str(log_dir), ["tensorboard", "csv", "json"])


def prepare_mlflow(config: Dict[str, Any], symbol: str) -> None:
    mlflow.set_experiment(config["experiment"]["name"])
    run_name = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.start_run(run_name=run_name)
    meta = config.get("meta", {})
    if meta.get("fingerprint"):
        mlflow.set_tag("config_fingerprint", meta["fingerprint"])
    if meta.get("config_path"):
        mlflow.set_tag("config_path", meta["config_path"])
    mlflow.log_params(
        {
            "symbol": symbol,
            "policy": config["ppo"].get("policy", "MultiInputPolicy"),
            "learning_rate": config["ppo"].get("learning_rate"),
            "lr_schedule": config["ppo"].get("lr_schedule"),
            "n_steps": config["ppo"].get("n_steps"),
            "batch_size": config["ppo"].get("batch_size"),
            "n_epochs": config["ppo"].get("n_epochs"),
            "gamma": config["ppo"].get("gamma"),
            "gae_lambda": config["ppo"].get("gae_lambda"),
            "clip_range": config["ppo"].get("clip_range"),
            "clip_range_vf": config["ppo"].get("clip_range_vf"),
            "ent_coef": config["ppo"].get("ent_coef"),
            "vf_coef": config["ppo"].get("vf_coef"),
            "max_grad_norm": config["ppo"].get("max_grad_norm"),
            "target_kl": config["ppo"].get("target_kl"),
            "total_timesteps": config["training"].get("total_timesteps"),
            "n_envs": config["training"].get("n_envs", 1),
            "seed": config["training"].get("seed"),
        }
    )


def close_mlflow() -> None:
    if mlflow.active_run():
        mlflow.end_run()


@dataclass
class TrainingResult:
    symbol: str
    status: str
    duration_hours: float
    total_timesteps: int
    final_model_path: Optional[str] = None
    promoted_model_path: Optional[str] = None
    best_metric: Optional[float] = None
    best_summary: Optional[Dict[str, float]] = None
    final_summary: Optional[Dict[str, float]] = None
    checkpoints: Optional[List[str]] = None
    error: Optional[str] = None
    config_fingerprint: Optional[str] = None
    reward_snapshot: Optional[Dict[str, float]] = None
    halt_reason: Optional[str] = None


def _extract_metric_value(summary: Dict[str, float], metric_name: str) -> Optional[float]:
    if not summary or not metric_name:
        return None
    for candidate_key in Phase3EvaluationCallback._candidate_metric_keys(metric_name):
        value = summary.get(candidate_key)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            return float(value)
    return None


def train_symbol_agent(
    symbol: str,
    config: Dict[str, Any],
    *,
    resume: bool = False,
) -> TrainingResult:
    LOGGER.info("Starting training for %s", symbol)
    experiment_cfg = config["experiment"]
    training_cfg = config["training"]
    ppo_cfg = config["ppo"]

    checkpoint_dir = ensure_directory(Path(experiment_cfg["checkpoint_dir"]) / symbol)
    log_dir = ensure_directory(Path(experiment_cfg["log_dir"]) / symbol)
    total_timesteps = int(training_cfg.get("total_timesteps", 100_000))

    LOGGER.info(
        "[%s] Training settings: total_timesteps=%s, n_envs=%s, eval_freq=%s, save_freq=%s, seed=%s",
        symbol,
        training_cfg.get("total_timesteps"),
        training_cfg.get("n_envs"),
        training_cfg.get("eval_freq"),
        training_cfg.get("save_freq"),
        training_cfg.get("seed"),
    )

    stability_cfg = training_cfg.get("stability_guard", {})

    policy_kwargs = resolve_policy_kwargs(ppo_cfg.get("policy_kwargs", {}))
    optimizer_cfg = dict(ppo_cfg.get("optimizer", {}) or {})
    raw_lr_schedule = create_lr_schedule(
        ppo_cfg.get("learning_rate", 3e-4),
        ppo_cfg.get("lr_schedule", "constant"),
        ppo_cfg.get("lr_min", 1e-5),
    )
    lr_schedule_callable = ensure_schedule_callable(raw_lr_schedule)
    lr_min_value = float(ppo_cfg.get("lr_min", 1e-5) or 0.0)
    warmup_steps = int(optimizer_cfg.get("warmup_steps", 0) or 0)
    warmup_floor = float(optimizer_cfg.get("warmup_floor", lr_min_value))
    if warmup_steps > 0:
        lr_schedule_callable = WarmupSchedule(
            lr_schedule_callable,
            warmup_steps=warmup_steps,
            total_timesteps=total_timesteps,
            floor=warmup_floor,
        )
    lr_schedule = ScaledSchedule(lr_schedule_callable, min_value=lr_min_value)
    initial_lr_value = float(lr_schedule.current_value(1.0))
    beta1 = float(optimizer_cfg.get("beta1", optimizer_cfg.get("beta_1", 0.95)))
    beta2 = float(optimizer_cfg.get("beta2", optimizer_cfg.get("beta_2", 0.95)))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))
    optimizer_eps = optimizer_cfg.get("eps")
    LOGGER.info(
        "[%s] PPO hyperparameters: lr=%s (%s schedule, lr_min=%s, initial=%.3e, warmup_steps=%s), n_steps=%s, batch_size=%s, n_epochs=%s, gamma=%s, gae_lambda=%s, clip_range=%s, ent_coef=%s, vf_coef=%s, max_grad_norm=%s, target_kl=%s",
        symbol,
        ppo_cfg.get("learning_rate"),
        ppo_cfg.get("lr_schedule", "constant"),
        ppo_cfg.get("lr_min"),
        initial_lr_value,
        warmup_steps,
        ppo_cfg.get("n_steps"),
        ppo_cfg.get("batch_size"),
        ppo_cfg.get("n_epochs"),
        ppo_cfg.get("gamma"),
        ppo_cfg.get("gae_lambda"),
        ppo_cfg.get("clip_range"),
        ppo_cfg.get("ent_coef"),
        ppo_cfg.get("vf_coef"),
        ppo_cfg.get("max_grad_norm"),
        ppo_cfg.get("target_kl"),
    )
    entropy_scheduler_cfg = ppo_cfg.get("entropy_scheduler")
    if entropy_scheduler_cfg:
        entropy_initial = entropy_scheduler_cfg.get("initial", ppo_cfg.get("ent_coef", 0.01))
        entropy_final = entropy_scheduler_cfg.get(
            "final",
            entropy_scheduler_cfg.get("target", ppo_cfg.get("ent_min", entropy_initial)),
        )
        entropy_min_value = float(entropy_scheduler_cfg.get("min", ppo_cfg.get("ent_min", entropy_final)))
        raw_entropy_schedule = create_piecewise_entropy_schedule(
            initial=float(entropy_initial),
            final=float(entropy_final),
            hold_steps=int(entropy_scheduler_cfg.get("hold_steps", 0) or 0),
            decay_steps=entropy_scheduler_cfg.get("decay_steps"),
            minimum=entropy_min_value,
            total_timesteps=total_timesteps,
            strategy=str(entropy_scheduler_cfg.get("strategy", "hold_then_linear")),
        )
    else:
        raw_entropy_schedule = create_entropy_schedule(
            ppo_cfg.get("ent_coef", 0.01),
            ppo_cfg.get("ent_decay", 1.0),
            ppo_cfg.get("ent_min", 0.0),
            total_timesteps=total_timesteps,
            n_steps=ppo_cfg.get("n_steps", 2048),
            n_envs=training_cfg.get("n_envs", 1),
        )
        entropy_min_value = float(ppo_cfg.get("ent_min", 0.0) or 0.0)
    entropy_callback: Optional[EntropyScheduleCallback] = None
    entropy_schedule: Optional[ScaledSchedule]
    if callable(raw_entropy_schedule):
        entropy_schedule = ScaledSchedule(
            ensure_schedule_callable(raw_entropy_schedule),
            min_value=entropy_min_value,
        )
        ent_coef_value = float(entropy_schedule.current_value(1.0))
        entropy_callback = EntropyScheduleCallback(
            entropy_schedule,
            log_freq=training_cfg.get("log_interval", 100),
        )
    else:
        entropy_schedule = None
        ent_coef_value = float(raw_entropy_schedule)

    LOGGER.info(
        "[%s] Optimizer settings: beta1=%.3f, beta2=%.3f, weight_decay=%.2e, eps=%s, warmup_steps=%s",
        symbol,
        beta1,
        beta2,
        weight_decay,
        optimizer_eps,
        warmup_steps,
    )
    entropy_bonus_cfg = dict(ppo_cfg.get("entropy_bonus", {}) or {})
    covariance_guard_cfg = dict(ppo_cfg.get("kl_covariance_guard", {}) or {})

    device_preference = training_cfg.get("device", "auto")
    try:
        device, device_note = resolve_device(device_preference)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to resolve compute device (%s): %s", device_preference, exc)
        raise

    if device_note:
        LOGGER.warning(device_note)

    monitor_device_label = device
    if device.startswith("cuda"):
        cuda_index = 0
        if ":" in device:
            _, index_str = device.split(":", 1)
            cuda_index = int(index_str)
        torch.cuda.set_device(cuda_index)
        gpu_name = torch.cuda.get_device_name(cuda_index)
        monitor_device_label = f"{device} ({gpu_name})"
        LOGGER.info("Using GPU device %s (%s)", device, gpu_name)
    else:
        LOGGER.info("Using device: %s", device)

    status_refresh_steps = max(1, int(training_cfg.get("status_refresh_steps", max(64, ppo_cfg.get("n_steps", 2048) // 4))))

    stability_guard: Optional[TrainingStabilityGuard] = None
    if stability_cfg.get("enabled", False):
        guard_min_lr = float(stability_cfg.get("min_lr", lr_min_value))
        lr_schedule.min_value = max(lr_schedule.min_value, guard_min_lr)
        stability_guard = TrainingStabilityGuard(
            symbol=symbol,
            lr_schedule=lr_schedule,
            total_timesteps=total_timesteps,
            approx_kl_threshold=float(stability_cfg.get("approx_kl_threshold", 0.12)),
            clip_fraction_threshold=float(stability_cfg.get("clip_fraction_threshold", 0.6)),
            value_loss_threshold=float(stability_cfg.get("value_loss_threshold", 300.0)),
            trigger_patience=int(stability_cfg.get("trigger_patience", 1)),
            cooldown_steps=int(stability_cfg.get("cooldown_steps", 8000)),
            lr_decay_factor=float(stability_cfg.get("lr_decay_factor", 0.5)),
            min_lr=guard_min_lr,
            logger_tag=str(stability_cfg.get("logger_tag", "stability")),
            max_mitigations=stability_cfg.get("max_mitigations"),
        )
        LOGGER.info(
            "[%s] Stability guard enabled: approx_kl>%.3g, clip_fraction>%.3g, value_loss>%.3g, lr_decay_factor=%.3f, cooldown=%s, min_lr=%.3e",
            symbol,
            stability_guard.approx_kl_threshold,
            stability_guard.clip_fraction_threshold,
            stability_guard.value_loss_threshold,
            stability_guard.lr_decay_factor,
            stability_guard.cooldown_steps,
            stability_guard.min_lr,
        )
    else:
        LOGGER.info("[%s] Stability guard disabled", symbol)

    if mlflow.active_run() is not None:
        optimizer_params: Dict[str, Any] = {
            "optimizer/beta1": beta1,
            "optimizer/beta2": beta2,
            "optimizer/weight_decay": weight_decay,
            "optimizer/warmup_steps": warmup_steps,
            "optimizer/warmup_floor": warmup_floor,
        }
        if optimizer_eps is not None:
            optimizer_params["optimizer/eps"] = float(optimizer_eps)
        mlflow.log_params(optimizer_params)

        if entropy_scheduler_cfg:
            scheduler_params = {
                "entropy_scheduler/strategy": entropy_scheduler_cfg.get("strategy", "hold_then_linear"),
                "entropy_scheduler/initial": entropy_scheduler_cfg.get("initial", ppo_cfg.get("ent_coef", 0.01)),
                "entropy_scheduler/final": entropy_scheduler_cfg.get("final", entropy_scheduler_cfg.get("target", ppo_cfg.get("ent_min", 0.0))),
                "entropy_scheduler/hold_steps": entropy_scheduler_cfg.get("hold_steps", 0),
                "entropy_scheduler/decay_steps": entropy_scheduler_cfg.get("decay_steps", "auto"),
                "entropy_scheduler/min": entropy_min_value,
            }
            mlflow.log_params(scheduler_params)

        if entropy_bonus_cfg:
            bonus_params = {
                f"entropy_bonus/{k}": v
                for k, v in entropy_bonus_cfg.items()
                if isinstance(v, (int, float, str, bool))
            }
            if bonus_params:
                mlflow.log_params(bonus_params)

        if covariance_guard_cfg:
            guard_params_extra = {
                f"kl_covariance_guard/{k}": v
                for k, v in covariance_guard_cfg.items()
                if isinstance(v, (int, float, str, bool))
            }
            if guard_params_extra:
                mlflow.log_params(guard_params_extra)

        guard_params: Dict[str, Any] = {"stability_guard/enabled": bool(stability_cfg.get("enabled", False))}
        if stability_guard is not None:
            guard_params.update(
                {
                    "stability_guard/approx_kl_threshold": stability_guard.approx_kl_threshold,
                    "stability_guard/clip_fraction_threshold": stability_guard.clip_fraction_threshold,
                    "stability_guard/value_loss_threshold": stability_guard.value_loss_threshold,
                    "stability_guard/trigger_patience": stability_guard.trigger_patience,
                    "stability_guard/cooldown_steps": stability_guard.cooldown_steps,
                    "stability_guard/lr_decay_factor": stability_guard.lr_decay_factor,
                    "stability_guard/min_lr": stability_guard.min_lr,
                    "stability_guard/max_mitigations": stability_guard.max_mitigations if stability_guard.max_mitigations is not None else -1,
                }
            )
        mlflow.log_params(guard_params)

        performance_guard_cfg = training_cfg.get("performance_guard", {})
        if performance_guard_cfg:
            mlflow.log_params({f"performance_guard/{k}": v for k, v in performance_guard_cfg.items()})

        entropy_guard_cfg = training_cfg.get("action_entropy_guard", {})
        if entropy_guard_cfg:
            mlflow.log_params({f"action_entropy_guard/{k}": v for k, v in entropy_guard_cfg.items()})

    env_kwargs_train = make_env_kwargs(symbol, "train", config)
    env_kwargs_val = make_env_kwargs(symbol, "val", config)

    reward_config = env_kwargs_train.get("reward_config")
    reward_snapshot: Optional[Dict[str, float]] = None
    if reward_config is not None:
        reward_snapshot = {
            "pnl_weight": reward_config.pnl_weight,
            "transaction_cost_weight": reward_config.transaction_cost_weight,
            "time_efficiency_weight": reward_config.time_efficiency_weight,
            "sharpe_weight": reward_config.sharpe_weight,
            "drawdown_weight": reward_config.drawdown_weight,
            "sizing_weight": reward_config.sizing_weight,
            "hold_penalty_weight": reward_config.hold_penalty_weight,
            "pnl_scale": reward_config.pnl_scale,
            "target_sharpe": reward_config.target_sharpe,
            "base_transaction_cost_pct": reward_config.base_transaction_cost_pct,
            "quick_win_bonus": reward_config.quick_win_bonus,
            "loss_penalty_multiplier": reward_config.loss_penalty_multiplier,
            "reward_clip": reward_config.reward_clip,
        }
        LOGGER.info("[%s] Reward configuration snapshot: %s", symbol, reward_snapshot)
        if mlflow.active_run() is not None:
            mlflow.log_dict(reward_snapshot, f"reward_config_{symbol}.json")
            mlflow.set_tag("reward_snapshot", json.dumps(reward_snapshot))

    set_random_seed(training_cfg.get("seed", 42))

    norm_cfg = training_cfg.get("reward_normalization", {})
    use_vecnormalize = bool(norm_cfg.get("enabled", False))
    vecnorm_kwargs = {
        "norm_obs": bool(norm_cfg.get("norm_obs", False)),
        "norm_reward": bool(norm_cfg.get("norm_reward", True)),
        "clip_reward": float(norm_cfg.get("clip_reward", 5.0)),
        "gamma": float(norm_cfg.get("gamma", ppo_cfg.get("gamma", 0.99))),
    }
    if use_vecnormalize:
        LOGGER.info(
            "[%s] VecNormalize enabled (norm_obs=%s, norm_reward=%s, clip_reward=%s, gamma=%s)",
            symbol,
            vecnorm_kwargs["norm_obs"],
            vecnorm_kwargs["norm_reward"],
            vecnorm_kwargs["clip_reward"],
            vecnorm_kwargs["gamma"],
        )
        if mlflow.active_run() is not None:
            mlflow.log_dict(
                {
                    "norm_obs": vecnorm_kwargs["norm_obs"],
                    "norm_reward": vecnorm_kwargs["norm_reward"],
                    "clip_reward": vecnorm_kwargs["clip_reward"],
                    "gamma": vecnorm_kwargs["gamma"],
                },
                f"vecnormalize_{symbol}.json",
            )

        monitor_info_lines = [
            f"Run start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Host: {platform.node()} ({platform.system()} {platform.release()})",
            f"Python {platform.python_version()} • Torch {torch.__version__} • SB3 {SB3_VERSION}",
            f"Device: {monitor_device_label}",
            f"Timesteps: {total_timesteps:,} • n_envs={training_cfg.get('n_envs', 1)}",
            f"Seeds → train={training_cfg.get('seed', 42)} | eval={training_cfg.get('seed', 42) + 123}",
        ]
        if use_vecnormalize:
            monitor_info_lines.append(
                "VecNormalize: on (norm_obs=%s, norm_reward=%s, clip=%.2f)"
                % (
                    "on" if vecnorm_kwargs["norm_obs"] else "off",
                    "on" if vecnorm_kwargs["norm_reward"] else "off",
                    vecnorm_kwargs["clip_reward"],
                )
            )
        else:
            monitor_info_lines.append("VecNormalize: off")
        meta = config.get("meta", {}) if isinstance(config, dict) else {}
        fingerprint = meta.get("fingerprint")
        if fingerprint:
            monitor_info_lines.append(f"Config fingerprint: {fingerprint[:12]}…")

        monitor = RichTrainingMonitor(
            symbol,
            total_timesteps=total_timesteps,
            device=monitor_device_label,
            system_info=monitor_info_lines,
        )
    else:
        LOGGER.info("[%s] VecNormalize disabled", symbol)
    vec_stats_path = checkpoint_dir / "vecnormalize.pkl"
    ret_rms = None
    obs_rms = None

    train_env: Optional[VecEnv] = None
    eval_env: Optional[VecEnv] = None
    tensorboard_logger: Optional[Logger] = None
    model: Optional[PPO] = None
    best_model_path = checkpoint_dir / "best_model.zip"
    eval_callback: Optional[Phase3EvaluationCallback] = None
    evaluation_callback_summary: Optional[Dict[str, float]] = None
    best_metric: Optional[float] = None
    summary: Dict[str, float] = {}
    episodes: List[Dict[str, Any]] = []
    duration_hours = 0.0
    promoted_model_path: Optional[Path] = None
    best_eval_summary: Optional[Dict[str, float]] = None
    best_eval_episodes: List[Dict[str, Any]] = []

    stability_halted = False
    guard_halt_reason: Optional[str] = None

    noisy_loggers = [
        "core.rl.environments",
        "core.rl.environments.reward_shaper",
        "core.rl.environments.portfolio_manager",
        "core.rl.environments.trading_env",
        "core.rl.environments.feature_extractor",
        "core.rl.environments.regime_indicators",
    ]
    silence_level = logging.CRITICAL + 10

    try:
        with monitor, TemporaryLogLevel(noisy_loggers, silence_level):
            monitor.update(status="Preparing environments", step=0)

            train_env_log_level = env_kwargs_train.get("log_level")
            n_envs = training_cfg.get("n_envs", 1)
            
            # WINDOWS FIX: Use DummyVecEnv on Windows for stability
            # SubprocVecEnv with multiprocessing on Windows is unstable (BrokenPipeError)
            # DummyVecEnv is slower but much more stable
            use_subprocess = training_cfg.get("n_envs", 1) > 1
            start_method = None
            
            if platform.system() == "Windows":
                if n_envs > 1:
                    LOGGER.warning(
                        f"⚠️  Windows detected with n_envs={n_envs}. "
                        f"Using DummyVecEnv for stability (slower but more reliable). "
                        f"To use SubprocVecEnv, run on Linux or reduce n_envs to 1."
                    )
                    use_subprocess = False  # Force DummyVecEnv on Windows
                    # Alternative: use_subprocess = True with start_method = "spawn"
            
            train_env = make_vec_trading_env(
                symbol=symbol,
                data_dir=Path(experiment_cfg["data_dir"]) / symbol,
                num_envs=n_envs,
                seed=training_cfg.get("seed", 42),
                use_subprocess=use_subprocess,
                start_method=start_method,
                env_kwargs=env_kwargs_train,
                env_log_level=train_env_log_level,
            )

            if use_vecnormalize:
                train_env = VecNormalize(train_env, **vecnorm_kwargs)

            if resume and use_vecnormalize and vec_stats_path.exists():
                train_env = VecNormalize.load(str(vec_stats_path), train_env)

            eval_env_log_level = env_kwargs_val.get("log_level")
            eval_env = make_vec_trading_env(
                symbol=symbol,
                data_dir=Path(experiment_cfg["data_dir"]) / symbol,
                num_envs=1,
                seed=training_cfg.get("seed", 42) + 123,
                use_subprocess=False,
                env_kwargs=env_kwargs_val,
                env_log_level=eval_env_log_level,
            )

            if use_vecnormalize:
                eval_env = VecNormalize(eval_env, **vecnorm_kwargs)
                eval_env.training = False
                eval_env.norm_reward = bool(vecnorm_kwargs["norm_reward"])  # type: ignore[attr-defined]

            tensorboard_logger = configure_sb3_logger(log_dir)

            if resume and best_model_path.exists():
                LOGGER.info("Resuming %s from %s", symbol, best_model_path)
                model = PPO.load(best_model_path, env=train_env, device=device)
                model.policy.to(device)
                model.verbose = 0
                model.set_logger(tensorboard_logger)
            else:
                model = PPO(
                    policy=ppo_cfg.get("policy", "MultiInputPolicy"),
                    env=train_env,
                    learning_rate=lr_schedule,
                    n_steps=ppo_cfg.get("n_steps", 2048),
                    batch_size=ppo_cfg.get("batch_size", 256),
                    n_epochs=ppo_cfg.get("n_epochs", 10),
                    gamma=ppo_cfg.get("gamma", 0.99),
                    gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
                    clip_range=ppo_cfg.get("clip_range", 0.2),
                    clip_range_vf=ppo_cfg.get("clip_range_vf"),
                    ent_coef=ent_coef_value,
                    vf_coef=ppo_cfg.get("vf_coef", 0.5),
                    max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=str(log_dir),
                    device=device,
                    verbose=0,
                    normalize_advantage=ppo_cfg.get("normalize_advantage", True),
                    target_kl=ppo_cfg.get("target_kl"),
                )
                model.set_logger(tensorboard_logger)

            model.lr_schedule = lr_schedule
            model.learning_rate = lr_schedule
            model.target_kl = ppo_cfg.get("target_kl")
            optimizer = getattr(model.policy, "optimizer", None)
            if optimizer is not None and hasattr(optimizer, "param_groups"):
                for group in optimizer.param_groups:
                    group["betas"] = (beta1, beta2)
                    group["weight_decay"] = weight_decay
                    if optimizer_eps is not None:
                        group["eps"] = float(optimizer_eps)
                optimizer.defaults["betas"] = (beta1, beta2)
                optimizer.defaults["weight_decay"] = weight_decay
                if optimizer_eps is not None:
                    optimizer.defaults["eps"] = float(optimizer_eps)
            progress_remaining = 1.0
            if total_timesteps > 0:
                progress_remaining = max(0.0, min(1.0, 1.0 - float(model.num_timesteps) / float(total_timesteps)))
            current_lr = float(lr_schedule(progress_remaining))
            update_learning_rate(model.policy.optimizer, current_lr)
            attach_training_guard(model, stability_guard)

            callbacks: List[BaseCallback] = []

            checkpoint_callback = CheckpointCallback(
                save_freq=int(training_cfg.get("save_freq", 10_000) // max(1, training_cfg.get("n_envs", 1))),
                save_path=str(checkpoint_dir),
                name_prefix=f"{symbol}_checkpoint",
            )
            callbacks.append(checkpoint_callback)

            reward_logger = RewardComponentLogger(RewardConfig().component_keys, log_freq=1000)
            callbacks.append(reward_logger)

            telemetry_cfg = dict(training_cfg.get("telemetry", {}) or {})
            telemetry_max_samples = telemetry_cfg.get("max_samples")
            if telemetry_max_samples is None:
                telemetry_max_samples = int(ppo_cfg.get("n_steps", 2048)) * int(training_cfg.get("n_envs", 1) or 1)
            telemetry_bins = telemetry_cfg.get("histogram_bins", training_cfg.get("telemetry_bins", 20) or 20)
            telemetry_log_every = telemetry_cfg.get(
                "log_every",
                telemetry_cfg.get("rollout_interval", training_cfg.get("telemetry_rollout_interval", 1) or 1),
            )
            telemetry_callback = PolicyTelemetryCallback(
                action_names=ALL_TRADE_ACTION_NAMES,
                max_samples=int(telemetry_max_samples),
                histogram_bins=int(telemetry_bins),
                log_every=int(max(1, telemetry_log_every)),
            )
            callbacks.append(telemetry_callback)

            if mlflow.active_run() is not None and telemetry_cfg:
                telemetry_params = {
                    f"telemetry/{k}": v
                    for k, v in telemetry_cfg.items()
                    if isinstance(v, (int, float, str, bool))
                }
                if telemetry_params:
                    mlflow.log_params(telemetry_params)

            if entropy_callback is not None:
                callbacks.append(entropy_callback)

            entropy_controller_enabled = entropy_schedule is not None and (
                bool(entropy_bonus_cfg.get("enabled", True)) or bool(covariance_guard_cfg.get("enabled"))
            )
            if entropy_controller_enabled and entropy_schedule is not None:
                guard_cfg = dict(covariance_guard_cfg)
                nested_guard = entropy_bonus_cfg.get("covariance_guard")
                if isinstance(nested_guard, dict):
                    guard_cfg.update(nested_guard)

                target_entropy = float(
                    entropy_bonus_cfg.get(
                        "target_entropy",
                        entropy_bonus_cfg.get("target", entropy_bonus_cfg.get("entropy_target", 0.0)),
                    )
                )
                bonus_scale = float(
                    entropy_bonus_cfg.get(
                        "bonus_scale",
                        entropy_bonus_cfg.get("scale", entropy_bonus_cfg.get("multiplier", 0.0)),
                    )
                )
                decay_rate = float(entropy_bonus_cfg.get("decay_rate", entropy_bonus_cfg.get("decay", 0.1)))
                warmup_bonus = int(entropy_bonus_cfg.get("warmup_steps", entropy_bonus_cfg.get("warmup", 0)))
                max_multiplier = entropy_bonus_cfg.get("max_multiplier")
                entropy_floor = entropy_bonus_cfg.get("floor", entropy_bonus_cfg.get("min", entropy_min_value))
                entropy_controller = EntropyAdaptiveController(
                    schedule=entropy_schedule,
                    telemetry=telemetry_callback,
                    total_timesteps=total_timesteps,
                    target_entropy=target_entropy,
                    bonus_scale=bonus_scale,
                    decay_rate=decay_rate,
                    warmup_steps=warmup_bonus,
                    max_multiplier=max_multiplier,
                    floor=entropy_floor,
                    covariance_guard=guard_cfg,
                    logger_tag=str(entropy_bonus_cfg.get("logger_tag", "entropy_adaptive")),
                )
                callbacks.append(entropy_controller)
                LOGGER.info(
                    "[%s] Entropy controller enabled (target=%.3f, bonus_scale=%.3f, warmup_steps=%s, decay=%.3f, max_multiplier=%s, guard_threshold=%s)",
                    symbol,
                    target_entropy,
                    bonus_scale,
                    warmup_bonus,
                    decay_rate,
                    max_multiplier,
                    guard_cfg.get("threshold"),
                )

            mlflow_callback = MLflowCallback(log_freq=training_cfg.get("log_interval", 100))
            callbacks.append(mlflow_callback)

            callbacks.append(
                RichStatusCallback(
                    monitor,
                    total_timesteps=total_timesteps,
                    refresh_steps=status_refresh_steps,
                )
            )

            if training_cfg.get("early_stopping", {}).get("enabled", True):
                early_cfg = training_cfg.get("early_stopping", {})
                eval_callback = Phase3EvaluationCallback(
                    eval_env=eval_env,
                    eval_freq=training_cfg.get("eval_freq", 5000),
                    n_eval_episodes=training_cfg.get("n_eval_episodes", 10),
                    patience=early_cfg.get("patience", 5),
                    min_delta=early_cfg.get("min_delta", 0.0),
                    monitor_metric=early_cfg.get("metric", "sharpe"),
                    checkpoint_on_best=training_cfg.get("checkpoint_on_best", True),
                    checkpoint_dir=checkpoint_dir,
                    performance_guard=training_cfg.get("performance_guard"),
                    action_entropy_guard=training_cfg.get("action_entropy_guard"),
                    success_thresholds=config.get("validation", {}).get("success_thresholds", {}),
                    deterministic=config.get("validation", {}).get("deterministic", True),
                    monitor=monitor,
                    entropy_controller=entropy_callback,
                    total_timesteps=total_timesteps,
                )
                callbacks.append(eval_callback)
            else:
                eval_callback = None

            start_time = datetime.now()
            monitor.update(status="Training", step=0)

            try:
                model.learn(
                    total_timesteps=total_timesteps,
                    callback=CallbackList(callbacks),
                    log_interval=training_cfg.get("log_interval", 100),
                    tb_log_name=symbol,
                )
            except TrainingHalted as halt_exc:
                stability_halted = True
                LOGGER.warning(
                    "[%s] Training halted by stability guard at %s steps: %s",
                    symbol,
                    model.num_timesteps,
                    halt_exc,
                )
                if monitor and monitor.enabled:
                    monitor.update(status="Stability Halt", step=model.num_timesteps)
                if mlflow.active_run() is not None:
                    mlflow.log_metric("training/stability_halted", 1, step=model.num_timesteps)
            finally:
                if use_vecnormalize and isinstance(train_env, VecNormalize):
                    ret_rms = train_env.ret_rms
                    if vecnorm_kwargs.get("norm_obs", False):
                        obs_rms = train_env.obs_rms
                    train_env.save(str(vec_stats_path))
                    if mlflow.active_run() is not None:
                        mlflow.log_artifact(str(vec_stats_path))
                if train_env is not None:
                    train_env.close()
                    train_env = None
                if model is not None:
                    detach_training_guard(model)

            if eval_callback is not None and eval_callback.halt_reason:
                guard_halt_reason = eval_callback.halt_reason

            duration_hours = (datetime.now() - start_time).total_seconds() / 3600.0
            LOGGER.info("Training complete for %s in %.2f hours", symbol, duration_hours)
            if stability_halted:
                LOGGER.info("[%s] Stability guard halted further updates after %.0f%% of scheduled timesteps", symbol, 100.0 * model.num_timesteps / max(1, total_timesteps))
            if mlflow.active_run() is not None:
                if stability_guard is not None:
                    mlflow.log_metric("stability/mitigation_events", stability_guard.mitigation_count, step=model.num_timesteps)
                mlflow.log_metric("training/stability_halted_final", 1 if stability_halted else 0, step=model.num_timesteps)
                mlflow.log_metric("training/guard_halted_final", 1 if guard_halt_reason else 0, step=model.num_timesteps)
                if guard_halt_reason:
                    mlflow.set_tag("training_guard_reason", guard_halt_reason)

            ensure_directory(checkpoint_dir)
            final_model_path = checkpoint_dir / "final_model.zip"
            model.save(final_model_path)

            monitor.update(status="Evaluating", step=total_timesteps)

            if eval_env is not None:
                if use_vecnormalize and isinstance(eval_env, VecNormalize):
                    if ret_rms is not None:
                        eval_env.ret_rms = ret_rms  # type: ignore[attr-defined]
                    if vecnorm_kwargs.get("norm_obs", False) and obs_rms is not None:
                        eval_env.obs_rms = obs_rms  # type: ignore[attr-defined]
                    eval_env.training = False

                n_validation_episodes = config.get("validation", {}).get("n_val_episodes", training_cfg.get("n_eval_episodes", 10))
                deterministic_eval = config.get("validation", {}).get("deterministic", True)

                summary, episodes = run_evaluation(
                    model,
                    eval_env,
                    n_validation_episodes,
                    deterministic=deterministic_eval,
                )

                promotion_metric_name = (
                    training_cfg.get("promotion_metric")
                    or training_cfg.get("early_stopping", {}).get("metric")
                    or "sharpe"
                )
                promotion_min_delta = float(training_cfg.get("promotion_min_delta", 0.0))

                best_eval_metric: Optional[float] = None

                if eval_callback is not None:
                    if eval_callback.best_metric != -float("inf"):
                        best_eval_metric = float(eval_callback.best_metric)
                    if eval_callback.best_summary:
                        best_eval_summary = dict(eval_callback.best_summary)
                    if not best_eval_episodes and eval_callback.history:
                        target_metric = best_eval_metric
                        for entry in eval_callback.history:
                            candidate_value = _extract_metric_value(entry.summary, promotion_metric_name)
                            if target_metric is None or candidate_value is None:
                                continue
                            if math.isclose(candidate_value, target_metric, rel_tol=1e-9, abs_tol=1e-9):
                                best_eval_episodes = entry.episodes
                                if best_eval_summary is None:
                                    best_eval_summary = dict(entry.summary)
                                break

                final_metric_value = _extract_metric_value(summary, promotion_metric_name)

                if best_eval_summary is not None:
                    evaluation_callback_summary = best_eval_summary

                if best_eval_metric is not None and (best_metric is None or best_eval_metric > best_metric):
                    best_metric = best_eval_metric

                if (
                    best_eval_metric is not None
                    and final_metric_value is not None
                    and best_model_path.exists()
                    and best_eval_metric > final_metric_value + promotion_min_delta
                ):
                    promoted_model_path = checkpoint_dir / "promoted_model.zip"
                    shutil.copy2(best_model_path, promoted_model_path)
                    LOGGER.info(
                        "[%s] Promoted best checkpoint (metric %.4f) over final model (metric %.4f)",
                        symbol,
                        best_eval_metric,
                        final_metric_value,
                    )
                    if mlflow.active_run() is not None:
                        mlflow.log_metric("promotion/final_metric", final_metric_value, step=total_timesteps)
                        mlflow.log_metric("promotion/best_metric", best_eval_metric, step=total_timesteps)
                        mlflow.log_param("promotion_metric", promotion_metric_name)
                        mlflow.log_artifact(str(promoted_model_path))

                eval_env.close()
                eval_env = None
            else:
                summary = {}
                episodes = []

            if mlflow.active_run() is not None:
                mlflow.log_metrics(
                    {f"final_eval/{k}": float(v) for k, v in summary.items() if isinstance(v, (int, float))},
                    step=total_timesteps,
                )
                mlflow.log_metric("training/duration_hours", duration_hours, step=total_timesteps)
                mlflow.log_dict(summary, "final_evaluation_summary.json")
                mlflow.log_dict({"episodes": episodes}, "final_evaluation_episodes.json")
                mlflow.log_artifact(str(final_model_path))
                if best_eval_summary is not None:
                    mlflow.log_dict(best_eval_summary, "best_evaluation_summary.json")
                    mlflow.log_dict({"episodes": best_eval_episodes}, "best_evaluation_episodes.json")

            if eval_callback is not None:
                callback_best_summary = eval_callback.best_summary
                if callback_best_summary and best_eval_summary is None:
                    best_eval_summary = callback_best_summary
                metric_candidate = eval_callback.best_metric if eval_callback.best_metric != -float("inf") else None
                if metric_candidate is not None and (best_metric is None or metric_candidate > best_metric):
                    best_metric = metric_candidate
                evaluation_callback_summary = best_eval_summary or callback_best_summary
                history_payload = [
                    {
                        "step": entry.step,
                        "summary": entry.summary,
                        "episodes": entry.episodes,
                    }
                    for entry in eval_callback.history
                ]
                history_path = checkpoint_dir / "evaluation_history.json"
                history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
                if mlflow.active_run() is not None:
                    mlflow.log_artifact(str(history_path))

            monitor.complete(
                metrics={
                    "eval_reward_mean": summary.get("episode_reward_mean"),
                    "eval_reward_std": summary.get("episode_reward_std"),
                    "eval_episode_length": summary.get("episode_length_mean"),
                    "last_eval_sharpe": summary.get("sharpe_ratio_mean"),
                    "eval_return_pct": summary.get("total_return_pct_mean"),
                    "best_sharpe": best_metric if best_metric is not None else summary.get("sharpe_ratio_mean"),
                    "last_eval_step": total_timesteps,
                    "stability_halted": 1.0 if stability_halted else 0.0,
                    "guard_halted": 1.0 if guard_halt_reason else 0.0,
                }
            )
    except Exception:
        if eval_env is not None:
            eval_env.close()
        raise

    meta = config.get("meta", {})

    selected_model_path = promoted_model_path if promoted_model_path is not None else final_model_path
    resolved_best_summary = best_eval_summary or evaluation_callback_summary

    status_value = "success"
    if stability_halted:
        status_value = "stability_halted"
    elif guard_halt_reason:
        status_value = "guard_halted"

    result = TrainingResult(
        symbol=symbol,
        status=status_value,
        duration_hours=duration_hours,
        total_timesteps=total_timesteps,
        final_model_path=str(selected_model_path),
        promoted_model_path=str(promoted_model_path) if promoted_model_path is not None else None,
        best_metric=best_metric,
        best_summary=resolved_best_summary,
        final_summary=summary,
        checkpoints=[str(path) for path in checkpoint_dir.glob("*.zip")],
        config_fingerprint=meta.get("fingerprint"),
        reward_snapshot=reward_snapshot,
        halt_reason=guard_halt_reason,
    )

    if resolved_best_summary is not None:
        best_summary_path = checkpoint_dir / "best_evaluation_summary.json"
        best_summary_payload = {
            "summary": resolved_best_summary,
            "episodes": best_eval_episodes,
            "config_fingerprint": meta.get("fingerprint"),
            "stability_halted": stability_halted,
            "halt_reason": guard_halt_reason,
        }
        best_summary_path.write_text(json.dumps(best_summary_payload, indent=2), encoding="utf-8")

    summary_path = checkpoint_dir / "final_evaluation_summary.json"
    summary_payload = {
        "summary": summary,
        "episodes": episodes,
        "best_summary": resolved_best_summary,
        "best_episodes": best_eval_episodes,
        "promotion_metric": training_cfg.get("promotion_metric")
        or training_cfg.get("early_stopping", {}).get("metric")
        or "sharpe",
        "promoted_model_path": str(promoted_model_path) if promoted_model_path is not None else None,
        "final_model_path": str(final_model_path),
        "reward_snapshot": reward_snapshot,
        "config_fingerprint": meta.get("fingerprint"),
        "stability_halted": stability_halted,
        "halt_reason": guard_halt_reason,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    LOGGER.info("Final evaluation for %s: %s", symbol, summary)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Agent Training")
    parser.add_argument("--config", required=True, help="Path to configuration YAML")
    parser.add_argument("--symbols", nargs="*", help="Optional subset of symbols to train")
    parser.add_argument("--resume", type=str, default=None, help="Symbol to resume from best checkpoint")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total timesteps for quick experiments")
    parser.add_argument("--n-envs", type=int, default=None, help="Override number of parallel environments")
    parser.add_argument("--eval-freq", type=int, default=None, help="Override evaluation frequency in timesteps")
    parser.add_argument("--save-freq", type=int, default=None, help="Override checkpoint save frequency in timesteps")
    parser.add_argument("--device", type=str, default=None, help="Compute device to use (auto, cpu, cuda, or cuda:<index>)")
    parser.add_argument("--seed", type=int, default=None, help="Override training seed for reproducibility sweeps")
    parser.add_argument("--sequential", action="store_true", help="Reserved for future multi-process orchestration")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_config(config_path)

    training_cfg = config.setdefault("training", {})

    if args.total_timesteps is not None:
        training_cfg["total_timesteps"] = int(args.total_timesteps)
        LOGGER.info("Overriding total_timesteps to %s", training_cfg["total_timesteps"])

    if args.n_envs is not None:
        training_cfg["n_envs"] = max(1, int(args.n_envs))
        LOGGER.info("Overriding n_envs to %s", training_cfg["n_envs"])

    if args.eval_freq is not None:
        training_cfg["eval_freq"] = max(1, int(args.eval_freq))
        LOGGER.info("Overriding eval_freq to %s", training_cfg["eval_freq"])

    if args.save_freq is not None:
        training_cfg["save_freq"] = max(1, int(args.save_freq))
        LOGGER.info("Overriding save_freq to %s", training_cfg["save_freq"])

    if args.device is not None:
        training_cfg["device"] = args.device

    if args.seed is not None:
        training_cfg["seed"] = int(args.seed)
        LOGGER.info("Overriding seed to %s", training_cfg["seed"])

    symbols = args.symbols if args.symbols else config["experiment"].get("symbols", [])
    if not symbols:
        raise ValueError("No symbols specified for training")

    if config["experiment"].get("mlflow_uri"):
        mlflow.set_tracking_uri(config["experiment"]["mlflow_uri"])

    results: List[TrainingResult] = []

    for symbol in symbols:
        resume_flag = args.resume == symbol if args.resume else False
        try:
            prepare_mlflow(config, symbol)
            result = train_symbol_agent(symbol, config, resume=resume_flag)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Training failed for %s", symbol)
            results.append(
                TrainingResult(
                    symbol=symbol,
                    status="failed",
                    duration_hours=0.0,
                    total_timesteps=0,
                    error=str(exc),
                    config_fingerprint=config.get("meta", {}).get("fingerprint"),
                )
            )
        finally:
            close_mlflow()

    summary_path = Path(config["experiment"].get("checkpoint_dir", "models/phase3_checkpoints")) / "training_summary.json"
    ensure_directory(summary_path.parent)
    summary_path.write_text(
        json.dumps([result.__dict__ for result in results], indent=2),
        encoding="utf-8",
    )

    LOGGER.info("Training summary written to %s", summary_path)

    successes = [r for r in results if r.status == "success"]
    failures = [r for r in results if r.status != "success"]

    LOGGER.info("Successful: %s / %s", len(successes), len(results))
    if successes:
        avg_duration = np.mean([r.duration_hours for r in successes])
        LOGGER.info("Average duration per symbol: %.2f hours", avg_duration)

    if failures:
        LOGGER.warning("Failures encountered for symbols: %s", ", ".join(r.symbol for r in failures))


if __name__ == "__main__":
    main()