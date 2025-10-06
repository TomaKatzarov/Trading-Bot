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

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from rich.console import Console, Group
from rich.errors import LiveError
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

from core.rl.environments import PortfolioConfig, RewardConfig
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


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {config_path} is not a mapping")
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


def build_reward_config(weights: Dict[str, float], base: Optional[RewardConfig] = None) -> RewardConfig:
    """Convert reward weight dictionary into a RewardConfig instance."""

    config = RewardConfig() if base is None else base
    mapping = {
        "pnl_weight": weights.get("pnl", config.pnl_weight),
        "transaction_cost_weight": weights.get("cost", config.transaction_cost_weight),
        "time_efficiency_weight": weights.get("time", config.time_efficiency_weight),
        "sharpe_weight": weights.get("sharpe", config.sharpe_weight),
        "drawdown_weight": weights.get("drawdown", config.drawdown_weight),
        "sizing_weight": weights.get("sizing", config.sizing_weight),
        "hold_penalty_weight": weights.get("hold", config.hold_penalty_weight),
        "base_transaction_cost_pct": weights.get("transaction_cost_pct", config.base_transaction_cost_pct),
    }

    for attr, value in mapping.items():
        setattr(config, attr, float(value))

    return config


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
    portfolio_config = build_portfolio_config(env_cfg)

    env_kwargs: Dict[str, Any] = {
        "symbol": symbol,
        "data_path": data_path,
        "sl_checkpoints": env_cfg.get("sl_checkpoints", {}),
        "lookback_window": int(env_cfg.get("lookback_window", 24)),
        "episode_length": int(env_cfg.get("episode_length", 168)),
        "initial_capital": float(env_cfg.get("initial_capital", 100_000.0)),
        "commission_rate": float(env_cfg.get("commission_rate", 0.001)),
        "slippage_bps": float(env_cfg.get("slippage_pct", 0.0005)) * 10_000.0,
        "stop_loss": float(env_cfg.get("stop_loss_pct", 0.02)),
        "take_profit": float(env_cfg.get("take_profit_pct", 0.025)),
        "max_hold_hours": int(env_cfg.get("max_hold_hours", 8)),
        "reward_config": reward_config,
        "portfolio_config": portfolio_config,
        "log_trades": bool(env_cfg.get("log_trades", False)),
    }

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
    ) -> None:
        self.symbol = symbol
        self.total_timesteps = max(1, int(total_timesteps))
        self.console = console or Console(force_terminal=True, highlight=False)
        self.refresh_per_second = max(1, int(refresh_per_second))
        disable_live = os.environ.get("PHASE3_DISABLE_RICH") or os.environ.get("RICH_NO_LIVE")
        if disable_live is None:
            disable_live = os.environ.get("CI")
        self.enabled = not (isinstance(disable_live, str) and disable_live.strip().lower() in {"1", "true", "yes", "on"})
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
        self._start_time = time.perf_counter()
        self._last_render = 0.0
        self.progress = Progress(
            SpinnerColumn(style="bold magenta"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
            refresh_per_second=self.refresh_per_second,
        )
        self._task_id = self.progress.add_task(self._progress_description(), total=self.total_timesteps)
        self.live: Optional[Live] = None

    def __enter__(self) -> "RichTrainingMonitor":
        if not self.enabled:
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

    def _progress_description(self) -> str:
        return f"{self.symbol} • {self.status}"

    def render(self) -> Panel:
        if self.enabled:
            self.progress.update(self._task_id, completed=self.current_step, description=self._progress_description())

        info_table = Table.grid(expand=True)
        info_table.add_column(ratio=1, justify="left")
        info_table.add_column(ratio=1, justify="right")

        elapsed = time.perf_counter() - self._start_time
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

        content = Group(self.progress, info_table)
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
        self._start_time = time.perf_counter()

    def _on_training_start(self) -> None:
        if self.monitor.enabled:
            device = getattr(self.model, "device", None)
            self.monitor.update(metrics={"device": str(device) if device is not None else "--"}, status="Training")

    def _on_step(self) -> bool:
        if not self.monitor.enabled:
            return True
        if self.num_timesteps - self._last_update_step < self.refresh_steps and self.num_timesteps < self.total_timesteps:
            return True
        self._last_update_step = self.num_timesteps

        elapsed = time.perf_counter() - self._start_time
        fps = self.num_timesteps / elapsed if elapsed > 0 else float("nan")
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

    return summary


def run_evaluation(model: PPO, eval_env: VecEnv, n_episodes: int, deterministic: bool = True) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Run evaluation episodes and return summary plus per-episode metrics."""

    episodes: List[Dict[str, Any]] = []

    for _ in range(n_episodes):
        obs = eval_env.reset()
        state = None
        episode_reward = 0.0
        while True:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, rewards, dones, infos = eval_env.step(action)
            episode_reward += float(rewards[0])
            if dones[0]:
                base_env = eval_env.envs[0] if hasattr(eval_env, "envs") else None
                if base_env is None:
                    raise RuntimeError("Evaluation environment does not expose underlying environment")
                info = infos[0]
                info.setdefault("episode", {})
                info["episode"].setdefault("r", episode_reward)
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

    def _on_step(self) -> bool:
        progress_remaining = getattr(self.model, "_current_progress_remaining", None)
        if progress_remaining is None:
            total = float(getattr(self.model, "_total_timesteps", 1) or 1)
            progress_remaining = 1.0 - float(self.num_timesteps) / total
        progress_remaining = float(max(0.0, min(1.0, progress_remaining)))

        entropy_value = float(self.schedule(progress_remaining))
        self.model.ent_coef = entropy_value

        if self.model.logger is not None and self.n_calls % self.log_freq == 0:
            self.model.logger.record("train/entropy_coef", entropy_value)

        if mlflow.active_run() is not None and self.n_calls % self.log_freq == 0:
            mlflow.log_metric("train/entropy_coef", entropy_value, step=self.num_timesteps)

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
        success_thresholds: Optional[Dict[str, float]] = None,
        deterministic: bool = True,
        verbose: int = 0,
        monitor: Optional[RichTrainingMonitor] = None,
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

        self.history: List[EvaluationHistoryEntry] = []
        self.best_metric: float = -float("inf")
        self.best_summary: Optional[Dict[str, float]] = None
        self.no_improvement_steps: int = 0
        self._last_eval_step: int = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True

        self._last_eval_step = self.num_timesteps

        if self.monitor and self.monitor.enabled:
            self.monitor.update(status=f"Evaluating @ {self.num_timesteps:,} steps")

        summary, episodes = run_evaluation(
            self.model,
            self.eval_env,
            self.n_eval_episodes,
            deterministic=self.deterministic,
        )
        self.history.append(EvaluationHistoryEntry(self.num_timesteps, summary, episodes))

        if self.monitor and self.monitor.enabled:
            self.monitor.update_evaluation(summary=summary, step=self.num_timesteps)

        log_evaluation_to_logger(self.model.logger, summary, self.num_timesteps)

        if mlflow.active_run() is not None:
            mlflow.log_metrics(
                {f"eval/{k}": float(v) for k, v in summary.items() if isinstance(v, (int, float))},
                step=self.num_timesteps,
            )

        metric_key = f"{self.monitor_metric}_mean"
        metric_value = summary.get(metric_key)

        if metric_value is not None and np.isfinite(metric_value):
            if metric_value > self.best_metric + self.min_delta:
                self.best_metric = float(metric_value)
                self.best_summary = dict(summary)
                self.no_improvement_steps = 0

                if self.checkpoint_on_best:
                    ensure_directory(self.checkpoint_dir)
                    best_path = self.checkpoint_dir / "best_model.zip"
                    self.model.save(best_path)
                    LOGGER.info("New best %s=%.4f at %s timesteps. Saved %s", metric_key, metric_value, self.num_timesteps, best_path)
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
            "ent_coef": config["ppo"].get("ent_coef"),
            "vf_coef": config["ppo"].get("vf_coef"),
            "max_grad_norm": config["ppo"].get("max_grad_norm"),
            "total_timesteps": config["training"].get("total_timesteps"),
            "n_envs": config["training"].get("n_envs", 1),
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
    best_metric: Optional[float] = None
    best_summary: Optional[Dict[str, float]] = None
    final_summary: Optional[Dict[str, float]] = None
    checkpoints: Optional[List[str]] = None
    error: Optional[str] = None


def train_symbol_agent(symbol: str, config: Dict[str, Any], resume: bool = False) -> TrainingResult:
    LOGGER.info("Starting training for %s", symbol)
    experiment_cfg = config["experiment"]
    training_cfg = config["training"]
    ppo_cfg = config["ppo"]

    checkpoint_dir = ensure_directory(Path(experiment_cfg["checkpoint_dir"]) / symbol)
    log_dir = ensure_directory(Path(experiment_cfg["log_dir"]) / symbol)

    policy_kwargs = resolve_policy_kwargs(ppo_cfg.get("policy_kwargs", {}))
    lr_schedule = create_lr_schedule(
        ppo_cfg.get("learning_rate", 3e-4),
        ppo_cfg.get("lr_schedule", "constant"),
        ppo_cfg.get("lr_min", 1e-5),
    )
    entropy_schedule = create_entropy_schedule(
        ppo_cfg.get("ent_coef", 0.01),
        ppo_cfg.get("ent_decay", 1.0),
        ppo_cfg.get("ent_min", 0.0),
        total_timesteps=training_cfg.get("total_timesteps", 100_000),
        n_steps=ppo_cfg.get("n_steps", 2048),
        n_envs=training_cfg.get("n_envs", 1),
    )
    entropy_callback: Optional[EntropyScheduleCallback] = None
    if callable(entropy_schedule):
        ent_coef_value = float(entropy_schedule(1.0))
        entropy_callback = EntropyScheduleCallback(
            entropy_schedule,
            log_freq=training_cfg.get("log_interval", 100),
        )
    else:
        ent_coef_value = float(entropy_schedule)

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

    total_timesteps = int(training_cfg.get("total_timesteps", 100_000))
    status_refresh_steps = max(1, int(training_cfg.get("status_refresh_steps", max(64, ppo_cfg.get("n_steps", 2048) // 4))))
    monitor = RichTrainingMonitor(symbol, total_timesteps=total_timesteps, device=monitor_device_label)

    env_kwargs_train = make_env_kwargs(symbol, "train", config)
    env_kwargs_val = make_env_kwargs(symbol, "val", config)

    set_random_seed(training_cfg.get("seed", 42))

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

            train_env = make_vec_trading_env(
                symbol=symbol,
                data_dir=Path(experiment_cfg["data_dir"]) / symbol,
                num_envs=training_cfg.get("n_envs", 1),
                seed=training_cfg.get("seed", 42),
                use_subprocess=training_cfg.get("n_envs", 1) > 1,
                env_kwargs=env_kwargs_train,
            )

            eval_env = make_vec_trading_env(
                symbol=symbol,
                data_dir=Path(experiment_cfg["data_dir"]) / symbol,
                num_envs=1,
                seed=training_cfg.get("seed", 42) + 123,
                use_subprocess=False,
                env_kwargs=env_kwargs_val,
            )

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
                )
                model.set_logger(tensorboard_logger)

            callbacks: List[BaseCallback] = []

            checkpoint_callback = CheckpointCallback(
                save_freq=int(training_cfg.get("save_freq", 10_000) // max(1, training_cfg.get("n_envs", 1))),
                save_path=str(checkpoint_dir),
                name_prefix=f"{symbol}_checkpoint",
            )
            callbacks.append(checkpoint_callback)

            reward_logger = RewardComponentLogger(RewardConfig().component_keys, log_freq=1000)
            callbacks.append(reward_logger)

            if entropy_callback is not None:
                callbacks.append(entropy_callback)

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
                    success_thresholds=config.get("validation", {}).get("success_thresholds", {}),
                    deterministic=config.get("validation", {}).get("deterministic", True),
                    monitor=monitor,
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
            finally:
                if train_env is not None:
                    train_env.close()
                    train_env = None

            duration_hours = (datetime.now() - start_time).total_seconds() / 3600.0
            LOGGER.info("Training complete for %s in %.2f hours", symbol, duration_hours)

            ensure_directory(checkpoint_dir)
            final_model_path = checkpoint_dir / "final_model.zip"
            model.save(final_model_path)

            monitor.update(status="Evaluating", step=total_timesteps)

            if eval_env is not None:
                summary, episodes = run_evaluation(
                    model,
                    eval_env,
                    config.get("validation", {}).get("n_val_episodes", training_cfg.get("n_eval_episodes", 10)),
                    deterministic=config.get("validation", {}).get("deterministic", True),
                )
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

            if eval_callback is not None:
                evaluation_callback_summary = eval_callback.best_summary
                metric_candidate = eval_callback.best_metric if eval_callback.best_metric != -float("inf") else None
                if metric_candidate is not None:
                    best_metric = metric_candidate
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
                }
            )
    except Exception:
        if eval_env is not None:
            eval_env.close()
        raise

    result = TrainingResult(
        symbol=symbol,
        status="success",
        duration_hours=duration_hours,
        total_timesteps=total_timesteps,
        final_model_path=str(final_model_path),
        best_metric=best_metric,
        best_summary=evaluation_callback_summary,
        final_summary=summary,
        checkpoints=[str(path) for path in checkpoint_dir.glob("*.zip")],
    )

    summary_path = checkpoint_dir / "final_evaluation_summary.json"
    summary_path.write_text(json.dumps({"summary": summary, "episodes": episodes}, indent=2), encoding="utf-8")

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
