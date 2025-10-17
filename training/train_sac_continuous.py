"""Train Soft Actor-Critic on the continuous trading environment."""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import logging
import math
import os
import shutil
import warnings
from contextlib import nullcontext
from collections import Counter, deque, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Mapping

# Suppress PyTorch TF32 deprecation warnings before torch import
warnings.filterwarnings("ignore", message=".*TF32.*", category=UserWarning)

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from importlib import util as importlib_util

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import polyak_update, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import DictReplayBufferSamples, ReplayBufferSamples
from gymnasium import spaces

from core.rl.curiosity.icm import ICMConfig, TradingICM
from core.rl.environments.action_space_migrator import ActionSpaceMigrator
from core.rl.policies import EncoderConfig, FeatureEncoder
from training.rl.env_factory import build_trading_config, load_yaml
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

LOGGER = logging.getLogger("training.sac")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s")


def _mirror_artifact(source: Path, dest_dir: Path, base_name: str) -> None:
    """Mirror a saved artifact to canonical names for compatibility."""
    if source is None or not source.exists():
        return

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Unable to create mirror directory %s: %s", dest_dir, exc)
        return

    suffix = "".join(source.suffixes)
    targets = [dest_dir / f"{base_name}{suffix}", dest_dir / f"{base_name}_latest{suffix}"]
    for target in targets:
        try:
            shutil.copy2(source, target)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to mirror %s to %s: %s", source, target, exc)

# Enable high-throughput math kernels when CUDA is available.
if torch.cuda.is_available():
    # Use new PyTorch 2.9+ API for TF32 precision control (suppresses warnings)
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except RuntimeError:
            pass
    
    # Configure TF32 for cuDNN convolutions using new API
    if hasattr(torch.backends, "cudnn"):
        cudnn = torch.backends.cudnn
        # Try new API first (PyTorch 2.9+)
        if hasattr(cudnn, "conv") and hasattr(cudnn.conv, "fp32_precision"):
            try:
                cudnn.conv.fp32_precision = "tf32"
            except (AttributeError, RuntimeError):
                pass
        
        # Try new API for matmul (PyTorch 2.9+)
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                try:
                    torch.backends.cuda.matmul.fp32_precision = "tf32"
                except (AttributeError, RuntimeError):
                    pass
    
    # Enable cudnn benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True

def build_cosine_warmup_schedule(
    base_lr: float,
    warmup_pct: float,
    *,
    min_lr_fraction: float = 0.05,
    num_cycles: int = 3,
) -> Callable[[float], float]:
    base_lr = float(base_lr)
    warmup_pct = float(np.clip(warmup_pct, 0.0, 1.0))
    min_lr_fraction = float(np.clip(min_lr_fraction, 0.0, 1.0))
    num_cycles = max(1, int(num_cycles))
    min_lr = base_lr * min_lr_fraction

    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - float(progress_remaining)
        progress = float(np.clip(progress, 0.0, 1.0))

        if warmup_pct > 0.0 and progress < warmup_pct:
            fraction = progress / max(1e-6, warmup_pct)
            lr = base_lr * fraction
        else:
            if warmup_pct >= 1.0:
                lr = base_lr
            else:
                post_progress = (progress - warmup_pct) / max(1e-6, 1.0 - warmup_pct)
                post_progress = float(np.clip(post_progress, 0.0, 1.0 - 1e-8))
                cycle_position = (post_progress * num_cycles) % 1.0
                cosine = 0.5 * (1.0 + np.cos(np.pi * cycle_position))
                lr = base_lr * (min_lr_fraction + (1.0 - min_lr_fraction) * cosine)

        return float(max(min_lr, lr))

    return schedule


def _discover_phase3_split_paths(sym: str, experiment_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data_dir = experiment_cfg.get("data_dir") if isinstance(experiment_cfg, dict) else None
    if not data_dir:
        return None

    root = Path(data_dir).expanduser() / sym
    train_path = root / "train.parquet"
    if not train_path.exists():
        return None

    val_path = root / "val.parquet"
    test_path = root / "test.parquet"
    metadata_path = root / "metadata.json"
    metadata: Optional[Dict[str, Any]] = None

    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load metadata for %s: %s", sym, exc)
            metadata = None

    return {
        "train": train_path,
        "val": val_path if val_path.exists() else None,
        "test": test_path if test_path.exists() else None,
        "metadata": metadata,
    }


def resolve_symbol_data_paths(
    sym: str,
    base_env_cfg: Dict[str, Any],
    experiment_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    split_info = _discover_phase3_split_paths(sym, experiment_cfg)
    if split_info is not None:
        return split_info

    existing = base_env_cfg.get("data_path")
    if existing:
        existing_path = Path(existing).expanduser()
        if existing_path.exists():
            return {"train": existing_path, "val": None, "test": None, "metadata": None}
        base_symbol = base_env_cfg.get("symbol")
        if isinstance(base_symbol, str) and base_symbol:
            try:
                substituted = Path(str(existing_path).replace(base_symbol, sym))
            except Exception:
                substituted = None
            else:
                if substituted and substituted.exists():
                    return {"train": substituted, "val": None, "test": None, "metadata": None}

    candidate = Path(f"data/historical/{sym}/1Hour/data.parquet")
    if candidate.exists():
        return {"train": candidate, "val": None, "test": None, "metadata": None}

    demo = Path(f"data/demo/{sym}.parquet")
    return {"train": demo if demo.exists() else candidate, "val": None, "test": None, "metadata": None}


class SharedFrozenFeatureExtractor(BaseFeaturesExtractor):
    """Wrapper that injects a pretrained transformer encoder into SB3."""

    def __init__(self, observation_space, shared_encoder: FeatureEncoder) -> None:  # type: ignore[override]
        super().__init__(observation_space, features_dim=shared_encoder.config.output_dim)
        object.__setattr__(self, "_shared_encoder", shared_encoder)
        self._shared_encoder.eval()
        for param in self._shared_encoder.parameters():
            param.requires_grad_(False)

    def forward(self, observations):  # type: ignore[override]
        with torch.no_grad():
            return self._shared_encoder(observations)


class ContinuousActionMonitor(BaseCallback):
    """Track raw and smoothed continuous actions during training."""

    def __init__(self, log_freq: int = 100) -> None:
        super().__init__()
        self.log_freq = max(1, int(log_freq))
        self.samples: List[float] = []
        self.final_stats: Dict[str, float] = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos:
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            continuous = info0.get("continuous_action") if isinstance(info0, dict) else None
            if continuous:
                smoothed = continuous.get("smoothed", continuous.get("raw"))
                if smoothed is not None:
                    self.samples.append(float(smoothed))
        if self.n_calls % self.log_freq == 0 and self.samples:
            data = np.asarray(self.samples, dtype=np.float32)
            self.final_stats["action_mean"] = float(np.nanmean(data))
            self.final_stats["action_std"] = float(np.nanstd(data))
            self.final_stats["action_min"] = float(np.nanmin(data))
            self.final_stats["action_max"] = float(np.nanmax(data))
            
            self.logger.record("continuous/action_mean", self.final_stats["action_mean"])
            self.logger.record("continuous/action_std", self.final_stats["action_std"])
            self.logger.record("continuous/action_min", self.final_stats["action_min"])
            self.logger.record("continuous/action_max", self.final_stats["action_max"])

            hist, _ = np.histogram(data, bins=20, range=(-1.0, 1.0))
            total = hist.sum()
            if total > 0:
                for idx, count in enumerate(hist):
                    self.logger.record(f"continuous/action_bin_{idx}", int(count))

            self.samples.clear()
        return True


class EntropyTracker(BaseCallback):
    """Approximate action entropy using a rolling histogram."""

    def __init__(self, window: int = 512, bins: int = 21, log_freq: int = 100) -> None:
        super().__init__()
        self.window = max(10, int(window))
        self.bins = max(5, int(bins))
        self.log_freq = max(1, int(log_freq))
        self.buffer: deque[float] = deque(maxlen=self.window)
        self.final_entropy: Optional[float] = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos:
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            continuous = info0.get("continuous_action") if isinstance(info0, dict) else None
            if continuous:
                smoothed = continuous.get("smoothed", continuous.get("raw"))
                if smoothed is not None:
                    self.buffer.append(float(smoothed))

        if self.n_calls % self.log_freq == 0 and len(self.buffer) >= 5:
            data = np.asarray(self.buffer, dtype=np.float32)
            counts, _ = np.histogram(data, bins=self.bins, range=(-1.0, 1.0))
            total = counts.sum()
            if total > 0:
                probs = counts / total
                probs = probs[probs > 0]
                entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
                self.final_entropy = entropy
                self.logger.record("continuous/entropy", entropy)
        return True


class RewardBreakdownLogger(BaseCallback):
    """Aggregate reward breakdown components from env infos for diagnostics."""

    def __init__(self, log_freq: int = 256) -> None:
        super().__init__()
        self.log_freq = max(1, int(log_freq))
        self.buffer: defaultdict[str, List[float]] = defaultdict(list)
        self.components_buffer: defaultdict[str, List[float]] = defaultdict(list)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos:
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            breakdown = info0.get("reward_breakdown") if isinstance(info0, Mapping) else None
            if isinstance(breakdown, Mapping):
                for key, value in breakdown.items():
                    try:
                        self.buffer[key].append(float(value))
                    except (TypeError, ValueError):
                        continue

            components = info0.get("reward_components") if isinstance(info0, Mapping) else None
            if isinstance(components, Mapping):
                for key, value in components.items():
                    try:
                        self.components_buffer[key].append(float(value))
                    except (TypeError, ValueError):
                        continue

        if self.n_calls % self.log_freq == 0 and self.buffer:
            for key, values in self.buffer.items():
                if values:
                    mean_val = float(np.mean(values))
                    self.logger.record(f"reward_breakdown/{key}", mean_val)
                    values.clear()
            for key, values in self.components_buffer.items():
                if values:
                    mean_val = float(np.mean(values))
                    self.logger.record(f"reward_components/{key}", mean_val)
                    values.clear()
        return True


class TradeMetricsCallback(BaseCallback):
    """Log trade execution metrics from environment info payloads."""

    def __init__(self, log_freq: int = 1000) -> None:
        super().__init__()
        self.log_freq = max(1, int(log_freq))
        self.counter: Counter[str] = Counter()
        self.total_steps = 0
        self.final_trade_rate: float = 0.0
        self.holding_period_sum: float = 0.0
        self.closed_trade_count: int = 0
        self.commission_sum: float = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        self.total_steps += 1
        if infos:
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            executed = info0.get("executed_action_name") if isinstance(info0, dict) else None
            if executed:
                self.counter[executed] += 1
            closed = info0.get("position_closed") if isinstance(info0, dict) else None
            if isinstance(closed, Mapping):
                holding_steps = float(closed.get("holding_period", 0.0))
                commission = float(closed.get("commission", 0.0))
                self.holding_period_sum += holding_steps
                self.commission_sum += commission
                self.closed_trade_count += 1

        if self.total_steps % self.log_freq == 0 and self.counter:
            total_trades = sum(count for name, count in self.counter.items() if name != "HOLD")
            trade_rate = total_trades / max(1, self.total_steps)
            self.final_trade_rate = trade_rate
            self.logger.record("trading/trade_rate", trade_rate)
            for name, count in self.counter.items():
                self.logger.record(f"trading/actions/{name}", int(count))
            if self.closed_trade_count > 0:
                avg_hold = self.holding_period_sum / float(self.closed_trade_count)
                self.logger.record("trading/avg_hold_steps", avg_hold)
                self.logger.record("trading/commission_sum", self.commission_sum)
        return True


class SACMetricsCapture(BaseCallback):
    """Capture final SAC training metrics for summary reporting."""

    def __init__(self) -> None:
        super().__init__()
        self.final_metrics: Dict[str, float] = {}

    def _on_step(self) -> bool:
        # Capture latest metrics from the logger
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            try:
                # Try to get the last logged values
                name_to_value = getattr(self.model.logger, 'name_to_value', {})
                if name_to_value:
                    for key in ['train/actor_loss', 'train/critic_loss', 'train/ent_coef', 'train/learning_rate', 'train/n_updates']:
                        if key in name_to_value:
                            self.final_metrics[key] = float(name_to_value[key])
            except Exception:
                pass
        return True


class SACMetricsCapture(BaseCallback):
    """Capture final SAC training metrics for summary reporting."""

    def __init__(self) -> None:
        super().__init__()
        self.final_metrics: Dict[str, float] = {}

    def _on_step(self) -> bool:
        # Capture latest metrics from the logger
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            try:
                # Try to get the last logged values
                name_to_value = getattr(self.model.logger, 'name_to_value', {})
                if name_to_value:
                    for key in ['train/actor_loss', 'train/critic_loss', 'train/ent_coef', 'train/learning_rate', 'train/n_updates']:
                        if key in name_to_value:
                            self.final_metrics[key] = float(name_to_value[key])
            except Exception:
                pass
        return True


class PeriodicCheckpointCallback(BaseCallback):
    """Save policy checkpoints at a fixed step interval during training."""

    def __init__(self, save_freq: int, save_path: Path, prefix: str = "checkpoint") -> None:
        super().__init__()
        self.save_freq = max(1, int(save_freq))
        self.save_path = Path(save_path)
        self.prefix = prefix

    def _init_callback(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            checkpoint_path = self.save_path / f"{self.prefix}_{self.num_timesteps:07d}.zip"
            try:
                self.model.save(str(checkpoint_path), exclude=["env", "eval_env", "replay_buffer"])
            except Exception as exc:
                if checkpoint_path.exists():
                    try:
                        checkpoint_path.unlink()
                    except OSError:
                        pass
                fallback_path = self.save_path / f"{self.prefix}_{self.num_timesteps:07d}_policy.pt"
                torch.save(self.model.policy.state_dict(), fallback_path)
                LOGGER.warning(
                    "Periodic checkpoint fallback saved policy weights to %s after serialization error: %s",
                    fallback_path.resolve(),
                    exc,
                )
            else:
                LOGGER.info("Periodic checkpoint saved to %s", checkpoint_path.resolve())
        return True


# ------------------------------
# Rich UI: Monitor & Callbacks
# ------------------------------


def _fmt_float(v: float, prec: int = 2) -> str:
    try:
        f = float(v)
    except Exception:
        return "--"
    return f"{f:.{prec}f}" if np.isfinite(f) else "--"


def _fmt_time(seconds: float) -> str:
    try:
        s = int(seconds)
    except Exception:
        return "--"
    if s < 0:
        return "--"
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class RichTrainingMonitor:
    def __init__(self, symbol: str, total_timesteps: int, *, console: Optional[Console] = None, refresh_per_second: int = 4) -> None:
        self.symbol = symbol
        self.total_timesteps = max(1, int(total_timesteps))
        self.console = console or Console(highlight=False)
        self.refresh_per_second = max(1, int(refresh_per_second))
        self.status = "Initializing"
        self.current_step = 0
        self.metrics: Dict[str, Any] = {
            "device": "--",
            "train_reward_mean": float("nan"),
            "train_reward_std": float("nan"),
            "eval_reward_mean": float("nan"),
            "eval_reward_std": float("nan"),
            "eval_pnl_mean": float("nan"),
            "eval_pnl_std": float("nan"),
            "eval_episode_length": float("nan"),
            "fps": float("nan"),
            "eta_seconds": float("nan"),
            "best_sharpe": float("nan"),
            "last_eval_sharpe": float("nan"),
            "eval_return_pct": float("nan"),
            "eval_return_pct_std": float("nan"),
            "eval_sharpe_std": float("nan"),
            "last_eval_step": None,
        }
        try:
            from datetime import UTC
            self._start_time = datetime.now(UTC)
        except ImportError:
            self._start_time = datetime.now(timezone.utc)
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
        self._task_id = self.progress.add_task(self._desc(), total=self.total_timesteps, fps="--")
        self.live: Optional[Live] = None

    def _desc(self) -> str:
        return f"{self.symbol} • {self.status}"

    def __enter__(self) -> "RichTrainingMonitor":
        info = Text()
        info.append("• ", style="dim"); info.append(f"Symbol: {self.symbol}", style="cyan"); info.append("\n")
        start_str = self._start_time.strftime('%Y-%m-%d %H:%M:%S UTC') if hasattr(self._start_time, 'strftime') else str(self._start_time)
        info.append("• ", style="dim"); info.append(f"Start: {start_str}", style="cyan")
        self.console.print(Panel(Align.left(info), border_style="bright_blue", title="[bold cyan]Run info"))
        self.live = Live(self.render(), console=self.console, refresh_per_second=self.refresh_per_second, transient=False)
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:  # noqa: ANN001
        try:
            self.update(status="Error" if exc_type else "Complete", step=self.total_timesteps)
            if self.live is not None:
                self.live.update(self.render())
        finally:
            if self.live is not None:
                self.live.__exit__(exc_type, exc, exc_tb)
                self.live = None
        self._emit_summary()

    def _emit_summary(self) -> None:
        text = Text()
        pnl_mean = self.metrics.get("eval_pnl_mean")
        pnl_std = self.metrics.get("eval_pnl_std")
        pnl_str = _fmt_float(pnl_mean)
        if pnl_str != "--" and pnl_std is not None and np.isfinite(float(pnl_std)):
            pnl_str = f"{pnl_str} ±{_fmt_float(pnl_std)}"

        roi_mean = self.metrics.get("eval_return_pct")
        roi_std = self.metrics.get("eval_return_pct_std")
        roi_str = _fmt_float(roi_mean, 2)
        if roi_str != "--" and roi_std is not None and np.isfinite(float(roi_std)):
            roi_str = f"{roi_str} ±{_fmt_float(roi_std, 2)}"

        sharpe_last = self.metrics.get("last_eval_sharpe")
        sharpe_std = self.metrics.get("eval_sharpe_std")
        sharpe_str = _fmt_float(sharpe_last, 3)
        if sharpe_str != "--" and sharpe_std is not None and np.isfinite(float(sharpe_std)):
            sharpe_str = f"{sharpe_str} ±{_fmt_float(sharpe_std, 3)}"

        pairs = [
            ("Status", str(self.status)),
            ("Best Sharpe", _fmt_float(self.metrics.get("best_sharpe"), 3)),
            ("Eval Sharpe", sharpe_str),
            ("Eval Return %", roi_str),
            ("Eval PnL μ/σ", pnl_str),
            ("Last Eval Step", str(self.metrics.get("last_eval_step") or "--")),
        ]
        for i, (k, v) in enumerate(pairs):
            text.append(f"{k}: ", style="dim"); text.append(str(v), style="bold")
            if i < len(pairs) - 1:
                text.append("\n")
        self.console.print(Panel(Align.left(text), border_style="bright_magenta", title=f"[bold]SAC Summary • {self.symbol}"))

    def update(self, *, step: Optional[int] = None, status: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None) -> None:
        if step is not None:
            self.current_step = max(0, min(int(step), self.total_timesteps))
        if status is not None:
            self.status = status
        if metrics:
            self.metrics.update(metrics)
        if self.live is not None:
            self.live.update(self.render())

    def update_eval(
        self,
        *,
        sharpe_mean: Optional[float],
        ret_pct_mean: Optional[float],
        reward_mean: Optional[float],
        reward_std: Optional[float],
        ep_len_mean: Optional[float],
        step: int,
        sharpe_std: Optional[float] = None,
        ret_pct_std: Optional[float] = None,
    ) -> None:
        best = self.metrics.get("best_sharpe")
        best_val = float(best) if isinstance(best, (int, float)) and np.isfinite(float(best)) else float("-inf")
        if sharpe_mean is not None and np.isfinite(float(sharpe_mean)) and float(sharpe_mean) > best_val:
            self.metrics["best_sharpe"] = float(sharpe_mean)
        updates = {
            "last_eval_sharpe": float(sharpe_mean) if sharpe_mean is not None else float("nan"),
            "eval_return_pct": float(ret_pct_mean) if ret_pct_mean is not None else float("nan"),
            "eval_reward_mean": float(reward_mean) if reward_mean is not None else float("nan"),
            "eval_reward_std": float(reward_std) if reward_std is not None else float("nan"),
            "eval_episode_length": float(ep_len_mean) if ep_len_mean is not None else float("nan"),
            "last_eval_step": int(step),
        }
        updates["eval_pnl_mean"] = updates["eval_reward_mean"]
        updates["eval_pnl_std"] = updates["eval_reward_std"]
        if sharpe_std is not None and np.isfinite(float(sharpe_std)):
            updates["eval_sharpe_std"] = float(sharpe_std)
        if ret_pct_std is not None and np.isfinite(float(ret_pct_std)):
            updates["eval_return_pct_std"] = float(ret_pct_std)
        self.update(metrics=updates)

    def render(self) -> Panel:
        # update progress fields
        fps_text = _fmt_float(self.metrics.get("fps"), 0)
        self.progress.update(self._task_id, completed=self.current_step, description=self._desc(), fps=fps_text)

        # info table
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=1, justify="right")

        roi_mean = self.metrics.get("eval_return_pct")
        roi_std = self.metrics.get("eval_return_pct_std")
        roi_display = _fmt_float(roi_mean, 2)
        if roi_display != "--" and roi_std is not None and np.isfinite(float(roi_std)):
            roi_display = f"{roi_display} ±{_fmt_float(roi_std, 2)}"

        pnl_mean = self.metrics.get("eval_pnl_mean")
        pnl_std = self.metrics.get("eval_pnl_std")
        pnl_display = _fmt_float(pnl_mean)
        if pnl_display != "--" and pnl_std is not None and np.isfinite(float(pnl_std)):
            pnl_display = f"{pnl_display} ±{_fmt_float(pnl_std)}"

        sharpe_last = self.metrics.get("last_eval_sharpe")
        sharpe_std = self.metrics.get("eval_sharpe_std")
        sharpe_display = _fmt_float(sharpe_last, 3)
        if sharpe_display != "--" and sharpe_std is not None and np.isfinite(float(sharpe_std)):
            sharpe_display = f"{sharpe_display} ±{_fmt_float(sharpe_std, 3)}"

        table.add_row(
            Text.from_markup(f"[dim]Status:[/] [bold]{self.status}[/]"),
            Text.from_markup(f"[dim]Device:[/] [bold]{self.metrics.get('device','--')}[/]"),
        )
        table.add_row(
            Text.from_markup(
                f"[dim]Train Reward μ/σ:[/] [bold]{_fmt_float(self.metrics.get('train_reward_mean'))}/{_fmt_float(self.metrics.get('train_reward_std'))}[/]"
            ),
            Text.from_markup(f"[dim]Eval PnL μ/σ:[/] [bold]{pnl_display}[/]"),
        )
        table.add_row(
            Text.from_markup(f"[dim]Best Sharpe:[/] [bold]{_fmt_float(self.metrics.get('best_sharpe'),3)}[/]"),
            Text.from_markup(f"[dim]Last Eval Sharpe:[/] [bold]{sharpe_display}[/]"),
        )
        table.add_row(
            Text.from_markup(f"[dim]Eval Return %:[/] [bold]{roi_display}[/]"),
            Text.from_markup(f"[dim]Last Eval Step:[/] [bold]{self.metrics.get('last_eval_step') or '--'}[/]"),
        )
        table.add_row(
            Text.from_markup(f"[dim]FPS:[/] [bold]{_fmt_float(self.metrics.get('fps'),0)}[/]"),
            Text.from_markup(f"[dim]ETA:[/] [bold]{_fmt_time(self.metrics.get('eta_seconds'))}[/]"),
        )

        content = Group(self.progress, table)
        return Panel(content, border_style="bright_magenta", title=f"[bold]SAC Training • {self.symbol}")


class RichStatusCallback(BaseCallback):
    def __init__(self, monitor: RichTrainingMonitor, *, total_timesteps: int, refresh_steps: int = 512) -> None:
        super().__init__()
        self.monitor = monitor
        self.total_timesteps = max(1, int(total_timesteps))
        self.refresh_steps = max(1, int(refresh_steps))
        self._last_step = 0
        try:
            from datetime import UTC
            self._last_time = datetime.now(UTC).timestamp()
        except ImportError:
            self._last_time = datetime.now(timezone.utc).timestamp()
        self._smoothed_fps = float("nan")

    def _on_training_start(self) -> None:
        device = getattr(self.model, "device", None)
        self.monitor.update(status="Training", metrics={"device": str(device) if device is not None else "--"})

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_step) < self.refresh_steps and self.num_timesteps < self.total_timesteps:
            return True
        try:
            from datetime import UTC
            now = datetime.now(UTC).timestamp()
        except ImportError:
            now = datetime.now(timezone.utc).timestamp()
        dt = max(1e-6, now - self._last_time)
        dsteps = max(0, self.num_timesteps - self._last_step)
        inst_fps = dsteps / dt
        if np.isfinite(inst_fps):
            if np.isfinite(self._smoothed_fps):
                fps = 0.7 * self._smoothed_fps + 0.3 * inst_fps
            else:
                fps = inst_fps
            self._smoothed_fps = fps
        else:
            fps = self._smoothed_fps

        remaining = max(0, self.total_timesteps - self.num_timesteps)
        eta = float(remaining / fps) if np.isfinite(fps) and fps > 1e-9 else float("nan")

        ep_rewards = [info.get("r", 0.0) for info in self.model.ep_info_buffer][-10:]
        reward_mean = float(np.mean(ep_rewards)) if ep_rewards else float("nan")
        reward_std = float(np.std(ep_rewards)) if ep_rewards else float("nan")

        if self.logger is not None:
            if np.isfinite(reward_mean):
                self.logger.record("train/reward_mean", reward_mean)
            if np.isfinite(reward_std):
                self.logger.record("train/reward_std", reward_std)

        self.monitor.update(
            step=self.num_timesteps,
            metrics={
                "fps": fps,
                "eta_seconds": eta,
                "train_reward_mean": reward_mean,
                "train_reward_std": reward_std,
            },
        )
        self._last_step = self.num_timesteps
        self._last_time = now
        return True


class ContinuousEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        monitor: RichTrainingMonitor,
        *,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        best_model_save_path: Optional[str] = None,
        latest_model_mirror_dir: Optional[str] = None,
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        self.eval_env = eval_env
        self.monitor = monitor
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = max(1, int(n_eval_episodes))
        self.best_model_save_path = Path(best_model_save_path) if best_model_save_path else None
        self.latest_model_mirror_dir = Path(latest_model_mirror_dir) if latest_model_mirror_dir else None
        self.deterministic = deterministic
        self.best_sharpe: float = float("-inf")
        self._saved_best = False
        self._best_artifact_path: Optional[Path] = None
        self._next_eval_step: int = self.eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval_step:
            return True

        freq = max(1, self.eval_freq)
        # Run evaluations until caught up with current timestep
        while self.num_timesteps >= self._next_eval_step:
            try:
                summary = self._run_evaluation()
                self._process_evaluation_summary(summary, step=self.num_timesteps)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Evaluation step failed: %s", exc)
            finally:
                self._next_eval_step += freq
        return True

    def _run_evaluation(self) -> Dict[str, float]:
        """Run evaluation episodes, utilizing parallel environments if available.
        
        PERFORMANCE OPTIMIZATION: If eval_env has multiple parallel environments,
        episodes run simultaneously with GPU batching and multiprocessing, reducing
        evaluation time from minutes to seconds.
        """
        num_eval_envs = getattr(self.eval_env, 'num_envs', 1)
        episodes: List[Dict[str, Any]] = []
        
        # Calculate how many parallel batches we need
        episodes_remaining = self.n_eval_episodes
        
        while episodes_remaining > 0:
            # Run up to num_eval_envs episodes in parallel
            batch_size = min(episodes_remaining, num_eval_envs)
            
            obs = self.eval_env.reset()
            episode_rewards = np.zeros(batch_size, dtype=np.float32)
            episode_lengths = np.zeros(batch_size, dtype=np.int32)
            active_envs = np.ones(batch_size, dtype=bool)
            
            # Run until all environments in this batch finish
            while active_envs.any():
                # Batched GPU inference for all active environments
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, rewards, dones, infos = self.eval_env.step(action)
                
                # Update metrics for active environments
                for env_idx in range(batch_size):
                    if active_envs[env_idx]:
                        episode_rewards[env_idx] += float(rewards[env_idx])
                        episode_lengths[env_idx] += 1
                        
                        if bool(dones[env_idx]):
                            # Episode finished - collect metrics
                            active_envs[env_idx] = False
                            info = infos[env_idx] if env_idx < len(infos) else {}
                            metrics = info.get("terminal_metrics", {})
                            total_pnl = float(metrics.get("total_pnl", episode_rewards[env_idx]))
                            sharpe = float(metrics.get("sharpe_ratio", float("nan")))
                            ret_pct = float(metrics.get("total_return_pct", float("nan")))
                            
                            episodes.append({
                                "episode_reward": total_pnl,
                                "episode_length": int(episode_lengths[env_idx]),
                                "sharpe_ratio": sharpe,
                                "total_return_pct": ret_pct,
                                "total_pnl": total_pnl,
                            })
            
            episodes_remaining -= batch_size
        
        if not episodes:
            return {}
        def sm(key: str) -> float:
            vals = [float(ep.get(key, float("nan"))) for ep in episodes]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")
        def ss(key: str) -> float:
            vals = [float(ep.get(key, float("nan"))) for ep in episodes]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.std(vals)) if vals else float("nan")
        return {
            "episode_reward_mean": sm("episode_reward"),
            "episode_reward_std": ss("episode_reward"),
            "episode_length_mean": sm("episode_length"),
            "sharpe_ratio_mean": sm("sharpe_ratio"),
            "sharpe_ratio_std": ss("sharpe_ratio"),
            "total_return_pct_mean": sm("total_return_pct"),
            "total_return_pct_std": ss("total_return_pct"),
            "total_pnl_mean": sm("total_pnl"),
            "total_pnl_std": ss("total_pnl"),
        }

    def _process_evaluation_summary(self, summary: Dict[str, float], *, step: int) -> None:
        if not summary:
            return

        sharpe_mean = summary.get("sharpe_ratio_mean")
        ret_pct_mean = summary.get("total_return_pct_mean")
        pnl_mean = summary.get("total_pnl_mean", summary.get("episode_reward_mean"))
        pnl_std = summary.get("total_pnl_std", summary.get("episode_reward_std"))
        ep_len_mean = summary.get("episode_length_mean")
        sharpe_std = summary.get("sharpe_ratio_std")
        ret_pct_std = summary.get("total_return_pct_std")

        improved = False
        if sharpe_mean is not None and np.isfinite(float(sharpe_mean)):
            current = float(sharpe_mean)
            if current > self.best_sharpe:
                self.best_sharpe = current
                improved = True

        if improved:
            self._save_best()

        if self.model.logger is not None:
            for key, value in summary.items():
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    self.model.logger.record(f"eval/{key}", float(value))
            self.model.logger.dump(step)

        reward_mean = pnl_mean
        reward_std = pnl_std
        self.monitor.update_eval(
            sharpe_mean=sharpe_mean,
            ret_pct_mean=ret_pct_mean,
            reward_mean=reward_mean,
            reward_std=reward_std,
            ep_len_mean=ep_len_mean,
            step=step,
            sharpe_std=sharpe_std,
            ret_pct_std=ret_pct_std,
        )

    def _save_best(self, *, force: bool = False) -> None:
        if self.best_model_save_path is None:
            return

        try:
            self.best_model_save_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to create best-model directory %s: %s", self.best_model_save_path.parent, exc)

        saved_path: Optional[Path] = None
        try:
            self.model.save(str(self.best_model_save_path), exclude=["env", "eval_env", "replay_buffer"])
            saved_path = self.best_model_save_path
        except Exception as exc:  # noqa: BLE001
            fallback_path = self.best_model_save_path.with_suffix(".policy.pt")
            try:
                torch.save(self.model.policy.state_dict(), fallback_path)
                saved_path = fallback_path
                LOGGER.warning(
                    "Best-model serialization fell back to policy weights at %s: %s",
                    fallback_path.resolve(),
                    exc,
                )
            except Exception as fallback_exc:  # noqa: BLE001
                LOGGER.error(
                    "Failed to persist best model to %s (and fallback %s): %s / %s",
                    self.best_model_save_path.resolve(),
                    fallback_path.resolve(),
                    exc,
                    fallback_exc,
                )
        finally:
            if saved_path is None and force:
                try:
                    forced = self.best_model_save_path.with_suffix(".last_policy.pt")
                    torch.save(self.model.policy.state_dict(), forced)
                    saved_path = forced
                    LOGGER.warning(
                        "Forced fallback saved latest policy weights to %s",
                        forced.resolve(),
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error("Forced fallback failed, no best-model artifact produced: %s", exc)

        if saved_path is not None:
            self._saved_best = True
            self._best_artifact_path = saved_path
            LOGGER.info("Best model updated at %s", saved_path.resolve())
            self._mirror_best(saved_path)

    def _mirror_best(self, artifact: Path) -> None:
        if self.latest_model_mirror_dir is None:
            return
        try:
            self.latest_model_mirror_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Unable to create latest best-model directory %s: %s", self.latest_model_mirror_dir, exc)
            return
        dest = self.latest_model_mirror_dir / artifact.name
        try:
            shutil.copy2(artifact, dest)
            LOGGER.info("Best model mirror updated at %s", dest.resolve())
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to mirror best model to %s: %s", dest, exc)

    def _on_training_end(self) -> None:
        if self.best_model_save_path is None:
            return

        if not self._saved_best or not self.best_model_save_path.exists():
            try:
                summary = self._run_evaluation()
                self._process_evaluation_summary(summary, step=self.num_timesteps)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Final evaluation failed; forcing best-model save: %s", exc)
            if not self.best_model_save_path.exists() or not self._saved_best:
                self._save_best(force=True)

    def get_best_model_artifact(self) -> Optional[Path]:
        return self._best_artifact_path


class SACWithICM(SAC):
    """Soft Actor-Critic augmented with an Intrinsic Curiosity Module."""

    def __init__(
        self,
        *args,
        icm_settings: Optional[Dict[str, Any]] = None,
        ent_coef_lower_bound: float = 0.0,
        **kwargs,
    ) -> None:
        self.icm_settings = icm_settings or {}
        self.icm_enabled = bool(self.icm_settings.get("enabled", False))
        self.icm_module: Optional[TradingICM] = None
        self.icm_optimizer: Optional[torch.optim.Optimizer] = None
        self.icm_extrinsic_weight: float = float(self.icm_settings.get("extrinsic_weight", 0.9))
        self.icm_intrinsic_weight: float = float(self.icm_settings.get("intrinsic_weight", 0.1))
        self._icm_base_intrinsic_weight: float = self.icm_intrinsic_weight
        self._icm_last_decayed_weight: float = self._icm_base_intrinsic_weight
        self.icm_decay_final_weight: float = float(self.icm_settings.get("intrinsic_final_weight", 0.0))
        self.icm_decay_after_steps: int = int(self.icm_settings.get("intrinsic_decay_after_steps", -1))
        self.icm_decay_duration: int = (
            max(1, int(self.icm_settings.get("intrinsic_decay_duration", 1)))
            if self.icm_decay_after_steps >= 0
            else 1
        )
        self.icm_decay_type: str = str(self.icm_settings.get("intrinsic_decay_type", "linear")).strip().lower()
        self._icm_has_decay: bool = self.icm_decay_after_steps >= 0
        self.icm_min_intrinsic_weight: float = float(self.icm_settings.get("minimum_intrinsic_weight", 0.0))
        self.icm_warmup_weight: float = float(self.icm_settings.get("intrinsic_warmup_weight", 0.0))
        self.icm_warmup_steps: int = max(0, int(self.icm_settings.get("intrinsic_warmup_steps", 0)))
        self.icm_disable_when_positive_sharpe: bool = bool(
            self.icm_settings.get("disable_when_positive_sharpe", False)
        )
        self.icm_disable_sharpe_threshold: float = float(self.icm_settings.get("disable_sharpe_threshold", 0.0))
        self.icm_resume_sharpe_threshold: float = float(self.icm_settings.get("resume_sharpe_threshold", -0.1))
        self.icm_disable_reward_std_threshold: float = float(
            self.icm_settings.get("disable_reward_std_threshold", float("inf"))
        )
        self.icm_train_freq: int = max(1, int(self.icm_settings.get("train_freq", 1)))
        self.icm_warmup_steps: int = max(0, int(self.icm_settings.get("warmup_steps", 1_000)))
        self.icm_max_grad_norm: float = float(self.icm_settings.get("max_grad_norm", 1.0))
        self.icm_last_metrics: Dict[str, float] = {}
        self._icm_step_counter: int = 0
        self._icm_obs_keys: Optional[List[str]] = None
        self._icm_use_policy_features: bool = bool(self.icm_settings.get("use_policy_features", False))
        self._icm_gate_active: bool = False
        self._icm_current_intrinsic_weight: float = self.icm_intrinsic_weight
        self._icm_last_eval_sharpe: float = float("nan")
        self._icm_last_reward_std: float = float("nan")
        self.ent_coef_lower_bound: float = float(ent_coef_lower_bound)
        super().__init__(*args, **kwargs)

        if self.icm_enabled:
            self._init_icm()

    def _init_icm(self) -> None:
        features_dim = getattr(self.policy.features_extractor, "features_dim", None)
        fallback_dim: Optional[int]
        obs_space = self.observation_space
        if isinstance(obs_space, spaces.Dict):
            self._icm_obs_keys = sorted(obs_space.spaces.keys())
            fallback_dim = int(sum(np.prod(space.shape or (1,)) for space in obs_space.spaces.values()))
            if not self._icm_use_policy_features:
                features_dim = None  # Prefer flattened observations for ICM to avoid duplicate encodes
        elif hasattr(obs_space, "shape") and getattr(obs_space, "shape") is not None:
            fallback_dim = int(np.prod(obs_space.shape))  # type: ignore[arg-type]
        else:
            fallback_dim = features_dim if features_dim is not None else 512

        if "state_dim" in self.icm_settings:
            state_dim = int(self.icm_settings["state_dim"])
        else:
            candidate = features_dim if features_dim is not None else fallback_dim
            state_dim = int(candidate if candidate is not None else 512)
        action_dim = int(np.prod(self.action_space.shape))

        icm_config = ICMConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=int(self.icm_settings.get("hidden_dim", 256)),
            feature_dim=int(self.icm_settings.get("feature_dim", 128)),
            beta=float(self.icm_settings.get("beta", 0.2)),
            eta=float(self.icm_settings.get("eta", 0.01)),
            extrinsic_weight=self.icm_extrinsic_weight,
            intrinsic_weight=self.icm_intrinsic_weight,
        )

        self.icm_module = TradingICM(icm_config).to(self.device)
        icm_lr = float(self.icm_settings.get("icm_lr", 1e-4))
        icm_weight_decay = float(self.icm_settings.get("weight_decay", 1e-5))
        self.icm_optimizer = torch.optim.AdamW(self.icm_module.parameters(), lr=icm_lr, weight_decay=icm_weight_decay)
        self.icm_last_metrics = {
            "forward_loss": 0.0,
            "inverse_loss": 0.0,
            "total_loss": 0.0,
            "intrinsic_reward_mean": 0.0,
            "decayed_weight": float(self._icm_base_intrinsic_weight),
        }

    def _encode_observation(self, observations: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            encoded = self.policy.extract_features(observations)
        return encoded.detach()

    def _flatten_observations(self, observations: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(observations, Mapping):
            if not self._icm_obs_keys:
                self._icm_obs_keys = sorted(observations.keys())
            parts: List[torch.Tensor] = []
            for key in self._icm_obs_keys:
                tensor = observations[key]
                parts.append(tensor.view(tensor.shape[0], -1))
            return torch.cat(parts, dim=1)
        return observations.view(observations.shape[0], -1)

    def _get_last_eval_sharpe(self) -> float:
        logger = getattr(self, "logger", None)
        if logger is None:
            return float("nan")
        name_to_value = getattr(logger, "name_to_value", None)
        if not isinstance(name_to_value, dict):
            return float("nan")
        candidate = name_to_value.get("eval/sharpe_ratio_mean")
        if candidate is None:
            candidate = name_to_value.get("eval/sharpe_ratio")
        try:
            return float(candidate) if candidate is not None else float("nan")
        except (TypeError, ValueError):
            return float("nan")

    def _compute_intrinsic_weight_with_decay(self) -> float:
        if self.icm_warmup_steps > 0 and self.num_timesteps <= self.icm_warmup_steps:
            decayed = float(self.icm_warmup_weight)
            self._icm_last_decayed_weight = decayed
            return decayed

        base = self._icm_base_intrinsic_weight
        if not self._icm_has_decay:
            self._icm_last_decayed_weight = base
            return base

        if self.num_timesteps <= self.icm_decay_after_steps:
            decayed = base
        else:
            progress = (self.num_timesteps - self.icm_decay_after_steps) / float(self.icm_decay_duration)
            progress = float(np.clip(progress, 0.0, 1.0))
            final = self.icm_decay_final_weight
            if self.icm_decay_type == "cosine":
                decayed = final + (base - final) * 0.5 * (1.0 + np.cos(np.pi * progress))
            else:
                decayed = base + (final - base) * progress

        decayed = float(decayed)
        self._icm_last_decayed_weight = decayed
        return decayed

    def _update_icm_gate(self, reward_std: float) -> None:
        sharpe = self._get_last_eval_sharpe()
        self._icm_last_eval_sharpe = sharpe
        self._icm_last_reward_std = float(reward_std)

        decayed_weight = self._compute_intrinsic_weight_with_decay()

        should_disable = False
        if self.icm_disable_when_positive_sharpe and np.isfinite(sharpe):
            if sharpe >= self.icm_disable_sharpe_threshold:
                should_disable = True

        if np.isfinite(self.icm_disable_reward_std_threshold):
            if reward_std <= self.icm_disable_reward_std_threshold:
                should_disable = True

        if should_disable:
            self._icm_gate_active = True
        elif self._icm_gate_active:
            resume = False
            if self.icm_disable_when_positive_sharpe and np.isfinite(sharpe):
                if sharpe <= self.icm_resume_sharpe_threshold:
                    resume = True
            if np.isfinite(self.icm_disable_reward_std_threshold):
                if reward_std > self.icm_disable_reward_std_threshold:
                    resume = True
            if resume:
                self._icm_gate_active = False

        if self._icm_gate_active:
            target_weight = self.icm_min_intrinsic_weight
        else:
            target_weight = decayed_weight

        target_weight = float(max(self.icm_min_intrinsic_weight, target_weight))
        self._icm_current_intrinsic_weight = target_weight

        self.icm_last_metrics.update(
            {
                "gate_active": float(1.0 if self._icm_gate_active else 0.0),
                "intrinsic_weight": float(self._icm_current_intrinsic_weight),
                "reward_std": float(reward_std),
                "decayed_weight": float(decayed_weight),
            }
        )
        if np.isfinite(sharpe):
            self.icm_last_metrics["eval_sharpe"] = float(sharpe)

        if self.logger is not None:
            self.logger.record("icm/gate_active", 1.0 if self._icm_gate_active else 0.0)
            self.logger.record("icm/intrinsic_weight", float(self._icm_current_intrinsic_weight))
            self.logger.record("icm/decayed_intrinsic_weight", float(decayed_weight))
            if np.isfinite(sharpe):
                self.logger.record("icm/last_eval_sharpe", float(sharpe))
            if np.isfinite(reward_std):
                self.logger.record("icm/replay_reward_std", float(reward_std))

    def _apply_icm(self, replay_data: Union[DictReplayBufferSamples, ReplayBufferSamples]) -> Union[DictReplayBufferSamples, ReplayBufferSamples]:
        if not self.icm_enabled or self.icm_module is None or self.icm_optimizer is None:
            return replay_data

        if self.num_timesteps <= self.icm_warmup_steps:
            return replay_data

        observations = replay_data.observations
        next_observations = replay_data.next_observations

        rewards_tensor = replay_data.rewards.detach()
        reward_std = (
            float(torch.std(rewards_tensor, unbiased=False).item()) if rewards_tensor.numel() > 1 else float(0.0)
        )
        self._update_icm_gate(reward_std)

        if self._icm_current_intrinsic_weight <= 1e-9:
            return replay_data

        if self._icm_use_policy_features or not isinstance(observations, Mapping):
            state_features = self._encode_observation(observations)
            next_state_features = self._encode_observation(next_observations)
        else:
            state_features = self._flatten_observations(observations)
            next_state_features = self._flatten_observations(next_observations)

        self.icm_module.train()
        intrinsic_reward, losses = self.icm_module(state_features, next_state_features, replay_data.actions)
        intrinsic_detached = intrinsic_reward.detach()

        intrinsic_scale = float(self._icm_current_intrinsic_weight)
        augmented_rewards = (
            self.icm_extrinsic_weight * replay_data.rewards
            + intrinsic_scale * intrinsic_detached.unsqueeze(-1)
        )

        if isinstance(replay_data, DictReplayBufferSamples):
            updated_samples: Union[DictReplayBufferSamples, ReplayBufferSamples] = DictReplayBufferSamples(
                observations=replay_data.observations,
                actions=replay_data.actions,
                next_observations=replay_data.next_observations,
                dones=replay_data.dones,
                rewards=augmented_rewards,
                discounts=replay_data.discounts,
            )
        else:
            updated_samples = ReplayBufferSamples(
                observations=replay_data.observations,
                actions=replay_data.actions,
                next_observations=replay_data.next_observations,
                dones=replay_data.dones,
                rewards=augmented_rewards,
                discounts=replay_data.discounts,
            )

        self._icm_step_counter += 1
        if (self._icm_step_counter % self.icm_train_freq == 0) and not self._icm_gate_active:
            self.icm_optimizer.zero_grad()
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.icm_module.parameters(), self.icm_max_grad_norm)
            self.icm_optimizer.step()

            self.icm_last_metrics = {
                "forward_loss": float(losses["forward_loss"].detach().item()),
                "inverse_loss": float(losses["inverse_loss"].detach().item()),
                "total_loss": float(losses["total_loss"].detach().item()),
                "intrinsic_reward_mean": float(intrinsic_detached.mean().item()),
                "decayed_weight": float(self._icm_last_decayed_weight),
            }
            self.icm_last_metrics.update(
                {
                    "gate_active": float(1.0 if self._icm_gate_active else 0.0),
                    "intrinsic_weight": float(intrinsic_scale),
                }
            )

        # Log curiosity statistics regardless of training frequency for monitoring
        self.logger.record("icm/intrinsic_reward_mean", float(intrinsic_detached.mean().item()))
        self.logger.record("icm/forward_loss", float(losses["forward_loss"].detach().item()))
        self.logger.record("icm/inverse_loss", float(losses["inverse_loss"].detach().item()))
        self.logger.record("icm/total_loss", float(losses["total_loss"].detach().item()))

        return updated_samples

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:  # type: ignore[override]
        self.policy.set_training_mode(True)
        if getattr(self, "_anomaly_detection_enabled", False):
            try:
                if not torch.is_anomaly_enabled():
                    torch.autograd.set_detect_anomaly(True)
            except AttributeError:
                torch.autograd.set_detect_anomaly(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)

        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        td_error_means: List[float] = []

        use_amp = bool(getattr(self, "amp_enabled", False) and self.device.type == "cuda")
        scaler = None
        autocast_callable: Optional[Callable[..., Any]] = None
        autocast_args: Sequence[Any] = ()
        autocast_kwargs: Dict[str, Any] = {}

        if use_amp:
            grad_scaler_cls = None
            grad_scaler_args: Sequence[Any] = ()
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                grad_scaler_cls = torch.amp.GradScaler
                try:
                    grad_scaler_sig = inspect.signature(grad_scaler_cls)
                except (TypeError, ValueError):  # pragma: no cover
                    grad_scaler_sig = None
                if grad_scaler_sig and "device_type" in grad_scaler_sig.parameters:
                    grad_scaler_args = ("cuda",)
            elif hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
                grad_scaler_cls = torch.cuda.amp.GradScaler

            if grad_scaler_cls is not None:
                try:
                    scaler = grad_scaler_cls(*grad_scaler_args, enabled=True)
                except TypeError:
                    scaler = grad_scaler_cls(*grad_scaler_args)

            if scaler is None:
                use_amp = False
            else:
                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                    autocast_callable = torch.amp.autocast
                    try:
                        autocast_sig = inspect.signature(autocast_callable)
                    except (TypeError, ValueError):  # pragma: no cover
                        autocast_sig = None
                    autocast_kwargs = {"dtype": torch.float16}
                    if autocast_sig:
                        parameters = list(autocast_sig.parameters.values())
                        if parameters:
                            first_param = parameters[0]
                            if first_param.kind in (
                                inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            ) and first_param.name in {"device_type", "device"}:
                                autocast_args = ("cuda",)
                        if "dtype" not in autocast_sig.parameters:
                            autocast_kwargs.pop("dtype", None)
                elif hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                    autocast_callable = torch.cuda.amp.autocast
                    try:
                        autocast_sig = inspect.signature(autocast_callable)
                    except (TypeError, ValueError):  # pragma: no cover
                        autocast_sig = None
                    autocast_kwargs = {"dtype": torch.float16}
                    if autocast_sig and "device_type" in autocast_sig.parameters:
                        autocast_kwargs = {"device_type": "cuda", "dtype": torch.float16}
                    elif autocast_sig and "dtype" not in autocast_sig.parameters:
                        autocast_kwargs = {}

                if autocast_callable is None:
                    scaler = None
                    use_amp = False

        if not use_amp:
            scaler = None
            autocast_callable = None
            autocast_args = ()
            autocast_kwargs = {}

        def _finite_check(name: str, value: Any, *, max_entries: int = 5) -> Tuple[bool, Optional[Dict[str, Any]]]:
            """Recursively verify finiteness for nested tensors and collect first offending entry."""
            if isinstance(value, torch.Tensor):
                finite_mask = torch.isfinite(value)
                if bool(finite_mask.all()):
                    return True, None
                nonfinite_indices = torch.nonzero(~finite_mask, as_tuple=False)
                sampled_indices = nonfinite_indices[:max_entries].cpu().tolist()
                sampled_values = value[~finite_mask][:max_entries].detach().cpu().tolist()
                return False, {
                    "path": name or "<tensor>",
                    "indices": sampled_indices,
                    "values": sampled_values,
                }
            if isinstance(value, Mapping):
                for key, sub_value in value.items():
                    ok, info = _finite_check(f"{name}.{key}" if name else str(key), sub_value, max_entries=max_entries)
                    if not ok:
                        return False, info
                return True, None
            if isinstance(value, (list, tuple)):
                for idx, sub_value in enumerate(value):
                    ok, info = _finite_check(f"{name}[{idx}]", sub_value, max_entries=max_entries)
                    if not ok:
                        return False, info
                return True, None
            return True, None

        def _sample_tensor(tensor: torch.Tensor, *, limit: int = 5) -> List[float]:
            flat = tensor.detach().reshape(-1)
            if flat.numel() == 0:
                return []
            limit = max(0, min(int(limit), flat.numel()))
            return flat[:limit].cpu().tolist()

        def _max_abs(name: str, value: Any) -> Union[float, Dict[str, Any], None]:
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return 0.0
                data = value.detach().to(torch.float32)
                return float(torch.max(torch.abs(data)).item())
            if isinstance(value, Mapping):
                return {key: _max_abs(f"{name}.{key}" if name else str(key), sub_value) for key, sub_value in value.items()}
            if isinstance(value, (list, tuple)):
                return {str(idx): _max_abs(f"{name}[{idx}]", sub_value) for idx, sub_value in enumerate(value)}
            return None

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            if self.icm_enabled:
                replay_data = self._apply_icm(replay_data)

            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            if use_amp and autocast_callable is not None:
                context_manager = autocast_callable(*autocast_args, **autocast_kwargs)
            else:
                context_manager = nullcontext()
            with context_manager:
                actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)

                ent_coef_loss = None
                if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                    ent_coef_tensor = torch.exp(self.log_ent_coef.detach())
                    assert isinstance(self.target_entropy, float)
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    ent_coef_losses.append(ent_coef_loss.item())
                    ent_coef: Union[torch.Tensor, float] = ent_coef_tensor
                else:
                    ent_coef = self.ent_coef_tensor

                if self.ent_coef_lower_bound > 0.0:
                    if isinstance(ent_coef, torch.Tensor):
                        ent_coef = torch.clamp(ent_coef, min=self.ent_coef_lower_bound)
                    else:
                        ent_coef = float(max(self.ent_coef_lower_bound, float(ent_coef)))

                ent_coefs.append(float(ent_coef.item() if isinstance(ent_coef, torch.Tensor) else ent_coef))

                with torch.no_grad():
                    next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                    next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                    next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

                current_q_values = self.critic(replay_data.observations, replay_data.actions)
                with torch.no_grad():
                    mse_terms: List[torch.Tensor] = []
                    for current_q in current_q_values:
                        mse_terms.append(torch.mean((target_q_values - current_q) ** 2))
                    if mse_terms:
                        stacked = torch.stack(mse_terms)
                        td_error_means.append(float(torch.mean(stacked).item()))

                critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
                if not torch.isfinite(critic_loss):
                    obs_isfinite, obs_issue = _finite_check("replay_data.observations", replay_data.observations)
                    next_obs_isfinite, next_obs_issue = _finite_check(
                        "replay_data.next_observations", replay_data.next_observations
                    )
                    discounts_isfinite: bool
                    discounts_sample: Union[float, List[float]]
                    if isinstance(discounts, torch.Tensor):
                        discounts_isfinite = bool(torch.isfinite(discounts).all())
                        discounts_sample = _sample_tensor(discounts)
                    else:
                        try:
                            discounts_sample = float(discounts)
                        except (TypeError, ValueError):
                            discounts_sample = float(self.gamma)
                        discounts_isfinite = bool(np.isfinite(discounts_sample))
                    raise ValueError(
                        "Critic loss became non-finite",
                        {
                            "critic_loss": critic_loss.detach().cpu(),
                            "target_q_values_isfinite": bool(torch.isfinite(target_q_values).all()),
                            "current_q_isfinite": [bool(torch.isfinite(q).all()) for q in current_q_values],
                            "target_q_values_sample": _sample_tensor(target_q_values),
                            "next_q_values_isfinite": bool(torch.isfinite(next_q_values).all()),
                            "next_q_values_sample": _sample_tensor(next_q_values),
                            "next_log_prob_isfinite": bool(torch.isfinite(next_log_prob).all()),
                            "next_log_prob_sample": _sample_tensor(next_log_prob),
                            "next_actions_isfinite": bool(torch.isfinite(next_actions).all()),
                            "next_actions_sample": _sample_tensor(next_actions),
                            "actor_log_prob_isfinite": bool(torch.isfinite(log_prob).all()),
                            "actor_log_prob_sample": _sample_tensor(log_prob),
                            "actions_pi_isfinite": bool(torch.isfinite(actions_pi).all()),
                            "actions_pi_sample": _sample_tensor(actions_pi),
                            "replay_rewards_isfinite": bool(torch.isfinite(replay_data.rewards).all()),
                            "replay_rewards_sample": _sample_tensor(replay_data.rewards),
                            "replay_dones_unique": torch.unique(replay_data.dones.detach()).cpu().tolist(),
                            "discounts_isfinite": discounts_isfinite,
                            "discounts_sample": discounts_sample,
                            "replay_actions_isfinite": bool(torch.isfinite(replay_data.actions).all()),
                            "replay_obs_isfinite": obs_isfinite,
                            "replay_obs_issue": obs_issue,
                            "replay_next_obs_isfinite": next_obs_isfinite,
                            "replay_next_obs_issue": next_obs_issue,
                            "replay_obs_max_abs": _max_abs("replay_data.observations", replay_data.observations),
                            "replay_next_obs_max_abs": _max_abs("replay_data.next_observations", replay_data.next_observations),
                        },
                    )
                critic_losses.append(float(critic_loss.item()))

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                if use_amp:
                    scaler.scale(ent_coef_loss).backward()
                    scaler.step(self.ent_coef_optimizer)
                else:
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

            self.critic.optimizer.zero_grad()
            if use_amp:
                scaler.scale(critic_loss).backward()
                # CRITICAL FIX: Add gradient clipping to prevent NaN from exploding gradients
                if scaler._enabled:
                    scaler.unscale_(self.critic.optimizer)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                scaler.step(self.critic.optimizer)
            else:
                critic_loss.backward()
                # CRITICAL FIX: Add gradient clipping to prevent NaN from exploding gradients
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.critic.optimizer.step()

            q_values_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            if not torch.isfinite(actor_loss):
                obs_isfinite, obs_issue = _finite_check("replay_data.observations", replay_data.observations)
                next_obs_isfinite, next_obs_issue = _finite_check(
                    "replay_data.next_observations", replay_data.next_observations
                )
                raise ValueError(
                    "Actor loss became non-finite",
                    {
                        "actor_loss": actor_loss.detach().cpu(),
                        "log_prob_isfinite": bool(torch.isfinite(log_prob).all()),
                        "min_qf_pi_isfinite": bool(torch.isfinite(min_qf_pi).all()),
                        "actions_pi_isfinite": bool(torch.isfinite(actions_pi).all()),
                        "replay_obs_isfinite": obs_isfinite,
                        "replay_obs_issue": obs_issue,
                        "replay_next_obs_isfinite": next_obs_isfinite,
                        "replay_next_obs_issue": next_obs_issue,
                    },
                )
            actor_losses.append(float(actor_loss.item()))

            self.actor.optimizer.zero_grad()
            if use_amp:
                scaler.scale(actor_loss).backward()
                # CRITICAL FIX: Add gradient clipping to prevent NaN from exploding gradients
                if scaler._enabled:
                    scaler.unscale_(self.actor.optimizer)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                scaler.step(self.actor.optimizer)
            else:
                actor_loss.backward()
                # CRITICAL FIX: Add gradient clipping to prevent NaN from exploding gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor.optimizer.step()

            if use_amp:
                scaler.update()

            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            self._n_updates += 1

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs) if ent_coefs else 0.0)
        self.logger.record("train/actor_loss", np.mean(actor_losses) if actor_losses else 0.0)
        self.logger.record("train/critic_loss", np.mean(critic_losses) if critic_losses else 0.0)
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        if td_error_means:
            self.logger.record("train/td_error_mse", np.mean(td_error_means))

    def get_icm_metrics(self) -> Dict[str, float]:
        return dict(self.icm_last_metrics)

class TradingSAC:
    """Wrapper that initialises and trains a SAC agent with custom callbacks."""

    def __init__(self, env, eval_env, config: Dict[str, Any], shared_encoder: Optional[FeatureEncoder] = None) -> None:
        self.env = env
        self.eval_env = eval_env
        self.config = config
        self.shared_encoder = shared_encoder
        sac_cfg = config.get("sac", {})
        icm_cfg = config.get("icm", {})
        if not isinstance(icm_cfg, dict):
            icm_cfg = {}

        configured_device = sac_cfg.get("device", "auto")
        if isinstance(configured_device, str) and configured_device.lower() == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target_device = configured_device

        self.device = torch.device(target_device)

        if self.shared_encoder is not None:
            self.shared_encoder.to(self.device)

        noise_sigma = float(sac_cfg.get("action_noise_sigma", 0.1))
        noise_theta = float(sac_cfg.get("action_noise_theta", 0.15))
        noise_dt = float(sac_cfg.get("action_noise_dt", 1e-2))

        n_actions = env.action_space.shape[0]
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions, dtype=np.float32),
            sigma=noise_sigma * np.ones(n_actions, dtype=np.float32),
            theta=noise_theta,
            dt=noise_dt,
        )

        tensorboard_log = sac_cfg.get("tensorboard_log")
        policy_kwargs = copy.deepcopy(sac_cfg.get("policy_kwargs", {}))
        if self.shared_encoder is not None:
            policy_kwargs.setdefault("features_extractor_class", SharedFrozenFeatureExtractor)
            policy_kwargs.setdefault("features_extractor_kwargs", {})
            policy_kwargs["features_extractor_kwargs"]["shared_encoder"] = self.shared_encoder

        buffer_size = int(sac_cfg.get("buffer_size", 100000))
        learning_starts = int(sac_cfg.get("learning_starts", 1000))
        batch_size = int(sac_cfg.get("batch_size", 256))
        tau = float(sac_cfg.get("tau", 0.005))
        gamma = float(sac_cfg.get("gamma", 0.99))
        train_freq = sac_cfg.get("train_freq", 1)

        base_gradient_steps = sac_cfg.get("gradient_steps", 1)
        try:
            gradient_steps = int(base_gradient_steps)
        except (TypeError, ValueError):
            gradient_steps = 1

        if bool(sac_cfg.get("auto_scale_gradient_steps", True)):
            num_envs = getattr(env, "num_envs", 1) or 1
            gradient_steps = max(1, gradient_steps * num_envs)

        ent_coef = sac_cfg.get("ent_coef", "auto_0.1")
        target_update_interval = int(sac_cfg.get("target_update_interval", 1))
        target_entropy = sac_cfg.get("target_entropy", "auto")
        use_sde = bool(sac_cfg.get("use_sde", True))
        sde_sample_freq = int(sac_cfg.get("sde_sample_freq", 64))

        lr_schedule = build_cosine_warmup_schedule(
            sac_cfg.get("base_learning_rate", 3e-4),
            sac_cfg.get("warmup_fraction", 0.05),
            min_lr_fraction=sac_cfg.get("lr_min_fraction", 0.05),
            num_cycles=sac_cfg.get("lr_cycles", 3),
        )

        replay_buffer_kwargs = sac_cfg.get("replay_buffer_kwargs", {})
        if not isinstance(replay_buffer_kwargs, dict):
            replay_buffer_kwargs = {}

        optimize_memory_usage = bool(sac_cfg.get("optimize_memory_usage", False))

        # Get optimizer_kwargs for paper's symmetric beta fix
        optimizer_kwargs = sac_cfg.get("optimizer_kwargs", None)

        sac_class = SACWithICM if bool(icm_cfg.get("enabled", False)) else SAC

        sac_kwargs = dict(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=lr_schedule,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=1,
            device=target_device,
            optimize_memory_usage=optimize_memory_usage,
            replay_buffer_kwargs=replay_buffer_kwargs,
        )
        if sac_class is SACWithICM:
            sac_kwargs["icm_settings"] = icm_cfg
            sac_kwargs["ent_coef_lower_bound"] = sac_cfg.get("ent_coef_lower_bound", 0.0)

        def _space_max_abs(space: spaces.Space) -> float:
            if isinstance(space, spaces.Box):
                candidates: List[float] = []
                if np.isfinite(space.high).any():
                    candidates.append(float(np.max(np.abs(space.high[np.isfinite(space.high)]))))
                if np.isfinite(space.low).any():
                    candidates.append(float(np.max(np.abs(space.low[np.isfinite(space.low)]))))
                if not candidates:
                    return float("inf")
                return float(max(candidates))
            if isinstance(space, spaces.Dict):
                maxima = [
                    _space_max_abs(sub_space)
                    for sub_space in space.spaces.values()
                ]
                return float(max(maxima)) if maxima else 0.0
            if isinstance(space, spaces.Tuple):
                maxima = [_space_max_abs(sub_space) for sub_space in space.spaces]
                return float(max(maxima)) if maxima else 0.0
            return 0.0

        self.model = sac_class(**sac_kwargs)

        amp_requested = bool(sac_cfg.get("use_amp", True))
        amp_available = bool(torch.cuda.is_available() and str(self.device).startswith("cuda"))
        amp_should_enable = amp_requested and amp_available
        if amp_should_enable:
            obs_space_max = _space_max_abs(env.observation_space)
            if not math.isfinite(obs_space_max) or obs_space_max >= 60_000:
                LOGGER.warning(
                    "Disabling AMP: observation magnitude %.1f exceeds float16 safety threshold.",
                    obs_space_max,
                )
                amp_should_enable = False
        setattr(self.model, "amp_enabled", amp_should_enable)

        anomaly_detection = bool(sac_cfg.get("enable_anomaly_detection", False))
        setattr(self.model, "_anomaly_detection_enabled", anomaly_detection)
        if anomaly_detection:
            try:
                if not torch.is_anomaly_enabled():
                    torch.autograd.set_detect_anomaly(True)
            except AttributeError:
                torch.autograd.set_detect_anomaly(True)
            LOGGER.warning("PyTorch autograd anomaly detection enabled; expect slower training.")
        
        # Apply paper's symmetric beta fix if optimizer_kwargs provided
        if optimizer_kwargs is not None:
            LOGGER.info(f"Applying custom optimizer settings: {optimizer_kwargs}")
            # Re-initialize optimizers with custom kwargs
            if hasattr(self.model, 'actor') and hasattr(self.model.actor, 'optimizer'):
                lr = self.model.actor.optimizer.param_groups[0]['lr']
                self.model.actor.optimizer = torch.optim.Adam(
                    self.model.actor.parameters(),
                    lr=lr,
                    **optimizer_kwargs
                )
            if hasattr(self.model, 'critic') and hasattr(self.model.critic, 'optimizer'):
                lr = self.model.critic.optimizer.param_groups[0]['lr']
                self.model.critic.optimizer = torch.optim.Adam(
                    self.model.critic.parameters(),
                    lr=lr,
                    **optimizer_kwargs
                )
            if hasattr(self.model, 'ent_coef_optimizer') and self.model.ent_coef_optimizer is not None:
                lr = self.model.ent_coef_optimizer.param_groups[0]['lr']
                self.model.ent_coef_optimizer = torch.optim.Adam(
                    [self.model.log_ent_coef],
                    lr=lr,
                    **optimizer_kwargs
                )

        compile_requested = bool(sac_cfg.get("compile_policy", False))
        have_triton = importlib_util.find_spec("triton") is not None
        if compile_requested:
            if os.name == "nt":
                LOGGER.warning("torch.compile requested but disabled on Windows due to stability issues.")
            elif not torch.cuda.is_available() or not hasattr(torch, "compile"):
                LOGGER.warning("torch.compile requested but CUDA compile support is unavailable; skipping.")
            elif not have_triton:
                LOGGER.warning("torch.compile requested but Triton is not installed; skipping compilation.")
            else:
                try:
                    self.model.policy.actor = torch.compile(self.model.policy.actor)
                    self.model.policy.critic = torch.compile(self.model.policy.critic)
                    self.model.policy.critic_target = torch.compile(self.model.policy.critic_target)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("torch.compile acceleration disabled (actor/critic): %s", exc)

    def train(self, total_timesteps: int, callbacks: List[BaseCallback]) -> SAC:
        callback_list = CallbackList(callbacks) if callbacks else None
        self.model.learn(total_timesteps=total_timesteps, callback=callback_list, progress_bar=True)
        return self.model


def make_env_factory(
    config_dict: Dict[str, Any],
    seed: int,
    *,
    mode: str = "continuous",
    is_eval: bool = False,
):
    trading_cfg = build_trading_config(config_dict)

    def _factory():
        # Suppress noisy logs in subprocess environments (critical for SubprocVecEnv with n-envs>1)
        # Use CRITICAL level and disabled flag to completely silence INFO and WARNING logs
        pm_logger = logging.getLogger("core.rl.environments.portfolio_manager")
        te_logger = logging.getLogger("core.rl.environments.trading_env")
        cte_logger = logging.getLogger("core.rl.environments.continuous_trading_env")

        pm_logger.setLevel(logging.CRITICAL)
        te_logger.setLevel(logging.CRITICAL)
        cte_logger.setLevel(logging.CRITICAL)

        pm_logger.propagate = False
        te_logger.propagate = False
        cte_logger.propagate = False

        pm_logger.disabled = True
        te_logger.disabled = True
        cte_logger.disabled = True

        cfg = copy.deepcopy(trading_cfg)
        if is_eval:
            eval_override = config_dict.get("val_data_path") or config_dict.get("test_data_path")
            if eval_override:
                eval_path = Path(eval_override).expanduser()
                if eval_path.exists():
                    cfg.data_path = eval_path
                else:
                    LOGGER.warning("Evaluation data override for %s not found: %s", cfg.symbol, eval_path)
            # Disable exploration-specific features; evaluation should be policy-only.
            if hasattr(cfg, "epsilon_greedy_enabled"):
                cfg.epsilon_greedy_enabled = False
                if hasattr(cfg, "epsilon_current"):
                    cfg.epsilon_current = 0.0
            if hasattr(cfg, "exploration_curriculum_enabled"):
                cfg.exploration_curriculum_enabled = False
            if hasattr(cfg, "curriculum_action_coverage_enabled"):
                cfg.curriculum_action_coverage_enabled = False
            setattr(cfg, "evaluation_mode", True)
            val_start = getattr(cfg, "val_start", None)
            val_end = getattr(cfg, "val_end", None)
            if val_start:
                cfg.train_start = val_start
            if val_end:
                cfg.train_end = val_end

        env = ActionSpaceMigrator.create_hybrid_environment(cfg, seed=seed, mode="hybrid" if mode == "hybrid" else mode)
        return env

    return _factory


def prepare_vec_env(
    config_dict: Dict[str, Any],
    seed: int,
    *,
    mode: str = "continuous",
    num_envs: int = 1,
    is_eval: bool = False,
):
    factories = [make_env_factory(config_dict, seed=seed + idx, mode=mode, is_eval=is_eval) for idx in range(num_envs)]
    if num_envs == 1:
        vec_env = DummyVecEnv(factories)
    else:
        # Windows spawn overhead can dominate for light-weight environments; allow config override
        vec_type = config_dict.get("vec_env_type", "subproc").lower()
        if vec_type not in {"subproc", "dummy"}:
            vec_type = "subproc"
        vec_env = DummyVecEnv(factories) if vec_type == "dummy" else SubprocVecEnv(factories)
    return VecMonitor(vec_env)


def setup_logging(output_dir: Path, experiment_cfg: Dict[str, Any], *, run_subdir: Optional[str] = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow_uri = experiment_cfg.get("mlflow_uri")
    if mlflow_uri:
        mlflow.set_tracking_uri(str(mlflow_uri))
        mlflow.set_experiment(experiment_cfg.get("name", "sac_continuous"))

    tensorboard_dir = output_dir / "tensorboard"
    if run_subdir:
        tensorboard_dir = tensorboard_dir / run_subdir
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    configure(str(tensorboard_dir), ["stdout", "tensorboard"])
    return tensorboard_dir


def start_mlflow_run(config: Dict[str, Any], output_dir: Path, run_name_override: Optional[str] = None) -> Optional[str]:
    experiment_cfg = config.get("experiment", {})
    if not experiment_cfg.get("mlflow_uri"):
        return None

    run_name = run_name_override or experiment_cfg.get("run_name") or f"sac-continuous-{datetime.now(timezone.utc).isoformat()}"
    mlflow_run = mlflow.start_run(run_name=run_name)
    mlflow.log_dict(config, "config.yaml")
    mlflow.log_param("output_dir", str(output_dir))
    return mlflow_run.info.run_id if mlflow_run else None


def print_training_summary(
    symbol: str,
    total_timesteps: int,
    duration_seconds: float,
    monitor: RichTrainingMonitor,
    output_dir: Path,
    *,
    saved_models: Optional[Dict[str, Path]] = None,
    callbacks: Optional[List[BaseCallback]] = None,
    param_stats: Optional[Dict[str, float]] = None,
    gpu_memory_mb: Optional[float] = None,
    icm_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Print a comprehensive training summary in printf style."""
    print("\n" + "=" * 80)
    print(f"  SAC TRAINING COMPLETE - {symbol}")
    print("=" * 80)
    
    # Training info
    print(f"\n  TRAINING OVERVIEW:")
    print("  " + "-" * 76)
    print(f"  Total Timesteps:        {total_timesteps:,}")
    print(f"  Training Duration:      {_fmt_time(duration_seconds)}")
    avg_fps = total_timesteps / max(1e-6, duration_seconds)
    print(f"  Average FPS:            {avg_fps:.0f}")
    if gpu_memory_mb is not None:
        print(f"  Peak GPU Memory:        {gpu_memory_mb:.1f} MB")
    run_label = monitor.metrics.get("run_label")
    if isinstance(run_label, str) and run_label:
        print(f"  Run Label:              {run_label}")
    
    # Performance metrics
    print("\n  EVALUATION PERFORMANCE:")
    print("  " + "-" * 76)
    
    best_sharpe = monitor.metrics.get("best_sharpe")
    last_sharpe = monitor.metrics.get("last_eval_sharpe")
    eval_return = monitor.metrics.get("eval_return_pct")
    eval_return_std = monitor.metrics.get("eval_return_pct_std")
    eval_reward_mean = monitor.metrics.get("eval_reward_mean")
    eval_reward_std = monitor.metrics.get("eval_reward_std")
    eval_sharpe_std = monitor.metrics.get("eval_sharpe_std")
    train_reward_mean = monitor.metrics.get("train_reward_mean")
    train_reward_std = monitor.metrics.get("train_reward_std")
    last_eval_step = monitor.metrics.get("last_eval_step")
    
    if best_sharpe is not None and np.isfinite(float(best_sharpe)):
        print(f"  Best Sharpe Ratio:      {float(best_sharpe):.4f}")
    else:
        print(f"  Best Sharpe Ratio:      N/A (no eval completed)")
    
    if last_sharpe is not None and np.isfinite(float(last_sharpe)):
        print(f"  Last Eval Sharpe:       {float(last_sharpe):.4f}")
        if eval_sharpe_std is not None and np.isfinite(float(eval_sharpe_std)):
            print(f"  Eval Sharpe σ:          {float(eval_sharpe_std):.4f}")
    else:
        print(f"  Last Eval Sharpe:       N/A")
    
    if eval_return is not None and np.isfinite(float(eval_return)):
        line = f"  Eval Return %:          {float(eval_return):.2f}%"
        if eval_return_std is not None and np.isfinite(float(eval_return_std)):
            line += f" (σ={float(eval_return_std):.2f}%)"
        print(line)
    else:
        print(f"  Eval Return %:          N/A")
    
    if eval_reward_mean is not None and np.isfinite(float(eval_reward_mean)):
        eval_std_str = f"{float(eval_reward_std):.2f}" if eval_reward_std and np.isfinite(float(eval_reward_std)) else "N/A"
        print(f"  Eval PnL (μ ± σ):       {float(eval_reward_mean):.2f} ± {eval_std_str}")
    else:
        print(f"  Eval Reward (μ ± σ):    N/A")
    
    if train_reward_mean is not None and np.isfinite(float(train_reward_mean)):
        train_std_str = f"{float(train_reward_std):.2f}" if train_reward_std and np.isfinite(float(train_reward_std)) else "N/A"
        print(f"  Train Reward (μ ± σ):   {float(train_reward_mean):.2f} ± {train_std_str}")
    else:
        print(f"  Train Reward (μ ± σ):   N/A")
    
    if last_eval_step is not None:
        print(f"  Last Eval Step:         {last_eval_step:,}")
    else:
        print(f"  Last Eval Step:         N/A")
    
    # Action & Exploration Metrics
    if callbacks:
        action_monitor = next((cb for cb in callbacks if isinstance(cb, ContinuousActionMonitor)), None)
        entropy_tracker = next((cb for cb in callbacks if isinstance(cb, EntropyTracker)), None)
        trade_metrics = next((cb for cb in callbacks if isinstance(cb, TradeMetricsCallback)), None)
        sac_metrics = next((cb for cb in callbacks if isinstance(cb, SACMetricsCapture)), None)
        
        if action_monitor or entropy_tracker or trade_metrics or sac_metrics:
            print("\n  ACTION & EXPLORATION METRICS:")
            print("  " + "-" * 76)
            
            if action_monitor and action_monitor.final_stats:
                stats = action_monitor.final_stats
                print(f"  Action Mean:            {stats.get('action_mean', float('nan')):.4f}")
                print(f"  Action Std:             {stats.get('action_std', float('nan')):.4f}")
                print(f"  Action Range:           [{stats.get('action_min', float('nan')):.4f}, {stats.get('action_max', float('nan')):.4f}]")
            
            if entropy_tracker and entropy_tracker.final_entropy is not None:
                print(f"  Action Entropy:         {entropy_tracker.final_entropy:.4f}")
            
            if trade_metrics:
                print(f"  Trade Rate:             {trade_metrics.final_trade_rate:.4f}")
                if trade_metrics.counter:
                    print("\n  ACTION DISTRIBUTION:")
                    total_actions = sum(trade_metrics.counter.values())
                    for action_name, count in sorted(trade_metrics.counter.items(), key=lambda x: -x[1]):
                        pct = (count / total_actions * 100) if total_actions > 0 else 0
                        print(f"    {action_name:20s}  {count:6d} ({pct:5.1f}%)")
            
            # SAC training metrics
            if sac_metrics and sac_metrics.final_metrics:
                print("\n  SAC TRAINING METRICS:")
                metrics = sac_metrics.final_metrics
                if 'train/actor_loss' in metrics:
                    print(f"  Actor Loss:             {metrics['train/actor_loss']:.4f}")
                if 'train/critic_loss' in metrics:
                    print(f"  Critic Loss:            {metrics['train/critic_loss']:.4f}")
                if 'train/ent_coef' in metrics:
                    print(f"  Entropy Coefficient:    {metrics['train/ent_coef']:.6f}")
                if 'train/learning_rate' in metrics:
                    print(f"  Learning Rate:          {metrics['train/learning_rate']:.6f}")
                if 'train/n_updates' in metrics:
                    print(f"  Total Updates:          {int(metrics['train/n_updates'])}")

            if icm_metrics:
                print("\n  CURIOSITY METRICS:")
                print("  " + "-" * 76)
                forward = icm_metrics.get("forward_loss")
                inverse = icm_metrics.get("inverse_loss")
                total = icm_metrics.get("total_loss")
                intrinsic_mean = icm_metrics.get("intrinsic_reward_mean")
                if forward is not None:
                    print(f"  ICM Forward Loss:       {forward:.6f}")
                if inverse is not None:
                    print(f"  ICM Inverse Loss:       {inverse:.6f}")
                if total is not None:
                    print(f"  ICM Total Loss:         {total:.6f}")
                if intrinsic_mean is not None:
                    print(f"  ICM Intrinsic Reward μ: {intrinsic_mean:.6f}")

    if param_stats:
        print("\n  MODEL PARAMETERS:")
        print("  " + "-" * 76)
        total = int(param_stats.get("total", 0))
        trainable = int(param_stats.get("trainable", 0))
        frozen = int(param_stats.get("frozen", max(total - trainable, 0)))
        print(f"  Total Parameters:       {total:,}")
        print(f"  Trainable Parameters:   {trainable:,}")
        print(f"  Frozen Parameters:      {frozen:,}")
    
    # Saved artifacts
    print("\n  SAVED ARTIFACTS:")
    print("  " + "-" * 76)
    print(f"  Output Directory:       {output_dir.resolve()}")
    
    if saved_models:
        for name, path in saved_models.items():
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                display_path = path
                try:
                    display_path = path.relative_to(output_dir)
                except ValueError:
                    display_path = path.resolve()
                print(f"  {name:22s}  {display_path} ({size_mb:.1f} MB)")
    
    tensorboard_dir = output_dir / "tensorboard"
    if isinstance(run_label, str) and run_label:
        tensorboard_dir = tensorboard_dir / run_label
    if tensorboard_dir.exists():
        print(f"  TensorBoard Logs:       {tensorboard_dir.resolve()}")
    
    checkpoint_dir = output_dir / "checkpoints"
    run_checkpoint_dir = checkpoint_dir / run_label if isinstance(run_label, str) and run_label else checkpoint_dir
    if run_checkpoint_dir.exists():
        checkpoints = list(run_checkpoint_dir.glob("checkpoint_*"))
        if checkpoints:
            print(f"  Checkpoints:            {len(checkpoints)} saved in {run_checkpoint_dir.resolve()}")
    latest_dir = checkpoint_dir / "latest"
    if latest_dir.exists():
        print(f"  Latest Snapshots:       {latest_dir.resolve()}")
    
    print("\n" + "=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on the continuous trading environment")
    parser.add_argument("--config", required=True, type=Path, help="Path to training configuration YAML")
    parser.add_argument("--total-timesteps", type=int, help="Override total timesteps")
    parser.add_argument("--seed", type=int, help="Random seed override")
    parser.add_argument("--symbol", type=str, help="Override trading symbol")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols to train sequentially (overrides --symbol)")
    parser.add_argument("--n-envs", type=int, help="Override number of parallel environments")
    parser.add_argument("--eval-freq", type=int, help="Override evaluation frequency in steps")
    parser.add_argument("--save-freq", type=int, help="Override periodic checkpoint frequency in steps")
    parser.add_argument(
        "--log-reward-breakdown",
        action="store_true",
        help="Log averaged reward breakdown components during training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    # Base sections
    env_cfg = config.setdefault("environment", {})
    training_cfg = config.setdefault("training", {})
    experiment_cfg = config.get("experiment", {})

    # Build symbol list from CLI or config
    if args.symbols:
        raw = args.symbols.replace(";", ",")
        symbols = [s for s in (seg.strip() for seg in raw.split(",")) if s]
    elif args.symbol:
        symbols = [args.symbol]
    else:
        cfg_sym = env_cfg.get("symbol")
        if cfg_sym:
            symbols = [cfg_sym]
        else:
            exp_symbols = experiment_cfg.get("symbols", [])
            if isinstance(exp_symbols, (list, tuple)):
                symbols = [str(sym).strip() for sym in exp_symbols if str(sym).strip()]
            else:
                symbols = []

    # Apply CLI overrides
    if args.n_envs is not None:
        training_cfg["n_envs"] = max(1, int(args.n_envs))
    if args.eval_freq is not None:
        training_cfg["eval_freq"] = max(1, int(args.eval_freq))
    if args.save_freq is not None:
        try:
            override = int(args.save_freq)
        except (TypeError, ValueError):
            override = None

        if override is None:
            LOGGER.warning("Ignoring save_freq override '%s' (not an integer)", args.save_freq)
        elif override <= 0:
            training_cfg["save_interval"] = 0
            LOGGER.info("Checkpoint autosave disabled via CLI (save_interval=0)")
        else:
            training_cfg["save_interval"] = override
            LOGGER.info("Overriding save_interval to %s via CLI", training_cfg["save_interval"])
    if args.total_timesteps is not None:
        training_cfg["total_timesteps"] = int(args.total_timesteps)
    if args.seed is not None:
        training_cfg["seed"] = int(args.seed)

    if not env_cfg:
        raise KeyError("Configuration must include an 'environment' section")

    total_timesteps = int(training_cfg.get("total_timesteps", 100_000))
    seed = int(training_cfg.get("seed", 42))
    set_random_seed(seed)

    shared_encoder_cfg = config.get("shared_encoder", {})
    shared_encoder: Optional[FeatureEncoder] = None
    if bool(shared_encoder_cfg.get("enabled", True)):
        encoder_kwargs = shared_encoder_cfg.get("config", {})
        if not isinstance(encoder_kwargs, dict):
            encoder_kwargs = {}
        try:
            encoder_config = EncoderConfig(**encoder_kwargs)
        except TypeError as exc:
            LOGGER.warning("Invalid shared encoder config overrides ignored: %s", exc)
            encoder_config = EncoderConfig()
        shared_encoder = FeatureEncoder(encoder_config)

        checkpoint_path = shared_encoder_cfg.get("checkpoint")
        if checkpoint_path:
            ckpt = Path(checkpoint_path)
            if ckpt.exists():
                try:
                    state_dict = torch.load(ckpt, map_location="cpu")
                    missing = shared_encoder.load_state_dict(state_dict, strict=False)
                    if missing.missing_keys or missing.unexpected_keys:
                        LOGGER.warning(
                            "Shared encoder checkpoint %s loaded with mismatched keys (missing=%s, unexpected=%s)",
                            ckpt,
                            missing.missing_keys,
                            missing.unexpected_keys,
                        )
                    else:
                        LOGGER.info("Shared encoder weights restored from %s", ckpt)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Failed to load shared encoder checkpoint %s: %s", ckpt, exc)
            else:
                LOGGER.warning("Shared encoder checkpoint %s not found; using default initialization", ckpt)

    if not symbols:
        raise ValueError("No symbol specified. Provide --symbol, --symbols, or set environment.symbol in YAML.")

    base_output_dir = Path(experiment_cfg.get("output_dir", "models/sac_continuous"))
    reward_breakdown_logging = bool(training_cfg.get("log_reward_breakdown", False)) or bool(
        getattr(args, "log_reward_breakdown", False)
    )
    reward_breakdown_log_freq = int(training_cfg.get("reward_breakdown_log_freq", 256))

    for idx, sym in enumerate(symbols):
        # Per-symbol environment config
        run_env_cfg = dict(env_cfg)
        run_env_cfg["symbol"] = sym
        data_sources = resolve_symbol_data_paths(sym, run_env_cfg, experiment_cfg)
        train_path = data_sources.get("train")
        if train_path is None:
            raise FileNotFoundError(f"Unable to locate training data for {sym}")
        run_env_cfg["data_path"] = str(Path(train_path))

        val_path = data_sources.get("val")
        test_path = data_sources.get("test")
        if val_path is not None:
            run_env_cfg["val_data_path"] = str(Path(val_path))
        if test_path is not None:
            run_env_cfg["test_data_path"] = str(Path(test_path))

        metadata = data_sources.get("metadata")
        if isinstance(metadata, dict):
            period_mappings = {
                "train_period": ("train_start", "train_end"),
                "val_period": ("val_start", "val_end"),
            }
            for period_key, (start_key, end_key) in period_mappings.items():
                period = metadata.get(period_key)
                if not isinstance(period, dict):
                    continue
                start = period.get("start")
                end = period.get("end")
                if start and start_key not in run_env_cfg:
                    run_env_cfg[start_key] = start
                if end and end_key not in run_env_cfg:
                    run_env_cfg[end_key] = end
        symbol_config = copy.deepcopy(config)
        symbol_config["environment"] = run_env_cfg

        # Per-symbol output directory and run labelling
        symbol_output_dir = base_output_dir / sym
        checkpoint_root = symbol_output_dir / "checkpoints"
        run_seed_value = seed + idx * 10
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        label_parts = [timestamp, f"seed{run_seed_value}"]
        extra_tag = experiment_cfg.get("run_tag") or training_cfg.get("run_tag")
        if extra_tag:
            sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(extra_tag))
            sanitized = sanitized.strip("-_")
            if sanitized:
                label_parts.append(sanitized)
        run_label_base = "_".join(label_parts)
        run_label = run_label_base
        run_checkpoint_dir = checkpoint_root / run_label
        suffix_idx = 1
        while run_checkpoint_dir.exists():
            run_label = f"{run_label_base}_{suffix_idx:02d}"
            run_checkpoint_dir = checkpoint_root / run_label
            suffix_idx += 1

        tensorboard_dir = setup_logging(symbol_output_dir, experiment_cfg, run_subdir=run_label)

        sac_section = symbol_config.setdefault("sac", {})
        sac_section["tensorboard_log"] = str(tensorboard_dir)

        num_envs = max(1, int(training_cfg.get("n_envs", 1)))
        env = prepare_vec_env(run_env_cfg, seed=seed + idx * 10, mode="continuous", num_envs=num_envs)
        
        # PERFORMANCE OPTIMIZATION: Match eval environments to training environments
        # Running evaluation with the same number of parallel environments as training
        # ensures consistent GPU utilization and maximizes throughput during evaluation
        n_eval_episodes = int(training_cfg.get("n_eval_episodes", 5))
        
        # IMPORTANT: Set eval_num_envs = num_envs (match training environments)
        # This allows all eval episodes to run in parallel batches efficiently
        # Example: 16 training envs + 16 eval episodes = 1 parallel batch (fastest)
        eval_num_envs = num_envs  # Always match training environment count
        
        eval_env = prepare_vec_env(
            run_env_cfg,
            seed=seed + idx * 10 + 1,
            mode="continuous",
            num_envs=eval_num_envs,  # Same as training for consistent performance
            is_eval=True,
        )

        # Configure console with proper encoding for Windows compatibility
        console = Console(highlight=False, force_terminal=True, legacy_windows=False, safe_box=True)
        monitor = RichTrainingMonitor(sym, total_timesteps, console=console)
        monitor.metrics["run_label"] = run_label
        callbacks: List[BaseCallback] = [
            ContinuousActionMonitor(log_freq=training_cfg.get("log_freq", 100)),
            EntropyTracker(window=training_cfg.get("entropy_window", 512)),
            TradeMetricsCallback(log_freq=training_cfg.get("metrics_log_freq", 1000)),
            SACMetricsCapture(),
            RichStatusCallback(monitor, total_timesteps=total_timesteps, refresh_steps=max(64, total_timesteps // 200)),
        ]
        if reward_breakdown_logging:
            callbacks.append(RewardBreakdownLogger(log_freq=reward_breakdown_log_freq))

        eval_freq = int(training_cfg.get("eval_freq", 5000))
        n_eval_episodes = int(training_cfg.get("n_eval_episodes", 5))
        save_interval = int(training_cfg.get("save_interval", 0))
        save_best_model = bool(training_cfg.get("save_best_model", True))
        save_final_model = bool(training_cfg.get("save_final_model", True))
        latest_checkpoint_dir = checkpoint_root / "latest"
        if save_best_model or save_interval > 0:
            run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            latest_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if save_best_model:
            best_model_file = run_checkpoint_dir / f"best_model_{run_label}.zip"
            eval_cb = ContinuousEvalCallback(
                eval_env,
                monitor,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                best_model_save_path=str(best_model_file),
                latest_model_mirror_dir=str(latest_checkpoint_dir),
                deterministic=True,
            )
        else:
            best_model_file = None
            eval_cb = ContinuousEvalCallback(
                eval_env,
                monitor,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                best_model_save_path=None,
                latest_model_mirror_dir=None,
                deterministic=True,
            )
        callbacks.append(eval_cb)

        if save_interval > 0:
            callbacks.append(PeriodicCheckpointCallback(save_interval, run_checkpoint_dir))

        if experiment_cfg.get("mlflow_uri"):
            run_id = start_mlflow_run(
                symbol_config,
                symbol_output_dir,
                run_name_override=f"sac-continuous-{sym}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
            )
            LOGGER.info("MLflow run started for %s: %s", sym, run_id)
        else:
            run_id = None

        try:
            trainer = TradingSAC(env, eval_env, symbol_config, shared_encoder=shared_encoder)
            # Train inside Rich monitor context; disable SB3 progress_bar to avoid double UI
            try:
                from datetime import UTC
                training_start = datetime.now(UTC)
            except ImportError:
                training_start = datetime.now(timezone.utc)
            
            # Temporarily suppress excessive warnings during training to keep UI clean
            portfolio_logger = logging.getLogger("core.rl.environments.portfolio_manager")
            trading_env_logger = logging.getLogger("core.rl.environments.trading_env")
            continuous_env_logger = logging.getLogger("core.rl.environments.continuous_trading_env")

            original_pm_level = portfolio_logger.level
            original_te_level = trading_env_logger.level
            original_cte_level = continuous_env_logger.level
            original_pm_propagate = portfolio_logger.propagate
            original_te_propagate = trading_env_logger.propagate
            original_cte_propagate = continuous_env_logger.propagate
            original_pm_disabled = portfolio_logger.disabled
            original_te_disabled = trading_env_logger.disabled
            original_cte_disabled = continuous_env_logger.disabled

            # Use CRITICAL level and disabled flag to completely silence INFO and WARNING
            portfolio_logger.setLevel(logging.CRITICAL)
            trading_env_logger.setLevel(logging.CRITICAL)
            continuous_env_logger.setLevel(logging.CRITICAL)
            portfolio_logger.propagate = False
            trading_env_logger.propagate = False
            continuous_env_logger.propagate = False
            portfolio_logger.disabled = True
            trading_env_logger.disabled = True
            continuous_env_logger.disabled = True
            
            try:
                if torch.cuda.is_available() and trainer.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(trainer.device)
                with monitor:
                    model = trainer.model
                    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks), progress_bar=False)
            finally:
                # Restore original log levels and propagation
                portfolio_logger.setLevel(original_pm_level)
                trading_env_logger.setLevel(original_te_level)
                continuous_env_logger.setLevel(original_cte_level)
                portfolio_logger.propagate = original_pm_propagate
                trading_env_logger.propagate = original_te_propagate
                continuous_env_logger.propagate = original_cte_propagate
                portfolio_logger.disabled = original_pm_disabled
                trading_env_logger.disabled = original_te_disabled
                continuous_env_logger.disabled = original_cte_disabled
            
            try:
                from datetime import UTC
                training_end = datetime.now(UTC)
            except ImportError:
                training_end = datetime.now(timezone.utc)
            training_duration = (training_end - training_start).total_seconds()

            gpu_peak_mb: Optional[float] = None
            if torch.cuda.is_available() and trainer.device.type == "cuda":
                gpu_peak_mb = torch.cuda.max_memory_allocated(trainer.device) / (1024 * 1024)
            
            saved_models: Dict[str, Path] = {}
            if save_final_model:
                final_model_path = symbol_output_dir / f"sac_continuous_final_{run_label}.zip"
                try:
                    model.save(str(final_model_path), exclude=["env", "eval_env", "replay_buffer"])
                    saved_models[f"Final Model [{run_label}]"] = final_model_path
                    _mirror_artifact(final_model_path, symbol_output_dir, "sac_continuous_final")
                except Exception as exc:
                    if final_model_path.exists():
                        try:
                            final_model_path.unlink()
                        except OSError:
                            pass
                    LOGGER.warning("Falling back to policy-only save because model serialization failed: %s", exc)
                    policy_path = symbol_output_dir / f"sac_continuous_policy_{run_label}.pt"
                    torch.save(model.policy.state_dict(), policy_path)
                    saved_models[f"Policy Weights [{run_label}]"] = policy_path
                    _mirror_artifact(policy_path, symbol_output_dir, "sac_continuous_policy")
                    LOGGER.info("Policy weights saved to %s", policy_path.resolve())
                    if run_id:
                        mlflow.log_artifact(str(policy_path), artifact_path="models")
                else:
                    LOGGER.info("Training complete for %s. Model saved to %s", sym, final_model_path.resolve())
                    if run_id:
                        mlflow.log_artifact(str(final_model_path), artifact_path="models")
            else:
                LOGGER.info("Training complete for %s. Skipping final model save per configuration.", sym)

            best_artifact_path = eval_cb.get_best_model_artifact() if save_best_model else None
            if best_artifact_path and best_artifact_path.exists():
                saved_models[f"Best Model [{run_label}]"] = best_artifact_path
                _mirror_artifact(best_artifact_path, checkpoint_root, "best_model")
                if run_id:
                    try:
                        mlflow.log_artifact(str(best_artifact_path), artifact_path="models")
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning("Failed to log best model artifact to MLflow: %s", exc)

            total_params = sum(p.numel() for p in model.policy.parameters())
            trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
            param_stats = {
                "total": float(total_params),
                "trainable": float(trainable_params),
                "frozen": float(total_params - trainable_params),
            }
            monitor.metrics["total_params"] = total_params
            monitor.metrics["trainable_params"] = trainable_params

            icm_metrics: Optional[Dict[str, float]]
            if hasattr(model, "get_icm_metrics") and callable(getattr(model, "get_icm_metrics")):
                icm_metrics = model.get_icm_metrics()
            else:
                icm_metrics = None

            if run_id:
                scalar_metrics: Dict[str, float] = {}
                for key, value in monitor.metrics.items():
                    if isinstance(value, (int, float, np.floating, np.integer)):
                        scalar_metrics[key] = float(value)
                scalar_metrics["training_duration_seconds"] = float(training_duration)
                scalar_metrics["total_timesteps"] = float(total_timesteps)
                if gpu_peak_mb is not None:
                    scalar_metrics["gpu_peak_memory_mb"] = float(gpu_peak_mb)
                sac_metrics_cb = next((cb for cb in callbacks if isinstance(cb, SACMetricsCapture)), None)
                if sac_metrics_cb and sac_metrics_cb.final_metrics:
                    for key, value in sac_metrics_cb.final_metrics.items():
                        if isinstance(value, (int, float, np.floating, np.integer)):
                            scalar_metrics[f"final_{key}"] = float(value)
                if icm_metrics:
                    for key, value in icm_metrics.items():
                        if isinstance(value, (int, float, np.floating, np.integer)):
                            scalar_metrics[f"icm_{key}"] = float(value)
                if scalar_metrics:
                    mlflow.log_metrics(scalar_metrics)
            
            # Print comprehensive summary
            print_training_summary(
                symbol=sym,
                total_timesteps=total_timesteps,
                duration_seconds=training_duration,
                monitor=monitor,
                output_dir=symbol_output_dir,
                saved_models=saved_models,
                callbacks=callbacks,
                param_stats=param_stats,
                gpu_memory_mb=gpu_peak_mb,
                icm_metrics=icm_metrics,
            )
        finally:
            if run_id:
                mlflow.end_run()
            env.close()
            eval_env.close()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
if __name__ == "__main__":
    main()
