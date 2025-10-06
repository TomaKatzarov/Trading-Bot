"""Demo: Training with the vectorized trading environment using Stable-Baselines3."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Literal

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    Console = Group = Live = Panel = Progress = None
    SpinnerColumn = TextColumn = BarColumn = TimeElapsedColumn = TimeRemainingColumn = None
    Table = None
    RICH_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.rl.environments.vec_trading_env import configure_env_loggers, make_vec_trading_env

logger = logging.getLogger(__name__)


def _supports_rich_progress() -> bool:
    return RICH_AVAILABLE


def _supports_tqdm_progress() -> bool:
    try:
        import tqdm  # noqa: F401
    except ImportError:  # pragma: no cover - optional dependency
        return False
    return True


def _build_vector_env(
    *,
    symbol: str,
    data_dir: Path,
    num_envs: int,
    subprocess: bool,
    seed: int,
    episode_length: int,
    start_method: str | None,
    env_log_level: int | None,
) -> VecEnv:
    return make_vec_trading_env(
        symbol=symbol,
        data_dir=data_dir,
        num_envs=num_envs,
        seed=seed,
        use_subprocess=subprocess,
        start_method=start_method,
        env_kwargs={"episode_length": episode_length},
        env_log_level=env_log_level,
    )


def _set_env_log_level(level: str) -> int:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        logger.warning("Unknown env log level '%s'; defaulting to WARNING", level)
        numeric_level = logging.WARNING

    configure_env_loggers(numeric_level)
    return numeric_level


if RICH_AVAILABLE:

    class RichProgressCallback(BaseCallback):
        """Rich-powered training progress display with rolling metrics."""

        def __init__(
            self,
            *,
            total_timesteps: int,
            refresh_per_second: float = 8.0,
            console=None,
        ) -> None:
            super().__init__()
            self.total_timesteps = total_timesteps
            self.refresh_per_second = max(refresh_per_second, 1.0)
            self.console = console or Console()
            self.progress = Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                TextColumn("{task.completed:,}/{task.total:,}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            self.task_id = self.progress.add_task("Training", total=total_timesteps)
            self.live = None
            self.episode_rewards: deque[float] = deque(maxlen=20)
            self.episode_lengths: deque[int] = deque(maxlen=20)
            self.num_episodes = 0
            self.best_reward: float | None = None
            self.start_time = 0.0
            self._last_render = 0.0

        def _on_training_start(self) -> None:
            self.start_time = time.time()
            self.progress.start()
            self.live = Live(console=self.console, refresh_per_second=self.refresh_per_second, transient=False)
            self.live.start()
            self._refresh(force=True)

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", ())
            for info in infos or ():
                episode = info.get("episode") if isinstance(info, dict) else None
                if episode:
                    reward = float(episode.get("r", 0.0))
                    length = int(episode.get("l", 0))
                    self.num_episodes += 1
                    self.episode_rewards.append(reward)
                    self.episode_lengths.append(length)
                    if self.best_reward is None or reward > self.best_reward:
                        self.best_reward = reward

            stats = self._gather_stats()
            description = self._build_description(stats)
            completed = min(stats["current_steps"], self.total_timesteps)
            self.progress.update(self.task_id, completed=completed, description=description)

            now = time.time()
            if now - self._last_render >= 1.0 / self.refresh_per_second:
                self._refresh(stats=stats)
                self._last_render = now

            return True

        def _on_training_end(self) -> None:
            stats = self._gather_stats()
            self.progress.update(self.task_id, completed=self.total_timesteps, description=self._build_description(stats))
            self._refresh(stats=stats, force=True)

            if self.live is not None:
                self.progress.stop()
                self.live.stop()
                summary_panel = Panel(
                    self._build_metrics_table(stats),
                    title="Training Summary",
                    border_style="green",
                    padding=(0, 1),
                )
                self.console.print(summary_panel)
                self.live = None

        # ------------------------------------------------------------------
        def _gather_stats(self) -> dict[str, float | int | None]:
            current_steps = int(self.model.num_timesteps) if self.model is not None else 0
            elapsed = max(time.time() - self.start_time, 0.0)
            fps = current_steps / elapsed if elapsed > 0 else 0.0
            remaining_steps = max(self.total_timesteps - current_steps, 0)
            eta = remaining_steps / fps if fps > 0 else None
            avg_reward = self._mean(self.episode_rewards)
            avg_length = self._mean(self.episode_lengths)

            return {
                "current_steps": current_steps,
                "elapsed": elapsed,
                "fps": fps,
                "eta": eta,
                "avg_reward": avg_reward,
                "avg_length": avg_length,
            }

        def _build_description(self, stats: dict[str, float | int | None]) -> str:
            parts = ["[cyan]Training[/]"]
            avg_reward = stats.get("avg_reward")
            if avg_reward is not None:
                parts.append(f"avgR {avg_reward:.2f}")
            if self.best_reward is not None:
                parts.append(f"bestR {self.best_reward:.2f}")
            fps = stats.get("fps") or 0.0
            if fps > 0:
                parts.append(f"{fps:,.0f} steps/s")
            return " | ".join(parts)

        def _refresh(self, *, stats: dict[str, float | int | None] | None = None, force: bool = False) -> None:
            if self.live is None:
                return
            if stats is None or force:
                stats = self._gather_stats()
            renderable = Group(
                self.progress,
                Panel(
                    self._build_metrics_table(stats),
                    title="Recent Performance",
                    border_style="cyan",
                    padding=(0, 1),
                ),
            )
            self.live.update(renderable, refresh=True)

        def _build_metrics_table(self, stats: dict[str, float | int | None]):
            table = Table.grid(expand=True)
            table.add_column("Metric", justify="left")
            table.add_column("Value", justify="right")

            table.add_row("Timesteps", f"{stats['current_steps']:,}/{self.total_timesteps:,}")
            table.add_row("Episodes", f"{self.num_episodes:,}")

            avg_reward = stats.get("avg_reward")
            best_reward = self.best_reward
            table.add_row("Avg Reward (last 20)", f"{avg_reward:.2f}" if avg_reward is not None else "—")
            table.add_row("Best Reward", f"{best_reward:.2f}" if best_reward is not None else "—")

            avg_length = stats.get("avg_length")
            table.add_row("Avg Episode Len", f"{avg_length:.1f}" if avg_length is not None else "—")

            fps = stats.get("fps") or 0.0
            table.add_row("Steps / sec", f"{fps:,.1f}")

            elapsed = stats.get("elapsed") or 0.0
            eta = stats.get("eta")
            table.add_row("Elapsed", self._format_seconds(elapsed))
            table.add_row("ETA", self._format_seconds(eta) if eta is not None else "—")

            return table

        @staticmethod
        def _mean(values: deque[float]) -> float | None:
            return (sum(values) / len(values)) if values else None

        @staticmethod
        def _format_seconds(seconds: float) -> str:
            if seconds is None:
                return "—"
            seconds = max(float(seconds), 0.0)
            total_seconds = int(seconds)
            hrs, remainder = divmod(total_seconds, 3600)
            mins, secs = divmod(remainder, 60)
            return f"{hrs:02d}:{mins:02d}:{secs:02d}"

else:  # pragma: no cover - fallback when rich is unavailable

    class RichProgressCallback(BaseCallback):
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[override]
            raise RuntimeError("Rich progress callback requires the 'rich' package. Install it or run with --no-rich-progress.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo vectorized environment training")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Symbol to train on")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/historical",
        help="Directory containing <symbol>.parquet files",
    )
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "A2C"],
        help="RL algorithm to use",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10_000,
        help="Total number of training timesteps",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=256,
        help="Episode length passed to each environment instance",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for all environments",
    )
    parser.add_argument(
        "--start-method",
        type=str,
        default=None,
        choices=["spawn", "fork", "forkserver", None],
        help="Multiprocessing start method for SubprocVecEnv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/rl_demo",
        help="Directory where checkpoints and logs will be stored",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=5_000,
        help="How often (timesteps) to run evaluation. 0 disables evaluation.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="Number of evaluation episodes when eval is enabled",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--env-log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for environment internals (default: WARNING)",
    )
    parser.add_argument(
        "--no-rich-progress",
        action="store_true",
        help="Disable the rich progress display and fall back to the default tqdm bar.",
    )
    parser.add_argument(
        "--progress-refresh-rate",
        type=float,
        default=8.0,
        help="Refresh rate (updates per second) for the rich progress display.",
    )
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    )


def _create_model(
    algorithm: Literal["PPO", "A2C"],
    vec_env: VecEnv,
    output_dir: Path,
) -> PPO | A2C:
    tensorboard_dir = output_dir / "tensorboard"

    if algorithm == "PPO":
        logger.info("Initialising PPO agent (MultiInputPolicy)")
        return PPO(
            policy="MultiInputPolicy",
            env=vec_env,
            verbose=1,
            tensorboard_log=str(tensorboard_dir),
            learning_rate=3e-4,
            n_steps=max(128, 2048 // vec_env.num_envs),
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
        )

    logger.info("Initialising A2C agent (MultiInputPolicy)")
    return A2C(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=str(tensorboard_dir),
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
    )


def main() -> None:
    args = parse_args()
    _configure_logging(args.log_level)
    env_log_level = _set_env_log_level(args.env_log_level)

    symbol = args.symbol.upper()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    tensorboard_dir = output_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("Vectorized Environment Training Demo")
    logger.info("=" * 72)
    logger.info("Symbol: %s", symbol)
    logger.info("Data directory: %s", data_dir)
    logger.info("Algorithm: %s", args.algorithm)
    logger.info("Parallel environments: %s", args.num_envs)

    vec_env = _build_vector_env(
        symbol=symbol,
        data_dir=data_dir,
        num_envs=args.num_envs,
        subprocess=True,
        seed=args.seed,
        episode_length=args.episode_length,
        start_method=args.start_method,
        env_log_level=env_log_level,
    )

    try:
        logger.info("Observation space: %s", vec_env.observation_space)
        logger.info("Action space: %s", vec_env.action_space)

        model = _create_model(args.algorithm, vec_env, output_dir)

        checkpoint_callback = CheckpointCallback(
            save_freq=max(1_000 // max(args.num_envs, 1), 1),
            save_path=str(checkpoint_dir),
            name_prefix=f"{args.algorithm.lower()}_{symbol.lower()}",
        )

        callbacks: list[BaseCallback] = []
        use_rich_progress = _supports_rich_progress() and not args.no_rich_progress
        progress_bar = False

        if use_rich_progress:
            logger.info(
                "Rich progress display enabled (refresh %.1f Hz)",
                args.progress_refresh_rate,
            )
            callbacks.append(
                RichProgressCallback(
                    total_timesteps=args.total_timesteps,
                    refresh_per_second=args.progress_refresh_rate,
                )
            )
        else:
            if not _supports_rich_progress() and not args.no_rich_progress:
                logger.warning("Rich console not available; falling back to default progress bar.")
            progress_bar = _supports_tqdm_progress()
            if not progress_bar:
                logger.warning("Progress bar disabled (install 'tqdm' for default progress display)")

        callbacks.append(checkpoint_callback)
        if args.eval_frequency > 0:
            logger.info(
                "Evaluation enabled: every %s timesteps, %s episodes",
                args.eval_frequency,
                args.eval_episodes,
            )
            eval_env = _build_vector_env(
                symbol=symbol,
                data_dir=data_dir,
                num_envs=1,
                subprocess=False,
                seed=args.seed + 1,
                episode_length=args.episode_length,
                start_method=None,
                env_log_level=env_log_level,
            )
            callbacks.append(
                EvalCallback(
                    eval_env=eval_env,
                    best_model_save_path=str(output_dir / "best_model"),
                    log_path=str(output_dir / "eval"),
                    eval_freq=args.eval_frequency,
                    n_eval_episodes=args.eval_episodes,
                    deterministic=True,
                )
            )
        else:
            eval_env = None

        logger.info(
            "Starting training for %s timesteps (~%s iterations per environment)",
            args.total_timesteps,
            args.total_timesteps // max(args.num_envs, 1),
        )

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=progress_bar,
        )

        output_path = output_dir / f"{args.algorithm.lower()}_{symbol.lower()}_final"
        model.save(output_path)
        logger.info("Training complete. Model saved to %s.zip", output_path)
        logger.info("TensorBoard logs available at %s", tensorboard_dir)

    finally:
        vec_env.close()
        if "eval_env" in locals() and eval_env is not None:
            eval_env.close()

    logger.info("Run `tensorboard --logdir %s` to inspect training curves", tensorboard_dir)


if __name__ == "__main__":
    main()
