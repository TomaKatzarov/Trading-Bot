"""Phase 3 Training Smoke Test.

Quick verification that the Phase 3 PPO training infrastructure is wired
correctly. The test instantiates the trading environment for AAPL, runs a
small PPO model for 1,000 timesteps, and performs a deterministic inference
step to confirm end-to-end compatibility.
"""

from __future__ import annotations

import sys
import multiprocessing
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rl.environments import PortfolioConfig, RewardConfig, TradingConfig, TradingEnvironment


def _configure_torch() -> None:
    cpu_count = multiprocessing.cpu_count()
    if torch.cuda.is_available():
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("medium")
    else:
        torch.set_num_threads(max(1, cpu_count - 1))
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, cpu_count // 2))


def make_environment(symbol: str) -> TradingEnvironment:
    data_path = Path("data/phase3_splits") / symbol / "train.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Expected training data at {data_path}")

    reward_config = RewardConfig(
        pnl_weight=0.40,
        transaction_cost_weight=0.15,
        time_efficiency_weight=0.15,
        sharpe_weight=0.05,
        drawdown_weight=0.10,
        sizing_weight=0.05,
        hold_penalty_weight=0.0,
    )

    portfolio_config = PortfolioConfig(
        initial_capital=10_000.0,
        commission_rate=0.001,
        slippage_bps=5.0,
        max_position_size_pct=0.20,
        max_total_exposure_pct=0.90,
        max_position_loss_pct=0.08,
        max_portfolio_loss_pct=0.30,
        max_positions=1,
    )

    config = TradingConfig(
        symbol=symbol,
        data_path=data_path,
        sl_checkpoints={},
        episode_length=168,
        lookback_window=24,
        initial_capital=10_000.0,
        commission_rate=0.001,
        slippage_bps=5.0,
        stop_loss=0.02,
        take_profit=0.025,
        max_hold_hours=8,
        reward_config=reward_config,
        portfolio_config=portfolio_config,
        log_trades=False,
    )

    return TradingEnvironment(config)


def smoke_test() -> None:
    _configure_torch()
    set_random_seed(42)
    print("=" * 70)
    print("PHASE 3 TRAINING SMOKE TEST")
    print("=" * 70)

    symbol = "AAPL"
    cpu_envs = max(1, multiprocessing.cpu_count() // 2)
    num_envs = min(4, cpu_envs)
    env_fns = [lambda sym=symbol: make_environment(sym) for _ in range(num_envs)]
    env = DummyVecEnv(env_fns)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nSymbol: {symbol}")
    print(f"Device: {device}")
    print(f"Parallel environments: {num_envs}")

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
    )

    print("\n✅ Environment and PPO model created")
    total_timesteps = 1000 * num_envs
    print(f"Training for {total_timesteps:,} timesteps across {num_envs} env(s)...")
    model.learn(total_timesteps=total_timesteps)

    obs = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    print(f"\n✅ Inference successful (action: {action})")

    env.close()

    print("\n" + "=" * 70)
    print("🎉 SMOKE TEST PASSED - Infrastructure ready!")
    print("=" * 70)


if __name__ == "__main__":
    smoke_test()
