# Phase 3 Training Guide

## Overview

This guide explains how to train and monitor the 10-symbol Phase 3 prototype reinforcement learning agents. The workflow builds on the validated data pipeline from Task 3.1 and the training infrastructure created in Task 3.2.

## Configuration

**Baseline file:** `training/config_templates/phase3_ppo_baseline.yaml`

Key parameters:
- **Learning rate:** 3e-4 with cosine decay to 1e-5
- **Entropy:** initial 0.01, decays toward 0.001 with patience-based early stopping
- **Rollout:** 2048 steps per environment (8 envs ⇒ 16,384 samples/update)
- **Batch size:** 256, **epochs per update:** 10
- **Discounts:** gamma 0.99, GAE lambda 0.95
- **Risk / reward weights:** PnL 40%, cost 15%, time 15%, Sharpe 5%, drawdown 10%, sizing 5%
- **Risk limits:** max 20% position, 90% exposure, 30% drawdown stop, 8-hour hold limit

Tweak `training.total_timesteps`, `training.n_envs`, or individual PPO hyperparameters for experiments. All configuration changes are automatically logged to MLflow.

## How to Train

```bash
# Activate the RL environment (Git Bash / WSL)
source trading_rl_env/Scripts/activate

# Train all 10 symbols
python training/train_phase3_agents.py \
  --config training/config_templates/phase3_ppo_baseline.yaml

# Train a symbol subset
python training/train_phase3_agents.py \
  --config training/config_templates/phase3_ppo_baseline.yaml \
  --symbols AAPL MSFT NVDA

# Resume from best checkpoint for a single symbol
python training/train_phase3_agents.py \
  --config training/config_templates/phase3_ppo_baseline.yaml \
  --resume AAPL
```

The script writes artifacts to:
- `models/phase3_checkpoints/{SYMBOL}/` – periodic checkpoints, `best_model.zip`, `final_model.zip`, evaluation history
- `logs/phase3_training/{SYMBOL}/` – TensorBoard event files for training + evaluation curves
- `mlruns/` – MLflow experiment directory (metrics, params, artifacts) when using the default file URI

## Monitoring

### MLflow

```bash
mlflow ui
```

Open http://localhost:5000 and select the `phase3_10symbol_baseline` experiment. Each symbol run logs:
- Training params & hardware
- Sharpe, return, drawdown, win rate, profit factor per evaluation
- Reward component aggregates (`reward/pnl_mean`, `reward/transaction_cost_mean`, ...)
- Training duration and final evaluation summaries

### TensorBoard

```bash
tensorboard --logdir=logs/phase3_training
```

Useful tabs: **Scalars** for policy/value losses, entropy, KL divergence, evaluation Sharpe; **Custom Scalars** for episode reward trajectories; **Graphs** confirms PPO network wiring.

## Outputs & Directory Layout

```
models/phase3_checkpoints/
├── AAPL/
│   ├── best_model.zip
│   ├── final_model.zip
│   ├── evaluation_history.json
│   └── AAPL_checkpoint_1250_steps.zip
├── MSFT/
│   └── ...
└── ...

logs/phase3_training/
├── AAPL/
│   └── PPO_AAPL/events.out.tfevents...
└── ...

mlruns/
└── 0/
    └── <run-id>/
        ├── metrics/
        ├── params/
        └── artifacts/
```

## Expected Timeline

- **Per symbol:** ~2–3 hours for 100k timesteps with 8 parallel environments (RTX 5070 Ti)
- **Full 10 symbols:** 25–30 wall-clock hours sequentially; shorter with multi-GPU scheduling

## Troubleshooting

| Issue | Symptoms | Remedy |
| --- | --- | --- |
| CUDA out-of-memory | Training aborts at rollout collection | Reduce `training.n_envs` to 4 or lower batch size to 128 |
| Entropy collapse | Actions → HOLD, entropy < 0.1 early | Increase `ppo.ent_coef` to 0.02 temporarily or raise `reward_weights.time` |
| Value instability | `explained_variance` < 0, high value loss | Enable reward normalization (`VecNormalize`), raise `ppo.vf_coef`, check reward scaling |
| NaN losses | PPO prints `nan` for `policy_loss` | Decrease learning rate to 1e-4, tighten `max_grad_norm` to 0.3, enable reward clipping |
| Slow training | FPS < 400 | Disable debug logging, ensure GPU active, consider turning off MLflow artifact logging |

## Next Steps

- Run the full Task 3.3 prototype training sweep using the baseline config.
- Collect validation metrics and compare against the SL baseline (Sharpe ≥ 0.50, return ≥ +12%).
- Prepare for Task 3.4 hyperparameter tuning (Optuna search space defined in the config stub).

The infrastructure is ready—launch the training jobs and keep an eye on MLflow/TensorBoard dashboards for convergence sanity checks. Good luck! 🚀
