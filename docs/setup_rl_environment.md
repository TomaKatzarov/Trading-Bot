# RL Environment Setup Guide

This document captures the end-to-end procedure for provisioning and verifying the reinforcement learning (RL) development environment used by the Trading Bot project.

## System Requirements

| Component | Minimum Specification | Recommended |
| --- | --- | --- |
| Operating System | Windows 11 / Ubuntu 22.04 | Windows 11 build 26220+ |
| Python | 3.10 or newer | 3.12.10 (current)
| CUDA Toolkit | 12.1 | 12.8 (driver/runtime bundled with PyTorch build)
| GPU | 1× NVIDIA RTX-class GPU, ≥8 GB VRAM | NVIDIA GeForce RTX 5070 Ti (15.9 GB)
| RAM | 32 GB | 64 GB+
| Storage | 200 GB free SSD | 1 TB SSD for data/checkpoints

## Virtual Environment Provisioning

1. **Create the dedicated RL virtual environment**
   ```bash
   C:/TradingBotAI/.venv/Scripts/python.exe -m venv trading_rl_env
   ```
2. **Upgrade `pip` inside the new environment**
   ```bash
   C:/TradingBotAI/trading_rl_env/Scripts/python.exe -m pip install --upgrade pip
   ```
3. **Install project base dependencies**
   ```bash
   C:/TradingBotAI/trading_rl_env/Scripts/python.exe -m pip install -r requirements.txt
   ```
4. **Install RL-specific dependencies**
   ```bash
   C:/TradingBotAI/trading_rl_env/Scripts/python.exe -m pip install -r requirements_rl.txt
   ```

5. **(Optional) Install additional development tooling**
   Pytest is bundled with `requirements_rl.txt`, but if you maintain a separate dev requirements file, run:
   ```bash
   C:/TradingBotAI/trading_rl_env/Scripts/python.exe -m pip install pytest
   ```

### Activation Helpers

- **Windows:** `activate_rl_env.bat`
- **Linux / macOS:** `source activate_rl_env.sh`

Both scripts live in the repository root and activate `trading_rl_env` from any shell.

## Verification Procedures

1. **Environment audit**
   ```bash
   C:/TradingBotAI/.venv/Scripts/python.exe scripts/verify_rl_environment.py > docs/environment_verification.txt
   cat docs/environment_verification.txt
   ```
   Key checks (2025-10-05 run):
   - Python 3.12.10 (CPython)
   - PyTorch 2.8.0.dev20250415+cu128
   - CUDA 12.8 detected with 1× NVIDIA GeForce RTX 5070 Ti (15.92 GB VRAM)

2. **RL library validation**
   ```bash
   C:/TradingBotAI/.venv/Scripts/python.exe scripts/verify_rl_libraries.py > docs/rl_library_verification.txt
   cat docs/rl_library_verification.txt
   ```
   Confirms Gymnasium 1.1.1, Stable-Baselines3 2.7.0, Ray 2.49.2, and CartPole sanity check.

3. **GPU readiness benchmark**
   ```bash
   C:/TradingBotAI/.venv/Scripts/python.exe scripts/test_gpu_rl_readiness.py > docs/gpu_readiness_report.txt
   cat docs/gpu_readiness_report.txt
   ```
   Highlights: matrix multiply (10×5000²) in 0.187 s, NN forward pass (100×256 batches) in 0.061 s, confirming target <5 s / <1 s thresholds.

4. **Run core RL unit tests (sanity check)**
   Activate the environment before running tests:
   ```bash
   source activate_rl_env.sh
   python -m pytest tests/test_trading_env.py
   ```
   This ensures `gymnasium` and related dependencies resolve correctly inside `trading_rl_env`.

## Troubleshooting

| Symptom | Likely Cause | Resolution |
| --- | --- | --- |
| `torch.cuda.is_available()` returns `False` | GPU drivers or CUDA runtime mismatch | Install latest NVIDIA drivers aligned with CUDA 12.8, reboot, rerun verification script |
| `pip install` fails with build errors | Missing compiler toolchain | Install Visual Studio Build Tools (Windows) or `build-essential` (Linux) |
| `ray` import errors inside venv | Package installed in wrong interpreter | Ensure commands use `trading_rl_env/Scripts/python.exe` or activate environment before installation |
| Unicode errors in verification logs | Windows console codepage limitations | Redirect output to file (as shown) or run `chcp 65001` before executing scripts |

## Artifact Locations

- `docs/environment_verification.txt`
- `docs/rl_library_verification.txt`
- `docs/gpu_readiness_report.txt`
- `scripts/verify_rl_environment.py`
- `scripts/verify_rl_libraries.py`
- `scripts/test_gpu_rl_readiness.py`
- `requirements_rl.txt`

## Change Log

- **2025-10-05:** Initial environment provisioning and verification completed.
