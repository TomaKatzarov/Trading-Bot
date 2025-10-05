# TradingBotAI

This project aims to develop an AI-powered trading system for autonomous trading, leveraging custom Neural Networks (NN) and Reinforcement Learning (RL) techniques. The system operates locally, connects to paper trading accounts via Alpaca API, and is designed to maximize risk-adjusted profitability.

## Project Structure

- `core/`: Contains the core logic of the trading bot, including data handling, model architectures, backtesting engine, and strategy implementations.
- `scripts/`: Houses various utility scripts for training, backtesting, data processing, and other operational tasks.
- `models/`: Stores trained machine learning models and scalers.
- `data/`: (Will be created by data pipeline) Stores historical market data and processed features.
- `logs/`: (Will be created by logging) Stores application logs.
- `memory-bank/`: Contains project documentation, architectural decisions, and progress tracking.

## Key Features

- **Custom Neural Networks:** Implementation of MLP, LSTM/GRU, and CNN-LSTM hybrid architectures for financial time series forecasting.
- **Reinforcement Learning Integration:** Future integration of RL algorithms (e.g., PPO, SAC, DQN) for autonomous trading decisions.
- **Robust Data Pipeline:** Modules for historical data loading, technical indicator generation, sentiment analysis integration, and data preprocessing.
- **High-Fidelity Backtesting Engine:** Event-driven simulation with cost modeling (commission, slippage) and support for Out-of-Sample (OOS) testing and Walk-Forward Validation.
- **MLflow Integration:** For experiment tracking, model versioning, and hyperparameter tuning with Optuna.
- **Risk Management:** Basic stop-loss/take-profit rules and dynamic position sizing.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TomaKatzarov/TradingBotAI.git
    cd TradingBotAI
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Variables:**
    Create a `.env` file in the project root based on `Credential.env.example` and fill in your API credentials.

    ```
    # .env
    ALPACA_API_KEY=your_alpaca_api_key
    ALPACA_SECRET_KEY=your_alpaca_secret_key
    # Add other sensitive variables here
    ```

## Usage

### End-to-end data refresh pipeline

Use `scripts/run_full_data_update.py` to refresh historical data, recompute
derived features, and regenerate the supervised-learning dataset that powers the
training campaigns. The script stitches together historical ingestion,
technical indicator calculation, sentiment attachment, data gap remediation,
and dataset validation in a single runnable entry point.

```bash
python scripts/run_full_data_update.py
```

Key options include `--start-date/--end-date` for append-only refreshes,
`--max-workers` to control parallelism, and `--skip-*` flags to omit individual
steps when troubleshooting. By default, the regenerated dataset is written to
`data/training_data_v2_final`, matching the path expected by the training and
evaluation scripts documented under `training/`.

Detailed usage instructions for training models, running backtests, and
deploying the trading system will be provided in specific documentation within
the `scripts/` and `core/` directories.

## Contributing

Contributions are welcome! Please refer to the `CONTRIBUTING.md` (to be created) for guidelines.

## License

This project is licensed under the MIT License. See the `LICENSE` (to be created) file for details.