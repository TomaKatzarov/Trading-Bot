# Backtesting Results Logging Documentation

## Overview

The backtesting engine provides comprehensive results logging with optional MLflow integration for tracking experiments and performance metrics.

## Logged Metrics

The following performance metrics are calculated and logged by the backtesting engine:

### Portfolio Metrics
- **Initial Capital**: Starting capital amount
- **Final Portfolio Value**: End value of the portfolio
- **Total Return (%)**: Overall percentage return
- **Annualized Return (%)**: Annualized percentage return (assumes 252 trading days)
- **Annualized Volatility (%)**: Annualized volatility of returns
- **Sharpe Ratio**: Risk-adjusted return metric (assumes 0% risk-free rate)
- **Sortino Ratio**: Downside risk-adjusted return metric (assumes 0% target return)
- **Calmar Ratio**: Measures risk-adjusted return by dividing the annualized return by the maximum drawdown.
- **Max Drawdown (%)**: Maximum peak-to-trough decline

### Signal Quality Metrics (for SupervisedNNStrategy BUY Signals)
- **Precision**: The proportion of positive identifications that were actually correct.
- **Recall**: The proportion of actual positives that were identified correctly.
- **F1-score**: The harmonic mean of precision and recall, providing a single metric that balances both.

### Trade Metrics
- **Total Trades**: Number of completed trades
- **Winning Trades**: Number of profitable trades
- **Losing Trades**: Number of unprofitable trades
- **Win Rate (%)**: Percentage of winning trades
- **Loss Rate (%)**: Percentage of losing trades
- **Total PnL**: Total profit and loss in currency units
- **Total PnL (% of Initial Capital)**: Total PnL as percentage of starting capital
- **Average PnL per Trade**: Mean profit/loss per trade
- **Average Winning PnL**: Mean profit of winning trades
- **Average Losing PnL**: Mean loss of losing trades
- **Gross Profit**: Total profits from winning trades
- **Gross Loss**: Total losses from losing trades (absolute value)
- **Profit Factor**: Ratio of gross profit to gross loss
- **Average Holding Period (hours)**: Mean duration of trades

## MLflow Integration

### Configuration

MLflow logging can be enabled/disabled via the `mlflow_logging_enabled` parameter when initializing the `BacktestingEngine`:

```python
backtester = BacktestingEngine(
    csv_dir, symbol_list, initial_capital, heartbeat, start_date, end_date,
    strategy, portfolio, execution_handler,
    mlflow_logging_enabled=True  # Enable MLflow logging
)
```

### Robustness Features

- **Optional Dependency**: MLflow is treated as an optional dependency. If not installed, logging is automatically disabled with a warning message.
- **Graceful Degradation**: If MLflow logging is requested but unavailable, the backtest continues normally without MLflow integration.
- **Clear Status Messages**: The engine provides clear console output indicating MLflow status:
  - "MLflow logging enabled. Starting MLflow run..."
  - "MLflow logging requested but MLflow is not available. Skipping MLflow logging."
  - "MLflow logging disabled."

### Logged Parameters

The following strategy and backtest configuration parameters are logged to MLflow:

- `strategy_name`: Class name of the strategy
- `symbol_list`: List of symbols being backtested
- `initial_capital`: Starting capital amount
- `start_date`: Backtest start date
- `end_date`: Backtest end date

#### Strategy-Specific Parameters (if available)
- `model_path`: Path to the ML model file
- `scaler_path`: Path to the feature scaler file
- `lookback_window`: Number of historical periods used for predictions
- `signal_threshold`: Threshold for generating buy signals
- `exit_threshold`: Threshold for generating exit signals
- `max_holding_period_hours`: Maximum time to hold a position

### Logged Metrics

All performance metrics listed above (Portfolio, Trade, and Signal Quality Metrics) are logged to MLflow as metrics, allowing for easy comparison across different runs and experiments.

### Advanced Validation Techniques (Task 1.3.8)

The backtesting engine now includes comprehensive advanced validation techniques for robust strategy evaluation:

#### Advanced Performance Metrics

**Risk-Adjusted Metrics:**
- **Sharpe Ratio (annualized)**: Risk-adjusted return metric assuming 0% risk-free rate
- **Sortino Ratio (annualized)**: Downside risk-adjusted return metric assuming 0% target return
- **Calmar Ratio**: Measures risk-adjusted return by dividing annualized return by maximum drawdown
- **Maximum Drawdown**: Comprehensive peak-to-trough decline calculation with peak/trough dates

**Trade Performance Metrics:**
- **Profit Factor**: Ratio of gross profit to gross loss (enhanced calculation)
- **Average PnL per Trade**: Mean profit/loss per completed trade
- **Win Rate**: Percentage of profitable trades
- **Average Holding Period**: Mean duration of trades in hours

**Signal Quality Metrics (for SupervisedNNStrategy):**
- **Precision**: Proportion of positive identifications that were actually correct
- **Recall**: Proportion of actual positives that were identified correctly
- **F1-Score**: Harmonic mean of precision and recall

#### Threshold Optimization Mechanism

The `PerformanceMetrics` class includes a sophisticated utility to determine an optimal `signal_threshold` for the `SupervisedNNStrategy`:

**How it Works:**
1. Iterates through a predefined range of possible `signal_threshold` values (default: 0.1 to 0.9 in steps of 0.05)
2. For each threshold, simulates the strategy's BUY signals based on prediction probabilities
3. Calculates precision, recall, and F1-score for the simulated signals against true labels
4. Identifies the threshold that maximizes F1-score as the "optimal" threshold

**Usage:**
```python
from core.backtesting.metrics import PerformanceMetrics
import pandas as pd

# Initialize metrics calculator
metrics_calculator = PerformanceMetrics(initial_capital=100000)

# Perform threshold optimization
optimization_results = metrics_calculator.optimize_signal_threshold(
    prediction_probabilities_series,
    true_labels_series
)

optimal_threshold = optimization_results['optimal_threshold']
max_f1_score = optimization_results['max_f1_score']
metrics_at_optimal_threshold = optimization_results['metrics_at_optimal_threshold']

print(f"Optimal Threshold: {optimal_threshold}")
print(f"Max F1-Score: {max_f1_score:.4f}")
print(f"Precision: {metrics_at_optimal_threshold['precision']:.4f}")
print(f"Recall: {metrics_at_optimal_threshold['recall']:.4f}")
```

**Integration with Backtesting:**
The threshold optimization is automatically integrated into the backtesting workflow. During backtesting, prediction probabilities and true labels are collected, and threshold optimization can be performed post-backtest.

**MLflow Integration:**
When threshold optimization is performed within an MLflow-enabled backtest, the following results are logged as nested run metrics:
- `optimal_signal_threshold`
- `max_f1_score_at_optimal_threshold`
- `precision_at_optimal_threshold`
- `recall_at_optimal_threshold`

#### Testing and Validation

A comprehensive test suite is available in `scripts/test_advanced_validation.py` that demonstrates:
- Independent testing of all advanced metrics calculations
- Threshold optimization with synthetic data
- Full integration testing with the backtesting engine
- Verification of MLflow logging for all advanced metrics

**Running the Test Suite:**
```bash
python scripts/test_advanced_validation.py
```

This test suite verifies that all advanced validation techniques are working correctly and provides examples of their usage.

All performance metrics listed above are logged to MLflow as metrics, allowing for easy comparison across different runs and experiments.

### Logged Artifacts

The following artifacts are logged to MLflow (when available):

1. **Trade History CSV** (`closed_trades.csv`): Detailed record of all completed trades including:
   - Symbol
   - Entry/exit times and prices
   - Quantity
   - PnL
   - Holding period
   - Trade type

2. **Strategy Configuration JSON** (`strategy_config.json`): Complete strategy configuration as a JSON file for reproducibility.

## Environment Variables

MLflow behavior can be controlled through standard MLflow environment variables:

- `MLFLOW_TRACKING_URI`: Set the MLflow tracking server URI
- `MLFLOW_EXPERIMENT_NAME`: Set the experiment name for organizing runs

## Example Usage

```python
from core.backtesting.engine import BacktestingEngine

# Initialize with MLflow logging enabled
backtester = BacktestingEngine(
    csv_dir="data/historical_data",
    symbol_list=["AAPL"],
    initial_capital=100000.0,
    heartbeat=0.0,
    start_date=start_date,
    end_date=end_date,
    strategy=strategy,
    portfolio=portfolio,
    execution_handler=execution_handler,
    mlflow_logging_enabled=True
)

# Run backtest - metrics will be automatically logged
backtester.simulate_trading()
```

## Viewing Results

### Console Output
All metrics are printed to the console in a formatted summary.

### MLflow UI
If MLflow is configured, results can be viewed in the MLflow UI:

```bash
mlflow ui
```

Navigate to `http://localhost:5000` to view experiments, compare runs, and analyze results.

## Implementation Notes

- Metrics calculations handle edge cases (e.g., no trades, zero volatility)
- All monetary values are logged in the base currency units
- Time-based metrics use hours as the standard unit
- The Sharpe and Sortino ratios assume a 0% risk-free rate and 0% target return respectively
- Annualized calculations assume 252 trading days per year