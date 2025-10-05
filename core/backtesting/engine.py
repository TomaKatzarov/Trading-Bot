import pandas as pd
import numpy as np
from datetime import datetime
import time
import pytz
import mlflow
import mlflow.pyfunc
try:
    import mlflow
    import mlflow.pyfunc
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    print("MLflow not installed. MLflow logging will be disabled.")

from core.backtesting.strategy import Strategy
from core.backtesting.portfolio import Portfolio
from core.backtesting.execution import ExecutionHandler
from core.backtesting.data import HistoricCSVDataHandler
from core.backtesting.metrics import PerformanceMetrics

class BacktestingEngine:
    """
    Encapsulates the event-driven backtesting system.
    """
    def __init__(self, csv_dir, symbol_list, initial_capital,
                 heartbeat, start_date, end_date,
                 strategy, portfolio, execution_handler,
                 data_handler_cls=HistoricCSVDataHandler,
                 mlflow_logging_enabled=False):
        
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date
        self.end_date = end_date

        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.events = [] # Event queue
        self.data_handler = data_handler_cls(csv_dir, symbol_list, start_date, end_date, self.events)
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1 # Assuming one strategy for now
        self.mlflow_logging_enabled = mlflow_logging_enabled
        self.metrics = PerformanceMetrics(initial_capital, self.portfolio.closed_trades)
        self.all_prediction_probabilities = [] # To store probabilities for threshold optimization
        self.all_true_labels = [] # To store true labels for signal quality metrics and optimization

    def _run_backtest(self):
        """
        Carries out the backtest.
        """
        i = 0
        while True:
            i += 1
            # Update the market bars
            if self.data_handler.continue_backtest:
                self.data_handler.update_bars()
            else:
                break

            # Handle events
            while True:
                try:
                    event = self.events.pop(0)
                except IndexError:
                    break
                else:
                    if event.type == 'MARKET':
                        self.portfolio.update_timeindex(event)
                        # Capture prediction probabilities and true labels if available from the strategy
                        if hasattr(self.strategy, 'latest_prediction_probabilities') and \
                           hasattr(self.strategy, 'latest_true_label'):
                            self.all_prediction_probabilities.append(self.strategy.latest_prediction_probabilities)
                            self.all_true_labels.append(self.strategy.latest_true_label)
                        self.strategy.calculate_signals(event)

                    elif event.type == 'SIGNAL':
                        self.signals += 1
                        self.portfolio.update_signal(event)

                    elif event.type == 'ORDER':
                        self.orders += 1
                        self.execution_handler.execute_order(event)

                    elif event.type == 'FILL':
                        self.fills += 1
                        self.portfolio.update_fill(event)

            time.sleep(self.heartbeat) # Simulate real-time

    def simulate_trading(self):
        """
        Simulates the backtest and outputs the metrics, with optional MLflow logging.
        """
        if self.mlflow_logging_enabled and _MLFLOW_AVAILABLE:
            print("MLflow logging enabled. Starting MLflow run...")
            with mlflow.start_run():
                mlflow.log_param("strategy_name", self.strategy.__class__.__name__)
                mlflow.log_param("symbol_list", self.symbol_list)
                mlflow.log_param("initial_capital", self.initial_capital)
                mlflow.log_param("start_date", self.start_date.strftime('%Y-%m-%d'))
                mlflow.log_param("end_date", self.end_date.strftime('%Y-%m-%d'))
                
                # Assuming strategy has these attributes for logging
                if hasattr(self.strategy, 'model_path'):
                    mlflow.log_param("model_path", self.strategy.model_path)
                if hasattr(self.strategy, 'scaler_path'):
                    mlflow.log_param("scaler_path", self.strategy.scaler_path)
                if hasattr(self.strategy, 'lookback_window'):
                    mlflow.log_param("lookback_window", self.strategy.lookback_window)
                if hasattr(self.strategy, 'signal_threshold'):
                    mlflow.log_param("signal_threshold", self.strategy.signal_threshold)
                if hasattr(self.strategy, 'exit_threshold'):
                    mlflow.log_param("exit_threshold", self.strategy.exit_threshold)
                if hasattr(self.strategy, 'max_holding_period_hours'):
                    mlflow.log_param("max_holding_period_hours", self.strategy.max_holding_period_hours)

                self._run_backtest()
                self.portfolio.create_equity_curve_dataframe() # Generate equity curve in portfolio
                self.metrics.create_equity_curve_dataframe(self.portfolio.equity_curve) # Pass to metrics
                
                # Convert lists of probabilities and labels to pandas Series for metric calculation
                # Assuming they are collected in the same order as market events
                all_probs_series = pd.Series(self.all_prediction_probabilities) if self.all_prediction_probabilities else None
                all_labels_series = pd.Series(self.all_true_labels) if self.all_true_labels else None

                # Get metrics and log them, passing signal data if available
                metrics_dict = self.metrics.output_summary_stats(
                    strategy_signals=(all_probs_series >= self.strategy.signal_threshold).astype(int) if all_probs_series is not None else None,
                    true_labels=all_labels_series
                )
                mlflow.log_metrics(metrics_dict)

                # Log artifacts
                if self.portfolio.closed_trades:
                    trades_df = pd.DataFrame(self.portfolio.closed_trades)
                    trades_csv_path = "closed_trades.csv"
                    trades_df.to_csv(trades_csv_path, index=False)
                    mlflow.log_artifact(trades_csv_path)
                    print(f"Logged closed trades to MLflow as {trades_csv_path}")
                
                # Log strategy configuration if available
                if hasattr(self.strategy, 'config'):
                    import json
                    config_json_path = "strategy_config.json"
                    with open(config_json_path, 'w') as f:
                        json.dump(self.strategy.config, f, indent=4)
                    mlflow.log_artifact(config_json_path)
                    print(f"Logged strategy config to MLflow as {config_json_path}")

        elif self.mlflow_logging_enabled and not _MLFLOW_AVAILABLE:
            print("MLflow logging requested but MLflow is not available. Skipping MLflow logging.")
            self._run_backtest()
            self.portfolio.create_equity_curve_dataframe()
            self.metrics.create_equity_curve_dataframe(self.portfolio.equity_curve)
            
            all_probs_series = pd.Series(self.all_prediction_probabilities) if self.all_prediction_probabilities else None
            all_labels_series = pd.Series(self.all_true_labels) if self.all_true_labels else None

            self.metrics.output_summary_stats(
                strategy_signals=(all_probs_series >= self.strategy.signal_threshold).astype(int) if all_probs_series is not None else None,
                true_labels=all_labels_series
            )
            return # Exit early if MLflow is not available but was requested
        else:
            print("MLflow logging disabled.")
            self._run_backtest()
            self.portfolio.create_equity_curve_dataframe()
            self.metrics.create_equity_curve_dataframe(self.portfolio.equity_curve)
            
            all_probs_series = pd.Series(self.all_prediction_probabilities) if self.all_prediction_probabilities else None
            all_labels_series = pd.Series(self.all_true_labels) if self.all_true_labels else None

            self.metrics.output_summary_stats(
                strategy_signals=(all_probs_series >= self.strategy.signal_threshold).astype(int) if all_probs_series is not None else None,
                true_labels=all_labels_series
            )


if __name__ == "__main__":
    import time
    from core.backtesting.data import HistoricCSVDataHandler
    from core.backtesting.strategy import BuyAndHoldStrategy # Example strategy
    from core.backtesting.portfolio import NaivePortfolio # Example portfolio
    from core.backtesting.execution import SimulatedExecutionHandler # Example execution

    # Example Usage:
    csv_dir = "data/historical_data" # Directory where your CSVs are
    symbol_list = ["SPY"] # Example symbol
    initial_capital = 100000.0
    heartbeat = 0.0 # No delay for backtesting
    start_date = datetime(2020, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
    end_date = datetime(2020, 12, 31, 0, 0, 0, tzinfo=pytz.UTC)

    # Create instances of components
    events_queue = [] # Shared event queue
    data_handler = HistoricCSVDataHandler(csv_dir, symbol_list, start_date, end_date, events_queue)
    strategy = BuyAndHoldStrategy(data_handler, events_queue)
    portfolio = NaivePortfolio(data_handler, events_queue, initial_capital, start_date)
    execution_handler = SimulatedExecutionHandler(events_queue)

    # Initialize and run the backtesting engine
    backtester = BacktestingEngine(
        csv_dir, symbol_list, initial_capital, heartbeat, start_date, end_date,
        strategy, portfolio, execution_handler, data_handler_cls=HistoricCSVDataHandler
    )
    backtester.simulate_trading()