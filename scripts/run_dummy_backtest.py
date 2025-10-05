#!/usr/bin/env python3
"""
Script to run a dummy backtest using SupervisedNNStrategy with sample data and dummy model.
This tests the complete integration of the strategy with dummy components.
"""

# Add the project root to the path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytz
from datetime import datetime

from core.backtesting.engine import BacktestingEngine
from core.backtesting.data import HistoricCSVDataHandler
from core.backtesting.portfolio import Portfolio
from core.backtesting.execution import SimulatedExecutionHandler
from core.backtesting.event import Event # Although not directly used, good to have for context

import torch
import torch.nn as nn
from core.strategies.supervised_nn_strategy import SupervisedNNStrategy

# Define the dummy model class (needed for loading)
class DummyNNModel(nn.Module):
    """
    A simple dummy neural network model that produces predictable outputs.
    """
    
    def __init__(self, num_features=17, lookback_window=24):
        super(DummyNNModel, self).__init__()
        self.num_features = num_features
        self.lookback_window = lookback_window
        
        # Simple linear layer (not actually used in forward pass for predictable behavior)
        self.linear = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, asset_id_tensor=None):
        """
        Forward pass with predictable logic.
        
        Args:
            x: Input tensor of shape (batch_size, lookback_window, num_features)
            asset_id_tensor: Optional asset ID tensor (not used in dummy model)
            
        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        batch_size = x.shape[0]
        
        # Get the last timestep's features for each sample in the batch
        last_features = x[:, -1, :]  # Shape: (batch_size, num_features)
        
        # Sum the features for each sample
        feature_sums = torch.sum(last_features, dim=1)  # Shape: (batch_size,)
        
        # Apply simple logic: if sum > 10, high probability (0.8), else low (0.2)
        probabilities = torch.where(feature_sums > 10.0,
                                   torch.tensor(0.8, dtype=torch.float32),
                                   torch.tensor(0.2, dtype=torch.float32))
        
        # Reshape to (batch_size, 1)
        probabilities = probabilities.unsqueeze(1)
        
        return probabilities


def run_backtest():
    """Run the dummy backtest using the core BacktestingEngine."""
    print("=" * 60)
    print("DUMMY BACKTEST - SupervisedNNStrategy Integration Test with BacktestingEngine")
    print("=" * 60)
    
    # Configuration
    csv_dir = "data/sample_test_data" # Directory containing your CSVs
    symbol_list = ["AAPL"] # Example symbol
    initial_capital = 100000.0
    heartbeat = 0.0 # No delay for backtesting
    
    # Define start and end dates for the backtest
    # These should align with your sample data
    start_date = datetime(2020, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
    end_date = datetime(2020, 1, 31, 23, 0, 0, tzinfo=pytz.UTC) # Adjust based on your sample data

    # Configure the strategy
    strategy_config = {
        'model_path': 'models/dummy_test_artifacts/dummy_model.pt',
        'scaler_path': 'models/dummy_test_artifacts/dummy_scaler.joblib',
        'asset_id_map_path': None,  # Not using asset IDs for single symbol test
        'feature_list': [
            'SMA_10', 'SMA_20', 'MACD_line', 'MACD_signal', 'MACD_hist',
            'RSI_14', 'Stoch_K', 'Stoch_D', 'ADX_14', 'ATR_14',
            'BB_bandwidth', 'OBV', 'Volume_SMA_20', 'Return_1h',
            'sentiment_score_hourly_ffill', 'DayOfWeek_sin', 'DayOfWeek_cos'
        ],
        'lookback_window': 24,
        'signal_threshold': 0.7,
        'exit_threshold': 0.4,
        'max_holding_period_hours': 8
    }
    
    print(f"\nStrategy Configuration:")
    for key, value in strategy_config.items():
        print(f"  {key}: {value}")
    
    # Initialize strategy
    print(f"\nInitializing SupervisedNNStrategy...")
    try:
        strategy = SupervisedNNStrategy(strategy_config)
        print("Strategy initialized successfully!")
    except Exception as e:
        print(f"Error initializing strategy: {e}")
        return

    # Create instances of components for BacktestingEngine
    events_queue = [] # Shared event queue
    data_handler = HistoricCSVDataHandler(csv_dir, symbol_list, start_date, end_date, events_queue)
    
    # Pass the strategy instance to the portfolio
    portfolio = Portfolio(data_handler, events_queue, initial_capital, start_date, strategy)
    execution_handler = SimulatedExecutionHandler(events_queue)

    # Initialize and run the backtesting engine
    print(f"\nInitializing BacktestingEngine...")
    backtester = BacktestingEngine(
        csv_dir, symbol_list, initial_capital, heartbeat, start_date, end_date,
        strategy, portfolio, execution_handler,
        data_handler_cls=HistoricCSVDataHandler,
        mlflow_logging_enabled=True # Enable MLflow logging for this dummy run
    )
    print(f"BacktestingEngine initialized with ${initial_capital:,.2f} capital.")
    
    print(f"\nRunning backtest from {start_date} to {end_date}...")
    print("-" * 40)
    
    backtester.simulate_trading()
    
    # --- Demonstrate Threshold Optimization ---
    print("\n" + "=" * 60)
    print("THRESHOLD OPTIMIZATION DEMONSTRATION")
    print("=" * 60)

    # Ensure we have data for optimization
    if backtester.all_prediction_probabilities and backtester.all_true_labels:
        # Convert lists to pandas Series for the optimization function
        prediction_probs_series = pd.Series(backtester.all_prediction_probabilities)
        true_labels_series = pd.Series(backtester.all_true_labels)

        # Perform threshold optimization
        optimization_results = backtester.metrics.optimize_signal_threshold(
            prediction_probs_series,
            true_labels_series
        )

        optimal_threshold = optimization_results['optimal_threshold']
        max_f1_score = optimization_results['max_f1_score']
        metrics_at_optimal_threshold = optimization_results['metrics_at_optimal_threshold']

        print(f"\nOptimal Signal Threshold: {optimal_threshold:.2f}")
        print(f"Max F1-Score at Optimal Threshold: {max_f1_score:.4f}")
        print(f"Precision at Optimal Threshold: {metrics_at_optimal_threshold.get('precision', 0.0):.4f}")
        print(f"Recall at Optimal Threshold: {metrics_at_optimal_threshold.get('recall', 0.0)::.4f}")

        # Try to log optimization results to MLflow if available
        try:
            import mlflow
            if backtester.mlflow_logging_enabled:
                with mlflow.start_run(nested=True): # Log optimization results as a nested run
                    mlflow.log_metric("optimal_signal_threshold", optimal_threshold)
                    mlflow.log_metric("max_f1_score_at_optimal_threshold", max_f1_score)
                    mlflow.log_metric("precision_at_optimal_threshold", metrics_at_optimal_threshold.get('precision', 0.0))
                    mlflow.log_metric("recall_at_optimal_threshold", metrics_at_optimal_threshold.get('recall', 0.0))
                    print("Logged threshold optimization results to MLflow.")
        except ImportError:
            print("MLflow not available for logging optimization results.")
    else:
        print("Not enough prediction probabilities or true labels collected for threshold optimization.")
    
    print("\n" + "=" * 60)
    print("BACKTEST COMPLETED")
    print("=" * 60)
    
    # Verification (simplified as metrics are now handled by PerformanceMetrics)
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    print("✓ Strategy initialized and loaded dependencies successfully")
    print("✓ BacktestingEngine executed without critical errors")
    print("✓ Performance metrics calculated and printed (see above)")
    print("✓ MLflow logging attempted (check your MLflow UI if configured)")
    
    print(f"\nTest completed successfully!")

if __name__ == "__main__":
    run_backtest()