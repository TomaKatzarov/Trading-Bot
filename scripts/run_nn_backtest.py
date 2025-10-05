#!/usr/bin/env python3
"""
Neural Network Model Backtesting Orchestration Script

This script orchestrates comprehensive backtesting of trained Neural Network models
using the SupervisedNNStrategy and the core backtesting engine.

Implements Activity 7.3: Integration with Backtesting Framework from the operational plan.

Features:
- Load trained models from local checkpoints or MLflow artifacts
- Configure and run backtests on specified symbols and date ranges
- Calculate comprehensive trading performance metrics
- Log results to MLflow for experiment tracking
- Generate detailed backtesting reports

Author: Flow-Code
Date: 2025-09-30
Version: 1.0
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.backtesting.engine import BacktestingEngine
from core.backtesting.data import BacktestDataHandler
from core.strategies.supervised_nn_strategy import SupervisedNNStrategy
from core.backtesting.metrics import PerformanceMetrics
from core.experiment_management.enhanced_logging import EnhancedMLflowLogger
from core.experiment_management.experiment_organizer import ExperimentOrganizer
from core.experiment_management.reporting import ExperimentReporter

# MLflow integration
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NNBacktestOrchestrator:
    """
    Orchestrates backtesting of Neural Network models with comprehensive logging and reporting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the backtest orchestrator.
        
        Args:
            config: Configuration dictionary containing:
                - model_source: 'local' or 'mlflow'
                - model_path: Path to model checkpoint (if local)
                - mlflow_run_id: MLflow run ID (if mlflow)
                - scaler_path: Path to scaler (if local)
                - asset_id_map_path: Path to asset ID mapping (if local)
                - symbols: List of symbols to backtest
                - start_date: Backtest start date
                - end_date: Backtest end date
                - initial_capital: Initial capital for backtesting
                - strategy_config: Strategy-specific configuration
                - mlflow_experiment_name: MLflow experiment name for logging
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize experiment tracking
        self.exp_organizer = None
        self.enhanced_logger = None
        self.reporter = None
        
        if MLFLOW_AVAILABLE and config.get('use_mlflow_logging', True):
            self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow experiment tracking."""
        try:
            experiment_name = self.config.get(
                'mlflow_experiment_name',
                'NN_Backtesting'
            )
            
            self.exp_organizer = ExperimentOrganizer(
                base_experiment_name=experiment_name
            )
            
            # Create experiment run name
            model_type = self.config.get('strategy_config', {}).get('model_type', 'unknown')
            symbols_str = '_'.join(self.config.get('symbols', ['multi'])[:3])
            run_name = f"backtest_{model_type}_{symbols_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Setup MLflow run
            experiment_id = mlflow.create_experiment(experiment_name) \
                if not mlflow.get_experiment_by_name(experiment_name) \
                else mlflow.get_experiment_by_name(experiment_name).experiment_id
            
            mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name
            )
            
            self.enhanced_logger = EnhancedMLflowLogger()
            self.reporter = ExperimentReporter()
            
            # Log configuration parameters
            mlflow.log_params({
                'model_source': self.config.get('model_source'),
                'symbols': ','.join(self.config.get('symbols', [])),
                'start_date': self.config.get('start_date'),
                'end_date': self.config.get('end_date'),
                'initial_capital': self.config.get('initial_capital'),
                'backtest_type': 'nn_model'
            })
            
            # Log strategy configuration
            for key, value in self.config.get('strategy_config', {}).items():
                mlflow.log_param(f"strategy_{key}", value)
            
            self.logger.info(f"MLflow tracking initialized for experiment: {experiment_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup MLflow tracking: {e}")
            self.exp_organizer = None
            self.enhanced_logger = None
    
    def load_strategy(self) -> SupervisedNNStrategy:
        """
        Load and configure the SupervisedNNStrategy with trained model.
        
        Returns:
            Configured SupervisedNNStrategy instance
        """
        self.logger.info("Loading SupervisedNNStrategy...")
        
        strategy_config = self.config.get('strategy_config', {})
        strategy = SupervisedNNStrategy(strategy_config)
        
        # Load model dependencies based on source
        model_source = self.config.get('model_source', 'local')
        
        if model_source == 'mlflow':
            if not MLFLOW_AVAILABLE:
                raise ImportError("MLflow is required for loading models from MLflow")
            
            mlflow_run_id = self.config.get('mlflow_run_id')
            if not mlflow_run_id:
                raise ValueError("mlflow_run_id must be specified when model_source is 'mlflow'")
            
            self.logger.info(f"Loading model from MLflow run: {mlflow_run_id}")
            strategy.load_dependencies_from_mlflow(
                run_id=mlflow_run_id,
                artifact_path=self.config.get('mlflow_artifact_path', 'best_model')
            )
        else:
            # Load from local paths
            if not strategy.model:  # If not already loaded in __init__
                strategy.load_dependencies()
        
        self.logger.info("Strategy loaded successfully")
        return strategy
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Execute the backtest and return comprehensive results.
        
        Returns:
            Dictionary containing backtest results and metrics
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING NEURAL NETWORK MODEL BACKTEST")
        self.logger.info("=" * 80)
        
        # Load strategy
        strategy = self.load_strategy()
        
        # Get backtest parameters
        symbols = self.config.get('symbols', [])
        start_date = self.config.get('start_date')
        end_date = self.config.get('end_date')
        initial_capital = self.config.get('initial_capital', 100000)
        
        self.logger.info(f"Symbols: {symbols}")
        self.logger.info(f"Period: {start_date} to {end_date}")
        self.logger.info(f"Initial Capital: ${initial_capital:,.2f}")
        
        # Initialize data handler
        data_handler = BacktestDataHandler(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        # Initialize backtesting engine
        engine = BacktestingEngine(
            strategy=strategy,
            data_handler=data_handler,
            initial_capital=initial_capital
        )
        
        # Run backtest
        self.logger.info("\nExecuting backtest...")
        results = engine.run()
        
        # Calculate comprehensive metrics
        metrics_calculator = PerformanceMetrics(
            initial_capital=initial_capital,
            closed_trades=results.get('closed_trades', [])
        )
        
        if 'equity_curve' in results:
            metrics_calculator.create_equity_curve_dataframe(results['equity_curve'])
        
        comprehensive_metrics = metrics_calculator.calculate_all_metrics()
        
        # Log results
        self.logger.info("\n" + "=" * 80)
        self.logger.info("BACKTEST RESULTS")
        self.logger.info("=" * 80)
        self._log_backtest_results(comprehensive_metrics)
        
        # Log to MLflow if available
        if self.enhanced_logger and mlflow and mlflow.active_run():
            self._log_to_mlflow(comprehensive_metrics, results)
        
        # Generate report if reporter available
        if self.reporter and mlflow and mlflow.active_run():
            self._generate_backtest_report(comprehensive_metrics, results)
        
        return {
            'metrics': comprehensive_metrics,
            'results': results,
            'strategy_config': self.config.get('strategy_config', {})
        }
    
    def _log_backtest_results(self, metrics: Dict[str, Any]):
        """Log backtest results to console."""
        self.logger.info("\nPerformance Metrics:")
        self.logger.info(f"  Total Return: {metrics.get('total_return', 0):.2f}%")
        self.logger.info(f"  Annualized Return: {metrics.get('annual_return', 0):.2f}%")
        self.logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        self.logger.info(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.4f}")
        self.logger.info(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.4f}")
        self.logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        
        self.logger.info("\nTrade Metrics:")
        self.logger.info(f"  Total Trades: {metrics.get('total_trades', 0)}")
        self.logger.info(f"  Winning Trades: {metrics.get('winning_trades', 0)}")
        self.logger.info(f"  Losing Trades: {metrics.get('losing_trades', 0)}")
        self.logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.2f}%")
        self.logger.info(f"  Profit Factor: {metrics.get('profit_factor', 0):.4f}")
        self.logger.info(f"  Avg PnL per Trade: ${metrics.get('avg_pnl_per_trade', 0):.2f}")
        
        if metrics.get('signal_quality'):
            self.logger.info("\nSignal Quality Metrics:")
            sq = metrics['signal_quality']
            self.logger.info(f"  Precision: {sq.get('precision', 0):.4f}")
            self.logger.info(f"  Recall: {sq.get('recall', 0):.4f}")
            self.logger.info(f"  F1-Score: {sq.get('f1_score', 0):.4f}")
    
    def _log_to_mlflow(self, metrics: Dict[str, Any], results: Dict[str, Any]):
        """Log comprehensive metrics to MLflow."""
        try:
            self.logger.info("\nLogging results to MLflow...")
            
            # Log performance metrics
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"backtest_{metric_name}", metric_value)
                elif isinstance(metric_value, dict):
                    # Handle nested metrics (e.g., signal_quality)
                    for sub_name, sub_value in metric_value.items():
                        if isinstance(sub_value, (int, float)):
                            mlflow.log_metric(f"backtest_{metric_name}_{sub_name}", sub_value)
            
            # Log equity curve as artifact if available
            if 'equity_curve' in results and results['equity_curve'] is not None:
                equity_df = results['equity_curve']
                equity_path = Path("temp_equity_curve.csv")
                equity_df.to_csv(equity_path)
                mlflow.log_artifact(str(equity_path), "backtest_results")
                equity_path.unlink()  # Clean up temp file
            
            # Log trade details if available
            if 'closed_trades' in results and results['closed_trades']:
                trades_df = pd.DataFrame(results['closed_trades'])
                trades_path = Path("temp_trades.csv")
                trades_df.to_csv(trades_path, index=False)
                mlflow.log_artifact(str(trades_path), "backtest_results")
                trades_path.unlink()  # Clean up temp file
            
            self.logger.info("Results logged to MLflow successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to log to MLflow: {e}")
    
    def _generate_backtest_report(self, metrics: Dict[str, Any], results: Dict[str, Any]):
        """Generate comprehensive backtest report."""
        try:
            self.logger.info("\nGenerating backtest report...")
            
            # This will be enhanced when we add backtesting report capabilities
            # to ExperimentReporter in Activity 7.4
            self.logger.info("Report generation capability will be enhanced in Activity 7.4")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate report: {e}")
    
    def cleanup(self):
        """Cleanup and finalize MLflow run."""
        if MLFLOW_AVAILABLE and mlflow and mlflow.active_run():
            mlflow.end_run()
            self.logger.info("MLflow run ended")


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix in ['.yaml', '.yml']:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def create_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Create configuration dictionary from command-line arguments."""
    
    # Load base config from file if provided
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = {}
    
    # Override with command-line arguments
    if args.symbols:
        config['symbols'] = args.symbols
    if args.start_date:
        config['start_date'] = args.start_date
    if args.end_date:
        config['end_date'] = args.end_date
    if args.initial_capital:
        config['initial_capital'] = args.initial_capital
    
    # Model source configuration
    config['model_source'] = args.model_source
    
    if args.model_source == 'mlflow':
        if not args.mlflow_run_id:
            raise ValueError("--mlflow-run-id is required when model_source is 'mlflow'")
        config['mlflow_run_id'] = args.mlflow_run_id
        config['mlflow_artifact_path'] = args.mlflow_artifact_path
    else:
        if not args.model_path:
            raise ValueError("--model-path is required when model_source is 'local'")
        config['strategy_config'] = config.get('strategy_config', {})
        config['strategy_config']['model_path'] = args.model_path
        config['strategy_config']['scaler_path'] = args.scaler_path
        config['strategy_config']['asset_id_map_path'] = args.asset_id_map_path
    
    # Strategy parameters
    if 'strategy_config' not in config:
        config['strategy_config'] = {}
    
    if args.signal_threshold is not None:
        config['strategy_config']['signal_threshold'] = args.signal_threshold
    if args.exit_threshold is not None:
        config['strategy_config']['exit_threshold'] = args.exit_threshold
    if args.max_holding_period is not None:
        config['strategy_config']['max_holding_period_hours'] = args.max_holding_period
    
    # MLflow logging
    config['use_mlflow_logging'] = args.use_mlflow_logging
    if args.mlflow_experiment_name:
        config['mlflow_experiment_name'] = args.mlflow_experiment_name
    
    return config


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run Neural Network Model Backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest with local model
  python scripts/run_nn_backtest.py --model-source local \\
      --model-path models/best_model.pt \\
      --scaler-path models/scalers.joblib \\
      --symbols AAPL MSFT \\
      --start-date 2024-01-01 --end-date 2024-12-31
  
  # Backtest with MLflow model
  python scripts/run_nn_backtest.py --model-source mlflow \\
      --mlflow-run-id abc123def456 \\
      --symbols AAPL MSFT NVDA \\
      --start-date 2024-01-01 --end-date 2024-12-31
  
  # Use configuration file
  python scripts/run_nn_backtest.py --config config/backtest_config.yaml
        """
    )
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to configuration file (YAML or JSON)')
    
    # Model source
    parser.add_argument('--model-source', type=str, choices=['local', 'mlflow'],
                       default='local', help='Model source: local file or MLflow')
    
    # Local model arguments
    parser.add_argument('--model-path', type=str, help='Path to model checkpoint (for local source)')
    parser.add_argument('--scaler-path', type=str, help='Path to scaler file (for local source)')
    parser.add_argument('--asset-id-map-path', type=str, help='Path to asset ID mapping JSON')
    
    # MLflow model arguments
    parser.add_argument('--mlflow-run-id', type=str, help='MLflow run ID (for mlflow source)')
    parser.add_argument('--mlflow-artifact-path', type=str, default='best_model',
                       help='Artifact path within MLflow run')
    
    # Backtest parameters
    parser.add_argument('--symbols', type=str, nargs='+', help='List of symbols to backtest')
    parser.add_argument('--start-date', type=str, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float, default=100000,
                       help='Initial capital for backtesting')
    
    # Strategy parameters
    parser.add_argument('--signal-threshold', type=float, help='Signal probability threshold')
    parser.add_argument('--exit-threshold', type=float, help='Exit probability threshold')
    parser.add_argument('--max-holding-period', type=int, help='Maximum holding period in hours')
    
    # MLflow logging
    parser.add_argument('--use-mlflow-logging', action='store_true', default=True,
                       help='Enable MLflow logging (default: True)')
    parser.add_argument('--no-mlflow-logging', action='store_false', dest='use_mlflow_logging',
                       help='Disable MLflow logging')
    parser.add_argument('--mlflow-experiment-name', type=str, help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.config and not args.symbols:
        parser.error("Either --config or --symbols must be provided")
    if not args.config and not (args.start_date and args.end_date):
        parser.error("--start-date and --end-date are required when not using --config")
    
    try:
        # Create configuration
        config = create_config_from_args(args)
        
        # Initialize orchestrator
        orchestrator = NNBacktestOrchestrator(config)
        
        # Run backtest
        backtest_results = orchestrator.run_backtest()
        
        # Cleanup
        orchestrator.cleanup()
        
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Backtest failed with error: {str(e)}", exc_info=True)
        
        # Ensure MLflow run is ended even on failure
        if MLFLOW_AVAILABLE and mlflow and mlflow.active_run():
            mlflow.end_run(status='FAILED')
        
        return 1


if __name__ == "__main__":
    sys.exit(main())