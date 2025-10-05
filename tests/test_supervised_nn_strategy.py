"""
Unit tests for SupervisedNNStrategy class.

This module provides comprehensive testing for the trade entry/exit logic
implemented in the SupervisedNNStrategy class.

Part of Task 1.3.2: Implement trade entry/exit rules aligned with NN signals.
"""

import unittest
import logging
import tempfile
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import StandardScaler
from core.strategies.supervised_nn_strategy import SupervisedNNStrategy


class TestSupervisedNNStrategy(unittest.TestCase):
    """Test cases for SupervisedNNStrategy class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Suppress logging during tests
        logging.disable(logging.CRITICAL)
        
        # Default configuration for most tests
        self.default_config = {
            'signal_threshold': 0.7,
            'exit_threshold': 0.3,
            'max_holding_period_hours': 8
        }
        
        # Configuration without exit_threshold
        self.config_no_exit_threshold = {
            'signal_threshold': 0.7,
            'max_holding_period_hours': 8
        }
        
        # Configuration with custom thresholds
        self.custom_config = {
            'signal_threshold': 0.8,
            'exit_threshold': 0.2,
            'max_holding_period_hours': 6
        }
    
    def tearDown(self):
        """Clean up after each test method."""
        # Re-enable logging
        logging.disable(logging.NOTSET)
    
    def test_init_with_default_config(self):
        """Test strategy initialization with default configuration."""
        strategy = SupervisedNNStrategy(self.default_config)
        self.assertEqual(strategy.config, self.default_config)
        self.assertIsNotNone(strategy.logger)
    
    def test_init_with_empty_config(self):
        """Test strategy initialization with empty configuration."""
        strategy = SupervisedNNStrategy({})
        self.assertEqual(strategy.config, {})
    
    def test_buy_signal_from_flat_above_threshold(self):
        """Test BUY signal generation when FLAT and probability above threshold."""
        strategy = SupervisedNNStrategy(self.default_config)
        
        # Test with probability exactly at threshold
        result = strategy.generate_trade_action(0.7, "FLAT", 0)
        self.assertEqual(result, "BUY")
        
        # Test with probability above threshold
        result = strategy.generate_trade_action(0.85, "FLAT", 0)
        self.assertEqual(result, "BUY")
        
        # Test with very high probability
        result = strategy.generate_trade_action(0.99, "FLAT", 0)
        self.assertEqual(result, "BUY")
    
    def test_hold_signal_from_flat_below_threshold(self):
        """Test HOLD signal generation when FLAT and probability below threshold."""
        strategy = SupervisedNNStrategy(self.default_config)
        
        # Test with probability just below threshold
        result = strategy.generate_trade_action(0.69, "FLAT", 0)
        self.assertEqual(result, "HOLD")
        
        # Test with low probability
        result = strategy.generate_trade_action(0.3, "FLAT", 0)
        self.assertEqual(result, "HOLD")
        
        # Test with very low probability
        result = strategy.generate_trade_action(0.01, "FLAT", 0)
        self.assertEqual(result, "HOLD")
    
    def test_sell_signal_from_long_max_holding_period(self):
        """Test SELL signal generation when LONG and max holding period reached."""
        strategy = SupervisedNNStrategy(self.default_config)
        
        # Test with time exactly at max holding period
        result = strategy.generate_trade_action(0.8, "LONG", 8)
        self.assertEqual(result, "SELL")
        
        # Test with time exceeding max holding period
        result = strategy.generate_trade_action(0.9, "LONG", 10)
        self.assertEqual(result, "SELL")
        
        # Test with very high probability but max time reached
        result = strategy.generate_trade_action(0.99, "LONG", 8)
        self.assertEqual(result, "SELL")
    
    def test_sell_signal_from_long_below_exit_threshold(self):
        """Test SELL signal generation when LONG and probability below exit threshold."""
        strategy = SupervisedNNStrategy(self.default_config)
        
        # Test with probability exactly at exit threshold
        result = strategy.generate_trade_action(0.3, "LONG", 2)
        self.assertEqual(result, "SELL")
        
        # Test with probability below exit threshold
        result = strategy.generate_trade_action(0.2, "LONG", 4)
        self.assertEqual(result, "SELL")
        
        # Test with very low probability
        result = strategy.generate_trade_action(0.05, "LONG", 1)
        self.assertEqual(result, "SELL")
    
    def test_hold_signal_from_long_conditions_not_met(self):
        """Test HOLD signal generation when LONG and no exit conditions met."""
        strategy = SupervisedNNStrategy(self.default_config)
        
        # Test with probability above exit threshold and time below max
        result = strategy.generate_trade_action(0.6, "LONG", 4)
        self.assertEqual(result, "HOLD")
        
        # Test with high probability and short time
        result = strategy.generate_trade_action(0.85, "LONG", 2)
        self.assertEqual(result, "HOLD")
        
        # Test with probability just above exit threshold
        result = strategy.generate_trade_action(0.31, "LONG", 7)
        self.assertEqual(result, "HOLD")
    
    def test_no_exit_threshold_configured(self):
        """Test behavior when exit_threshold is not configured (None)."""
        strategy = SupervisedNNStrategy(self.config_no_exit_threshold)
        
        # Should only exit due to max holding period, not probability
        result = strategy.generate_trade_action(0.1, "LONG", 4)
        self.assertEqual(result, "HOLD")
        
        # Should still exit when max holding period reached
        result = strategy.generate_trade_action(0.1, "LONG", 8)
        self.assertEqual(result, "SELL")
        
        # Should still enter when probability above signal threshold
        result = strategy.generate_trade_action(0.8, "FLAT", 0)
        self.assertEqual(result, "BUY")
    
    def test_custom_thresholds(self):
        """Test strategy with custom threshold configurations."""
        strategy = SupervisedNNStrategy(self.custom_config)
        
        # Test custom signal threshold (0.8)
        result = strategy.generate_trade_action(0.75, "FLAT", 0)
        self.assertEqual(result, "HOLD")  # Below custom threshold
        
        result = strategy.generate_trade_action(0.85, "FLAT", 0)
        self.assertEqual(result, "BUY")  # Above custom threshold
        
        # Test custom exit threshold (0.2)
        result = strategy.generate_trade_action(0.25, "LONG", 2)
        self.assertEqual(result, "HOLD")  # Above custom exit threshold
        
        result = strategy.generate_trade_action(0.15, "LONG", 2)
        self.assertEqual(result, "SELL")  # Below custom exit threshold
        
        # Test custom max holding period (6 hours)
        result = strategy.generate_trade_action(0.9, "LONG", 5)
        self.assertEqual(result, "HOLD")  # Below custom max period
        
        result = strategy.generate_trade_action(0.9, "LONG", 6)
        self.assertEqual(result, "SELL")  # At custom max period
    
    def test_default_parameter_values(self):
        """Test that default parameter values are used when not specified in config."""
        # Empty config should use defaults
        strategy = SupervisedNNStrategy({})
        
        # Test default signal_threshold (0.7)
        result = strategy.generate_trade_action(0.69, "FLAT", 0)
        self.assertEqual(result, "HOLD")
        
        result = strategy.generate_trade_action(0.7, "FLAT", 0)
        self.assertEqual(result, "BUY")
        
        # Test default max_holding_period_hours (8)
        result = strategy.generate_trade_action(0.8, "LONG", 7)
        self.assertEqual(result, "HOLD")
        
        result = strategy.generate_trade_action(0.8, "LONG", 8)
        self.assertEqual(result, "SELL")
        
        # Test default exit_threshold (None - not used)
        result = strategy.generate_trade_action(0.1, "LONG", 4)
        self.assertEqual(result, "HOLD")
    
    def test_unknown_position_status(self):
        """Test behavior with unknown position status."""
        strategy = SupervisedNNStrategy(self.default_config)
        
        # Test with unknown position status
        result = strategy.generate_trade_action(0.8, "UNKNOWN", 0)
        self.assertEqual(result, "HOLD")
        
        result = strategy.generate_trade_action(0.8, "SHORT", 0)
        self.assertEqual(result, "HOLD")
        
        result = strategy.generate_trade_action(0.8, "", 0)
        self.assertEqual(result, "HOLD")
    
    def test_edge_case_probabilities(self):
        """Test behavior with edge case probability values."""
        strategy = SupervisedNNStrategy(self.default_config)
        
        # Test with probability 0.0
        result = strategy.generate_trade_action(0.0, "FLAT", 0)
        self.assertEqual(result, "HOLD")
        
        # Test with probability 1.0
        result = strategy.generate_trade_action(1.0, "FLAT", 0)
        self.assertEqual(result, "BUY")
        
        # Test with probability exactly at thresholds
        result = strategy.generate_trade_action(0.7, "FLAT", 0)
        self.assertEqual(result, "BUY")
        
        result = strategy.generate_trade_action(0.3, "LONG", 2)
        self.assertEqual(result, "SELL")
    
    def test_edge_case_time_values(self):
        """Test behavior with edge case time values."""
        strategy = SupervisedNNStrategy(self.default_config)
        
        # Test with time_in_position_hours = 0
        result = strategy.generate_trade_action(0.8, "LONG", 0)
        self.assertEqual(result, "HOLD")
        
        # Test with very large time value
        result = strategy.generate_trade_action(0.8, "LONG", 100)
        self.assertEqual(result, "SELL")
    
    def test_priority_of_exit_conditions(self):
        """Test that max holding period takes priority over exit threshold."""
        strategy = SupervisedNNStrategy(self.default_config)
        
        # When both conditions are met, should still return SELL
        # (max holding period is checked first)
        result = strategy.generate_trade_action(0.2, "LONG", 8)
        self.assertEqual(result, "SELL")
        
        # When only max holding period is met
        result = strategy.generate_trade_action(0.8, "LONG", 8)
        self.assertEqual(result, "SELL")
        
        # When only exit threshold condition is met
        result = strategy.generate_trade_action(0.2, "LONG", 4)
        self.assertEqual(result, "SELL")
    
    @patch('core.strategies.supervised_nn_strategy.logging.getLogger')
    def test_logging_calls(self, mock_get_logger):
        """Test that appropriate logging calls are made."""
        mock_logger = mock_get_logger.return_value
        strategy = SupervisedNNStrategy(self.default_config)
        
        # Test BUY signal logging
        strategy.generate_trade_action(0.8, "FLAT", 0)
        mock_logger.info.assert_called()
        
        # Test SELL signal logging
        strategy.generate_trade_action(0.8, "LONG", 8)
        mock_logger.info.assert_called()
        
        # Test debug logging
        strategy.generate_trade_action(0.5, "FLAT", 0)
        mock_logger.debug.assert_called()


class TestSupervisedNNStrategyIntegration(unittest.TestCase):
    """Integration tests for SupervisedNNStrategy class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up after each test method."""
        logging.disable(logging.NOTSET)
    
    def test_realistic_trading_scenario(self):
        """Test a realistic trading scenario with multiple state transitions."""
        config = {
            'signal_threshold': 0.75,
            'exit_threshold': 0.25,
            'max_holding_period_hours': 6
        }
        strategy = SupervisedNNStrategy(config)
        
        # Scenario: Start FLAT, get BUY signal, hold for a while, then exit
        
        # 1. Start FLAT with low probability - should HOLD
        result = strategy.generate_trade_action(0.5, "FLAT", 0)
        self.assertEqual(result, "HOLD")
        
        # 2. Still FLAT, probability rises above threshold - should BUY
        result = strategy.generate_trade_action(0.8, "FLAT", 0)
        self.assertEqual(result, "BUY")
        
        # 3. Now LONG, probability stays high, time is low - should HOLD
        result = strategy.generate_trade_action(0.8, "LONG", 1)
        self.assertEqual(result, "HOLD")
        
        # 4. Still LONG, probability drops but not below exit threshold - should HOLD
        result = strategy.generate_trade_action(0.4, "LONG", 3)
        self.assertEqual(result, "HOLD")
        
        # 5. Still LONG, probability drops below exit threshold - should SELL
        result = strategy.generate_trade_action(0.2, "LONG", 4)
        self.assertEqual(result, "SELL")
        
        # 6. Back to FLAT, low probability - should HOLD
        result = strategy.generate_trade_action(0.3, "FLAT", 0)
        self.assertEqual(result, "HOLD")
    
    def test_max_holding_period_scenario(self):
        """Test scenario where position is held until max holding period."""
        config = {
            'signal_threshold': 0.7,
            'exit_threshold': 0.3,
            'max_holding_period_hours': 4
        }
        strategy = SupervisedNNStrategy(config)
        
        # Enter position
        result = strategy.generate_trade_action(0.8, "FLAT", 0)
        self.assertEqual(result, "BUY")
        
        # Hold for increasing time periods with high probability
        for hours in range(1, 4):
            result = strategy.generate_trade_action(0.85, "LONG", hours)
            self.assertEqual(result, "HOLD")
        
        # At max holding period, should exit regardless of high probability
        result = strategy.generate_trade_action(0.9, "LONG", 4)
        self.assertEqual(result, "SELL")


# Define dummy model class at module level so it can be pickled
class DummyModel(nn.Module):
    """Simple dummy PyTorch model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1)  # 5 features -> 1 output
    
    def forward(self, x, asset_ids=None):
        # x shape: (batch_size, seq_len, features)
        # Take mean across sequence dimension
        x_mean = x.mean(dim=1)  # (batch_size, features)
        return self.linear(x_mean)  # (batch_size, 1)


class TestSupervisedNNStrategyModelInference(unittest.TestCase):
    """Test cases for model inference methods in SupervisedNNStrategy."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        logging.disable(logging.CRITICAL)
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create dummy model
        self.dummy_model = DummyModel()
        self.model_path = os.path.join(self.temp_dir, 'dummy_model.pt')
        torch.save(self.dummy_model, self.model_path)
        
        # Create dummy scaler
        self.dummy_scaler = StandardScaler()
        # Fit on dummy data
        dummy_data = np.random.randn(100, 5)
        self.dummy_scaler.fit(dummy_data)
        self.scaler_path = os.path.join(self.temp_dir, 'dummy_scaler.pkl')
        joblib.dump(self.dummy_scaler, self.scaler_path)
        
        # Create dummy asset ID mapping
        self.asset_id_map = {
            'symbol_to_id': {'AAPL': 0, 'MSFT': 1, 'TSLA': 2},
            'id_to_symbol': {'0': 'AAPL', '1': 'MSFT', '2': 'TSLA'}
        }
        self.asset_id_map_path = os.path.join(self.temp_dir, 'asset_id_mapping.json')
        with open(self.asset_id_map_path, 'w') as f:
            json.dump(self.asset_id_map, f)
        
        # Configuration with model paths
        self.config_with_model = {
            'signal_threshold': 0.7,
            'max_holding_period_hours': 8,
            'model_path': self.model_path,
            'scaler_path': self.scaler_path,
            'asset_id_map_path': self.asset_id_map_path,
            'feature_list': ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
            'lookback_window': 10
        }
    
    def tearDown(self):
        """Clean up after each test method."""
        logging.disable(logging.NOTSET)
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    
    def _create_sample_historical_data(self, num_rows=20):
        """Create sample historical data for testing."""
        np.random.seed(42)  # For reproducible tests
        data = {
            'feature1': np.random.randn(num_rows),
            'feature2': np.random.randn(num_rows),
            'feature3': np.random.randn(num_rows),
            'feature4': np.random.randn(num_rows),
            'feature5': np.random.randn(num_rows),
            'extra_feature': np.random.randn(num_rows)  # Not in feature_list
        }
        return pd.DataFrame(data)
    
    def test_load_dependencies_success(self):
        """Test successful loading of dependencies."""
        # Create strategy without auto-loading dependencies
        config_no_auto_load = self.config_with_model.copy()
        del config_no_auto_load['model_path']  # Remove to prevent auto-loading
        del config_no_auto_load['scaler_path']
        
        strategy = SupervisedNNStrategy(config_no_auto_load)
        
        # Manually set the paths and call load_dependencies
        strategy.config['model_path'] = self.model_path
        strategy.config['scaler_path'] = self.scaler_path
        strategy.config['asset_id_map_path'] = self.asset_id_map_path
        
        strategy.load_dependencies()
        
        # Verify model and scaler are loaded
        self.assertIsNotNone(strategy.model)
        self.assertIsNotNone(strategy.scaler)
        self.assertIsNotNone(strategy.asset_id_map)
    
    def test_load_dependencies_missing_model_path(self):
        """Test error handling when model_path is missing."""
        config_no_model = self.config_with_model.copy()
        del config_no_model['model_path']
        
        strategy = SupervisedNNStrategy(config_no_model)
        
        # Test actual method
        with self.assertRaises(ValueError) as context:
            strategy.load_dependencies()
        
        self.assertIn("model_path must be specified", str(context.exception))
    
    def test_load_dependencies_missing_scaler_path(self):
        """Test error handling when scaler_path is missing."""
        config_no_scaler = self.config_with_model.copy()
        del config_no_scaler['scaler_path']
        
        strategy = SupervisedNNStrategy(config_no_scaler)
        
        # Test actual method
        with self.assertRaises(ValueError) as context:
            strategy.load_dependencies()
        
        self.assertIn("scaler_path must be specified", str(context.exception))
    
    @patch('core.strategies.supervised_nn_strategy.SupervisedNNStrategy.load_dependencies')
    def test_prepare_input_sequence_success(self, mock_load_deps):
        """Test successful input sequence preparation."""
        mock_load_deps.return_value = None
        
        strategy = SupervisedNNStrategy(self.config_with_model)
        strategy.scaler = self.dummy_scaler
        strategy.asset_id_map = self.asset_id_map['symbol_to_id']
        
        historical_data = self._create_sample_historical_data(20)
        
        feature_tensor, asset_id_tensor = strategy.prepare_input_sequence(
            historical_data, 'AAPL'
        )
        
        # Verify output shapes and types
        self.assertIsNotNone(feature_tensor)
        self.assertIsNotNone(asset_id_tensor)
        self.assertEqual(feature_tensor.shape, (1, 10, 5))  # (batch, seq_len, features)
        self.assertEqual(asset_id_tensor.shape, (1,))
        self.assertEqual(asset_id_tensor.item(), 0)  # AAPL -> 0
    
    @patch('core.strategies.supervised_nn_strategy.SupervisedNNStrategy.load_dependencies')
    def test_prepare_input_sequence_insufficient_data(self, mock_load_deps):
        """Test input sequence preparation with insufficient data."""
        mock_load_deps.return_value = None
        
        strategy = SupervisedNNStrategy(self.config_with_model)
        strategy.scaler = self.dummy_scaler
        
        # Create data with fewer rows than lookback_window
        historical_data = self._create_sample_historical_data(5)  # < 10 required
        
        feature_tensor, asset_id_tensor = strategy.prepare_input_sequence(
            historical_data, 'AAPL'
        )
        
        # Should return None for both
        self.assertIsNone(feature_tensor)
        self.assertIsNone(asset_id_tensor)
    
    @patch('core.strategies.supervised_nn_strategy.SupervisedNNStrategy.load_dependencies')
    def test_prepare_input_sequence_missing_features(self, mock_load_deps):
        """Test input sequence preparation with missing features."""
        mock_load_deps.return_value = None
        
        config_missing_features = self.config_with_model.copy()
        config_missing_features['feature_list'] = ['feature1', 'missing_feature']
        
        strategy = SupervisedNNStrategy(config_missing_features)
        strategy.scaler = self.dummy_scaler
        
        historical_data = self._create_sample_historical_data(20)
        
        feature_tensor, asset_id_tensor = strategy.prepare_input_sequence(
            historical_data, 'AAPL'
        )
        
        # Should return None due to missing features
        self.assertIsNone(feature_tensor)
        self.assertIsNone(asset_id_tensor)
    
    @patch('core.strategies.supervised_nn_strategy.SupervisedNNStrategy.load_dependencies')
    def test_get_model_prediction_success(self, mock_load_deps):
        """Test successful model prediction."""
        mock_load_deps.return_value = None
        
        strategy = SupervisedNNStrategy(self.config_with_model)
        strategy.model = self.dummy_model
        strategy.model.eval()
        
        # Create dummy input tensors
        feature_tensor = torch.randn(1, 10, 5)
        asset_id_tensor = torch.tensor([0])
        
        prediction = strategy.get_model_prediction(feature_tensor, asset_id_tensor)
        
        # Verify prediction is a float between 0 and 1
        self.assertIsInstance(prediction, float)
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)
    
    @patch('core.strategies.supervised_nn_strategy.SupervisedNNStrategy.load_dependencies')
    def test_get_model_prediction_without_asset_id(self, mock_load_deps):
        """Test model prediction without asset ID."""
        mock_load_deps.return_value = None
        
        strategy = SupervisedNNStrategy(self.config_with_model)
        strategy.model = self.dummy_model
        strategy.model.eval()
        
        # Create dummy input tensor without asset ID
        feature_tensor = torch.randn(1, 10, 5)
        
        prediction = strategy.get_model_prediction(feature_tensor, None)
        
        # Verify prediction is valid
        self.assertIsInstance(prediction, float)
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)
    
    @patch('core.strategies.supervised_nn_strategy.SupervisedNNStrategy.load_dependencies')
    def test_on_bar_data_success(self, mock_load_deps):
        """Test successful on_bar_data processing."""
        mock_load_deps.return_value = None
        
        strategy = SupervisedNNStrategy(self.config_with_model)
        strategy.model = self.dummy_model
        strategy.scaler = self.dummy_scaler
        strategy.asset_id_map = self.asset_id_map['symbol_to_id']
        
        # Create test inputs
        bar_data = {'symbol': 'AAPL'}
        historical_window_df = self._create_sample_historical_data(20)
        current_portfolio_status = {
            'position_type': 'FLAT',
            'time_in_position_hours': 0
        }
        
        action = strategy.on_bar_data(
            bar_data, historical_window_df, current_portfolio_status
        )
        
        # Verify action is valid
        self.assertIn(action, ['BUY', 'SELL', 'HOLD'])
    
    @patch('core.strategies.supervised_nn_strategy.SupervisedNNStrategy.load_dependencies')
    def test_on_bar_data_insufficient_data(self, mock_load_deps):
        """Test on_bar_data with insufficient historical data."""
        mock_load_deps.return_value = None
        
        strategy = SupervisedNNStrategy(self.config_with_model)
        strategy.model = self.dummy_model
        strategy.scaler = self.dummy_scaler
        
        # Create test inputs with insufficient data
        bar_data = {'symbol': 'AAPL'}
        historical_window_df = self._create_sample_historical_data(5)  # < 10 required
        current_portfolio_status = {
            'position_type': 'FLAT',
            'time_in_position_hours': 0
        }
        
        action = strategy.on_bar_data(
            bar_data, historical_window_df, current_portfolio_status
        )
        
        # Should return HOLD due to insufficient data
        self.assertEqual(action, 'HOLD')
    
    @patch('core.strategies.supervised_nn_strategy.SupervisedNNStrategy.load_dependencies')
    def test_on_bar_data_missing_symbol(self, mock_load_deps):
        """Test on_bar_data with missing symbol in bar_data."""
        mock_load_deps.return_value = None
        
        strategy = SupervisedNNStrategy(self.config_with_model)
        
        # Create test inputs without symbol
        bar_data = {}  # No symbol
        historical_window_df = self._create_sample_historical_data(20)
        current_portfolio_status = {
            'position_type': 'FLAT',
            'time_in_position_hours': 0
        }
        
        action = strategy.on_bar_data(
            bar_data, historical_window_df, current_portfolio_status
        )
        
        # Should return HOLD due to missing symbol
        self.assertEqual(action, 'HOLD')


if __name__ == '__main__':
    # Configure logging for test execution
    logging.basicConfig(level=logging.DEBUG)
    
    # Run the tests
    unittest.main(verbosity=2)