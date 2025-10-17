"""
Unit Tests for Neural Network Data Preparation Module

This module contains comprehensive unit tests for the NNDataPreparer class
in core/data_preparation_nn.py. Tests cover all key methods including:
- Initialization and configuration loading
- Asset ID mapping creation and loading
- Data loading for single/multiple symbols
- Feature selection and preprocessing
- Label generation logic
- Sequence generation
- Data splitting
- Feature scaling
- Sample weight calculation
- Overall output structure and content

Created: 2025-05-23
Phase: 2 - Task 2.10 - Unit Testing and Documentation Finalization
"""

import unittest
import tempfile
import shutil
import os
import json
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler

# Import the class under test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.data_preparation_nn import NNDataPreparer


class TestNNDataPreparer(unittest.TestCase):
    """Test suite for NNDataPreparer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.test_config = {
            'symbols_config_path': 'config/symbols.json',
            'feature_list': [
                'SMA_10', 'SMA_20', 'MACD', 'RSI_14', 'sentiment_score_hourly_ffill',
                'DayOfWeek_sin', 'DayOfWeek_cos'
            ],
            'nan_handling_features': 'ffill',
            'lookback_window': 24,
            'prediction_horizon': 8,
            'profit_target': 0.05,
            'stop_loss_target': 0.02,
            'data_base_path': self.temp_dir,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'shuffle_before_split': False,
            'scaling_method': 'standard',
            'calculate_sample_weights': True,
            'sample_weight_strategy': 'inverse_frequency',
            'output_path_scalers': os.path.join(self.temp_dir, 'scalers.joblib'),
            'symbols_list': ['AAPL', 'MSFT']
        }
        
        # Create test symbols.json content
        self.test_symbols_config = {
            "sectors": {
                "technology": ["AAPL", "MSFT", "GOOGL"],
                "finance": ["JPM", "BAC"]
            },
            "etfs": {
                "broad_market": ["SPY", "QQQ"]
            },
            "crypto": {
                "major": ["BTC-USD", "ETH-USD"]
            }
        }
        
        # Create test data
        self.create_test_data()
    
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_data(self):
        """Create test data files for testing."""
        # Create test symbols.json
        symbols_path = os.path.join(self.temp_dir, 'config', 'symbols.json')
        os.makedirs(os.path.dirname(symbols_path), exist_ok=True)
        with open(symbols_path, 'w') as f:
            json.dump(self.test_symbols_config, f)
        
        # Create test historical data for AAPL and MSFT
        for symbol in ['AAPL', 'MSFT']:
            data_dir = os.path.join(self.temp_dir, 'data', 'historical', symbol, '1Hour')
            os.makedirs(data_dir, exist_ok=True)
            
            # Create test DataFrame with required columns
            dates = pd.date_range('2024-01-01', periods=100, freq='h')
            np.random.seed(42)  # For reproducible tests
            
            test_data = pd.DataFrame({
                'Open': 150 + np.random.randn(100) * 5,
                'High': 155 + np.random.randn(100) * 5,
                'Low': 145 + np.random.randn(100) * 5,
                'Close': 150 + np.random.randn(100) * 5,
                'Volume': 1000000 + np.random.randint(-100000, 100000, 100),
                'VWAP': 150 + np.random.randn(100) * 3,
                'SMA_10': 150 + np.random.randn(100) * 2,
                'SMA_20': 150 + np.random.randn(100) * 2,
                'MACD': np.random.randn(100) * 0.5,
                'MACD_signal': np.random.randn(100) * 0.3,
                'MACD_histogram': np.random.randn(100) * 0.2,
                'RSI_14': 30 + np.random.randn(100) * 20,
                'Stoch_K': 50 + np.random.randn(100) * 20,
                'Stoch_D': 50 + np.random.randn(100) * 20,
                'ADX_14': 25 + np.random.randn(100) * 10,
                'ATR_14': 2 + np.random.randn(100) * 0.5,
                'BB_bandwidth': 0.1 + np.random.randn(100) * 0.02,
                'OBV': np.cumsum(np.random.randn(100) * 1000),
                'Volume_SMA_20': 1000000 + np.random.randint(-50000, 50000, 100),
                'Returns_1h': np.random.randn(100) * 0.01,
                'sentiment_score_hourly_ffill': np.random.randn(100) * 0.3,
                'DayOfWeek_sin': np.sin(2 * np.pi * dates.dayofweek / 7),
                'DayOfWeek_cos': np.cos(2 * np.pi * dates.dayofweek / 7)
            }, index=dates)
            
            # Ensure High >= Close >= Low for realistic price data
            test_data['High'] = np.maximum(test_data['High'], test_data['Close'])
            test_data['Low'] = np.minimum(test_data['Low'], test_data['Close'])
            
            # Save as parquet
            data_path = os.path.join(data_dir, 'data.parquet')
            test_data.to_parquet(data_path)
    
    @patch('core.data_preparation_nn.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_initialization_with_existing_mapping(self, mock_file, mock_exists):
        """Test initialization when asset ID mapping file already exists."""
        # Mock existing mapping file
        mock_exists.return_value = True
        mock_mapping = {
            'symbol_to_id': {'AAPL': 0, 'MSFT': 1, 'GOOGL': 2}
        }
        mock_file.return_value.read.return_value = json.dumps(mock_mapping)
        
        with patch('json.load', return_value=mock_mapping):
            preparer = NNDataPreparer(self.test_config)
            
            self.assertIsNotNone(preparer.asset_id_map)
            self.assertEqual(preparer.asset_id_map['AAPL'], 0)
            self.assertEqual(preparer.asset_id_map['MSFT'], 1)
    
    def test_initialization_creates_new_mapping(self):
        """Test initialization when asset ID mapping file doesn't exist."""
        # Update config to point to our test symbols.json
        config = self.test_config.copy()
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        
        preparer = NNDataPreparer(config)
        
        # Check that asset_id_map was created
        self.assertIsNotNone(preparer.asset_id_map)
        self.assertIn('AAPL', preparer.asset_id_map)
        self.assertIn('MSFT', preparer.asset_id_map)
        self.assertIn('GOOGL', preparer.asset_id_map)
        
        # Check that mapping file was created
        mapping_path = "config/asset_id_mapping.json"
        self.assertTrue(os.path.exists(mapping_path))
    
    def test_create_asset_id_mapping(self):
        """Test asset ID mapping creation from symbols.json."""
        symbols_path = os.path.join(self.temp_dir, 'config', 'symbols.json')
        mapping_path = os.path.join(self.temp_dir, 'asset_mapping.json')
        
        # Create a minimal preparer instance
        preparer = NNDataPreparer({'symbols_config_path': symbols_path})
        preparer._create_asset_id_mapping(symbols_path, mapping_path)
        
        # Verify mapping file was created
        self.assertTrue(os.path.exists(mapping_path))
        
        # Load and verify mapping content
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        self.assertIn('symbol_to_id', mapping_data)
        self.assertIn('id_to_symbol', mapping_data)
        self.assertIn('metadata', mapping_data)
        
        # Check that all symbols from test config are included
        expected_symbols = {'AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC', 'SPY', 'QQQ', 'BTC-USD', 'ETH-USD'}
        actual_symbols = set(mapping_data['symbol_to_id'].keys())
        self.assertEqual(expected_symbols, actual_symbols)
    
    def test_load_data_for_symbol(self):
        """Test loading data for a single symbol."""
        config = self.test_config.copy()
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        config['data_base_path'] = os.path.join(self.temp_dir, 'data')
        
        preparer = NNDataPreparer(config)
        
        # Test loading AAPL data
        aapl_data = preparer.load_data_for_symbol('AAPL')
        
        self.assertIsInstance(aapl_data, pd.DataFrame)
        self.assertFalse(aapl_data.empty)
        self.assertIsInstance(aapl_data.index, pd.DatetimeIndex)
        self.assertIn('Close', aapl_data.columns)
        self.assertIn('High', aapl_data.columns)
        self.assertIn('Low', aapl_data.columns)
        
        # Test caching - second call should return cached data
        aapl_data_cached = preparer.load_data_for_symbol('AAPL')
        pd.testing.assert_frame_equal(aapl_data, aapl_data_cached)
    
    def test_load_data_for_nonexistent_symbol(self):
        """Test loading data for a symbol that doesn't exist."""
        config = self.test_config.copy()
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        config['data_base_path'] = os.path.join(self.temp_dir, 'data')
        
        preparer = NNDataPreparer(config)
        
        with self.assertRaises(FileNotFoundError):
            preparer.load_data_for_symbol('NONEXISTENT')
    
    def test_select_features(self):
        """Test feature selection functionality."""
        config = self.test_config.copy()
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        config['data_base_path'] = os.path.join(self.temp_dir, 'data')
        
        preparer = NNDataPreparer(config)
        
        # Load test data
        aapl_data = preparer.load_data_for_symbol('AAPL')
        
        # Test feature selection
        selected_features = preparer._select_features(aapl_data)
        
        expected_features = self.test_config['feature_list']
        self.assertEqual(list(selected_features.columns), expected_features)
        self.assertEqual(len(selected_features.columns), len(expected_features))
    
    def test_select_features_with_missing_features(self):
        """Test feature selection when some features are missing."""
        config = self.test_config.copy()
        config['feature_list'] = ['SMA_10', 'MISSING_FEATURE', 'RSI_14']
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        config['data_base_path'] = os.path.join(self.temp_dir, 'data')
        
        preparer = NNDataPreparer(config)
        aapl_data = preparer.load_data_for_symbol('AAPL')
        
        selected_features = preparer._select_features(aapl_data)
        
        # Should only include available features
        expected_features = ['SMA_10', 'RSI_14']
        self.assertEqual(list(selected_features.columns), expected_features)
    
    def test_generate_day_of_week_features(self):
        """Test day of week feature generation."""
        config = self.test_config.copy()
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        
        preparer = NNDataPreparer(config)
        
        # Create test DataFrame without day of week features
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        test_df = pd.DataFrame({'price': range(10)}, index=dates)
        
        result_df = preparer._generate_day_of_week_features(test_df)
        
        self.assertIn('DayOfWeek_sin', result_df.columns)
        self.assertIn('DayOfWeek_cos', result_df.columns)
        
        # Check that values are in expected range [-1, 1]
        self.assertTrue((result_df['DayOfWeek_sin'] >= -1).all())
        self.assertTrue((result_df['DayOfWeek_sin'] <= 1).all())
        self.assertTrue((result_df['DayOfWeek_cos'] >= -1).all())
        self.assertTrue((result_df['DayOfWeek_cos'] <= 1).all())
    
    def test_generate_labels_for_symbol(self):
        """Test label generation logic with various price movement scenarios."""
        config = self.test_config.copy()
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        config['data_base_path'] = os.path.join(self.temp_dir, 'data')
        
        preparer = NNDataPreparer(config)
        
        # Create test data with known price movements
        dates = pd.date_range('2024-01-01', periods=20, freq='h')
        
        # Scenario 1: Price goes up 6% (should be label 1)
        test_data = pd.DataFrame({
            'Close': [100] * 4 + [106] * 16,  # Price jumps to 106 within prediction horizon
            'High': [100] * 4 + [106] * 16,
            'Low': [100] * 4 + [106] * 16,
            'Volume': [1000] * 20
        }, index=dates)
        
        result_df = preparer._generate_labels_for_symbol(test_data, 'TEST')
        
        # First few rows should have label 1 (profit target hit)
        self.assertEqual(result_df.iloc[0]['label'], 1)
        
        # Test scenario 2: Price goes down 3% (should be label 0)
        test_data_down = pd.DataFrame({
            'Close': [100] * 4 + [97] * 16,  # Price drops within prediction horizon
            'High': [100] * 4 + [97] * 16,
            'Low': [100] * 4 + [97] * 16,
            'Volume': [1000] * 20
        }, index=dates)
        
        result_df_down = preparer._generate_labels_for_symbol(test_data_down, 'TEST')
        self.assertEqual(result_df_down.iloc[0]['label'], 0)
    
    def test_generate_sequences_for_symbol(self):
        """Test sequence generation with correct shapes and alignment."""
        config = self.test_config.copy()
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        config['data_base_path'] = os.path.join(self.temp_dir, 'data')
        
        preparer = NNDataPreparer(config)
        
        # Create test features and labels
        dates = pd.date_range('2024-01-01', periods=50, freq='h')
        features_df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'feature3': np.random.randn(50)
        }, index=dates)
        
        labels_df = pd.DataFrame({
            'label': np.random.randint(0, 2, 50)
        }, index=dates)
        
        asset_id = 0
        X, y, asset_ids = preparer._generate_sequences_for_symbol(features_df, labels_df, asset_id)
        
        # Check shapes
        lookback_window = config['lookback_window']
        expected_sequences = len(dates) - lookback_window + 1
        
        self.assertEqual(X.shape, (expected_sequences, lookback_window, 3))
        self.assertEqual(y.shape, (expected_sequences,))
        self.assertEqual(asset_ids.shape, (expected_sequences,))
        
        # Check that all asset_ids are correct
        self.assertTrue((asset_ids == asset_id).all())
        
        # Check data types
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.int32)
        self.assertEqual(asset_ids.dtype, np.int32)
    
    def test_generate_sequences_insufficient_data(self):
        """Test sequence generation with insufficient data."""
        config = self.test_config.copy()
        config['lookback_window'] = 50  # More than available data
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        
        preparer = NNDataPreparer(config)
        
        # Create small dataset
        dates = pd.date_range('2024-01-01', periods=10, freq='h')
        features_df = pd.DataFrame({'feature1': range(10)}, index=dates)
        labels_df = pd.DataFrame({'label': [0] * 10}, index=dates)
        
        X, y, asset_ids = preparer._generate_sequences_for_symbol(features_df, labels_df, 0)
        
        # Should return empty arrays
        self.assertEqual(len(X), 0)
        self.assertEqual(len(y), 0)
        self.assertEqual(len(asset_ids), 0)
    
    def test_split_data(self):
        """Test data splitting respecting temporal order and ratios."""
        config = self.test_config.copy()
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        
        preparer = NNDataPreparer(config)
        
        # Create test data
        n_samples = 1000
        X_all = np.random.randn(n_samples, 24, 5).astype(np.float32)
        y_all = np.random.randint(0, 2, n_samples).astype(np.int32)
        asset_ids_all = np.random.randint(0, 5, n_samples).astype(np.int32)
        
        splits = preparer._split_data(X_all, y_all, asset_ids_all)
        
        # Check that all splits are present
        self.assertIn('train', splits)
        self.assertIn('val', splits)
        self.assertIn('test', splits)
        
        # Check split sizes approximately match ratios
        train_size = len(splits['train']['X'])
        val_size = len(splits['val']['X'])
        test_size = len(splits['test']['X'])
        
        self.assertAlmostEqual(train_size / n_samples, 0.7, delta=0.05)
        self.assertAlmostEqual(val_size / n_samples, 0.15, delta=0.05)
        self.assertAlmostEqual(test_size / n_samples, 0.15, delta=0.05)
        
        # Check that total samples are preserved
        self.assertEqual(train_size + val_size + test_size, n_samples)
        
        # Check shapes are consistent
        for split_name in ['train', 'val', 'test']:
            split_data = splits[split_name]
            self.assertEqual(len(split_data['X']), len(split_data['y']))
            self.assertEqual(len(split_data['y']), len(split_data['asset_ids']))
    
    def test_apply_scaling(self):
        """Test feature scaling including fitting on train only and correct transformation."""
        config = self.test_config.copy()
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        
        preparer = NNDataPreparer(config)
        
        # Create test data with different scales
        np.random.seed(42)
        X_train = np.random.randn(100, 24, 3).astype(np.float32) * 10 + 50
        X_val = np.random.randn(20, 24, 3).astype(np.float32) * 10 + 50
        X_test = np.random.randn(20, 24, 3).astype(np.float32) * 10 + 50
        
        X_train_scaled, X_val_scaled, X_test_scaled = preparer._apply_scaling(X_train, X_val, X_test)
        
        # Check that shapes are preserved
        self.assertEqual(X_train_scaled.shape, X_train.shape)
        self.assertEqual(X_val_scaled.shape, X_val.shape)
        self.assertEqual(X_test_scaled.shape, X_test.shape)
        
        # Check that training data is approximately standardized
        train_mean = X_train_scaled.mean()
        train_std = X_train_scaled.std()
        self.assertAlmostEqual(train_mean, 0.0, delta=0.1)
        self.assertAlmostEqual(train_std, 1.0, delta=0.1)
        
        # Check that scalers were stored
        self.assertIsNotNone(preparer.scalers)
        self.assertIn('global', preparer.scalers)
    
    def test_apply_scaling_robust(self):
        """Test robust scaling method."""
        config = self.test_config.copy()
        config['scaling_method'] = 'robust'
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        
        preparer = NNDataPreparer(config)
        
        X_train = np.random.randn(100, 24, 3).astype(np.float32)
        X_val = np.random.randn(20, 24, 3).astype(np.float32)
        X_test = np.random.randn(20, 24, 3).astype(np.float32)
        
        X_train_scaled, X_val_scaled, X_test_scaled = preparer._apply_scaling(X_train, X_val, X_test)
        
        # Check that RobustScaler was used
        self.assertIsInstance(preparer.scalers['global'], RobustScaler)
    
    def test_calculate_sample_weights(self):
        """Test sample weight calculation for class imbalance handling."""
        config = self.test_config.copy()
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        
        preparer = NNDataPreparer(config)
        
        # Create imbalanced dataset (90% class 0, 10% class 1)
        y_train = np.array([0] * 900 + [1] * 100)
        
        sample_weights = preparer._calculate_sample_weights(y_train)
        
        self.assertIsNotNone(sample_weights)
        self.assertEqual(len(sample_weights), len(y_train))
        
        # Check that minority class has higher weights
        class_0_weight = sample_weights[y_train == 0][0]
        class_1_weight = sample_weights[y_train == 1][0]
        self.assertGreater(class_1_weight, class_0_weight)
    
    def test_calculate_sample_weights_disabled(self):
        """Test sample weight calculation when disabled."""
        config = self.test_config.copy()
        config['calculate_sample_weights'] = False
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        
        preparer = NNDataPreparer(config)
        
        y_train = np.array([0] * 90 + [1] * 10)
        sample_weights = preparer._calculate_sample_weights(y_train)
        
        self.assertIsNone(sample_weights)
    
    def test_save_and_load_scalers(self):
        """Test scaler saving and loading functionality."""
        config = self.test_config.copy()
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        
        preparer = NNDataPreparer(config)
        
        # Create and fit a scaler
        scaler = StandardScaler()
        test_data = np.random.randn(100, 5)
        scaler.fit(test_data)
        preparer.scalers = {'test_scaler': scaler}
        
        # Test saving
        scaler_path = os.path.join(self.temp_dir, 'test_scalers.joblib')
        preparer.save_scalers(scaler_path)
        self.assertTrue(os.path.exists(scaler_path))
        
        # Test loading
        new_preparer = NNDataPreparer(config)
        new_preparer.load_scalers(scaler_path)
        
        self.assertIn('test_scaler', new_preparer.scalers)
        self.assertIsInstance(new_preparer.scalers['test_scaler'], StandardScaler)
    
    def test_load_scalers_file_not_found(self):
        """Test loading scalers when file doesn't exist."""
        config = self.test_config.copy()
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        
        preparer = NNDataPreparer(config)
        
        with self.assertRaises(FileNotFoundError):
            preparer.load_scalers('nonexistent_file.joblib')
    
    def test_aggregate_data_from_symbols(self):
        """Test data aggregation from multiple symbols."""
        config = self.test_config.copy()
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        config['data_base_path'] = os.path.join(self.temp_dir, 'data')
        
        preparer = NNDataPreparer(config)
        
        symbols = ['AAPL', 'MSFT']
        X_parts, y_parts, asset_id_parts = preparer._aggregate_data_from_symbols(symbols)
        
        # Should have data for both symbols
        self.assertEqual(len(X_parts), 2)
        self.assertEqual(len(y_parts), 2)
        self.assertEqual(len(asset_id_parts), 2)
        
        # Check that each part has correct shapes
        for i, symbol in enumerate(symbols):
            self.assertGreater(len(X_parts[i]), 0)
            self.assertEqual(len(X_parts[i]), len(y_parts[i]))
            self.assertEqual(len(y_parts[i]), len(asset_id_parts[i]))
            
            # Check that asset IDs are consistent
            expected_asset_id = preparer.asset_id_map[symbol]
            self.assertTrue((asset_id_parts[i] == expected_asset_id).all())
    
    def test_get_prepared_data_for_training_full_pipeline(self):
        """Test the complete data preparation pipeline."""
        config = self.test_config.copy()
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        config['data_base_path'] = os.path.join(self.temp_dir, 'data')
        
        preparer = NNDataPreparer(config)
        
        # Run full pipeline
        prepared_data = preparer.get_prepared_data_for_training()
        
        # Check main structure
        self.assertIn('train', prepared_data)
        self.assertIn('val', prepared_data)
        self.assertIn('test', prepared_data)
        self.assertIn('scalers', prepared_data)
        self.assertIn('asset_id_map', prepared_data)
        
        # Check train data structure
        train_data = prepared_data['train']
        self.assertIn('X', train_data)
        self.assertIn('y', train_data)
        self.assertIn('asset_ids', train_data)
        self.assertIn('sample_weights', train_data)  # Should be included based on config
        
        # Check shapes are consistent
        for split_name in ['train', 'val', 'test']:
            split_data = prepared_data[split_name]
            X_shape = split_data['X'].shape
            y_shape = split_data['y'].shape
            asset_ids_shape = split_data['asset_ids'].shape
            
            # Check 3D shape for X
            self.assertEqual(len(X_shape), 3)
            self.assertEqual(X_shape[1], config['lookback_window'])
            self.assertEqual(X_shape[2], len(config['feature_list']))
            
            # Check 1D shapes for y and asset_ids
            self.assertEqual(len(y_shape), 1)
            self.assertEqual(len(asset_ids_shape), 1)
            
            # Check consistent sample counts
            self.assertEqual(X_shape[0], y_shape[0])
            self.assertEqual(y_shape[0], asset_ids_shape[0])
        
        # Check that scalers were saved
        scaler_path = config['output_path_scalers']
        self.assertTrue(os.path.exists(scaler_path))
        
        # Check asset_id_map
        self.assertIsInstance(prepared_data['asset_id_map'], dict)
        self.assertIn('AAPL', prepared_data['asset_id_map'])
        self.assertIn('MSFT', prepared_data['asset_id_map'])
    
    def test_get_prepared_data_no_symbols_list(self):
        """Test data preparation when no symbols_list is provided in config."""
        config = self.test_config.copy()
        del config['symbols_list']  # Remove symbols_list
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        config['data_base_path'] = os.path.join(self.temp_dir, 'data')
        
        preparer = NNDataPreparer(config)
        
        # Should use all symbols from asset_id_map, but only AAPL and MSFT have data
        prepared_data = preparer.get_prepared_data_for_training()
        
        # Should still work with available symbols
        self.assertIn('train', prepared_data)
        self.assertGreater(len(prepared_data['train']['X']), 0)
    
    def test_error_handling_invalid_config(self):
        """Test error handling with invalid configuration."""
        # Test with missing required config
        invalid_config = {}
        
        with self.assertRaises(Exception):
            preparer = NNDataPreparer(invalid_config)
    
    def test_label_generation_edge_cases(self):
        """Test label generation with edge cases."""
        config = self.test_config.copy()
        config['symbols_config_path'] = os.path.join(self.temp_dir, 'config', 'symbols.json')
        
        preparer = NNDataPreparer(config)
        
        # Test with insufficient future data
        dates = pd.date_range('2024-01-01', periods=5, freq='h')
        test_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'High': [101, 102, 103, 104, 105],
            'Low': [99, 100, 101, 102, 103],
            'Volume': [1000] * 5
        }, index=dates)
        
        result_df = preparer._generate_labels_for_symbol(test_data, 'TEST')
        
        # Should have fewer rows due to insufficient future data for some rows
        self.assertLessEqual(len(result_df), len(test_data))
        
        # Check that labels use an integer-compatible dtype
        self.assertTrue(pd.api.types.is_integer_dtype(result_df['label'].dtype))


class TestNNDataPreparerIntegration(unittest.TestCase):
    """Integration tests for NNDataPreparer with realistic scenarios."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.create_realistic_test_data()
    
    def tearDown(self):
        """Clean up after integration tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_realistic_test_data(self):
        """Create more realistic test data for integration tests."""
        # Create symbols.json
        symbols_config = {
            "sectors": {
                "technology": ["AAPL", "MSFT", "GOOGL"],
                "finance": ["JPM"]
            }
        }
        
        symbols_path = os.path.join(self.temp_dir, 'config', 'symbols.json')
        os.makedirs(os.path.dirname(symbols_path), exist_ok=True)
        with open(symbols_path, 'w') as f:
            json.dump(symbols_config, f)
        
        # Create realistic price data with trends and volatility
        for symbol in ['AAPL', 'MSFT']:
            data_dir = os.path.join(self.temp_dir, 'data', 'historical', symbol, '1Hour')
            os.makedirs(data_dir, exist_ok=True)
            
            # Create 1000 hours of data (about 6 weeks)
            dates = pd.date_range('2024-01-01', periods=1000, freq='h')
            np.random.seed(42 if symbol == 'AAPL' else 43)
            
            # Generate realistic price series with trend and volatility
            base_price = 150 if symbol == 'AAPL' else 300
            returns = np.random.normal(0.0001, 0.02, 1000)  # Small positive drift, 2% hourly volatility
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Create OHLC data
            highs = prices * (1 + np.abs(np.random.normal(0, 0.005, 1000)))
            lows = prices * (1 - np.abs(np.random.normal(0, 0.005, 1000)))
            volumes = np.random.lognormal(14, 0.5, 1000).astype(int)
            
            # Create technical indicators
            sma_10 = pd.Series(prices).rolling(10).mean().bfill()
            sma_20 = pd.Series(prices).rolling(20).mean().bfill()
            
            test_data = pd.DataFrame({
                'Open': prices,
                'High': highs,
                'Low': lows,
                'Close': prices,
                'Volume': volumes,
                'VWAP': prices * (1 + np.random.normal(0, 0.001, 1000)),
                'SMA_10': sma_10,
                'SMA_20': sma_20,
                'MACD': np.random.normal(0, 0.5, 1000),
                'MACD_signal': np.random.normal(0, 0.3, 1000),
                'MACD_histogram': np.random.normal(0, 0.2, 1000),
                'RSI_14': 30 + 40 * np.random.beta(2, 2, 1000),  # RSI between 30-70 mostly
                'Stoch_K': 50 + 30 * np.random.normal(0, 1, 1000),
                'Stoch_D': 50 + 30 * np.random.normal(0, 1, 1000),
                'ADX_14': 20 + 30 * np.random.beta(2, 5, 1000),  # ADX typically 20-50
                'ATR_14': prices * 0.02 * np.random.lognormal(0, 0.3, 1000),
                'BB_bandwidth': 0.05 + 0.1 * np.random.beta(2, 5, 1000),
                'OBV': np.cumsum(volumes * np.sign(returns)),
                'Volume_SMA_20': pd.Series(volumes).rolling(20).mean().bfill(),
                'Returns_1h': returns,
                'sentiment_score_hourly_ffill': np.random.normal(0, 0.3, 1000),
                'DayOfWeek_sin': np.sin(2 * np.pi * dates.dayofweek / 7),
                'DayOfWeek_cos': np.cos(2 * np.pi * dates.dayofweek / 7)
            }, index=dates)
            
            # Save as parquet
            data_path = os.path.join(data_dir, 'data.parquet')
            test_data.to_parquet(data_path)
    
    def test_realistic_data_preparation_pipeline(self):
        """Test the complete pipeline with realistic data."""
        config = {
            'symbols_config_path': os.path.join(self.temp_dir, 'config', 'symbols.json'),
            'feature_list': [
                'SMA_10', 'SMA_20', 'MACD', 'RSI_14', 'sentiment_score_hourly_ffill',
                'DayOfWeek_sin', 'DayOfWeek_cos', 'ATR_14', 'Returns_1h'
            ],
            'nan_handling_features': 'ffill',
            'lookback_window': 24,
            'prediction_horizon': 8,
            'profit_target': 0.05,
            'stop_loss_target': 0.02,
            'data_base_path': os.path.join(self.temp_dir, 'data'),
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'shuffle_before_split': False,
            'scaling_method': 'standard',
            'calculate_sample_weights': True,
            'sample_weight_strategy': 'inverse_frequency',
            'output_path_scalers': os.path.join(self.temp_dir, 'scalers.joblib'),
            'symbols_list': ['AAPL', 'MSFT']
        }
        
        preparer = NNDataPreparer(config)
        prepared_data = preparer.get_prepared_data_for_training()
        
        # Verify the complete pipeline worked
        self.assertIsInstance(prepared_data, dict)
        
        # Check data quality
        train_data = prepared_data['train']
        
        # Should have reasonable number of samples
        self.assertGreater(len(train_data['X']), 100)
        
        # Verify label ratios are valid probabilities (full dataset retained)
        positive_ratio = np.mean(train_data['y'])
        self.assertGreaterEqual(positive_ratio, 0.0)
        self.assertLessEqual(positive_ratio, 1.0)
        
        # Check that scaling worked
        train_mean = train_data['X'].mean()
        train_std = train_data['X'].std()
        self.assertAlmostEqual(train_mean, 0.0, delta=0.2)
        self.assertAlmostEqual(train_std, 1.0, delta=0.2)
        
        # Check that asset IDs are valid
        unique_asset_ids = np.unique(train_data['asset_ids'])
        self.assertGreater(len(unique_asset_ids), 0)
        self.assertLessEqual(len(unique_asset_ids), 2)  # Should have at most 2 symbols
        
        # Verify sample weights exist and are reasonable
        self.assertIn('sample_weights', train_data)
        self.assertEqual(len(train_data['sample_weights']), len(train_data['y']))
        self.assertGreater(train_data['sample_weights'].min(), 0)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)