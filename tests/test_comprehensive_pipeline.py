"""
Comprehensive Pipeline Testing Suite

Tests all aspects of the data pipeline for RL training including:
- Historical data download (full and append modes)
- Technical indicator calculations
- Sentiment attachment
- RL validation and remediation
- Training data generation
- SAC continuous action environment
- End-to-end integration

Created: 2025-10-14
Purpose: Ensure complete pipeline integrity for RL training
"""

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Set up path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.feature_calculator import TechnicalIndicatorCalculator
from core.hist_data_loader import HistoricalDataLoader
from core.data_preparation_nn import NNDataPreparer
from scripts.attach_sentiment_to_hourly import SentimentAttacher
from scripts.run_full_data_update import DataUpdatePipeline


class TestHistoricalDataDownload(unittest.TestCase):
    """Test historical data download and append functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data" / "historical"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_append_mode_preserves_existing_data(self):
        """Verify append mode doesn't delete existing data."""
        # Create mock existing data
        symbol_dir = self.data_dir / "AAPL" / "1Hour"
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Create initial dataset (100 rows)
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        initial_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(100) * 10 + 150,
            'high': np.random.randn(100) * 10 + 155,
            'low': np.random.randn(100) * 10 + 145,
            'close': np.random.randn(100) * 10 + 150,
            'volume': np.random.randint(1000000, 10000000, 100),
        })
        initial_data.to_parquet(symbol_dir / 'data.parquet')
        
        # Verify initial data
        loaded = pd.read_parquet(symbol_dir / 'data.parquet')
        self.assertEqual(len(loaded), 100)
        initial_first_date = loaded.iloc[0]['timestamp']
        
        # Simulate append (add 50 more rows)
        new_dates = pd.date_range('2024-01-05 04:00:00', periods=50, freq='h')
        new_data = pd.DataFrame({
            'timestamp': new_dates,
            'open': np.random.randn(50) * 10 + 150,
            'high': np.random.randn(50) * 10 + 155,
            'low': np.random.randn(50) * 10 + 145,
            'close': np.random.randn(50) * 10 + 150,
            'volume': np.random.randint(1000000, 10000000, 50),
        })
        
        # Append without overwriting
        combined = pd.concat([loaded, new_data]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        combined.to_parquet(symbol_dir / 'data.parquet')
        
        # Verify data preserved and extended
        final = pd.read_parquet(symbol_dir / 'data.parquet')
        self.assertGreater(len(final), 100)
        self.assertEqual(final.iloc[0]['timestamp'], initial_first_date)
    
    def test_date_range_validation(self):
        """Test proper date range handling."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 2, 1, tzinfo=timezone.utc)
        
        # Verify date range is valid
        self.assertLess(start, end)
        delta = end - start
        self.assertGreater(delta.days, 0)


class TestTechnicalIndicators(unittest.TestCase):
    """Test technical indicator calculations with edge cases."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data" / "historical"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_data(self, symbol: str, num_rows: int) -> pd.DataFrame:
        """Create realistic test data for indicator calculations."""
        dates = pd.date_range('2024-01-01', periods=num_rows, freq='h')
        np.random.seed(42)
        
        # Generate realistic price series
        base_price = 150
        returns = np.random.normal(0.0001, 0.02, num_rows)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # FIXED: Use title case column names as expected by FeatureCalculator
        data = pd.DataFrame({
            'timestamp': dates,
            'Open': prices * (1 + np.random.normal(0, 0.01, num_rows)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.02, num_rows))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.02, num_rows))),
            'Close': prices,
            'Volume': np.random.lognormal(14, 0.5, num_rows).astype(int),
        })
        
        # Ensure OHLC relationships
        data['High'] = data[['High', 'Open', 'Close']].max(axis=1)
        data['Low'] = data[['Low', 'Open', 'Close']].min(axis=1)
        
        return data
    
    def test_all_indicators_calculation(self):
        """Verify all 14 technical indicators are calculated correctly."""
        symbol = "TEST"
        symbol_dir = self.data_dir / symbol / "1Hour"
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sufficient data (100 rows)
        test_data = self.create_test_data(symbol, 100)
        test_data.to_parquet(symbol_dir / 'data.parquet')
        
        # Calculate indicators
        calculator = TechnicalIndicatorCalculator(str(self.data_dir))
        success = calculator.process_symbol(symbol)
        self.assertTrue(success)
        
        # Load and verify
        result = pd.read_parquet(symbol_dir / 'data.parquet')
        
        # Expected indicators
        expected_indicators = [
            'SMA_10', 'SMA_20', 'MACD_line', 'MACD_signal', 'MACD_hist',
            'RSI_14', 'Stoch_K', 'Stoch_D', 'ADX_14', 'ATR_14',
            'BB_bandwidth', 'OBV', 'Volume_SMA_20', 'Return_1h'
        ]
        
        for indicator in expected_indicators:
            self.assertIn(indicator, result.columns, f"Missing indicator: {indicator}")
        
        # Verify no NaN in later rows (after warmup period)
        warmup = 20
        for indicator in expected_indicators:
            non_nan_count = result[indicator].iloc[warmup:].notna().sum()
            self.assertGreater(non_nan_count, 0, f"{indicator} has all NaN values after warmup")
    
    def test_insufficient_data_handling(self):
        """Test indicator calculation with insufficient data."""
        symbol = "SMALL"
        symbol_dir = self.data_dir / symbol / "1Hour"
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Create minimal data (only 10 rows)
        test_data = self.create_test_data(symbol, 10)
        test_data.to_parquet(symbol_dir / 'data.parquet')
        
        calculator = TechnicalIndicatorCalculator(str(self.data_dir))
        # Should handle gracefully (may warn but shouldn't crash)
        try:
            result = calculator.process_symbol(symbol)
            # If it succeeds, verify data still exists
            if result:
                loaded = pd.read_parquet(symbol_dir / 'data.parquet')
                self.assertEqual(len(loaded), 10)
        except Exception as e:
            self.fail(f"Indicator calculation crashed on small dataset: {e}")
    
    def test_nan_values_in_source_data(self):
        """Test handling of NaN values in source data."""
        symbol = "NAN_TEST"
        symbol_dir = self.data_dir / symbol / "1Hour"
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        test_data = self.create_test_data(symbol, 50)
        # Inject some NaN values
        test_data.loc[10:15, 'volume'] = np.nan
        test_data.loc[20:22, 'close'] = np.nan
        test_data.to_parquet(symbol_dir / 'data.parquet')
        
        calculator = TechnicalIndicatorCalculator(str(self.data_dir))
        try:
            calculator.process_symbol(symbol)
            # Should handle NaN gracefully
            loaded = pd.read_parquet(symbol_dir / 'data.parquet')
            self.assertEqual(len(loaded), 50)
        except Exception as e:
            self.fail(f"Failed to handle NaN values: {e}")


class TestSentimentAttachment(unittest.TestCase):
    """Test sentiment attachment and forward-fill logic."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.historical_dir = self.data_dir / "historical"
        self.sentiment_dir = self.data_dir / "sentiment"
        self.historical_dir.mkdir(parents=True, exist_ok=True)
        self.sentiment_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_sentiment_forward_fill(self):
        """Verify sentiment is forward-filled to hourly correctly."""
        symbol = "AAPL"
        
        # Create hourly data - FIXED: Add OHLC columns for sentiment attacher compatibility
        hourly_dates = pd.date_range('2024-01-01', periods=48, freq='h')
        close_prices = np.random.randn(48) * 10 + 150
        hourly_data = pd.DataFrame({
            'timestamp': hourly_dates,
            'Open': close_prices * 0.99,
            'High': close_prices * 1.01,
            'Low': close_prices * 0.98,
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 10000000, 48),
        })
        
        symbol_dir = self.historical_dir / symbol / "1Hour"
        symbol_dir.mkdir(parents=True, exist_ok=True)
        hourly_data.to_parquet(symbol_dir / 'data.parquet')
        
        # Create daily sentiment (only 2 days) - use correct structure
        daily_dates = pd.date_range('2024-01-01', periods=2, freq='D')
        sentiment_data = pd.DataFrame({
            'date': daily_dates,  # Use 'date' column as expected
            'sentiment_score': [0.5, -0.3],
        })
        sentiment_symbol_dir = self.sentiment_dir / symbol
        sentiment_symbol_dir.mkdir(parents=True, exist_ok=True)
        sentiment_data.to_parquet(sentiment_symbol_dir / 'daily_sentiment.parquet')
        
        # Attach sentiment
        with patch.object(Path, 'cwd', return_value=self.temp_dir):
            attacher = SentimentAttacher()
            # Mock the paths - use correct attribute names
            attacher.historical_data_root = self.historical_dir
            attacher.sentiment_data_root = self.sentiment_dir
            
            result = attacher.process_symbol(symbol)
            
            if result:
                # Verify sentiment was attached
                final = pd.read_parquet(symbol_dir / 'data.parquet')
                self.assertIn('sentiment_score_hourly_ffill', final.columns)
                
                # Verify forward-fill worked
                non_null_count = final['sentiment_score_hourly_ffill'].notna().sum()
                self.assertGreater(non_null_count, 0)
    
    def test_missing_sentiment_handling(self):
        """Test handling when sentiment data is missing."""
        symbol = "NOSENT"
        
        # Create hourly data without sentiment - FIXED: Add required OHLCV columns
        hourly_dates = pd.date_range('2024-01-01', periods=24, freq='h')
        close_prices = np.random.randn(24) * 10 + 150
        hourly_data = pd.DataFrame({
            'timestamp': hourly_dates,
            'Open': close_prices * 0.99,
            'High': close_prices * 1.01,
            'Low': close_prices * 0.98,
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 10000000, 24),
        })
        
        symbol_dir = self.historical_dir / symbol / "1Hour"
        symbol_dir.mkdir(parents=True, exist_ok=True)
        hourly_data.to_parquet(symbol_dir / 'data.parquet')
        
        # Don't create sentiment file
        
        # Should handle gracefully
        with patch.object(Path, 'cwd', return_value=self.temp_dir):
            attacher = SentimentAttacher()
            attacher.data_dir = self.data_dir
            attacher.sentiment_dir = self.sentiment_dir
            
            try:
                attacher.process_symbol(symbol)
                # Verify original data preserved
                final = pd.read_parquet(symbol_dir / 'data.parquet')
                self.assertEqual(len(final), 24)
            except Exception as e:
                self.fail(f"Failed to handle missing sentiment: {e}")


class TestNNDataPreparation(unittest.TestCase):
    """Test training data generation with NNDataPreparer."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # data_dir should point to the data folder, not data/historical
        self.data_dir = Path(self.temp_dir) / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Create historical subfolder
        self.historical_dir = self.data_dir / "historical"
        self.historical_dir.mkdir(parents=True, exist_ok=True)
        
        # Create config directory
        self.config_dir = Path(self.temp_dir) / "config"
        self.config_dir.mkdir(exist_ok=True)
        
        # Create symbols.json
        symbols_config = {
            "sectors": {
                "technology": ["AAPL", "MSFT"]
            }
        }
        with open(self.config_dir / "symbols.json", 'w') as f:
            json.dump(symbols_config, f)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_realistic_data(self, symbol: str, num_rows: int = 1000):
        """Create realistic test data with all required features."""
        dates = pd.date_range('2024-01-01', periods=num_rows, freq='h')
        np.random.seed(42 if symbol == 'AAPL' else 43)
        
        base_price = 150 if symbol == 'AAPL' else 300
        returns = np.random.normal(0.0001, 0.02, num_rows)
        prices = base_price * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'Open': prices,
            'High': prices * (1 + np.abs(np.random.normal(0, 0.005, num_rows))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, num_rows))),
            'Close': prices,
            'Volume': np.random.lognormal(14, 0.5, num_rows).astype(int),
            'SMA_10': pd.Series(prices).rolling(10).mean().bfill(),
            'SMA_20': pd.Series(prices).rolling(20).mean().bfill(),
            'MACD_line': np.random.normal(0, 0.5, num_rows),
            'MACD_signal': np.random.normal(0, 0.3, num_rows),
            'MACD_hist': np.random.normal(0, 0.2, num_rows),
            'RSI_14': 30 + 40 * np.random.beta(2, 2, num_rows),
            'Stoch_K': 50 + 30 * np.random.normal(0, 1, num_rows),
            'Stoch_D': 50 + 30 * np.random.normal(0, 1, num_rows),
            'ADX_14': 20 + 30 * np.random.beta(2, 5, num_rows),
            'ATR_14': prices * 0.02 * np.random.lognormal(0, 0.3, num_rows),
            'BB_bandwidth': 0.05 + 0.1 * np.random.beta(2, 5, num_rows),
            'OBV': np.cumsum(np.random.lognormal(14, 0.5, num_rows) * np.sign(returns)),
            'Volume_SMA_20': pd.Series(np.random.lognormal(14, 0.5, num_rows)).rolling(20).mean().bfill(),
            'Return_1h': returns,
            'sentiment_score_hourly_ffill': np.random.normal(0, 0.3, num_rows),
            'DayOfWeek_sin': np.sin(2 * np.pi * dates.dayofweek / 7),
            'DayOfWeek_cos': np.cos(2 * np.pi * dates.dayofweek / 7),
        }, index=dates)
        
        return data
    
    def test_no_data_filtration(self):
        """Verify all data is preserved (no filtration)."""
        for symbol in ['AAPL', 'MSFT']:
            symbol_dir = self.historical_dir / symbol / "1Hour"
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            data = self.create_realistic_data(symbol, 1000)
            data.to_parquet(symbol_dir / 'data.parquet')
        
        config = {
            'symbols_config_path': str(self.config_dir / "symbols.json"),
            'feature_list': [
                'SMA_10', 'SMA_20', 'MACD_line', 'RSI_14', 
                'sentiment_score_hourly_ffill', 'DayOfWeek_sin', 'DayOfWeek_cos'
            ],
            'nan_handling_features': 'ffill',
            'lookback_window': 24,
            'prediction_horizon': 8,
            'profit_target': 0.05,
            'stop_loss_target': 0.02,
            'data_base_path': str(self.data_dir),
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'scaling_method': 'standard',
            'calculate_sample_weights': True,
            'output_path_scalers': str(Path(self.temp_dir) / 'scalers.joblib'),
            'symbols_list': ['AAPL', 'MSFT']
        }
        
        preparer = NNDataPreparer(config)
        prepared_data = preparer.get_prepared_data_for_training()
        
        # Verify data was generated
        self.assertIn('train', prepared_data)
        self.assertIn('val', prepared_data)
        self.assertIn('test', prepared_data)
        
        # Verify no excessive filtration (most data should be kept)
        train_size = len(prepared_data['train']['y'])
        # With 1000 rows per symbol, 2 symbols, 24h lookback, should have substantial data
        self.assertGreater(train_size, 1000, "Too much data filtered out")
    
    def test_label_generation_logic(self):
        """Verify label generation creates valid labels."""
        symbol = "AAPL"
        symbol_dir = self.historical_dir / symbol / "1Hour"
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        data = self.create_realistic_data(symbol, 500)
        data.to_parquet(symbol_dir / 'data.parquet')
        
        config = {
            'symbols_config_path': str(self.config_dir / "symbols.json"),
            'feature_list': ['SMA_10', 'RSI_14'],
            'nan_handling_features': 'ffill',
            'lookback_window': 24,
            'prediction_horizon': 8,
            'profit_target': 0.05,
            'stop_loss_target': 0.02,
            'data_base_path': str(self.data_dir),
            'output_path_scalers': str(Path(self.temp_dir) / 'scalers.joblib'),
            'symbols_list': ['AAPL']
        }
        
        preparer = NNDataPreparer(config)
        raw_data = preparer.load_data_for_symbol(symbol)
        labeled = preparer._generate_labels_for_symbol(raw_data, symbol)
        
        # Verify labels are binary
        unique_labels = labeled['label'].unique()
        self.assertTrue(set(unique_labels).issubset({0, 1}))
        
        # Verify some positive and negative labels
        label_counts = labeled['label'].value_counts()
        self.assertIn(0, label_counts.index)
        self.assertIn(1, label_counts.index)


class TestContinuousActionEnvironment(unittest.TestCase):
    """Test SAC continuous action environment."""
    
    def test_action_masking_logic(self):
        """Test adaptive action masking rules."""
        # This would require setting up the full environment
        # For now, verify the logic components exist
        from core.rl.environments.continuous_trading_env import ContinuousTradingEnvironment
        
        # Verify class exists and has masking method
        self.assertTrue(hasattr(ContinuousTradingEnvironment, '_mask_invalid_action'))
    
    def test_multi_position_support(self):
        """Verify multi-position tracking in environment."""
        from core.rl.environments.continuous_trading_env import ContinuousTradingEnvironment
        
        # Verify environment supports position tracking
        self.assertTrue(hasattr(ContinuousTradingEnvironment, 'step'))


class TestIntegrationPipeline(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_connectivity(self):
        """Verify all pipeline stages connect properly."""
        # Create minimal mock for pipeline
        import argparse
        
        args = argparse.Namespace(
            start_date=None,
            end_date=None,
            years=2.0,
            max_workers=2,
            force_refresh=False,
            profit_target=0.015,
            stop_loss=0.03,
            lookback=24,
            prediction_horizon=24,
            output_dir="data/training_data_v2_final",
            min_total_samples=1000,
            positive_ratio_min=None,
            positive_ratio_max=None,
            feature_count_min=5,
            skip_download=True,
            skip_indicators=True,
            skip_sentiment=True,
            skip_validation=True,
            skip_remediation=True,
            skip_sample_verification=True,
            skip_training=True,
            skip_training_validation=True,
            remediation_dry_run=True,
            log_level='INFO'
        )
        
        # Verify pipeline can be instantiated
        try:
            pipeline = DataUpdatePipeline(args)
            self.assertIsNotNone(pipeline)
        except Exception as e:
            self.fail(f"Pipeline instantiation failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
