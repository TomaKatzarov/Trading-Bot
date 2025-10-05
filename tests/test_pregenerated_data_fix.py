#!/usr/bin/env python3
"""
Test script to validate the fix for the use_pregenerated_data parameter issue.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training.train_nn_model import ModelTrainer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_path_resolution():
    """Test the path resolution logic in the ModelTrainer."""
    
    # Create a minimal config for testing
    test_config = {
        'use_pregenerated_data': True,
        'pregenerated_data_path': '../data/training_data',  # Relative path from training directory
        'data_config': {
            'symbols': ['AAPL', 'GOOGL'],
            'feature_columns': ['close', 'volume'],
            'lookback_window': 24,
            'target_profit_pct': 5.0,
            'target_stop_loss_pct': 2.0,
            'target_horizon_hours': 8,
            'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
            'min_samples_per_symbol': 100,
            'nan_handling': 'ffill',
            'use_robust_scaler_for': ['volume'],
            'multi_symbol_training': True
        },
        'model_config': {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.1
        },
        'training_config': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 1,
            'early_stopping_patience': 10
        },
        'model_type': 'lstm',
        'experiment_name': 'path_test',
        'use_mlflow': False  # Disable MLflow for this test
    }
    
    logger.info("Creating ModelTrainer instance...")
    trainer = ModelTrainer(test_config)
    
    # Test the path resolution logic
    use_pregenerated = trainer.config.get('use_pregenerated_data', False)
    pregenerated_data_path = trainer.config.get('pregenerated_data_path', 'data/training_data')
    
    logger.info(f"use_pregenerated_data: {use_pregenerated}")
    logger.info(f"pregenerated_data_path: {pregenerated_data_path}")
      # Resolve path relative to project root if it's a relative path
    if not Path(pregenerated_data_path).is_absolute():
        # Get project root (parent of training directory)
        project_root = Path(__file__).parent
        resolved_path = project_root / pregenerated_data_path
        
        # If the resolved path doesn't exist, try interpreting it as relative to project root directly
        if not resolved_path.exists():
            # Try removing the leading "../" if present
            cleaned_path = pregenerated_data_path
            if cleaned_path.startswith('../'):
                cleaned_path = cleaned_path[3:]
            alternative_path = project_root / cleaned_path
            if alternative_path.exists():
                resolved_path = alternative_path
    else:
        resolved_path = Path(pregenerated_data_path)
    
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Test script location: {Path(__file__).parent}")
    logger.info(f"Original path: {pregenerated_data_path}")
    logger.info(f"Resolved path: {resolved_path}")
    logger.info(f"Resolved path exists: {resolved_path.exists()}")
    
    if resolved_path.exists():
        logger.info("✅ Pre-generated data path resolved successfully!")
        
        # Check if all required files exist
        required_files = [
            'train_X.npy', 'train_y.npy', 'train_asset_ids.npy',
            'val_X.npy', 'val_y.npy', 'val_asset_ids.npy', 
            'test_X.npy', 'test_y.npy', 'test_asset_ids.npy',
            'scalers.joblib', 'asset_id_mapping.json', 'metadata.json'
        ]
        
        missing_files = []
        for file in required_files:
            if not (resolved_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.warning(f"❌ Missing required files: {missing_files}")
            return False
        else:
            logger.info("✅ All required pre-generated data files found!")
            return True
    else:
        logger.error(f"❌ Pre-generated data path does not exist: {resolved_path}")
        return False

def test_trainer_load_data():
    """Test the actual load_data method with the fix."""
    
    # Create a minimal config for testing
    test_config = {
        'use_pregenerated_data': True,
        'pregenerated_data_path': '../data/training_data',
        'data_config': {
            'symbols': ['AAPL', 'GOOGL'],
            'feature_columns': ['close', 'volume'],
            'lookback_window': 24,
            'target_profit_pct': 5.0,
            'target_stop_loss_pct': 2.0,
            'target_horizon_hours': 8,
            'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
            'min_samples_per_symbol': 100,
            'nan_handling': 'ffill',
            'use_robust_scaler_for': ['volume'],
            'multi_symbol_training': True
        },
        'model_config': {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.1
        },
        'training_config': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 1,
            'early_stopping_patience': 10
        },
        'model_type': 'lstm',
        'experiment_name': 'data_load_test',
        'use_mlflow': False
    }
    
    try:
        logger.info("Testing ModelTrainer.load_data() with the fix...")
        trainer = ModelTrainer(test_config)
        trainer.load_data()
        
        # Check if data loaders were created
        if hasattr(trainer, 'train_loader') and trainer.train_loader is not None:
            logger.info(f"✅ Training data loaded successfully!")
            logger.info(f"   Train loader: {len(trainer.train_loader)} batches")
            logger.info(f"   Val loader: {len(trainer.val_loader)} batches")
            logger.info(f"   Test loader: {len(trainer.test_loader)} batches")
            return True
        else:
            logger.error("❌ Training data loaders not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing load_data: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing pre-generated data parameter fix")
    logger.info("=" * 60)
    
    # Test 1: Path resolution logic
    logger.info("\n1. Testing path resolution logic...")
    path_test_passed = test_path_resolution()
    
    # Test 2: Actual load_data method (only if path test passed)
    if path_test_passed:
        logger.info("\n2. Testing ModelTrainer.load_data() method...")
        load_data_test_passed = test_trainer_load_data()
    else:
        logger.warning("\n2. Skipping load_data test due to path resolution failure")
        load_data_test_passed = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Path resolution: {'✅ PASSED' if path_test_passed else '❌ FAILED'}")
    logger.info(f"Data loading: {'✅ PASSED' if load_data_test_passed else '❌ FAILED'}")
    
    if path_test_passed and load_data_test_passed:
        logger.info("\n🎉 ALL TESTS PASSED! The fix is working correctly.")
        sys.exit(0)
    else:
        logger.error("\n💥 SOME TESTS FAILED! Please check the implementation.")
        sys.exit(1)
