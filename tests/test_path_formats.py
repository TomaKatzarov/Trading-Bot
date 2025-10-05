#!/usr/bin/env python3
"""
Test script to validate that the fix works with different path formats.
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

def test_different_path_formats():
    """Test different path formats to ensure robustness."""
    
    test_cases = [
        {
            'name': 'Relative path with ../',
            'path': '../data/training_data',
            'expected': True
        },
        {
            'name': 'Direct relative path',
            'path': 'data/training_data',
            'expected': True
        },
        {
            'name': 'Absolute path',
            'path': str(Path(__file__).parent / 'data' / 'training_data'),
            'expected': True
        },
        {
            'name': 'Non-existent path',
            'path': 'data/nonexistent_data',
            'expected': False
        }
    ]
    
    base_config = {
        'use_pregenerated_data': True,
        'data_config': {
            'symbols': ['AAPL'],
            'feature_columns': ['close'],
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
        'model_config': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.1},
        'training_config': {'batch_size': 32, 'learning_rate': 0.001, 'epochs': 1, 'early_stopping_patience': 10},
        'model_type': 'lstm',
        'experiment_name': 'path_format_test',
        'use_mlflow': False
    }
    
    for test_case in test_cases:
        logger.info(f"\nTesting: {test_case['name']}")
        logger.info(f"Path: {test_case['path']}")
        
        config = base_config.copy()
        config['pregenerated_data_path'] = test_case['path']
        
        try:
            trainer = ModelTrainer(config)
            
            # Test path resolution logic
            use_pregenerated = trainer.config.get('use_pregenerated_data', False)
            pregenerated_data_path = trainer.config.get('pregenerated_data_path', 'data/training_data')
            
            # Resolve path (same logic as in ModelTrainer)
            if not Path(pregenerated_data_path).is_absolute():
                project_root = Path(__file__).parent
                resolved_path = project_root / pregenerated_data_path
                
                # If the resolved path doesn't exist, try interpreting it as relative to project root directly
                if not resolved_path.exists():
                    cleaned_path = pregenerated_data_path
                    if cleaned_path.startswith('../'):
                        cleaned_path = cleaned_path[3:]
                    alternative_path = project_root / cleaned_path
                    if alternative_path.exists():
                        resolved_path = alternative_path
            else:
                resolved_path = Path(pregenerated_data_path)
            
            exists = resolved_path.exists()
            logger.info(f"Resolved to: {resolved_path}")
            logger.info(f"Exists: {exists}")
            
            assert exists == test_case['expected'], (
                f"Path resolution mismatch for {test_case['name']}: "
                f"expected exists={test_case['expected']}, got {exists}"
            )
            logger.info("✅ PASSED")

        except Exception as e:
            logger.error(f"❌ FAILED - Exception: {e}")
            raise AssertionError(
                f"Path format test '{test_case['name']}' raised an exception: {e}"
            ) from e

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing different path formats")
    logger.info("=" * 60)
    
    success = True
    try:
        test_different_path_formats()
    except AssertionError as err:
        logger.error(str(err))
        success = False
    except Exception as err:
        logger.error(f"Unexpected error: {err}")
        success = False
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    if success:
        logger.info("🎉 ALL PATH FORMAT TESTS PASSED!")
        sys.exit(0)
    else:
        logger.error("💥 SOME PATH FORMAT TESTS FAILED!")
        sys.exit(1)
