#!/usr/bin/env python3
"""
Verification Script for core/data_preparation_nn.py Outputs

This script verifies the data preparation pipeline outputs as outlined in 
Section 1: Prerequisites and Data Finalization of plan_1.4_train_tune_nn_models.md

Created: 2025-05-28
Author: Roo (AI Assistant)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.data_preparation_nn import NNDataPreparer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_config() -> Dict[str, Any]:
    """Create test configuration for data preparation."""
    return {
        'symbols_config_path': str(project_root / 'config' / 'symbols.json'),
        'feature_list': [
            'open', 'high', 'low', 'close', 'volume', 'vwap',
            'SMA_10', 'SMA_20', 'MACD_line', 'MACD_signal', 'MACD_hist',
            'RSI_14', 'Stoch_K', 'Stoch_D', 'ADX_14', 'ATR_14',
            'BB_bandwidth', 'OBV', 'Volume_SMA_20', '1h_return',
            'sentiment_score_hourly_ffill', 'DayOfWeek_sin', 'DayOfWeek_cos'
        ],
        'nan_handling_features': 'ffill',
        'lookback_window': 24,
        'prediction_horizon_hours': 8,
        'profit_target': 0.05,
        'stop_loss_target': 0.02,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'scaling_method': 'standard',
        'sample_weight_strategy': 'balanced',
        'shuffle_splits': False
    }

def verify_activity_1_data_preparation_outputs():
    """Activity 1: Verify core/data_preparation_nn.py outputs."""
    logger.info("=" * 60)
    logger.info("ACTIVITY 1: Verify core/data_preparation_nn.py Outputs")
    logger.info("=" * 60)
    
    results = {'status': 'UNKNOWN', 'checks': {}, 'issues': []}
    
    try:
        config = create_test_config()
        preparer = NNDataPreparer(config)
        
        # Check 1.1: Asset ID mapping
        logger.info("Check 1.1: Verifying asset ID mapping...")
        mapping_file = project_root / 'config' / 'asset_id_mapping.json'
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
            logger.info(f"✓ Asset ID mapping exists with {len(mapping_data.get('symbol_to_id', {}))} symbols")
            results['checks']['asset_id_mapping'] = True
        else:
            logger.error("✗ Asset ID mapping file not found")
            results['issues'].append("Asset ID mapping file not found")
            results['checks']['asset_id_mapping'] = False
        
        # Check 1.2: Single symbol data loading
        logger.info("Check 1.2: Testing single symbol data loading...")
        test_symbol = 'AAPL'
        data_file = project_root / 'data' / 'historical' / test_symbol / '1Hour' / 'data.parquet'
        
        if data_file.exists():
            df = preparer.load_data_for_symbol(test_symbol)
            if df is not None and not df.empty:
                logger.info(f"✓ Successfully loaded {test_symbol} data: {df.shape}")
                logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
                logger.info(f"  Columns: {len(df.columns)} features")
                results['checks']['data_loading'] = True
            else:
                logger.error(f"✗ Failed to load data for {test_symbol}")
                results['issues'].append(f"Failed to load data for {test_symbol}")
                results['checks']['data_loading'] = False
        else:
            logger.error(f"✗ Data file not found for {test_symbol}: {data_file}")
            results['issues'].append(f"Data file not found for {test_symbol}")
            results['checks']['data_loading'] = False
        
        # Check 1.3: Feature processing
        logger.info("Check 1.3: Verifying feature processing...")
        if results['checks']['data_loading']:
            processed_df = preparer._preprocess_single_symbol_data(test_symbol)
            if processed_df is not None:
                expected_features = config['feature_list']
                missing_features = [f for f in expected_features if f not in processed_df.columns]
                
                if not missing_features:
                    logger.info(f"✓ All {len(expected_features)} expected features present")
                    results['checks']['feature_processing'] = True
                else:
                    logger.error(f"✗ Missing features: {missing_features}")
                    results['issues'].append(f"Missing features: {missing_features}")
                    results['checks']['feature_processing'] = False
                
                # Check for NaN values
                nan_counts = processed_df.isnull().sum()
                features_with_nans = nan_counts[nan_counts > 0]
                if len(features_with_nans) == 0:
                    logger.info("✓ No NaN values in processed features")
                else:
                    logger.warning(f"⚠ Features with NaN values: {dict(features_with_nans)}")
            else:
                logger.error("✗ Feature processing returned None")
                results['issues'].append("Feature processing returned None")
                results['checks']['feature_processing'] = False
        
        # Check 1.4: Label generation
        logger.info("Check 1.4: Testing label generation...")
        if results['checks']['data_loading']:
            raw_df = preparer.load_data_for_symbol(test_symbol)
            labeled_df = preparer._generate_labels_for_symbol(raw_df, test_symbol)
            
            if 'target' in labeled_df.columns:
                labels = labeled_df['target'].dropna()
                unique_labels = labels.unique()
                label_counts = labels.value_counts()
                
                if set(unique_labels).issubset({0, 1}):
                    logger.info(f"✓ Binary labels generated: {dict(label_counts)}")
                    pos_pct = (label_counts.get(1, 0) / len(labels) * 100) if len(labels) > 0 else 0
                    logger.info(f"  Positive label percentage: {pos_pct:.2f}%")
                    results['checks']['label_generation'] = True
                else:
                    logger.error(f"✗ Non-binary labels found: {unique_labels}")
                    results['issues'].append(f"Non-binary labels: {unique_labels}")
                    results['checks']['label_generation'] = False
            else:
                logger.error("✗ Target column not generated")
                results['issues'].append("Target column not generated")
                results['checks']['label_generation'] = False
        
        # Check 1.5: Sequence generation
        logger.info("Check 1.5: Testing sequence generation...")
        if results['checks']['feature_processing'] and results['checks']['label_generation']:
            try:
                asset_id = preparer.asset_id_map.get(test_symbol, 0)
                processed_df = preparer._preprocess_single_symbol_data(test_symbol)
                raw_df = preparer.load_data_for_symbol(test_symbol)
                labeled_df = preparer._generate_labels_for_symbol(raw_df, test_symbol)
                
                X, y, asset_ids = preparer._generate_sequences_for_symbol(processed_df, labeled_df, asset_id)
                
                if X is not None and y is not None and asset_ids is not None:
                    lookback_window = config['lookback_window']
                    n_features = len(config['feature_list'])
                    
                    logger.info(f"✓ Sequences generated successfully")
                    logger.info(f"  X shape: {X.shape} (expected: (n_samples, {lookback_window}, {n_features}))")
                    logger.info(f"  y shape: {y.shape}")
                    logger.info(f"  asset_ids shape: {asset_ids.shape}")
                    logger.info(f"  Data types: X={X.dtype}, y={y.dtype}, asset_ids={asset_ids.dtype}")
                    
                    # Verify shapes
                    shape_ok = (X.shape[1] == lookback_window and 
                               X.shape[2] == n_features and
                               y.shape[0] == X.shape[0] and
                               asset_ids.shape[0] == X.shape[0])
                    
                    if shape_ok:
                        logger.info("✓ All shapes are correct")
                        results['checks']['sequence_generation'] = True
                    else:
                        logger.error("✗ Shape mismatch in generated sequences")
                        results['issues'].append("Shape mismatch in sequences")
                        results['checks']['sequence_generation'] = False
                else:
                    logger.error("✗ Sequence generation returned None values")
                    results['issues'].append("Sequence generation returned None")
                    results['checks']['sequence_generation'] = False
                    
            except Exception as e:
                logger.error(f"✗ Exception in sequence generation: {str(e)}")
                results['issues'].append(f"Sequence generation exception: {str(e)}")
                results['checks']['sequence_generation'] = False
        
        # Determine overall status
        all_passed = all(results['checks'].values())
        results['status'] = 'PASSED' if all_passed else 'FAILED'
        
        logger.info(f"\nActivity 1 Status: {results['status']}")
        logger.info(f"Checks passed: {sum(results['checks'].values())}/{len(results['checks'])}")
        if results['issues']:
            logger.info(f"Issues found: {len(results['issues'])}")
            for issue in results['issues']:
                logger.info(f"  - {issue}")
        
    except Exception as e:
        logger.error(f"Critical error in Activity 1: {str(e)}")
        results['status'] = 'ERROR'
        results['issues'].append(f"Critical error: {str(e)}")
    
    return results

def verify_activity_2_data_splits():
    """Activity 2: Verify data splits."""
    logger.info("\n" + "=" * 60)
    logger.info("ACTIVITY 2: Verify Data Splits")
    logger.info("=" * 60)
    
    results = {'status': 'UNKNOWN', 'checks': {}, 'issues': []}
    
    try:
        config = create_test_config()
        preparer = NNDataPreparer(config)
        
        logger.info("Preparing data for split verification...")
        prepared_data = preparer.get_prepared_data_for_training()
        
        # Check split ratios
        train_data = prepared_data.get('train', {})
        val_data = prepared_data.get('val', {})
        test_data = prepared_data.get('test', {})
        
        train_size = len(train_data.get('X', []))
        val_size = len(val_data.get('X', []))
        test_size = len(test_data.get('X', []))
        total_size = train_size + val_size + test_size
        
        if total_size > 0:
            train_ratio = train_size / total_size
            val_ratio = val_size / total_size
            test_ratio = test_size / total_size
            
            logger.info(f"✓ Data split sizes:")
            logger.info(f"  Train: {train_size} ({train_ratio:.3f})")
            logger.info(f"  Val: {val_size} ({val_ratio:.3f})")
            logger.info(f"  Test: {test_size} ({test_ratio:.3f})")
            
            expected_ratios = (config['train_ratio'], config['val_ratio'], config['test_ratio'])
            actual_ratios = (train_ratio, val_ratio, test_ratio)
            
            # Allow 5% tolerance
            ratio_ok = all(abs(actual - expected) < 0.05 for actual, expected in zip(actual_ratios, expected_ratios))
            
            if ratio_ok:
                logger.info("✓ Split ratios are within expected range")
                results['checks']['split_ratios'] = True
            else:
                logger.error(f"✗ Split ratios outside expected range. Expected: {expected_ratios}, Actual: {actual_ratios}")
                results['issues'].append("Split ratios outside expected range")
                results['checks']['split_ratios'] = False
        else:
            logger.error("✗ No data in splits")
            results['issues'].append("No data in splits")
            results['checks']['split_ratios'] = False
        
        # Check data types and shapes consistency
        logger.info("Checking data consistency across splits...")
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            X = split_data.get('X')
            y = split_data.get('y')
            asset_ids = split_data.get('asset_ids')
            
            if X is not None and y is not None and asset_ids is not None:
                logger.info(f"✓ {split_name}: X{X.shape}, y{y.shape}, asset_ids{asset_ids.shape}")
                
                # Check data types
                if X.dtype == np.float32 and y.dtype == np.int32 and asset_ids.dtype == np.int32:
                    logger.info(f"✓ {split_name}: Correct data types")
                else:
                    logger.error(f"✗ {split_name}: Incorrect data types - X:{X.dtype}, y:{y.dtype}, asset_ids:{asset_ids.dtype}")
                    results['issues'].append(f"{split_name} has incorrect data types")
            else:
                logger.error(f"✗ {split_name}: Missing data arrays")
                results['issues'].append(f"{split_name} missing data arrays")
        
        results['checks']['data_consistency'] = len([issue for issue in results['issues'] if 'data types' in issue or 'Missing data' in issue]) == 0
        
        # Determine overall status
        all_passed = all(results['checks'].values())
        results['status'] = 'PASSED' if all_passed else 'FAILED'
        
        logger.info(f"\nActivity 2 Status: {results['status']}")
        logger.info(f"Checks passed: {sum(results['checks'].values())}/{len(results['checks'])}")
        
    except Exception as e:
        logger.error(f"Critical error in Activity 2: {str(e)}")
        results['status'] = 'ERROR'
        results['issues'].append(f"Critical error: {str(e)}")
    
    return results

def verify_activity_4_normalization():
    """Activity 4: Verify normalization procedures."""
    logger.info("\n" + "=" * 60)
    logger.info("ACTIVITY 4: Verify Normalization Procedures")
    logger.info("=" * 60)
    
    results = {'status': 'UNKNOWN', 'checks': {}, 'issues': []}
    
    try:
        config = create_test_config()
        preparer = NNDataPreparer(config)
        
        # Check scaler types
        logger.info("Checking scaler configuration...")
        scaling_method = config.get('scaling_method', 'standard')
        logger.info(f"✓ Scaling method configured: {scaling_method}")
        
        # Get prepared data to check normalization
        logger.info("Preparing data to check normalization...")
        prepared_data = preparer.get_prepared_data_for_training()
        
        # Check if scalers are saved
        if hasattr(preparer, 'scalers') and preparer.scalers:
            logger.info(f"✓ Scalers available: {list(preparer.scalers.keys())}")
            results['checks']['scaler_availability'] = True
        else:
            logger.error("✗ No scalers found")
            results['issues'].append("No scalers found")
            results['checks']['scaler_availability'] = False
        
        # Check normalization quality on training data
        train_X = prepared_data.get('train', {}).get('X')
        if train_X is not None:
            # Reshape to 2D for analysis
            train_X_2d = train_X.reshape(-1, train_X.shape[-1])
            
            means = np.mean(train_X_2d, axis=0)
            stds = np.std(train_X_2d, axis=0)
            
            # For StandardScaler, means should be close to 0, stds close to 1
            mean_ok = np.allclose(means, 0, atol=0.1)
            std_ok = np.allclose(stds, 1, atol=0.2)
            
            logger.info(f"Training data normalization check:")
            logger.info(f"  Mean range: [{means.min():.4f}, {means.max():.4f}] (should be ~0)")
            logger.info(f"  Std range: [{stds.min():.4f}, {stds.max():.4f}] (should be ~1)")
            
            if mean_ok and std_ok:
                logger.info("✓ Training data properly normalized")
                results['checks']['normalization_quality'] = True
            else:
                logger.error("✗ Training data normalization issues")
                results['issues'].append("Training data normalization issues")
                results['checks']['normalization_quality'] = False
        else:
            logger.error("✗ No training data available for normalization check")
            results['issues'].append("No training data for normalization check")
            results['checks']['normalization_quality'] = False
        
        # Check scaler persistence
        logger.info("Testing scaler persistence...")
        try:
            temp_path = project_root / 'temp_scalers_test.joblib'
            preparer.save_scalers(temp_path)
            
            if temp_path.exists():
                logger.info("✓ Scalers can be saved")
                
                # Test loading
                test_preparer = NNDataPreparer(config)
                test_preparer.load_scalers(temp_path)
                
                if test_preparer.scalers:
                    logger.info("✓ Scalers can be loaded")
                    results['checks']['scaler_persistence'] = True
                else:
                    logger.error("✗ Scalers could not be loaded")
                    results['issues'].append("Scalers could not be loaded")
                    results['checks']['scaler_persistence'] = False
                
                # Clean up
                temp_path.unlink()
            else:
                logger.error("✗ Scalers could not be saved")
                results['issues'].append("Scalers could not be saved")
                results['checks']['scaler_persistence'] = False
                
        except Exception as e:
            logger.error(f"✗ Scaler persistence test failed: {str(e)}")
            results['issues'].append(f"Scaler persistence test failed: {str(e)}")
            results['checks']['scaler_persistence'] = False
        
        # Determine overall status
        all_passed = all(results['checks'].values())
        results['status'] = 'PASSED' if all_passed else 'FAILED'
        
        logger.info(f"\nActivity 4 Status: {results['status']}")
        logger.info(f"Checks passed: {sum(results['checks'].values())}/{len(results['checks'])}")
        
    except Exception as e:
        logger.error(f"Critical error in Activity 4: {str(e)}")
        results['status'] = 'ERROR'
        results['issues'].append(f"Critical error: {str(e)}")
    
    return results

def main():
    """Run all verification activities."""
    logger.info("STARTING DATA PREPARATION VERIFICATION")
    logger.info("=" * 80)
    
    all_results = {}
    
    # Run all activities
    all_results['activity_1'] = verify_activity_1_data_preparation_outputs()
    all_results['activity_2'] = verify_activity_2_data_splits()
    all_results['activity_4'] = verify_activity_4_normalization()
    
    # Generate summary
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)
    
    total_activities = len(all_results)
    passed_activities = sum(1 for result in all_results.values() if result['status'] == 'PASSED')
    
    logger.info(f"Activities completed: {total_activities}")
    logger.info(f"Activities passed: {passed_activities}")
    logger.info(f"Overall success rate: {passed_activities/total_activities*100:.1f}%")
    
    for activity, result in all_results.items():
        status_symbol = "✓" if result['status'] == 'PASSED' else "✗"
        logger.info(f"{status_symbol} {activity.upper()}: {result['status']}")
        if result['issues']:
            for issue in result['issues']:
                logger.info(f"    - {issue}")
    
    # Final recommendation
    if passed_activities == total_activities:
        logger.info("\n🎉 ALL VERIFICATIONS PASSED - Data preparation pipeline is ready for NN training!")
    else:
        logger.info(f"\n⚠️  {total_activities - passed_activities} activities failed - Please address issues before proceeding")
    
    return all_results

if __name__ == "__main__":
    results = main()