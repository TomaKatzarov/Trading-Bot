#!/usr/bin/env python3
"""
Pre-compute and Save Training Dataset

This script performs one-time data preparation using the optimized NNDataPreparer
and saves the results to disk in an efficient format. This eliminates the need to
reprocess data for every training run, dramatically reducing training startup time.

Usage:
    python scripts/prepare_training_dataset.py --config path/to/config.yaml --output data/training_data

Author: Performance Optimization
Date: 2025-09-30
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_preparation_nn import NNDataPreparer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_prepared_data(prepared_data: dict, output_dir: str) -> None:
    """
    Save prepared data to disk in an efficient format.
    
    Args:
        prepared_data: Dictionary from NNDataPreparer.get_prepared_data_for_training()
        output_dir: Directory to save the data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving prepared data to {output_path}")
    
    # Save train data
    np.save(output_path / "train_X.npy", prepared_data['train']['X'])
    np.save(output_path / "train_y.npy", prepared_data['train']['y'])
    np.save(output_path / "train_asset_ids.npy", prepared_data['train']['asset_ids'])
    if 'sample_weights' in prepared_data['train']:
        np.save(output_path / "train_sample_weights.npy", prepared_data['train']['sample_weights'])
    
    # Save validation data
    np.save(output_path / "val_X.npy", prepared_data['validation']['X'])
    np.save(output_path / "val_y.npy", prepared_data['validation']['y'])
    np.save(output_path / "val_asset_ids.npy", prepared_data['validation']['asset_ids'])
    
    # Save test data
    np.save(output_path / "test_X.npy", prepared_data['test']['X'])
    np.save(output_path / "test_y.npy", prepared_data['test']['y'])
    np.save(output_path / "test_asset_ids.npy", prepared_data['test']['asset_ids'])
    
    # Save scalers
    joblib.dump(prepared_data['scalers'], output_path / "scalers.joblib")
    
    # Save asset ID mapping
    with open(output_path / "asset_id_mapping.json", 'w') as f:
        json.dump(prepared_data['asset_id_map'], f, indent=2)
    
    # Save metadata
    metadata = {
        'generation_timestamp': datetime.now().isoformat(),
        'train_samples': len(prepared_data['train']['X']),
        'val_samples': len(prepared_data['validation']['X']),
        'test_samples': len(prepared_data['test']['X']),
        'n_features': prepared_data['train']['X'].shape[2],
        'lookback_window': prepared_data['train']['X'].shape[1],
        'num_assets': len(prepared_data['asset_id_map']),
        'has_sample_weights': 'sample_weights' in prepared_data['train']
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Data saved successfully!")
    logger.info(f"  Train: {metadata['train_samples']} samples")
    logger.info(f"  Val:   {metadata['val_samples']} samples")
    logger.info(f"  Test:  {metadata['test_samples']} samples")
    logger.info(f"  Features: {metadata['n_features']}")
    logger.info(f"  Lookback window: {metadata['lookback_window']}")
    logger.info(f"  Assets: {metadata['num_assets']}")
    
    # Calculate and log size
    total_size_mb = sum(
        (output_path / f).stat().st_size 
        for f in output_path.iterdir() 
        if f.is_file()
    ) / (1024 * 1024)
    logger.info(f"  Total size: {total_size_mb:.2f} MB")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_default_config() -> dict:
    """Create a default configuration for data preparation."""
    return {
        'symbols_config_path': 'config/symbols.json',
        'feature_list': [
            'open', 'high', 'low', 'close', 'volume', 'vwap',
            'SMA_20', 'EMA_12', 'RSI_14', 'MACD_line', 'MACD_signal', 'MACD_hist',
            'BB_upper', 'BB_middle', 'BB_lower', 'BB_bandwidth',
            'DayOfWeek_sin', 'DayOfWeek_cos', 'sentiment_score_hourly_ffill'
        ],
        'lookback_window': 24,
        'prediction_horizon_hours': 8,
        'profit_target': 0.05,
        'stop_loss_target': 0.02,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'scaling_method': 'standard',
        'nan_handling_features': 'ffill',
        'calculate_sample_weights': True,
        'sample_weight_strategy': 'inverse_frequency',
        'data_base_path': 'data',
        'output_path_scalers': 'models/scalers.joblib'
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Pre-compute and save training dataset for faster training'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/training_data',
        help='Output directory for prepared data (default: data/training_data)'
    )
    parser.add_argument(
        '--symbols-list',
        type=str,
        nargs='+',
        help='List of symbols to process (optional, uses all symbols if not specified)'
    )
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    else:
        logger.info("Using default configuration")
        config = create_default_config()
    
    # Override symbols list if provided
    if args.symbols_list:
        config['symbols_list'] = args.symbols_list
        logger.info(f"Processing {len(args.symbols_list)} symbols: {args.symbols_list}")
    
    # Initialize data preparer
    logger.info("Initializing NNDataPreparer with optimized functions...")
    start_time = time.time()
    preparer = NNDataPreparer(config)
    
    # Prepare data
    logger.info("Starting data preparation (this may take a while for large datasets)...")
    prepared_data = preparer.get_prepared_data_for_training()
    
    preparation_time = time.time() - start_time
    logger.info(f"Data preparation completed in {preparation_time:.2f} seconds ({preparation_time/60:.2f} minutes)")
    
    # Save data
    save_prepared_data(prepared_data, args.output)
    
    total_time = time.time() - start_time
    logger.info(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"Prepared data saved to: {args.output}")
    logger.info("\nTo use this pre-generated data in training, set in your training config:")
    logger.info(f"  use_pregenerated_data: true")
    logger.info(f"  pregenerated_data_path: {args.output}")


if __name__ == '__main__':
    main()