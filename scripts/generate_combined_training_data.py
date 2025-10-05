#!/usr/bin/env python3
"""
Generate Combined Training Data for All Available Symbols

This script generates and saves combined training data for all symbols that have
historical data available. The generated data can then be reused for training
different neural network models without regenerating it each time.

Created: 2025-05-28
Author: Roo (AI Assistant)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import joblib
import argparse

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

def get_available_symbols() -> List[str]:
    """
    Get list of symbols that have actual data files available.
    
    Returns:
        List[str]: List of available symbol names
    """
    data_dir = Path("data/historical")
    available_symbols = []
    
    if not data_dir.exists():
        logger.warning(f"Data directory {data_dir} does not exist")
        return available_symbols
    
    for symbol_dir in data_dir.iterdir():
        if symbol_dir.is_dir():
            data_file = symbol_dir / "1Hour" / "data.parquet"
            if data_file.exists():
                available_symbols.append(symbol_dir.name)
    
    logger.info(f"Found {len(available_symbols)} symbols with data: {available_symbols}")
    return available_symbols

def create_training_config(symbols_list: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create configuration for training data generation.
    
    Args:
        symbols_list: Optional list of symbols to process. If None, uses all available symbols.
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config = {
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
        'profit_target': 0.025,  # 2.5%
        'stop_loss': 0.02,      # 2%
        'train_ratio': 0.70,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'scaling_method': 'standard',
        'scaling_scope': 'global',
        'calculate_sample_weights': False,
        'data_base_path': 'data'
    }
    
    # Add symbols list if provided
    if symbols_list:
        config['symbols_list'] = symbols_list
        
    return config

def save_training_data(data_splits: Dict[str, Any], output_dir: str = "data/training_data") -> None:
    """
    Save the generated training data to disk.
    
    Args:
        data_splits: Dictionary containing train/val/test splits and metadata
        output_dir: Directory to save the data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the main data splits
    splits_to_save = ['train', 'val', 'test']
    
    for split in splits_to_save:
        if split in data_splits:
            split_data = data_splits[split]
            
            # Save features (X), labels (y), and asset_ids separately
            np.save(output_path / f"{split}_X.npy", split_data['X'])
            np.save(output_path / f"{split}_y.npy", split_data['y'])
            np.save(output_path / f"{split}_asset_ids.npy", split_data['asset_ids'])
            
            logger.info(f"Saved {split} data: X shape {split_data['X'].shape}, y shape {split_data['y'].shape}")
    
    # Save scalers
    if 'scalers' in data_splits:
        scalers_path = output_path / "scalers.joblib"
        joblib.dump(data_splits['scalers'], scalers_path)
        logger.info(f"Saved scalers to {scalers_path}")
    
    # Save asset ID mapping
    if 'asset_id_map' in data_splits:
        asset_map_path = output_path / "asset_id_mapping.json"
        with open(asset_map_path, 'w') as f:
            json.dump(data_splits['asset_id_map'], f, indent=2)
        logger.info(f"Saved asset ID mapping to {asset_map_path}")
    
    # Save metadata
    metadata = {
        'generation_timestamp': datetime.now().isoformat(),
        'num_symbols': len(data_splits.get('symbols_processed', [])),
        'symbols_processed': data_splits.get('symbols_processed', []),
        'feature_count': data_splits['train']['X'].shape[2] if 'train' in data_splits else 0,
        'lookback_window': data_splits.get('lookback_window', 24),
        'total_sequences': sum(data_splits[split]['X'].shape[0] for split in splits_to_save if split in data_splits),
        'train_samples': data_splits['train']['X'].shape[0] if 'train' in data_splits else 0,
        'val_samples': data_splits['val']['X'].shape[0] if 'val' in data_splits else 0,
        'test_samples': data_splits['test']['X'].shape[0] if 'test' in data_splits else 0,
        'positive_ratio_train': float(np.mean(data_splits['train']['y'])) if 'train' in data_splits else 0.0,
        'positive_ratio_val': float(np.mean(data_splits['val']['y'])) if 'val' in data_splits else 0.0,
        'positive_ratio_test': float(np.mean(data_splits['test']['y'])) if 'test' in data_splits else 0.0,
    }
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")
    
    logger.info(f"All training data saved to {output_path}")

def load_training_data(data_dir: str = "data/training_data") -> Dict[str, Any]:
    """
    Load previously generated training data from disk.
    
    Args:
        data_dir: Directory containing the saved training data
        
    Returns:
        Dict[str, Any]: Dictionary containing loaded data splits and metadata
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Training data directory {data_path} does not exist")
    
    # Load metadata first
    metadata_path = data_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loading training data generated on {metadata.get('generation_timestamp', 'unknown')}")
    else:
        metadata = {}
    
    # Load data splits
    data_splits = {'metadata': metadata}
    splits_to_load = ['train', 'val', 'test']
    
    for split in splits_to_load:
        X_path = data_path / f"{split}_X.npy"
        y_path = data_path / f"{split}_y.npy"
        asset_ids_path = data_path / f"{split}_asset_ids.npy"
        
        if X_path.exists() and y_path.exists() and asset_ids_path.exists():
            data_splits[split] = {
                'X': np.load(X_path),
                'y': np.load(y_path),
                'asset_ids': np.load(asset_ids_path)
            }
            logger.info(f"Loaded {split} data: X shape {data_splits[split]['X'].shape}, y shape {data_splits[split]['y'].shape}")
        else:
            logger.warning(f"Missing {split} data files in {data_path}")
    
    # Load scalers
    scalers_path = data_path / "scalers.joblib"
    if scalers_path.exists():
        data_splits['scalers'] = joblib.load(scalers_path)
        logger.info(f"Loaded scalers from {scalers_path}")
    
    # Load asset ID mapping
    asset_map_path = data_path / "asset_id_mapping.json"
    if asset_map_path.exists():
        with open(asset_map_path, 'r') as f:
            data_splits['asset_id_map'] = json.load(f)
        logger.info(f"Loaded asset ID mapping from {asset_map_path}")
    
    return data_splits

def main():
    """Main function to generate combined training data."""
    parser = argparse.ArgumentParser(description="Generate combined training data")
    parser.add_argument('--symbols', type=str, help='Path to symbols JSON file')
    parser.add_argument('--output-dir', type=str, default='data/training_data', help='Output directory for training data')
    parser.add_argument('--lookback-window', type=int, default=24, help='Lookback window size')
    parser.add_argument('--profit-target', type=float, default=0.025, help='Profit target (e.g., 0.025 for 2.5%)')
    parser.add_argument('--stop-loss', type=float, default=0.02, help='Stop loss (e.g., 0.02 for 2%)')
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("GENERATING COMBINED TRAINING DATA")
    logger.info(f"Output dir: {args.output_dir}, Profit target: {args.profit_target}, Lookback: {args.lookback_window}")
    logger.info("=" * 80)
    
    # Get available symbols or from file
    if args.symbols:
        with open(args.symbols, 'r') as f:
            symbols_data = json.load(f)
        # Assume 'crypto' key for complete list, or flatten all
        available_symbols = []
        if 'crypto' in symbols_data:
            for cat in symbols_data['crypto'].values():
                available_symbols.extend(cat)
        else:
            # Flatten all categories if no 'crypto'
            for cat in symbols_data.values():
                if isinstance(cat, dict):
                    for subcat in cat.values():
                        available_symbols.extend(subcat)
                else:
                    available_symbols.extend(cat)
        available_symbols = list(set(available_symbols))  # Dedup
    else:
        available_symbols = get_available_symbols()
    
    if not available_symbols:
        logger.error("No symbols found.")
        return
    
    logger.info(f"Processing {len(available_symbols)} symbols: {available_symbols[:5]}...")
    
    # Create configuration
    config = create_training_config(available_symbols)
    config['lookback_window'] = args.lookback_window
    config['profit_target'] = args.profit_target
    config['stop_loss'] = args.stop_loss
    
    # Initialize data preparer
    logger.info("Initializing NNDataPreparer...")
    data_preparer = NNDataPreparer(config)
    
    # Generate training data
    logger.info("Generating combined training data...")
    try:
        data_splits = data_preparer.get_prepared_data_for_training()
        
        # Add symbols processed to the results
        data_splits['symbols_processed'] = available_symbols
        data_splits['lookback_window'] = config['lookback_window']
        
        # Save the data
        logger.info("Saving training data...")
        save_training_data(data_splits, args.output_dir)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("TRAINING DATA GENERATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Symbols processed: {len(available_symbols)}")
        logger.info(f"Total sequences: {sum(data_splits[split]['X'].shape[0] for split in ['train', 'val', 'test'])}")
        logger.info(f"Features per sequence: {data_splits['train']['X'].shape[2]}")
        logger.info(f"Lookback window: {config['lookback_window']} hours")
        logger.info(f"Train samples: {data_splits['train']['X'].shape[0]}")
        logger.info(f"Val samples: {data_splits['val']['X'].shape[0]}")
        logger.info(f"Test samples: {data_splits['test']['X'].shape[0]}")
        logger.info(f"Positive ratio - Train: {np.mean(data_splits['train']['y']):.3f}, Val: {np.mean(data_splits['val']['y']):.3f}, Test: {np.mean(data_splits['test']['y']):.3f}")
        logger.info("Data saved to: data/training_data/")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error generating training data: {e}")
        raise

if __name__ == "__main__":
    main()