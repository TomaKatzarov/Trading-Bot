#!/usr/bin/env python3
"""
Unified Training Data Update Script

This script performs all necessary steps to update and prepare training data:
1. Download new historical data
2. Add technical indicators to all symbols
3. Generate training dataset
4. Validate the dataset

Usage:
    python scripts/update_training_data.py --start-date 2025-05-29 --end-date 2025-10-01
    python scripts/update_training_data.py  # Uses default 2-year lookback

Created: 2025-10-02
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.hist_data_loader import HistoricalDataLoader
from core.feature_calculator import TechnicalIndicatorCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_symbols_with_data():
    """Get all symbols that have historical data files"""
    data_dir = Path('data/historical')
    symbols = []
    
    if data_dir.exists():
        for symbol_dir in data_dir.iterdir():
            if symbol_dir.is_dir():
                data_file = symbol_dir / '1Hour' / 'data.parquet'
                if data_file.exists():
                    symbols.append(symbol_dir.name)
    
    return sorted(symbols)

def step1_download_historical_data(start_date=None, end_date=None, years=2, workers=4):
    """Step 1: Download/update historical data"""
    logger.info("="*80)
    logger.info("STEP 1: DOWNLOADING HISTORICAL DATA")
    logger.info("="*80)
    
    loader = HistoricalDataLoader()
    
    if start_date and end_date:
        logger.info(f"Downloading data from {start_date} to {end_date} (append mode)")
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        loader.load_all_symbols(
            timeframe='hour',
            start_date=start_dt,
            end_date=end_dt,
            verbose=True,
            max_workers=workers,
            append=True
        )
    else:
        logger.info(f"Downloading {years} years of historical data")
        loader.load_all_symbols(
            timeframe='hour',
            years=years,
            verbose=True,
            max_workers=workers
        )
    
    logger.info("✅ Historical data download complete\n")

def step2_add_technical_indicators():
    """Step 2: Add technical indicators to all symbols"""
    logger.info("="*80)
    logger.info("STEP 2: ADDING TECHNICAL INDICATORS")
    logger.info("="*80)
    
    symbols = get_symbols_with_data()
    logger.info(f"Found {len(symbols)} symbols with historical data")
    
    if not symbols:
        logger.error("❌ No symbols found with historical data!")
        return False
    
    calculator = TechnicalIndicatorCalculator()
    
    results = {}
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"Processing {i}/{len(symbols)}: {symbol}")
        results[symbol] = calculator.process_symbol(symbol)
    
    successful = sum(1 for success in results.values() if success)
    failed = len(results) - successful
    
    logger.info(f"Technical indicators added: {successful}/{len(results)} successful")
    
    if failed > 0:
        failed_symbols = [sym for sym, success in results.items() if not success]
        logger.warning(f"Failed symbols: {failed_symbols}")
    
    logger.info("✅ Technical indicators complete\n")
    return successful > 0

def step3_generate_sentiment(start_date, end_date):
    """Step 3: Generate sentiment data for new period"""
    logger.info("="*80)
    logger.info("STEP 3: GENERATING SENTIMENT DATA")
    logger.info("="*80)
    logger.info(f"Date range: {start_date} to {end_date}")
    
    import subprocess
    
    try:
        result = subprocess.run(
            ['python', 'scripts/generate_sentiment_data.py',
             '--start-date', start_date,
             '--end-date', end_date],
            check=True,
            capture_output=False
        )
        logger.info("✅ Sentiment generation complete\n")
        return True
    except Exception as e:
        logger.error(f"❌ Sentiment generation failed: {e}")
        return False

def step4_attach_sentiment():
    """Step 4: Attach sentiment to hourly data"""
    logger.info("="*80)
    logger.info("STEP 4: ATTACHING SENTIMENT TO HOURLY DATA")
    logger.info("="*80)
    
    import subprocess
    
    try:
        result = subprocess.run(
            ['python', 'scripts/attach_sentiment_to_hourly.py'],
            check=True,
            capture_output=False
        )
        logger.info("✅ Sentiment attachment complete\n")
        return True
    except Exception as e:
        logger.error(f"❌ Sentiment attachment failed: {e}")
        return False

def step3_generate_training_dataset(output_dir, profit_target=0.025, lookback=24, stop_loss=0.02):
    """Step 3: Generate combined training dataset"""
    logger.info("="*80)
    logger.info("STEP 3: GENERATING TRAINING DATASET")
    logger.info("="*80)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Profit target: {profit_target*100}%, Stop loss: {stop_loss*100}%")
    logger.info(f"Lookback window: {lookback} hours")
    
    # Import here to avoid circular dependency
    from scripts.generate_combined_training_data import main as generate_main
    
    # Temporarily modify sys.argv to pass arguments
    original_argv = sys.argv
    sys.argv = [
        'generate_combined_training_data.py',
        '--output-dir', output_dir,
        '--profit-target', str(profit_target),
        '--lookback-window', str(lookback),
        '--stop-loss', str(stop_loss)
    ]
    
    try:
        generate_main()
        logger.info("✅ Training dataset generation complete\n")
        return True
    except Exception as e:
        logger.error(f"❌ Error generating training dataset: {e}")
        return False
    finally:
        sys.argv = original_argv

def step4_validate_dataset(data_dir):
    """Step 4: Validate the generated dataset"""
    logger.info("="*80)
    logger.info("STEP 4: VALIDATING DATASET")
    logger.info("="*80)
    
    import numpy as np
    import json
    
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"❌ Dataset directory {data_path} does not exist")
        return False
    
    # Load metadata
    metadata_path = data_path / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        logger.error("❌ metadata.json not found")
        return False
    
    # Validate all splits
    total_samples = 0
    total_positive = 0
    
    for split in ['train', 'val', 'test']:
        x_path = data_path / f'{split}_X.npy'
        y_path = data_path / f'{split}_y.npy'
        
        if x_path.exists() and y_path.exists():
            X = np.load(x_path)
            y = np.load(y_path)
            
            n_samples = len(y)
            n_positive = np.sum(y)
            positive_ratio = n_positive / n_samples if n_samples > 0 else 0
            
            total_samples += n_samples
            total_positive += n_positive
            
            logger.info(f"{split.upper()}: {n_samples:,} samples, {n_positive:,} positive ({positive_ratio*100:.1f}%), shape {X.shape}")
        else:
            logger.error(f"❌ Missing files for {split}")
            return False
    
    overall_positive_ratio = total_positive / total_samples if total_samples > 0 else 0
    feature_count = metadata.get('feature_count', 0)
    
    logger.info(f"\nOverall: {total_samples:,} samples, {overall_positive_ratio*100:.2f}% positive, {feature_count} features")
    
    # Quality checks
    checks_passed = (
        total_samples >= 800000 and
        0.01 <= overall_positive_ratio <= 0.15 and
        feature_count >= 22
    )
    
    if checks_passed:
        logger.info("✅ All validation checks passed - Dataset ready for HPO\n")
    else:
        logger.warning("⚠️ Some validation checks failed\n")
    
    return checks_passed

def main():
    parser = argparse.ArgumentParser(
        description='Unified script to update and prepare training data'
    )
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD) for data download')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD) for data download')
    parser.add_argument('--years', type=int, default=2, help='Years of historical data (if no start-date)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--output-dir', type=str, default='data/training_data_v2_full', help='Output directory for training data')
    parser.add_argument('--profit-target', type=float, default=0.025, help='Profit target (e.g., 0.025 for 2.5%)')
    parser.add_argument('--lookback', type=int, default=24, help='Lookback window in hours')
    parser.add_argument('--stop-loss', type=float, default=0.02, help='Stop loss (e.g., 0.02 for 2%)')
    parser.add_argument('--skip-download', action='store_true', help='Skip data download step')
    parser.add_argument('--skip-indicators', action='store_true', help='Skip technical indicators step')
    parser.add_argument('--skip-generation', action='store_true', help='Skip dataset generation step')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation step')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("UNIFIED TRAINING DATA UPDATE PIPELINE")
    logger.info("="*80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Profit target: {args.profit_target*100}%")
    logger.info(f"Lookback window: {args.lookback} hours")
    logger.info("="*80 + "\n")
    
    try:
        # Step 1: Download historical data
        if not args.skip_download:
            step1_download_historical_data(
                start_date=args.start_date,
                end_date=args.end_date,
                years=args.years,
                workers=args.workers
            )
        else:
            logger.info("⏭️  Skipping data download (--skip-download)\n")
        
        # Step 2: Add technical indicators
        if not args.skip_indicators:
            if not step2_add_technical_indicators():
                logger.error("❌ Technical indicators step failed")
                return False
        else:
            logger.info("⏭️  Skipping technical indicators (--skip-indicators)\n")
        
        # Step 3: Generate training dataset
        if not args.skip_generation:
            if not step3_generate_training_dataset(
                output_dir=args.output_dir,
                profit_target=args.profit_target,
                lookback=args.lookback,
                stop_loss=args.stop_loss
            ):
                logger.error("❌ Dataset generation failed")
                return False
        else:
            logger.info("⏭️  Skipping dataset generation (--skip-generation)\n")
        
        # Step 4: Validate dataset
        if not args.skip_validation:
            if not step4_validate_dataset(args.output_dir):
                logger.error("❌ Dataset validation failed")
                return False
        else:
            logger.info("⏭️  Skipping validation (--skip-validation)\n")
        
        # Success!
        logger.info("="*80)
        logger.info("✅ TRAINING DATA UPDATE PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Training data saved to: {args.output_dir}")
        logger.info("Next step: Create HPO configurations and run hyperparameter optimization")
        logger.info("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)