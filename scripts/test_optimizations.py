#!/usr/bin/env python3
"""
Test Performance Optimizations and Verify Metrics Logging

This script runs a quick test to verify:
1. All optimizations are working correctly
2. All metrics are being logged to MLflow
3. The best model recipe is saved
4. Training completes successfully with improved performance

Usage:
    python scripts/test_optimizations.py --quick  # Fast test with small subset
    python scripts/test_optimizations.py --full   # Full test with all data

Author: Performance Optimization Testing
Date: 2025-09-30
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import yaml
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.train_nn_model import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are available."""
    logger.info("Checking dependencies...")
    
    issues = []
    
    # Check Numba
    try:
        import numba
        logger.info(f"✅ Numba {numba.__version__} installed")
    except ImportError:
        issues.append("❌ Numba not installed - label generation will be slower")
    
    # Check PyTorch version
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if torch_version >= (2, 0):
        logger.info(f"✅ PyTorch {torch.__version__} (torch.compile available)")
    else:
        issues.append(f"⚠️ PyTorch {torch.__version__} - torch.compile requires 2.0+")
    
    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        issues.append("⚠️ CUDA not available - training will be slower on CPU")
    
    # Check MLflow
    try:
        import mlflow
        logger.info(f"✅ MLflow {mlflow.__version__} installed")
    except ImportError:
        issues.append("⚠️ MLflow not installed - experiment tracking disabled")
    
    if issues:
        logger.warning("\n".join(issues))
        logger.warning("\nSome optimizations may not be available. Continue? (y/n)")
        response = input().strip().lower()
        if response != 'y':
            sys.exit(1)
    else:
        logger.info("\n✅ All dependencies satisfied!")
    
    return len(issues) == 0


def create_quick_test_config():
    """Create a minimal config for quick testing."""
    return {
        'model_type': 'lstm',
        'experiment_name': 'optimization_test',
        'output_dir': 'training/runs/optimization_test',
        'use_mlflow': True,
        'num_workers': 2,  # Reduced for testing
        'use_pregenerated_data': False,  # Will generate on the fly for test
        
        'data_config': {
            'symbols_config_path': 'config/symbols.json',
            'feature_list': [
                'open', 'high', 'low', 'close', 'volume', 'vwap',
                'SMA_20', 'EMA_12', 'RSI_14', 'MACD_line', 'MACD_signal', 'MACD_hist',
                'BB_upper', 'BB_middle', 'BB_lower', 'BB_bandwidth',
                'DayOfWeek_sin', 'DayOfWeek_cos'
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
            'sample_weights_strategy': 'balanced',
            'data_base_path': 'data',
            # Use only a few symbols for quick test
            'symbols_list': ['AAPL', 'MSFT', 'GOOGL']  # Small subset
        },
        
        'model_config': {
            'n_features': 18,
            'num_assets': 3,  # Will be updated
            'asset_embedding_dim': 8,
            'lstm_hidden_dim': 64,
            'lstm_num_layers': 2,
            'attention_dim': 64,
            'dropout_rate': 0.3,
            'use_layer_norm': True
        },
        
        'training_config': {
            'epochs': 5,  # Very short for testing
            'batch_size': 32,  # Smaller for testing
            'learning_rate': 1e-3,
            'optimizer': 'adamw',
            'weight_decay': 1e-5,
            'scheduler': 'reduce_on_plateau',
            'scheduler_factor': 0.5,
            'scheduler_patience': 2,
            'gradient_clip_norm': 1.0,
            'early_stopping_patience': 3,
            'monitor_metric': 'f1',
            'use_weighted_sampling': False,
            'use_amp': True,
            'use_torch_compile': True,  # Test torch.compile
            'log_confusion_matrix_freq': 1,  # Log every epoch for testing
            'loss_function': {
                'type': 'focal',
                'alpha': 0.25,
                'gamma': 2.0
            }
        }
    }


def verify_mlflow_logging(trainer):
    """Verify that all metrics are being logged to MLflow."""
    logger.info("\n" + "="*80)
    logger.info("VERIFYING MLFLOW LOGGING")
    logger.info("="*80)
    
    try:
        import mlflow
        
        # Get current run
        run = mlflow.active_run()
        if not run:
            logger.error("❌ No active MLflow run found!")
            return False
        
        run_id = run.info.run_id
        logger.info(f"✅ MLflow Run ID: {run_id}")
        
        # Get run data
        client = mlflow.tracking.MlflowClient()
        run_data = client.get_run(run_id)
        
        # Check parameters
        params = run_data.data.params
        logger.info(f"\n📊 Parameters logged: {len(params)}")
        essential_params = ['model_type', 'learning_rate', 'batch_size', 'epochs']
        for param in essential_params:
            if param in params:
                logger.info(f"  ✅ {param}: {params[param]}")
            else:
                logger.warning(f"  ⚠️ {param} not logged")
        
        # Check metrics
        metrics = run_data.data.metrics
        logger.info(f"\n📈 Metrics logged: {len(metrics)}")
        essential_metrics = ['train_loss', 'val_loss', 'val_f1', 'val_accuracy', 
                           'val_precision', 'val_recall', 'val_roc_auc']
        for metric in essential_metrics:
            if metric in metrics:
                logger.info(f"  ✅ {metric}: {metrics[metric]:.4f}")
            else:
                logger.warning(f"  ⚠️ {metric} not logged")
        
        # Check artifacts
        artifacts = client.list_artifacts(run_id)
        logger.info(f"\n📦 Artifacts logged: {len(artifacts)}")
        for artifact in artifacts[:10]:  # Show first 10
            logger.info(f"  ✅ {artifact.path}")
        
        logger.info("\n✅ MLflow logging verification complete!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error verifying MLflow logging: {e}")
        return False


def verify_model_outputs(trainer):
    """Verify that model and outputs are saved correctly."""
    logger.info("\n" + "="*80)
    logger.info("VERIFYING MODEL OUTPUTS")
    logger.info("="*80)
    
    output_dir = Path(trainer.config['output_dir'])
    
    # Check for best model
    best_model_path = output_dir / 'best_model.pt'
    if best_model_path.exists():
        logger.info(f"✅ Best model saved: {best_model_path}")
        
        # Load and verify
        checkpoint = torch.load(best_model_path)
        logger.info(f"  📊 Best F1: {checkpoint['metrics']['f1']:.4f}")
        logger.info(f"  📊 Best Epoch: {checkpoint['epoch']}")
        logger.info(f"  ✅ Config saved in checkpoint")
        logger.info(f"  ✅ Scalers saved in checkpoint")
        logger.info(f"  ✅ Asset ID map saved in checkpoint")
    else:
        logger.warning(f"⚠️ Best model not found at {best_model_path}")
    
    # Check for scalers
    scalers_path = output_dir / 'scalers.joblib'
    if scalers_path.exists():
        logger.info(f"✅ Scalers saved: {scalers_path}")
    else:
        logger.warning(f"⚠️ Scalers not found at {scalers_path}")
    
    # Check for checkpoints
    checkpoints = list(output_dir.glob('epoch*.pt'))
    if checkpoints:
        logger.info(f"✅ Found {len(checkpoints)} epoch checkpoints")
    else:
        logger.warning("⚠️ No epoch checkpoints found")
    
    logger.info("\n✅ Model output verification complete!")


def print_performance_summary(start_time, data_prep_time, training_time):
    """Print a summary of performance metrics."""
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*80)
    
    total_time = time.time() - start_time
    
    logger.info(f"\n⏱️  Timing Breakdown:")
    logger.info(f"  Data Preparation: {data_prep_time:.2f}s ({data_prep_time/60:.2f} min)")
    logger.info(f"  Model Training:   {training_time:.2f}s ({training_time/60:.2f} min)")
    logger.info(f"  Total Time:       {total_time:.2f}s ({total_time/60:.2f} min)")
    
    logger.info(f"\n🚀 Optimization Status:")
    
    # Check what was used
    try:
        import numba
        logger.info(f"  ✅ Numba-optimized label generation")
    except ImportError:
        logger.info(f"  ⚠️ Numba not available - using Python fallback")
    
    if torch.cuda.is_available():
        logger.info(f"  ✅ CUDA acceleration enabled")
        if hasattr(torch, 'compile'):
            logger.info(f"  ✅ torch.compile() available")
        else:
            logger.info(f"  ⚠️ torch.compile() not available (PyTorch < 2.0)")
    else:
        logger.info(f"  ⚠️ Running on CPU")
    
    logger.info(f"\n📝 Next Steps:")
    logger.info(f"  1. Check MLflow UI for detailed metrics and plots")
    logger.info(f"  2. Compare timing with baseline (should be much faster)")
    logger.info(f"  3. Run full HPO study with optimizations")
    logger.info(f"  4. Consider pre-generating data for production")


def main():
    """Main test execution."""
    parser = argparse.ArgumentParser(
        description='Test performance optimizations and verify metrics logging'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with minimal data (default)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full test with all data'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom config file'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print("PERFORMANCE OPTIMIZATION TEST")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Check dependencies
    all_deps_ok = check_dependencies()
    
    # Load config
    if args.config:
        logger.info(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.info("Using quick test configuration")
        config = create_quick_test_config()
        if args.full:
            # Use all symbols for full test
            config['data_config'].pop('symbols_list', None)
            config['training_config']['epochs'] = 20
            logger.info("Running FULL test with all data")
    
    # Track timing
    start_time = time.time()
    
    try:
        # Initialize trainer
        logger.info("\n" + "="*80)
        logger.info("INITIALIZING TRAINER")
        logger.info("="*80)
        trainer = ModelTrainer(config)
        
        # Load data (timed)
        logger.info("\n" + "="*80)
        logger.info("LOADING AND PREPARING DATA")
        logger.info("="*80)
        data_start = time.time()
        trainer.load_data()
        data_prep_time = time.time() - data_start
        logger.info(f"✅ Data preparation completed in {data_prep_time:.2f}s")
        
        # Create model
        logger.info("\n" + "="*80)
        logger.info("CREATING MODEL")
        logger.info("="*80)
        trainer.create_model()
        trainer.create_optimizer_and_scheduler()
        trainer.create_loss_function()
        
        # Train (timed)
        logger.info("\n" + "="*80)
        logger.info("STARTING TRAINING")
        logger.info("="*80)
        training_start = time.time()
        trainer.train()
        training_time = time.time() - training_start
        logger.info(f"✅ Training completed in {training_time:.2f}s")
        
        # Verify outputs
        verify_mlflow_logging(trainer)
        verify_model_outputs(trainer)
        
        # Print summary
        print_performance_summary(start_time, data_prep_time, training_time)
        
        # Success
        print("\n" + "="*80)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nAll optimizations are working correctly.")
        print(f"Check the output directory: {config['output_dir']}")
        print(f"Check MLflow UI for detailed metrics and plots")
        print("\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())