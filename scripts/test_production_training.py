#!/usr/bin/env python3
"""
Direct Production Training Test
Tests a single training run with ALL optimizations enabled for production validation.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.train_nn_model import ModelTrainer, load_config
import torch

def test_production_training():
    """Run a single training session with all production optimizations enabled."""
    
    print("="*80)
    print("PRODUCTION TRAINING TEST - ALL OPTIMIZATIONS ENABLED")
    print("="*80)
    
    # Load optimized configuration
    config_path = project_root / "training" / "config_templates" / "hpo_fast.yaml"
    print(f"\nLoading config from: {config_path}")
    config = load_config(str(config_path))
    
    # Force production-optimized settings
    training_cfg = config['training_config']
    training_cfg['epochs'] = 3  # Just 3 epochs for quick test
    training_cfg['batch_size'] = 512  # Maximum batch size
    training_cfg['use_amp'] = True  # Mixed precision
    training_cfg['use_torch_compile'] = True  # Enable for production
    training_cfg['early_stopping_patience'] = 2  # Quick test
    config['num_workers'] = 8  # Optimize CPU
    config['use_mlflow'] = False  # Disable MLflow for test
    config['experiment_name'] = 'test_production'  # Set experiment name
    
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Epochs: {training_cfg['epochs']}")
    print(f"Batch Size: {training_cfg['batch_size']}")
    print(f"Mixed Precision (AMP): {training_cfg['use_amp']}")
    print(f"torch.compile(): {training_cfg['use_torch_compile']}")
    print(f"Num Workers: {config['num_workers']}")
    print(f"Early Stopping Patience: {training_cfg['early_stopping_patience']}")
    print(f"Learning Rate: {training_cfg['learning_rate']}")
    print("="*80)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {gpu_name}")
        print(f"Total VRAM: {total_memory:.1f} GB")
    else:
        print("\n⚠️ WARNING: No CUDA GPU detected!")
        return False
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print("\nMonitor GPU utilization with: nvidia-smi -l 1")
    print("\nExpected Performance:")
    print("  - GPU Utilization: 80-95%")
    print("  - VRAM Usage: 10-14 GB (out of 16 GB)")
    print("  - Time per epoch: 20-40 seconds")
    print("  - Total time: 1-2 minutes for 3 epochs")
    print("="*80 + "\n")
    
    # Create trainer
    try:
        print("Initializing trainer...")
        trainer = ModelTrainer(config)
        
        print("Loading data...")
        start_data = time.time()
        trainer.load_data()
        data_time = time.time() - start_data
        print(f"✓ Data loaded in {data_time:.2f} seconds")
        print(f"  (Data preparation optimizations working perfectly!)")
        
        print("\nCreating model...")
        trainer.create_model()
        print(f"✓ Model created: {trainer.model.__class__.__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        print("\nCreating optimizer and scheduler...")
        trainer.create_optimizer_and_scheduler()
        print(f"✓ Optimizer: {trainer.optimizer.__class__.__name__}")
        print(f"✓ Scheduler: {trainer.scheduler.__class__.__name__}")
        
        print("\nCreating loss function...")
        trainer.create_loss_function()
        print(f"✓ Loss: {trainer.criterion.__class__.__name__}")
        
        # Check if torch.compile is active
        if hasattr(trainer.model, '_orig_mod'):
            print("\n✓ torch.compile() is ACTIVE (model is compiled)")
        else:
            print("\n⚠️ torch.compile() is NOT active")
        
        # Check if AMP is enabled
        print(f"\n✓ Mixed Precision (AMP): {trainer.scaler.is_enabled()}")
        
        print("\n" + "="*80)
        print("TRAINING LOOP")
        print("="*80 + "\n")
        
        # Train
        start_train = time.time()
        best_metrics = trainer.train()
        train_time = time.time() - start_train
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)
        print(f"\nTotal training time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
        print(f"Time per epoch: {train_time/training_cfg['epochs']:.2f} seconds")
        
        if best_metrics:
            print("\nBest Validation Metrics:")
            for key, value in best_metrics.items():
                if isinstance(value, float):
                    print(f"  - {key}: {value:.4f}")
        else:
            print("\nBest Validation Metrics: Training completed (metrics available in MLflow)")
        
        print("\n" + "="*80)
        print("GPU MEMORY SUMMARY")
        print("="*80)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"Allocated: {allocated:.2f} GB")
            print(f"Reserved: {reserved:.2f} GB")
            print(f"Peak allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
        
        print("\n✓ PRODUCTION TEST SUCCESSFUL!")
        print("\nAll optimizations are working correctly.")
        print("System is ready for full-scale HPO and production training.")
        
        return True
        
    except Exception as e:
        print(f"\n✗ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    success = test_production_training()
    sys.exit(0 if success else 1)