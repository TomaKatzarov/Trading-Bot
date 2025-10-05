#!/usr/bin/env python3
"""
Baseline Training Campaign - Phase 2
Trains all 4 neural network architectures with baseline configurations

This script orchestrates the training of:
1. MLP (Multi-Layer Perceptron) - Baseline feedforward
2. LSTM with Attention - Primary recurrent model
3. GRU with Attention - Alternative recurrent model
4. CNN-LSTM Hybrid - Exploratory architecture

Author: Roo (AI Assistant)
Date: 2025-09-30
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Model configurations mapping
MODELS = {
    'mlp': {
        'config': 'training/config_templates/mlp_baseline.yaml',
        'name': 'MLP Baseline',
        'description': 'Simple feedforward baseline'
    },
    'lstm': {
        'config': 'training/config_templates/lstm_baseline.yaml',
        'name': 'LSTM with Attention',
        'description': 'Primary recurrent model'
    },
    'gru': {
        'config': 'training/config_templates/gru_baseline.yaml',
        'name': 'GRU with Attention',
        'description': 'Alternative recurrent model'
    },
    'cnn_lstm': {
        'config': 'training/config_templates/cnn_lstm_baseline.yaml',
        'name': 'CNN-LSTM Hybrid',
        'description': 'Exploratory architecture'
    }
}

def print_header(text: str, char: str = "="):
    """Print formatted header"""
    print(f"\n{char * 80}")
    print(f"{text.center(80)}")
    print(f"{char * 80}\n")

def print_section(text: str):
    """Print section header"""
    print(f"\n{'─' * 80}")
    print(f"  {text}")
    print(f"{'─' * 80}\n")

def check_data_exists() -> Tuple[bool, Path]:
    """
    Check if training data exists, preferring v2_final (with sentiment + full features)
    
    Returns:
        Tuple[bool, Path]: (exists, data_path)
    """
    # Prefer v2_final (complete dataset with sentiment)
    data_paths = [
        Path("data/training_data_v2_final"),
        Path("data/training_data_v2_full"),
        Path("data/training_data_v2"),
        Path("data/training_data")
    ]
    
    required_files = [
        "train_X.npy",
        "train_y.npy",
        "train_asset_ids.npy",
        "val_X.npy",
        "val_y.npy",
        "val_asset_ids.npy",
        "test_X.npy",
        "test_y.npy",
        "test_asset_ids.npy",
        "scalers.joblib",
        "metadata.json"
    ]
    
    for data_path in data_paths:
        if data_path.exists():
            missing_files = [f for f in required_files if not (data_path / f).exists()]
            if not missing_files:
                print(f"✅ Using training data from: {data_path}")
                return True, data_path
    
    print(f"❌ No complete training data found")
    return False, Path("data/training_data")

def generate_training_data() -> bool:
    """Generate training data using the data preparation script"""
    print_section("Generating Training Data")
    
    print("Running: python scripts/generate_combined_training_data.py")
    
    try:
        result = subprocess.run(
            [sys.executable, "scripts/generate_combined_training_data.py"],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ Training data generation completed successfully")
            print(result.stdout)
            return True
        else:
            print(f"❌ Training data generation failed with return code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Training data generation timed out (30 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error generating training data: {e}")
        return False

def verify_data_quality(data_path: Path) -> Tuple[bool, Dict]:
    """Verify training data quality and return metadata"""
    print_section("Verifying Data Quality")
    
    metadata_path = data_path / "metadata.json"
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"✓ Symbols processed: {metadata.get('num_symbols', 0)}")
        print(f"✓ Total sequences: {metadata.get('total_sequences', 0):,}")
        print(f"✓ Train samples: {metadata.get('train_samples', 0):,}")
        print(f"✓ Val samples: {metadata.get('val_samples', 0):,}")
        print(f"✓ Test samples: {metadata.get('test_samples', 0):,}")
        print(f"✓ Features per sequence: {metadata.get('feature_count', 0)}")
        print(f"✓ Lookback window: {metadata.get('lookback_window', 0)} hours")
        print(f"✓ Positive ratio - Train: {metadata.get('positive_ratio_train', 0):.3f}")
        print(f"✓ Positive ratio - Val: {metadata.get('positive_ratio_val', 0):.3f}")
        print(f"✓ Positive ratio - Test: {metadata.get('positive_ratio_test', 0):.3f}")
        
        # Verify minimum requirements
        if metadata.get('train_samples', 0) < 1000:
            print("⚠️ Warning: Training samples < 1000, may not be sufficient")
        
        if metadata.get('num_symbols', 0) < 10:
            print("⚠️ Warning: Less than 10 symbols, diversity may be limited")
        
        return True, metadata
        
    except Exception as e:
        print(f"❌ Error verifying data quality: {e}")
        return False, {}

def train_model(model_name: str, model_config: Dict) -> Tuple[bool, float, Dict]:
    """
    Train a single model
    
    Returns:
        Tuple[bool, float, Dict]: (success, duration_hours, results)
    """
    print_header(f"Training {model_config['name']}", "=")
    print(f"Description: {model_config['description']}")
    print(f"Config: {model_config['config']}\n")
    
    # Get the actual data path
    _, data_path = check_data_exists()
    
    cmd = [
        sys.executable,
        "training/train_nn_model.py",
        "--config", model_config['config'],
        "--use-pregenerated",
        "--pregenerated-path", str(data_path)
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        duration = time.time() - start_time
        duration_hours = duration / 3600
        
        if result.returncode == 0:
            print(f"\n✅ {model_config['name']} training completed in {duration_hours:.2f} hours")
            return True, duration_hours, {'returncode': 0}
        else:
            print(f"\n❌ {model_config['name']} training failed with return code {result.returncode}")
            return False, duration_hours, {'returncode': result.returncode}
            
    except Exception as e:
        duration = time.time() - start_time
        duration_hours = duration / 3600
        print(f"\n❌ {model_config['name']} training failed with exception: {e}")
        return False, duration_hours, {'error': str(e)}

def print_summary(results: Dict[str, Tuple[bool, float, Dict]], total_duration: float, metadata: Dict):
    """Print final campaign summary"""
    print_header("BASELINE TRAINING CAMPAIGN COMPLETE", "=")
    
    print(f"Total Campaign Duration: {total_duration/3600:.2f} hours\n")
    
    print("Model Training Results:")
    print("-" * 80)
    
    successful = 0
    for model_name, (success, duration, info) in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        model_display = MODELS[model_name]['name']
        print(f"{model_display:25} {status:15} {duration:.2f}h")
        if success:
            successful += 1
    
    print("-" * 80)
    print(f"\n{successful}/{len(results)} models trained successfully")
    
    # Data summary
    print(f"\nData Summary:")
    print(f"  - Symbols: {metadata.get('num_symbols', 'N/A')}")
    print(f"  - Total sequences: {metadata.get('total_sequences', 'N/A'):,}")
    print(f"  - Train/Val/Test: {metadata.get('train_samples', 'N/A'):,} / {metadata.get('val_samples', 'N/A'):,} / {metadata.get('test_samples', 'N/A'):,}")
    
    # Next steps
    if successful == len(results):
        print("\n" + "=" * 80)
        print("✅ ALL BASELINE MODELS TRAINED SUCCESSFULLY")
        print("=" * 80)
        print("\nNext Steps:")
        print("  1. Review MLflow UI to compare model performance: http://localhost:5000")
        print("  2. Identify best performing architecture for HPO")
        print("  3. Proceed to Phase 3: Hyperparameter Optimization")
        print("  4. Run: python training/run_hpo.py --model-type <best_model>")
    else:
        print("\n" + "=" * 80)
        print("⚠️ SOME MODELS FAILED - REVIEW REQUIRED")
        print("=" * 80)
        print("\nAction Required:")
        print("  1. Review logs for failed models")
        print("  2. Check MLflow UI for partial results: http://localhost:5000")
        print("  3. Fix issues before proceeding to Phase 3")
        
    return successful == len(results)

def main():
    """Main orchestration function"""
    print_header("PHASE 2: BASELINE MODEL TRAINING CAMPAIGN", "=")
    
    print("Campaign Goals:")
    print("  ✓ Train 4 neural network architectures with baseline configurations")
    print("  ✓ Establish performance benchmarks for each architecture")
    print("  ✓ Identify best candidates for hyperparameter optimization")
    print("  ✓ Document baseline model performance for comparison\n")
    
    # Step 1: Verify or generate training data
    print_section("Step 1: Verify Training Data Availability")
    
    data_exists, data_path = check_data_exists()
    
    if data_exists:
        print(f"✅ Training data exists at: {data_path}")
        data_valid, metadata = verify_data_quality(data_path)
        if not data_valid:
            print("\n❌ Data quality verification failed")
            return 1
    else:
        print("❌ No training data found")
        print("\nPlease generate training data first:")
        print("  python scripts/update_training_data.py --start-date 2025-05-29 --end-date 2025-10-01")
        print("  OR")
        print("  python scripts/generate_combined_training_data.py --output-dir data/training_data_v2_final")
        return 1
    
    # Step 2: Train all models
    print_section("Step 2: Training All Model Architectures")
    
    results = {}
    total_start = time.time()
    
    for model_name, model_config in MODELS.items():
        success, duration, info = train_model(model_name, model_config)
        results[model_name] = (success, duration, info)
        
        # Brief pause between models
        if model_name != list(MODELS.keys())[-1]:
            time.sleep(5)
    
    total_duration = time.time() - total_start
    
    # Step 3: Print summary
    print_section("Step 3: Campaign Summary")
    all_successful = print_summary(results, total_duration, metadata)
    
    return 0 if all_successful else 1

if __name__ == "__main__":
    sys.exit(main())