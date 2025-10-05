import numpy as np
import json
from pathlib import Path

def validate_training_data_v2(data_dir='data/training_data_v2'):
    """Validate the generated training dataset v2"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Directory {data_path} does not exist")
        return
    
    # Load metadata
    metadata_path = data_path / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print("Metadata loaded:")
        print(json.dumps(metadata, indent=2))
    else:
        print("No metadata.json found")
        metadata = {}
    
    # Load data splits
    splits = ['train', 'val', 'test']
    total_samples = 0
    total_positive = 0
    split_stats = {}
    
    for split in splits:
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
            
            split_stats[split] = {
                'samples': n_samples,
                'positive': n_positive,
                'positive_ratio': positive_ratio,
                'X_shape': X.shape,
                'features': X.shape[2] if len(X.shape) == 3 else 'Invalid shape'
            }
            
            print(f"{split.capitalize()}: {n_samples} samples, {n_positive} positive ({positive_ratio:.3f}), X shape: {X.shape}")
            
            # Check for NaNs
            nan_in_y = np.isnan(y).sum()
            if nan_in_y > 0:
                print(f"  WARNING: {nan_in_y} NaN values in y")
            
            if len(X.shape) == 3:
                nan_in_X = np.isnan(X).sum()
                if nan_in_X > 0:
                    print(f"  WARNING: {nan_in_X} NaN values in X")
        else:
            print(f"Missing files for {split}")
    
    overall_positive_ratio = total_positive / total_samples if total_samples > 0 else 0
    
    print("\nOverall Statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Total positive samples: {total_positive}")
    print(f"Overall positive ratio: {overall_positive_ratio:.3f} ({overall_positive_ratio*100:.1f}%)")
    print(f"Features per timestep: {split_stats.get('train', {}).get('features', 'N/A')}")
    print(f"Lookback window: {metadata.get('lookback_window', 'N/A')}")
    print(f"Symbols processed: {metadata.get('num_symbols', 'N/A')}")
    
    # Expected improvements check
    expected_old_ratio = 0.006
    expected_new_ratio = 0.012  # Target 1.2%
    actual_improvement = overall_positive_ratio / expected_old_ratio if expected_old_ratio > 0 else 0
    
    print(f"\nImprovement Analysis:")
    print(f"Old positive ratio (5% target): {expected_old_ratio*100:.1f}%")
    print(f"Target new ratio (2.5% target): {expected_new_ratio*100:.1f}%")
    print(f"Actual new ratio: {overall_positive_ratio*100:.1f}%")
    print(f"Improvement factor: {actual_improvement:.1f}x")
    
    if overall_positive_ratio > expected_new_ratio:
        print("✅ Positive class balance improved as expected")
    else:
        print("⚠️ Positive class balance may need further adjustment")
    
    # Data volume check
    expected_old_total = 872326
    data_increase = (total_samples - expected_old_total) / expected_old_total * 100 if expected_old_total > 0 else 0
    print(f"\nData Volume:")
    print(f"Old total samples: {expected_old_total:,}")
    print(f"New total samples: {total_samples:,}")
    print(f"Increase: {data_increase:.1f}% (expected ~25% from 5 months new data)")
    
    if data_increase > 0:
        print("✅ Data volume increased")
    else:
        print("⚠️ No data volume increase detected")
    
    # Feature check
    expected_features = 23
    actual_features = split_stats.get('train', {}).get('features', 0)
    print(f"\nFeature Quality:")
    print(f"Expected features: {expected_features}")
    print(f"Actual features: {actual_features}")
    if actual_features == expected_features:
        print("✅ Full feature set present")
    else:
        print(f"⚠️ Missing features: {expected_features - actual_features} (only {actual_features} found)")
        print("   Note: Run core/feature_calculator.py to add technical indicators if missing")
    
    return {
        'total_samples': total_samples,
        'positive_ratio': overall_positive_ratio,
        'feature_count': actual_features,
        'quality_passed': actual_features == expected_features and overall_positive_ratio > expected_new_ratio,
        'recommendations': []
    }

if __name__ == "__main__":
    results = validate_training_data_v2()
    print("\nValidation Summary:")
    if results['quality_passed']:
        print("✅ All critical checks passed")
    else:
        print("⚠️ Some checks failed - see recommendations")
    print(f"Ready for HPO: {'Yes' if results['quality_passed'] else 'No - fix features first'}")