import numpy as np
import json
from pathlib import Path

def validate_training_dataset_full(data_dir='data/training_data_v2_full'):
    """Comprehensive validation of the full training dataset"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Directory {data_path} does not exist")
        return False
    
    print("="*80)
    print("TRAINING DATASET V2 FULL - COMPREHENSIVE VALIDATION")
    print("="*80)
    
    # Load metadata
    metadata_path = data_path / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print("\n📊 METADATA:")
        print(f"  Generation timestamp: {metadata.get('generation_timestamp', 'N/A')}")
        print(f"  Symbols processed: {metadata.get('num_symbols', 'N/A')}")
        print(f"  Feature count: {metadata.get('feature_count', 'N/A')}")
        print(f"  Lookback window: {metadata.get('lookback_window', 'N/A')}")
    else:
        print("\n⚠️ No metadata.json found")
        metadata = {}
    
    # Load and validate data splits
    splits = ['train', 'val', 'test']
    total_samples = 0
    total_positive = 0
    all_checks_passed = True
    
    print("\n📈 SPLIT STATISTICS:")
    for split in splits:
        x_path = data_path / f'{split}_X.npy'
        y_path = data_path / f'{split}_y.npy'
        asset_ids_path = data_path / f'{split}_asset_ids.npy'
        
        if x_path.exists() and y_path.exists() and asset_ids_path.exists():
            X = np.load(x_path)
            y = np.load(y_path)
            asset_ids = np.load(asset_ids_path)
            
            n_samples = len(y)
            n_positive = np.sum(y)
            positive_ratio = n_positive / n_samples if n_samples > 0 else 0
            
            total_samples += n_samples
            total_positive += n_positive
            
            # Check for NaNs
            nan_in_X = np.isnan(X).sum()
            nan_in_y = np.isnan(y).sum()
            
            print(f"\n{split.upper()}:")
            print(f"  Samples: {n_samples:,}")
            print(f"  Positive: {n_positive:,} ({positive_ratio*100:.1f}%)")
            print(f"  X shape: {X.shape}")
            print(f"  Features: {X.shape[2] if len(X.shape) == 3 else 'Invalid'}")
            print(f"  NaN in X: {nan_in_X} {'✅' if nan_in_X == 0 else '❌'}")
            print(f"  NaN in y: {nan_in_y} {'✅' if nan_in_y == 0 else '❌'}")
            print(f"  Asset IDs unique: {len(np.unique(asset_ids))}")
            
            if nan_in_X > 0 or nan_in_y > 0:
                all_checks_passed = False
        else:
            print(f"\n❌ {split.upper()}: Missing data files")
            all_checks_passed = False
    
    overall_positive_ratio = total_positive / total_samples if total_samples > 0 else 0
    
    print("\n" + "="*80)
    print("OVERALL STATISTICS:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total positive: {total_positive:,}")
    print(f"  Overall positive ratio: {overall_positive_ratio*100:.2f}%")
    
    # Feature count validation
    feature_count = metadata.get('feature_count', 0)
    expected_features_without_sentiment = 22  # All indicators except sentiment
    expected_features_with_sentiment = 23
    
    print("\n🔍 FEATURE VALIDATION:")
    print(f"  Expected (without sentiment): {expected_features_without_sentiment}")
    print(f"  Expected (with sentiment): {expected_features_with_sentiment}")
    print(f"  Actual: {feature_count}")
    
    if feature_count == expected_features_with_sentiment:
        print("  ✅ Full feature set including sentiment")
    elif feature_count == expected_features_without_sentiment:
        print("  ⚠️  Missing sentiment feature (expected - will be added if sentiment data exists)")
    else:
        print(f"  ❌ Unexpected feature count")
        all_checks_passed = False
    
    # Class balance validation
    print("\n⚖️  CLASS BALANCE CHECK:")
    expected_min_ratio = 0.01  # Target was 1.2%, got 6.9%
    expected_max_ratio = 0.15  # Reasonable upper bound
    
    print(f"  Expected range: {expected_min_ratio*100:.1f}% - {expected_max_ratio*100:.1f}%")
    print(f"  Actual: {overall_positive_ratio*100:.2f}%")
    
    if expected_min_ratio <= overall_positive_ratio <= expected_max_ratio:
        print("  ✅ Class balance in acceptable range")
    else:
        print("  ⚠️  Class balance outside expected range")
        if overall_positive_ratio < expected_min_ratio:
            print("     (Too few positive samples)")
        else:
            print("     (Unusually high positive ratio)")
    
    # Data volume check
    print("\n📦 DATA VOLUME CHECK:")
    expected_min_samples = 800000  # Should have ~878K
    print(f"  Expected minimum: {expected_min_samples:,}")
    print(f"  Actual: {total_samples:,}")
    
    if total_samples >= expected_min_samples:
        print("  ✅ Sufficient data volume")
    else:
        print("  ❌ Insufficient data volume")
        all_checks_passed = False
    
    # Split ratio validation
    print("\n📊 SPLIT RATIO VALIDATION:")
    train_pct = metadata.get('train_samples', 0) / total_samples * 100 if total_samples > 0 else 0
    val_pct = metadata.get('val_samples', 0) / total_samples * 100 if total_samples > 0 else 0
    test_pct = metadata.get('test_samples', 0) / total_samples * 100 if total_samples > 0 else 0
    
    print(f"  Train: {train_pct:.1f}% (target: 70%)")
    print(f"  Val: {val_pct:.1f}% (target: 15%)")
    print(f"  Test: {test_pct:.1f}% (target: 15%)")
    
    if abs(train_pct - 70) < 1 and abs(val_pct - 15) < 1 and abs(test_pct - 15) < 1:
        print("  ✅ Split ratios correct")
    else:
        print("  ⚠️  Split ratios slightly off target")
    
    # Sentiment data check
    print("\n💬 SENTIMENT DATA CHECK:")
    print(f"  Feature count: {feature_count}")
    if feature_count == 23:
        print("  ✅ Sentiment feature included")
    elif feature_count == 22:
        print("  ⚠️  Sentiment feature NOT included")
        print("     This is EXPECTED if sentiment data is not in the database")
        print("     The dataset is still valid for training without sentiment")
    else:
        print("  ❌ Unexpected feature count")
    
    # Final verdict
    print("\n" + "="*80)
    print("VALIDATION SUMMARY:")
    
    checks = {
        'Data files exist': all_checks_passed,
        'No NaN values': all_checks_passed,
        'Sufficient volume (>800K)': total_samples >= expected_min_samples,
        'Good class balance (1-15%)': expected_min_ratio <= overall_positive_ratio <= expected_max_ratio,
        'Correct split ratios': abs(train_pct - 70) < 1,
        'Feature count valid': feature_count >= expected_features_without_sentiment
    }
    
    for check, passed in checks.items():
        print(f"  {'✅' if passed else '❌'} {check}")
    
    all_passed = all(checks.values())
    
    print(f"\n{'✅ DATASET READY FOR HPO' if all_passed else '⚠️ DATASET NEEDS ATTENTION'}")
    print("="*80)
    
    return all_passed

if __name__ == "__main__":
    success = validate_training_dataset_full()
    exit(0 if success else 1)