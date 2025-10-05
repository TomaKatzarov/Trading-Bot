#!/usr/bin/env python3
"""
Verification script for Day of Week feature calculations
Tests the correctness of sine and cosine components for cyclical day-of-week encoding
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from core.feature_calculator import TechnicalIndicatorCalculator

def test_day_of_week_calculation():
    """Test Day of Week feature calculation with known timestamps"""
    print("Testing Day of Week feature calculations...")
    
    calculator = TechnicalIndicatorCalculator()
    
    # Create test timestamps for a full week
    # Start with Monday 2024-01-01 (which was actually a Monday)
    start_date = datetime(2024, 1, 1, 9, 0)  # Monday 9 AM
    test_timestamps = []
    
    for i in range(7):  # One week
        test_timestamps.append(start_date + timedelta(days=i))
    
    # Convert to DatetimeIndex
    timestamps = pd.DatetimeIndex(test_timestamps)
    
    # Calculate features
    dow_features = calculator.calculate_day_of_week_features(timestamps)
    
    print("\nTest Results:")
    print("Date\t\tDay\tDayOfWeek\tSin\t\tCos")
    print("-" * 70)
    
    expected_results = []
    for i, ts in enumerate(timestamps):
        day_name = ts.strftime('%A')
        day_of_week = ts.dayofweek  # 0=Monday, 6=Sunday
        sin_val = dow_features['day_of_week_sin'].iloc[i]
        cos_val = dow_features['day_of_week_cos'].iloc[i]
        
        # Calculate expected values
        expected_sin = math.sin(2 * math.pi * day_of_week / 7)
        expected_cos = math.cos(2 * math.pi * day_of_week / 7)
        
        print(f"{ts.strftime('%Y-%m-%d')}\t{day_name}\t{day_of_week}\t\t{sin_val:.6f}\t{cos_val:.6f}")
        
        expected_results.append({
            'day': day_name,
            'day_of_week': day_of_week,
            'expected_sin': expected_sin,
            'expected_cos': expected_cos,
            'actual_sin': sin_val,
            'actual_cos': cos_val
        })
    
    # Verify calculations
    print("\nVerification:")
    all_correct = True
    for result in expected_results:
        sin_diff = abs(result['actual_sin'] - result['expected_sin'])
        cos_diff = abs(result['actual_cos'] - result['expected_cos'])
        
        if sin_diff > 1e-10 or cos_diff > 1e-10:
            print(f"ERROR: {result['day']} - Sin diff: {sin_diff}, Cos diff: {cos_diff}")
            all_correct = False
    
    if all_correct:
        print("✓ All Day of Week calculations are correct!")
    else:
        print("✗ Some calculations are incorrect!")
    
    # Test cyclical properties
    print("\nTesting cyclical properties:")
    
    # Test that sin^2 + cos^2 = 1 for all values
    sin_squared_plus_cos_squared = dow_features['day_of_week_sin']**2 + dow_features['day_of_week_cos']**2
    if all(abs(val - 1.0) < 1e-10 for val in sin_squared_plus_cos_squared):
        print("✓ Cyclical property (sin²+cos²=1) verified for all values")
    else:
        print("✗ Cyclical property failed!")
    
    # Test specific known values
    print("\nTesting specific known values:")
    
    # Monday (0): sin(0) = 0, cos(0) = 1
    monday_sin = dow_features['day_of_week_sin'].iloc[0]
    monday_cos = dow_features['day_of_week_cos'].iloc[0]
    if abs(monday_sin - 0) < 1e-10 and abs(monday_cos - 1) < 1e-10:
        print("✓ Monday values correct (sin≈0, cos≈1)")
    else:
        print(f"✗ Monday values incorrect: sin={monday_sin}, cos={monday_cos}")
    
    # Wednesday (2): sin(4π/7), cos(4π/7)
    wednesday_sin = dow_features['day_of_week_sin'].iloc[2]
    wednesday_cos = dow_features['day_of_week_cos'].iloc[2]
    expected_wed_sin = math.sin(4 * math.pi / 7)
    expected_wed_cos = math.cos(4 * math.pi / 7)
    if abs(wednesday_sin - expected_wed_sin) < 1e-10 and abs(wednesday_cos - expected_wed_cos) < 1e-10:
        print("✓ Wednesday values correct")
    else:
        print(f"✗ Wednesday values incorrect")
    
    return all_correct

def test_with_real_data():
    """Test Day of Week features with real market data"""
    print("\n" + "="*70)
    print("Testing with real market data...")
    
    calculator = TechnicalIndicatorCalculator()
    
    # Try to find a symbol with data
    from pathlib import Path
    import json
    
    # Load symbols from config
    config_file = Path("config/symbols.json")
    if not config_file.exists():
        print("No symbols.json config file found")
        return False
    
    with open(config_file, 'r') as f:
        config = json.load(f)
        symbols = []
        for category, symbol_list in config.items():
            if isinstance(symbol_list, list):
                symbols.extend(symbol_list)
    
    # Find a symbol with data
    test_symbol = None
    for symbol in symbols[:10]:  # Test first 10 symbols
        symbol_dir = Path("data/historical") / symbol / "1Hour"
        data_file = symbol_dir / "data.parquet"
        if data_file.exists():
            test_symbol = symbol
            break
    
    if not test_symbol:
        print("No symbols with data found for testing")
        return False
    
    print(f"Testing with symbol: {test_symbol}")
    
    # Load data
    symbol_dir = Path("data/historical") / test_symbol / "1Hour"
    data_file = symbol_dir / "data.parquet"
    df = pd.read_parquet(data_file)
    
    if df.empty:
        print(f"No data found for {test_symbol}")
        return False
    
    print(f"Loaded {len(df)} rows of data")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Calculate Day of Week features
    dow_features = calculator.calculate_day_of_week_features(df.index)
    
    # Show sample of results
    print("\nSample Day of Week features:")
    sample_size = min(10, len(df))
    sample_indices = np.linspace(0, len(df)-1, sample_size, dtype=int)
    
    print("Date\t\t\tDay\tSin\t\tCos")
    print("-" * 60)
    
    for i in sample_indices:
        ts = df.index[i]
        day_name = ts.strftime('%A')
        sin_val = dow_features['day_of_week_sin'].iloc[i]
        cos_val = dow_features['day_of_week_cos'].iloc[i]
        print(f"{ts.strftime('%Y-%m-%d %H:%M')}\t{day_name}\t{sin_val:.6f}\t{cos_val:.6f}")
    
    # Verify no NaN values
    sin_nan_count = dow_features['day_of_week_sin'].isna().sum()
    cos_nan_count = dow_features['day_of_week_cos'].isna().sum()
    
    if sin_nan_count == 0 and cos_nan_count == 0:
        print("✓ No NaN values in Day of Week features")
    else:
        print(f"✗ Found NaN values: Sin={sin_nan_count}, Cos={cos_nan_count}")
    
    # Test range of values
    sin_min, sin_max = dow_features['day_of_week_sin'].min(), dow_features['day_of_week_sin'].max()
    cos_min, cos_max = dow_features['day_of_week_cos'].min(), dow_features['day_of_week_cos'].max()
    
    print(f"\nValue ranges:")
    print(f"Sin: [{sin_min:.6f}, {sin_max:.6f}]")
    print(f"Cos: [{cos_min:.6f}, {cos_max:.6f}]")
    
    # Values should be in [-1, 1]
    if -1 <= sin_min <= sin_max <= 1 and -1 <= cos_min <= cos_max <= 1:
        print("✓ All values are within expected range [-1, 1]")
    else:
        print("✗ Some values are outside expected range!")
    
    return True

def main():
    """Main verification function"""
    print("Day of Week Feature Verification Script")
    print("="*70)
    
    # Test 1: Known timestamp calculations
    test1_passed = test_day_of_week_calculation()
    
    # Test 2: Real data integration
    test2_passed = test_with_real_data()
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print(f"Known timestamp test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Real data test: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✓ All Day of Week feature tests PASSED!")
        print("Day of Week features are correctly calculated and ready for integration.")
    else:
        print("\n✗ Some tests FAILED!")
        print("Please review the implementation before proceeding.")

if __name__ == "__main__":
    main()