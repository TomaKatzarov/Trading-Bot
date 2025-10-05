#!/usr/bin/env python3
"""
Script to create sample test data for backtesting integration testing using the existing pipeline.
Creates AAPL_1Hour_sample.parquet with 1 week of hourly data and processes it through the data pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.feature_calculator import TechnicalIndicatorCalculator

def create_base_ohlcv_data():
    """Create base OHLCV data for AAPL."""
    
    # Create 1 week of hourly data (5 trading days, 8 hours/day = 40 rows)
    start_date = datetime(2024, 1, 2, 9, 0)  # Tuesday 9 AM (trading day)
    timestamps = []
    
    # Generate 5 trading days, 8 hours each (9 AM to 4 PM)
    for day in range(5):
        base_date = start_date + timedelta(days=day)
        for hour in range(8):
            timestamps.append(base_date + timedelta(hours=hour))
    
    # Create base OHLCV data with some realistic variation
    np.random.seed(42)  # For reproducible data
    base_price = 150.0
    
    data = []
    current_price = base_price
    
    for i, timestamp in enumerate(timestamps):
        # Generate realistic OHLCV with some trend and volatility
        price_change = np.random.normal(0, 0.5)  # Small random changes
        current_price += price_change
        
        # Ensure realistic OHLC relationships
        open_price = current_price
        high_price = open_price + abs(np.random.normal(0, 0.3))
        low_price = open_price - abs(np.random.normal(0, 0.3))
        close_price = open_price + np.random.normal(0, 0.2)
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        volume = int(np.random.normal(1000000, 200000))  # Random volume
        volume = max(volume, 100000)  # Minimum volume
        
        vwap = (high_price + low_price + close_price) / 3  # Simple VWAP approximation
        
        current_price = close_price  # Update for next iteration
        
        data.append({
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume,
            'vwap': round(vwap, 2)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

def add_sentiment_data(df):
    """Add sentiment data to the DataFrame."""
    # Generate random sentiment values between 0.3 and 0.7 (moderate sentiment)
    np.random.seed(123)  # Different seed for sentiment
    sentiment_values = np.random.uniform(0.3, 0.7, len(df))
    df['sentiment_score_hourly_ffill'] = sentiment_values
    return df

def process_with_feature_calculator(df):
    """Process the data through the TechnicalIndicatorCalculator to add all technical indicators."""
    
    # Initialize the technical indicator calculator
    calculator = TechnicalIndicatorCalculator()
    
    # The calculator expects columns with capital letters (Open, High, Low, Close, Volume)
    # Let's rename our columns to match
    df_renamed = df.copy()
    df_renamed.columns = [col.title() if col.lower() in ['open', 'high', 'low', 'close', 'volume'] else col for col in df_renamed.columns]
    
    # Calculate all indicators
    df_with_features = calculator.calculate_all_indicators(df_renamed)
    
    # Rename columns back to lowercase for consistency
    df_with_features.columns = [col.lower() if col in ['Open', 'High', 'Low', 'Close', 'Volume'] else col for col in df_with_features.columns]
    
    return df_with_features

def create_directory_structure():
    """Create the necessary directory structure."""
    directories = [
        "data/sample_test_data",
        "data/historical/AAPL/1Hour",
        "models/dummy_test_artifacts"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """Main function to create and save sample data using the existing pipeline."""
    print("Creating sample AAPL hourly data using existing pipeline...")
    
    # Create directory structure
    create_directory_structure()
    
    # Step 1: Create base OHLCV data
    print("Step 1: Creating base OHLCV data...")
    df = create_base_ohlcv_data()
    print(f"Created base data with shape: {df.shape}")
    
    # Step 2: Add sentiment data
    print("Step 2: Adding sentiment data...")
    df = add_sentiment_data(df)
    
    # Step 3: Process through feature calculator to add technical indicators
    print("Step 3: Processing through FeatureCalculator...")
    try:
        df = process_with_feature_calculator(df)
        print(f"Processed data with shape: {df.shape}")
        print(f"Features added: {list(df.columns)}")
    except Exception as e:
        print(f"Error in feature calculation: {e}")
        print("Continuing with basic features...")
    
    # Step 4: Save to both locations
    # Save as sample data
    sample_path = "data/sample_test_data/AAPL_1Hour_sample.parquet"
    df.to_parquet(sample_path)
    print(f"Sample data saved to: {sample_path}")
    
    # Also save to historical data location for pipeline testing
    historical_path = "data/historical/AAPL/1Hour/data.parquet"
    df.to_parquet(historical_path)
    print(f"Historical data saved to: {historical_path}")
    
    # Step 5: Display summary
    print(f"\nData creation completed successfully!")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    
    # Display first few rows
    print("\nFirst 3 rows:")
    print(df.head(3))
    
    # Check for any NaN values
    nan_counts = df.isnull().sum()
    if nan_counts.sum() > 0:
        print(f"\nWarning: Found NaN values:")
        print(nan_counts[nan_counts > 0])
    else:
        print("\nNo NaN values found - data is clean!")
    
    # Verify we have the expected features from feature_set_NN.md
    expected_features = [
        'open', 'high', 'low', 'close', 'volume', 'vwap',
        'SMA_10', 'SMA_20', 'MACD_line', 'MACD_signal', 'MACD_hist',
        'RSI_14', 'Stoch_K', 'Stoch_D', 'ADX_14', 'ATR_14',
        'BB_bandwidth', 'OBV', 'Volume_SMA_20', 'Return_1h',
        'sentiment_score_hourly_ffill', 'DayOfWeek_sin', 'DayOfWeek_cos'
    ]
    
    missing_features = [f for f in expected_features if f not in df.columns]
    if missing_features:
        print(f"\nWarning: Missing expected features: {missing_features}")
    else:
        print(f"\nAll expected features present! ✓")
    
    return df

if __name__ == "__main__":
    main()