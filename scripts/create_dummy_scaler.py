#!/usr/bin/env python3
"""
Script to create a dummy scaler for backtesting integration testing.
"""

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

def create_dummy_scaler():
    """Create and save a dummy StandardScaler fitted on sample data."""
    
    # Create dummy data with the same number of features as our sample data
    # We have 17 base features (excluding OHLCV, VWAP, and asset ID)
    # Features: SMA_10, SMA_20, MACD_line, MACD_signal, MACD_hist, RSI_14,
    #          Stoch_K, Stoch_D, ADX_14, ATR_14, BB_bandwidth, OBV,
    #          Volume_SMA_20, Return_1h, sentiment_score_hourly_ffill,
    #          DayOfWeek_sin, DayOfWeek_cos
    
    num_features = 17
    num_samples = 100
    
    # Generate random data with realistic ranges for each feature type
    np.random.seed(42)  # For reproducible scaler
    
    dummy_data = np.random.normal(0, 1, (num_samples, num_features))
    
    # Adjust ranges to be more realistic for different feature types
    # Price-based features (SMA, MACD): around 150 (like our AAPL data)
    dummy_data[:, 0:5] = dummy_data[:, 0:5] * 10 + 150  # SMA_10, SMA_20, MACD components
    
    # Percentage-based features (RSI, Stoch): 0-100 range
    dummy_data[:, 5:8] = (dummy_data[:, 5:8] + 1) * 50  # RSI_14, Stoch_K, Stoch_D
    
    # ADX: 0-100 range
    dummy_data[:, 8] = (dummy_data[:, 8] + 1) * 50  # ADX_14
    
    # ATR: positive values around 1-5
    dummy_data[:, 9] = np.abs(dummy_data[:, 9]) * 2 + 1  # ATR_14
    
    # BB_bandwidth: small positive values
    dummy_data[:, 10] = np.abs(dummy_data[:, 10]) * 0.1  # BB_bandwidth
    
    # OBV: large numbers (volume-based)
    dummy_data[:, 11] = dummy_data[:, 11] * 1000000  # OBV
    
    # Volume_SMA_20: large positive numbers
    dummy_data[:, 12] = np.abs(dummy_data[:, 12]) * 1000000 + 500000  # Volume_SMA_20
    
    # Return_1h: small percentage changes
    dummy_data[:, 13] = dummy_data[:, 13] * 0.02  # Return_1h
    
    # Sentiment: 0-1 range
    dummy_data[:, 14] = (dummy_data[:, 14] + 1) / 2  # sentiment_score_hourly_ffill
    
    # Day of week features: -1 to 1 range (already in correct range)
    dummy_data[:, 15] = np.clip(dummy_data[:, 15], -1, 1)  # DayOfWeek_sin
    dummy_data[:, 16] = np.clip(dummy_data[:, 16], -1, 1)  # DayOfWeek_cos
    
    # Create and fit the scaler
    scaler = StandardScaler()
    scaler.fit(dummy_data)
    
    # Save the scaler
    output_path = "models/dummy_test_artifacts/dummy_scaler.joblib"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(scaler, output_path)
    
    print(f"Dummy scaler created and saved to: {output_path}")
    print(f"Scaler fitted on data shape: {dummy_data.shape}")
    print(f"Feature means: {scaler.mean_}")
    print(f"Feature scales: {scaler.scale_}")
    
    return scaler

if __name__ == "__main__":
    create_dummy_scaler()