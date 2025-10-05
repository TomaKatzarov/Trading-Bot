# scripts/test_feature_calculator.py
"""
Simple test script for the feature calculator
Tests with sample data to ensure basic functionality
"""

import logging
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.feature_calculator import TechnicalIndicatorCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(n_periods: int = 100) -> pd.DataFrame:
    """
    Create sample OHLCV data for testing
    """
    # Generate sample price data with some realistic patterns
    np.random.seed(42)  # For reproducible results
    
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='H')
    
    # Start with a base price and add random walk
    base_price = 100.0
    price_changes = np.random.normal(0, 0.02, n_periods)  # 2% volatility
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # Ensure positive prices
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.01))  # Intraday volatility
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        
        # Open is previous close with small gap
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, 0.005)  # Small overnight gap
            open_price = prices[i-1] * (1 + gap)
        
        # Volume with some correlation to price movement
        base_volume = 10000
        volume_multiplier = 1 + abs(price_changes[i]) * 5  # Higher volume on big moves
        volume = int(base_volume * volume_multiplier * (1 + np.random.normal(0, 0.3)))
        
        data.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df


def test_basic_functionality():
    """Test basic functionality of the calculator"""
    logger.info("Testing basic functionality...")
    
    # Create sample data
    df = create_sample_data(100)
    logger.info(f"Created sample data with {len(df)} rows")
    
    # Initialize calculator
    calculator = TechnicalIndicatorCalculator()
    
    # Calculate indicators
    try:
        result_df = calculator.calculate_all_indicators(df)
        logger.info(f"Successfully calculated indicators. Result shape: {result_df.shape}")
        
        # Check that all expected columns are present
        expected_indicators = [
            'SMA_10', 'SMA_20', 'MACD_Line', 'MACD_Signal', 'MACD_Histogram',
            'RSI_14', 'Stoch_K', 'Stoch_D', 'ADX_14', 'ATR_14',
            'Bollinger_Bandwidth', 'OBV', 'Volume_SMA_20', 'Return_1h'
        ]
        
        missing_indicators = []
        present_indicators = []
        
        for indicator in expected_indicators:
            if indicator in result_df.columns:
                present_indicators.append(indicator)
            else:
                missing_indicators.append(indicator)
        
        logger.info(f"Present indicators ({len(present_indicators)}): {present_indicators}")
        if missing_indicators:
            logger.warning(f"Missing indicators ({len(missing_indicators)}): {missing_indicators}")
        
        return len(missing_indicators) == 0, result_df
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return False, None


def test_data_quality(df: pd.DataFrame):
    """Test the quality of calculated indicators"""
    logger.info("Testing data quality...")
    
    expected_indicators = [
        'SMA_10', 'SMA_20', 'MACD_Line', 'MACD_Signal', 'MACD_Histogram',
        'RSI_14', 'Stoch_K', 'Stoch_D', 'ADX_14', 'ATR_14',
        'Bollinger_Bandwidth', 'OBV', 'Volume_SMA_20', 'Return_1h'
    ]
    
    quality_report = {}
    
    for indicator in expected_indicators:
        if indicator in df.columns:
            series = df[indicator]
            
            quality_report[indicator] = {
                'total_values': len(series),
                'non_null_values': series.notna().sum(),
                'null_values': series.isna().sum(),
                'null_percentage': (series.isna().sum() / len(series)) * 100,
                'min_value': series.min() if series.notna().any() else None,
                'max_value': series.max() if series.notna().any() else None,
                'mean_value': series.mean() if series.notna().any() else None,
                'std_value': series.std() if series.notna().any() else None
            }
            
            # Check for infinite values
            if series.notna().any():
                inf_count = np.isinf(series).sum()
                quality_report[indicator]['infinite_values'] = inf_count
    
    # Print quality report
    logger.info("\nData Quality Report:")
    logger.info("-" * 60)
    
    for indicator, stats in quality_report.items():
        logger.info(f"{indicator}:")
        logger.info(f"  Non-null: {stats['non_null_values']}/{stats['total_values']} "
                   f"({100 - stats['null_percentage']:.1f}%)")
        
        if stats['min_value'] is not None:
            logger.info(f"  Range: {stats['min_value']:.4f} to {stats['max_value']:.4f}")
            logger.info(f"  Mean: {stats['mean_value']:.4f}, Std: {stats['std_value']:.4f}")
        
        if stats.get('infinite_values', 0) > 0:
            logger.warning(f"  WARNING: {stats['infinite_values']} infinite values detected!")
    
    return quality_report


def test_mathematical_properties(df: pd.DataFrame):
    """Test mathematical properties of indicators"""
    logger.info("\nTesting mathematical properties...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: RSI bounds (0-100)
    total_tests += 1
    if 'RSI_14' in df.columns:
        rsi = df['RSI_14'].dropna()
        if len(rsi) > 0:
            rsi_valid = (rsi >= 0).all() and (rsi <= 100).all()
            if rsi_valid:
                tests_passed += 1
                logger.info("✓ RSI bounds test passed (0-100)")
            else:
                logger.warning(f"✗ RSI bounds test failed. Range: {rsi.min():.2f} to {rsi.max():.2f}")
    
    # Test 2: Stochastic bounds (0-100)
    total_tests += 1
    if 'Stoch_K' in df.columns and 'Stoch_D' in df.columns:
        stoch_k = df['Stoch_K'].dropna()
        stoch_d = df['Stoch_D'].dropna()
        if len(stoch_k) > 0 and len(stoch_d) > 0:
            stoch_valid = ((stoch_k >= 0).all() and (stoch_k <= 100).all() and 
                          (stoch_d >= 0).all() and (stoch_d <= 100).all())
            if stoch_valid:
                tests_passed += 1
                logger.info("✓ Stochastic bounds test passed (0-100)")
            else:
                logger.warning("✗ Stochastic bounds test failed")
    
    # Test 3: MACD consistency (Histogram = Line - Signal)
    total_tests += 1
    if all(col in df.columns for col in ['MACD_Line', 'MACD_Signal', 'MACD_Histogram']):
        macd_line = df['MACD_Line'].dropna()
        macd_signal = df['MACD_Signal'].dropna()
        macd_hist = df['MACD_Histogram'].dropna()
        
        # Find common indices
        common_idx = macd_line.index.intersection(macd_signal.index).intersection(macd_hist.index)
        if len(common_idx) > 10:
            calculated_hist = macd_line.loc[common_idx] - macd_signal.loc[common_idx]
            actual_hist = macd_hist.loc[common_idx]
            max_diff = np.abs(calculated_hist - actual_hist).max()
            
            if max_diff < 1e-10:
                tests_passed += 1
                logger.info("✓ MACD consistency test passed")
            else:
                logger.warning(f"✗ MACD consistency test failed. Max difference: {max_diff}")
    
    # Test 4: SMA smoothness (SMA should be less volatile than price)
    total_tests += 1
    if 'SMA_20' in df.columns:
        price_volatility = df['Close'].std()
        sma_volatility = df['SMA_20'].dropna().std()
        
        if sma_volatility < price_volatility:
            tests_passed += 1
            logger.info("✓ SMA smoothness test passed")
        else:
            logger.warning("✗ SMA smoothness test failed")
    
    # Test 5: Returns calculation
    total_tests += 1
    if 'Return_1h' in df.columns:
        our_returns = df['Return_1h'].dropna()
        manual_returns = df['Close'].pct_change().dropna()
        
        # Find common indices
        common_idx = our_returns.index.intersection(manual_returns.index)
        if len(common_idx) > 10:
            max_diff = np.abs(our_returns.loc[common_idx] - manual_returns.loc[common_idx]).max()
            
            if max_diff < 1e-10:
                tests_passed += 1
                logger.info("✓ Returns calculation test passed")
            else:
                logger.warning(f"✗ Returns calculation test failed. Max difference: {max_diff}")
    
    logger.info(f"\nMathematical properties test summary: {tests_passed}/{total_tests} tests passed")
    return tests_passed, total_tests


def main():
    """Main test execution"""
    logger.info("Starting Feature Calculator Tests")
    logger.info("=" * 50)
    
    # Test 1: Basic functionality
    success, result_df = test_basic_functionality()
    
    if not success:
        logger.error("Basic functionality test failed. Stopping tests.")
        return False
    
    # Test 2: Data quality
    quality_report = test_data_quality(result_df)
    
    # Test 3: Mathematical properties
    tests_passed, total_tests = test_mathematical_properties(result_df)
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Basic functionality: {'PASS' if success else 'FAIL'}")
    logger.info(f"Mathematical properties: {tests_passed}/{total_tests} tests passed")
    
    # Check for any major issues
    major_issues = []
    for indicator, stats in quality_report.items():
        if stats['null_percentage'] > 50:
            major_issues.append(f"{indicator} has {stats['null_percentage']:.1f}% null values")
        if stats.get('infinite_values', 0) > 0:
            major_issues.append(f"{indicator} has infinite values")
    
    if major_issues:
        logger.warning("Major issues detected:")
        for issue in major_issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("No major data quality issues detected")
    
    overall_success = success and (tests_passed >= total_tests * 0.8) and len(major_issues) == 0
    logger.info(f"\nOverall test result: {'PASS' if overall_success else 'FAIL'}")
    
    return overall_success


if __name__ == "__main__":
    main()