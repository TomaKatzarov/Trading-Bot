# core/feature_calculator.py
"""
Feature Calculator for Neural Network Training Data Preparation
Calculates all 14 technical indicators specified in feature_set_NN.md
"""

import logging
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalIndicatorCalculator:
    """
    Calculates technical indicators for 1-hour bar data as specified in feature_set_NN.md
    """
    
    def __init__(self, data_dir: str = 'data/historical'):
        self.data_dir = Path(data_dir)
        
    def calculate_sma(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=window, min_periods=1).mean()
    
    def calculate_ema(self, prices: pd.Series, span: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=span, adjust=False).mean()
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD Line, Signal Line, and Histogram
        Returns dict with 'macd_line', 'macd_signal', 'macd_histogram'
        """
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        macd_signal = self.calculate_ema(macd_line, signal)
        macd_histogram = macd_line - macd_signal
        
        return {
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        }
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rs = rs.fillna(100)  # When loss is zero, assign RSI of 100
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator %K and %D
        Returns dict with 'stoch_k' and 'stoch_d'
        """
        lowest_low = low.rolling(window=k_window, min_periods=1).min()
        highest_high = high.rolling(window=k_window, min_periods=1).max()
        
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_k = stoch_k.fillna(50)  # Fill NaN with neutral value
        
        stoch_d = stoch_k.rolling(window=d_window, min_periods=1).mean()
        
        return {
            'stoch_k': stoch_k,
            'stoch_d': stoch_d
        }
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), 
                          np.maximum(high - high.shift(1), 0), 0)
        dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), 
                           np.maximum(low.shift(1) - low, 0), 0)
        
        dm_plus = pd.Series(dm_plus, index=high.index)
        dm_minus = pd.Series(dm_minus, index=high.index)
        
        # Smoothed values
        tr_smooth = tr.rolling(window=window, min_periods=1).mean()
        dm_plus_smooth = dm_plus.rolling(window=window, min_periods=1).mean()
        dm_minus_smooth = dm_minus.rolling(window=window, min_periods=1).mean()
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        dx = dx.fillna(0)
        adx = dx.rolling(window=window, min_periods=1).mean()
        
        return adx
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=window, min_periods=1).mean()
        return atr
    
    def calculate_bollinger_bandwidth(self, prices: pd.Series, window: int = 20, std_dev: float = 2.0) -> pd.Series:
        """Calculate Bollinger Bandwidth: (Upper Band - Lower Band) / Middle Band"""
        sma = self.calculate_sma(prices, window)
        std = prices.rolling(window=window, min_periods=1).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        bandwidth = (upper_band - lower_band) / sma
        return bandwidth.fillna(0)
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        price_change = close.diff()
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def calculate_day_of_week_features(self, timestamps: pd.DatetimeIndex) -> Dict[str, pd.Series]:
        """
        Calculate Day of Week sine and cosine components from timestamps
        Monday=0, Tuesday=1, ..., Sunday=6
        Returns dict with 'day_of_week_sin' and 'day_of_week_cos'
        """
        # Handle both DatetimeIndex and Series
        if isinstance(timestamps, pd.Series):
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(timestamps):
                timestamps = pd.to_datetime(timestamps)
            day_of_week = timestamps.dt.dayofweek
            index_to_use = timestamps.index
        else:
            # Assume it's a DatetimeIndex
            day_of_week = timestamps.dayofweek
            index_to_use = timestamps
        
        # Calculate sine and cosine components
        day_of_week_sin = np.sin(2 * math.pi * day_of_week / 7)
        day_of_week_cos = np.cos(2 * math.pi * day_of_week / 7)
        
        return {
            'day_of_week_sin': pd.Series(day_of_week_sin, index=index_to_use),
            'day_of_week_cos': pd.Series(day_of_week_cos, index=index_to_use)
        }
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all 14 technical indicators + Day of Week features for a DataFrame with OHLCV data
        Expected columns: Open, High, Low, Close, Volume
        """
        result_df = df.copy()
        
        # Handle timestamp column and set proper datetime index if needed
        if 'timestamp' in result_df.columns and not isinstance(result_df.index, pd.DatetimeIndex):
            # Convert timestamp column to datetime and set as index
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
            timestamp_series = result_df['timestamp']
            # Keep timestamp column but also create datetime index for calculations
        elif isinstance(result_df.index, pd.DatetimeIndex):
            timestamp_series = result_df.index
        else:
            raise ValueError("DataFrame must have either a 'timestamp' column or DatetimeIndex")
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info("Calculating technical indicators...")
        
        # 1. SMA (10-hour)
        result_df['SMA_10'] = self.calculate_sma(df['Close'], 10)
        
        # 2. SMA (20-hour)
        result_df['SMA_20'] = self.calculate_sma(df['Close'], 20)
        
        # 3-5. MACD components
        macd_data = self.calculate_macd(df['Close'])
        result_df['MACD_line'] = macd_data['macd_line']
        result_df['MACD_signal'] = macd_data['macd_signal']
        result_df['MACD_hist'] = macd_data['macd_histogram']
        
        # 6. RSI (14-hour)
        result_df['RSI_14'] = self.calculate_rsi(df['Close'], 14)
        
        # 7-8. Stochastic Oscillator
        stoch_data = self.calculate_stochastic(df['High'], df['Low'], df['Close'])
        result_df['Stoch_K'] = stoch_data['stoch_k']
        result_df['Stoch_D'] = stoch_data['stoch_d']
        
        # 9. ADX (14-hour)
        result_df['ADX_14'] = self.calculate_adx(df['High'], df['Low'], df['Close'], 14)
        
        # 10. ATR (14-hour)
        result_df['ATR_14'] = self.calculate_atr(df['High'], df['Low'], df['Close'], 14)
        
        # 11. Bollinger Bandwidth
        result_df['BB_bandwidth'] = self.calculate_bollinger_bandwidth(df['Close'], 20, 2.0)
        
        # 12. OBV
        result_df['OBV'] = self.calculate_obv(df['Close'], df['Volume'])
        
        # 13. Volume SMA (20-hour)
        result_df['Volume_SMA_20'] = self.calculate_sma(df['Volume'], 20)
        
        # 14. 1-hour Return
        result_df['Return_1h'] = df['Close'].pct_change().fillna(0)
        
        # 15-16. Day of Week features (contextual) - use timestamp_series instead of df.index
        dow_features = self.calculate_day_of_week_features(timestamp_series)
        result_df['DayOfWeek_sin'] = dow_features['day_of_week_sin']
        result_df['DayOfWeek_cos'] = dow_features['day_of_week_cos']
        
        logger.info(f"Calculated all 14 technical indicators + 2 Day of Week features. DataFrame shape: {result_df.shape}")
        return result_df
    
    def process_symbol(self, symbol: str) -> bool:
        """
        Process a single symbol: load data, calculate indicators, save back to Parquet
        Returns True if successful, False otherwise
        """
        try:
            # Construct file path
            symbol_dir = self.data_dir / symbol / "1Hour"
            data_file = symbol_dir / "data.parquet"
            
            if not data_file.exists():
                logger.warning(f"Data file not found for {symbol}: {data_file}")
                return False
            
            # Load data
            logger.info(f"Processing {symbol}...")
            df = pd.read_parquet(data_file)
            
            # Verify data structure
            if df.empty:
                logger.warning(f"Empty data for {symbol}")
                return False
            
            # Calculate indicators
            df_with_indicators = self.calculate_all_indicators(df)
            
            # Save back to Parquet
            df_with_indicators.to_parquet(data_file, index=True)
            logger.info(f"Successfully updated {symbol} with technical indicators")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            return False
    
    def process_all_symbols(self, symbols: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Process all symbols or a specified list of symbols
        Returns dict mapping symbol to success status
        """
        if symbols is None:
            # Load symbols from config
            config_file = Path("config/symbols.json")
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    symbols = []
                    # Extract all symbols from all categories
                    for category, symbol_list in config.items():
                        if isinstance(symbol_list, list):
                            symbols.extend(symbol_list)
            else:
                logger.error("No symbols.json config file found and no symbols provided")
                return {}
        
        logger.info(f"Processing {len(symbols)} symbols...")
        results = {}
        
        for symbol in symbols:
            results[symbol] = self.process_symbol(symbol)
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        logger.info(f"Processing complete: {successful}/{total} symbols successful")
        
        return results
    
    def verify_indicators(self, symbol: str, sample_size: int = 10) -> Dict[str, any]:
        """
        Verify calculated indicators for a symbol by showing sample values
        """
        try:
            symbol_dir = self.data_dir / symbol / "1Hour"
            data_file = symbol_dir / "data.parquet"
            
            if not data_file.exists():
                return {"error": f"Data file not found for {symbol}"}
            
            df = pd.read_parquet(data_file)
            
            # Expected indicator columns
            indicator_cols = [
                'SMA_10', 'SMA_20', 'MACD_Line', 'MACD_Signal', 'MACD_Histogram',
                'RSI_14', 'Stoch_K', 'Stoch_D', 'ADX_14', 'ATR_14',
                'Bollinger_Bandwidth', 'OBV', 'Volume_SMA_20', 'Return_1h',
                'Day_of_Week_Sin', 'Day_of_Week_Cos'
            ]
            
            verification = {
                "symbol": symbol,
                "total_rows": len(df),
                "date_range": f"{df.index.min()} to {df.index.max()}",
                "indicators_present": [],
                "indicators_missing": [],
                "sample_data": {}
            }
            
            for col in indicator_cols:
                if col in df.columns:
                    verification["indicators_present"].append(col)
                    # Get sample of non-NaN values
                    non_nan_values = df[col].dropna()
                    if len(non_nan_values) > 0:
                        sample_values = non_nan_values.tail(min(sample_size, len(non_nan_values)))
                        verification["sample_data"][col] = {
                            "non_nan_count": len(non_nan_values),
                            "nan_count": df[col].isna().sum(),
                            "sample_values": sample_values.tolist()
                        }
                else:
                    verification["indicators_missing"].append(col)
            
            return verification
            
        except Exception as e:
            return {"error": f"Error verifying {symbol}: {str(e)}"}


def main():
    """Main execution function"""
    calculator = TechnicalIndicatorCalculator()
    
    # Process all symbols
    results = calculator.process_all_symbols()
    
    # Show verification for a few symbols
    successful_symbols = [symbol for symbol, success in results.items() if success]
    if successful_symbols:
        logger.info("\nVerification samples:")
        for symbol in successful_symbols[:3]:  # Show first 3 successful symbols
            verification = calculator.verify_indicators(symbol)
            if "error" not in verification:
                logger.info(f"\n{symbol}:")
                logger.info(f"  Total rows: {verification['total_rows']}")
                logger.info(f"  Date range: {verification['date_range']}")
                logger.info(f"  Indicators present: {len(verification['indicators_present'])}/16")
                logger.info(f"  Missing indicators: {verification['indicators_missing']}")


if __name__ == "__main__":
    main()