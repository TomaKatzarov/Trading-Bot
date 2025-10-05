# scripts/verify_feature_calculations.py
"""
Verification script for technical indicator calculations
Compares our implementations against pandas-ta library for accuracy
"""

import logging
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.feature_calculator import TechnicalIndicatorCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureVerifier:
    """
    Verifies technical indicator calculations by comparing against reference implementations
    """
    
    def __init__(self, data_dir: str = 'data/historical'):
        self.data_dir = Path(data_dir)
        self.calculator = TechnicalIndicatorCalculator(data_dir)
        self.pandas_ta_available = self._check_pandas_ta()
    
    def _check_pandas_ta(self) -> bool:
        """Check if pandas-ta is available for reference calculations"""
        try:
            import pandas_ta as ta
            logger.info("pandas-ta library available for reference calculations")
            return True
        except ImportError:
            logger.warning("pandas-ta not available. Will use manual verification methods.")
            return False
    
    def compare_with_pandas_ta(self, df: pd.DataFrame, symbol: str) -> Dict[str, Dict[str, float]]:
        """
        Compare our calculations with pandas-ta library
        Returns comparison metrics for each indicator
        """
        if not self.pandas_ta_available:
            return {"error": "pandas-ta not available"}
        
        try:
            import pandas_ta as ta
            
            # Calculate our indicators
            our_df = self.calculator.calculate_all_indicators(df)
            
            # Calculate reference indicators using pandas-ta
            ref_df = df.copy()
            
            # SMA
            ref_df['SMA_10_ref'] = ta.sma(df['Close'], length=10)
            ref_df['SMA_20_ref'] = ta.sma(df['Close'], length=20)
            
            # MACD
            macd_data = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            ref_df['MACD_Line_ref'] = macd_data['MACD_12_26_9']
            ref_df['MACD_Signal_ref'] = macd_data['MACDs_12_26_9']
            ref_df['MACD_Histogram_ref'] = macd_data['MACDh_12_26_9']
            
            # RSI
            ref_df['RSI_14_ref'] = ta.rsi(df['Close'], length=14)
            
            # Stochastic
            stoch_data = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
            ref_df['Stoch_K_ref'] = stoch_data['STOCHk_14_3_3']
            ref_df['Stoch_D_ref'] = stoch_data['STOCHd_14_3_3']
            
            # ADX
            ref_df['ADX_14_ref'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
            
            # ATR
            ref_df['ATR_14_ref'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            
            # Bollinger Bands (for bandwidth calculation)
            bb_data = ta.bbands(df['Close'], length=20, std=2.0)
            ref_df['Bollinger_Bandwidth_ref'] = (bb_data['BBU_20_2.0'] - bb_data['BBL_20_2.0']) / bb_data['BBM_20_2.0']
            
            # OBV
            ref_df['OBV_ref'] = ta.obv(df['Close'], df['Volume'])
            
            # Volume SMA
            ref_df['Volume_SMA_20_ref'] = ta.sma(df['Volume'], length=20)
            
            # Returns
            ref_df['Return_1h_ref'] = df['Close'].pct_change()
            
            # Compare indicators
            comparisons = {}
            indicator_pairs = [
                ('SMA_10', 'SMA_10_ref'),
                ('SMA_20', 'SMA_20_ref'),
                ('MACD_Line', 'MACD_Line_ref'),
                ('MACD_Signal', 'MACD_Signal_ref'),
                ('MACD_Histogram', 'MACD_Histogram_ref'),
                ('RSI_14', 'RSI_14_ref'),
                ('Stoch_K', 'Stoch_K_ref'),
                ('Stoch_D', 'Stoch_D_ref'),
                ('ADX_14', 'ADX_14_ref'),
                ('ATR_14', 'ATR_14_ref'),
                ('Bollinger_Bandwidth', 'Bollinger_Bandwidth_ref'),
                ('OBV', 'OBV_ref'),
                ('Volume_SMA_20', 'Volume_SMA_20_ref'),
                ('Return_1h', 'Return_1h_ref')
            ]
            
            for our_col, ref_col in indicator_pairs:
                if our_col in our_df.columns and ref_col in ref_df.columns:
                    our_values = our_df[our_col].dropna()
                    ref_values = ref_df[ref_col].dropna()
                    
                    # Align the series (in case of different NaN handling)
                    common_index = our_values.index.intersection(ref_values.index)
                    if len(common_index) > 10:  # Need sufficient data points
                        our_aligned = our_values.loc[common_index]
                        ref_aligned = ref_values.loc[common_index]
                        
                        # Calculate comparison metrics
                        correlation = our_aligned.corr(ref_aligned)
                        mae = np.mean(np.abs(our_aligned - ref_aligned))
                        rmse = np.sqrt(np.mean((our_aligned - ref_aligned) ** 2))
                        max_diff = np.max(np.abs(our_aligned - ref_aligned))
                        
                        comparisons[our_col] = {
                            'correlation': correlation,
                            'mae': mae,
                            'rmse': rmse,
                            'max_diff': max_diff,
                            'data_points': len(common_index),
                            'our_mean': our_aligned.mean(),
                            'ref_mean': ref_aligned.mean()
                        }
            
            return comparisons
            
        except Exception as e:
            logger.error(f"Error comparing with pandas-ta for {symbol}: {str(e)}")
            return {"error": str(e)}
    
    def manual_verification(self, df: pd.DataFrame, symbol: str) -> Dict[str, any]:
        """
        Manual verification of calculations using known mathematical properties
        """
        verification = {
            "symbol": symbol,
            "tests": {}
        }
        
        try:
            # Calculate indicators
            result_df = self.calculator.calculate_all_indicators(df)
            
            # Test 1: SMA properties
            sma_10 = result_df['SMA_10'].dropna()
            sma_20 = result_df['SMA_20'].dropna()
            
            verification["tests"]["sma_monotonicity"] = {
                "description": "SMA should be smoother than raw prices",
                "sma_10_volatility": sma_10.std(),
                "sma_20_volatility": sma_20.std(),
                "price_volatility": df['Close'].std(),
                "passed": sma_20.std() < sma_10.std() < df['Close'].std()
            }
            
            # Test 2: RSI bounds
            rsi = result_df['RSI_14'].dropna()
            verification["tests"]["rsi_bounds"] = {
                "description": "RSI should be between 0 and 100",
                "min_value": rsi.min(),
                "max_value": rsi.max(),
                "passed": (rsi.min() >= 0) and (rsi.max() <= 100)
            }
            
            # Test 3: Stochastic bounds
            stoch_k = result_df['Stoch_K'].dropna()
            stoch_d = result_df['Stoch_D'].dropna()
            verification["tests"]["stochastic_bounds"] = {
                "description": "Stochastic should be between 0 and 100",
                "stoch_k_min": stoch_k.min(),
                "stoch_k_max": stoch_k.max(),
                "stoch_d_min": stoch_d.min(),
                "stoch_d_max": stoch_d.max(),
                "passed": (stoch_k.min() >= 0) and (stoch_k.max() <= 100) and 
                         (stoch_d.min() >= 0) and (stoch_d.max() <= 100)
            }
            
            # Test 4: MACD relationship
            macd_line = result_df['MACD_Line'].dropna()
            macd_signal = result_df['MACD_Signal'].dropna()
            macd_hist = result_df['MACD_Histogram'].dropna()
            
            # Check if histogram = line - signal (approximately)
            common_idx = macd_line.index.intersection(macd_signal.index).intersection(macd_hist.index)
            if len(common_idx) > 10:
                calculated_hist = macd_line.loc[common_idx] - macd_signal.loc[common_idx]
                actual_hist = macd_hist.loc[common_idx]
                hist_diff = np.abs(calculated_hist - actual_hist).max()
                
                verification["tests"]["macd_consistency"] = {
                    "description": "MACD Histogram should equal Line - Signal",
                    "max_difference": hist_diff,
                    "passed": hist_diff < 1e-10
                }
            
            # Test 5: OBV monotonicity with volume
            obv = result_df['OBV'].dropna()
            verification["tests"]["obv_properties"] = {
                "description": "OBV should accumulate volume",
                "obv_range": obv.max() - obv.min(),
                "volume_sum": df['Volume'].sum(),
                "passed": (obv.max() - obv.min()) > 0
            }
            
            # Test 6: Returns calculation
            returns = result_df['Return_1h'].dropna()
            manual_returns = df['Close'].pct_change().dropna()
            common_idx = returns.index.intersection(manual_returns.index)
            
            if len(common_idx) > 10:
                returns_diff = np.abs(returns.loc[common_idx] - manual_returns.loc[common_idx]).max()
                verification["tests"]["returns_accuracy"] = {
                    "description": "Returns should match pct_change calculation",
                    "max_difference": returns_diff,
                    "passed": returns_diff < 1e-10
                }
            
            # Test 7: NaN handling
            total_rows = len(result_df)
            nan_counts = {}
            for col in ['SMA_10', 'SMA_20', 'MACD_Line', 'RSI_14', 'Stoch_K', 'ADX_14', 'ATR_14']:
                if col in result_df.columns:
                    nan_counts[col] = result_df[col].isna().sum()
            
            verification["tests"]["nan_handling"] = {
                "description": "NaN counts should be reasonable for each indicator",
                "total_rows": total_rows,
                "nan_counts": nan_counts,
                "passed": all(count < total_rows * 0.1 for count in nan_counts.values())  # Less than 10% NaN
            }
            
        except Exception as e:
            verification["error"] = str(e)
        
        return verification
    
    def verify_symbol(self, symbol: str) -> Dict[str, any]:
        """
        Comprehensive verification for a single symbol
        """
        try:
            # Load data
            symbol_dir = self.data_dir / symbol / "1Hour"
            data_file = symbol_dir / "data.parquet"
            
            if not data_file.exists():
                return {"error": f"Data file not found for {symbol}"}
            
            df = pd.read_parquet(data_file)
            
            if len(df) < 50:  # Need sufficient data for meaningful verification
                return {"error": f"Insufficient data for {symbol} ({len(df)} rows)"}
            
            verification_results = {
                "symbol": symbol,
                "data_rows": len(df),
                "date_range": f"{df.index.min()} to {df.index.max()}"
            }
            
            # Manual verification
            verification_results["manual_tests"] = self.manual_verification(df, symbol)
            
            # pandas-ta comparison if available
            if self.pandas_ta_available:
                verification_results["pandas_ta_comparison"] = self.compare_with_pandas_ta(df, symbol)
            
            return verification_results
            
        except Exception as e:
            return {"error": f"Error verifying {symbol}: {str(e)}"}
    
    def run_verification_suite(self, symbols: Optional[List[str]] = None, max_symbols: int = 5) -> Dict[str, any]:
        """
        Run verification suite on multiple symbols
        """
        if symbols is None:
            # Load symbols from config
            config_file = Path("config/symbols.json")
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    all_symbols = []
                    for category, symbol_list in config.items():
                        if isinstance(symbol_list, list):
                            all_symbols.extend(symbol_list)
                    symbols = all_symbols[:max_symbols]  # Limit for testing
            else:
                logger.error("No symbols.json config file found")
                return {"error": "No symbols configuration found"}
        
        logger.info(f"Running verification suite on {len(symbols)} symbols...")
        
        suite_results = {
            "summary": {
                "total_symbols": len(symbols),
                "successful_verifications": 0,
                "failed_verifications": 0
            },
            "symbol_results": {}
        }
        
        for symbol in symbols:
            logger.info(f"Verifying {symbol}...")
            result = self.verify_symbol(symbol)
            
            if "error" in result:
                suite_results["summary"]["failed_verifications"] += 1
                logger.warning(f"Verification failed for {symbol}: {result['error']}")
            else:
                suite_results["summary"]["successful_verifications"] += 1
                logger.info(f"Verification completed for {symbol}")
            
            suite_results["symbol_results"][symbol] = result
        
        return suite_results


def main():
    """Main execution function"""
    verifier = FeatureVerifier()
    
    # Run verification suite
    results = verifier.run_verification_suite(max_symbols=3)
    
    # Print summary
    print("\n" + "="*60)
    print("FEATURE CALCULATION VERIFICATION RESULTS")
    print("="*60)
    
    summary = results["summary"]
    print(f"Total symbols tested: {summary['total_symbols']}")
    print(f"Successful verifications: {summary['successful_verifications']}")
    print(f"Failed verifications: {summary['failed_verifications']}")
    
    # Print detailed results for successful verifications
    for symbol, result in results["symbol_results"].items():
        if "error" not in result:
            print(f"\n--- {symbol} ---")
            print(f"Data rows: {result['data_rows']}")
            print(f"Date range: {result['date_range']}")
            
            # Manual tests
            if "manual_tests" in result and "tests" in result["manual_tests"]:
                print("Manual verification tests:")
                for test_name, test_result in result["manual_tests"]["tests"].items():
                    status = "PASS" if test_result.get("passed", False) else "FAIL"
                    print(f"  {test_name}: {status}")
            
            # pandas-ta comparison
            if "pandas_ta_comparison" in result and "error" not in result["pandas_ta_comparison"]:
                print("pandas-ta comparison:")
                for indicator, metrics in result["pandas_ta_comparison"].items():
                    if isinstance(metrics, dict) and "correlation" in metrics:
                        corr = metrics["correlation"]
                        print(f"  {indicator}: correlation = {corr:.4f}")


if __name__ == "__main__":
    main()