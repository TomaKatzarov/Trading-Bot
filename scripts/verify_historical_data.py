#!/usr/bin/env python3
"""
Historical Data Verification Script for Task 0.2

This script verifies the integrity and completeness of historical data files
as required by the NN Data Preparation implementation plan.

Verification Criteria:
- Existence of Parquet files for all symbols in config/symbols.json
- Data integrity: presence of OHLCV, VWAP columns; reasonable date ranges; minimal NaNs
- Sufficient historical depth (at least 2 years of 1-hour data)
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('historical_data_verification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HistoricalDataVerifier:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.data_root = self.project_root / "data" / "historical"
        self.config_path = self.project_root / "config" / "symbols.json"
        
        # Verification criteria
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
        self.min_historical_days = 365 * 2  # 2 years
        self.max_nan_percentage = 5.0  # Maximum 5% NaN values allowed
        self.timeframe = "1Hour"
        
        # Results storage
        self.verification_results = {}
        self.summary_stats = {
            'total_symbols': 0,
            'symbols_with_files': 0,
            'symbols_with_valid_data': 0,
            'symbols_with_sufficient_depth': 0,
            'symbols_missing_files': [],
            'symbols_with_issues': [],
            'symbols_insufficient_depth': []
        }

    def load_symbols_config(self) -> List[str]:
        """Load all symbols from the configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Extract all unique symbols from all sectors
            all_symbols = set()
            for sector_symbols in config.get('sectors', {}).values():
                all_symbols.update(symbol.strip().upper() for symbol in sector_symbols)
            
            # Add ETF symbols
            for etf_category in config.get('etfs', {}).values():
                all_symbols.update(symbol.strip().upper() for symbol in etf_category)
            
            # Add crypto symbols
            for crypto_category in config.get('crypto', {}).values():
                all_symbols.update(symbol.strip().upper() for symbol in crypto_category)
            
            return sorted(list(all_symbols))
            
        except Exception as e:
            logger.error(f"Failed to load symbols configuration: {e}")
            return []

    def check_file_exists(self, symbol: str) -> bool:
        """Check if the Parquet file exists for a symbol."""
        file_path = self.data_root / symbol / self.timeframe / "data.parquet"
        return file_path.exists()

    def verify_data_integrity(self, symbol: str) -> Dict:
        """Verify data integrity for a single symbol."""
        file_path = self.data_root / symbol / self.timeframe / "data.parquet"
        result = {
            'symbol': symbol,
            'file_exists': False,
            'columns_valid': False,
            'date_range_valid': False,
            'sufficient_depth': False,
            'nan_percentage': 100.0,
            'row_count': 0,
            'first_date': None,
            'last_date': None,
            'missing_columns': [],
            'issues': []
        }
        
        try:
            if not file_path.exists():
                result['issues'].append("File does not exist")
                return result
            
            result['file_exists'] = True
            
            # Load the data
            df = pd.read_parquet(file_path)
            result['row_count'] = len(df)
            
            if df.empty:
                result['issues'].append("File is empty")
                return result
            
            # Check required columns
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            result['missing_columns'] = missing_cols
            result['columns_valid'] = len(missing_cols) == 0
            
            if missing_cols:
                result['issues'].append(f"Missing columns: {missing_cols}")
            
            # Check date range
            if isinstance(df.index, pd.DatetimeIndex):
                result['first_date'] = df.index[0].strftime('%Y-%m-%d %H:%M:%S')
                result['last_date'] = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                
                # Calculate data span
                data_span = df.index[-1] - df.index[0]
                result['sufficient_depth'] = data_span.days >= self.min_historical_days
                result['date_range_valid'] = True
                
                if not result['sufficient_depth']:
                    result['issues'].append(f"Insufficient historical depth: {data_span.days} days (need {self.min_historical_days})")
            else:
                result['issues'].append("Index is not DatetimeIndex")
            
            # Check for NaN values
            if result['columns_valid']:
                total_values = len(df) * len(self.required_columns)
                nan_count = df[self.required_columns].isnull().sum().sum()
                result['nan_percentage'] = (nan_count / total_values) * 100
                
                if result['nan_percentage'] > self.max_nan_percentage:
                    result['issues'].append(f"High NaN percentage: {result['nan_percentage']:.2f}%")
            
            # Additional data quality checks
            if 'Close' in df.columns:
                # Check for zero or negative prices
                invalid_prices = (df['Close'] <= 0).sum()
                if invalid_prices > 0:
                    result['issues'].append(f"Found {invalid_prices} invalid price values (<=0)")
                
                # Check for extreme price movements (>50% in one hour)
                if len(df) > 1:
                    price_changes = df['Close'].pct_change().abs()
                    extreme_moves = (price_changes > 0.5).sum()
                    if extreme_moves > 0:
                        result['issues'].append(f"Found {extreme_moves} extreme price movements (>50%)")
            
            if 'Volume' in df.columns:
                # Check for negative volume
                negative_volume = (df['Volume'] < 0).sum()
                if negative_volume > 0:
                    result['issues'].append(f"Found {negative_volume} negative volume values")
            
        except Exception as e:
            result['issues'].append(f"Error reading file: {str(e)}")
            logger.error(f"Error verifying {symbol}: {e}")
        
        return result

    def run_verification(self) -> Dict:
        """Run complete verification for all symbols."""
        logger.info("Starting historical data verification...")
        
        # Load symbols
        symbols = self.load_symbols_config()
        if not symbols:
            logger.error("No symbols loaded from configuration")
            return self.summary_stats
        
        self.summary_stats['total_symbols'] = len(symbols)
        logger.info(f"Verifying data for {len(symbols)} symbols...")
        
        # Verify each symbol
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"Verifying {symbol} ({i}/{len(symbols)})")
            result = self.verify_data_integrity(symbol)
            self.verification_results[symbol] = result
            
            # Update summary statistics
            if result['file_exists']:
                self.summary_stats['symbols_with_files'] += 1
                
                if result['columns_valid'] and result['date_range_valid'] and not result['issues']:
                    self.summary_stats['symbols_with_valid_data'] += 1
                
                if result['sufficient_depth']:
                    self.summary_stats['symbols_with_sufficient_depth'] += 1
                else:
                    self.summary_stats['symbols_insufficient_depth'].append(symbol)
                
                if result['issues']:
                    self.summary_stats['symbols_with_issues'].append(symbol)
            else:
                self.summary_stats['symbols_missing_files'].append(symbol)
        
        return self.summary_stats

    def generate_report(self) -> str:
        """Generate a comprehensive verification report."""
        report = []
        report.append("=" * 80)
        report.append("HISTORICAL DATA VERIFICATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Verification Criteria:")
        report.append(f"  - Required columns: {', '.join(self.required_columns)}")
        report.append(f"  - Minimum historical depth: {self.min_historical_days} days")
        report.append(f"  - Maximum NaN percentage: {self.max_nan_percentage}%")
        report.append("")
        
        # Summary statistics
        stats = self.summary_stats
        report.append("SUMMARY STATISTICS")
        report.append("-" * 40)
        report.append(f"Total symbols in config: {stats['total_symbols']}")
        report.append(f"Symbols with files: {stats['symbols_with_files']}")
        report.append(f"Symbols with valid data: {stats['symbols_with_valid_data']}")
        report.append(f"Symbols with sufficient depth: {stats['symbols_with_sufficient_depth']}")
        report.append("")
        
        # Coverage percentages
        if stats['total_symbols'] > 0:
            file_coverage = (stats['symbols_with_files'] / stats['total_symbols']) * 100
            valid_coverage = (stats['symbols_with_valid_data'] / stats['total_symbols']) * 100
            depth_coverage = (stats['symbols_with_sufficient_depth'] / stats['total_symbols']) * 100
            
            report.append("COVERAGE ANALYSIS")
            report.append("-" * 40)
            report.append(f"File coverage: {file_coverage:.1f}%")
            report.append(f"Valid data coverage: {valid_coverage:.1f}%")
            report.append(f"Sufficient depth coverage: {depth_coverage:.1f}%")
            report.append("")
        
        # Missing files
        if stats['symbols_missing_files']:
            report.append(f"MISSING FILES ({len(stats['symbols_missing_files'])} symbols)")
            report.append("-" * 40)
            for symbol in sorted(stats['symbols_missing_files']):
                report.append(f"  {symbol}")
            report.append("")
        
        # Insufficient depth
        if stats['symbols_insufficient_depth']:
            report.append(f"INSUFFICIENT DEPTH ({len(stats['symbols_insufficient_depth'])} symbols)")
            report.append("-" * 40)
            for symbol in sorted(stats['symbols_insufficient_depth']):
                if symbol in self.verification_results:
                    result = self.verification_results[symbol]
                    if result['first_date'] and result['last_date']:
                        first_date = pd.to_datetime(result['first_date'])
                        last_date = pd.to_datetime(result['last_date'])
                        days = (last_date - first_date).days
                        report.append(f"  {symbol}: {days} days ({result['first_date']} to {result['last_date']})")
                    else:
                        report.append(f"  {symbol}: Date range unknown")
            report.append("")
        
        # Data quality issues
        if stats['symbols_with_issues']:
            report.append(f"DATA QUALITY ISSUES ({len(stats['symbols_with_issues'])} symbols)")
            report.append("-" * 40)
            for symbol in sorted(stats['symbols_with_issues']):
                if symbol in self.verification_results:
                    result = self.verification_results[symbol]
                    report.append(f"  {symbol}:")
                    for issue in result['issues']:
                        report.append(f"    - {issue}")
            report.append("")
        
        # Detailed statistics by symbol (top 10 by row count)
        valid_symbols = [(s, r) for s, r in self.verification_results.items() 
                        if r['file_exists'] and r['row_count'] > 0]
        if valid_symbols:
            valid_symbols.sort(key=lambda x: x[1]['row_count'], reverse=True)
            report.append("TOP 10 SYMBOLS BY DATA VOLUME")
            report.append("-" * 40)
            for symbol, result in valid_symbols[:10]:
                report.append(f"  {symbol}: {result['row_count']:,} rows, "
                            f"NaN: {result['nan_percentage']:.2f}%, "
                            f"Range: {result['first_date']} to {result['last_date']}")
            report.append("")
        
        report.append("=" * 80)
        return "\n".join(report)

    def save_detailed_results(self, filename: str = "historical_data_verification_detailed.json"):
        """Save detailed verification results to JSON file."""
        output_path = self.project_root / filename
        try:
            with open(output_path, 'w') as f:
                json.dump({
                    'summary': self.summary_stats,
                    'detailed_results': self.verification_results,
                    'verification_timestamp': datetime.now().isoformat(),
                    'criteria': {
                        'required_columns': self.required_columns,
                        'min_historical_days': self.min_historical_days,
                        'max_nan_percentage': self.max_nan_percentage
                    }
                }, f, indent=2, default=str)
            logger.info(f"Detailed results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save detailed results: {e}")


def main():
    """Main execution function."""
    verifier = HistoricalDataVerifier()
    
    # Run verification
    summary = verifier.run_verification()
    
    # Generate and display report
    report = verifier.generate_report()
    print(report)
    
    # Save detailed results
    verifier.save_detailed_results()
    
    # Save report to file
    report_path = verifier.project_root / "historical_data_verification_report.txt"
    try:
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
    
    # Return exit code based on verification success
    total_symbols = summary['total_symbols']
    valid_symbols = summary['symbols_with_valid_data']
    
    if total_symbols == 0:
        logger.error("No symbols found in configuration")
        return 1
    
    success_rate = (valid_symbols / total_symbols) * 100
    logger.info(f"Verification completed. Success rate: {success_rate:.1f}%")
    
    # Consider verification successful if >90% of symbols have valid data
    if success_rate >= 90:
        logger.info("Verification PASSED")
        return 0
    else:
        logger.warning("Verification FAILED - insufficient data coverage")
        return 1


if __name__ == "__main__":
    sys.exit(main())