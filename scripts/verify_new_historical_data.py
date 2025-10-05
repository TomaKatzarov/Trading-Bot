#!/usr/bin/env python3
"""
Verify New Historical Data (May 29 - Oct 1, 2025)
Checks date range, symbol coverage, completeness, quality for all symbols.
"""

import os
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
from typing import Dict, Any

def validate_symbol_data(data_path: str) -> Dict[str, Any]:
    """Validate data for a single symbol."""
    try:
        df = pd.read_parquet(data_path)
        if df.empty:
            return {'valid': False, 'error': 'Empty DataFrame'}
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Date range check
        # Convert index to UTC for comparison
        df_index_utc = df.index.tz_convert('UTC') if df.index.tz is not None else df.index
        min_date = df_index_utc.min()
        max_date = df_index_utc.max()
        expected_min = pd.Timestamp('2025-05-29', tz='UTC')
        expected_max = pd.Timestamp('2025-10-01', tz='UTC')
        date_valid = min_date < expected_max and max_date > expected_min
        
        # Completeness: Check full rows and new period rows
        expected_rows = 150 * 24  # Approximate for new period
        row_count = len(df)
        new_period_mask = df_index_utc >= expected_min
        new_period_rows = new_period_mask.sum()
        has_new_data = new_period_rows > 0
        completeness = row_count > 0 and new_period_rows >= expected_rows * 0.9
        
        # Check for gaps in new period
        time_diffs = df_index_utc.to_series().diff().dt.total_seconds() / 3600
        gaps = time_diffs > 1.1  # Allow small tolerance
        gap_count = gaps[new_period_mask.shift(1).fillna(False)].sum()  # Gaps in new period
        no_gaps = gap_count <= 5  # Allow few gaps for holidays
        
        # Quality: OHLCV columns, no NaN, high >= low, volume >=0
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        cols_present = all(col in df.columns for col in required_cols)
        no_nan = df[required_cols].isna().sum().sum() == 0
        high_low_valid = (df['High'] >= df['Low']).all()
        volume_valid = (df['Volume'] >= 0).all()
        quality_valid = cols_present and no_nan and high_low_valid and volume_valid
        completeness = completeness and has_new_data
        
        return {
            'valid': date_valid and completeness and no_gaps and quality_valid,
            'min_date': str(min_date.date()),
            'max_date': str(max_date.date()),
            'row_count': row_count,
            'gap_count': int(gap_count),
            'issues': []
        }
    except Exception as e:
        return {'valid': False, 'error': str(e)}

def main():
    historical_dir = Path('data/historical')
    if not historical_dir.exists():
        print("No historical data directory found.")
        return
    
    symbols = [d.name for d in historical_dir.iterdir() if d.is_dir()]
    print(f"Found {len(symbols)} symbols to validate.")
    
    validation_results = {}
    valid_count = 0
    for symbol in symbols:
        hour_dir = historical_dir / symbol / '1Hour'
        data_file = hour_dir / 'data.parquet'
        if data_file.exists():
            result = validate_symbol_data(data_file)
            validation_results[symbol] = result
            if result['valid']:
                valid_count += 1
            print(f"{symbol}: {'PASS' if result['valid'] else 'FAIL'} (rows: {result.get('row_count', 0)}, gaps: {result.get('gap_count', 'N/A')})")
        else:
            validation_results[symbol] = {'valid': False, 'error': 'No data.parquet'}
    
    # Summary
    total_symbols = len(symbols)
    print(f"\nValidation Summary:")
    print(f"Total symbols: {total_symbols}")
    print(f"Valid: {valid_count} ({valid_count/total_symbols*100:.1f}%)")
    print(f"Invalid: {total_symbols - valid_count}")
    
    # Save report
    report = {
        'total_symbols': total_symbols,
        'valid_symbols': valid_count,
        'validation_date': datetime.now().isoformat(),
        'details': validation_results
    }
    with open('data/validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("Report saved to data/validation_report.json")

if __name__ == "__main__":
    main()