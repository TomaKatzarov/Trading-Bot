import pandas as pd
from pathlib import Path
import json

def check_historical_coverage(symbols_file='config/symbols.json', historical_dir='data/historical'):
    """Check date coverage in historical data for all symbols"""
    historical_path = Path(historical_dir)
    
    # Load symbols
    with open(symbols_file, 'r') as f:
        symbols_config = json.load(f)
    
    all_symbols = []
    for category in symbols_config.values():
        if isinstance(category, dict):
            for subcat in category.values():
                if isinstance(subcat, list):
                    all_symbols.extend(subcat)
        elif isinstance(category, list):
            all_symbols.extend(category)
    
    all_symbols = list(set(all_symbols))  # Dedup
    print(f"Total unique symbols from config: {len(all_symbols)}")
    
    coverage_summary = {
        'total_symbols': len(all_symbols),
        'symbols_with_data': 0,
        'symbols_without_data': 0,
        'date_ranges': {},
        'new_data_2025': 0,
        'total_rows': 0
    }
    
    for symbol in all_symbols:
        symbol_dir = historical_path / symbol / '1Hour'
        data_file = symbol_dir / 'data.parquet'
        
        if data_file.exists():
            try:
                df = pd.read_parquet(data_file)
                coverage_summary['symbols_with_data'] += 1
                coverage_summary['total_rows'] += len(df)
                
                min_date = df.index.min()
                max_date = df.index.max()
                coverage_summary['date_ranges'][symbol] = f"{min_date.date()} to {max_date.date()}"
                
                # Check for 2025 data (May-Oct)
                if pd.to_datetime('2025-05-01') <= max_date:
                    coverage_summary['new_data_2025'] += 1
                
                print(f"{symbol}: {len(df)} rows, {min_date.date()} to {max_date.date()}")
                
            except Exception as e:
                print(f"Error reading {symbol}: {e}")
                coverage_summary['symbols_without_data'] += 1
        else:
            coverage_summary['symbols_without_data'] += 1
            print(f"{symbol}: No data file")
    
    print("\nSummary:")
    print(f"Symbols with data: {coverage_summary['symbols_with_data']}/{len(all_symbols)}")
    print(f"Symbols without data: {coverage_summary['symbols_without_data']}")
    print(f"Symbols with 2025 data: {coverage_summary['new_data_2025']}")
    print(f"Total rows across all symbols: {coverage_summary['total_rows']:,}")
    
    # Estimate new data coverage
    expected_new_rows_per_symbol = 150 * 24  # 5 months * 24 hours
    expected_total_new = expected_new_rows_per_symbol * coverage_summary['new_data_2025']
    print(f"Expected new rows (May-Oct 2025): ~{expected_total_new:,}")
    
    return coverage_summary

if __name__ == "__main__":
    coverage = check_historical_coverage()