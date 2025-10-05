import pandas as pd
from pathlib import Path
from datetime import datetime
import pytz

def verify_data_updates():
    """Verify that historical data was successfully updated to October 2025"""
    historical_dir = Path('data/historical')
    
    if not historical_dir.exists():
        print("❌ Historical data directory not found")
        return False
    
    stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'SPY', 'QQQ']
    success_count = 0
    oct_2025_count = 0
    total_checked = 0
    
    print("Checking sample symbols for October 2025 data...\n")
    
    for symbol in stock_symbols:
        data_file = historical_dir / symbol / '1Hour' / 'data.parquet'
        if data_file.exists():
            try:
                df = pd.read_parquet(data_file)
                total_checked += 1
                
                # Ensure index is datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                # Convert to UTC if needed for comparison
                if df.index.tz is not None:
                    max_date = df.index.max().astimezone(pytz.UTC)
                else:
                    max_date = df.index.max().replace(tzinfo=pytz.UTC)
                
                # Check if we have October 2025 data
                oct_2025_start = datetime(2025, 10, 1, tzinfo=pytz.UTC)
                has_oct_data = max_date >= oct_2025_start
                
                print(f"{symbol}:")
                print(f"  Total rows: {len(df):,}")
                print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
                print(f"  Latest timestamp: {max_date}")
                print(f"  Has Oct 2025 data: {'✅ YES' if has_oct_data else '❌ NO'}")
                print()
                
                if has_oct_data:
                    oct_2025_count += 1
                success_count += 1
                
            except Exception as e:
                print(f"❌ Error reading {symbol}: {e}\n")
        else:
            print(f"❌ {symbol}: No data file found\n")
    
    print("="*60)
    print(f"SUMMARY:")
    print(f"  Symbols checked: {total_checked}")
    print(f"  Successfully read: {success_count}")
    print(f"  With October 2025 data: {oct_2025_count}")
    
    if oct_2025_count >= 7:  # At least 7 out of 9 sample symbols
        print("\n✅ DATA UPDATE SUCCESSFUL!")
        print("   Historical data contains October 2025 records")
        return True
    else:
        print("\n⚠️ DATA UPDATE INCOMPLETE")
        print(f"   Only {oct_2025_count}/{total_checked} symbols have October 2025 data")
        return False

if __name__ == "__main__":
    success = verify_data_updates()
    exit(0 if success else 1)