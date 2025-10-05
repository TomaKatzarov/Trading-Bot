from pathlib import Path
import pandas as pd

sentiment_dir = Path('data/sentiment')

print("Checking sentiment data availability...\n")

# Check AAPL as example
aapl_dir = sentiment_dir / 'AAPL'
print(f"AAPL sentiment dir exists: {aapl_dir.exists()}")

if aapl_dir.exists():
    parquet_files = list(aapl_dir.glob('*.parquet'))
    print(f"Parquet files in AAPL: {[f.name for f in parquet_files]}")
    
    if parquet_files:
        df = pd.read_parquet(parquet_files[0])
        print(f"\nSample data from {parquet_files[0].name}:")
        print(df.head())
        print(f"\nShape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "No date column")
    else:
        print("No parquet files found in AAPL directory")
else:
    print("AAPL sentiment directory does not exist")

# Check a few more symbols
test_symbols = ['MSFT', 'GOOGL', 'TSLA']
print(f"\n\nChecking other symbols: {test_symbols}")
for symbol in test_symbols:
    symbol_dir = sentiment_dir / symbol
    if symbol_dir.exists():
        parquet_files = list(symbol_dir.glob('*.parquet'))
        print(f"  {symbol}: {len(parquet_files)} parquet file(s)")
    else:
        print(f"  {symbol}: No directory")