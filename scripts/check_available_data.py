import os
import json

# Load symbols
with open('config/symbols.json', 'r') as f:
    symbols_data = json.load(f)

# Collect all unique symbols
all_symbols = set()

# Get symbols from sectors
sectors = symbols_data.get('sectors', {})
for sector_name, sector_symbols in sectors.items():
    all_symbols.update(sector_symbols)
    print(f"  {sector_name}: {len(sector_symbols)} symbols")

# Get symbols from indices
indices = symbols_data.get('indices', {})
for idx_name, idx_symbols in indices.items():
    all_symbols.update(idx_symbols)

# Get symbols from ETFs
etfs = symbols_data.get('etfs', {})
for etf_type, etf_symbols in etfs.items():
    all_symbols.update(etf_symbols)

# Get symbols from crypto
crypto = symbols_data.get('crypto', {})
for crypto_type, crypto_symbols in crypto.items():
    all_symbols.update(crypto_symbols)

all_symbols = sorted(list(all_symbols))
print(f"\n✅ Total unique symbols configured: {len(all_symbols)}")

# Check which have 1Hour data
data_dir = 'data/historical'
available_symbols = []
for symbol in all_symbols:
    path = os.path.join(data_dir, symbol, '1Hour')
    if os.path.exists(path):
        # Check if data.parquet exists
        data_file = os.path.join(path, 'data.parquet')
        if os.path.exists(data_file):
            available_symbols.append(symbol)

print(f"✅ Symbols with 1Hour data available: {len(available_symbols)}/{len(all_symbols)}")
print(f"\nAvailable symbols ({len(available_symbols)}):")
print(", ".join(available_symbols[:50]))
if len(available_symbols) > 50:
    print(f"... and {len(available_symbols) - 50} more")

missing = set(all_symbols) - set(available_symbols)
if missing:
    print(f"\n❌ Missing data for {len(missing)} symbols:")
    print(", ".join(sorted(list(missing))[:20]))
    if len(missing) > 20:
        print(f"... and {len(missing) - 20} more")