#!/usr/bin/env python3
"""
Quick script to count total symbols in config/symbols.json
"""

import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
config_path = project_root / "config" / "symbols.json"

def count_symbols():
    """Count all unique symbols in the configuration."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        all_symbols = set()
        
        # Count symbols from sectors
        sectors = config.get('sectors', {})
        for sector, symbols in sectors.items():
            all_symbols.update(symbol.strip().upper() for symbol in symbols)
            print(f"{sector}: {len(symbols)} symbols")
        
        # Count ETF symbols
        etfs = config.get('etfs', {})
        etf_count = 0
        for category, symbols in etfs.items():
            all_symbols.update(symbol.strip().upper() for symbol in symbols)
            etf_count += len(symbols)
            print(f"ETF {category}: {len(symbols)} symbols")
        print(f"Total ETFs: {etf_count}")
        
        # Count crypto symbols
        crypto = config.get('crypto', {})
        crypto_count = 0
        for category, symbols in crypto.items():
            all_symbols.update(symbol.strip().upper() for symbol in symbols)
            crypto_count += len(symbols)
            print(f"Crypto {category}: {len(symbols)} symbols")
        print(f"Total Crypto: {crypto_count}")
        
        print(f"\nTotal unique symbols: {len(all_symbols)}")
        return len(all_symbols)
        
    except Exception as e:
        print(f"Error reading config: {e}")
        return 0

if __name__ == "__main__":
    count_symbols()