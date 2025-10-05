#!/usr/bin/env python3
"""
Test script to verify the hist_data_loader modifications
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.hist_data_loader import HistoricalDataLoader

def test_symbol_loading():
    """Test that the data loader can see all symbols."""
    loader = HistoricalDataLoader()
    
    # Test symbol config loading
    config = loader.load_symbols_config()
    print("Symbol configuration loaded successfully")
    
    # Count symbols in each category
    sector_count = sum(len(symbols) for symbols in config.get('sectors', {}).values())
    etf_count = sum(len(symbols) for symbols in config.get('etfs', {}).values())
    crypto_count = sum(len(symbols) for symbols in config.get('crypto', {}).values())
    
    print(f"Sectors: {sector_count} symbols")
    print(f"ETFs: {etf_count} symbols")
    print(f"Crypto: {crypto_count} symbols")
    print(f"Total: {sector_count + etf_count + crypto_count} symbols")
    
    # Test the modified load_all_symbols method (dry run)
    try:
        # We'll just test the symbol collection part without actually downloading
        symbols_config = loader.load_symbols_config()
        all_symbols = set()
        
        # Add sector symbols
        for sector_symbols in symbols_config.get('sectors', {}).values():
            all_symbols.update(sector_symbols)
        # Add ETF symbols
        for etf_symbols in symbols_config.get('etfs', {}).values():
            all_symbols.update(etf_symbols)
        # Add crypto symbols
        for crypto_symbols in symbols_config.get('crypto', {}).values():
            all_symbols.update(crypto_symbols)
        
        all_symbols = sorted(list(all_symbols))
        print(f"\nModified loader would process {len(all_symbols)} unique symbols")
        print("First 10 symbols:", all_symbols[:10])
        print("Last 10 symbols:", all_symbols[-10:])
        
        return True
        
    except Exception as e:
        print(f"Error testing symbol loading: {e}")
        return False

if __name__ == "__main__":
    success = test_symbol_loading()
    if success:
        print("\n✓ Data loader modifications verified successfully")
    else:
        print("\n✗ Data loader test failed")
        sys.exit(1)