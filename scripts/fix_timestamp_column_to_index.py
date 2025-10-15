"""
Fix existing historical data files by converting timestamp column to DatetimeIndex.

This script repairs files that were broken by the sentiment attacher saving
with index=False, which converted DatetimeIndex to RangeIndex + timestamp column.
"""

import logging
from pathlib import Path
import pandas as pd
import json
from typing import List, Set
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_ROOT = Path("data/historical")
SYMBOLS_CONFIG = Path("config/symbols.json")


def load_symbols() -> List[str]:
    """Load symbols from config"""
    with SYMBOLS_CONFIG.open('r') as f:
        config = json.load(f)
    
    symbols: Set[str] = set()
    
    def collect(values):
        for item in values:
            if isinstance(item, list):
                collect(item)
            elif isinstance(item, dict):
                collect(item.values())
            else:
                symbols.add(str(item))
    
    collect(config.get("sectors", {}).values())
    collect(config.get("indices", {}).values())
    collect(config.get("etfs", {}).values())
    collect(config.get("crypto", {}).values())
    
    return sorted(symbols)


def fix_symbol_file(symbol: str) -> dict:
    """Fix a single symbol's data file"""
    file_path = DATA_ROOT / symbol / "1Hour" / "data.parquet"
    
    if not file_path.exists():
        return {
            'symbol': symbol,
            'status': 'missing',
            'message': 'File not found'
        }
    
    try:
        df = pd.read_parquet(file_path)
        
        # Check if timestamp is a column (broken format)
        if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            logger.info(f"{symbol}: Converting timestamp column to DatetimeIndex")
            
            # Convert timestamp column to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.set_index('timestamp')
            
            # Save back with proper index
            df.to_parquet(file_path, index=True, engine='pyarrow', compression='snappy')
            
            return {
                'symbol': symbol,
                'status': 'fixed',
                'rows': len(df),
                'start': df.index.min().isoformat(),
                'end': df.index.max().isoformat()
            }
        
        # Check if already has DatetimeIndex (correct format)
        elif isinstance(df.index, pd.DatetimeIndex):
            # Ensure timezone-aware
            if df.index.tz is None:
                logger.info(f"{symbol}: Adding timezone to DatetimeIndex")
                df.index = df.index.tz_localize('UTC')
                df.to_parquet(file_path, index=True, engine='pyarrow', compression='snappy')
                status = 'tz_fixed'
            else:
                status = 'ok'
            
            return {
                'symbol': symbol,
                'status': status,
                'rows': len(df),
                'start': df.index.min().isoformat(),
                'end': df.index.max().isoformat()
            }
        
        else:
            logger.error(f"{symbol}: No timestamp column or DatetimeIndex found")
            return {
                'symbol': symbol,
                'status': 'error',
                'message': 'No timestamp data'
            }
    
    except Exception as e:
        logger.error(f"{symbol}: Error fixing file: {e}")
        return {
            'symbol': symbol,
            'status': 'error',
            'message': str(e)
        }


def main():
    """Fix all symbol files"""
    logger.info("Starting timestamp column to index conversion...")
    
    symbols = load_symbols()
    logger.info(f"Found {len(symbols)} symbols to process")
    
    results = []
    status_counter = defaultdict(int)
    
    for symbol in symbols:
        result = fix_symbol_file(symbol)
        results.append(result)
        status_counter[result['status']] += 1
        
        if result['status'] == 'fixed':
            logger.info(f"✓ {symbol}: Fixed ({result['rows']} rows, {result['start']} to {result['end']})")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY:")
    logger.info(f"Total symbols: {len(symbols)}")
    for status, count in sorted(status_counter.items()):
        logger.info(f"  {status}: {count}")
    
    # Save detailed results
    results_path = Path("data/timestamp_fix_results.json")
    with results_path.open('w') as f:
        json.dump({
            'summary': dict(status_counter),
            'details': results
        }, f, indent=2)
    
    logger.info(f"\nDetailed results saved to {results_path}")
    
    return status_counter


if __name__ == "__main__":
    main()
