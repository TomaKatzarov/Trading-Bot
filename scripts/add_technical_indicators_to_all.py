import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.feature_calculator import TechnicalIndicatorCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_all_symbols_with_data():
    """Get all symbols that have historical data files"""
    data_dir = Path('data/historical')
    symbols = []
    
    if data_dir.exists():
        for symbol_dir in data_dir.iterdir():
            if symbol_dir.is_dir():
                data_file = symbol_dir / '1Hour' / 'data.parquet'
                if data_file.exists():
                    symbols.append(symbol_dir.name)
    
    return sorted(symbols)

def main():
    logger.info("="*80)
    logger.info("ADDING TECHNICAL INDICATORS TO ALL HISTORICAL DATA")
    logger.info("="*80)
    
    # Get all symbols with data
    symbols = get_all_symbols_with_data()
    logger.info(f"Found {len(symbols)} symbols with historical data")
    
    if not symbols:
        logger.error("No symbols found with historical data!")
        return
    
    # Initialize calculator
    calculator = TechnicalIndicatorCalculator()
    
    # Process all symbols
    results = {}
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"Processing {i}/{len(symbols)}: {symbol}")
        results[symbol] = calculator.process_symbol(symbol)
    
    # Summary
    successful = sum(1 for success in results.values() if success)
    failed = len(results) - successful
    
    logger.info("="*80)
    logger.info("PROCESSING COMPLETE")
    logger.info(f"Total symbols: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    if failed > 0:
        failed_symbols = [sym for sym, success in results.items() if not success]
        logger.warning(f"Failed symbols: {failed_symbols}")
    
    logger.info("="*80)

if __name__ == "__main__":
    main()