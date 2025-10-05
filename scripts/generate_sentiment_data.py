#!/usr/bin/env python3
"""
Generate Sentiment Data for Historical Period

Wrapper script to properly generate sentiment data for a date range.
Handles the sys.path setup correctly.

Usage:
    python scripts/generate_sentiment_data.py --start-date 2025-05-29 --end-date 2025-10-01
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.news_sentiment import NewsSentimentAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Generate sentiment data for date range')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols (optional, default is all)')
    parser.add_argument('--max-workers', type=int, default=4, help='Max parallel workers')
    
    args = parser.parse_args()
    
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        logger.error("Use YYYY-MM-DD format, e.g., 2025-05-29")
        return False
    
    logger.info("="*80)
    logger.info("GENERATING SENTIMENT DATA")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info("="*80)
    
    try:
        # Initialize analyzer
        analyzer = NewsSentimentAnalyzer(max_workers=args.max_workers)
        
        # Get symbols to process
        if args.symbols:
            symbols = args.symbols
            logger.info(f"Processing {len(symbols)} specified symbols")
        else:
            symbols = analyzer.get_all_symbols()
            logger.info(f"Processing all {len(symbols)} symbols from config")
        
        # Generate sentiment for the date range
        logger.info(f"Starting sentiment generation...")
        results = analyzer.process_historical_sentiment(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        successful = sum(1 for r in results.values() if r)
        failed = len(results) - successful
        
        logger.info("="*80)
        logger.info("SENTIMENT GENERATION COMPLETE")
        logger.info(f"Successful: {successful}/{len(results)}")
        logger.info(f"Failed: {failed}")
        
        if failed > 0:
            failed_symbols = [sym for sym, success in results.items() if not success]
            logger.warning(f"Failed symbols: {failed_symbols[:10]}{'...' if len(failed_symbols) > 10 else ''}")
        
        logger.info("="*80)
        
        return successful > 0
        
    except Exception as e:
        logger.error(f"Error generating sentiment data: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)