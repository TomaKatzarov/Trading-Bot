#!/usr/bin/env python3
"""
Demonstration Script for Sentiment Attachment

This script creates sample sentiment data and demonstrates the sentiment attachment
process for a few symbols. It's designed to show the complete workflow without
requiring actual sentiment data to be present.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from typing import List

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our sentiment attacher
from scripts.attach_sentiment_to_hourly import SentimentAttacher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentAttachmentDemo:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.historical_data_root = self.project_root / "data" / "historical"
        self.sentiment_data_root = self.project_root / "data" / "sentiment"
        self.attacher = SentimentAttacher()
        
    def get_available_symbols(self) -> List[str]:
        """Get symbols that have historical data available"""
        symbols = []
        if self.historical_data_root.exists():
            for symbol_dir in self.historical_data_root.iterdir():
                if symbol_dir.is_dir():
                    data_file = symbol_dir / "1Hour" / "data.parquet"
                    if data_file.exists():
                        symbols.append(symbol_dir.name)
        return sorted(symbols)
    
    def create_sample_sentiment_data(self, symbol: str, days: int = 30) -> bool:
        """Create sample sentiment data for a symbol"""
        try:
            # Create sentiment directory for symbol
            sentiment_dir = self.sentiment_data_root / symbol
            sentiment_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate sample sentiment data for the last 30 days
            end_date = datetime.now(timezone.utc).date()
            start_date = end_date - timedelta(days=days)
            
            dates = []
            sentiment_scores = []
            news_counts = []
            
            # Generate data for weekdays only (simulating real market data)
            current_date = start_date
            np.random.seed(hash(symbol) % 2**32)  # Consistent random data per symbol
            
            while current_date <= end_date:
                # Only add data for weekdays (Monday=0 to Friday=4)
                if current_date.weekday() < 5:
                    dates.append(current_date)
                    
                    # Generate realistic sentiment scores (0.2 to 0.8 range)
                    base_sentiment = 0.5
                    daily_variation = np.random.normal(0, 0.1)
                    sentiment_score = np.clip(base_sentiment + daily_variation, 0.0, 1.0)
                    sentiment_scores.append(sentiment_score)
                    
                    # Generate realistic news counts (5 to 25 articles per day)
                    news_count = np.random.randint(5, 26)
                    news_counts.append(news_count)
                
                current_date += timedelta(days=1)
            
            # Create DataFrame
            sentiment_df = pd.DataFrame({
                'date': pd.to_datetime(dates),
                'sentiment_score': sentiment_scores,
                'news_count': news_counts,
                'model_used': ['finbert'] * len(dates)
            })
            
            # Save to Parquet
            sentiment_file = sentiment_dir / "daily_sentiment.parquet"
            sentiment_df.to_parquet(sentiment_file, index=False)
            
            logger.info(f"Created sample sentiment data for {symbol}: {len(sentiment_df)} records")
            return True
            
        except Exception as e:
            logger.error(f"Error creating sample sentiment data for {symbol}: {e}")
            return False
    
    def demonstrate_attachment(self, symbols: List[str]) -> dict:
        """Demonstrate the sentiment attachment process"""
        logger.info(f"Starting sentiment attachment demonstration for {len(symbols)} symbols")
        
        results = {
            'symbols_processed': [],
            'symbols_failed': [],
            'total_records_updated': 0,
            'sample_data': {}
        }
        
        for symbol in symbols:
            try:
                logger.info(f"Processing {symbol}...")
                
                # Create sample sentiment data
                if not self.create_sample_sentiment_data(symbol):
                    results['symbols_failed'].append(symbol)
                    continue
                
                # Process the symbol using our sentiment attacher
                result = self.attacher.process_symbol(symbol)
                
                if result['status'] == 'success':
                    results['symbols_processed'].append(symbol)
                    results['total_records_updated'] += result['records_updated']
                    
                    # Load a sample of the updated data for inspection
                    historical_df = self.attacher.load_historical_data(symbol)
                    if historical_df is not None and len(historical_df) > 0:
                        # Get a sample of recent data
                        sample = historical_df.tail(24)[['timestamp', self.attacher.sentiment_column_name]].copy()
                        sample['weekday'] = sample['timestamp'].dt.day_name()
                        results['sample_data'][symbol] = sample.to_dict('records')
                    
                    logger.info(f"SUCCESS {symbol}: {result['records_updated']} records updated")
                else:
                    results['symbols_failed'].append(symbol)
                    logger.warning(f"FAILED {symbol}: {result.get('reason', 'unknown error')}")
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results['symbols_failed'].append(symbol)
        
        return results
    
    def show_sample_results(self, results: dict):
        """Display sample results from the attachment process"""
        logger.info("\n" + "="*60)
        logger.info("SENTIMENT ATTACHMENT DEMONSTRATION RESULTS")
        logger.info("="*60)
        
        logger.info(f"Symbols processed successfully: {len(results['symbols_processed'])}")
        logger.info(f"Symbols failed: {len(results['symbols_failed'])}")
        logger.info(f"Total records updated: {results['total_records_updated']}")
        
        if results['symbols_processed']:
            logger.info(f"\nSuccessfully processed symbols: {', '.join(results['symbols_processed'])}")
        
        if results['symbols_failed']:
            logger.info(f"\nFailed symbols: {', '.join(results['symbols_failed'])}")
        
        # Show sample data for the first processed symbol
        if results['sample_data']:
            first_symbol = list(results['sample_data'].keys())[0]
            sample_data = results['sample_data'][first_symbol]
            
            logger.info(f"\nSample data for {first_symbol} (last 24 hours):")
            logger.info("-" * 80)
            logger.info(f"{'Timestamp':<25} {'Weekday':<10} {'Sentiment':<10}")
            logger.info("-" * 80)
            
            for record in sample_data[-10:]:  # Show last 10 records
                timestamp = str(record['timestamp'])[:19]  # Remove timezone for display
                weekday = record['weekday']
                sentiment = f"{record[self.attacher.sentiment_column_name]:.3f}"
                logger.info(f"{timestamp:<25} {weekday:<10} {sentiment:<10}")
        
        logger.info("\n" + "="*60)

def main():
    """Main execution function"""
    try:
        demo = SentimentAttachmentDemo()
        
        # Get available symbols (limit to first 3 for demo)
        available_symbols = demo.get_available_symbols()
        
        if not available_symbols:
            logger.error("No symbols with historical data found!")
            sys.exit(1)
        
        # Use first 3 symbols for demonstration
        demo_symbols = available_symbols[:3]
        logger.info(f"Available symbols: {len(available_symbols)}")
        logger.info(f"Using for demo: {demo_symbols}")
        
        # Run the demonstration
        results = demo.demonstrate_attachment(demo_symbols)
        
        # Show results
        demo.show_sample_results(results)
        
        # Cleanup: Remove sample sentiment data (optional)
        cleanup = input("\nRemove sample sentiment data? (y/N): ").lower().strip()
        if cleanup == 'y':
            import shutil
            for symbol in demo_symbols:
                sentiment_dir = demo.sentiment_data_root / symbol
                if sentiment_dir.exists():
                    shutil.rmtree(sentiment_dir)
                    logger.info(f"Removed sample sentiment data for {symbol}")
        
        logger.info("Demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()