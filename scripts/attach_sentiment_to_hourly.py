#!/usr/bin/env python3
"""
Sentiment Score Attachment to Hourly Data Script for Task 1.2

This script implements the logic to attach daily sentiment scores to hourly historical data
by forward-filling daily sentiment scores to match hourly timestamps.

Key Features:
- Loads hourly historical data from data/historical/{SYMBOL}/1Hour/data.parquet
- Loads daily sentiment scores from data/sentiment/{SYMBOL}/daily_sentiment.parquet
- Forward-fills daily sentiment to hourly timestamps
- Handles weekends and holidays (carries forward last trading day sentiment)
- Merges sentiment as new column: sentiment_score_hourly_ffill
- Overwrites existing Parquet files with updated data
- Comprehensive verification and error handling
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_attachment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentAttacher:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.historical_data_root = self.project_root / "data" / "historical"
        self.sentiment_data_root = self.project_root / "data" / "sentiment"
        self.config_path = self.project_root / "config" / "symbols.json"
        
        # Configuration
        self.timeframe = "1Hour"
        self.sentiment_column_name = "sentiment_score_hourly_ffill"
        
        # Statistics tracking
        self.stats = {
            'symbols_processed': 0,
            'symbols_skipped': 0,
            'symbols_failed': 0,
            'total_records_updated': 0,
            'errors': []
        }
        
    def load_symbols(self) -> List[str]:
        """Load all symbols from config/symbols.json"""
        try:
            with open(self.config_path, 'r') as f:
                symbols_config = json.load(f)
            
            # Extract all unique symbols from all categories
            all_symbols = set()
            
            # Add symbols from sectors
            if 'sectors' in symbols_config:
                for sector_symbols in symbols_config['sectors'].values():
                    all_symbols.update(sector_symbols)
            
            # Add symbols from indices
            if 'indices' in symbols_config:
                for index_symbols in symbols_config['indices'].values():
                    all_symbols.update(index_symbols)
            
            # Add symbols from ETFs
            if 'etfs' in symbols_config:
                for etf_symbols in symbols_config['etfs'].values():
                    all_symbols.update(etf_symbols)
            
            # Add symbols from crypto
            if 'crypto' in symbols_config:
                for crypto_symbols in symbols_config['crypto'].values():
                    all_symbols.update(crypto_symbols)
            
            symbols_list = sorted(list(all_symbols))
            logger.info(f"Loaded {len(symbols_list)} unique symbols from configuration")
            return symbols_list
            
        except Exception as e:
            logger.error(f"Error loading symbols configuration: {e}")
            raise
    
    def load_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load hourly historical data for a symbol"""
        try:
            data_path = self.historical_data_root / symbol / self.timeframe / "data.parquet"
            
            if not data_path.exists():
                logger.warning(f"Historical data file not found for {symbol}: {data_path}")
                return None
            
            df = pd.read_parquet(data_path)
            
            # Handle both DatetimeIndex and timestamp column formats
            if 'timestamp' in df.columns:
                # Already has timestamp column, ensure it's timezone-aware
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if df['timestamp'].dt.tz is None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
                # Set timestamp as index for consistent processing
                df = df.set_index('timestamp')
            elif isinstance(df.index, pd.DatetimeIndex):
                # Already has DatetimeIndex, just ensure timezone-aware
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
            else:
                logger.error(f"No timestamp column or DatetimeIndex found in historical data for {symbol}")
                return None
            
            logger.debug(f"Loaded {len(df)} hourly records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}")
            return None
    
    def load_sentiment_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load daily sentiment data for a symbol"""
        try:
            sentiment_path = self.sentiment_data_root / symbol / "daily_sentiment.parquet"
            
            if not sentiment_path.exists():
                logger.warning(f"Sentiment data file not found for {symbol}: {sentiment_path}")
                return None
            
            df = pd.read_parquet(sentiment_path)
            
            # Ensure date column is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            else:
                logger.error(f"No date column found in sentiment data for {symbol}")
                return None
            
            # Ensure sentiment_score column exists
            if 'sentiment_score' not in df.columns:
                logger.error(f"No sentiment_score column found in sentiment data for {symbol}")
                return None
            
            # Sort by date to ensure proper forward-filling
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.debug(f"Loaded {len(df)} daily sentiment records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading sentiment data for {symbol}: {e}")
            return None
    
    def forward_fill_sentiment(self, historical_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill daily sentiment scores to hourly timestamps.
        
        Logic:
        - Each day's sentiment applies to all trading hours of that day
        - For non-trading periods (weekends, holidays), carry forward last trading day sentiment
        - Use pandas merge_asof for efficient time-based joining
        """
        try:
            # Create a copy to avoid modifying original data
            hist_df = historical_df.copy()
            
            # Prepare sentiment data for merging
            sent_df = sentiment_df.copy()
            
            # Handle DatetimeIndex - convert to column for merge_asof
            if isinstance(hist_df.index, pd.DatetimeIndex):
                hist_df = hist_df.reset_index()
                if 'index' in hist_df.columns:
                    hist_df = hist_df.rename(columns={'index': 'timestamp'})
            
            # Ensure both timestamp columns are datetime with consistent timezone and precision
            # Convert to datetime64[ns, UTC] for consistent merge (pandas default)
            hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'], utc=True).astype('datetime64[ns, UTC]')
            sent_df['date'] = pd.to_datetime(sent_df['date'], utc=True).astype('datetime64[ns, UTC]')
            
            # Sort both dataframes by timestamp/date
            hist_df = hist_df.sort_values('timestamp')
            sent_df = sent_df.sort_values('date').reset_index(drop=True)
            
            # Use merge_asof for forward-fill behavior
            # This will match each hourly timestamp with the most recent sentiment score
            merged_df = pd.merge_asof(
                hist_df,
                sent_df[['date', 'sentiment_score']],
                left_on='timestamp',
                right_on='date',
                direction='backward',  # Use most recent sentiment score
                suffixes=('', '_sentiment')
            )
            
            # Handle sentiment column naming
            # If sentiment already existed, merge_asof created sentiment_score_sentiment or similar
            # Drop the old sentiment column and use the new one
            if self.sentiment_column_name in merged_df.columns:
                # Drop existing sentiment column (old data)
                merged_df = merged_df.drop(columns=[self.sentiment_column_name])
            
            # Rename the newly merged sentiment_score to our target name
            if 'sentiment_score' in merged_df.columns:
                merged_df = merged_df.rename(columns={'sentiment_score': self.sentiment_column_name})
            
            # Drop the temporary date column from sentiment data
            merged_df = merged_df.drop(columns=['date'], errors='ignore')
            
            # Check if sentiment column exists after merge
            if self.sentiment_column_name not in merged_df.columns:
                raise ValueError(f"Sentiment column '{self.sentiment_column_name}' not found after merge. Available columns: {list(merged_df.columns)}")
            
            # Check for any missing sentiment values
            missing_count = int(merged_df[self.sentiment_column_name].isna().sum())
            if missing_count > 0:
                logger.warning(f"Found {missing_count} records with missing sentiment scores")
                
                # Forward-fill any remaining NaN values
                merged_df[self.sentiment_column_name] = merged_df[self.sentiment_column_name].ffill()
                
                # If still NaN at the beginning, backward-fill
                merged_df[self.sentiment_column_name] = merged_df[self.sentiment_column_name].bfill()
            
            # Restore DatetimeIndex format (consistent with rest of pipeline)
            if 'timestamp' in merged_df.columns:
                merged_df = merged_df.set_index('timestamp')
            
            return merged_df
            
        except Exception as e:
            import traceback
            logger.error(f"Error in forward_fill_sentiment: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def verify_sentiment_attachment(self, df: pd.DataFrame, symbol: str) -> Dict[str, any]:
        """
        Verify the correct attachment and forward-filling of sentiment scores.
        
        Returns verification results including:
        - Total records
        - Records with sentiment
        - Missing sentiment count
        - Sentiment value range
        - Sample verification for weekends/holidays
        """
        try:
            verification = {
                'symbol': symbol,
                'total_records': len(df),
                'records_with_sentiment': 0,
                'missing_sentiment': 0,
                'sentiment_min': None,
                'sentiment_max': None,
                'sentiment_mean': None,
                'weekend_holiday_check': 'PASS',
                'value_range_check': 'PASS'
            }
            
            if self.sentiment_column_name in df.columns:
                sentiment_series = df[self.sentiment_column_name]
                
                verification['records_with_sentiment'] = sentiment_series.notna().sum()
                verification['missing_sentiment'] = sentiment_series.isna().sum()
                
                if verification['records_with_sentiment'] > 0:
                    verification['sentiment_min'] = float(sentiment_series.min())
                    verification['sentiment_max'] = float(sentiment_series.max())
                    verification['sentiment_mean'] = float(sentiment_series.mean())
                    
                    # Check if sentiment values are in expected range [0, 1]
                    if verification['sentiment_min'] < 0 or verification['sentiment_max'] > 1:
                        verification['value_range_check'] = 'FAIL'
                        logger.warning(f"Sentiment values outside [0,1] range for {symbol}")
                
                # Check weekend/holiday forward-filling
                df_with_weekday = df.copy()
                # Use index for weekday calculation (DatetimeIndex)
                if isinstance(df_with_weekday.index, pd.DatetimeIndex):
                    df_with_weekday['weekday'] = df_with_weekday.index.weekday
                elif 'timestamp' in df_with_weekday.columns:
                    df_with_weekday['weekday'] = df_with_weekday['timestamp'].dt.weekday
                else:
                    # Skip weekday check if no timestamp info
                    logger.warning(f"Cannot perform weekday check for {symbol}: no timestamp info")
                    return verification
                
                # Check if weekend records (Saturday=5, Sunday=6) have sentiment
                weekend_records = df_with_weekday[df_with_weekday['weekday'].isin([5, 6])]
                if len(weekend_records) > 0:
                    weekend_missing = weekend_records[self.sentiment_column_name].isna().sum()
                    if weekend_missing > 0:
                        verification['weekend_holiday_check'] = 'FAIL'
                        logger.warning(f"Found {weekend_missing} weekend records without sentiment for {symbol}")
            else:
                verification['weekend_holiday_check'] = 'FAIL'
                verification['value_range_check'] = 'FAIL'
                logger.error(f"Sentiment column {self.sentiment_column_name} not found in data for {symbol}")
            
            return verification
            
        except Exception as e:
            logger.error(f"Error in verify_sentiment_attachment for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def save_updated_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Save the updated DataFrame back to the Parquet file"""
        try:
            data_path = self.historical_data_root / symbol / self.timeframe / "data.parquet"
            
            # Ensure the directory exists
            data_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the updated data with DatetimeIndex (index=True to preserve index structure)
            df.to_parquet(data_path, index=True)
            
            logger.info(f"Successfully updated historical data for {symbol} with sentiment scores")
            return True
            
        except Exception as e:
            logger.error(f"Error saving updated data for {symbol}: {e}")
            return False
    
    def process_symbol(self, symbol: str) -> Dict[str, any]:
        """Process sentiment attachment for a single symbol"""
        try:
            logger.info(f"Processing sentiment attachment for {symbol}")
            
            # Load historical data
            historical_df = self.load_historical_data(symbol)
            if historical_df is None:
                self.stats['symbols_skipped'] += 1
                return {'symbol': symbol, 'status': 'skipped', 'reason': 'no_historical_data'}
            
            # Load sentiment data
            sentiment_df = self.load_sentiment_data(symbol)
            if sentiment_df is None:
                self.stats['symbols_skipped'] += 1
                return {'symbol': symbol, 'status': 'skipped', 'reason': 'no_sentiment_data'}
            
            # Check if sentiment column already exists
            if self.sentiment_column_name in historical_df.columns:
                logger.info(f"Sentiment column already exists for {symbol}, updating...")
            
            # Forward-fill sentiment to hourly data
            updated_df = self.forward_fill_sentiment(historical_df, sentiment_df)
            
            # Verify the attachment
            verification = self.verify_sentiment_attachment(updated_df, symbol)
            
            # Save the updated data
            if self.save_updated_data(updated_df, symbol):
                self.stats['symbols_processed'] += 1
                self.stats['total_records_updated'] += len(updated_df)
                
                return {
                    'symbol': symbol,
                    'status': 'success',
                    'records_updated': len(updated_df),
                    'verification': verification
                }
            else:
                self.stats['symbols_failed'] += 1
                return {'symbol': symbol, 'status': 'failed', 'reason': 'save_error'}
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            self.stats['symbols_failed'] += 1
            self.stats['errors'].append(f"{symbol}: {str(e)}")
            return {'symbol': symbol, 'status': 'failed', 'reason': str(e)}
    
    def process_all_symbols(self, symbols: Optional[List[str]] = None) -> Dict[str, any]:
        """Process sentiment attachment for all symbols"""
        try:
            if symbols is None:
                symbols = self.load_symbols()
            
            logger.info(f"Starting sentiment attachment for {len(symbols)} symbols")
            
            results = []
            
            for symbol in symbols:
                result = self.process_symbol(symbol)
                results.append(result)
                
                # Log progress every 10 symbols
                if len(results) % 10 == 0:
                    logger.info(f"Processed {len(results)}/{len(symbols)} symbols")
            
            # Generate summary report
            summary = {
                'total_symbols': len(symbols),
                'processed': self.stats['symbols_processed'],
                'skipped': self.stats['symbols_skipped'],
                'failed': self.stats['symbols_failed'],
                'total_records_updated': self.stats['total_records_updated'],
                'results': results,
                'errors': self.stats['errors']
            }
            
            logger.info(f"Sentiment attachment completed:")
            logger.info(f"  - Processed: {summary['processed']}")
            logger.info(f"  - Skipped: {summary['skipped']}")
            logger.info(f"  - Failed: {summary['failed']}")
            logger.info(f"  - Total records updated: {summary['total_records_updated']}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in process_all_symbols: {e}")
            raise

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Attach sentiment scores to hourly historical data')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to process (default: all)')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing sentiment attachments')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Set logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        attacher = SentimentAttacher()
        
        if args.verify_only:
            logger.info("Running verification-only mode")
            # TODO: Implement verification-only mode if needed
            symbols = args.symbols if args.symbols else attacher.load_symbols()
            for symbol in symbols:
                historical_df = attacher.load_historical_data(symbol)
                if historical_df is not None and attacher.sentiment_column_name in historical_df.columns:
                    verification = attacher.verify_sentiment_attachment(historical_df, symbol)
                    logger.info(f"Verification for {symbol}: {verification}")
        else:
            # Process sentiment attachment
            summary = attacher.process_all_symbols(args.symbols)
            
            # Save summary report
            summary_path = Path('sentiment_attachment_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Summary report saved to {summary_path}")
            
            # Return appropriate exit code
            if summary['failed'] > 0:
                logger.warning("Some symbols failed processing")
                sys.exit(1)
            else:
                logger.info("All symbols processed successfully")
                sys.exit(0)
                
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()