#!/usr/bin/env python3
"""
Verification Script for Sentiment Attachment to Hourly Data

This script verifies the correct implementation of sentiment score attachment
by testing the logic on sample data and checking for proper forward-filling
across weekends and holidays.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

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

class SentimentAttachmentVerifier:
    def __init__(self):
        self.attacher = SentimentAttacher()
        
    def create_sample_hourly_data(self) -> pd.DataFrame:
        """Create sample hourly data for testing"""
        # Create 7 days of hourly data (including weekend)
        start_date = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)  # Monday 9 AM
        end_date = start_date + timedelta(days=7)
        
        # Generate hourly timestamps
        timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
        
        # Create sample OHLCV data
        np.random.seed(42)  # For reproducible results
        n_records = len(timestamps)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'Open': 150.0 + np.random.randn(n_records) * 2,
            'High': 152.0 + np.random.randn(n_records) * 2,
            'Low': 148.0 + np.random.randn(n_records) * 2,
            'Close': 150.0 + np.random.randn(n_records) * 2,
            'Volume': 1000000 + np.random.randint(-100000, 100000, n_records),
            'VWAP': 150.0 + np.random.randn(n_records) * 1.5,
            # Add some technical indicators (dummy values)
            'SMA_10': 150.0 + np.random.randn(n_records) * 1,
            'SMA_20': 150.0 + np.random.randn(n_records) * 1,
            'RSI': 50.0 + np.random.randn(n_records) * 10,
            'MACD': np.random.randn(n_records) * 0.5,
            'day_of_week_sin': np.sin(2 * np.pi * timestamps.weekday / 7),
            'day_of_week_cos': np.cos(2 * np.pi * timestamps.weekday / 7)
        })
        
        return df
    
    def create_sample_sentiment_data(self) -> pd.DataFrame:
        """Create sample daily sentiment data for testing"""
        # Create sentiment data for weekdays only (simulating real market data)
        start_date = datetime(2024, 1, 15, tzinfo=timezone.utc)  # Monday
        
        # Create sentiment for 5 trading days
        dates = []
        sentiments = []
        
        # Monday to Friday
        for i in range(5):
            date = start_date + timedelta(days=i)
            dates.append(date.date())
            sentiments.append(0.3 + i * 0.15)  # Increasing sentiment: 0.3, 0.45, 0.6, 0.75, 0.9
        
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'sentiment_score': sentiments,
            'news_count': [10, 15, 8, 12, 20],
            'model_used': ['finbert'] * 5
        })
        
        return df
    
    def test_forward_fill_logic(self) -> Dict[str, any]:
        """Test the forward-fill logic with sample data"""
        logger.info("Testing forward-fill logic with sample data")
        
        try:
            # Create sample data
            hourly_df = self.create_sample_hourly_data()
            sentiment_df = self.create_sample_sentiment_data()
            
            logger.info(f"Created {len(hourly_df)} hourly records")
            logger.info(f"Created {len(sentiment_df)} daily sentiment records")
            
            # Test the forward-fill function
            result_df = self.attacher.forward_fill_sentiment(hourly_df, sentiment_df)
            
            # Verify results
            verification = self.attacher.verify_sentiment_attachment(result_df, "TEST_SYMBOL")
            
            # Additional specific tests
            test_results = {
                'basic_attachment': False,
                'weekend_forward_fill': False,
                'value_range_check': False,
                'no_missing_values': False,
                'correct_daily_mapping': False
            }
            
            # Check if sentiment column exists
            if self.attacher.sentiment_column_name in result_df.columns:
                test_results['basic_attachment'] = True
                sentiment_col = result_df[self.attacher.sentiment_column_name]
                
                # Check value range [0, 1]
                if sentiment_col.min() >= 0 and sentiment_col.max() <= 1:
                    test_results['value_range_check'] = True
                
                # Check for missing values
                if sentiment_col.isna().sum() == 0:
                    test_results['no_missing_values'] = True
                
                # Check weekend forward-fill
                result_df['weekday'] = result_df['timestamp'].dt.weekday
                weekend_data = result_df[result_df['weekday'].isin([5, 6])]
                if len(weekend_data) > 0 and weekend_data[self.attacher.sentiment_column_name].notna().all():
                    test_results['weekend_forward_fill'] = True
                
                # Check correct daily mapping (Monday should have 0.3, Tuesday 0.45, etc.)
                monday_data = result_df[result_df['weekday'] == 0]  # Monday
                tuesday_data = result_df[result_df['weekday'] == 1]  # Tuesday
                
                if (len(monday_data) > 0 and len(tuesday_data) > 0 and
                    abs(monday_data[self.attacher.sentiment_column_name].iloc[0] - 0.3) < 0.01 and
                    abs(tuesday_data[self.attacher.sentiment_column_name].iloc[0] - 0.45) < 0.01):
                    test_results['correct_daily_mapping'] = True
            
            # Log detailed results
            logger.info("Test Results:")
            for test_name, passed in test_results.items():
                status = "PASS" if passed else "FAIL"
                logger.info(f"  {test_name}: {status}")
            
            # Sample data inspection
            logger.info("\nSample of result data:")
            sample_data = result_df[['timestamp', self.attacher.sentiment_column_name]].head(10)
            for _, row in sample_data.iterrows():
                weekday = row['timestamp'].strftime('%A')
                logger.info(f"  {row['timestamp']} ({weekday}): {row[self.attacher.sentiment_column_name]:.3f}")
            
            return {
                'verification': verification,
                'test_results': test_results,
                'sample_data': result_df.head(24).to_dict('records'),  # First 24 hours
                'all_tests_passed': all(test_results.values())
            }
            
        except Exception as e:
            logger.error(f"Error in test_forward_fill_logic: {e}")
            return {'error': str(e), 'all_tests_passed': False}
    
    def test_weekend_holiday_handling(self) -> Dict[str, any]:
        """Test specific weekend and holiday handling scenarios"""
        logger.info("Testing weekend and holiday handling")
        
        try:
            # Create data spanning a weekend
            start_date = datetime(2024, 1, 19, 15, 0, 0, tzinfo=timezone.utc)  # Friday 3 PM
            end_date = datetime(2024, 1, 22, 10, 0, 0, tzinfo=timezone.utc)   # Monday 10 AM
            
            timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
            
            hourly_df = pd.DataFrame({
                'timestamp': timestamps,
                'Close': [150.0] * len(timestamps),
                'Volume': [1000000] * len(timestamps)
            })
            
            # Create sentiment for Friday only
            sentiment_df = pd.DataFrame({
                'date': [datetime(2024, 1, 19, tzinfo=timezone.utc)],  # Friday
                'sentiment_score': [0.7],
                'news_count': [15],
                'model_used': ['finbert']
            })
            
            # Apply forward-fill
            result_df = self.attacher.forward_fill_sentiment(hourly_df, sentiment_df)
            
            # Check that weekend hours have Friday's sentiment
            friday_sentiment = 0.7
            weekend_records = result_df[
                result_df['timestamp'].dt.weekday.isin([5, 6])  # Saturday, Sunday
            ]
            
            weekend_test_passed = (
                len(weekend_records) > 0 and
                weekend_records[self.attacher.sentiment_column_name].notna().all() and
                abs(weekend_records[self.attacher.sentiment_column_name].iloc[0] - friday_sentiment) < 0.01
            )
            
            logger.info(f"Weekend forward-fill test: {'PASS' if weekend_test_passed else 'FAIL'}")
            
            return {
                'weekend_test_passed': weekend_test_passed,
                'weekend_records_count': len(weekend_records),
                'friday_sentiment': friday_sentiment,
                'weekend_sentiment_sample': weekend_records[self.attacher.sentiment_column_name].iloc[0] if len(weekend_records) > 0 else None
            }
            
        except Exception as e:
            logger.error(f"Error in test_weekend_holiday_handling: {e}")
            return {'error': str(e), 'weekend_test_passed': False}
    
    def run_all_tests(self) -> Dict[str, any]:
        """Run all verification tests"""
        logger.info("Starting comprehensive sentiment attachment verification")
        
        results = {
            'forward_fill_test': self.test_forward_fill_logic(),
            'weekend_holiday_test': self.test_weekend_holiday_handling(),
            'overall_success': False
        }
        
        # Determine overall success
        forward_fill_success = results['forward_fill_test'].get('all_tests_passed', False)
        weekend_success = results['weekend_holiday_test'].get('weekend_test_passed', False)
        
        results['overall_success'] = forward_fill_success and weekend_success
        
        logger.info(f"Overall verification result: {'SUCCESS' if results['overall_success'] else 'FAILURE'}")
        
        return results

def main():
    """Main execution function"""
    try:
        verifier = SentimentAttachmentVerifier()
        results = verifier.run_all_tests()
        
        if results['overall_success']:
            logger.info("All sentiment attachment tests passed!")
            sys.exit(0)
        else:
            logger.error("Some sentiment attachment tests failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Verification script failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()