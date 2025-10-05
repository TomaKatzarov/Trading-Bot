#!/usr/bin/env python3
"""
Verification script for sentiment processing functionality.
Tests the enhanced news_sentiment.py module with FinBERT processing.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.news_sentiment import NewsSentimentAnalyzer

def setup_logging():
    """Setup logging for verification."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_model_initialization():
    """Test that the FinBERT model initializes correctly."""
    logger = logging.getLogger(__name__)
    logger.info("Testing FinBERT model initialization...")
    
    try:
        analyzer = NewsSentimentAnalyzer(max_workers=2, batch_size=4)
        logger.info(f"✅ Model initialized successfully")
        logger.info(f"   - Model: {analyzer.model_name}")
        logger.info(f"   - Device: {analyzer.device}")
        logger.info(f"   - Batch size: {analyzer.batch_size}")
        logger.info(f"   - GPU available: {analyzer.gpu_info['available']}")
        return analyzer
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        return None

def test_sentiment_analysis():
    """Test sentiment analysis with sample texts."""
    logger = logging.getLogger(__name__)
    logger.info("Testing sentiment analysis functionality...")
    
    analyzer = NewsSentimentAnalyzer(max_workers=2, batch_size=4)
    
    # Test texts with known sentiment
    test_texts = [
        "Apple stock surges to new all-time high on strong earnings report",  # Positive
        "Company faces major lawsuit and regulatory scrutiny",  # Negative
        "Stock price remains stable in sideways trading",  # Neutral
        "",  # Empty
        "The quarterly results exceeded analyst expectations significantly"  # Positive
    ]
    
    try:
        # Test batch processing
        scores = analyzer.analyze_sentiment(test_texts)
        logger.info(f"✅ Sentiment analysis completed")
        
        for i, (text, score) in enumerate(zip(test_texts, scores)):
            text_preview = text[:50] + "..." if len(text) > 50 else text
            logger.info(f"   Text {i+1}: '{text_preview}' -> Score: {score:.3f}")
        
        # Test normalization
        for i, text in enumerate(test_texts[:3]):  # Test first 3
            normalized_score = analyzer.get_normalized_score([text], f"TEST{i+1}")
            logger.info(f"   Normalized score for text {i+1}: {normalized_score:.3f}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Sentiment analysis failed: {e}")
        return False

def test_symbol_loading():
    """Test loading symbols from configuration."""
    logger = logging.getLogger(__name__)
    logger.info("Testing symbol loading...")
    
    try:
        analyzer = NewsSentimentAnalyzer(max_workers=2)
        symbols = analyzer.get_all_symbols()
        
        if symbols:
            logger.info(f"✅ Loaded {len(symbols)} symbols")
            logger.info(f"   Sample symbols: {symbols[:5]}")
            return symbols
        else:
            logger.warning("⚠️ No symbols loaded")
            return []
    except Exception as e:
        logger.error(f"❌ Symbol loading failed: {e}")
        return []

def test_news_fetching():
    """Test news fetching for a sample symbol."""
    logger = logging.getLogger(__name__)
    logger.info("Testing news fetching...")
    
    try:
        analyzer = NewsSentimentAnalyzer(max_workers=2)
        
        # Test with a major symbol
        test_symbol = "AAPL"
        test_date = datetime.now() - timedelta(days=1)  # Yesterday
        
        logger.info(f"Fetching news for {test_symbol} on {test_date.strftime('%Y-%m-%d')}")
        news_articles = analyzer.fetch_news(test_symbol, test_date)
        
        if news_articles is not None:
            logger.info(f"✅ News fetching successful")
            logger.info(f"   Found {len(news_articles)} articles")
            if news_articles:
                logger.info(f"   Sample headline: {news_articles[0][:100]}...")
            return True
        else:
            logger.warning("⚠️ News fetching returned None (API issue or no news)")
            return False
    except Exception as e:
        logger.error(f"❌ News fetching failed: {e}")
        return False

def test_single_symbol_processing():
    """Test processing sentiment for a single symbol."""
    logger = logging.getLogger(__name__)
    logger.info("Testing single symbol sentiment processing...")
    
    try:
        analyzer = NewsSentimentAnalyzer(max_workers=2)
        
        # Test with a major symbol
        test_symbol = "AAPL"
        test_date = datetime.now() - timedelta(days=1)
        
        logger.info(f"Processing sentiment for {test_symbol} on {test_date.strftime('%Y-%m-%d')}")
        sentiment_score = analyzer.process_symbol_sentiment(test_symbol, test_date)
        
        if sentiment_score is not None:
            logger.info(f"✅ Single symbol processing successful")
            logger.info(f"   Sentiment score: {sentiment_score:.3f}")
            
            # Check if file was created
            file_path = analyzer._get_symbol_sentiment_file_path(test_symbol)
            if file_path.exists():
                logger.info(f"   ✅ Parquet file created: {file_path}")
                
                # Load and verify data
                import pandas as pd
                df = pd.read_parquet(file_path)
                logger.info(f"   Data shape: {df.shape}")
                logger.info(f"   Columns: {list(df.columns)}")
                if not df.empty:
                    logger.info(f"   Latest entry: {df.iloc[-1].to_dict()}")
            else:
                logger.warning("   ⚠️ Parquet file not found")
            
            return True
        else:
            logger.warning("⚠️ Single symbol processing returned None")
            return False
    except Exception as e:
        logger.error(f"❌ Single symbol processing failed: {e}")
        return False

def test_concurrent_processing():
    """Test concurrent processing of multiple symbols."""
    logger = logging.getLogger(__name__)
    logger.info("Testing concurrent symbol processing...")
    
    try:
        analyzer = NewsSentimentAnalyzer(max_workers=2)
        
        # Test with a few major symbols
        test_symbols = ["AAPL", "MSFT", "GOOGL"]
        test_date = datetime.now() - timedelta(days=1)
        
        logger.info(f"Processing sentiment for {len(test_symbols)} symbols concurrently")
        results = analyzer.process_symbols_concurrent(test_symbols, test_date, max_workers=2)
        
        logger.info(f"✅ Concurrent processing completed")
        logger.info(f"   Processed {len(results)}/{len(test_symbols)} symbols successfully")
        
        for symbol, score in results.items():
            logger.info(f"   {symbol}: {score:.3f}")
        
        return len(results) > 0
    except Exception as e:
        logger.error(f"❌ Concurrent processing failed: {e}")
        return False

def test_data_persistence():
    """Test that sentiment data is properly saved and can be reloaded."""
    logger = logging.getLogger(__name__)
    logger.info("Testing data persistence...")
    
    try:
        analyzer = NewsSentimentAnalyzer(max_workers=2)
        
        test_symbol = "AAPL"
        
        # Load existing data
        existing_data = analyzer._load_existing_sentiment_data(test_symbol)
        logger.info(f"✅ Data loading successful")
        logger.info(f"   Existing records: {len(existing_data)}")
        
        if not existing_data.empty:
            logger.info(f"   Date range: {existing_data['date'].min()} to {existing_data['date'].max()}")
            logger.info(f"   Average sentiment: {existing_data['sentiment_score'].mean():.3f}")
            logger.info(f"   Models used: {existing_data['model_used'].unique()}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Data persistence test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    logger = setup_logging()
    logger.info("Starting sentiment processing verification...")
    
    tests = [
        ("Model Initialization", test_model_initialization),
        ("Sentiment Analysis", test_sentiment_analysis),
        ("Symbol Loading", test_symbol_loading),
        ("News Fetching", test_news_fetching),
        ("Single Symbol Processing", test_single_symbol_processing),
        ("Concurrent Processing", test_concurrent_processing),
        ("Data Persistence", test_data_persistence),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "⚠️ PARTIAL"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"{test_name}: ❌ FAILED - {e}")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("VERIFICATION SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Sentiment processing is ready.")
    elif passed > total // 2:
        logger.info("⚠️ Most tests passed. Some issues may need attention.")
    else:
        logger.error("❌ Multiple test failures. Please review the implementation.")

if __name__ == "__main__":
    main()