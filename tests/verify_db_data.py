import logging
import sys
from pathlib import Path
from collections import defaultdict
from sqlalchemy import func, inspect, text # Import text
from sqlalchemy.orm import Session

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import necessary components after setting path
from core.db_models import TradeJournal, SentimentAnalysis # Ensure models are imported
try:
    from utils.db_setup import SessionLocal, engine # Base might not be needed directly here
except ImportError as e:
    # Keep error handling for db_setup import
    print(f"Error importing project modules (db_setup): {e}")
    print(f"Ensure the script is in the correct location relative to the project root: {project_root}")
    sys.exit(1)

# --- Configuration ---
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

INDICATOR_COLUMNS = [
    'open_price', 'high_price', 'low_price', 'close_price', 'volume',
    'wap', 'returns', 'hl_diff', 'ohlc_avg', 'rsi', 'volume_pct', 'returns_1h'
]
MIN_RECORDS_PER_SYMBOL = 50
FALLBACK_SENTIMENT_SCORE = 0.5
RSI_MIN = 0
RSI_MAX = 100
MAX_EXAMPLES_TO_LOG = 10

# --- Verification Functions ---

def check_table_exists(engine_instance, table_name: str) -> bool:
    """Check if a specific table exists in the database using the engine."""
    try:
        inspector = inspect(engine_instance)
        return table_name in inspector.get_table_names()
    except Exception as e:
        logger.error(f"Error checking table existence for '{table_name}': {e}")
        return False

def verify_data():
    """Performs data verification checks on the TradeJournal and SentimentAnalysis tables."""
    logger.info("Starting database data verification...")
    session = SessionLocal()
    sentiment_table_exists = False # Assume false initially

    try:
        # 1. Check if tables exist using the engine directly
        trade_journal_exists = check_table_exists(engine, 'trade_journal')
        sentiment_table_exists = check_table_exists(engine, 'sentiment_analysis')
        
        if not trade_journal_exists:
            logger.error("Critical error: trade_journal table does not exist in the database.")
            return False
        
        if not sentiment_table_exists:
            logger.error("Critical error: sentiment_analysis table does not exist in the database.")
            return False
            
        logger.info(f"Tables verified: trade_journal={trade_journal_exists}, sentiment_analysis={sentiment_table_exists}")
        
        # 2. Count records in both tables
        trade_count = session.query(func.count(TradeJournal.id)).scalar()
        sentiment_count = session.query(func.count(SentimentAnalysis.id)).scalar()
        
        logger.info(f"Record counts: trade_journal={trade_count}, sentiment_analysis={sentiment_count}")
        
        # 3. Find trades missing sentiment analysis
        missing_sentiment_query = text("""
            SELECT tj.id, tj.symbol, tj.timestamp 
            FROM trade_journal tj
            LEFT JOIN sentiment_analysis sa ON tj.id = sa.trade_journal_id
            WHERE sa.id IS NULL
        """)
        
        missing_sentiment = session.execute(missing_sentiment_query).fetchall()
        missing_count = len(missing_sentiment)
        
        logger.info(f"Found {missing_count} trades missing sentiment analysis")
        
        if missing_count > 0:
            # Log some examples of missing sentiment trades
            examples = missing_sentiment[:min(MAX_EXAMPLES_TO_LOG, missing_count)]
            logger.info("Examples of trades missing sentiment analysis:")
            for idx, (trade_id, symbol, timestamp) in enumerate(examples):
                logger.info(f"  {idx+1}. ID: {trade_id}, Symbol: {symbol}, Timestamp: {timestamp}")
            
            return False  # Verification failed due to missing sentiment data
        
        # 4. Check for trades with fallback sentiment score (0.5)
        fallback_query = text(f"""
            SELECT COUNT(*) 
            FROM sentiment_analysis 
            WHERE ABS(score - {FALLBACK_SENTIMENT_SCORE}) < 0.001
        """)
        
        fallback_count = session.execute(fallback_query).scalar()
        total_fallback_pct = (fallback_count / sentiment_count * 100) if sentiment_count > 0 else 0
        
        logger.info(f"Found {fallback_count} sentiment records with fallback score 0.5 ({total_fallback_pct:.1f}%)")
        
        # 5. Check for indicator columns in TradeJournal
        symbol_with_indicators_query = text(f"""
            SELECT symbol, COUNT(*) as count 
            FROM trade_journal 
            WHERE {' AND '.join([f'{col} IS NOT NULL' for col in INDICATOR_COLUMNS])}
            GROUP BY symbol
            HAVING COUNT(*) >= {MIN_RECORDS_PER_SYMBOL}
        """)
        
        symbols_with_indicators = session.execute(symbol_with_indicators_query).fetchall()
        
        logger.info(f"Found {len(symbols_with_indicators)} symbols with complete indicator data (min {MIN_RECORDS_PER_SYMBOL} records)")
        
        # 6. Check RSI bounds
        invalid_rsi_query = text(f"""
            SELECT COUNT(*) 
            FROM trade_journal 
            WHERE rsi IS NOT NULL AND (rsi < {RSI_MIN} OR rsi > {RSI_MAX})
        """)
        
        invalid_rsi_count = session.execute(invalid_rsi_query).scalar()
        
        if invalid_rsi_count > 0:
            logger.warning(f"Found {invalid_rsi_count} trades with RSI values outside expected range [{RSI_MIN}-{RSI_MAX}]")
        else:
            logger.info("All RSI values are within expected bounds")
        
        # Final verification result - successful if we have trades and all have sentiment
        verification_success = trade_count > 0 and missing_count == 0
        logger.info(f"Database verification {'successful' if verification_success else 'failed'}")
        return verification_success
        
    except Exception as e:
        logger.error(f"Error during database verification: {e}", exc_info=True)
        return False
    finally:
        session.close()
        
if __name__ == "__main__":
    # Force a WAL checkpoint before verification
    try:
        conn = engine.raw_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA wal_checkpoint(FULL);")
        checkpoint_result = cursor.fetchone()
        logger.info(f"Initial WAL checkpoint completed: {checkpoint_result}")
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error during initial WAL checkpoint: {e}")
    
    verification_result = verify_data()
    logger.info(f"Verification completed with result: {verification_result}")
    
    if not verification_result:
        from core.news_sentiment import NewsSentimentAnalyzer
        from utils.db_setup import checkpoint_database
        
        # If verification failed, perform repair by importing required modules
        try:
            from tests.initial_population import attach_news_sentiment
            
            logger.info("Attempting to repair missing sentiment data...")
            attach_news_sentiment()
            
            # After repair, checkpoint database and verify again
            checkpoint_database()
            logger.info("Database checkpoint after repair completed")
            
            # Verify again after repair
            final_verification = verify_data()
            logger.info(f"Final verification after repair: {final_verification}")
            
        except Exception as repair_e:
            logger.error(f"Error during database repair: {repair_e}", exc_info=True)