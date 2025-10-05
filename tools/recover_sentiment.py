import logging
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.db_setup import SessionLocal, engine, checkpoint_database
from core.db_models import TradeJournal, SentimentAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def recover_missing_sentiment():
    """Check for missing sentiment records and rerun sentiment analysis if needed."""
    logger.info("Checking for missing sentiment data...")
    
    session = SessionLocal()
    try:
        # Find trades missing sentiment
        missing_sentiment_query = session.query(TradeJournal).outerjoin(
            SentimentAnalysis, TradeJournal.id == SentimentAnalysis.trade_journal_id
        ).filter(SentimentAnalysis.id == None)
        
        missing_count = missing_sentiment_query.count()
        
        if missing_count == 0:
            logger.info("No missing sentiment data found. Database is complete.")
            return True
            
        logger.warning(f"Found {missing_count} trades missing sentiment data. Running recovery...")
        
        # Import here to avoid circular imports
        from tests.initial_population import attach_news_sentiment
        
        # Close current session before running attach_news_sentiment
        session.close()
        
        # Run sentiment attachment
        attach_news_sentiment()
        
        # Verify recovery
        checkpoint_database()
        
        return True
        
    except Exception as e:
        logger.error(f"Error during sentiment recovery: {e}", exc_info=True)
        return False
    finally:
        if session:
            session.close()

if __name__ == "__main__":
    recover_missing_sentiment()