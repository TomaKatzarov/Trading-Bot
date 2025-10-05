#!/usr/bin/env python
# rebuild_tables.py - Emergency table recovery utility
import logging
import sys
from pathlib import Path
from sqlalchemy import inspect, text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import critical components with proper order
from utils.db_setup import engine, Base, SessionLocal, checkpoint_database
# Import models after Base to ensure proper registration
from core.db_models import TradeJournal, SentimentAnalysis

def direct_create_sentiment_table():
    """
    Directly creates the sentiment_analysis table if missing.
    This is a targeted emergency recovery function.
    """
    logger.info("Starting direct sentiment_analysis table creation...")
    
    session = SessionLocal()
    try:
        # Check if table exists
        inspector = inspect(engine)
        exists = 'sentiment_analysis' in inspector.get_table_names()
        
        if exists:
            logger.info("sentiment_analysis table already exists, no action needed.")
            return True
        
        # Table doesn't exist, create it directly to ensure it matches model
        logger.warning("sentiment_analysis table missing! Creating it now...")
        
        # Create only the sentiment_analysis table
        SentimentAnalysis.__table__.create(engine, checkfirst=True)
        
        # Verify creation
        inspector = inspect(engine)
        created = 'sentiment_analysis' in inspector.get_table_names()
        
        if created:
            logger.info("Successfully created sentiment_analysis table!")
            
            # Force WAL checkpoint to ensure persistence
            session.execute(text("PRAGMA wal_checkpoint(TRUNCATE);"))
            session.commit()
            
            return True
        else:
            logger.error("Failed to create sentiment_analysis table!")
            return False
            
    except Exception as e:
        logger.error(f"Error creating sentiment table: {e}", exc_info=True)
        return False
    finally:
        session.close()
        engine.dispose()

def verify_and_rebuild_all_tables():
    """
    Checks for all required tables and rebuilds any that are missing.
    Preserves existing data in tables that are present.
    """
    logger.info("Starting database table verification and recovery...")
    
    # First, check what tables exist
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    
    logger.info(f"Found existing tables: {existing_tables}")
    
    # Required tables based on our models
    required_tables = {'trade_journal', 'sentiment_analysis'}
    missing_tables = required_tables - existing_tables
    
    if not missing_tables:
        logger.info("All required tables exist. No recovery needed.")
        return True
    
    logger.warning(f"Missing tables detected: {missing_tables}")
    
    # If tables are missing, we need to create them
    # But first ensure Base has all models registered
    known_models = {table_name: table_obj for table_name, table_obj in Base.metadata.tables.items()}
    logger.info(f"Models registered with Base: {list(known_models.keys())}")
    
    # Create only the missing tables to preserve existing data
    try:
        for table_name in missing_tables:
            if table_name == 'sentiment_analysis':
                direct_create_sentiment_table()
            elif table_name == 'trade_journal':
                logger.warning("trade_journal table missing! Creating it...")
                TradeJournal.__table__.create(engine, checkfirst=True)
            else:
                logger.warning(f"Unknown table {table_name} requested for creation")
        
        # Verify all tables now exist
        inspector = inspect(engine)
        final_tables = set(inspector.get_table_names())
        still_missing = required_tables - final_tables
        
        if still_missing:
            logger.error(f"Failed to create some tables: {still_missing}")
            return False
        else:
            logger.info("All required tables successfully created/verified!")
            
            # Force checkpoint
            checkpoint_database()
            return True
            
    except Exception as e:
        logger.error(f"Error during table recovery: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Database Table Recovery Utility")
    logger.info("This script will check and rebuild missing database tables")
    
    # Force WAL checkpoint before we start
    session = SessionLocal()
    try:
        session.execute(text("PRAGMA wal_checkpoint(FULL);"))
        result = session.execute(text("PRAGMA journal_mode;")).fetchone()
        logger.info(f"Current journal mode: {result}")
    except Exception as e:
        logger.error(f"Error during initial checkpoint: {e}")
    finally:
        session.close()
    
    # Run the verification and rebuild process
    success = verify_and_rebuild_all_tables()
    
    if success:
        logger.info("Database recovery completed successfully")
    else:
        logger.error("Database recovery failed")