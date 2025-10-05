# db_setup.py
import logging
# Original import moved below
from sqlalchemy import create_engine, event, inspect # Import inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import sys
from pathlib import Path

# Add project root to Python path if necessary
# Assuming this file is in utils/ relative to project root
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Configure logger
logger = logging.getLogger(__name__)
# Ensure logging is configured only once if run multiple times or imported
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

DATABASE_URL = f"sqlite:///{project_root.as_posix()}/trading_bot.db" # Absolute path to DB file
# Use connect_args for thread check, listeners for PRAGMA
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# --- Add Event Listener for journal_mode ---
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    try:
        # Change to WAL mode for better crash recovery
        cursor.execute("PRAGMA journal_mode=WAL;")
        # Use FULL synchronous for maximum data protection
        cursor.execute("PRAGMA synchronous=FULL;")
        # Add foreign key enforcement
        cursor.execute("PRAGMA foreign_keys=ON;")
        # Set busy timeout to prevent "database is locked" errors
        cursor.execute("PRAGMA busy_timeout=10000;")
        logger.info("SQLite PRAGMAs set: journal_mode=WAL, synchronous=FULL, foreign_keys=ON, busy_timeout=10000")
    except Exception as e:
        logger.error(f"Failed to set PRAGMAs: {e}")
    finally:
        cursor.close()
# --- End Event Listener ---

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

# Initialize database on module import
def init_db():
    """Initialize database, ensuring models are registered and tables created."""
    logger.info("Attempting database initialization...")
    try:
        # --- Explicitly import all models HERE to ensure registration ---
        try:
            # Import the module containing model definitions
            from core import db_models
            logger.info("Successfully imported core.db_models for metadata registration.")
            # Verify Base has tables registered AFTER import
            if not Base.metadata.tables:
                 logger.warning("No SQLAlchemy models seem registered with Base metadata after import.")
            else:
                 logger.info(f"Registered tables: {list(Base.metadata.tables.keys())}")
        except ImportError as e:
             logger.error(f"Could not import core.db_models in init_db: {e}. Tables might not be created correctly.", exc_info=True)
             # Allow create_all to proceed, but log the potential issue
        except Exception as e:
             logger.error(f"Unexpected error during model import for DB init: {e}", exc_info=True)


        # Create tables. create_all checks for existence by default.
        logger.info("Executing Base.metadata.create_all(bind=engine)...")
        Base.metadata.create_all(bind=engine)
        logger.info("Base.metadata.create_all execution finished.")

        # Verify tables were created using inspect
        inspector = inspect(engine)
        current_tables = inspector.get_table_names()
        logger.info(f"Tables found via inspect: {current_tables}")
        required_tables = ['trade_journal', 'sentiment_analysis'] # Add all required tables
        missing_tables = [table for table in required_tables if table not in current_tables]
        if missing_tables:
            logger.error(f"Verification failed: Missing tables after create_all: {missing_tables}")
            # raise RuntimeError(f"Database initialization failed to create tables: {missing_tables}")
        else:
            logger.info("Verification successful: All required tables exist.")

    except Exception as e:
        logger.critical(f"Critical database initialization error during create_all or verification: {e}", exc_info=True)
        # Re-raise to ensure failure is propagated
        raise RuntimeError("Database initialization failed critically.") from e
    finally:
        logger.info("Database initialization attempt finished.")

def checkpoint_database():
    """Force a WAL checkpoint before shutdown to ensure data is written to main db file."""
    try:
        # First dispose any existing connections to ensure no transactions are active
        engine.dispose()
        logger.info("All existing database connections disposed before checkpoint")
        
        # Create a new connection for the checkpoint
        conn = engine.raw_connection()
        cursor = conn.cursor()
        
        # First try a PASSIVE checkpoint which doesn't block other connections
        cursor.execute("PRAGMA wal_checkpoint(PASSIVE);")
        passive_result = cursor.fetchone()
        logger.info(f"PASSIVE database checkpoint result: {passive_result}")
        
        # Then do a FULL checkpoint which ensures complete sync
        cursor.execute("PRAGMA wal_checkpoint(FULL);")
        checkpoint_result = cursor.fetchone()
        
        # If FULL checkpoint indicates pending frames, try RESTART mode
        if checkpoint_result and checkpoint_result[0] > 0:
            logger.warning(f"FULL checkpoint has {checkpoint_result[0]} frames pending. Trying RESTART...")
            cursor.execute("PRAGMA wal_checkpoint(RESTART);")
            restart_result = cursor.fetchone()
            logger.info(f"RESTART checkpoint result: {restart_result}")
        
        # Finally do a TRUNCATE checkpoint to minimize WAL file size
        cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        truncate_result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        logger.info(f"Database checkpoints completed: FULL={checkpoint_result}, TRUNCATE={truncate_result}")
        
        # Report WAL file size for diagnostics
        db_path = Path(DATABASE_URL.replace("sqlite:///", ""))
        wal_path = Path(f"{db_path}-wal")
        if wal_path.exists():
            wal_size = wal_path.stat().st_size
            logger.info(f"WAL file size after checkpoint: {wal_size} bytes")
    except Exception as e:
        logger.error(f"Error during database checkpoint: {e}", exc_info=True)
    finally:
        # Safely dispose engine connections
        engine.dispose()
        logger.info("Database engine disposed.")

def verify_database_integrity():
    """Verify database integrity and check for missing sentiment records."""
    session = SessionLocal()
    try:
        # Check for trades missing sentiment analysis
        from core.db_models import TradeJournal, SentimentAnalysis
        total_trades = session.query(TradeJournal).count()
        trades_with_sentiment = session.query(TradeJournal).join(
            SentimentAnalysis, TradeJournal.id == SentimentAnalysis.trade_journal_id
        ).count()
        
        logger.info(f"Database integrity check: {trades_with_sentiment}/{total_trades} trades have sentiment data")
        return trades_with_sentiment, total_trades
    except Exception as e:
        logger.error(f"Error during database integrity check: {e}")
        return 0, 0
    finally:
        session.close()
        
# Removed auto-init to allow explicit initialization by calling scripts
# init_db()