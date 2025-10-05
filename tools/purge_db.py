from pathlib import Path
import sys
import time  # Import time
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import necessary components AFTER setting path
try:
    # Import shared engine, Base, init_db, and SessionLocal
    from utils.db_setup import engine, Base, init_db, SessionLocal, checkpoint_database
    # Import models ONLY to access __tablename__ if needed, rely on init_db for creation
    from core.db_models import TradeJournal, SentimentAnalysis
    from sqlalchemy import inspect, func, text # Import inspect, func, and text
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)

def purge_database():
    print("Dropping all tables...")
    try:
        # Drop all tables associated with the Base metadata
        Base.metadata.drop_all(bind=engine)
        print("Tables dropped.")
    except Exception as e:
        print(f"Error dropping tables: {e}")
        # Decide if we should proceed or exit
        sys.exit(1) # Exit if drop fails critically

    print("Recreating tables using init_db()...")
    try:
        # Use the centralized init_db function to recreate tables
        init_db()
        print("Tables recreated via init_db().")
    except Exception as e:
        print(f"Error recreating tables via init_db(): {e}")
        sys.exit(1) # Exit if recreation fails

    # Verification step
    print("Verification: Checking table counts...")
    session = SessionLocal()
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"Tables found after recreation: {tables}")

        tj_count = session.query(func.count(TradeJournal.id)).scalar() if 'trade_journal' in tables else 'N/A (Table Missing)'
        sa_count = session.query(func.count(SentimentAnalysis.id)).scalar() if 'sentiment_analysis' in tables else 'N/A (Table Missing)'

        print(f"Verification: TradeJournal count = {tj_count}")
        print(f"Verification: SentimentAnalysis count = {sa_count}")

        if tj_count == 0 and sa_count == 0:
            print("Verification successful: Both tables exist and are empty.")
        else:
            print("Warning: One or more tables may contain residual records.")
            print("This might indicate a problem with the purge process.")

    except Exception as e:
        print(f"Error during verification: {e}")
    finally:
        session.close()
        
    # Checkpoint database to ensure changes are persisted
    try:
        # Force a WAL checkpoint to ensure all data is written to the main db file
        print("Performing final WAL checkpoint...")
        checkpoint_database()
        print("Database checkpoint completed.")
    except Exception as e:
        print(f"Error during database checkpoint: {e}")

    # Add a small delay before exiting
    print("Adding short delay before exit...")
    time.sleep(1)

if __name__ == "__main__":
    # Check if we're running in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    print(f"Running in virtual environment: {in_venv}")
    
    # Check current working directory and database file
    cwd = os.getcwd()
    db_path = project_root / "trading_bot.db"
    db_wal_path = project_root / "trading_bot.db-wal"
    db_shm_path = project_root / "trading_bot.db-shm"
    
    print(f"Current working directory: {cwd}")
    print(f"Project root directory: {project_root}")
    print(f"Database file exists: {db_path.exists()}")
    print(f"WAL file exists: {db_wal_path.exists()}")
    print(f"SHM file exists: {db_shm_path.exists()}")
    
    user_input = input("This will PERMANENTLY DELETE all database data. Type 'YES' to continue: ")
    if user_input.strip().upper() == "YES":
        purge_database()
        print("Database purge completed successfully.")
    else:
        print("Database purge cancelled.")