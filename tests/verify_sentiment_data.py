# filepath: c:\TradingBotAI\verify_sentiment_data.py
import os
import sys
from pathlib import Path
import logging
import subprocess
import time

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / "sentiment_verification.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("sentiment_verifier")

def main():
    logger.info("Starting sentiment data verification...")
    
    # Check if trading_bot.db exists
    db_path = project_root / "trading_bot.db"
    wal_path = project_root / "trading_bot.db-wal"
    shm_path = project_root / "trading_bot.db-shm"
    
    if not db_path.exists():
        logger.error(f"Database file not found at {db_path}")
        return
    
    logger.info(f"Database file found: {db_path}")
    logger.info(f"WAL file exists: {wal_path.exists()}")
    logger.info(f"SHM file exists: {shm_path.exists()}")
    
    # Import required modules for verification
    try:
        from utils.db_setup import verify_database_integrity, checkpoint_database
        
        # First, ensure any WAL changes are checkpointed
        logger.info("Running initial database checkpoint...")
        checkpoint_database()
        logger.info("Initial checkpoint completed")
        
        # Check if sentiment data exists
        sentiment_count, total_count = verify_database_integrity()
        logger.info(f"Verification results: {sentiment_count}/{total_count} trades have sentiment data")
        
        if sentiment_count < total_count:
            logger.warning(f"Missing sentiment data detected: {total_count - sentiment_count} trades need repair")
            
            # Ask user if they want to run repair
            user_input = input(f"Found {total_count - sentiment_count} trades missing sentiment data. Run repair? (y/n): ")
            
            if user_input.lower() == 'y':
                logger.info("User confirmed repair. Running tests/verify_db_data.py for automatic repair...")
                
                # Run the verification and repair script
                try:
                    verify_script = project_root / "tests" / "verify_db_data.py"
                    logger.info(f"Running {verify_script}")
                    
                    result = subprocess.run([sys.executable, str(verify_script)], 
                                           capture_output=True, text=True, check=False)
                    
                    logger.info(f"Repair script stdout: {result.stdout}")
                    if result.stderr:
                        logger.warning(f"Repair script stderr: {result.stderr}")
                    
                    # Check if repair was successful
                    sentiment_count, total_count = verify_database_integrity()
                    logger.info(f"Post-repair verification: {sentiment_count}/{total_count} trades have sentiment data")
                    
                    if sentiment_count == total_count:
                        logger.info("✅ Repair successful! All sentiment data is now present.")
                        print("\n✅ Repair successful! All sentiment data is now present.\n")
                    else:
                        logger.error(f"❌ Repair partially successful. Still missing sentiment for {total_count - sentiment_count} trades.")
                        print(f"\n❌ Repair partially successful. Still missing sentiment for {total_count - sentiment_count} trades.\n")
                        
                except Exception as e:
                    logger.error(f"Error running repair script: {e}", exc_info=True)
            else:
                logger.info("User declined repair. No changes made.")
        else:
            logger.info("✅ All sentiment data is present! No repair needed.")
            print("\n✅ All sentiment data is present! No repair needed.\n")
            
    except Exception as e:
        logger.error(f"Error during verification: {e}", exc_info=True)
        print(f"\n❌ Error during verification: {str(e)}\n")
    
    # Final checkpoint to ensure all changes are written
    try:
        from utils.db_setup import checkpoint_database
        logger.info("Performing final database checkpoint...")
        checkpoint_database()
        logger.info("Final checkpoint completed.")
    except Exception as e:
        logger.error(f"Error during final checkpoint: {e}")

if __name__ == "__main__":
    print("\n=== SQLite Sentiment Data Verification Tool ===\n")
    print("This tool will check if sentiment analysis data is intact and repair if needed.")
    main()
    print("\nVerification complete. See sentiment_verification.log for details.")
    time.sleep(1)  # Give user time to read the message