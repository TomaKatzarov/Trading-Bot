import json
from collections import Counter
from pathlib import Path
import logging
import sys
import sqlite3
from math import log2

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
project_root = Path(__file__).parent.parent.absolute()
data_dir = project_root / "data" / "training data"
# Default data file, can be changed if needed
data_file = data_dir / "training_data_pnl_v1.jsonl"

def analyze_distribution(file_path: Path):
    """Reads the JSONL file and analyzes the distribution of target_signal."""
    if not file_path.exists():
        logger.error(f"Data file not found: {file_path}")
        return

    logger.info(f"Analyzing target signal distribution in: {file_path}")
    signal_counts = Counter()
    total_lines = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                try:
                    data = json.loads(line)
                    signal = data.get('target_signal')
                    if signal is not None and isinstance(signal, int):
                        signal_counts[signal] += 1
                    else:
                        logger.warning(f"Skipping line {total_lines}: Invalid or missing 'target_signal'. Content: {line.strip()}")
                except json.JSONDecodeError:
                    logger.warning(f"Skipping line {total_lines}: Invalid JSON. Content: {line.strip()}")
                except Exception as e:
                    logger.warning(f"Skipping line {total_lines}: Unexpected error ({e}). Content: {line.strip()}")

    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return

    logger.info(f"Total lines processed: {total_lines}")
    logger.info("Target Signal Distribution:")
    if not signal_counts:
        logger.warning("No valid target signals found.")
        return

    # Sort by signal class for clarity
    sorted_counts = sorted(signal_counts.items())
    total_valid_signals = sum(signal_counts.values())

    for signal, count in sorted_counts:
        percentage = (count / total_valid_signals) * 100 if total_valid_signals > 0 else 0
        logger.info(f"  Signal {signal}: {count} samples ({percentage:.2f}%)")

    logger.info(f"Total valid signals counted: {total_valid_signals}")

    # Additional metrics
    if signal_counts:
        logger.info("\nAdditional Metrics:")
        
        # 1. Class imbalance ratio (max count / min count)
        min_count = min(signal_counts.values())
        max_count = max(signal_counts.values())
        imbalance_ratio = max_count / min_count
        logger.info(f"  Class imbalance ratio: {imbalance_ratio:.2f}")
        
        # 2. Shannon entropy (measure of balance)
        total_samples = sum(signal_counts.values())
        probabilities = [count/total_samples for count in signal_counts.values()]
        entropy = -sum(p * log2(p) for p in probabilities)
        max_entropy = log2(len(signal_counts))  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy
        logger.info(f"  Entropy (balance measure): {normalized_entropy:.4f} (1.0 is perfectly balanced)")
        
        # 3. Feature vector analysis (first few samples)
        feature_lens = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 10:  # Just check first 10 lines
                        break
                    data = json.loads(line)
                    vec = data.get('context_vector', [])
                    feature_lens.append(len(vec))
            
            if feature_lens:
                unique_lens = set(feature_lens)
                if len(unique_lens) == 1:
                    logger.info(f"  Feature vector length: {next(iter(unique_lens))} (consistent)")
                else:
                    logger.info(f"  Feature vector lengths vary: {unique_lens} (inconsistent)")
                    
                # Check if the expected sentiment feature is included
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        first_sample = json.loads(f.readline())
                        if 'feature_names' in first_sample:
                            logger.info(f"  Feature names included: {first_sample['feature_names']}")
                            if 'sentiment_score' in first_sample['feature_names']:
                                logger.info("  Sentiment score is included in features")
                except:
                    pass
        except Exception as e:
            logger.warning(f"  Could not analyze feature vectors: {e}")
            
        # 4. File size and efficiency analysis
        file_size_bytes = file_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        bytes_per_sample = file_size_bytes / total_samples
        logger.info(f"  File size: {file_size_mb:.2f} MB ({bytes_per_sample:.2f} bytes per sample)")

def verify_database_counts():
    """Connects directly to the SQLite database to verify record counts and relationship integrity."""
    db_path = project_root / "trading_bot.db"
    if not db_path.exists():
        logger.error(f"Database file not found at {db_path}")
        return
    
    logger.info(f"Verifying database integrity at {db_path}")
    
    try:
        # Use direct SQLite connection to avoid any ORM issues
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get database file information
        wal_file = Path(f"{db_path}-wal")
        shm_file = Path(f"{db_path}-shm")
        
        logger.info(f"Database files:")
        logger.info(f"  Main DB: {db_path.exists()} ({db_path.stat().st_size} bytes)")
        logger.info(f"  WAL file: {wal_file.exists()} ({wal_file.stat().st_size if wal_file.exists() else 0} bytes)")
        logger.info(f"  SHM file: {shm_file.exists()} ({shm_file.stat().st_size if shm_file.exists() else 0} bytes)")
        
        # Get WAL mode status
        cursor.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0]
        
        # Get WAL checkpoint info
        cursor.execute("PRAGMA wal_checkpoint")
        checkpoint_info = cursor.fetchone()
        
        logger.info(f"Database Verification Results:")
        logger.info(f"  Journal mode: {journal_mode}")
        logger.info(f"  WAL checkpoint info: {checkpoint_info}")
        
        # Check table existence
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"  Tables found: {tables}")
        
        if 'trade_journal' not in tables or 'sentiment_analysis' not in tables:
            logger.error("CRITICAL: Required tables are missing!")
            logger.info("Attempting to rebuild missing tables...")
            conn.close()
            
            # Run the table rebuild script
            try:
                from tools.rebuild_tables import verify_and_rebuild_all_tables
                rebuild_success = verify_and_rebuild_all_tables()
                logger.info(f"Table rebuild {'succeeded' if rebuild_success else 'failed'}")
                
                # Reconnect to see the changes
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
            except ImportError:
                logger.error("Cannot find rebuild_tables module. Please create this file first.")
                return
        
        # Now check counts (after possible rebuild)
        try:
            # Check trade_journal count
            cursor.execute("SELECT COUNT(*) FROM trade_journal")
            trade_count = cursor.fetchone()[0]
            
            # Check sentiment_analysis count
            cursor.execute("SELECT COUNT(*) FROM sentiment_analysis")
            sentiment_count = cursor.fetchone()[0]
            
            # Check missing sentiment records
            cursor.execute("""
                SELECT COUNT(*) 
                FROM trade_journal tj
                LEFT JOIN sentiment_analysis sa ON tj.id = sa.trade_journal_id
                WHERE sa.id IS NULL
            """)
            missing_count = cursor.fetchone()[0]
            
            logger.info(f"  Trade records: {trade_count}")
            logger.info(f"  Sentiment records: {sentiment_count}")
            logger.info(f"  Missing sentiment: {missing_count} ({missing_count/trade_count*100:.2f}% of trades)" if trade_count > 0 else "  No trade records found")
            
            if missing_count > 0:
                logger.warning("Database integrity issue: Some trades are missing sentiment analysis")
            else:
                logger.info("Database integrity OK: All trades have sentiment analysis")
                
        except sqlite3.OperationalError as e:
            logger.error(f"Error querying tables: {e}")
        
        conn.close()
        
    except sqlite3.Error as e:
        logger.error(f"SQLite error during verification: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during verification: {e}")

if __name__ == "__main__":
    # Check if we should verify the database instead of analyzing JSONL
    if len(sys.argv) > 1 and sys.argv[1] == "--verify-db":
        verify_database_counts()
    else:
        analyze_distribution(data_file)