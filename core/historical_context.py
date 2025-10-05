# core/historical_context.py
import logging
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import numpy as np
import os
from sqlalchemy import Column, DateTime, Float, String, Text, Integer
from sqlalchemy.orm import relationship
from utils.db_setup import Base
from core.db_models import TradeJournal  # Import the model from core/db_models
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from utils.db_setup import SessionLocal
import uuid
import json

# Load .env with explicit path
env_path = Path(__file__).resolve().parent.parent / "Credential.env"
load_dotenv(dotenv_path=env_path, override=True)
logger = logging.getLogger(__name__)


class HistoricalContext:
    def __init__(self, data_dir='data/historical', rsi_period=14, n_neighbors=3):
        """
        Initialize historical context system with a directory of parquet files,
        the RSI period, and the number of neighbors for similarity searches.
        """
        self.data_dir = data_dir
        self.rsi_period = rsi_period
        self.n_neighbors = n_neighbors
        self.df = self._load_data()
        self.scaler = None
        self.nn = None
        self._prepare_system()

    def _calculate_rsi(self, close_prices):
        """
        Calculate the Relative Strength Index (RSI) over the given close prices.
        If there is no loss (i.e. loss==0), the RSI is set to 100.
        """
        delta = close_prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=self.rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=1).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rs = rs.fillna(100)  # When loss is zero, assign an RSI of 100.
        return 100 - (100 / (1 + rs))

    def _load_data(self):
        """Load and process parquet files."""
        all_files = []
        for root, _, files in os.walk(self.data_dir):
            for f in files:
                if f.endswith('.parquet'):
                    all_files.append((os.path.join(root, f), os.path.basename(os.path.dirname(root))))
        
        if not all_files:
            logger.error(f"No parquet files found in {self.data_dir}")
            return pd.DataFrame()

        dfs = []
        for file_path, symbol in all_files:
            try:
                # Read parquet file and preserve the index
                df = pd.read_parquet(file_path)
                # Convert index to timestamp column
                df = df.reset_index()
                df.rename(columns={'index': 'timestamp'}, inplace=True)
                df['Symbol'] = symbol
                if not {"Close", "Volume"}.issubset(df.columns):
                    logger.error(f"Missing required columns in {file_path}")
                    continue
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {str(e)}")

        if not dfs:
            logger.error("No data could be loaded from parquet files.")
            return pd.DataFrame()

        # Combine all DataFrames and ensure timestamp column exists
        combined = pd.concat(dfs, ignore_index=True)
        
        # Calculate additional features
        combined['RSI'] = self._calculate_rsi(combined['Close'])
        combined['returns_1h'] = combined['Close'].pct_change()
        combined['volume_pct'] = combined['Volume'].pct_change()
        combined = combined.dropna()
        
        logger.info(f"Loaded historical data with shape: {combined.shape}")
        return combined

    def _prepare_system(self):
        """
        Prepare the feature scaling and nearest neighbors system using the calculated RSI
        and volume percentage change as features.
        """
        features = self.df[['RSI', 'volume_pct']].values
        self.scaler = StandardScaler().fit(features)
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.scaler.transform(features))
        logger.info("Historical context system prepared with NearestNeighbors and StandardScaler.")

    # MUST INCLUDE THIS METHOD
    def find_similar(self, current_rsi: float, current_volume_pct: float):
        """
        Find similar historical data rows based on the current RSI and volume percentage.
        Returns a DataFrame subset containing the timestamp, RSI, Volume, and 1-hour returns.
        """
        query = np.array([[current_rsi, current_volume_pct]])
        scaled_query = self.scaler.transform(query)
        distances, indices = self.nn.kneighbors(scaled_query)
        return self.df.iloc[indices[0]][['timestamp', 'RSI', 'Volume', 'returns_1h']]

    def run_and_upload_data(self):
        """Upload processed data to TradeJournal table."""
        session = SessionLocal()
        try:
            for idx, row in self.df.iterrows():
                # Get timestamp from index if it's not in columns
                if 'timestamp' in row.index:
                    timestamp = pd.to_datetime(row['timestamp'])
                else:
                    timestamp = pd.to_datetime(row.name)  # Use index as timestamp
                
                trade_id = f"hist_{uuid.uuid4()}"
                journal_entry = TradeJournal(
                    id=trade_id,
                    timestamp=timestamp,
                    symbol=row['Symbol'],
                    status="historical",
                    open_price=row.get('Open'),
                    high_price=row.get('High'),
                    low_price=row.get('Low'),
                    close_price=row.get('Close'),
                    volume=row.get('Volume'),
                    wap=row.get('VWAP'),
                    returns=row.get('Returns'),
                    hl_diff=row.get('HL_diff'),
                    ohlc_avg=row.get('OHLC_avg'),
                    rsi=row.get('RSI'),
                    volume_pct=row.get('volume_pct'),
                    returns_1h=row.get('returns_1h'),
                    checksum=str(uuid.uuid4())
                )
                session.add(journal_entry)
                
                if idx > 0 and idx % 1000 == 0:
                    session.commit()
                    logger.info(f"Committed {idx} rows...")

            session.commit()
            logger.info(f"Uploaded {len(self.df)} historical rows to the DB.")
        except Exception as e:
            logger.error(f"Error uploading historical data: {e}")
            session.rollback()
            raise
        finally:
            session.close()