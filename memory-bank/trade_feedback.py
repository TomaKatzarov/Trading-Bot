# trade_feedback.py
import logging
from datetime import datetime, timedelta
import numpy as np
import os
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from core.db_models import TradeJournal

# Load environment variables from Credential.env
env_path = Path(__file__).resolve().parent.parent / "Credential.env"
load_dotenv(dotenv_path=env_path, override=True)

logger = logging.getLogger(__name__)

# Configure the database URL; using SQLite for simplicity.
DATABASE_URL = os.getenv("TRADE_DB_URL", "sqlite:///trades.db")
engine = create_engine(DATABASE_URL, echo=False)
SessionFactory = sessionmaker(bind=engine)
Session = scoped_session(SessionFactory)

class TradeFeedbackAnalyzer:
    def __init__(self):
        # Use a scoped session to manage session lifecycle automatically.
        self.session = Session()

    def get_recent_trades(self, days: int = 7):
        """
        Retrieve trade records from the last 'days' days from the local database.
        """
        since = datetime.utcnow() - timedelta(days=days)
        try:
            trades = self.session.query(TradeJournal).filter(TradeJournal.timestamp >= since).all()
            logger.info(f"Retrieved {len(trades)} trades from the last {days} days")
            return trades
        except Exception as e:
            logger.error(f"Error retrieving trades: {e}")
            return []

    def calculate_performance_metrics(self, trades):
        """
        Compute aggregated performance metrics from the trade records.
        Metrics include the average realized/unrealized PnL and win rate.
        """
        realized = [t.pnl_realized for t in trades if t.pnl_realized is not None]
        unrealized = [t.pnl_unrealized for t in trades if t.pnl_unrealized is not None]
        metrics = {
            "num_trades": len(trades),
            "avg_realized_pnl": np.mean(realized) if realized else 0,
            "avg_unrealized_pnl": np.mean(unrealized) if unrealized else 0,
            "win_rate": np.mean([1 if pnl > 0 else 0 for pnl in realized]) if realized else 0
        }
        logger.info(f"Calculated performance metrics: {metrics}")
        return metrics

    def analyze_and_report(self, days: int = 7):
        """
        Retrieve, calculate, and report trade performance metrics.
        Returns a dictionary with the metrics.
        """
        trades = self.get_recent_trades(days=days)
        if not trades:
            logger.warning("No recent trades found for performance analysis.")
            return None
        metrics = self.calculate_performance_metrics(trades)
        return metrics

if __name__ == "__main__":
    analyzer = TradeFeedbackAnalyzer()
    performance_metrics = analyzer.analyze_and_report(days=7)
    if performance_metrics:
        print("Local Trade Performance Metrics:", performance_metrics)