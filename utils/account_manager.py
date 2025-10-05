# account_manager.py
import os
import sys
import logging
import datetime
import uuid
import pytz
from pathlib import Path
from utils.db_setup import SessionLocal
from utils.audit_models import AuditLog
from typing import Optional, Dict, List
import json

from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Text
from sqlalchemy.orm import sessionmaker, declarative_base

from dotenv import load_dotenv

# Import the TradeJournal model (used for historical trades)
from core.db_models import TradeJournal

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load environment variables from Credential.env (using an absolute path)
env_path = Path(__file__).resolve().parent.parent / "Credential.env"
load_dotenv(dotenv_path=env_path, override=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define the Base for SQLAlchemy models.
Base = declarative_base()

# -------------------------
# Persistent Position Model
# -------------------------
class Position(Base):
    __tablename__ = 'positions'
    symbol = Column(String, primary_key=True, index=True)
    trade_id = Column(String, unique=True, nullable=False)
    paper_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    opened_at = Column(DateTime, default=datetime.datetime.utcnow)

# -------------------------
# AccountManager Definition
# -------------------------
class AccountManager:
    """
    AccountManager tracks the starting balance, current available funds, open positions,
    alerts, and provides helper methods for the UI to retrieve or update account info.
    Uses SQLAlchemy for persistent storage of account state.
    """
    def set_decision_engine(self, engine):
        """Attach a DecisionEngine instance so that manual closures are routed properly."""
        self.decision_engine = engine
        logger.info("Decision engine attached to AccountManager.")
    
    def __init__(self, starting_balance: float = 100000.0):
        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        self.equity = starting_balance
        self.current_symbol: Optional[str] = None  # Active trading symbol
        self.alerts: List[Dict] = []  # In-memory alerts for UI visualization
        self._init_db()
        self.open_positions = self._load_positions()  # Load persistent positions from DB
        logger.info(f"Initialized AccountManager with starting balance: {starting_balance}")

    def _init_db(self):
        # Create engine and bind session
        engine = create_engine('sqlite:///trading_bot.db')
        Base.metadata.create_all(engine)
        self.Session = sessionmaker(bind=engine)

    def _load_positions(self) -> Dict[str, Dict]:
        """
        Load open positions from the persistent database.
        Returns a dictionary mapping symbol to a position dictionary.
        """
        session = self.Session()
        try:
            positions = {}
            for p in session.query(Position).all():
                positions[p.symbol] = {
                    "trade_id": p.trade_id,
                    "paper_price": p.paper_price,
                    "quantity": p.quantity,
                    "timestamp": p.opened_at.isoformat()
                }
            return positions
        finally:
            session.close()

    def add_position(self, trade_record: Dict):
        """
        Record a new open position using a trade record dictionary.
        trade_record should include keys: symbol, trade_id, paper_price, quantity, timestamp.
        Stores the position both in memory and in the persistent database.
        """
        symbol = trade_record.get("symbol")
        if symbol:
            self.open_positions[symbol] = trade_record
            logger.info(f"Added open position for {symbol}: {trade_record}")
            # Persist the new position to the DB
            session = self.Session()
            try:
                pos = Position(
                    symbol=symbol,
                    trade_id=trade_record.get("trade_id"),
                    paper_price=trade_record.get("paper_price"),
                    quantity=trade_record.get("quantity"),
                    opened_at=datetime.datetime.utcnow()
                )
                session.add(pos)
                session.commit()
            except Exception as e:
                logger.error(f"Error persisting position for {symbol}: {e}")
                session.rollback()
            finally:
                session.close()
        else:
            logger.error("Trade record missing 'symbol' key.")

    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Retrieve the open position for a given symbol.
        Returns None if no open position exists.
        """
        return self.open_positions.get(symbol)

    def get_all_open_positions(self) -> Dict[str, Dict]:
        """
        Returns all currently open positions as a dictionary.
        """
        return self.open_positions

    def close_position(self, symbol: str, exit_price: float, manual: bool = False, reason: Optional[str] = None) -> float:
        """
        Close the open position for the given symbol.
        Computes PnL and updates the current balance.
        Also removes the position from persistent storage.
        Returns the computed PnL.
        """
        if symbol in self.open_positions:
            position = self.open_positions.pop(symbol)
            entry_price = position.get("paper_price", 0.0)
            quantity = position.get("quantity", 0)
            pnl = (exit_price - entry_price) * quantity
            self.current_balance += pnl
            logger.info(f"Closed position for {symbol}: Entry {entry_price}, Exit {exit_price}, "
                        f"Qty {quantity}, PnL {pnl:.2f}. New balance: {self.current_balance:.2f}")
            if manual:
                self._log_audit(
                    action="MANUAL_CLOSE",
                    details={
                        "symbol": symbol,
                        "exit_price": exit_price,
                        "reason": reason,
                        "pnl": pnl
                    }
                )
            # Remove the position from the DB.
            session = self.Session()
            try:
                pos_obj = session.query(Position).filter(Position.symbol == symbol).first()
                if pos_obj:
                    session.delete(pos_obj)
                    session.commit()
            except Exception as e:
                logger.error(f"Error removing position {symbol} from DB: {e}")
                session.rollback()
            finally:
                session.close()
            return pnl
        else:
            logger.warning(f"No open position found for {symbol} to close.")
            return 0.0

    def manual_close_position(self, symbol: str, exit_price: float, reason: str) -> float:
        """
        Trigger a manual closure through the attached DecisionEngine.
        This ensures that the Alpaca API is called and the TradeJournal is updated.
        """
        if hasattr(self, 'decision_engine') and self.decision_engine is not None:
            return self.decision_engine.manual_close_position(symbol, exit_price, reason)
        else:
            logger.error("No decision engine attached, cannot process manual close.")
            return 0.0

    def get_current_balance(self) -> float:
        """
        Return the current available balance.
        """
        return self.current_balance

    def _log_audit(self, action: str, details: Dict):
        """Log audit trail entries"""
        session = SessionLocal()
        try:
            entry = AuditLog(
                timestamp=datetime.datetime.utcnow(),
                action_type=action,
                details=json.dumps(details),
                account_balance=self.current_balance
            )
            session.add(entry)
            session.commit()
        except Exception as e:
            logger.error(f"Audit log failed: {e}")
            session.rollback()
        finally:
            session.close()

    def update_balance(self, new_balance: float, reason: Optional[str] = None):
        """
        Update the account balance manually with audit logging.
        """
        prev_balance = self.current_balance
        self.current_balance = new_balance
        logger.info(f"Account balance manually updated from {prev_balance} to {new_balance}")
        self._log_audit(
            action="BALANCE_ADJUST",
            details={
                "previous": prev_balance,
                "new": new_balance,
                "reason": reason
            }
        )

    def get_historical_trades(self, days: int = 30) -> List:
        """
        Retrieve historical trades from the DB (using SessionLocal) over the past 'days' days.
        Useful for the UI to display past trading decisions.
        """
        since = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        session = SessionLocal()
        try:
            trades = session.query(TradeJournal).filter(TradeJournal.timestamp >= since).all()
            return trades
        except Exception as e:
            logger.error(f"Error fetching historical trades: {e}")
            return []
        finally:
            session.close()

    def log_trading_decision(self, symbol: str, decision: str, details: Optional[Dict] = None) -> Dict:
        """
        Log a trading decision (e.g., BUY, SELL, HOLD) along with any additional details.
        Can be used for UI visualization.
        """
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "symbol": symbol,
            "decision": decision,
            "details": details
        }
        logger.info(f"Trading decision: {log_entry}")
        return log_entry

    def calculate_profit_loss(self, days: int = 30, pnl_type: str = "realized") -> Dict:
        """
        Calculate P/L for a specified time window and type.
        For 'realized' pnl, only closed trades are considered;
        for 'unrealized', open trades are used.
        """
        since = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        session = SessionLocal()
        try:
            query = session.query(TradeJournal).filter(TradeJournal.timestamp >= since)
            if pnl_type == "realized":
                query = query.filter(TradeJournal.status == "closed")
            elif pnl_type == "unrealized":
                query = query.filter(TradeJournal.status == "open")
            trades = query.all()
            total_pnl = sum(trade.pnl_realized or 0.0 for trade in trades)
            return {
                "period": f"{days} days",
                "type": pnl_type,
                "total_pnl": total_pnl,
                "trade_count": len(trades)
            }
        except Exception as e:
            logger.error(f"P/L calculation failed: {e}")
            return {"error": str(e)}
        finally:
            session.close()

    def get_audit_logs(self, days: int = 7) -> List[Dict]:
        """Retrieve audit logs for UI display."""
        session = SessionLocal()
        since = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        try:
            logs = session.query(AuditLog).filter(AuditLog.timestamp >= since).all()
            return [{
                "timestamp": log.timestamp.isoformat(),
                "action": log.action_type,
                "details": json.loads(log.details),
                "balance": log.account_balance
            } for log in logs]
        except Exception as e:
            logger.error(f"Audit log retrieval failed: {e}")
            return []
        finally:
            session.close()

    # -------------------------
    # Additional Account Functions
    # -------------------------
    def deposit_funds(self, amount: float, reason: Optional[str] = None):
        """
        Increase the current balance by a deposit amount.
        """
        if amount <= 0:
            logger.error("Deposit amount must be positive.")
            return
        self.current_balance += amount
        logger.info(f"Deposited ${amount:.2f}. New balance: {self.current_balance:.2f}")
        self._log_audit("DEPOSIT", {"amount": amount, "reason": reason})

    def withdraw_funds(self, amount: float, reason: Optional[str] = None):
        """
        Withdraw funds from the current balance.
        """
        if amount <= 0:
            logger.error("Withdrawal amount must be positive.")
            return
        if amount > self.current_balance:
            logger.error("Insufficient funds for withdrawal.")
            return
        self.current_balance -= amount
        logger.info(f"Withdrew ${amount:.2f}. New balance: {self.current_balance:.2f}")
        self._log_audit("WITHDRAWAL", {"amount": amount, "reason": reason})

    def get_closed_positions(self, days: int = 365) -> List[Dict]:
        """
        Retrieve closed positions (from the TradeJournal) for the past 'days' days.
        """
        since = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        session = SessionLocal()
        try:
            positions = session.query(TradeJournal).filter(
                TradeJournal.timestamp >= since,
                TradeJournal.status == "closed"
            ).all()
            closed_positions = []
            for pos in positions:
                closed_positions.append({
                    "id": pos.id,
                    "timestamp": pos.timestamp.isoformat(),
                    "symbol": pos.symbol,
                    "action": pos.action,
                    "quantity": pos.quantity,
                    "paper_price": pos.paper_price,
                    "real_price": pos.real_price,
                    "pnl": pos.pnl_realized,
                    "model_metadata": pos.model_metadata,
                    "checksum": pos.checksum
                })
            return closed_positions
        except Exception as e:
            logger.error(f"Error retrieving closed positions: {e}")
            return []
        finally:
            session.close()

    def compute_open_positions_pl(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Compute unrealized profit/loss for each open position given a mapping of current prices.
        Returns a dictionary mapping symbol to its computed P/L.
        """
        pl_dict = {}
        for symbol, pos in self.open_positions.items():
            current_price = current_prices.get(symbol)
            if current_price is None:
                logger.warning(f"Current price for {symbol} not provided; skipping P/L calculation.")
                continue
            entry_price = pos.get("paper_price", 0.0)
            quantity = pos.get("quantity", 0)
            pnl = (current_price - entry_price) * quantity
            pl_dict[symbol] = pnl
        return pl_dict

    def reset_metrics(self):
        """
        Reset all in-memory account metrics to their initial state and clear open positions from the database.
        Note: Historical trades remain preserved.
        """
        self.current_balance = self.starting_balance
        self.open_positions = {}
        logger.info("Reset account metrics to initial state.")
        self._log_audit("RESET_METRICS", {"reset_to": self.starting_balance})
        session = self.Session()
        try:
            session.query(Position).delete()
            session.commit()
        except Exception as e:
            logger.error(f"Failed to reset positions in DB: {e}")
            session.rollback()
        finally:
            session.close()

    def get_account_summary(self, current_prices: Optional[Dict[str, float]] = None, days: int = 365) -> Dict:
        """
        Return a summary of account metrics including current balance, open positions,
        historical trades, and (if provided) unrealized P/L on open positions.
        """
        summary = {
            "starting_balance": self.starting_balance,
            "current_balance": self.current_balance,
            "open_positions_count": len(self.open_positions),
            "open_positions": self.open_positions,
            "historical_trades": [trade.__dict__ for trade in self.get_historical_trades(days=days)],
            "closed_positions": self.get_closed_positions(days=days),
            "profit_loss": self.calculate_profit_loss(days=days, pnl_type="realized")
        }
        if current_prices:
            open_pl = self.compute_open_positions_pl(current_prices)
            summary["open_positions_pl"] = open_pl
            summary["total_open_pl"] = sum(open_pl.values())
        return summary

    def set_trading_symbol(self, symbol: str):
        """
        Set or update the currently active trading symbol.
        """
        self.current_symbol = symbol
        logger.info(f"Set current trading symbol to {symbol}")

    def add_alert(self, alert: Dict):
        """
        Add an alert for visualization in the UI.
        """
        self.alerts.append(alert)
        logger.info(f"Alert added: {alert}")

    def get_recent_alerts(self, days: int = 7) -> List[Dict]:
        """
        Retrieve recent alerts by combining in-memory alerts and trade journal entries.
        """
        alerts = list(self.alerts)  # Copy in-memory alerts
        since = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        session = SessionLocal()
        try:
            trades = session.query(TradeJournal).filter(TradeJournal.timestamp >= since).all()
            for trade in trades:
                if trade.action in ["BUY", "SELL", "MANUAL_SELL"]:
                    alerts.append({
                        "timestamp": trade.timestamp.isoformat(),
                        "symbol": trade.symbol,
                        "action": trade.action,
                        "quantity": trade.quantity,
                        "price": trade.paper_price,
                        "model_metadata": trade.model_metadata
                    })
            return alerts
        except Exception as e:
            logger.error(f"Error retrieving recent alerts: {e}")
            return alerts
        finally:
            session.close()

    def get_metrics(self) -> Dict:
        """
        Return basic account metrics.
        """
        metrics = {
            "starting_balance": self.starting_balance,
            "current_balance": self.current_balance,
            "open_positions_count": len(self.open_positions),
            "open_positions": self.open_positions
        }
        return metrics


# For Module Testing
if __name__ == "__main__":
    manager = AccountManager(starting_balance=100000.0)
    
    sample_trade = {
        "symbol": "AAPL",
        "trade_id": str(uuid.uuid4()),
        "paper_price": 150.0,
        "quantity": 10,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    manager.add_position(sample_trade)
    
    pos = manager.get_position("AAPL")
    print("Open position for AAPL:", pos)
    
    manager.update_balance(120000.0)
    print("Current balance:", manager.get_current_balance())
    
    pnl = manager.close_position("AAPL", exit_price=155.0)
    print("PnL from closing AAPL position:", pnl)
    print("Current balance after closing:", manager.get_current_balance())
    
    print("All open positions:", manager.get_all_open_positions())
    
    historical_trades = manager.get_historical_trades(days=7)
    print(f"Historical trades in last 7 days: {len(historical_trades)}")
    
    decision_log = manager.log_trading_decision("AAPL", "BUY", {"reason": "LLM confidence high"})
    print("Logged trading decision:", decision_log)
    
    manager.deposit_funds(5000, reason="Profit deposit")
    print("Balance after deposit:", manager.get_current_balance())
    manager.withdraw_funds(3000, reason="Partial withdrawal")
    print("Balance after withdrawal:", manager.get_current_balance())
    
    manager.set_trading_symbol("GOOG")
    print("Current trading symbol:", manager.current_symbol)
    
    summary = manager.get_account_summary(current_prices={"AAPL": 157.0, "GOOG": 2800.0})
    print("Account summary:", summary)
    
    manager.reset_metrics()
    print("Metrics after reset:", manager.get_metrics())
