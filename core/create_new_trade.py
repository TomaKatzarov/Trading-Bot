# create_new_trade.py
import uuid
from datetime import datetime
from utils.db_setup import SessionLocal
from core.db_models import TradeJournal

def create_trade(symbol: str, action: str, qty: int):
    """
    Creates a new TradeJournal row for the given symbol and action,
    commits it, and returns the new trade's ID (not the detached object).
    """
    session = SessionLocal()
    try:
        new_id = str(uuid.uuid4())
        new_trade = TradeJournal(
            id=new_id,
            timestamp=datetime.utcnow(),
            symbol=symbol,
            action=action,
            quantity=qty,
            paper_price=150.0,
            status="open",
            pnl_unrealized=0.0,
            pnl_realized=0.0,
            model_metadata="{}",
            checksum=str(uuid.uuid4())
        )
        session.add(new_trade)
        session.commit()
        print(f"Created trade {new_id} in DB.")
        return new_id
    except Exception as e:
        session.rollback()
        print(f"Error creating trade: {e}")
        return None
    finally:
        session.close()