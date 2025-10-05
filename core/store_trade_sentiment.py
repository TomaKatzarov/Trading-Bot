# store_trade_sentiment.py
from datetime import datetime
from utils.db_setup import SessionLocal
from core.news_sentiment import SentimentAnalysis, NewsSentimentAnalyzer
from core.db_models import TradeJournal

def store_trade_sentiment(trade_id: str):
    """
    Re-load the TradeJournal row by trade_id in a new session,
    compute sentiment for that symbol, and store it in SentimentAnalysis.
    """
    analyzer = NewsSentimentAnalyzer()
    session = SessionLocal()
    try:
        # 1) Re-load the trade so it's attached to *this* session
        trade = session.query(TradeJournal).get(trade_id)
        if not trade:
            print(f"No TradeJournal row found with id={trade_id}")
            return

        symbol = trade.symbol  # now safe, because `trade` is attached
        score = analyzer.get_normalized_score(symbol)

        # 2) Create SentimentAnalysis row linked to that trade
        sent_row = SentimentAnalysis(
            trade_journal_id=trade_id,  # link to the same ID
            symbol=symbol,
            score=score,
            timestamp=datetime.utcnow(),
            source="my_pipeline",
            raw_data=""
        )
        session.add(sent_row)
        session.commit()
        print(f"Stored sentiment {score:.3f} for trade {trade_id} (symbol={symbol}).")
    except Exception as e:
        session.rollback()
        print(f"Error storing sentiment for trade {trade_id}: {e}")
    finally:
        session.close()
