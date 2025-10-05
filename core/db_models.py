# core/models.py
from sqlalchemy import Column, DateTime, Float, String, Text, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from utils.db_setup import Base

class TradeJournal(Base):
    __tablename__ = "trade_journal"
    id = Column(String, primary_key=True, index=True)
    timestamp = Column(DateTime)
    symbol = Column(String, index=True)
    action = Column(String)
    quantity = Column(Integer)
    paper_price = Column(Float)
    real_price = Column(Float, nullable=True)
    status = Column(String)
    pnl_unrealized = Column(Float)
    pnl_realized = Column(Float)
    model_metadata = Column(Text)
    checksum = Column(String)
    open_price = Column(Float, nullable=True)
    high_price = Column(Float, nullable=True)
    low_price = Column(Float, nullable=True)
    close_price = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    wap = Column(Float, nullable=True)
    returns = Column(Float, nullable=True)
    hl_diff = Column(Float, nullable=True)
    ohlc_avg = Column(Float, nullable=True)
    rsi = Column(Float, nullable=True)
    volume_pct = Column(Float, nullable=True)
    returns_1h = Column(Float, nullable=True)

    sentiment_analysis = relationship("SentimentAnalysis", back_populates="trade_journal", uselist=False)

class SentimentAnalysis(Base):
    __tablename__ = "sentiment_analysis"
    trade_journal_id = Column(String, ForeignKey("trade_journal.id"), unique=True)
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    score = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    source = Column(String(50))
    raw_data = Column(String(500))

    trade_journal = relationship("TradeJournal", back_populates="sentiment_analysis", uselist=False)
    __table_args__ = (
        UniqueConstraint("trade_journal_id", name="uq_sentiment_one_per_trade"),
    )