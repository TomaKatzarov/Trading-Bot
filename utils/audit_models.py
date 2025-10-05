# utils/audit_models.py
from sqlalchemy import Column, Integer, Float, String, DateTime, Text
from utils.db_setup import Base

class AuditLog(Base):
    __tablename__ = "audit_log"
    __table_args__ = {"extend_existing": True}  # ensures no conflict

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime)
    action_type = Column(String)
    details = Column(Text)
    account_balance = Column(Float)
