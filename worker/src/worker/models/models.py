from worker.models.base import Base
from sqlalchemy import Column, Integer, String, DateTime, Text
from datetime import datetime

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String, nullable=False)
    response = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
