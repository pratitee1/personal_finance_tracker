from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from db.models import Base
class LineItem(Base):
    __tablename__ = "line_items"
    id = Column(Integer, primary_key=True)
    receipt_id = Column(Integer, ForeignKey("receipts.id"))
    name = Column(String)
    quantity = Column(Float)
    price_per_unit = Column(Float)
    total_price = Column(Float)
    confidence_score = Column(Integer)
    category = Column(String)
    receipt = relationship("Receipt", back_populates="line_items")