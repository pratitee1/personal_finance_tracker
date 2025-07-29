from sqlalchemy import Column, Integer, String, Float, Date, Text, ForeignKey
from sqlalchemy.orm import relationship
from db.models import Base
class Receipt(Base):
    __tablename__ = "receipts"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    store_name = Column(String)
    store_address = Column(String)
    store_number = Column(String)
    date = Column(Date)
    total = Column(Float)
    taxes = Column(Float)
    payment_method = Column(String)
    confidence_score_ocr = Column(Integer)
    raw_text = Column(Text)
    user = relationship("User", back_populates="receipts")
    line_items = relationship("LineItem", back_populates="receipt", cascade="all, delete")