from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from db.models import Base
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    receipts = relationship("Receipt", back_populates="user")