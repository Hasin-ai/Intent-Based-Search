from sqlalchemy import Column, Integer, String, Text
from    db import Base

class Product(Base):
    __tablename__ = "product"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)