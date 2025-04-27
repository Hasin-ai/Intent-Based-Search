from sqlalchemy import Column, Integer, String, Text, Numeric, JSON
from db import Base

class Product(Base):
    __tablename__ = "product_new"
    
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(255))
    cluster_id = Column(Integer)
    brand = Column(String(255))
    title = Column(String(500), nullable=False)
    description = Column(Text)
    price = Column(Numeric(10, 2))
    spectablecontent = Column(JSON)  