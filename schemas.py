from pydantic import BaseModel, constr
from typing import Optional

class ProductBase(BaseModel):
    """Base class for Product schemas"""
    class Config:
        from_attributes = True

class ProductModel(ProductBase):
    """Schema for creating and displaying products"""
    name: constr(min_length=1, max_length=100)
    description: constr(min_length=1)

class ProductResponse(ProductModel):
    """Schema for returning products with ID"""
    id: int
    score: Optional[float] = None  # Optional score for search results

class ProductUpdate(ProductBase):
    """Schema for updating products"""
    name: Optional[constr(min_length=1, max_length=100)] = None
    description: Optional[constr(min_length=1)] = None