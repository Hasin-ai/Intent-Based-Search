from pydantic import BaseModel, constr, Field
from typing import Optional, Dict, Any

class ProductBase(BaseModel):
    """Base class for Product schemas"""
    class Config:
        from_attributes = True
        populate_by_name = True  # Allow both casing versions

class ProductModel(ProductBase):
    """Schema for creating and displaying products"""
    category: Optional[str] = None
    cluster_id: Optional[int] = None
    brand: Optional[str] = None
    title: constr(min_length=1, max_length=500)
    description: Optional[str] = None
    price: Optional[float] = None
    specTableContent: Optional[Dict[str, Any]] = Field(default=None, alias="spectablecontent")

class ProductResponse(ProductModel):
    """Schema for returning products with ID"""
    id: int
    score: Optional[float] = None  # Optional score for search results

class ProductUpdate(ProductBase):
    """Schema for updating products"""
    category: Optional[str] = None
    cluster_id: Optional[int] = None
    brand: Optional[str] = None
    title: Optional[constr(min_length=1, max_length=500)] = None
    description: Optional[str] = None
    price: Optional[float] = None
    specTableContent: Optional[Dict[str, Any]] = Field(default=None, alias="spectablecontent")