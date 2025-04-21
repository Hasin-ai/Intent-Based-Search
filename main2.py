from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import or_, func, desc, text
from sqlalchemy.sql.expression import literal_column
from db import engine, Base, get_db
from models import Product
from typing import List
from schemas import ProductModel, ProductResponse, ProductUpdate

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup

@app.post("/products/", response_model=ProductModel)
async def create_product(
    product: ProductModel,
    db: Session = Depends(get_db)
):
    try:
        db_product = Product(
            name=product.name,
            description=product.description
        )
        db.add(db_product)
        db.commit()
        db.refresh(db_product)
        return db_product
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products/")
async def get_products(db: Session = Depends(get_db)):
    try:
        return db.query(Product).all()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products/{product_id}", response_model=ProductResponse)
async def get_product(product_id: int, db: Session = Depends(get_db)):
    try:
        product = db.query(Product).filter(Product.id == product_id).first()
        if product is None:
            raise HTTPException(status_code=404, detail="Product not found")
        return product
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/products/{product_id}", response_model=ProductResponse)
async def update_product(
    product_id: int,
    product: ProductUpdate,
    db: Session = Depends(get_db)
):
    try:
        db_product = db.query(Product).filter(Product.id == product_id).first()
        if db_product is None:
            raise HTTPException(status_code=404, detail="Product not found")
        
        update_data = product.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_product, key, value)
            
        db.commit()
        db.refresh(db_product)
        return db_product
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products/search/", response_model=List[ProductResponse])
async def search_products(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum number of results to return"),
    offset: int = Query(0, description="Offset for pagination"),
    db: Session = Depends(get_db)
):
    try:
        # Return empty list for empty queries
        if not query.strip():
            return []
        
        # Create a PostgreSQL tsvector from name and description with different weights
        # Weight A (4.0) for name and weight B (0.4) for description
        search_vector = func.to_tsvector('english', 
            func.concat_ws(' ', 
                Product.name, 
                Product.description
            )
        )
        
        # Create a tsquery from the search terms
        search_query = func.plainto_tsquery('english', query)
        
        # Perform the search with ranking
        results = db.query(
            Product,
            func.ts_rank(search_vector, search_query).label('rank')
        ).filter(
            search_vector.op('@@')(search_query)
        ).order_by(
            desc('rank')
        ).offset(offset).limit(limit).all()
        
        # Extract just the products from the results
        products = [item[0] for item in results]
        
        return products
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)