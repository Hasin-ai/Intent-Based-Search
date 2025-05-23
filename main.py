import logging
import traceback
from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from db import engine, Base, get_db
from models import Product
from typing import List
from schemas import ProductModel, ProductResponse, ProductUpdate
import vector_search
import time
from redis_cache import get_cache_statistics 
from create_tables import create_tables_if_not_exist


logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import redis_cache
redis_cache.setup_redis()  # Initialize Redis immediately


@app.on_event("startup")
async def startup_event():
    # Create tables if they don't exist
    create_tables_if_not_exist()
    
    # Set up vector search collection
    vector_search.setup_collection()



@app.post("/products/", response_model=ProductModel)
async def create_product(
    product: ProductModel,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    try:
        db_product = Product(
            category=product.category,
            cluster_id=product.cluster_id,
            brand=product.brand,
            title=product.title,
            description=product.description,
            price=product.price,
            spectablecontent=product.specTableContent  
        )
        db.add(db_product)
        db.commit()
        db.refresh(db_product)
        
        # Index the product in the vector database in the background
        background_tasks.add_task(vector_search.index_product, db_product)
        
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

@app.get("/products/search/", response_model=List[ProductResponse])
async def search_products(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum number of results to return"),
    offset: int = Query(0, description="Offset for pagination"),
    db: Session = Depends(get_db)
):
    try:
        
        if not query.strip():
            return []
        
        # PostgreSQL tsvector from all relevant text fields with different weights
        # Weight A (4.0) for title and brand, Weight B (0.4) for description and category
        search_vector = func.to_tsvector('english', 
            func.concat_ws(' ', 
                Product.title,
                Product.brand,
                Product.category,
                Product.description
            )
        )
        
      
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

@app.put("/products/{product_id}", response_model=ProductResponse)
async def update_product(
    product_id: int,
    product: ProductUpdate,
    background_tasks: BackgroundTasks,
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
        
        # Reindex the updated product
        background_tasks.add_task(vector_search.index_product, db_product)
        
        return db_product
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/products/{product_id}")
async def delete_product(product_id: int, db: Session = Depends(get_db)):
    try:
        db_product = db.query(Product).filter(Product.id == product_id).first()
        if db_product is None:
            raise HTTPException(status_code=404, detail="Product not found")
        
        db.delete(db_product)
        db.commit()
        return {"message": "Product deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products/integrated-search/", response_model=List[ProductResponse])
async def integrated_search(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum number of results to return"),
    offset: int = Query(0, description="Offset for pagination"),
    db: Session = Depends(get_db)
):
    """Integrated search using both PostgreSQL and vector search"""
    try:
        
        if not query.strip():
            return []
        
        # Get similar product IDs from vector search
        similar_products = vector_search.get_similar_product_ids(query, limit, offset)
        
        if not similar_products:
            return []
        
        # Extract IDs
        product_ids = [item["id"] for item in similar_products]
        
        # Fetch complete product data from PostgreSQL
        products_map = {}
        products = db.query(Product).filter(Product.id.in_(product_ids)).all()
        for product in products:
            products_map[product.id] = product
        
        # Combine vector search scores with product data
        results = []
        for item in similar_products:
            product_id = item["id"]
            if product_id in products_map:
                product = products_map[product_id]
                
                # Skip products with missing or invalid title
                if not product.title or not isinstance(product.title, str):
                    continue
                product_dict = {
                    "id": product.id,
                    "category": product.category,
                    "cluster_id": product.cluster_id,
                    "brand": product.brand,
                    "title": product.title,
                    "description": product.description,
                    "price": product.price,
                    "specTableContent": product.spectablecontent,
                    "score": item["score"]
                }
                results.append(product_dict)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/products/reindex")
async def reindex_products(db: Session = Depends(get_db)):
    """Reindex all products from PostgreSQL into Qdrant"""
    logger.info("Reindex request received")
    try:
        count = vector_search.index_all_products(db)
        logger.info(f"Reindexed {count} products")
        return {"message": f"Successfully indexed {count} products", "status": "success"}
    except Exception as e:
        logger.error(f"Reindex error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Reindexing error: {str(e)}")

@app.get("/cache/stats")
async def cache_statistics():
    """Get statistics about the Redis cache"""
    return get_cache_statistics()

@app.get("/cache/test")
async def test_redis_connection():
    """Test Redis connection and caching"""
    try:
        from redis_cache import redis_client

        if not redis_client:
            return {"status": "error", "message": "Could not connect to Redis"}
            
        test_key = "test:connection"
        test_value = f"Connection test at {time.time()}"
        redis_client.set(test_key, test_value)
        read_back = redis_client.get(test_key)
        
        info = redis_client.info()
        
        return {
            "status": "success",
            "connection": "established",
            "write_test": "success",
            "read_test": read_back if read_back else None,
            "redis_version": info.get("redis_version", "unknown"),
            "clients_connected": info.get("connected_clients", "unknown")
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)