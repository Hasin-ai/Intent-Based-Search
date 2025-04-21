import logging
import traceback
from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import or_, func, desc, text
from sqlalchemy.sql.expression import literal_column
from db import engine, Base, get_db
from models import Product
from typing import List
from schemas import ProductModel, ProductResponse, ProductUpdate
import vector_search

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Initialize vector search on startup
@app.on_event("startup")
async def startup_event():
    vector_search.setup_collection()

# Database setup

@app.post("/products/", response_model=ProductModel)
async def create_product(
    product: ProductModel,
    background_tasks: BackgroundTasks,
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

@app.get("/products/vector-search/", response_model=List[ProductResponse])
async def search_products_vector(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum number of results to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    logger.info(f"Vector search request received: query='{query}', limit={limit}, offset={offset}")
    try:
        # First check if we need to set up the collection
        collection_setup_ok = vector_search.setup_collection()
        if not collection_setup_ok:
            logger.error("Collection setup failed")
            raise HTTPException(status_code=500, detail="Failed to set up vector search collection")
        
        # Try to get similar products
        logger.info("Retrieving similar product IDs")
        similar_product_ids = vector_search.get_similar_product_ids(query, limit, offset)
        
        # Check if we got any results
        if not similar_product_ids:
            logger.info("No similar products found")
            return []
        
        # Extract IDs
        product_ids = [item["id"] for item in similar_product_ids]
        logger.info(f"Found similar product IDs: {product_ids}")
        
        # Get product data from database
        db = next(get_db())
        products = db.query(Product).filter(Product.id.in_(product_ids)).all()
        logger.info(f"Retrieved {len(products)} products from database")
        
        # Map scores to products
        id_to_score = {item["id"]: item["score"] for item in similar_product_ids}
        result = []
        
        for product in products:
            product_dict = {
                "id": product.id,
                "name": product.name,
                "description": product.description,
                "score": id_to_score.get(product.id, 0)
            }
            result.append(product_dict)
        
        # Sort by score (highest first)
        result.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"Returning {len(result)} products")
        
        return result
    except Exception as e:
        logger.error(f"Error in vector search: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Vector search error: {str(e)}")

@app.get("/products/integrated-search/", response_model=List[ProductResponse])
async def integrated_search(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum number of results to return"),
    offset: int = Query(0, description="Offset for pagination"),
    db: Session = Depends(get_db)
):
    """
    Integrated search that uses:
    1. Vector search (Qdrant) to find similar products
    2. PostgreSQL to fetch the complete product data
    
    Architecture flow:
    User → FastAPI → Embedding Generation → Vector DB → PostgreSQL → Response
    """
    try:
        # Return empty list for empty queries
        if not query.strip():
            return []
        
        # 1. Get similar product IDs from vector search
        similar_products = vector_search.get_similar_product_ids(query, limit, offset)
        
        if not similar_products:
            return []
        
        # 2. Extract IDs
        product_ids = [item["id"] for item in similar_products]
        
        # 3. Fetch complete product data from PostgreSQL
        products_map = {}
        products = db.query(Product).filter(Product.id.in_(product_ids)).all()
        for product in products:
            products_map[product.id] = product
        
        # 4. Combine vector search scores with product data
        # Maintain the order of results from vector search
        results = []
        for item in similar_products:
            product_id = item["id"]
            if product_id in products_map:
                # Attach score to product model
                product = products_map[product_id]
                # Add as a dictionary so we can include the score
                product_dict = {
                    "id": product.id,
                    "name": product.name,
                    "description": product.description,
                    "score": item["score"]  # Include relevance score
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)