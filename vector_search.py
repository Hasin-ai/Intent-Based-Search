import os
import logging
import traceback
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from models import Product
from sqlalchemy.orm import Session
from dotenv import load_dotenv


# Redis
from redis_cache import (
    get_cached_embedding, cache_embedding,
    get_cache_key, get_redis_client,
    get_frequency_key
)

# logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()   
MODEL_PATH = os.getenv("MODEL_PATH")
# print(f"MODEL_PATH: {MODEL_PATH}")
if not MODEL_PATH:
    raise ValueError("MODEL_PATH is not set in environment variables")

# Initialize Qdrant client
try:
    logger.info("Initializing Qdrant client")
    qdrant = QdrantClient(":memory:")
    logger.info("Qdrant client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant client: {str(e)}")
    raise


VECTOR_DIM = 384  

def generate_embedding_from_transformers(text):
    """Generate embeddings using local transformer model"""
    try:
        
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        
        model_path = MODEL_PATH
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        
        encoded_input = tokenizer([text], padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            model_output = model(**encoded_input)
        
        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        # Convert embeddings to list for storage
        embedding = mean_pooled[0].numpy().tolist()
        
        # Ensure embedding is exactly 384 dimensions
        global VECTOR_DIM
        if len(embedding) != VECTOR_DIM:
            logger.warning(f"Embedding dimension mismatch: got {len(embedding)}, forcing to {VECTOR_DIM}")
            if len(embedding) > VECTOR_DIM:
                # Truncate if longer than needed
                embedding = embedding[:VECTOR_DIM]
            else:
                # Pad with zeros if shorter than needed 
                embedding = embedding + [0.0] * (VECTOR_DIM - len(embedding))
        
        logger.info(f"Successfully generated local embedding with {len(embedding)} dimensions")
        return embedding
        
    except Exception as e:
        logger.error(f"Error generating local embedding: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def generate_embedding(text):
    """Generate embeddings with Redis caching using local transformer model"""
    global VECTOR_DIM
    VECTOR_DIM = 384
    
    try:
        
        cached_embedding = get_cached_embedding(text)
        if cached_embedding:
           
            if len(cached_embedding) != VECTOR_DIM:
                logger.warning(f"Cached embedding has incorrect dimension: {len(cached_embedding)}, expected {VECTOR_DIM}")
                
              
                if len(cached_embedding) > VECTOR_DIM:
                    cached_embedding = cached_embedding[:VECTOR_DIM]
                else:
                    cached_embedding = cached_embedding + [0.0] * (VECTOR_DIM - len(cached_embedding))
            return cached_embedding

        logger.info(f"Generating embedding for: '{text[:30]}...'")
        
        embedding = generate_embedding_from_transformers(text)

        
        if VECTOR_DIM is None:
            VECTOR_DIM = len(embedding)
            logger.info(f"Discovered embedding dimension: {VECTOR_DIM}")

        
        if len(embedding) != VECTOR_DIM:
            logger.warning(f"Embedding dimension mismatch: got {len(embedding)}, expected {VECTOR_DIM}")
        
        
        if get_redis_client():
            redis_client = get_redis_client()
            logger.info("Caching embedding in Redis")
            query_hash = get_cache_key(text).split(":")[-1]
            logger.info(f"Caching embedding for query hash: {query_hash}")
            
            
            current_freq = redis_client.zscore(get_frequency_key(), query_hash) or 0
            new_freq = current_freq + 1
            
            
            redis_client.zadd(get_frequency_key(), {query_hash: new_freq})
            redis_client.expire(get_frequency_key(), 3600)  # Set expiry
            
            
            ttl = 3600 if new_freq >= 5 else 300
            
            
            cache_embedding(text, embedding, ttl)  # Pass TTL directly
            logger.info(f"Cached embedding with TTL {ttl}s")
        else:
            logger.warning("Redis client not available, skipping cache")

        return embedding

    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        logger.error(traceback.format_exc())

        # Fall back to mock embedding
        logger.info("Falling back to mock embedding")
        return generate_mock_embedding(text)

def generate_mock_embedding(text):
    """Fallback function to generate mock embeddings if API fails"""
    try:
        global VECTOR_DIM
        if VECTOR_DIM is None:
            VECTOR_DIM = 384  
            logger.info(f"Using default embedding dimension: {VECTOR_DIM}")
        
        text_hash = hash(text) % 10000
        np.random.seed(text_hash)
        mock_embedding = np.random.random(VECTOR_DIM).tolist()
        
        logger.info(f"Generated mock embedding with {len(mock_embedding)} dimensions")
        return mock_embedding
    except Exception as e:
        logger.error(f"Error generating mock embedding: {str(e)}")
        raise

def setup_collection():
    """Set up Qdrant collection with the correct dimensions"""
    global VECTOR_DIM
    
    if VECTOR_DIM is None:
        logger.info("VECTOR_DIM is None")
        VECTOR_DIM = 384
    
    try:
        logger.info("Setting up Qdrant collection")
        
        if qdrant.collection_exists("products"):
            logger.info("Collection 'products' exists, checking dimensions")
            try:
                collection_info = qdrant.get_collection("products")
                existing_dim = collection_info.config.params.vectors.size
                
                if existing_dim == VECTOR_DIM:
                    logger.info(f"Collection dimensions match ({VECTOR_DIM}), keeping existing collection")
                    return True
                else:
                    logger.warning(f"Dimension mismatch: collection={existing_dim}, required={VECTOR_DIM}")
                    logger.info("Recreating collection with correct dimensions")
                    qdrant.delete_collection("products")
            except Exception as e:
                logger.error(f"Error checking collection: {str(e)}")
                qdrant.delete_collection("products")
        
        # Create new collection with correct dimensions
        logger.info(f"Creating collection with {VECTOR_DIM} dimensions")
        qdrant.create_collection(
            collection_name="products",
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info("Collection 'products' created successfully")
        return True
    except Exception as e:
        logger.error(f"Error setting up collection: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def index_product(product):
    """Index a single product in Qdrant"""
    try:
        logger.info(f"Indexing product ID: {product.id}")
        
        # embedding using all relevant fields
        content = f"{product.category or ''} {product.brand or ''} {product.title or ''} {product.description or ''} {str(product.price) if product.price else ''}"
        embedding = generate_embedding(content)
        
        # Create point for Qdrant
        point = PointStruct(
            id=product.id,
            vector=embedding,
            payload={"id": product.id}
        )
        
        # Upsert point to Qdrant
        qdrant.upsert(
            collection_name="products",
            points=[point],
            wait=True
        )
        logger.info(f"Successfully indexed product ID: {product.id}")
        return point
    except Exception as e:
        logger.error(f"Error indexing product ID {product.id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def index_all_products(db: Session):
    """Index all products from database into Qdrant"""
    try:
        logger.info("Starting to index all products")
        setup_collection()
        products = db.query(Product).all()
        logger.info(f"Found {len(products)} products to index")
        
        indexed_count = 0
        for product in products:
            try:
                index_product(product)
                indexed_count += 1
            except Exception as e:
                logger.error(f"Failed to index product {product.id}: {str(e)}")
                # Continue with other products even if one fails
                
        logger.info(f"Successfully indexed {indexed_count} out of {len(products)} products")
        return indexed_count
    except Exception as e:
        logger.error(f"Error in index_all_products: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_similar_product_ids(query, limit=10, offset=0):
    """Get similar product IDs from vector search"""
    try:
        logger.info(f"Getting similar products for query: '{query}'")
        if not query.strip():
            logger.info("Empty query, returning empty result")
            return []
            
        # Generate embedding for query (now with Redis caching)
        logger.info("Generating embedding for query")
        query_vector = generate_embedding(query)
        
        # Search in Qdrant
        logger.info(f"Searching Qdrant with limit={limit}, offset={offset}")
        hits = qdrant.search(
            collection_name="products",
            query_vector=query_vector,
            limit=limit,
            offset=offset
        )
        
        # Process results
        result_ids = [{"id": hit.payload["id"], "score": hit.score} for hit in hits]
        logger.info(f"Vector search returned {len(result_ids)} results")
        return result_ids
    except Exception as e:
        logger.error(f"Error in vector search for query '{query}': {str(e)}")
        logger.error(traceback.format_exc())
        raise