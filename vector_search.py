import os
import logging
import traceback
import google.generativeai as genai
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from models import Product
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from google.genai import types
from google import genai
import redis
import hashlib
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Load environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY is not set in environment variables")
    raise ValueError("GEMINI_API_KEY is required")

# Redis client setup
redis_client = None
def setup_redis():
    """Initialize Redis client"""
    global redis_client
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        logger.info(f"Initializing Redis client: {redis_url}")
        redis_client = redis.from_url(redis_url)
        redis_client.ping()  # Test connection
        logger.info("Redis connection successful")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {str(e)}")
        redis_client = None
        return False

# Cache management functions
def get_cache_key(query):
    """Generate a cache key for embeddings based on query hash"""
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return f"search:embedding:{query_hash}"

def get_frequency_key():
    """Key for the frequency sorted set"""
    return "search:freq"

def is_hot_query(query_hash):
    """Check if a query is 'hot' (score >= 5)"""
    global redis_client
    if not redis_client:
        return False
        
    try:
        score = redis_client.zscore(get_frequency_key(), query_hash)
        return score is not None and score >= 5
    except Exception as e:
        logger.warning(f"Error checking hot query: {str(e)}")
        return False

def increment_query_frequency(query_hash):
    """Increment frequency counter for query"""
    global redis_client
    if not redis_client:
        return
        
    try:
        # Increment score in sorted set
        redis_client.zincrby(get_frequency_key(), 1, query_hash)
        # Ensure frequency data expires after 1 hour
        redis_client.expire(get_frequency_key(), 3600)
    except Exception as e:
        logger.warning(f"Failed to increment query frequency: {str(e)}")

def determine_ttl(query_hash):
    """Determine TTL based on query frequency"""
    if is_hot_query(query_hash):
        logger.info(f"Hot query detected - using 1 hour TTL")
        return 3600  # 1 hour for hot queries
    else:
        logger.info(f"Cold query - using 5 minute TTL")
        return 300   # 5 minutes for cold queries

def cache_embedding(query, embedding):
    """Cache embedding with adaptive TTL"""
    global redis_client
    if not redis_client:
        return
        
    try:
        cache_key = get_cache_key(query)
        query_hash = cache_key.split(":")[-1]
        ttl = determine_ttl(query_hash)
        
        redis_client.setex(
            cache_key, 
            ttl, 
            json.dumps(embedding)
        )
        logger.info(f"Cached embedding with TTL: {ttl}s")
    except Exception as e:
        logger.warning(f"Failed to cache embedding: {str(e)}")

def refresh_cache_ttl(query):
    """Refresh TTL for existing cache entry"""
    global redis_client
    if not redis_client:
        return
        
    try:
        cache_key = get_cache_key(query)
        query_hash = cache_key.split(":")[-1]
        ttl = determine_ttl(query_hash)
        redis_client.expire(cache_key, ttl)
        logger.info(f"Refreshed cache TTL to {ttl}s")
    except Exception as e:
        logger.warning(f"Failed to refresh cache TTL: {str(e)}")

# Gemini API setup
try:
    logger.info("Initializing Gemini API client")
    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("Gemini API client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini API client: {str(e)}")
    raise


# Initialize Qdrant client
try:
    logger.info("Initializing Qdrant client")
    qdrant = QdrantClient(":memory:")
    logger.info("Qdrant client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant client: {str(e)}")
    raise


# Variable to store the embedding dimension once we discover it
VECTOR_DIM = 768  # Default dimension for many models

def generate_embedding(text):
    """Generate embeddings using Gemini API with Redis caching"""
    global VECTOR_DIM, redis_client

    try:
        # Check cache if Redis is available
        if redis_client:
            start_time = time.time()
            cache_key = get_cache_key(text)
            query_hash = cache_key.split(":")[-1]
            
            logger.info(f"Checking cache for query: '{text[:30]}...'")
            cached_data = redis_client.get(cache_key)
            
            if cached_data:
                # Cache hit
                logger.info(f"Cache HIT! Embedding retrieved in {time.time() - start_time:.2f}s")
                embedding = json.loads(cached_data)
                
                # Update stats and refresh TTL
                increment_query_frequency(query_hash)
                refresh_cache_ttl(text)
                
                return embedding
            
            logger.info(f"Cache MISS. Generating new embedding.")
        else:
            logger.info("Redis not available, skipping cache check")

        # Generate new embedding
        logger.info(f"Generating embedding for: '{text[:30]}...'")

        result = client.models.embed_content(
            model="text-embedding-004",
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=VECTOR_DIM, 
                                     task_type="SEMANTIC_SIMILARITY")
        )

        # Extract embedding values
        embedding = result.embeddings[0].values

        # Update the global VECTOR_DIM if not set yet
        if VECTOR_DIM is None:
            VECTOR_DIM = len(embedding)
            logger.info(f"Discovered embedding dimension: {VECTOR_DIM}")

        # Ensure the dimension matches our expected dimension
        if len(embedding) != VECTOR_DIM:
            logger.warning(f"Embedding dimension mismatch: got {len(embedding)}, expected {VECTOR_DIM}")

        logger.info(f"Successfully generated embedding with {len(embedding)} dimensions")
        
        # Cache the newly generated embedding
        if redis_client:
            cache_key = get_cache_key(text)
            query_hash = cache_key.split(":")[-1]
            cache_embedding(text, embedding)
            increment_query_frequency(query_hash)

        return embedding

    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        logger.error(traceback.format_exc())

        # Fall back to mock embedding if API fails
        logger.info("Falling back to mock embedding")
        return generate_mock_embedding(text)

def generate_mock_embedding(text):
    """Fallback function to generate mock embeddings if API fails"""
    try:
        # If VECTOR_DIM is not set yet, use a default size
        global VECTOR_DIM
        if VECTOR_DIM is None:
            VECTOR_DIM = 768  # Default for many embedding models
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
    
    try:
        logger.info("Setting up Qdrant collection")
        
        # Generate a test embedding to determine dimensions if not known yet
        if VECTOR_DIM is None:
            logger.info("Generating test embedding to determine dimensions")
            test_embedding = generate_embedding("test query")
            VECTOR_DIM = len(test_embedding)
            logger.info(f"Determined vector dimension: {VECTOR_DIM}")
        
        # If collection exists, check if dimensions match
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
        
        # Generate embedding
        content = f"{product.name} {product.description}"
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