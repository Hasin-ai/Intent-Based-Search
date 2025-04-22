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


# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Load environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY is not set in environment variables")
    raise ValueError("GEMINI_API_KEY is required")


# Gemini API setup
genai.configure(api_key=GEMINI_API_KEY)


# Initialize Qdrant client
try:
    logger.info("Initializing Qdrant client")
    qdrant = QdrantClient(":memory:")
    logger.info("Qdrant client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant client: {str(e)}")
    raise


# Variable to store the embedding dimension once we discover it
VECTOR_DIM = None

def generate_embedding(text):
    """Generate embeddings using Gemini API"""
    global VECTOR_DIM
    
    try:
        logger.info(f"Generating embedding for: '{text[:30]}...'")
        
        # Generate embedding using Gemini embeddings API
        embedding_result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="SEMANTIC_SIMILARITY"
        )
        
        if embedding_result and hasattr(embedding_result, 'embedding'):
            embedding = embedding_result.embedding
        elif embedding_result and isinstance(embedding_result, dict) and 'embedding' in embedding_result:
            embedding = embedding_result['embedding']
        else:
            logger.error(f"Unexpected embedding result format: {type(embedding_result)}")
            raise ValueError("Failed to extract embedding from API response")
        
        # Update the global VECTOR_DIM if not set yet
        if VECTOR_DIM is None:
            VECTOR_DIM = len(embedding)
            logger.info(f"Discovered embedding dimension: {VECTOR_DIM}")
        
        # Ensure the dimension matches our expected dimension
        if len(embedding) != VECTOR_DIM:
            logger.warning(f"Embedding dimension mismatch: got {len(embedding)}, expected {VECTOR_DIM}")
            
        logger.info(f"Successfully generated embedding with {len(embedding)} dimensions")
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
            
        # Generate embedding for query
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