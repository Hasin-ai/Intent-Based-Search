import redis as redis_lib  # to avoid self-import conflict
import hashlib
import logging
import json
import time
import os

# Configure logging
logger = logging.getLogger(__name__)

# Redis client setup
redis_client = None

def setup_redis():
    """Initialize Redis client"""
    global redis_client
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        logger.info(f"Initializing Redis client: {redis_url}")
        redis_client = redis_lib.from_url(redis_url)  # Use redis_lib instead
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
        return 600  # for hot queries
    else:
        logger.info(f"Cold query - using 5 minute TTL")
        return 60   # for cold queries

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

def get_cached_embedding(query):
    """Get embedding from cache if available"""
    global redis_client
    if not redis_client:
        return None
        
    try:
        start_time = time.time()
        cache_key = get_cache_key(query)
        query_hash = cache_key.split(":")[-1]
        
        logger.info(f"Checking cache for query: '{query[:30]}...'")
        cached_data = redis_client.get(cache_key)
        
        if cached_data:
            # Cache hit
            logger.info(f"Cache HIT! Embedding retrieved in {time.time() - start_time:.2f}s")
            embedding = json.loads(cached_data)
            
            # Update stats and refresh TTL
            increment_query_frequency(query_hash)
            refresh_cache_ttl(query)
            
            return embedding
        
        logger.info(f"Cache MISS.")
        return None
    except Exception as e:
        logger.warning(f"Error retrieving from cache: {str(e)}")
        return None

def get_cache_statistics():
    """Get statistics about the Redis cache"""
    global redis_client
    try:
        if not redis_client:
            return {"status": "Redis not available"}
            
        # Get hot queries
        freq_key = get_frequency_key()
        hot_queries = redis_client.zrange(
            freq_key, 0, -1, desc=True, withscores=True
        )
        
        # Get memory info
        memory_info = redis_client.info("memory")
        
        # Format results
        hot_query_data = [
            {"query_hash": query.decode(), "score": score}
            for query, score in hot_queries if score >= 5
        ]
        
        return {
            "status": "ok",
            "hot_queries": hot_query_data,
            "hot_query_count": len(hot_query_data),
            "memory_used": memory_info.get("used_memory_human", "unknown"),
            "timestamp": time.time()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}