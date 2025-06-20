import os
import redis
import urllib.parse

def get_redis_client():
    """
    Initialize and return a Redis client based on environment configuration.
    Returns None if Redis connection fails.
    """
    redis_client = None
    
    try:
        if os.getenv("FLASK_ENV") == "production":
            redis_host = os.environ.get('REDIS_HOST', 'localhost')
            redis_port = int(os.environ.get('REDIS_PORT', 6379))
            redis_password = os.environ.get('REDIS_PASSWORD')
            
            if redis_password:
                print(f"⚠️  Connecting with redis password: {redis_password}")
                encoded_password = urllib.parse.quote(redis_password)
                redis_client = redis.from_url(f"redis://:{encoded_password}@{redis_host}:{redis_port}/0")
            else:
                redis_client = redis.from_url(f"redis://{redis_host}:{redis_port}/0")
        else:
            redis_client = redis.from_url('redis://localhost:6379')
        
        # Test Redis connection
        redis_client.ping()
        print("✅ Redis connected successfully")
        
    except Exception as e:
        print(f"⚠️  Redis connection failed: {e}")
        print("📝 Falling back to in-memory caching")
        redis_client = None
    
    return redis_client