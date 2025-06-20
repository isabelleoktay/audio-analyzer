import os
import json
from datetime import datetime, timedelta
import jwt


class JWTManager:
    def __init__(self, secret_key=None, redis_client=None, expiration_hours=24):
        self.secret_key = secret_key
        self.expiration_hours = expiration_hours
        self.redis_client = redis_client
        self.user_cache = {}  # fallback in-memory cache

    def generate_token(self, user_id: str) -> str:
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=self.expiration_hours),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def verify_token(self, token: str) -> str | None:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload.get('user_id')
        except jwt.ExpiredSignatureError:
            print("Token expired")
        except jwt.InvalidTokenError:
            print("Invalid token")
        return None

    def _get_cache_key(self, user_id: str) -> str:
        return f"user_cache:{user_id}"

    def get_user_cache(self, user_id: str) -> dict | None:
        cache_key = self._get_cache_key(user_id)
        # print(f"Fetching cache for user_id: {user_id}, cache_key: {cache_key}")
        if self.redis_client:
            data = self.redis_client.get(cache_key)
            return json.loads(data) if data else None
        return self.user_cache.get(user_id)

    def update_user_cache(self, user_id: str, cache_data: dict) -> None:
        cache_key = self._get_cache_key(user_id)
        if self.redis_client:
            print(f"Updating cache for user_id: {user_id}, cache_key: {cache_key}")
            self.redis_client.setex(
                cache_key,
                timedelta(hours=self.expiration_hours),
                json.dumps(cache_data, default=str)
            )
        else:
            self.user_cache[user_id] = cache_data


# --- Singleton pattern to store shared instance ---
_jwt_manager = None

def init_jwt_manager(secret_key=None, redis_client=None, expiration_hours=24):
    """
    Initializes the singleton JWT manager with optional overrides.
    Should be called during app startup (e.g., inside create_app()).
    """
    global _jwt_manager
    if _jwt_manager is None:
        if not secret_key:
            secret_key = os.getenv("JWT_SECRET_KEY", "dev-secret-key")
        _jwt_manager = JWTManager(secret_key=secret_key, redis_client=redis_client, expiration_hours=expiration_hours)

def get_jwt_manager() -> JWTManager:
    """
    Returns the singleton instance of JWTManager.
    Assumes `init_jwt_manager()` was already called.
    """
    if _jwt_manager is None:
        raise RuntimeError("JWTManager not initialized. Call init_jwt_manager() first.")
    return _jwt_manager
