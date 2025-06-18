import jwt
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class JWTManager:
    def __init__(self, secret_key: str, expiration_hours: int = 24):
        self.secret_key = secret_key
        self.expiration_hours = expiration_hours
        # In-memory cache that clears on restart
        self.user_cache: Dict[str, Dict] = {}
    
    def generate_token(self) -> str:
        """Generate a new JWT token with a unique user ID"""
        user_id = str(uuid.uuid4())
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=self.expiration_hours),
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        # Initialize cache for this user
        self.user_cache[user_id] = {
            'current_file_hash': None,
            'pitch': {
                'audio': None,
                'sr': None,
                'pitch': None,
                'smoothed_pitch': None,
                'highlighted_section': None,
                'x_axis': None,
                'audio_url': None,
                'hop_sec_duration': None
            },
            'dynamics': {
                'audio': None,
                'sr': None,
                'audio_url': None,
                'dynamics': None,
                'smoothed_dynamics': None,
                'highlighted_section': None,
                'x_axis': None,
            },
            'tempo': {
                'audio': None,
                'sr': None,
                'audio_url': None,
                'tempo': None,
                'beats': None
            },
            'vibrato': {
                'audio': None,
                'sr': None,
                'audio_url': None,
                'vibrato_rate': None,
                'vibrato_extent': None,
                'highlighted_section': None,
            },
            'phonation': {
                'audio': None,
                'sr': None,
                'audio_url': None,
                'jitter': None,
                'shimmer': None,
                'hnr': None
            }
        }
        
        return token
    
    def verify_token(self, token: str) -> Optional[str]:
        """Verify JWT token and return user_id if valid"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            user_id = payload['user_id']
            
            # Check if user exists in cache
            if user_id not in self.user_cache:
                return None
                
            return user_id
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def get_user_cache(self, user_id: str) -> Optional[Dict]:
        """Get user's cache data"""
        return self.user_cache.get(user_id)
    
    def clear_expired_users(self):
        """Clean up expired tokens (optional background task)"""
        # This would require storing expiration times separately
        # For now, we rely on restart to clear everything
        pass

# Global JWT manager instance
jwt_manager = None

def init_jwt_manager(secret_key: str):
    global jwt_manager
    jwt_manager = JWTManager(secret_key)
    return jwt_manager

def get_jwt_manager():
    global jwt_manager
    return jwt_manager