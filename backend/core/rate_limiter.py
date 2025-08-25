"""
Rate Limiter for Free Tier Users
=================================
Implements a sliding window rate limiter with Redis backend.
Free tier: 20 requests per 5 hours (18000 seconds).
"""

import time
import json
import logging
from typing import Optional, Dict, Tuple
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

# Try Redis for production, fallback to file-based for development
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.info("Redis not available - using file-based rate limiting")


class RateLimiter:
    """Rate limiter for API requests with configurable limits."""
    
    # Default limits
    FREE_TIER_LIMIT = 20  # requests
    FREE_TIER_WINDOW = 5 * 60 * 60  # 5 hours in seconds
    
    PREMIUM_TIER_LIMIT = 1000  # requests
    PREMIUM_TIER_WINDOW = 60 * 60  # 1 hour
    
    def __init__(self, redis_url: Optional[str] = None, data_dir: Optional[Path] = None):
        """
        Initialize rate limiter.
        
        Args:
            redis_url: Redis connection URL (for production)
            data_dir: Directory for file-based storage (for development)
        """
        self.redis_client = None
        self.data_dir = data_dir or Path("./data/rate_limits")
        
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("✅ Rate limiter using Redis backend")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        
        if not self.redis_client:
            # Use file-based storage
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Rate limiter using file backend: {self.data_dir}")
    
    def _get_client_id(self, ip: str, wallet: Optional[str] = None) -> str:
        """Get unique client identifier."""
        if wallet:
            # Authenticated users tracked by wallet
            return f"wallet:{wallet}"
        else:
            # Free users tracked by IP
            return f"ip:{ip}"
    
    def _get_tier(self, wallet: Optional[str] = None, api_key: Optional[str] = None) -> str:
        """Determine user tier."""
        if api_key or wallet:
            return "premium"
        return "free"
    
    def check_rate_limit(
        self, 
        ip: str, 
        wallet: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: str = "/chat"
    ) -> Tuple[bool, Dict]:
        """
        Check if request is within rate limit.
        
        Returns:
            (allowed, info) where:
            - allowed: True if request is allowed
            - info: Dict with limit details
        """
        client_id = self._get_client_id(ip, wallet)
        tier = self._get_tier(wallet, api_key)
        
        # Get tier limits
        if tier == "premium":
            limit = self.PREMIUM_TIER_LIMIT
            window = self.PREMIUM_TIER_WINDOW
        else:
            limit = self.FREE_TIER_LIMIT
            window = self.FREE_TIER_WINDOW
        
        current_time = time.time()
        window_start = current_time - window
        
        # Get request history
        requests = self._get_requests(client_id, endpoint)
        
        # Filter requests within window
        recent_requests = [t for t in requests if t > window_start]
        
        # Check if limit exceeded
        if len(recent_requests) >= limit:
            # Calculate when they can retry
            if recent_requests:
                oldest_request = min(recent_requests)
                retry_after = max(1, int(oldest_request + window - current_time))
            else:
                retry_after = 60  # Default 1 minute if no requests found
            
            # Sanity check - retry_after should never be more than window
            retry_after = min(retry_after, int(window))
            
            return False, {
                "tier": tier,
                "limit": limit,
                "window_hours": window / 3600,
                "used": len(recent_requests),
                "remaining": 0,
                "retry_after": retry_after,
                "retry_at": current_time + retry_after,
                "message": f"Rate limit exceeded. Free tier: {limit} requests per {window/3600:.1f} hours"
            }
        
        # Add current request
        self._add_request(client_id, endpoint, current_time)
        recent_requests.append(current_time)
        
        return True, {
            "tier": tier,
            "limit": limit,
            "window_hours": window / 3600,
            "used": len(recent_requests),
            "remaining": limit - len(recent_requests),
            "reset_at": window_start + window,  # Next window start
            "message": f"Request allowed. {limit - len(recent_requests)} remaining in {window/3600:.1f} hour window"
        }
    
    def _get_requests(self, client_id: str, endpoint: str) -> list:
        """Get request timestamps for client."""
        key = f"rate_limit:{client_id}:{endpoint}"
        
        if self.redis_client:
            # Redis backend
            try:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"Redis read error: {e}")
        else:
            # File backend
            file_path = self.data_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"File read error: {e}")
        
        return []
    
    def _add_request(self, client_id: str, endpoint: str, timestamp: float):
        """Add request timestamp for client."""
        key = f"rate_limit:{client_id}:{endpoint}"
        
        # Get existing requests
        requests = self._get_requests(client_id, endpoint)
        
        # Add new timestamp
        requests.append(timestamp)
        
        # Keep only recent requests (last 5 hours for free tier)
        cutoff = timestamp - max(self.FREE_TIER_WINDOW, self.PREMIUM_TIER_WINDOW)
        requests = [t for t in requests if t > cutoff]
        
        if self.redis_client:
            # Redis backend
            try:
                self.redis_client.setex(
                    key, 
                    self.FREE_TIER_WINDOW,  # TTL
                    json.dumps(requests)
                )
            except Exception as e:
                logger.error(f"Redis write error: {e}")
        else:
            # File backend
            file_path = self.data_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
            try:
                with open(file_path, 'w') as f:
                    json.dump(requests, f)
            except Exception as e:
                logger.error(f"File write error: {e}")
    
    def reset_client(self, ip: str, wallet: Optional[str] = None):
        """Reset rate limit for a client (admin function)."""
        client_id = self._get_client_id(ip, wallet)
        key_pattern = f"rate_limit:{client_id}:*"
        
        if self.redis_client:
            try:
                for key in self.redis_client.scan_iter(match=key_pattern):
                    self.redis_client.delete(key)
                logger.info(f"Reset rate limit for {client_id}")
            except Exception as e:
                logger.error(f"Redis reset error: {e}")
        else:
            # File backend
            for file_path in self.data_dir.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    # Check if this file belongs to the client
                    # (This is a simplification - in production use proper key mapping)
                    file_path.unlink()
                except:
                    pass
    
    def get_client_status(self, ip: str, wallet: Optional[str] = None) -> Dict:
        """Get current rate limit status for client."""
        client_id = self._get_client_id(ip, wallet)
        tier = self._get_tier(wallet)
        
        # Get limits
        if tier == "premium":
            limit = self.PREMIUM_TIER_LIMIT
            window = self.PREMIUM_TIER_WINDOW
        else:
            limit = self.FREE_TIER_LIMIT
            window = self.FREE_TIER_WINDOW
        
        # Get usage
        requests = self._get_requests(client_id, "/chat")
        current_time = time.time()
        window_start = current_time - window
        recent_requests = [t for t in requests if t > window_start]
        
        return {
            "client_id": client_id,
            "tier": tier,
            "limit": limit,
            "window_hours": window / 3600,
            "used": len(recent_requests),
            "remaining": max(0, limit - len(recent_requests)),
            "reset_at": window_start + window if recent_requests else current_time + window
        }


# Singleton instance
_rate_limiter = None

def get_rate_limiter() -> RateLimiter:
    """Get or create rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        import os
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _rate_limiter = RateLimiter(redis_url=redis_url)
    return _rate_limiter