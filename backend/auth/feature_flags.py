"""
Feature Flags for Zero-Downtime API Key V2 Rollout
===================================================

Enables gradual traffic shifting from V1 to V2 without restarts.
"""

import os
import json
import random
import hashlib
from typing import Dict, Optional, List
from datetime import datetime
from enum import Enum
import redis
import logging

logger = logging.getLogger(__name__)

class RolloutStrategy(str, Enum):
    """Different rollout strategies for V2 migration"""
    OFF = "off"                    # V2 disabled
    CANARY = "canary"              # Specific test users only
    PERCENTAGE = "percentage"       # Random percentage of traffic
    GRADUAL = "gradual"           # Time-based gradual increase
    FULL = "full"                  # All traffic to V2

class FeatureFlagManager:
    """
    Manages feature flags for V2 API key system rollout.
    Supports multiple strategies for zero-downtime deployment.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client or self._init_redis()
        self.config_key = "feature_flags:api_key_v2"
        self.canary_users_key = "feature_flags:canary_users"
        
        # Default configuration
        self.default_config = {
            "strategy": RolloutStrategy.OFF.value,
            "percentage": 0,
            "canary_users": [],
            "start_time": None,
            "ramp_duration_hours": 24,
            "enabled_endpoints": [],
            "override_users": {},  # user_id -> force V1/V2
            "metrics": {
                "v1_requests": 0,
                "v2_requests": 0,
                "v1_errors": 0,
                "v2_errors": 0,
                "last_updated": None
            }
        }
        
        # Load or initialize configuration
        self.load_config()
    
    def _init_redis(self) -> redis.Redis:
        """Initialize Redis connection"""
        return redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 1)),  # Different DB for feature flags
            decode_responses=True
        )
    
    def load_config(self):
        """Load configuration from Redis or use defaults"""
        try:
            stored = self.redis.get(self.config_key)
            if stored:
                self.config = json.loads(stored)
            else:
                self.config = self.default_config.copy()
                self.save_config()
        except Exception as e:
            logger.error(f"Error loading feature flags: {e}")
            self.config = self.default_config.copy()
    
    def save_config(self):
        """Save configuration to Redis"""
        try:
            self.config["metrics"]["last_updated"] = datetime.now().isoformat()
            self.redis.setex(
                self.config_key,
                86400,  # 24 hour TTL
                json.dumps(self.config)
            )
        except Exception as e:
            logger.error(f"Error saving feature flags: {e}")
    
    def should_use_v2(self, user_id: Optional[str] = None, endpoint: str = "") -> bool:
        """
        Determine if a request should use V2 based on rollout strategy.
        
        Args:
            user_id: Optional user identifier for consistent routing
            endpoint: API endpoint being accessed
            
        Returns:
            True if V2 should be used, False for V1
        """
        strategy = RolloutStrategy(self.config.get("strategy", "off"))
        
        # Check if endpoint is enabled for V2
        enabled_endpoints = self.config.get("enabled_endpoints", [])
        if enabled_endpoints and endpoint not in enabled_endpoints:
            self.track_request("v1", endpoint)
            return False
        
        # Check user overrides
        if user_id and user_id in self.config.get("override_users", {}):
            use_v2 = self.config["override_users"][user_id] == "v2"
            self.track_request("v2" if use_v2 else "v1", endpoint)
            return use_v2
        
        # Apply strategy
        if strategy == RolloutStrategy.OFF:
            self.track_request("v1", endpoint)
            return False
            
        elif strategy == RolloutStrategy.FULL:
            self.track_request("v2", endpoint)
            return True
            
        elif strategy == RolloutStrategy.CANARY:
            canary_users = self.get_canary_users()
            use_v2 = user_id in canary_users if user_id else False
            self.track_request("v2" if use_v2 else "v1", endpoint)
            return use_v2
            
        elif strategy == RolloutStrategy.PERCENTAGE:
            percentage = self.config.get("percentage", 0)
            # Use consistent hashing for user stickiness
            if user_id:
                hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
                use_v2 = (hash_val % 100) < percentage
            else:
                use_v2 = random.random() * 100 < percentage
            self.track_request("v2" if use_v2 else "v1", endpoint)
            return use_v2
            
        elif strategy == RolloutStrategy.GRADUAL:
            # Time-based gradual rollout
            start_time = self.config.get("start_time")
            if not start_time:
                self.track_request("v1", endpoint)
                return False
                
            start = datetime.fromisoformat(start_time)
            now = datetime.now()
            ramp_hours = self.config.get("ramp_duration_hours", 24)
            
            hours_elapsed = (now - start).total_seconds() / 3600
            percentage = min(100, (hours_elapsed / ramp_hours) * 100)
            
            # Use same logic as percentage strategy
            if user_id:
                hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
                use_v2 = (hash_val % 100) < percentage
            else:
                use_v2 = random.random() * 100 < percentage
                
            self.track_request("v2" if use_v2 else "v1", endpoint)
            return use_v2
        
        # Default to V1
        self.track_request("v1", endpoint)
        return False
    
    def get_canary_users(self) -> List[str]:
        """Get list of canary test users"""
        try:
            users = self.redis.smembers(self.canary_users_key)
            return list(users) if users else []
        except:
            return self.config.get("canary_users", [])
    
    def add_canary_user(self, user_id: str):
        """Add a user to canary test group"""
        try:
            self.redis.sadd(self.canary_users_key, user_id)
            logger.info(f"Added canary user: {user_id}")
        except Exception as e:
            logger.error(f"Error adding canary user: {e}")
    
    def track_request(self, version: str, endpoint: str):
        """Track metrics for monitoring"""
        try:
            metric_key = f"v{version[1]}_requests"
            self.config["metrics"][metric_key] = \
                self.config["metrics"].get(metric_key, 0) + 1
            
            # Save periodically (every 100 requests)
            if sum(self.config["metrics"].get(f"v{v}_requests", 0) 
                   for v in ["1", "2"]) % 100 == 0:
                self.save_config()
        except:
            pass
    
    def track_error(self, version: str, error_type: str):
        """Track errors for monitoring"""
        try:
            metric_key = f"v{version[1]}_errors"
            self.config["metrics"][metric_key] = \
                self.config["metrics"].get(metric_key, 0) + 1
            self.save_config()
        except:
            pass
    
    def set_rollout_strategy(self, strategy: RolloutStrategy, **kwargs):
        """
        Update rollout strategy.
        
        Args:
            strategy: The rollout strategy to use
            **kwargs: Additional parameters (percentage, ramp_duration_hours, etc.)
        """
        self.config["strategy"] = strategy.value
        
        if strategy == RolloutStrategy.PERCENTAGE:
            self.config["percentage"] = kwargs.get("percentage", 0)
            
        elif strategy == RolloutStrategy.GRADUAL:
            self.config["start_time"] = kwargs.get("start_time", datetime.now().isoformat())
            self.config["ramp_duration_hours"] = kwargs.get("ramp_duration_hours", 24)
        
        self.save_config()
        logger.info(f"Rollout strategy updated to: {strategy.value}")
    
    def enable_endpoints(self, endpoints: List[str]):
        """Enable specific endpoints for V2"""
        self.config["enabled_endpoints"] = endpoints
        self.save_config()
        logger.info(f"Enabled V2 for endpoints: {endpoints}")
    
    def override_user(self, user_id: str, version: str):
        """Force a specific user to V1 or V2"""
        if "override_users" not in self.config:
            self.config["override_users"] = {}
        
        if version.lower() in ["v1", "v2"]:
            self.config["override_users"][user_id] = version.lower()
            self.save_config()
            logger.info(f"User {user_id} overridden to {version}")
    
    def get_metrics(self) -> Dict:
        """Get current metrics for monitoring"""
        metrics = self.config.get("metrics", {})
        
        # Calculate percentages
        total_requests = metrics.get("v1_requests", 0) + metrics.get("v2_requests", 0)
        if total_requests > 0:
            metrics["v1_percentage"] = (metrics.get("v1_requests", 0) / total_requests) * 100
            metrics["v2_percentage"] = (metrics.get("v2_requests", 0) / total_requests) * 100
        
        # Calculate error rates
        v1_requests = metrics.get("v1_requests", 0)
        v2_requests = metrics.get("v2_requests", 0)
        
        if v1_requests > 0:
            metrics["v1_error_rate"] = (metrics.get("v1_errors", 0) / v1_requests) * 100
        
        if v2_requests > 0:
            metrics["v2_error_rate"] = (metrics.get("v2_errors", 0) / v2_requests) * 100
        
        metrics["current_strategy"] = self.config.get("strategy", "off")
        
        return metrics
    
    def reset_metrics(self):
        """Reset metrics counters"""
        self.config["metrics"] = {
            "v1_requests": 0,
            "v2_requests": 0,
            "v1_errors": 0,
            "v2_errors": 0,
            "last_updated": datetime.now().isoformat()
        }
        self.save_config()

# Global instance
feature_flags = FeatureFlagManager()

# Convenience decorators
def with_feature_flag(endpoint: str):
    """
    Decorator to automatically route requests based on feature flags.
    
    Example:
        @with_feature_flag("/api/auth/register")
        async def register_api_key(request):
            # Will automatically use V1 or V2 based on flags
            pass
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user_id from request if available
            request = args[0] if args else None
            user_id = None
            
            if hasattr(request, "headers"):
                # Try to extract user ID from auth header or session
                auth_header = request.headers.get("Authorization", "")
                if auth_header:
                    # Simple extraction - customize based on your auth
                    user_id = hashlib.md5(auth_header.encode()).hexdigest()[:8]
            
            # Determine which version to use
            use_v2 = feature_flags.should_use_v2(user_id, endpoint)
            
            # Add version info to request for handler to use
            if request:
                request.state.api_version = "v2" if use_v2 else "v1"
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

"""
Usage Examples:

# 1. Start with canary deployment
feature_flags.set_rollout_strategy(RolloutStrategy.CANARY)
feature_flags.add_canary_user("test_user_001")

# 2. Move to 10% rollout
feature_flags.set_rollout_strategy(RolloutStrategy.PERCENTAGE, percentage=10)

# 3. Gradual rollout over 48 hours
feature_flags.set_rollout_strategy(RolloutStrategy.GRADUAL, ramp_duration_hours=48)

# 4. Full rollout
feature_flags.set_rollout_strategy(RolloutStrategy.FULL)

# 5. Emergency rollback
feature_flags.set_rollout_strategy(RolloutStrategy.OFF)

# 6. Check metrics
metrics = feature_flags.get_metrics()
print(f"V2 adoption: {metrics['v2_percentage']:.1f}%")
print(f"V2 error rate: {metrics['v2_error_rate']:.2f}%")
"""