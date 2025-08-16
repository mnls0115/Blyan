"""
Advanced Key Revocation System with Redis TTL
=============================================

Handles immediate revocation, blocklists, and audit trails.
"""

import json
import time
import hashlib
from typing import Optional, Dict, List, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import redis
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class RevocationReason(str, Enum):
    """Standardized revocation reasons for audit"""
    COMPROMISED = "compromised"
    EXPIRED = "expired"
    USER_REQUEST = "user_request"
    ADMIN_ACTION = "admin_action"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ROTATION = "rotation"
    EMERGENCY = "emergency"

@dataclass
class RevocationRecord:
    """Detailed revocation record for audit trail"""
    key_id: str
    jti: str  # JWT ID
    revoked_at: str
    reason: RevocationReason
    revoked_by: Optional[str]
    metadata: Dict[str, any]
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, data: str) -> 'RevocationRecord':
        return cls(**json.loads(data))

class RevocationManager:
    """
    Manages API key revocation with distributed consistency.
    
    Features:
    - Immediate revocation with Redis
    - TTL-based automatic cleanup
    - Bloom filter for fast checking
    - Audit trail for compliance
    - Batch revocation support
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client or self._init_redis()
        
        # Key prefixes
        self.revocation_prefix = "revoked:"
        self.audit_prefix = "revocation_audit:"
        self.bloom_key = "revoked_bloom"
        
        # Configuration
        self.default_ttl = 86400 * 30  # 30 days
        self.audit_retention = 86400 * 365  # 1 year
        
        # Local cache for performance
        self._local_cache = {}
        self._cache_ttl = 60  # 1 minute local cache
        self._last_cache_clear = time.time()
    
    def _init_redis(self) -> redis.Redis:
        """Initialize Redis connection with retry"""
        import os
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    db=int(os.getenv("REDIS_DB", 2)),  # Separate DB for revocations
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                client.ping()
                logger.info(f"Revocation Redis connected")
                return client
            except redis.ConnectionError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to connect to Redis for revocations")
                    raise
                time.sleep(2 ** attempt)
    
    # ========================================================================
    # REVOCATION OPERATIONS
    # ========================================================================
    
    async def revoke_key(
        self,
        key_id: str,
        jti: str,
        reason: RevocationReason,
        revoked_by: Optional[str] = None,
        ttl: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Revoke an API key immediately.
        
        Args:
            key_id: The key identifier
            jti: JWT ID from the token
            reason: Revocation reason
            revoked_by: Who initiated the revocation
            ttl: Custom TTL in seconds (default: 30 days)
            metadata: Additional context
            
        Returns:
            True if successfully revoked
        """
        try:
            # Create revocation record
            record = RevocationRecord(
                key_id=key_id,
                jti=jti,
                revoked_at=datetime.now().isoformat(),
                reason=reason,
                revoked_by=revoked_by,
                metadata=metadata or {}
            )
            
            # Store in Redis with TTL
            revocation_key = f"{self.revocation_prefix}{jti}"
            ttl = ttl or self.default_ttl
            
            # Use pipeline for atomic operations
            pipe = self.redis.pipeline()
            
            # 1. Add to revocation list
            pipe.setex(revocation_key, ttl, record.to_json())
            
            # 2. Add to bloom filter for fast checking
            pipe.sadd(self.bloom_key, jti)
            
            # 3. Store audit record (longer retention)
            audit_key = f"{self.audit_prefix}{key_id}:{int(time.time())}"
            pipe.setex(audit_key, self.audit_retention, record.to_json())
            
            # 4. Publish revocation event for real-time updates
            pipe.publish("key_revoked", json.dumps({
                "jti": jti,
                "key_id": key_id,
                "reason": reason.value
            }))
            
            # Execute pipeline
            results = pipe.execute()
            
            # Clear local cache
            self._clear_local_cache(jti)
            
            logger.info(f"Key revoked: {key_id} (JTI: {jti}) - Reason: {reason.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke key {key_id}: {e}")
            return False
    
    async def is_revoked(self, jti: str) -> bool:
        """
        Check if a key is revoked (optimized for performance).
        
        Uses multiple layers:
        1. Local cache (fastest)
        2. Bloom filter (fast, may have false positives)
        3. Redis lookup (accurate)
        """
        # Check local cache first
        if self._check_local_cache(jti):
            return True
        
        try:
            # Quick check with bloom filter
            if not self.redis.sismember(self.bloom_key, jti):
                # Definitely not revoked
                self._update_local_cache(jti, False)
                return False
            
            # Bloom filter says maybe - check Redis
            revocation_key = f"{self.revocation_prefix}{jti}"
            exists = self.redis.exists(revocation_key)
            
            # Update local cache
            self._update_local_cache(jti, exists)
            
            return bool(exists)
            
        except redis.ConnectionError:
            # Redis down - fail open or closed based on security policy
            logger.error("Redis unavailable for revocation check")
            # Fail closed for security
            return True
        except Exception as e:
            logger.error(f"Error checking revocation for {jti}: {e}")
            return True  # Fail closed
    
    async def batch_revoke(
        self,
        keys: List[Tuple[str, str]],  # List of (key_id, jti) tuples
        reason: RevocationReason,
        revoked_by: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Revoke multiple keys at once (useful for emergency situations).
        
        Returns:
            Dict mapping key_id to success status
        """
        results = {}
        
        # Use pipeline for efficiency
        pipe = self.redis.pipeline()
        
        for key_id, jti in keys:
            record = RevocationRecord(
                key_id=key_id,
                jti=jti,
                revoked_at=datetime.now().isoformat(),
                reason=reason,
                revoked_by=revoked_by,
                metadata={"batch_revocation": True}
            )
            
            revocation_key = f"{self.revocation_prefix}{jti}"
            pipe.setex(revocation_key, self.default_ttl, record.to_json())
            pipe.sadd(self.bloom_key, jti)
        
        try:
            pipe.execute()
            results = {key_id: True for key_id, _ in keys}
            logger.info(f"Batch revoked {len(keys)} keys - Reason: {reason.value}")
        except Exception as e:
            logger.error(f"Batch revocation failed: {e}")
            results = {key_id: False for key_id, _ in keys}
        
        # Clear local cache for all revoked keys
        for _, jti in keys:
            self._clear_local_cache(jti)
        
        return results
    
    async def emergency_revoke_all(
        self,
        role: Optional[str] = None,
        created_before: Optional[datetime] = None,
        revoked_by: str = "system"
    ) -> int:
        """
        Emergency revocation of all keys or subset.
        
        Args:
            role: Revoke only keys with specific role
            created_before: Revoke keys created before this time
            revoked_by: Who initiated the emergency revocation
            
        Returns:
            Number of keys revoked
        """
        # This would need to scan all active keys in the main key store
        # For now, we'll add a marker that causes all validations to fail
        
        emergency_key = "emergency_revocation"
        emergency_data = {
            "activated_at": datetime.now().isoformat(),
            "role_filter": role,
            "created_before": created_before.isoformat() if created_before else None,
            "revoked_by": revoked_by
        }
        
        try:
            # Set emergency flag with 1 hour TTL (should be resolved by then)
            self.redis.setex(emergency_key, 3600, json.dumps(emergency_data))
            
            # Publish emergency event
            self.redis.publish("emergency_revocation", json.dumps(emergency_data))
            
            logger.critical(f"EMERGENCY REVOCATION activated by {revoked_by}")
            
            # In production, this would trigger additional workflows
            return -1  # Indicates emergency mode
            
        except Exception as e:
            logger.error(f"Failed to activate emergency revocation: {e}")
            return 0
    
    # ========================================================================
    # AUDIT & COMPLIANCE
    # ========================================================================
    
    async def get_revocation_history(
        self,
        key_id: Optional[str] = None,
        limit: int = 100
    ) -> List[RevocationRecord]:
        """
        Get revocation audit history.
        
        Args:
            key_id: Filter by specific key (None for all)
            limit: Maximum records to return
            
        Returns:
            List of revocation records
        """
        pattern = f"{self.audit_prefix}{key_id}:*" if key_id else f"{self.audit_prefix}*"
        
        records = []
        cursor = 0
        
        try:
            # Scan for audit records
            while len(records) < limit:
                cursor, keys = self.redis.scan(
                    cursor,
                    match=pattern,
                    count=min(100, limit - len(records))
                )
                
                if keys:
                    # Get records in batch
                    pipe = self.redis.pipeline()
                    for key in keys:
                        pipe.get(key)
                    
                    values = pipe.execute()
                    
                    for value in values:
                        if value:
                            records.append(RevocationRecord.from_json(value))
                
                if cursor == 0:
                    break
            
            # Sort by revocation time (newest first)
            records.sort(key=lambda r: r.revoked_at, reverse=True)
            
            return records[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get revocation history: {e}")
            return []
    
    async def get_revocation_stats(self) -> Dict[str, any]:
        """Get revocation statistics for monitoring"""
        try:
            # Count total revoked keys
            total_revoked = self.redis.scard(self.bloom_key)
            
            # Get recent revocations (last 24 hours)
            recent_pattern = f"{self.audit_prefix}*"
            recent_count = 0
            
            now = time.time()
            day_ago = now - 86400
            
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(cursor, match=recent_pattern, count=100)
                
                for key in keys:
                    # Extract timestamp from key
                    timestamp = int(key.split(":")[-1])
                    if timestamp > day_ago:
                        recent_count += 1
                
                if cursor == 0:
                    break
            
            # Check for emergency revocation
            emergency_active = self.redis.exists("emergency_revocation")
            
            return {
                "total_revoked": total_revoked,
                "revoked_last_24h": recent_count,
                "emergency_active": bool(emergency_active),
                "bloom_filter_size": self.redis.memory_usage(self.bloom_key) or 0,
                "audit_records": self.redis.dbsize()  # Approximate
            }
            
        except Exception as e:
            logger.error(f"Failed to get revocation stats: {e}")
            return {}
    
    # ========================================================================
    # CACHE MANAGEMENT
    # ========================================================================
    
    def _check_local_cache(self, jti: str) -> Optional[bool]:
        """Check local cache for revocation status"""
        # Clear old cache periodically
        if time.time() - self._last_cache_clear > self._cache_ttl:
            self._local_cache.clear()
            self._last_cache_clear = time.time()
            return None
        
        return self._local_cache.get(jti)
    
    def _update_local_cache(self, jti: str, revoked: bool):
        """Update local cache"""
        self._local_cache[jti] = revoked
    
    def _clear_local_cache(self, jti: Optional[str] = None):
        """Clear local cache"""
        if jti:
            self._local_cache.pop(jti, None)
        else:
            self._local_cache.clear()
    
    # ========================================================================
    # CLEANUP & MAINTENANCE
    # ========================================================================
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired revocation records.
        Redis handles TTL automatically, but this can force cleanup.
        
        Returns:
            Number of records cleaned
        """
        # Redis handles TTL automatically
        # This method is for forced cleanup if needed
        
        pattern = f"{self.revocation_prefix}*"
        cleaned = 0
        
        try:
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
                
                if keys:
                    # Check TTL for each key
                    pipe = self.redis.pipeline()
                    for key in keys:
                        pipe.ttl(key)
                    
                    ttls = pipe.execute()
                    
                    # Remove from bloom filter if expired
                    for key, ttl in zip(keys, ttls):
                        if ttl == -2:  # Key doesn't exist
                            jti = key.replace(self.revocation_prefix, "")
                            self.redis.srem(self.bloom_key, jti)
                            cleaned += 1
                
                if cursor == 0:
                    break
            
            if cleaned > 0:
                logger.info(f"Cleaned {cleaned} expired revocation records")
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0
    
    async def rebuild_bloom_filter(self) -> bool:
        """
        Rebuild bloom filter from current revocation records.
        Useful after Redis restart or data corruption.
        """
        try:
            # Clear existing bloom filter
            self.redis.delete(self.bloom_key)
            
            # Scan all revocation records
            pattern = f"{self.revocation_prefix}*"
            cursor = 0
            rebuilt = 0
            
            while True:
                cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
                
                if keys:
                    # Extract JTIs and add to bloom filter
                    jtis = [key.replace(self.revocation_prefix, "") for key in keys]
                    if jtis:
                        self.redis.sadd(self.bloom_key, *jtis)
                        rebuilt += len(jtis)
                
                if cursor == 0:
                    break
            
            logger.info(f"Rebuilt bloom filter with {rebuilt} entries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild bloom filter: {e}")
            return False

# Global instance
revocation_manager = RevocationManager()

"""
Integration Example:

from backend.auth.revocation import revocation_manager, RevocationReason

# In API key validation
async def validate_api_key(token: str):
    # Decode JWT to get JTI
    payload = jwt.decode(token, ...)
    jti = payload.get("jti")
    
    # Check if revoked
    if await revocation_manager.is_revoked(jti):
        raise HTTPException(401, "API key has been revoked")
    
    # Continue validation...

# Revoke a compromised key
await revocation_manager.revoke_key(
    key_id="user_123_key",
    jti="abc123",
    reason=RevocationReason.COMPROMISED,
    revoked_by="admin@example.com",
    metadata={"ip": "1.2.3.4", "user_agent": "suspicious"}
)

# Emergency revocation
await revocation_manager.emergency_revoke_all(
    role="node_operator",
    revoked_by="security_team"
)

# Get audit history
history = await revocation_manager.get_revocation_history(
    key_id="user_123_key",
    limit=10
)
"""