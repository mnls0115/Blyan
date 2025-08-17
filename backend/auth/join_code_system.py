#!/usr/bin/env python3
"""
Production JOIN_CODE system for secure node enrollment.
Single-use codes that are exchanged for permanent node credentials.
"""

import os
import json
import time
import secrets
import hashlib
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import redis
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class JoinCode:
    """Join code data structure"""
    code: str
    created_at: float
    expires_at: float
    node_type: str = "gpu"
    used: bool = False
    used_by: Optional[str] = None
    used_at: Optional[float] = None
    ip_address: Optional[str] = None
    metadata: Dict[str, Any] = None

    def is_valid(self) -> bool:
        """Check if code is still valid"""
        if self.used:
            return False
        if time.time() > self.expires_at:
            return False
        return True

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class NodeCredentials:
    """Node credentials issued after JOIN_CODE validation"""
    node_id: str
    node_key: str
    issued_at: float
    expires_at: float
    role: str = "node_operator"
    capabilities: Dict[str, Any] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


class JoinCodeManager:
    """Manages JOIN_CODE creation, validation, and exchange for credentials"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize JOIN_CODE manager
        
        Args:
            redis_client: Redis client for distributed storage
        """
        self.redis_client = redis_client or self._init_redis()
        
        # Configuration
        self.code_length = 8  # Characters in join code
        self.code_ttl = 1800  # 30 minutes
        self.rate_limit_window = 300  # 5 minutes
        self.rate_limit_max = 3  # Max codes per IP in window
        self.node_key_ttl = 7776000  # 90 days for node credentials
        
        # Prefixes for Redis keys
        self.prefix_code = "join_code:"
        self.prefix_node = "node_cred:"
        self.prefix_rate = "rate_limit:join:"
        self.prefix_audit = "audit:join:"
        
    def _init_redis(self) -> redis.Redis:
        """Initialize Redis connection"""
        try:
            client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=int(os.getenv("REDIS_DB", 0)),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            client.ping()
            logger.info("Redis connected for JOIN_CODE system")
            return client
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory storage: {e}")
            # Fallback to in-memory storage (not recommended for production)
            return None
    
    def generate_join_code(
        self, 
        node_type: str = "gpu",
        ip_address: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[JoinCode, str]:
        """
        Generate a new JOIN_CODE
        
        Args:
            node_type: Type of node (gpu, cpu, storage)
            ip_address: IP address of requester
            metadata: Additional metadata
            
        Returns:
            Tuple of (JoinCode object, error message if any)
        """
        # Check rate limit
        if ip_address and not self._check_rate_limit(ip_address):
            return None, "Rate limit exceeded. Please wait before requesting another code."
        
        # Generate secure random code
        code = self._generate_secure_code()
        
        # Create JoinCode object
        now = time.time()
        join_code = JoinCode(
            code=code,
            created_at=now,
            expires_at=now + self.code_ttl,
            node_type=node_type,
            used=False,
            ip_address=ip_address,
            metadata=metadata or {}
        )
        
        # Store in Redis
        if self.redis_client:
            key = f"{self.prefix_code}{code}"
            self.redis_client.setex(
                key,
                self.code_ttl,
                json.dumps(join_code.to_dict())
            )
            
            # Update rate limit
            if ip_address:
                rate_key = f"{self.prefix_rate}{ip_address}"
                pipe = self.redis_client.pipeline()
                pipe.incr(rate_key)
                pipe.expire(rate_key, self.rate_limit_window)
                pipe.execute()
            
            # Audit log
            self._audit_log("generate", code, ip_address, {"node_type": node_type})
        
        logger.info(f"Generated JOIN_CODE: {code[:4]}**** for {node_type} node from {ip_address}")
        
        return join_code, None
    
    def validate_and_exchange(
        self,
        code: str,
        node_meta: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ) -> Tuple[Optional[NodeCredentials], str]:
        """
        Validate JOIN_CODE and exchange for permanent node credentials
        
        Args:
            code: The JOIN_CODE to validate
            node_meta: Node metadata (GPU info, location, etc.)
            ip_address: IP address of the node
            
        Returns:
            Tuple of (NodeCredentials if successful, error message)
        """
        if not code or len(code) < self.code_length:
            return None, "Invalid code format"
        
        # Retrieve code from Redis
        if not self.redis_client:
            return None, "Storage system unavailable"
        
        key = f"{self.prefix_code}{code}"
        code_data = self.redis_client.get(key)
        
        if not code_data:
            self._audit_log("invalid_attempt", code, ip_address)
            return None, "Invalid or expired code"
        
        # Parse join code
        try:
            join_code = JoinCode(**json.loads(code_data))
        except Exception as e:
            logger.error(f"Failed to parse join code: {e}")
            return None, "Invalid code data"
        
        # Validate code
        if not join_code.is_valid():
            if join_code.used:
                self._audit_log("reuse_attempt", code, ip_address, {"used_by": join_code.used_by})
                return None, "Code has already been used"
            else:
                self._audit_log("expired_attempt", code, ip_address)
                return None, "Code has expired"
        
        # Generate node credentials
        node_id = self._generate_node_id(node_meta)
        node_key = self._generate_node_key()
        
        now = time.time()
        credentials = NodeCredentials(
            node_id=node_id,
            node_key=node_key,
            issued_at=now,
            expires_at=now + self.node_key_ttl,
            role="node_operator",
            capabilities={
                "node_type": join_code.node_type,
                "enrolled_from": ip_address,
                "meta": node_meta or {}
            }
        )
        
        # Mark code as used (atomic operation)
        join_code.used = True
        join_code.used_by = node_id
        join_code.used_at = now
        
        # Store updated code and new credentials
        pipe = self.redis_client.pipeline()
        
        # Update join code
        pipe.setex(
            key,
            86400,  # Keep used codes for 24h for audit
            json.dumps(join_code.to_dict())
        )
        
        # Store node credentials
        node_key_redis = f"{self.prefix_node}{node_id}"
        pipe.setex(
            node_key_redis,
            self.node_key_ttl,
            json.dumps(credentials.to_dict())
        )
        
        # Store reverse lookup (node_key -> node_id)
        key_hash = hashlib.sha256(node_key.encode()).hexdigest()
        pipe.setex(
            f"{self.prefix_node}key:{key_hash}",
            self.node_key_ttl,
            node_id
        )
        
        pipe.execute()
        
        # Audit log
        self._audit_log("exchange_success", code, ip_address, {
            "node_id": node_id,
            "node_type": join_code.node_type
        })
        
        logger.info(f"Successfully exchanged JOIN_CODE for node {node_id}")
        
        return credentials, None
    
    def verify_node_credentials(self, node_id: str, node_key: str) -> bool:
        """
        Verify node credentials
        
        Args:
            node_id: Node identifier
            node_key: Node key to verify
            
        Returns:
            True if credentials are valid
        """
        if not self.redis_client:
            return False
        
        # Get stored credentials
        key = f"{self.prefix_node}{node_id}"
        cred_data = self.redis_client.get(key)
        
        if not cred_data:
            return False
        
        try:
            stored_creds = NodeCredentials(**json.loads(cred_data))
            
            # Check expiry
            if time.time() > stored_creds.expires_at:
                logger.info(f"Node {node_id} credentials expired")
                return False
            
            # Verify key
            if stored_creds.node_key != node_key:
                logger.warning(f"Invalid key for node {node_id}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify credentials: {e}")
            return False
    
    def _generate_secure_code(self) -> str:
        """Generate cryptographically secure join code"""
        # Use uppercase alphanumeric without ambiguous characters
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        code = ''.join(secrets.choice(alphabet) for _ in range(self.code_length))
        return code
    
    def _generate_node_id(self, node_meta: Optional[Dict[str, Any]] = None) -> str:
        """Generate unique node identifier"""
        prefix = "gpu"
        if node_meta and "node_type" in node_meta:
            prefix = node_meta["node_type"][:3].lower()
        
        # Generate unique suffix
        unique_part = secrets.token_hex(4)
        timestamp_part = hex(int(time.time()))[2:][-4:]
        
        return f"{prefix}-{timestamp_part}-{unique_part}"
    
    def _generate_node_key(self) -> str:
        """Generate secure node key"""
        # 32 bytes = 256 bits of entropy
        return secrets.token_urlsafe(32)
    
    def _check_rate_limit(self, ip_address: str) -> bool:
        """Check if IP is within rate limit"""
        if not self.redis_client:
            return True  # Allow if Redis not available
        
        key = f"{self.prefix_rate}{ip_address}"
        current_count = self.redis_client.get(key)
        
        if current_count and int(current_count) >= self.rate_limit_max:
            return False
        
        return True
    
    def _audit_log(
        self, 
        action: str, 
        code: str, 
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log audit event"""
        if not self.redis_client:
            return
        
        audit_entry = {
            "action": action,
            "code": code[:4] + "****" if len(code) > 4 else code,
            "ip": ip_address,
            "timestamp": time.time(),
            "details": details or {}
        }
        
        # Store audit log with expiry
        audit_key = f"{self.prefix_audit}{int(time.time())}-{secrets.token_hex(4)}"
        self.redis_client.setex(
            audit_key,
            604800,  # Keep audit logs for 7 days
            json.dumps(audit_entry)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get JOIN_CODE system statistics"""
        if not self.redis_client:
            return {"status": "redis_unavailable"}
        
        stats = {
            "active_codes": 0,
            "used_codes": 0,
            "enrolled_nodes": 0,
            "recent_enrollments": []
        }
        
        # Count active and used codes
        for key in self.redis_client.scan_iter(f"{self.prefix_code}*"):
            code_data = self.redis_client.get(key)
            if code_data:
                try:
                    join_code = JoinCode(**json.loads(code_data))
                    if join_code.used:
                        stats["used_codes"] += 1
                    elif join_code.is_valid():
                        stats["active_codes"] += 1
                except:
                    pass
        
        # Count enrolled nodes
        for key in self.redis_client.scan_iter(f"{self.prefix_node}*"):
            if not key.startswith(f"{self.prefix_node}key:"):
                stats["enrolled_nodes"] += 1
        
        return stats


# Singleton instance
_join_code_manager = None

def get_join_code_manager() -> JoinCodeManager:
    """Get singleton JoinCodeManager instance"""
    global _join_code_manager
    if _join_code_manager is None:
        _join_code_manager = JoinCodeManager()
    return _join_code_manager