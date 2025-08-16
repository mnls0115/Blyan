"""
JWT Secret Rotation System
==========================

Zero-downtime secret rotation with multi-version support.
"""

import os
import secrets
import json
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import redis
import jwt
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

logger = logging.getLogger(__name__)

class RotationState(str, Enum):
    """Secret rotation states"""
    ACTIVE = "active"           # Current signing secret
    ROTATING = "rotating"       # New secret, signing with both
    PREVIOUS = "previous"       # Old secret, only for validation
    EXPIRED = "expired"         # No longer valid

@dataclass
class SecretVersion:
    """JWT secret version with metadata"""
    version_id: str
    secret: str
    created_at: str
    state: RotationState
    expires_at: Optional[str]
    created_by: str
    metadata: Dict[str, Any]
    
    def to_json(self) -> str:
        # Don't serialize the actual secret
        data = asdict(self)
        data["secret"] = "REDACTED"
        return json.dumps(data)
    
    def is_valid(self) -> bool:
        """Check if secret is still valid"""
        if self.state == RotationState.EXPIRED:
            return False
        
        if self.expires_at:
            expiry = datetime.fromisoformat(self.expires_at)
            if datetime.now() > expiry:
                return False
        
        return True

class SecretRotationManager:
    """
    Manages JWT secret rotation for zero-downtime updates.
    
    Features:
    - Multiple secret versions for gradual rotation
    - Encrypted secret storage in Redis
    - Automatic expiration of old secrets
    - Audit trail for compliance
    - Multi-node synchronization
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client or self._init_redis()
        
        # Key prefixes
        self.secret_prefix = "jwt_secret:"
        self.version_prefix = "jwt_version:"
        self.audit_prefix = "secret_rotation_audit:"
        
        # Configuration
        self.rotation_overlap = timedelta(hours=24)  # Keep old secret valid for 24h
        self.max_versions = 3  # Maximum concurrent secret versions
        
        # Encryption for secrets at rest
        self.master_key = self._get_or_create_master_key()
        self.cipher = Fernet(self.master_key)
        
        # Cache for performance
        self._secret_cache = {}
        self._cache_expiry = {}
        
        # Load current secrets
        self._load_secrets()
    
    def _init_redis(self) -> redis.Redis:
        """Initialize Redis connection"""
        return redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 3)),  # Separate DB for secrets
            decode_responses=False  # Binary data for encryption
        )
    
    def _get_or_create_master_key(self) -> bytes:
        """
        Get or create master encryption key.
        In production, this should use KMS or HSM.
        """
        key_file = os.getenv("MASTER_KEY_FILE", "/secure/master.key")
        
        # For demo, use environment variable
        if os.getenv("MASTER_KEY"):
            return os.getenv("MASTER_KEY").encode()[:32].ljust(32, b'0')
        
        # Generate new key
        key = Fernet.generate_key()
        logger.warning("Generated new master key - store securely!")
        return key
    
    def _encrypt_secret(self, secret: str) -> bytes:
        """Encrypt secret for storage"""
        return self.cipher.encrypt(secret.encode())
    
    def _decrypt_secret(self, encrypted: bytes) -> str:
        """Decrypt secret from storage"""
        return self.cipher.decrypt(encrypted).decode()
    
    # ========================================================================
    # SECRET MANAGEMENT
    # ========================================================================
    
    def _load_secrets(self):
        """Load all valid secrets into cache"""
        try:
            # Get all secret versions
            pattern = f"{self.version_prefix}*"
            cursor = 0
            
            self._secret_cache.clear()
            
            while True:
                cursor, keys = self.redis.scan(cursor, match=pattern.encode(), count=10)
                
                for key in keys:
                    version_data = self.redis.get(key)
                    if version_data:
                        version_info = json.loads(version_data)
                        version_id = version_info["version_id"]
                        
                        # Get encrypted secret
                        secret_key = f"{self.secret_prefix}{version_id}".encode()
                        encrypted_secret = self.redis.get(secret_key)
                        
                        if encrypted_secret:
                            # Decrypt and cache
                            secret = self._decrypt_secret(encrypted_secret)
                            
                            version = SecretVersion(
                                version_id=version_id,
                                secret=secret,
                                created_at=version_info["created_at"],
                                state=RotationState(version_info["state"]),
                                expires_at=version_info.get("expires_at"),
                                created_by=version_info["created_by"],
                                metadata=version_info.get("metadata", {})
                            )
                            
                            if version.is_valid():
                                self._secret_cache[version_id] = version
                                self._cache_expiry[version_id] = time.time() + 300  # 5 min cache
                
                if cursor == 0:
                    break
            
            logger.info(f"Loaded {len(self._secret_cache)} valid JWT secrets")
            
        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            # Fall back to environment variable
            default_secret = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
            self._secret_cache["default"] = SecretVersion(
                version_id="default",
                secret=default_secret,
                created_at=datetime.now().isoformat(),
                state=RotationState.ACTIVE,
                expires_at=None,
                created_by="system",
                metadata={}
            )
    
    def get_current_secret(self) -> str:
        """Get current active secret for signing"""
        # Check cache
        for version in self._secret_cache.values():
            if version.state == RotationState.ACTIVE:
                return version.secret
        
        # Reload if not found
        self._load_secrets()
        
        for version in self._secret_cache.values():
            if version.state == RotationState.ACTIVE:
                return version.secret
        
        # Fall back to default
        logger.warning("No active secret found, using default")
        return self._secret_cache.get("default", SecretVersion(
            version_id="fallback",
            secret=secrets.token_urlsafe(32),
            created_at=datetime.now().isoformat(),
            state=RotationState.ACTIVE,
            expires_at=None,
            created_by="system",
            metadata={}
        )).secret
    
    def get_all_valid_secrets(self) -> List[str]:
        """Get all secrets valid for verification"""
        # Clean expired from cache
        now = time.time()
        expired_keys = [
            k for k, expiry in self._cache_expiry.items()
            if expiry < now
        ]
        for k in expired_keys:
            del self._secret_cache[k]
            del self._cache_expiry[k]
        
        # Reload if cache is empty
        if not self._secret_cache:
            self._load_secrets()
        
        # Return all valid secrets
        valid_secrets = []
        for version in self._secret_cache.values():
            if version.is_valid() and version.state != RotationState.EXPIRED:
                valid_secrets.append(version.secret)
        
        return valid_secrets if valid_secrets else [self.get_current_secret()]
    
    # ========================================================================
    # ROTATION OPERATIONS
    # ========================================================================
    
    async def initiate_rotation(
        self,
        initiated_by: str,
        reason: str = "scheduled",
        immediate: bool = False
    ) -> Tuple[bool, str]:
        """
        Initiate secret rotation.
        
        Args:
            initiated_by: Who initiated the rotation
            reason: Reason for rotation
            immediate: If True, immediately expire old secret
            
        Returns:
            (success, new_version_id)
        """
        try:
            # Generate new secret
            new_secret = secrets.token_urlsafe(32)
            new_version_id = f"v_{int(time.time())}_{secrets.token_hex(4)}"
            
            # Get current active secret
            current_version = None
            for version in self._secret_cache.values():
                if version.state == RotationState.ACTIVE:
                    current_version = version
                    break
            
            # Create new version
            new_version = SecretVersion(
                version_id=new_version_id,
                secret=new_secret,
                created_at=datetime.now().isoformat(),
                state=RotationState.ROTATING,
                expires_at=None,
                created_by=initiated_by,
                metadata={"reason": reason}
            )
            
            # Store encrypted secret
            secret_key = f"{self.secret_prefix}{new_version_id}".encode()
            encrypted = self._encrypt_secret(new_secret)
            
            # Use pipeline for atomic operations
            pipe = self.redis.pipeline()
            
            # Store new secret
            pipe.set(secret_key, encrypted)
            
            # Store version info
            version_key = f"{self.version_prefix}{new_version_id}".encode()
            version_data = {
                "version_id": new_version_id,
                "created_at": new_version.created_at,
                "state": new_version.state.value,
                "expires_at": new_version.expires_at,
                "created_by": new_version.created_by,
                "metadata": new_version.metadata
            }
            pipe.set(version_key, json.dumps(version_data))
            
            # Update current version state
            if current_version:
                if immediate:
                    # Immediately expire old secret
                    current_version.state = RotationState.EXPIRED
                    current_version.expires_at = datetime.now().isoformat()
                else:
                    # Keep old secret valid for overlap period
                    current_version.state = RotationState.PREVIOUS
                    current_version.expires_at = (
                        datetime.now() + self.rotation_overlap
                    ).isoformat()
                
                # Update in Redis
                old_version_key = f"{self.version_prefix}{current_version.version_id}".encode()
                old_version_data = {
                    "version_id": current_version.version_id,
                    "created_at": current_version.created_at,
                    "state": current_version.state.value,
                    "expires_at": current_version.expires_at,
                    "created_by": current_version.created_by,
                    "metadata": current_version.metadata
                }
                pipe.set(old_version_key, json.dumps(old_version_data))
            
            # Create audit record
            audit_key = f"{self.audit_prefix}{int(time.time())}".encode()
            audit_data = {
                "action": "rotation_initiated",
                "new_version": new_version_id,
                "old_version": current_version.version_id if current_version else None,
                "initiated_by": initiated_by,
                "reason": reason,
                "immediate": immediate,
                "timestamp": datetime.now().isoformat()
            }
            pipe.setex(audit_key, 86400 * 365, json.dumps(audit_data))  # 1 year retention
            
            # Publish rotation event
            pipe.publish("secret_rotation", json.dumps({
                "event": "rotation_started",
                "new_version": new_version_id
            }))
            
            # Execute pipeline
            pipe.execute()
            
            # Update cache
            self._secret_cache[new_version_id] = new_version
            self._cache_expiry[new_version_id] = time.time() + 300
            
            logger.info(f"Secret rotation initiated: {new_version_id} by {initiated_by}")
            return True, new_version_id
            
        except Exception as e:
            logger.error(f"Failed to initiate rotation: {e}")
            return False, ""
    
    async def complete_rotation(self, version_id: str) -> bool:
        """
        Complete rotation by making new secret active.
        
        Args:
            version_id: Version to make active
            
        Returns:
            Success status
        """
        try:
            # Get the rotating version
            version = self._secret_cache.get(version_id)
            if not version or version.state != RotationState.ROTATING:
                logger.error(f"Invalid version for completion: {version_id}")
                return False
            
            # Update states
            pipe = self.redis.pipeline()
            
            # Make new version active
            version.state = RotationState.ACTIVE
            version_key = f"{self.version_prefix}{version_id}".encode()
            version_data = {
                "version_id": version.version_id,
                "created_at": version.created_at,
                "state": RotationState.ACTIVE.value,
                "expires_at": version.expires_at,
                "created_by": version.created_by,
                "metadata": version.metadata
            }
            pipe.set(version_key, json.dumps(version_data))
            
            # Expire old versions
            for v in self._secret_cache.values():
                if v.version_id != version_id and v.state == RotationState.PREVIOUS:
                    # Set expiration for old versions
                    if not v.expires_at:
                        v.expires_at = (datetime.now() + self.rotation_overlap).isoformat()
                    
                    old_key = f"{self.version_prefix}{v.version_id}".encode()
                    old_data = {
                        "version_id": v.version_id,
                        "created_at": v.created_at,
                        "state": v.state.value,
                        "expires_at": v.expires_at,
                        "created_by": v.created_by,
                        "metadata": v.metadata
                    }
                    pipe.set(old_key, json.dumps(old_data))
            
            # Audit
            audit_key = f"{self.audit_prefix}{int(time.time())}".encode()
            audit_data = {
                "action": "rotation_completed",
                "version": version_id,
                "timestamp": datetime.now().isoformat()
            }
            pipe.setex(audit_key, 86400 * 365, json.dumps(audit_data))
            
            # Publish completion event
            pipe.publish("secret_rotation", json.dumps({
                "event": "rotation_completed",
                "active_version": version_id
            }))
            
            pipe.execute()
            
            # Update cache
            self._load_secrets()
            
            logger.info(f"Secret rotation completed: {version_id} is now active")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete rotation: {e}")
            return False
    
    async def rollback_rotation(self, version_id: str, reason: str) -> bool:
        """
        Rollback a rotation in progress.
        
        Args:
            version_id: Version to rollback
            reason: Reason for rollback
            
        Returns:
            Success status
        """
        try:
            version = self._secret_cache.get(version_id)
            if not version or version.state != RotationState.ROTATING:
                return False
            
            # Mark as expired
            version.state = RotationState.EXPIRED
            version.expires_at = datetime.now().isoformat()
            
            # Update in Redis
            version_key = f"{self.version_prefix}{version_id}".encode()
            self.redis.delete(version_key)
            
            # Remove secret
            secret_key = f"{self.secret_prefix}{version_id}".encode()
            self.redis.delete(secret_key)
            
            # Audit
            audit_key = f"{self.audit_prefix}{int(time.time())}".encode()
            audit_data = {
                "action": "rotation_rollback",
                "version": version_id,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
            self.redis.setex(audit_key, 86400 * 365, json.dumps(audit_data))
            
            # Remove from cache
            del self._secret_cache[version_id]
            
            logger.info(f"Rotation rolled back: {version_id} - {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback rotation: {e}")
            return False
    
    # ========================================================================
    # JWT OPERATIONS WITH ROTATION SUPPORT
    # ========================================================================
    
    def sign_jwt(self, payload: Dict[str, Any], algorithm: str = "HS256") -> str:
        """
        Sign JWT with current active secret.
        
        Args:
            payload: JWT payload
            algorithm: Signing algorithm
            
        Returns:
            Signed JWT token
        """
        secret = self.get_current_secret()
        
        # Add key version to payload for tracking
        current_version = None
        for v in self._secret_cache.values():
            if v.secret == secret:
                current_version = v.version_id
                break
        
        if current_version:
            payload["kid"] = current_version  # Key ID header
        
        return jwt.encode(payload, secret, algorithm=algorithm)
    
    def verify_jwt(
        self,
        token: str,
        algorithm: str = "HS256",
        **kwargs
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify JWT with support for multiple secret versions.
        
        Args:
            token: JWT token to verify
            algorithm: Expected algorithm
            **kwargs: Additional JWT validation parameters
            
        Returns:
            (valid, payload)
        """
        # Try all valid secrets
        secrets_to_try = self.get_all_valid_secrets()
        
        for secret in secrets_to_try:
            try:
                payload = jwt.decode(
                    token,
                    secret,
                    algorithms=[algorithm],
                    **kwargs
                )
                
                # Check if secret version is still valid
                kid = payload.get("kid")
                if kid and kid in self._secret_cache:
                    version = self._secret_cache[kid]
                    if not version.is_valid():
                        continue
                
                return True, payload
                
            except jwt.ExpiredSignatureError:
                # Token expired, not a secret issue
                return False, {"error": "token_expired"}
            except jwt.InvalidTokenError:
                # Try next secret
                continue
            except Exception as e:
                logger.error(f"JWT verification error: {e}")
                continue
        
        return False, {"error": "invalid_token"}
    
    # ========================================================================
    # MONITORING & MAINTENANCE
    # ========================================================================
    
    async def get_rotation_status(self) -> Dict[str, Any]:
        """Get current rotation status"""
        active_version = None
        rotating_version = None
        previous_versions = []
        
        for version in self._secret_cache.values():
            if version.state == RotationState.ACTIVE:
                active_version = version
            elif version.state == RotationState.ROTATING:
                rotating_version = version
            elif version.state == RotationState.PREVIOUS:
                previous_versions.append(version)
        
        return {
            "active": {
                "version_id": active_version.version_id if active_version else None,
                "created_at": active_version.created_at if active_version else None,
                "created_by": active_version.created_by if active_version else None
            },
            "rotating": {
                "version_id": rotating_version.version_id if rotating_version else None,
                "created_at": rotating_version.created_at if rotating_version else None,
                "progress": "in_progress" if rotating_version else "none"
            },
            "previous_count": len(previous_versions),
            "total_versions": len(self._secret_cache),
            "last_rotation": active_version.created_at if active_version else None
        }
    
    async def cleanup_expired(self) -> int:
        """Clean up expired secret versions"""
        cleaned = 0
        
        for version_id in list(self._secret_cache.keys()):
            version = self._secret_cache[version_id]
            
            if not version.is_valid() or version.state == RotationState.EXPIRED:
                # Remove from Redis
                version_key = f"{self.version_prefix}{version_id}".encode()
                secret_key = f"{self.secret_prefix}{version_id}".encode()
                
                self.redis.delete(version_key, secret_key)
                
                # Remove from cache
                del self._secret_cache[version_id]
                cleaned += 1
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired secret versions")
        
        return cleaned

# Global instance
secret_rotation = SecretRotationManager()

"""
Usage Example:

# Initiate rotation
success, new_version = await secret_rotation.initiate_rotation(
    initiated_by="admin@example.com",
    reason="quarterly_rotation"
)

# Complete rotation after testing
await secret_rotation.complete_rotation(new_version)

# Sign JWT with current secret
token = secret_rotation.sign_jwt({"user_id": "123", "role": "basic"})

# Verify JWT (works with any valid secret version)
valid, payload = secret_rotation.verify_jwt(token)
"""