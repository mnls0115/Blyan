"""
Production-Grade API Key Management System
==========================================
Author: Senior Backend Engineering Team
Version: 2.0.0

This module provides a robust, scalable API key management system with:
- JWT-based signed tokens
- Role-based access control (RBAC)
- Automatic key rotation and refresh
- Distributed cache synchronization
- Comprehensive audit logging
"""

import os
import jwt
import hashlib
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Set, Tuple, Any
from enum import Enum
from dataclasses import dataclass, asdict
import asyncio
import redis
from fastapi import HTTPException, Request, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import logging
from functools import wraps

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class APIKeyConfig:
    """Centralized configuration for API key system"""
    
    # JWT Configuration
    JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
    JWT_ALGORITHM = "HS256"
    JWT_ISSUER = "blyan-api"
    
    # Key Lifetimes (in seconds)
    TTL_BASIC = 3600 * 24 * 7          # 7 days
    TTL_CONTRIBUTOR = 3600 * 24 * 30    # 30 days
    TTL_NODE_OPERATOR = 3600 * 24 * 90  # 90 days
    TTL_ADMIN = 3600 * 24 * 365         # 1 year
    
    # Refresh Windows (seconds before expiry)
    REFRESH_WINDOW = 3600 * 24          # 24 hours before expiry
    
    # Rate Limits (requests per hour)
    RATE_LIMITS = {
        "basic": 100,
        "contributor": 1000,
        "node_operator": 10000,
        "admin": -1  # Unlimited
    }
    
    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    REDIS_KEY_PREFIX = "api_key:"
    
    # Security
    HASH_ITERATIONS = 100_000
    SALT_LENGTH = 32

# ============================================================================
# ENUMS AND DATA MODELS
# ============================================================================

class APIKeyRole(str, Enum):
    """Standardized role definitions - single source of truth"""
    BASIC = "basic"
    CONTRIBUTOR = "contributor"
    NODE_OPERATOR = "node_operator"
    ADMIN = "admin"
    
    @classmethod
    def from_string(cls, value: str) -> Optional['APIKeyRole']:
        """Safe string to enum conversion with normalization"""
        normalized = value.lower().replace("-", "_")
        for role in cls:
            if role.value == normalized:
                return role
        return None

class APIKeyScope(str, Enum):
    """Granular permission scopes"""
    CHAT_READ = "chat:read"
    CHAT_WRITE = "chat:write"
    CHAT_HISTORY = "chat:history"
    DATASET_READ = "dataset:read"
    DATASET_WRITE = "dataset:write"
    NODE_REGISTER = "node:register"
    NODE_MANAGE = "node:manage"
    ADMIN_READ = "admin:read"
    ADMIN_WRITE = "admin:write"
    SYSTEM_MANAGE = "system:manage"

# Role-to-Scope Mapping (Permission Matrix)
ROLE_PERMISSIONS: Dict[APIKeyRole, Set[APIKeyScope]] = {
    APIKeyRole.BASIC: {
        APIKeyScope.CHAT_READ,
        APIKeyScope.CHAT_WRITE,
    },
    APIKeyRole.CONTRIBUTOR: {
        APIKeyScope.CHAT_READ,
        APIKeyScope.CHAT_WRITE,
        APIKeyScope.CHAT_HISTORY,
        APIKeyScope.DATASET_READ,
        APIKeyScope.DATASET_WRITE,
    },
    APIKeyRole.NODE_OPERATOR: {
        APIKeyScope.CHAT_READ,
        APIKeyScope.CHAT_WRITE,
        APIKeyScope.CHAT_HISTORY,
        APIKeyScope.NODE_REGISTER,
        APIKeyScope.NODE_MANAGE,
    },
    APIKeyRole.ADMIN: {
        # Admins have all scopes
        scope for scope in APIKeyScope
    }
}

# Who can create which key types
KEY_CREATION_PERMISSIONS: Dict[APIKeyRole, Set[APIKeyRole]] = {
    APIKeyRole.BASIC: {APIKeyRole.BASIC},  # Basic users can only create basic keys
    APIKeyRole.CONTRIBUTOR: {APIKeyRole.BASIC},
    APIKeyRole.NODE_OPERATOR: {APIKeyRole.BASIC},
    APIKeyRole.ADMIN: {APIKeyRole.BASIC, APIKeyRole.CONTRIBUTOR, APIKeyRole.NODE_OPERATOR, APIKeyRole.ADMIN}
}

@dataclass
class APIKeyPayload:
    """JWT payload structure for API keys"""
    sub: str           # Subject (user ID or key ID)
    role: str          # Role string
    scopes: List[str]  # List of permission scopes
    iat: int           # Issued at timestamp
    exp: int           # Expiration timestamp
    iss: str           # Issuer
    jti: str           # JWT ID for revocation tracking
    metadata: Dict[str, Any] = None  # Optional metadata

    def to_dict(self) -> dict:
        """Convert to dictionary for JWT encoding"""
        data = asdict(self)
        if data.get('metadata') is None:
            del data['metadata']
        return data

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class RegisterAPIKeyRequest(BaseModel):
    """Request model for API key registration"""
    name: str = Field(..., min_length=3, max_length=100, description="Key name/identifier")
    key_type: str = Field(..., description="Role type: basic, contributor, node_operator, admin")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Optional metadata")
    
    @validator('key_type')
    def validate_key_type(cls, v):
        role = APIKeyRole.from_string(v)
        if role is None:
            raise ValueError(f"Invalid key_type: {v}. Must be one of: {[r.value for r in APIKeyRole]}")
        return role.value

class RegisterAPIKeyResponse(BaseModel):
    """Response model for API key registration"""
    api_key: str
    key_id: str
    role: str
    scopes: List[str]
    expires_at: datetime
    refresh_after: datetime
    metadata: Dict[str, Any]

class RefreshAPIKeyRequest(BaseModel):
    """Request model for API key refresh"""
    current_key: str = Field(..., description="Current API key to refresh")

class APIErrorResponse(BaseModel):
    """Standardized error response"""
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    request_id: Optional[str] = Field(default=None, description="Request tracking ID")

# ============================================================================
# CORE API KEY MANAGER
# ============================================================================

class APIKeyManager:
    """
    Production-grade API key manager with distributed cache support
    """
    
    def __init__(self):
        self.config = APIKeyConfig()
        self.redis_client = self._init_redis()
        self._signing_key_cache = {}
        
    def _init_redis(self) -> redis.Redis:
        """Initialize Redis connection with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = redis.Redis(
                    host=self.config.REDIS_HOST,
                    port=self.config.REDIS_PORT,
                    db=self.config.REDIS_DB,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                client.ping()
                logger.info(f"Redis connection established: {self.config.REDIS_HOST}:{self.config.REDIS_PORT}")
                return client
            except redis.ConnectionError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to connect to Redis after {max_retries} attempts")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
    def _hash_key(self, api_key: str) -> str:
        """Securely hash API key for storage and logging"""
        salt = self.config.JWT_SECRET.encode()[:self.config.SALT_LENGTH]
        return hashlib.pbkdf2_hmac(
            'sha256',
            api_key.encode(),
            salt,
            self.config.HASH_ITERATIONS
        ).hex()
    
    def _get_ttl_for_role(self, role: APIKeyRole) -> int:
        """Get TTL based on role"""
        ttl_map = {
            APIKeyRole.BASIC: self.config.TTL_BASIC,
            APIKeyRole.CONTRIBUTOR: self.config.TTL_CONTRIBUTOR,
            APIKeyRole.NODE_OPERATOR: self.config.TTL_NODE_OPERATOR,
            APIKeyRole.ADMIN: self.config.TTL_ADMIN,
        }
        return ttl_map.get(role, self.config.TTL_BASIC)
    
    async def register_api_key(
        self,
        requester_role: Optional[APIKeyRole],
        request: RegisterAPIKeyRequest
    ) -> RegisterAPIKeyResponse:
        """
        Register a new API key with comprehensive validation
        """
        # Parse requested role
        requested_role = APIKeyRole.from_string(request.key_type)
        if requested_role is None:
            raise HTTPException(
                status_code=400,
                detail=APIErrorResponse(
                    error="invalid_role",
                    message=f"Invalid role: {request.key_type}",
                    details={"valid_roles": [r.value for r in APIKeyRole]}
                ).dict()
            )
        
        # Check creation permissions
        if requester_role:
            allowed_roles = KEY_CREATION_PERMISSIONS.get(requester_role, set())
            if requested_role not in allowed_roles:
                raise HTTPException(
                    status_code=403,
                    detail=APIErrorResponse(
                        error="insufficient_permissions",
                        message=f"Role {requester_role.value} cannot create {requested_role.value} keys",
                        details={
                            "your_role": requester_role.value,
                            "requested_role": requested_role.value,
                            "allowed_roles": [r.value for r in allowed_roles]
                        }
                    ).dict()
                )
        
        # Generate key components
        key_id = f"{requested_role.value}_{int(time.time())}_{secrets.token_hex(8)}"
        jti = secrets.token_urlsafe(16)
        
        # Calculate timestamps
        now = datetime.now(timezone.utc)
        ttl = self._get_ttl_for_role(requested_role)
        expires_at = now + timedelta(seconds=ttl)
        refresh_after = expires_at - timedelta(seconds=self.config.REFRESH_WINDOW)
        
        # Get scopes for role
        scopes = [s.value for s in ROLE_PERMISSIONS[requested_role]]
        
        # Create JWT payload
        payload = APIKeyPayload(
            sub=key_id,
            role=requested_role.value,
            scopes=scopes,
            iat=int(now.timestamp()),
            exp=int(expires_at.timestamp()),
            iss=self.config.JWT_ISSUER,
            jti=jti,
            metadata=request.metadata
        )
        
        # Generate JWT token
        api_key = jwt.encode(
            payload.to_dict(),
            self.config.JWT_SECRET,
            algorithm=self.config.JWT_ALGORITHM
        )
        
        # Store in Redis with TTL
        redis_key = f"{self.config.REDIS_KEY_PREFIX}{jti}"
        redis_value = {
            "key_id": key_id,
            "role": requested_role.value,
            "scopes": scopes,
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "metadata": request.metadata,
            "hashed": self._hash_key(api_key)
        }
        
        self.redis_client.setex(
            redis_key,
            ttl,
            json.dumps(redis_value)
        )
        
        # Log key creation (hash only)
        logger.info(f"API key created: role={requested_role.value}, key_id={key_id}, hash={self._hash_key(api_key)[:16]}...")
        
        return RegisterAPIKeyResponse(
            api_key=api_key,
            key_id=key_id,
            role=requested_role.value,
            scopes=scopes,
            expires_at=expires_at,
            refresh_after=refresh_after,
            metadata=request.metadata
        )
    
    async def validate_api_key(self, api_key: str) -> Tuple[bool, Optional[APIKeyPayload], Optional[str]]:
        """
        Validate API key and return (is_valid, payload, error_message)
        """
        try:
            # Decode JWT
            payload_dict = jwt.decode(
                api_key,
                self.config.JWT_SECRET,
                algorithms=[self.config.JWT_ALGORITHM],
                issuer=self.config.JWT_ISSUER
            )
            
            # Check if key is revoked (check Redis)
            jti = payload_dict.get('jti')
            if jti:
                redis_key = f"{self.config.REDIS_KEY_PREFIX}{jti}"
                if not self.redis_client.exists(redis_key):
                    return False, None, "Key has been revoked"
            
            # Convert to payload object
            payload = APIKeyPayload(**payload_dict)
            
            # Check expiration
            if payload.exp < time.time():
                return False, None, "Key has expired"
            
            return True, payload, None
            
        except jwt.ExpiredSignatureError:
            return False, None, "Key has expired"
        except jwt.InvalidTokenError as e:
            return False, None, f"Invalid key: {str(e)}"
        except Exception as e:
            logger.error(f"Key validation error: {str(e)}")
            return False, None, "Key validation failed"
    
    async def refresh_api_key(self, current_key: str) -> RegisterAPIKeyResponse:
        """
        Refresh an existing API key if within refresh window
        """
        # Validate current key
        is_valid, payload, error = await self.validate_api_key(current_key)
        
        if not is_valid:
            raise HTTPException(
                status_code=401,
                detail=APIErrorResponse(
                    error="invalid_key",
                    message=error or "Current key is invalid"
                ).dict()
            )
        
        # Check if within refresh window
        time_until_expiry = payload.exp - time.time()
        if time_until_expiry > self.config.REFRESH_WINDOW:
            raise HTTPException(
                status_code=400,
                detail=APIErrorResponse(
                    error="too_early_to_refresh",
                    message=f"Key can only be refreshed within {self.config.REFRESH_WINDOW/3600} hours of expiry",
                    details={
                        "expires_in_seconds": int(time_until_expiry),
                        "refresh_window_seconds": self.config.REFRESH_WINDOW
                    }
                ).dict()
            )
        
        # Revoke old key
        if payload.jti:
            old_redis_key = f"{self.config.REDIS_KEY_PREFIX}{payload.jti}"
            self.redis_client.delete(old_redis_key)
        
        # Create new key with same role and metadata
        role = APIKeyRole.from_string(payload.role)
        request = RegisterAPIKeyRequest(
            name=f"refreshed_{payload.sub}",
            key_type=payload.role,
            metadata=payload.metadata or {}
        )
        
        return await self.register_api_key(role, request)
    
    async def revoke_api_key(self, api_key: str):
        """
        Revoke an API key immediately
        """
        is_valid, payload, _ = await self.validate_api_key(api_key)
        
        if is_valid and payload.jti:
            redis_key = f"{self.config.REDIS_KEY_PREFIX}{payload.jti}"
            self.redis_client.delete(redis_key)
            logger.info(f"API key revoked: jti={payload.jti}")

# ============================================================================
# FASTAPI DEPENDENCIES AND MIDDLEWARE
# ============================================================================

# Global manager instance
api_key_manager = APIKeyManager()

# Security scheme
security = HTTPBearer()

async def get_current_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> APIKeyPayload:
    """
    FastAPI dependency to extract and validate API key from headers
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail=APIErrorResponse(
                error="missing_credentials",
                message="Authorization header missing"
            ).dict()
        )
    
    is_valid, payload, error = await api_key_manager.validate_api_key(credentials.credentials)
    
    if not is_valid:
        raise HTTPException(
            status_code=401,
            detail=APIErrorResponse(
                error="invalid_credentials",
                message=error or "Invalid API key"
            ).dict()
        )
    
    return payload

async def get_optional_api_key(
    request: Request
) -> Optional[APIKeyPayload]:
    """
    FastAPI dependency for optional authentication (free tier support)
    """
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    token = auth_header.replace("Bearer ", "")
    is_valid, payload, _ = await api_key_manager.validate_api_key(token)
    
    return payload if is_valid else None

def require_scopes(*required_scopes: APIKeyScope):
    """
    Decorator factory for scope-based authorization
    """
    async def verify_scopes(api_key: APIKeyPayload = Depends(get_current_api_key)):
        user_scopes = set(api_key.scopes)
        required = set(s.value for s in required_scopes)
        
        if not required.issubset(user_scopes):
            missing = required - user_scopes
            raise HTTPException(
                status_code=403,
                detail=APIErrorResponse(
                    error="insufficient_scopes",
                    message="Your API key lacks required permissions",
                    details={
                        "required_scopes": list(required),
                        "your_scopes": list(user_scopes),
                        "missing_scopes": list(missing)
                    }
                ).dict()
            )
        
        return api_key
    
    return verify_scopes

def require_role(minimum_role: APIKeyRole):
    """
    Decorator factory for role-based authorization
    """
    async def verify_role(api_key: APIKeyPayload = Depends(get_current_api_key)):
        user_role = APIKeyRole.from_string(api_key.role)
        
        # Define role hierarchy
        role_hierarchy = {
            APIKeyRole.BASIC: 0,
            APIKeyRole.CONTRIBUTOR: 1,
            APIKeyRole.NODE_OPERATOR: 1,  # Same level as contributor
            APIKeyRole.ADMIN: 2
        }
        
        if role_hierarchy.get(user_role, 0) < role_hierarchy.get(minimum_role, 0):
            raise HTTPException(
                status_code=403,
                detail=APIErrorResponse(
                    error="insufficient_role",
                    message=f"This endpoint requires at least {minimum_role.value} role",
                    details={
                        "your_role": api_key.role,
                        "required_role": minimum_role.value
                    }
                ).dict()
            )
        
        return api_key
    
    return verify_role

# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """
    Token bucket rate limiter with Redis backend
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.config = APIKeyConfig()
    
    async def check_rate_limit(self, api_key: APIKeyPayload) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limit
        Returns (is_allowed, rate_limit_info)
        """
        role = APIKeyRole.from_string(api_key.role)
        limit = self.config.RATE_LIMITS.get(role.value, 100)
        
        # Unlimited for admin
        if limit == -1:
            return True, {"limit": "unlimited", "remaining": "unlimited"}
        
        # Redis key for rate limiting
        bucket_key = f"rate_limit:{api_key.sub}:{int(time.time() // 3600)}"
        
        # Increment counter
        pipe = self.redis.pipeline()
        pipe.incr(bucket_key)
        pipe.expire(bucket_key, 3600)
        results = pipe.execute()
        
        current_count = results[0]
        
        if current_count > limit:
            return False, {
                "limit": limit,
                "remaining": 0,
                "reset_in": 3600 - (int(time.time()) % 3600)
            }
        
        return True, {
            "limit": limit,
            "remaining": limit - current_count,
            "reset_in": 3600 - (int(time.time()) % 3600)
        }

# Rate limiter instance
rate_limiter = RateLimiter(api_key_manager.redis_client)

async def enforce_rate_limit(api_key: APIKeyPayload = Depends(get_current_api_key)):
    """
    FastAPI dependency for rate limiting
    """
    is_allowed, info = await rate_limiter.check_rate_limit(api_key)
    
    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail=APIErrorResponse(
                error="rate_limit_exceeded",
                message="API rate limit exceeded",
                details=info
            ).dict(),
            headers={
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": str(info["remaining"]),
                "X-RateLimit-Reset": str(info["reset_in"])
            }
        )
    
    return api_key

# ============================================================================
# MONITORING AND AUDIT
# ============================================================================

class APIKeyAuditor:
    """
    Audit logging for API key operations
    """
    
    @staticmethod
    def log_key_usage(
        api_key: APIKeyPayload,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float
    ):
        """Log API key usage for audit and analytics"""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "key_id": api_key.sub,
            "role": api_key.role,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time_ms": response_time_ms
        }
        
        # In production, send to centralized logging (ELK, Datadog, etc.)
        logger.info(f"API_AUDIT: {json.dumps(audit_entry)}")
    
    @staticmethod
    async def check_suspicious_activity(api_key: APIKeyPayload) -> bool:
        """
        Check for suspicious activity patterns
        Returns True if suspicious activity detected
        """
        # Implement anomaly detection logic here
        # - Sudden spike in requests
        # - Unusual geographic location
        # - Accessing unusual endpoints
        # - Rapid key rotation attempts
        
        return False

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
Example FastAPI endpoint implementation:

from fastapi import FastAPI, Depends
from backend.auth.api_key_system import (
    APIKeyRole, APIKeyScope, APIKeyPayload,
    RegisterAPIKeyRequest, RegisterAPIKeyResponse,
    get_current_api_key, get_optional_api_key,
    require_scopes, require_role, enforce_rate_limit,
    api_key_manager
)

app = FastAPI()

# Public endpoint - no auth required
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Optional auth - works for both free and authenticated users
@app.post("/chat")
async def chat(
    request: ChatRequest,
    api_key: Optional[APIKeyPayload] = Depends(get_optional_api_key)
):
    if api_key:
        # Authenticated user - full features
        return await process_authenticated_chat(request, api_key)
    else:
        # Free tier - limited features
        return await process_free_tier_chat(request)

# Requires authentication and specific scope
@app.get("/chat/history")
async def get_chat_history(
    api_key: APIKeyPayload = Depends(require_scopes(APIKeyScope.CHAT_HISTORY))
):
    return await fetch_chat_history(api_key.sub)

# Requires specific role
@app.post("/datasets/upload")
async def upload_dataset(
    request: DatasetRequest,
    api_key: APIKeyPayload = Depends(require_role(APIKeyRole.CONTRIBUTOR))
):
    return await process_dataset_upload(request, api_key)

# Admin only with rate limiting
@app.get("/admin/users")
async def list_users(
    api_key: APIKeyPayload = Depends(require_role(APIKeyRole.ADMIN)),
    _: Any = Depends(enforce_rate_limit)
):
    return await fetch_all_users()

# API key management endpoints
@app.post("/auth/register", response_model=RegisterAPIKeyResponse)
async def register_api_key(
    request: RegisterAPIKeyRequest,
    current_key: Optional[APIKeyPayload] = Depends(get_optional_api_key)
):
    requester_role = APIKeyRole.from_string(current_key.role) if current_key else None
    return await api_key_manager.register_api_key(requester_role, request)

@app.post("/auth/refresh", response_model=RegisterAPIKeyResponse)
async def refresh_api_key(
    request: RefreshAPIKeyRequest
):
    return await api_key_manager.refresh_api_key(request.current_key)
"""

import json  # Add this import at the top