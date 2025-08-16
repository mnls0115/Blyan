#!/usr/bin/env python3
"""
API Authentication and Security Middleware
Enterprise-grade API protection with key management and security headers.
"""

import os
import hashlib
import secrets
import time
import json
import logging
from typing import Dict, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from fastapi import HTTPException, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import RedirectResponse, JSONResponse

logger = logging.getLogger(__name__)

@dataclass 
class APIKeyInfo:
    """API key information and metadata."""
    key_id: str
    key_hash: str  # SHA256 hash of the actual key
    name: str
    permissions: Set[str]  # Set of allowed endpoints/actions
    created_at: float
    last_used: float
    usage_count: int
    rate_limit_tier: str  # "basic", "premium", "enterprise"
    is_active: bool = True
    expires_at: Optional[float] = None


class APIKeyManager:
    """Secure API key management system."""
    
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or Path("./data/api_keys")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.keys_db: Dict[str, APIKeyInfo] = {}
        self.key_hash_to_id: Dict[str, str] = {}  # For fast lookups
        
        # Permission levels with sensitive endpoints restricted
        self.permission_sets = {
            "basic": {
                "/chat", "/chain/*/blocks", "/health", "/status", 
                "/auth/rate_limit_status"
            },
            "contributor": {
                "/upload_moe_experts", "/datasets/upload", "/auth/pol_challenge",
                "/experts/stats/*", "/experts/top", "/voting/create", "/voting/vote"
            },
            "node_operator": {
                "/p2p/register", "/p2p/heartbeat/*", "/p2p/nodes", 
                "/chat/distributed", "/chat/distributed_optimized", 
                "/chat/distributed_secure", "/p2p/register_optimized"
            },
            "financial": {
                "/economy/*", "/payment/*", "/wallet/*", "/ledger/*",
                "/rewards/*", "/balance/*"
            },
            "admin": {"*"}  # All endpoints
        }
        
        # Rate limits per tier (requests per minute)
        self.rate_limits = {
            "basic": 60,      # 1 req/sec
            "premium": 300,   # 5 req/sec
            "enterprise": 1000,  # ~16 req/sec
            "unlimited": float('inf')
        }
        
        self._load_keys_db()
        self._ensure_default_keys()
    
    def _load_keys_db(self):
        """Load API keys from persistent storage."""
        keys_file = self.storage_dir / "api_keys.json"
        if keys_file.exists():
            try:
                with open(keys_file) as f:
                    data = json.load(f)
                    for key_id, key_data in data.items():
                        key_info = APIKeyInfo(**key_data)
                        key_info.permissions = set(key_data["permissions"])
                        self.keys_db[key_id] = key_info
                        self.key_hash_to_id[key_info.key_hash] = key_id
            except Exception as e:
                print(f"Warning: Failed to load API keys: {e}")
    
    def _save_keys_db(self):
        """Save API keys to persistent storage."""
        keys_file = self.storage_dir / "api_keys.json"
        try:
            data = {}
            for key_id, key_info in self.keys_db.items():
                key_data = asdict(key_info)
                key_data["permissions"] = list(key_info.permissions)  # Convert set to list for JSON
                data[key_id] = key_data
            
            # Atomic write
            temp_file = keys_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(keys_file)
            
        except Exception as e:
            print(f"Warning: Failed to save API keys: {e}")
    
    def _ensure_default_keys(self):
        """Create default API keys if none exist."""
        if not self.keys_db:
            # Create default admin key for development
            admin_key = self.create_api_key(
                name="default_admin",
                permissions=self.permission_sets["admin"],
                rate_limit_tier="enterprise"
            )
            print(f"🔑 Created default admin API key: {admin_key}")
            print("⚠️  Save this key securely - it won't be shown again!")
    
    def create_api_key(self, name: str, permissions: Set[str], 
                      rate_limit_tier: str = "basic", expires_days: Optional[int] = None) -> str:
        """Create a new API key."""
        # Generate secure random key
        raw_key = secrets.token_urlsafe(32)  # 256-bit key
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_id = secrets.token_hex(8)  # 64-bit ID
        
        # Calculate expiration
        expires_at = None
        if expires_days:
            expires_at = time.time() + (expires_days * 86400)
        
        # Create key info
        key_info = APIKeyInfo(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            permissions=permissions,
            created_at=time.time(),
            last_used=0.0,
            usage_count=0,
            rate_limit_tier=rate_limit_tier,
            expires_at=expires_at
        )
        
        # Store in database
        self.keys_db[key_id] = key_info
        self.key_hash_to_id[key_hash] = key_id
        self._save_keys_db()
        
        print(f"✅ Created API key '{name}' with {len(permissions)} permissions")
        return raw_key
    
    def validate_api_key(self, raw_key: str) -> Optional[APIKeyInfo]:
        """Validate API key and return key info if valid."""
        if not raw_key:
            return None
        
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_id = self.key_hash_to_id.get(key_hash)
        
        if not key_id:
            return None
        
        key_info = self.keys_db.get(key_id)
        if not key_info or not key_info.is_active:
            return None
        
        # Check expiration
        if key_info.expires_at and time.time() > key_info.expires_at:
            key_info.is_active = False
            self._save_keys_db()
            return None
        
        # Update usage stats
        key_info.last_used = time.time()
        key_info.usage_count += 1
        
        return key_info
    
    def has_permission(self, key_info: APIKeyInfo, endpoint: str) -> bool:
        """Check if API key has permission for endpoint."""
        if "*" in key_info.permissions:
            return True
        
        # Direct match
        if endpoint in key_info.permissions:
            return True
        
        # Wildcard match
        for permission in key_info.permissions:
            if permission.endswith("*"):
                prefix = permission[:-1]
                if endpoint.startswith(prefix):
                    return True
        
        return False
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self.keys_db:
            self.keys_db[key_id].is_active = False
            self._save_keys_db()
            return True
        return False
    
    def list_keys(self) -> Dict[str, Dict]:
        """List all API keys (without sensitive data)."""
        return {
            key_id: {
                "name": key_info.name,
                "permissions_count": len(key_info.permissions),
                "rate_limit_tier": key_info.rate_limit_tier,
                "created_at": key_info.created_at,
                "last_used": key_info.last_used,
                "usage_count": key_info.usage_count,
                "is_active": key_info.is_active,
                "expires_at": key_info.expires_at
            }
            for key_id, key_info in self.keys_db.items()
        }


class ProductionSecurityMiddleware:
    """Production security middleware with HTTPS enforcement and security headers."""
    
    def __init__(self, api_key_manager: APIKeyManager):
        self.api_key_manager = api_key_manager
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # Security headers for all responses
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block", 
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "X-API-Version": "v2.0",
            "X-Powered-By": "Blyan-Blyanchain"
        }
        
        # Public endpoints that don't require authentication
        self.public_endpoints = {
            "/health", "/status", "/", "/docs", "/openapi.json", "/redoc",
            "/pol/status",  # PoL status check is public
            "/chain/A/blocks", "/chain/B/blocks",  # Read-only blockchain data  
            "/chain/*/blocks",  # All chain blocks are public
            "/experts/top",  # Top experts list is public
            "/leaderboard/*",  # All leaderboard endpoints are public
            "/metrics/*",  # Metrics are public
            "/auth/register_api_key",  # API key registration must be accessible
            # "/genesis/hash"  # Genesis verification should require auth in production
        }
        
        # Sensitive endpoints requiring enhanced security
        self.sensitive_endpoints = {
            "/wallet/", "/economy/", "/payment/", "/rewards/",
            "/voting/", "/admin/", "/keys/", "/security/"
        }
        
        # Rate limit tracking
        self.request_counts = {}  # {api_key_id: {minute: count}}
    
    async def __call__(self, request: Request, call_next):
        """Main middleware function with enhanced security."""
        # 0. Bypass authentication for health endpoint
        endpoint = request.url.path
        if endpoint == "/health":
            return await call_next(request)
        
        # 1. HTTPS enforcement in production
        if self.environment == "production" and request.url.scheme != "https":
            https_url = str(request.url).replace("http://", "https://", 1)
            return RedirectResponse(url=https_url, status_code=301)
        
        # 2. Check if endpoint requires authentication
        requires_auth = not any(
            endpoint == public or (public.endswith("*") and endpoint.startswith(public.rstrip("*")))
            for public in self.public_endpoints
        )
        
        # 3. Check if this is a sensitive endpoint
        is_sensitive = any(
            endpoint.startswith(sensitive)
            for sensitive in self.sensitive_endpoints
        )
        
        # 4. API key validation for protected endpoints
        if requires_auth:
            api_key = self._extract_api_key(request)
            if not api_key:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "API key required",
                        "message": "Add 'X-API-Key' header or use 'Authorization: Bearer' format",
                        "register_url": "/auth/register_api_key"
                    }
                )
            
            key_info = self.api_key_manager.validate_api_key(api_key)
            if not key_info:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Invalid or expired API key",
                        "register_url": "/auth/register_api_key"
                    }
                )
            
            # Check endpoint permissions
            if not self.api_key_manager.has_permission(key_info, endpoint):
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Insufficient permissions",
                        "required_endpoint": endpoint,
                        "your_permissions": list(key_info.permissions)[:5]  # Show first 5
                    }
                )
            
            # Enhanced validation for sensitive endpoints
            if is_sensitive:
                # Check for financial permissions on financial endpoints
                if endpoint.startswith(("/wallet/", "/economy/", "/payment/", "/rewards/")):
                    if "financial" not in key_info.permissions and "*" not in key_info.permissions:
                        return JSONResponse(
                            status_code=403,
                            content={"error": "Financial permissions required for this endpoint"}
                        )
                
                # Rate limiting for sensitive endpoints
                if not self._check_rate_limit(key_info, endpoint):
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Rate limit exceeded",
                            "rate_limit": self.rate_limits.get(key_info.rate_limit_tier, 60),
                            "retry_after": 60  # seconds
                        }
                    )
            
            # Add key info to request state for downstream use
            request.state.api_key_info = key_info
        
        # 4. Process request
        response = await call_next(request)
        
        # 5. Add security headers to response
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        # 6. Log request with masked API key
        api_key = locals().get('api_key', None)  # Get api_key if it was defined
        if api_key:
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
            logger.info(f"Request: {request.method} {endpoint} [key: {masked_key}] [status: {response.status_code}]")
        else:
            logger.info(f"Request: {request.method} {endpoint} [no key] [status: {response.status_code}]")
        
        return response
    
    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request headers."""
        # Try X-API-Key header first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key
        
        # Try Authorization Bearer format
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        
        return None
    
    def _check_rate_limit(self, key_info: APIKeyInfo, endpoint: str = "") -> bool:
        """Check if request is within rate limit."""
        current_minute = int(time.time() / 60)
        key_id = key_info.key_id
        
        # Special stricter limits for P2P and auth registration
        if endpoint.startswith("/p2p/") or endpoint == "/auth/register_api_key":
            # 5 requests per minute for these endpoints
            rate_limit = 5
        else:
            rate_limit = self.rate_limits.get(key_info.rate_limit_tier, 60)
        
        # Initialize tracking for this key
        if key_id not in self.request_counts:
            self.request_counts[key_id] = {}
        
        # Clean old entries (keep only last 2 minutes)
        self.request_counts[key_id] = {
            minute: count 
            for minute, count in self.request_counts[key_id].items()
            if minute >= current_minute - 1
        }
        
        # Count requests in current minute
        current_count = self.request_counts[key_id].get(current_minute, 0)
        
        if current_count >= rate_limit:
            return False
        
        # Increment counter
        self.request_counts[key_id][current_minute] = current_count + 1
        return True


# Global instances
api_key_manager = APIKeyManager()
security_middleware = ProductionSecurityMiddleware(api_key_manager)


class APIKeyGenerator:
    """Utility for generating API keys with different permission levels."""
    
    @staticmethod
    def create_basic_user_key(name: str) -> str:
        """Create basic user API key."""
        return api_key_manager.create_api_key(
            name=name,
            permissions=api_key_manager.permission_sets["basic"],
            rate_limit_tier="basic"
        )
    
    @staticmethod  
    def create_contributor_key(name: str) -> str:
        """Create contributor API key."""
        permissions = (
            api_key_manager.permission_sets["basic"] | 
            api_key_manager.permission_sets["contributor"]
        )
        return api_key_manager.create_api_key(
            name=name,
            permissions=permissions,
            rate_limit_tier="premium"
        )
    
    @staticmethod
    def create_node_operator_key(name: str) -> str:
        """Create node operator API key."""
        permissions = (
            api_key_manager.permission_sets["basic"] |
            api_key_manager.permission_sets["contributor"] |
            api_key_manager.permission_sets["node_operator"]
        )
        return api_key_manager.create_api_key(
            name=name,
            permissions=permissions,
            rate_limit_tier="enterprise"
        )
    
    @staticmethod
    def create_financial_key(name: str) -> str:
        """Create financial operations API key."""
        permissions = (
            api_key_manager.permission_sets["basic"] |
            api_key_manager.permission_sets["financial"]
        )
        return api_key_manager.create_api_key(
            name=name,
            permissions=permissions,
            rate_limit_tier="premium"
        )
    
    @staticmethod
    def create_admin_key(name: str) -> str:
        """Create admin API key."""
        return api_key_manager.create_api_key(
            name=name,
            permissions=api_key_manager.permission_sets["admin"],
            rate_limit_tier="enterprise"
        )


def get_api_key_info(request: Request) -> Optional[APIKeyInfo]:
    """Get API key info from request state (if authenticated)."""
    return getattr(request.state, "api_key_info", None)


def require_permission(permission: str):
    """Decorator to require specific permission for endpoint."""
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            key_info = get_api_key_info(request)
            if not key_info or not api_key_manager.has_permission(key_info, permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission required: {permission}"
                )
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator