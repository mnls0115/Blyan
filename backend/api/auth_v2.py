#!/usr/bin/env python3
"""
API Key V2 Endpoints for Main Service Node
Secure authentication system for GPU nodes and users
"""

import os
import json
import hmac
import hashlib
import time
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Header, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import structlog

# Import the existing API key system
from backend.auth.api_key_system import APIKeyManager, APIKeyRole

logger = structlog.get_logger()

# Initialize router
router = APIRouter(prefix="/api/auth/v2", tags=["authentication"])

# Security scheme
security = HTTPBearer()

# Environment variables
JWT_SECRET = os.environ.get("JWT_SECRET", "change_me_in_production")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
API_BOOTSTRAP_SECRET = os.environ.get("API_BOOTSTRAP_SECRET", "")
BLYAN_NODE_ENROLL_SECRET = os.environ.get("BLYAN_NODE_ENROLL_SECRET", "change_me_for_nodes")

# Config file for node registration
NODE_AUTH_CONFIG_PATH = Path("config/node_auth.json")

# Initialize API Key Manager
api_key_manager = None

def get_api_key_manager() -> APIKeyManager:
    """Get or create API key manager singleton."""
    global api_key_manager
    if api_key_manager is None:
        api_key_manager = APIKeyManager(
            jwt_secret=JWT_SECRET,
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            redis_db=REDIS_DB
        )
    return api_key_manager

def load_node_auth_config() -> Dict[str, Any]:
    """Load node authentication configuration."""
    if not NODE_AUTH_CONFIG_PATH.exists():
        # Create default config
        default_config = {
            "settings": {
                "allow_node_registration": False,
                "require_enrollment_secret": True,
                "max_nodes_per_ip": 5
            },
            "authorized_nodes": [],
            "blocked_ips": []
        }
        NODE_AUTH_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(NODE_AUTH_CONFIG_PATH, 'w') as f:
            json.dump(default_config, f, indent=2)
        return default_config
    
    with open(NODE_AUTH_CONFIG_PATH, 'r') as f:
        return json.load(f)

def save_node_auth_config(config: Dict[str, Any]):
    """Save node authentication configuration."""
    with open(NODE_AUTH_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)

def verify_enrollment_hmac(node_id: str, provided_hmac: str) -> bool:
    """Verify HMAC for node enrollment."""
    expected_hmac = hmac.new(
        BLYAN_NODE_ENROLL_SECRET.encode(),
        node_id.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected_hmac, provided_hmac)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Validate Bearer token and return user claims."""
    manager = get_api_key_manager()
    
    is_valid, claims = manager.validate_api_key(credentials.credentials)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return claims

async def require_admin(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Require admin role for endpoint."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

# Request/Response models
class RegisterRequest(BaseModel):
    wallet_address: str
    role: str = "basic"
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class NodeRegisterRequest(BaseModel):
    name: str
    node_id: str
    metadata: Optional[Dict[str, Any]] = None

class RefreshRequest(BaseModel):
    api_key: str

class RevokeRequest(BaseModel):
    jti: str  # JWT ID to revoke

class BootstrapRequest(BaseModel):
    description: Optional[str] = "Bootstrap admin key"

# Endpoints

@router.post("/register")
async def register_api_key(
    request: RegisterRequest,
    admin: Dict[str, Any] = Depends(require_admin)
):
    """Register a new API key (admin only)."""
    manager = get_api_key_manager()
    
    try:
        # Validate role
        if request.role not in ["basic", "contributor", "node_operator", "admin"]:
            raise HTTPException(status_code=400, detail="Invalid role")
        
        # Generate API key
        api_key, jti = manager.register_api_key(
            wallet_address=request.wallet_address,
            role=APIKeyRole(request.role),
            metadata=request.metadata
        )
        
        logger.info(
            "API key registered",
            wallet_address=request.wallet_address,
            role=request.role,
            jti=jti,
            admin_id=admin.get("jti")
        )
        
        return {
            "api_key": api_key,
            "jti": jti,
            "role": request.role,
            "expires_in": manager._get_ttl_for_role(APIKeyRole(request.role))
        }
        
    except Exception as e:
        logger.error(f"Failed to register API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to register API key")

@router.post("/node/register")
async def register_node_key(
    request: NodeRegisterRequest,
    x_node_enrollment: Optional[str] = Header(None),
    x_forwarded_for: Optional[str] = Header(None),
    request_obj: Request = None
):
    """Register a GPU node operator key (guarded endpoint)."""
    
    # Load config
    config = load_node_auth_config()
    settings = config.get("settings", {})
    
    # Check if node registration is allowed
    if not settings.get("allow_node_registration", False):
        logger.warning("Node registration attempted while disabled")
        raise HTTPException(status_code=403, detail="Node registration is currently disabled")
    
    # Verify enrollment HMAC if required
    if settings.get("require_enrollment_secret", True):
        if not x_node_enrollment:
            raise HTTPException(status_code=401, detail="Missing enrollment header")
        
        if not verify_enrollment_hmac(request.node_id, x_node_enrollment):
            logger.warning(f"Invalid enrollment HMAC for node {request.node_id}")
            raise HTTPException(status_code=401, detail="Invalid enrollment credentials")
    
    # Get client IP
    client_ip = x_forwarded_for or request_obj.client.host if request_obj else "unknown"
    
    # Check IP blocklist
    if client_ip in config.get("blocked_ips", []):
        logger.warning(f"Blocked IP attempted node registration: {client_ip}")
        raise HTTPException(status_code=403, detail="IP address is blocked")
    
    # Check max nodes per IP
    authorized_nodes = config.get("authorized_nodes", [])
    nodes_from_ip = [n for n in authorized_nodes if n.get("ip") == client_ip]
    max_per_ip = settings.get("max_nodes_per_ip", 5)
    
    if len(nodes_from_ip) >= max_per_ip:
        logger.warning(f"Max nodes per IP reached for {client_ip}")
        raise HTTPException(status_code=429, detail="Maximum nodes per IP reached")
    
    manager = get_api_key_manager()
    
    try:
        # Prepare metadata
        metadata = request.metadata or {}
        metadata.update({
            "node_id": request.node_id,
            "name": request.name,
            "ip": client_ip,
            "registered_at": datetime.utcnow().isoformat()
        })
        
        # Generate node operator key
        api_key, jti = manager.register_api_key(
            wallet_address=f"node_{request.node_id}",
            role=APIKeyRole.NODE_OPERATOR,
            metadata=metadata
        )
        
        # Add to authorized nodes
        authorized_nodes.append({
            "node_id": request.node_id,
            "name": request.name,
            "jti": jti,
            "ip": client_ip,
            "registered_at": datetime.utcnow().isoformat()
        })
        config["authorized_nodes"] = authorized_nodes
        save_node_auth_config(config)
        
        logger.info(
            "Node registered",
            node_id=request.node_id,
            name=request.name,
            ip=client_ip,
            jti=jti
        )
        
        return {
            "api_key": api_key,
            "jti": jti,
            "role": "node_operator",
            "node_id": request.node_id,
            "expires_in": 7776000  # 90 days for node operators
        }
        
    except Exception as e:
        logger.error(f"Failed to register node: {e}")
        raise HTTPException(status_code=500, detail="Failed to register node")

@router.get("/validate")
async def validate_api_key(user: Dict[str, Any] = Depends(get_current_user)):
    """Validate API key and return claims."""
    return {
        "valid": True,
        "claims": user,
        "expires_at": user.get("exp")
    }

@router.post("/refresh")
async def refresh_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Refresh API key within renewal window."""
    manager = get_api_key_manager()
    
    try:
        new_key, new_jti = manager.refresh_api_key(credentials.credentials)
        
        if not new_key:
            raise HTTPException(status_code=400, detail="Key not eligible for refresh")
        
        logger.info(f"API key refreshed: {new_jti}")
        
        return {
            "api_key": new_key,
            "jti": new_jti,
            "message": "Key refreshed successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to refresh key: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh key")

@router.post("/revoke")
async def revoke_api_key(
    request: RevokeRequest,
    admin: Dict[str, Any] = Depends(require_admin)
):
    """Revoke an API key by JTI (admin only)."""
    manager = get_api_key_manager()
    
    try:
        success = manager.revoke_api_key(request.jti)
        
        if not success:
            raise HTTPException(status_code=404, detail="Key not found or already revoked")
        
        logger.info(
            "API key revoked",
            revoked_jti=request.jti,
            admin_id=admin.get("jti")
        )
        
        return {
            "success": True,
            "message": f"Key {request.jti} revoked successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to revoke key: {e}")
        raise HTTPException(status_code=500, detail="Failed to revoke key")

@router.post("/bootstrap_admin")
async def bootstrap_admin(
    request: BootstrapRequest,
    x_bootstrap_secret: Optional[str] = Header(None)
):
    """Bootstrap first admin key (one-time use)."""
    
    if not API_BOOTSTRAP_SECRET:
        raise HTTPException(status_code=404, detail="Endpoint not available")
    
    if x_bootstrap_secret != API_BOOTSTRAP_SECRET:
        logger.warning("Invalid bootstrap secret attempted")
        raise HTTPException(status_code=401, detail="Invalid bootstrap secret")
    
    # Check if admin already exists
    config = load_node_auth_config()
    if config.get("bootstrap_used", False):
        raise HTTPException(status_code=403, detail="Bootstrap already used")
    
    manager = get_api_key_manager()
    
    try:
        # Generate admin key
        api_key, jti = manager.register_api_key(
            wallet_address="bootstrap_admin",
            role=APIKeyRole.ADMIN,
            metadata={"description": request.description}
        )
        
        # Mark bootstrap as used
        config["bootstrap_used"] = True
        config["bootstrap_admin_jti"] = jti
        save_node_auth_config(config)
        
        logger.info(f"Bootstrap admin created: {jti}")
        
        return {
            "api_key": api_key,
            "jti": jti,
            "role": "admin",
            "message": "Bootstrap admin created. This endpoint is now disabled."
        }
        
    except Exception as e:
        logger.error(f"Failed to bootstrap admin: {e}")
        raise HTTPException(status_code=500, detail="Failed to bootstrap admin")

@router.get("/status")
async def auth_status():
    """Get authentication system status (public)."""
    config = load_node_auth_config()
    manager = get_api_key_manager()
    
    return {
        "system": "API Key V2",
        "node_registration": config.get("settings", {}).get("allow_node_registration", False),
        "bootstrap_available": not config.get("bootstrap_used", False) and bool(API_BOOTSTRAP_SECRET),
        "authorized_nodes": len(config.get("authorized_nodes", [])),
        "redis_connected": manager.redis_client is not None if manager else False
    }

# Metrics endpoint (optional)
@router.get("/metrics")
async def auth_metrics(admin: Dict[str, Any] = Depends(require_admin)):
    """Get authentication metrics (admin only)."""
    manager = get_api_key_manager()
    
    return {
        "total_keys_issued": manager.metrics.get("keys_issued", 0),
        "total_validations": manager.metrics.get("validations", 0),
        "total_refreshes": manager.metrics.get("refreshes", 0),
        "total_revocations": manager.metrics.get("revocations", 0),
        "active_sessions": manager.metrics.get("active_sessions", 0)
    }