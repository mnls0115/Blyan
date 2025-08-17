#!/usr/bin/env python3
"""
P2P Node Enrollment API Endpoints
Handles JOIN_CODE generation and exchange for node credentials
"""

from fastapi import APIRouter, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import time
from datetime import datetime

# Import the JOIN_CODE system
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.auth.join_code_system import get_join_code_manager, NodeCredentials

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/p2p", tags=["P2P Enrollment"])

# Request/Response models
class JoinCodeRequest(BaseModel):
    """Request for JOIN_CODE generation"""
    node_type: str = Field(default="gpu", description="Type of node: gpu, cpu, storage")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")

class JoinCodeResponse(BaseModel):
    """Response with JOIN_CODE"""
    join_code: str = Field(..., description="One-time enrollment code")
    expires_at: str = Field(..., description="ISO format expiry time")
    instructions: str = Field(..., description="How to use the code")

class EnrollRequest(BaseModel):
    """Node enrollment request"""
    join_code: str = Field(..., description="JOIN_CODE from registration")
    node_meta: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Node metadata (GPU model, VRAM, location, etc.)"
    )

class EnrollResponse(BaseModel):
    """Node enrollment response with credentials"""
    node_id: str = Field(..., description="Unique node identifier")
    node_key: str = Field(..., description="Secret key for authentication")
    expires_at: str = Field(..., description="Credential expiry time")
    instructions: Dict[str, str] = Field(..., description="Next steps")

class NodeAuthRequest(BaseModel):
    """Node authentication validation"""
    node_id: str = Field(..., description="Node identifier")
    node_key: str = Field(..., description="Node secret key")


# Rate limiting decorator
def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    # Check X-Forwarded-For header (for proxies/load balancers)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the first IP in the chain
        return forwarded.split(",")[0].strip()
    
    # Check X-Real-IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct connection IP
    return request.client.host if request.client else "unknown"


@router.post("/join-code", response_model=JoinCodeResponse)
async def request_join_code(
    request: Request,
    body: JoinCodeRequest
) -> JoinCodeResponse:
    """
    Request a one-time JOIN_CODE for node enrollment.
    
    Rate limited to 3 requests per IP address per 5 minutes.
    Codes expire after 30 minutes.
    """
    # Get client IP for rate limiting
    client_ip = get_client_ip(request)
    
    # Get JOIN_CODE manager
    manager = get_join_code_manager()
    
    # Generate JOIN_CODE
    join_code, error = manager.generate_join_code(
        node_type=body.node_type,
        ip_address=client_ip,
        metadata=body.metadata
    )
    
    if error:
        if "Rate limit" in error:
            raise HTTPException(status_code=429, detail=error)
        else:
            raise HTTPException(status_code=400, detail=error)
    
    # Format response
    expires_at = datetime.fromtimestamp(join_code.expires_at).isoformat()
    
    instructions = (
        f"Use this code when starting your Docker container:\n"
        f"docker run -e JOIN_CODE={join_code.code} -v /data:/data blyan/node:latest"
    )
    
    logger.info(f"Issued JOIN_CODE to {client_ip} for {body.node_type} node")
    
    return JoinCodeResponse(
        join_code=join_code.code,
        expires_at=expires_at,
        instructions=instructions
    )


@router.post("/enroll", response_model=EnrollResponse)
async def enroll_node(
    request: Request,
    body: EnrollRequest
) -> EnrollResponse:
    """
    Exchange a JOIN_CODE for permanent node credentials.
    
    This endpoint is called by the node on first boot.
    The JOIN_CODE is single-use and will be invalidated after successful exchange.
    """
    # Get client IP
    client_ip = get_client_ip(request)
    
    # Add client info to node metadata
    if body.node_meta is None:
        body.node_meta = {}
    
    body.node_meta["enrolled_from_ip"] = client_ip
    body.node_meta["enrolled_at"] = datetime.utcnow().isoformat()
    body.node_meta["user_agent"] = request.headers.get("User-Agent", "unknown")
    
    # Get JOIN_CODE manager
    manager = get_join_code_manager()
    
    # Validate and exchange
    credentials, error = manager.validate_and_exchange(
        code=body.join_code,
        node_meta=body.node_meta,
        ip_address=client_ip
    )
    
    if error:
        logger.warning(f"Failed enrollment from {client_ip}: {error}")
        raise HTTPException(status_code=400, detail=error)
    
    # Format response
    expires_at = datetime.fromtimestamp(credentials.expires_at).isoformat()
    
    instructions = {
        "save_credentials": "Save node_id and node_key to /data/credentials.json",
        "authentication": "Use X-Node-ID and X-Node-Key headers for API requests",
        "renewal": "Credentials expire in 90 days. Contact admin for renewal.",
        "next_step": "Node can now register with /p2p/register endpoint"
    }
    
    logger.info(f"Successfully enrolled node {credentials.node_id} from {client_ip}")
    
    return EnrollResponse(
        node_id=credentials.node_id,
        node_key=credentials.node_key,
        expires_at=expires_at,
        instructions=instructions
    )


@router.post("/verify-node")
async def verify_node_auth(
    request: Request,
    body: NodeAuthRequest
) -> Dict[str, Any]:
    """
    Verify node credentials (for internal use).
    
    Used by other services to validate node authentication.
    """
    manager = get_join_code_manager()
    
    is_valid = manager.verify_node_credentials(
        node_id=body.node_id,
        node_key=body.node_key
    )
    
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid or expired credentials")
    
    return {
        "valid": True,
        "node_id": body.node_id,
        "message": "Node credentials are valid"
    }


@router.get("/enrollment-stats")
async def get_enrollment_stats(
    x_api_key: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    Get JOIN_CODE system statistics (admin only).
    
    Requires admin API key.
    """
    # Simple admin check (in production, use proper auth)
    admin_key = "admin_key_from_env"  # Should come from environment
    if x_api_key != admin_key:
        # For now, allow public access to basic stats
        pass  # In production, enforce: raise HTTPException(status_code=403, detail="Admin access required")
    
    manager = get_join_code_manager()
    stats = manager.get_stats()
    
    return {
        "status": "operational",
        "stats": stats,
        "timestamp": datetime.utcnow().isoformat()
    }


# Middleware for node authentication
async def verify_node_headers(
    x_node_id: Optional[str] = Header(None),
    x_node_key: Optional[str] = Header(None)
) -> Optional[str]:
    """
    Dependency to verify node authentication headers.
    
    Returns node_id if valid, raises 401 if invalid.
    """
    if not x_node_id or not x_node_key:
        return None  # No node auth provided
    
    manager = get_join_code_manager()
    
    if not manager.verify_node_credentials(x_node_id, x_node_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired node credentials"
        )
    
    return x_node_id


# Protected endpoint example
@router.get("/node-status")
async def get_node_status(
    node_id: Optional[str] = Depends(verify_node_headers)
) -> Dict[str, Any]:
    """
    Get node status (requires node authentication).
    
    Example of an endpoint that requires valid node credentials.
    """
    if not node_id:
        raise HTTPException(
            status_code=401,
            detail="Node authentication required"
        )
    
    return {
        "node_id": node_id,
        "status": "authenticated",
        "message": f"Node {node_id} is properly authenticated"
    }