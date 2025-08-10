"""
FastAPI Dependencies for Node Authentication
Provides actual enforcement of node authentication
"""

import os
from typing import Optional
from fastapi import HTTPException, Header, Depends
from backend.api.node_auth import NodeAuthenticator

# Initialize authenticator once
_authenticator = NodeAuthenticator()

async def verify_main_node(
    x_node_id: Optional[str] = Header(None, alias="X-Node-ID"),
    x_node_auth_token: Optional[str] = Header(None, alias="X-Node-Auth-Token")
) -> str:
    """
    FastAPI dependency to verify main node authentication
    Use this for endpoints that should only be accessible to the main node
    """
    if not x_node_id or not x_node_auth_token:
        # Log attempt
        _authenticator._log_security_event({
            "type": "missing_auth_headers",
            "endpoint": "protected",
            "timestamp": datetime.utcnow().isoformat()
        })
        raise HTTPException(
            status_code=401,
            detail="Missing authentication headers"
        )
    
    if not _authenticator.verify_main_node(x_node_id, x_node_auth_token):
        # Log unauthorized attempt
        _authenticator._log_security_event({
            "type": "unauthorized_main_node_access",
            "node_id": x_node_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        raise HTTPException(
            status_code=403,
            detail="Unauthorized: Invalid main node credentials"
        )
    
    return x_node_id

async def verify_any_authorized_node(
    x_node_id: Optional[str] = Header(None, alias="X-Node-ID"),
    x_node_auth_token: Optional[str] = Header(None, alias="X-Node-Auth-Token")
) -> dict:
    """
    Verify that the request comes from any authorized node (main, validator, or worker)
    Returns node info including role
    """
    if not x_node_id:
        raise HTTPException(
            status_code=401,
            detail="Missing node identification"
        )
    
    role = _authenticator.get_node_role(x_node_id)
    
    # Main node needs token verification
    if role == "main":
        if not x_node_auth_token or not _authenticator.verify_main_node(x_node_id, x_node_auth_token):
            raise HTTPException(
                status_code=403,
                detail="Invalid main node credentials"
            )
    
    return {
        "node_id": x_node_id,
        "role": role,
        "can_write": _authenticator.can_write_blocks(x_node_id, x_node_auth_token)
    }

# Optional: Environment-based enforcement toggle
ENFORCE_AUTH = os.environ.get("BLYAN_ENFORCE_AUTH", "true").lower() == "true"

async def maybe_verify_main_node(
    x_node_id: Optional[str] = Header(None, alias="X-Node-ID"),
    x_node_auth_token: Optional[str] = Header(None, alias="X-Node-Auth-Token")
) -> Optional[str]:
    """
    Conditionally verify main node based on BLYAN_ENFORCE_AUTH environment variable
    Useful for development vs production
    """
    if not ENFORCE_AUTH:
        return "dev-mode"
    
    return await verify_main_node(x_node_id, x_node_auth_token)

from datetime import datetime